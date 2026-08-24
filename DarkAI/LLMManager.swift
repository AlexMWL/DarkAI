import Foundation
import Combine
import LlamaSwift
import os
import UIKit
import Darwin

// MARK: - State Types

enum ModelLoadState {
    case unloaded
    case loading(progress: Double, status: String)
    case loaded(modelName: String, sizeGB: Double)
    case failed(error: String)
}

enum MemorySafetyStatus: Equatable {
    case safe
    case warning(requiredGB: Double, availableGB: Double)
    case dangerous(requiredGB: Double, availableGB: Double)
}

// MARK: - LlamaRunner Actor
// Wraps the raw llama.cpp C-API in a background actor to keep inference off the main thread.



actor LlamaRunner {


    /// Smallest context a streaming load is allowed to settle for before being refused outright.
    /// Below this the model cannot hold a system prompt and a question together, so "loaded" is
    /// a worse outcome than a clear failure.
    private static let minimumUsableContextTokens = 2048

    private var model: OpaquePointer? = nil
    private var context: OpaquePointer? = nil
    private var nCtxTokens: Int = 2048  // actual context window in tokens
    private var trainedCtxTokens: Int = 0
    private var isCancelled = false
    /// Actual tokenized prompt length (post-truncation) from the most recent generation —
    /// the real figure, as opposed to the char-count estimate used for UI budgeting.
    private var lastPromptTokenCount: Int = 0

    /// Tokens currently resident in the KV cache for sequence 0, in cache-position order —
    /// `residentTokens[i]` is the token actually decoded at position `i`. Kept in lockstep with
    /// `nPos` in `generateStream`: `residentTokens.count` equals `nPos` after every mutation
    /// (initial prefill, each per-token decode, each context-shift eviction), and empty whenever
    /// the cache holds nothing meaningful — before the first load, and after
    /// `unloadModelOnly`/`unload`.
    ///
    /// Exists purely to make cross-turn KV-cache reuse provably safe rather than a heuristic.
    /// `generateStream` used to call `llama_memory_clear` and redo the full prefill on *every*
    /// turn, even though most of a multi-turn conversation's prompt (system block plus prior
    /// history) is byte-identical to what was just decoded a moment ago. At the start of the next
    /// call, this turn's freshly-tokenized prompt is compared token-ID-for-token-ID against this
    /// array (see the reuse check in `generateStream`). Causal attention means a cached
    /// position's K/V entry depends only on the tokens at or before it and never changes once
    /// written, so as long as every ID in `residentTokens[0..<n]` is *literally* identical to the
    /// new prompt's first `n` tokens, those cached entries are exactly what a fresh decode would
    /// have produced — this holds regardless of how either string was tokenized, since it
    /// compares the resulting token IDs directly rather than reasoning about whether BPE/SPM
    /// merges are stable across the turn boundary (they are not, in general).
    ///
    /// Deliberately narrow: reuse is only attempted when the *entire* current cache passes that
    /// check — a strict prefix of the new prompt, nothing partial. A mismatch anywhere (edited or
    /// regenerated history, a truncated prompt, the shared background personality-analysis pass
    /// using this same actor/cache for an unrelated prompt, or simply the first message) falls
    /// back to the original full-clear-and-reprefill behaviour rather than attempting to splice
    /// out a divergent middle section — a partial-reuse "diff to the divergence point" was
    /// considered and rejected as too easy to get subtly wrong for a path with no way to be
    /// runtime-verified here.
    private var residentTokens: [llama_token] = []

    /// Longest run of tokens `a` and `b` agree on from the start. Used only to decide KV-cache
    /// reuse eligibility (see `residentTokens`) — a plain linear scan is more than fast enough
    /// against even a full context window, and staying this simple keeps the safety argument
    /// easy to check by inspection.
    private static func commonPrefixLength(_ a: [llama_token], _ b: [llama_token]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n && a[i] == b[i] { i += 1 }
        return i
    }

    /// True for the entire duration of an in-flight `generateStream` call.
    ///
    /// `LlamaRunner` being an `actor` serializes calls to it *between* suspension points, but
    /// `generateStream` itself suspends every token (`await Task.yield()`), and actors are
    /// reentrant across suspension points — a second call queued on this same actor can start
    /// running while the first is parked mid-stream. Two callers hit this in practice: an
    /// ordinary chat reply, and `LLMManager`'s periodic background personality-style analysis.
    /// `LLMManager.isGenerating` stops the background one from starting *during* a visible chat
    /// reply, but nothing stopped the reverse — a chat message sent while a background analysis
    /// was still streaming. Both calls share the same `context`, i.e. the same KV cache, and
    /// letting them interleave corrupts its position bookkeeping: llama.cpp then aborts a later
    /// decode with "inconsistent sequence positions" (the last position it recorded doesn't
    /// match where the next batch says it's continuing from), which is what actually sat behind
    /// reports of the model crashing and unloading after a couple of exchanges — slower models
    /// widen the window in which a background analysis is still running when the next message
    /// goes out, which is why it showed up on the largest catalog model first.
    ///
    /// Checked and set at the very top of `generateStream`, before its first `await` — the
    /// synchronous prefix of an actor method can't be preempted, so this check-and-set can't
    /// itself race.
    private var isBusyGenerating = false

    /// The context window actually applied to the loaded model, which can differ from the
    /// user's requested setting once `safeContextTokens` clamps it to available RAM.
    func getContextWindowTokens() -> Int { nCtxTokens }
    /// The context the model was actually trained for, which is a property of the weights and
    /// entirely separate from what this device could afford to allocate.
    func getTrainedContextTokens() -> Int { trainedCtxTokens }
    func getLastPromptTokenCount() -> Int { lastPromptTokenCount }

    init() {
        // Initialize the backend once for the lifetime of this actor
        llama_backend_init()
        
        llama_log_set({ level, text, user_data in
            guard let text = text, let str = String(cString: text, encoding: .utf8) else { return }
            let cleanStr = str.trimmingCharacters(in: .whitespacesAndNewlines)
            if !cleanStr.isEmpty {
                DispatchQueue.main.async {
                    LogManager.shared.log("LLAMA: \(cleanStr)")
                }
            }
        }, nil)
    }

    func load(path: String, availableMemoryGB: Double, modelSizeGB: Double, contextLimit: Int) throws {
        unloadModelOnly()

        // A missing file fails llama_model_load_from_file instantly and silently (no internal
        // llama.cpp logging at all, since it never gets far enough to parse anything) — which
        // previously surfaced as the same generic "fits in RAM" message as a true memory
        // failure, wrongly pointing at memory pressure instead of the real, unresolvable cause.
        guard FileManager.default.fileExists(atPath: path) else {
            throw NSError(domain: "LlamaRunner", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "Model file not found at \(path). It may have been moved or the app reinstalled — try reselecting it in Settings."])
        }

        let footprintBeforeLoadGB = MemoryBudget.footprintGB()

        let plan = planOffload(path: path,
                               modelSizeGB: modelSizeGB,
                               availableMemoryGB: availableMemoryGB,
                               contextLimit: contextLimit)

        // `n_batch`/`n_ubatch` tier: prefill throughput scales with batch size, but the
        // attention/activation scratch buffers scale with `n_ubatch` (see the comment on
        // `ctxParams.n_ubatch` below), so raising the ceiling has to be paired with
        // `computeOverheadGB` reserving more for it — not just a bigger number handed to
        // llama.cpp. Gated on `availableMemoryGB` (this load's real, current headroom, not a
        // static device tier) and kept at the historical 512 floor whenever the load is already
        // streaming weights from storage: that path independently halves `n_ubatch` to protect
        // its much tighter budget (see below), so doubling the *reservation* for a batch size it
        // doesn't actually use would only shrink an already-constrained context window for
        // nothing in return.
        let batchCeiling = (!plan.streamsFromStorage && availableMemoryGB >= 4.0) ? 1024 : 512

        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = plan.nGpuLayers
        modelParams.use_mmap = true
        modelParams.use_mlock = false

        let loaded = llama_model_load_from_file(path, modelParams)

        guard let mdl = loaded else {
            throw NSError(domain: "LlamaRunner", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to load GGUF. Check the file is a valid model and fits in RAM."])
        }
        self.model = mdl

        let trainedCtx = Int(llama_model_n_ctx_train(mdl))
        self.trainedCtxTokens = trainedCtx
        // Read during the probe; the loaded model reports the same values.
        let geometry = plan.geometry

        // Two different thread counts, because decode and prefill have opposite bottlenecks on
        // Apple Silicon. Decode (n_threads) runs one token at a time and is memory-bandwidth
        // bound, not compute bound — optimized for Apple Silicon: using all cores (including
        // E-cores) degrades performance, since their lower per-core memory bandwidth becomes the
        // new bottleneck the moment they're added to the pool. Prefill (n_threads_batch)
        // processes a whole batch of tokens per call and is compute bound, so unlike decode it
        // genuinely benefits from every core it can get, E-cores included.
        //
        // iOS has no public API to ask how many of `activeProcessorCount` are P-cores vs E-cores,
        // so `optimalThreads` stays this existing conservative estimate (roughly the P-core count
        // on every current Apple Silicon iPhone SoC) rather than guessing at a split.
        // `batchThreads` isn't trying to identify P-cores at all — it just allows more
        // parallelism for the compute-bound phase, capped well under `activeProcessorCount` so it
        // can't regress into the same all-cores-including-E-cores slowdown decode has to avoid.
        let optimalThreads = Int32(max(2, min(4, ProcessInfo.processInfo.activeProcessorCount / 2)))
        let batchThreads = Int32(max(Int(optimalThreads), min(ProcessInfo.processInfo.activeProcessorCount, 6)))

        // Prefer the quantised cache, but keep f16 as a fallback rather than failing the load.
        // Quantised K/V depends on backend support for this specific architecture's head
        // geometry, and that is only knowable by trying: a rejected combination surfaces as a
        // null context, not as a diagnosable error. Falling back re-derives the context window
        // first, because f16 doubles the per-token cost and the Q8_0-sized window would no
        // longer fit the same budget.
        var formats: [KVCacheFormat] = []
        if geometry?.supportsQuantizedCache ?? false {
            formats.append(.q8_0)
            // Only added for a load that's already streaming weights from storage — see
            // `KVCacheFormat.q4_0`. A load that comfortably fits never sees this option at all.
            if plan.streamsFromStorage {
                formats.append(.q4_0)
            }
        }
        formats.append(.f16)

        // A streaming load that can only afford a few hundred tokens is not a working model, it
        // is a model that will accept one message and then fail on it. `llama_init_from_model`
        // builds that context quite happily — the memory it could not spare gets spent later, in
        // prefill, and the failure there is a dead Metal backend and a corrupted UI rather than a
        // clean refusal. Better to refuse now, while there is still something useful to say.
        //
        // Checked against the most generous format in the list — the one with the smallest
        // `bytesPerElement`, which yields the largest window for a given budget — rather than
        // assuming it's always first: Q4_0 packs tighter than Q8_0 when it's present, so it's the
        // real best case now, not Q8_0. If that can't reach a usable size, nothing else in the
        // list, which all cost more per token, will either.
        //
        // Only streaming loads are held to this. A fully-resident model scraping the floor on a
        // small device is long-standing behaviour that works, and is not this problem.
        if plan.streamsFromStorage,
           let best = formats.min(by: { $0.bytesPerElement < $1.bytesPerElement }) {
            let bestCtx = safeContextTokens(geometry: geometry,
                                            kvFormat: best,
                                            plan: plan,
                                            availableMemoryGB: availableMemoryGB,
                                            modelSizeGB: modelSizeGB,
                                            requestedLimit: contextLimit,
                                            trainedCtx: trainedCtx,
                                            batchCeiling: batchCeiling)
            if bestCtx < Self.minimumUsableContextTokens {
                llama_model_free(mdl)
                self.model = nil
                throw NSError(domain: "LlamaRunner", code: 4, userInfo: [NSLocalizedDescriptionKey:
                    "This model is too large for this device. It would load with only a \(bestCtx)-token context — too small to hold a conversation, and it would run out of GPU memory on the first message. Try a smaller or more compressed model."])
            }
        }

        for (formatIndex, format) in formats.enumerated() {
            let nCtx = Int32(safeContextTokens(geometry: geometry,
                                               kvFormat: format,
                                               plan: plan,
                                               availableMemoryGB: availableMemoryGB,
                                               modelSizeGB: modelSizeGB,
                                               requestedLimit: contextLimit,
                                               trainedCtx: trainedCtx,
                                               batchCeiling: batchCeiling))

            // A format later in the list packs tighter than this one — e.g. Q8_0 landing under
            // the usable floor while Q4_0 is still ahead — so it's worth trying that one instead
            // of settling for (or worse, technically succeeding at) a context too small to hold a
            // conversation. The preflight check above already confirmed the most generous format
            // in the list can clear the floor, so this loop is guaranteed to reach a format that
            // passes before it runs out.
            if plan.streamsFromStorage,
               nCtx < Self.minimumUsableContextTokens,
               formats[(formatIndex + 1)...].contains(where: { $0.bytesPerElement < format.bytesPerElement }) {
                LogManager.shared.log("Offload plan — \(format.name) KV cache only reaches \(nCtx) tokens, trying a smaller-footprint format")
                continue
            }

            var ctxParams = llama_context_default_params()
            ctxParams.n_ctx   = UInt32(nCtx)
            // `batchCeiling` (computed once above, before this format loop) raises this past the
            // historical flat 512 on a load with enough headroom to afford a bigger prefill batch
            // — see its own comment for the memory tradeoff, and `computeOverheadGB` below for how
            // the KV budget was told about the larger compute buffer this implies.
            ctxParams.n_batch = UInt32(min(nCtx, Int32(batchCeiling)))
            // `n_batch` is the logical size a prompt gets chunked into; `n_ubatch` is the physical
            // size actually pushed through the compute graph at once, and it's what the attention/
            // activation scratch buffers scale with (`computeOverheadGB`'s 523 MiB measurement was
            // taken at 512/512). Left equal to `n_batch` by default. For a streaming load, where
            // every spare hundred MB is a meaningful share of the budget in `safeContextTokens`,
            // halving it trades some prefill throughput — the prompt gets chunked into more, smaller
            // passes — for a smaller real compute buffer, at no risk to correctness.
            ctxParams.n_ubatch = plan.streamsFromStorage
                ? min(ctxParams.n_batch, 256)
                : ctxParams.n_batch
            ctxParams.n_threads       = optimalThreads
            ctxParams.n_threads_batch = batchThreads
            // Quantised K/V requires flash attention, which is why this is set unconditionally
            // rather than left on AUTO.
            ctxParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED
            ctxParams.type_k = format.ggmlType
            ctxParams.type_v = format.ggmlType

            if plan.streamsFromStorage {
                // Anything left off the GPU has to stay off it. Op offloading copies host-side
                // weights to the GPU to run large-batch matmuls there, which is a good trade
                // normally and a catastrophic one here: those weights were left behind precisely
                // because they don't fit, and prefill runs at `n_batch` = 512, so the scheduler
                // stages layer after layer across in bulk and reintroduces — transiently, all at
                // once — the entire allocation the plan just avoided.
                //
                // This condition used to cover only pinned experts, and a dense 12B model with
                // ~20 layers on the CPU side was the result: it loaded cleanly, then died in
                // prefill with `kIOGPUCommandBufferCallbackErrorOutOfMemory` on the first
                // message, taking the shared Metal device down with it and corrupting the app's
                // own rendering. Dense streaming needs this more than expert streaming does, not
                // less — its CPU-side layers are touched by every single token.
                //
                // Costs prompt-ingestion speed. The header's warning about this setting applies
                // only when `n_seq_max > 1`, which is not the case for a single conversation.
                ctxParams.op_offload = false
            }

            if let ctx = llama_init_from_model(mdl, ctxParams) {
                // What the plan predicted would stay resident, against what actually did.
                //
                // Every budget in this file rests on a claim about which bytes iOS will charge
                // to this process — offloaded weights yes, mmap'd weights no, pinned experts
                // only in part. Those claims are estimates about another system's behaviour, and
                // when one is wrong it is wrong by gigabytes: the load appears to succeed, the
                // footprint quietly sits far above plan, and the app is killed shortly after by
                // a jetsam the user experiences as a crash with no explanation.
                //
                // Measuring here turns that into a refused load with a reason. The threshold is
                // deliberately loose — this is a backstop for a broken assumption, not a check
                // on the accuracy of the estimate.
                let actualGB = MemoryBudget.footprintGB() - footprintBeforeLoadGB
                let kvBytesPerToken = (geometry?.elementsPerToken ?? 0) * format.bytesPerElement
                let predictedGB = plan.residentWeightGB
                    + Double(nCtx) * kvBytesPerToken / (1024 * 1024 * 1024)
                if actualGB > predictedGB * 1.5 + 1.0 {
                    LogManager.shared.log(String(
                        format: "Load aborted: footprint grew %.2f GB against a predicted %.2f GB — the offload plan is not describing what this model actually does in memory.",
                        actualGB, predictedGB))
                    llama_free(ctx)
                    llama_model_free(mdl)
                    self.model = nil
                    throw NSError(domain: "LlamaRunner", code: 5, userInfo: [NSLocalizedDescriptionKey: String(
                        format: "This model used %.1f GB of memory when %.1f GB was expected, so it was unloaded before it could crash the app. It isn't usable on this device.",
                        actualGB, predictedGB)])
                }

                self.context = ctx
                self.nCtxTokens = Int(nCtx)
                LogManager.shared.log(
                    "Loaded with \(format.name) KV cache, \(nCtx) tokens — \(plan.note)"
                )
                return
            }

            LogManager.shared.log("Context creation failed with \(format.name) KV cache at \(nCtx) tokens")
        }

        llama_model_free(mdl)
        self.model = nil
        throw NSError(domain: "LlamaRunner", code: 2,
                      userInfo: [NSLocalizedDescriptionKey: "Failed to create inference context. The model may require more RAM."])
    }

    // MARK: - Offload planning

    /// How a model's layers get divided between Metal and CPU/mmap for one particular load.
    ///
    /// This division is the app's memory dial. Layers offloaded to Metal are wired into GPU
    /// buffers and cost full price against the process's dirty-memory allowance; layers left
    /// behind stay as clean, file-backed mmap pages, which iOS can evict and re-read on demand
    /// and which do not count against that allowance. Full offload is fastest, and partial
    /// offload is what lets a model larger than the memory budget run at all — slower, because
    /// the CPU-side layers are re-read from storage as they are touched, but running rather
    /// than refused.
    /// For a mixture-of-experts model there is a second, much cheaper dial. A dense layer left
    /// off the GPU is re-read in full on every token, so dense streaming is bounded by storage
    /// bandwidth. A routed expert is read only by the tokens that route to it, and a token
    /// touches a small fraction of the experts in each block — so pinning expert stacks to
    /// CPU/mmap gives up a fraction of the bandwidth that pinning whole layers would, while
    /// freeing the majority of the file. That is what makes a sparse model far larger than the
    /// device's memory a reasonable thing to run, where a dense one of the same size is not.
    private struct OffloadPlan {
        let nGpuLayers: Int32

        /// Weight bytes expected to stay charged against the process's dirty-memory allowance.
        ///
        /// Computed here rather than in `safeContextTokens` because this is where the placement
        /// decisions are actually known — the context budget only needs the total.
        let residentWeightGB: Double

        /// Attention shape, read during the probe and carried forward so the real load doesn't
        /// have to read it again.
        let geometry: KVGeometry?

        /// Any weight at all is being left off the GPU to be read from storage instead.
        ///
        /// Deliberately not derived from `nGpuLayers`: dense layer
        /// streaming leaves the expert list empty while keeping most of the model off the GPU,
        /// and the large-vocab correctness cap holds layers back for a reason that has nothing
        /// to do with memory but has exactly the same consequence for op offloading. Every path
        /// states its own answer so none of those cases can be missed by inference.
        let streamsFromStorage: Bool

        let note: String
    }

    /// Decides where each part of the model goes, from a single metadata-only read of the file.
    ///
    /// `ModelProfiler` parses the GGUF header — hyperparameters, vocabulary size and the tensor
    /// directory — without touching tensor data. It replaces a `vocab_only` model load that was
    /// used here originally and could not do this job: that mode populates only the vocabulary,
    /// so every hyperparameter came back as zero and the `nLayer > 0` guard below rejected every
    /// model that has ever been loaded, silently sending all of them down the full-offload path
    /// no matter how large. Asking it for attention geometry was worse still — it aborted the
    /// process. Both failures are invisible from here, which is why this now reads the file.
    private func planOffload(path: String,
                             modelSizeGB: Double,
                             availableMemoryGB: Double,
                             contextLimit: Int) -> OffloadPlan {
        let profile = ModelProfiler.profile(path: path)

        let nLayer = profile?.nLayer ?? 0
        let vocabSize = profile?.vocabSize ?? 0
        let geometry: KVGeometry? = (profile?.hasUsableGeometry ?? false)
            ? KVGeometry(nLayer: profile!.nLayer,
                         nHeadKV: profile!.nHeadKV,
                         headDimK: profile!.headDimK,
                         headDimV: profile!.headDimV)
            : nil

        // Very large vocabularies (Gemma's 256K-token vocab, in particular) have been observed
        // to produce zeroed Metal compute buffers when the output projection layer — which
        // scales with vocab size — is fully GPU-offloaded on this device; capping GPU layers
        // works around it. Queried from the model itself (not matched against the file name) so
        // this correctly protects any large-vocab model — e.g. Llama 3's 128K vocab stays
        // comfortably under the threshold and gets full offload, without needing a per-model
        // name allowlist.
        //
        // This is a correctness cap, not a memory one, which is why hitting it does not put the
        // load into the memory-constrained regime below.
        let correctnessCap: Int32 = vocabSize >= 150_000 ? 15 : 99

        // Coarse reserve: `safeContextTokens` sizes the KV cache precisely once the model is
        // open, so this only has to stop the layer budget from eating the memory that call will
        // go on to need. Deliberately small — a dense model asking for a large context should
        // get a smaller context, not start streaming layers to pay for one.
        let safetyMarginGB = MemoryBudget.safetyMarginGB(for: availableMemoryGB)
        let kvAndComputeReserveGB = min(1.25, availableMemoryGB * 0.20)
        // Bounded by what the GPU will hold as well as by what iOS will let the app dirty. These
        // are different numbers — 8.59 GB against 11.4 GB of RAM on an iPhone 17 Pro — and the
        // smaller one is the one that decides whether a fully-offloaded model survives its first
        // command buffer.
        let gpuBudgetGB = MemoryBudget.gpuResidentBudgetGB(processHeadroomGB: availableMemoryGB)
            - safetyMarginGB - kvAndComputeReserveGB

        guard nLayer > 0, modelSizeGB > 0 else {
            // Unreadable header. Without a layer count there is no way to hold back *some* of the
            // model, so the choice is all on the GPU or none of it — and "all" is precisely the
            // arrangement that put a 6.8 GB model onto an 8.59 GB GPU and killed it on the first
            // message. Small models still offload as before; anything that doesn't clearly fit
            // stays on mmap, which is slow but cannot fail this way.
            let fits = modelSizeGB <= gpuBudgetGB
            let note = fits
                ? "geometry unavailable, model fits — offloading up to \(correctnessCap) layers"
                : "geometry unavailable and model exceeds the GPU budget — keeping all weights on CPU/mmap"
            LogManager.shared.log("Offload plan — \(note)")
            return OffloadPlan(nGpuLayers: fits ? correctnessCap : 0,
                               residentWeightGB: fits ? modelSizeGB * 0.8 : modelSizeGB * 0.45,
                               geometry: geometry,
                               streamsFromStorage: !fits,
                               note: note)
        }

        // Blocks plus the token-embedding and output tensors, which together are what the file
        // size covers. Treating those two as roughly one layer each keeps this a slight
        // over-estimate of per-layer cost for large-vocab models, which errs toward offloading
        // fewer layers — the safe direction.
        let perLayerGB = modelSizeGB / Double(nLayer + 2)

        // Everything fits on the GPU. Keep the sentinel rather than an exact layer count so the
        // output tensor is offloaded too, and keep the previously tuned flat 0.8 residency
        // figure, so no model that already loaded sees its context window move.
        if modelSizeGB <= gpuBudgetGB {
            // The large-vocab cap still leaves layers on the CPU even though memory was never
            // the reason, and for op offloading the reason does not matter — only the fact.
            let allLayersOnGPU = Int(correctnessCap) >= nLayer
            let note = "fully offloaded: n_gpu_layers \(correctnessCap) of \(nLayer) layers"
            LogManager.shared.log("Offload plan — \(note)")
            return OffloadPlan(nGpuLayers: correctnessCap,
                               residentWeightGB: modelSizeGB * 0.8,
                               geometry: geometry,
                               streamsFromStorage: !allLayersOnGPU,
                               note: note)
        }

        // Expert pinning used to live here, and it does not work on Metal. The idea was sound
        // and the accounting was right: pin every routed expert stack to CPU/mmap, leave
        // attention on the GPU, and a 20B model needs under 2 GB resident. What defeats it is
        // how llama.cpp maps a model file. With mmap on, each device gets *one* buffer spanning
        // from its first tensor to its last, and a MoE model interleaves expert and attention
        // tensors throughout the file — so putting any attention tensor on the GPU forces the
        // Metal mapping to cover every expert lying between them too.
        //
        // Measured on an iPhone 17 Pro loading GPT-OSS-20B, which is how this was found:
        //
        //     load_tensors:  CPU_Mapped model buffer size = 10949.33 MiB
        //     load_tensors: MTL0_Mapped model buffer size = 11536.18 MiB
        //     ggml_metal_log_allocated_size: warning: current allocated size is greater
        //                                    than the recommended max working set size
        //
        // Both devices mapped essentially the whole 11.27 GiB file. Metal wired 12.5 GB against
        // an 8.19 GB working set and the first command buffer died. llama.cpp says as much
        // during the load — "tensor overrides to CPU are used with mmap enabled" — and the only
        // way to get exact per-device buffers is `use_mmap = false`, which on iOS means copying
        // 9.5 GB of experts into dirty memory: the same failure by a different route.
        //
        // Dense layer streaming below is not affected, and the reason is worth keeping in mind
        // if this is ever revisited: it splits the model at a *layer boundary*, so each device's
        // tensors occupy one contiguous run of the file and the two mappings barely overlap.
        // Any future attempt at this has to preserve that property to be worth trying.

        // Dense streaming: hold layers back from the GPU until the resident share fits.
        //
        // llama.cpp offloads a contiguous run of the *last* n_gpu_layers blocks — the ones
        // nearest the output — when n_gpu_layers < nLayer, so the layers that matter for this
        // budget are the trailing ones, walked from the end backward, not an arbitrary count of
        // average-sized ones.
        //
        // Real per-block byte counts (`ModelProfile.blockGBByLayer`) replace the flat `perLayerGB`
        // average whenever the tensor directory named every block — this is what actually catches
        // uneven layer sizes, which the average blurs together regardless of which layers happen
        // to be the ones offloaded. Falls back to the average when the directory is incomplete,
        // which was always this estimate's only source before.
        let realBlockGB = profile?.blockGBByLayer
        let hasCompleteBlockSizes = (realBlockGB?.count ?? 0) == nLayer

        // Cumulative GB of the trailing N blocks, index 0 meaning zero blocks offloaded.
        var cumulativeGB: [Double] = [0]
        for layer in stride(from: nLayer - 1, through: 0, by: -1) {
            let layerGB = hasCompleteBlockSizes ? (realBlockGB?[layer] ?? perLayerGB) : perLayerGB
            cumulativeGB.append(cumulativeGB[cumulativeGB.count - 1] + layerGB)
        }

        // The token-embedding and output tensors aren't part of any block, so they never show up
        // in `cumulativeGB` — reserved here the same way the flat estimate always folded them in,
        // as two layers' worth up front, so a model with real per-layer data doesn't get a more
        // permissive reserve than one without it.
        let reservedForEmbeddingAndOutputGB = 2.0 * perLayerGB
        let layerBudgetGB = max(0, gpuBudgetGB - reservedForEmbeddingAndOutputGB)

        // Largest trailing-block count whose real cumulative size still fits the budget.
        let affordableLayers = cumulativeGB.lastIndex(where: { $0 <= layerBudgetGB }) ?? 0
        let nGpuLayers = min(correctnessCap, Int32(affordableLayers))
        let gpuGB = cumulativeGB[Int(nGpuLayers)]
        let cpuGB = max(0, modelSizeGB - gpuGB)

        let note = String(format: "streaming: %d of %d layers on GPU (%.2f GB budget, %.3f GB/layer)",
                          Int(nGpuLayers), nLayer, gpuBudgetGB, perLayerGB)
        LogManager.shared.log("Offload plan — \(note)")

        return OffloadPlan(nGpuLayers: nGpuLayers,
                           // Layers left off the GPU are clean file-backed pages that iOS can
                           // evict, but they are re-read on every forward pass, so they are
                           // charged at better than half rather than discounted to nothing.
                           residentWeightGB: gpuGB * 0.8 + cpuGB * 0.45,
                           geometry: geometry,
                           streamsFromStorage: true,
                           note: note)
    }

    /// Context window the budget is allowed to aim for, before memory is considered.
    ///
    /// Shared by `planOffload` and `safeContextTokens` so the two cannot disagree about what
    /// they are budgeting for — a mismatch here is invisible until a model quietly ends up with
    /// a 512-token window.
    private func contextClamp(requestedLimit: Int, trainedCtx: Int, modelSizeGB: Double) -> Int {
        let trainedClamp = max(512, trainedCtx > 0 ? trainedCtx : requestedLimit)
        let requestedClamp = max(512, min(requestedLimit, trainedClamp))
        // Backstop independent of the detailed formula — never aim past this for a model of this
        // size, in case a given architecture's real memory behaviour (e.g. Gemma's mixed
        // local/global attention layers) doesn't match the generic per-layer estimate.
        let hardCeiling = modelSizeGB > 4.0 ? 16384 : (modelSizeGB > 2.0 ? 32768 : 65536)
        return min(requestedClamp, hardCeiling)
    }

    /// Compute and activation buffers — attention scratch space and batch buffers.
    ///
    /// This was `max(0.5, tokens / 8192 * 0.75)`, i.e. linear in the *requested* context, and it
    /// was reserving memory that is never allocated. Measured on an iPhone 17 Pro at a 16,384
    /// context with `n_batch` 512:
    ///
    ///     sched_reserve: MTL0 compute buffer size = 398.38 MiB
    ///     sched_reserve:  CPU compute buffer size = 124.46 MiB
    ///
    /// 523 MiB in total, where the old formula reserved 1.5 GB — and 6 GB had the slider been at
    /// 65,536. These buffers hold one batch of activations, so they scale with `n_batch`, which
    /// is capped at 512, not with the size of the cache. Reserving against the context made the
    /// budget self-defeating: the larger the window asked for, the more was withheld from the
    /// cache that would have provided it, which is how a 1.9 GB model on a 12 GB phone ended up
    /// with 512 tokens.
    ///
    /// Now essentially flat, with a slight context term for the bookkeeping that genuinely does
    /// grow, and still roughly double the measured figure.
    ///
    /// `nUbatch` defaults to 512 — the batch size the 523 MiB measurement above was actually
    /// taken at — so every existing caller keeps exactly today's numbers. `batchCeiling` (see
    /// `load`) can raise the real `n_ubatch` a fully-resident, high-memory load uses past that,
    /// and the buffer scales with it (per the `ctxParams.n_ubatch` comment in `load`), so this
    /// scales its own reservation the same way rather than quietly under-reserving for a bigger
    /// buffer. `max(1.0, …)` means the ratio only ever pushes the reservation up, never down —
    /// a smaller `nUbatch` (e.g. a streaming load's halved 256) still reserves the original flat
    /// amount rather than clawing back margin that was never the risk this guards against.
    private func computeOverheadGB(forContext tokens: Int, nUbatch: Int = 512) -> Double {
        let batchGB = 0.5 * max(1.0, Double(nUbatch) / 512.0)
        return 0.25 + batchGB + Double(tokens) / 65536.0 * 0.5
    }

    // MARK: - KV cache sizing

    /// The data type the K/V cache is stored in, and what one cached element costs.
    ///
    /// Q8_0 packs 32 quantised values plus a single f16 scale into 34 bytes — 1.0625 bytes per
    /// element against f16's 2.0 — so the same memory budget buys nearly twice the context
    /// window, at a quality cost that is very hard to detect in practice. It requires flash
    /// attention (enabled unconditionally at load) and head dimensions that divide evenly into
    /// the 32-element block.
    private enum KVCacheFormat {
        case q8_0
        /// Q4_0 packs 32 4-bit values plus one f16 scale into 18 bytes — 0.5625 bytes per element,
        /// against Q8_0's 1.0625 — buying nearly double Q8_0's own context window for the same
        /// memory. Unlike Q8_0, the quality cost here is real rather than "hard to detect," so
        /// this is only ever offered as a rescue for a load that's already streaming weights from
        /// storage and would otherwise be refused for too small a context — never a default
        /// preference over Q8_0. Same 32-element block alignment requirement as Q8_0.
        case q4_0
        case f16

        var ggmlType: ggml_type {
            switch self {
            case .q8_0: return GGML_TYPE_Q8_0
            case .q4_0: return GGML_TYPE_Q4_0
            case .f16:  return GGML_TYPE_F16
            }
        }

        /// Bytes per cached element. Q8_0 is 34 bytes per 32-element block, Q4_0 is 18, both scale
        /// included.
        var bytesPerElement: Double {
            switch self {
            case .q8_0: return 34.0 / 32.0
            case .q4_0: return 18.0 / 32.0
            case .f16:  return 2.0
            }
        }

        var name: String {
            switch self {
            case .q8_0: return "Q8_0"
            case .q4_0: return "Q4_0"
            case .f16:  return "F16"
            }
        }
    }

    /// KV-cache geometry read from the model's own GGUF metadata, rather than guessed from file
    /// size, so the per-token cost below tracks *this* model's real attention shape.
    private struct KVGeometry {
        let nLayer: Int
        let nHeadKV: Int
        let headDimK: Int
        let headDimV: Int

        /// K and V elements cached per token across every layer.
        var elementsPerToken: Double {
            Double(nLayer * nHeadKV) * Double(headDimK + headDimV)
        }

        /// A quantised cache needs each head dimension to divide evenly into the 32-element
        /// block. Nearly every current architecture uses a multiple of 64, but the ones that
        /// don't must stay on f16 rather than fail the load.
        var supportsQuantizedCache: Bool {
            headDimK % 32 == 0 && headDimV % 32 == 0
        }
    }

    /// Computes a safe context window using the model's actual KV-cache geometry
    /// (layer count, KV head count, per-head K/V dims read from GGUF metadata) rather
    /// than a generic size-based guess, so the limit tracks the true per-token memory
    /// cost for *this* model on *this* device's real, currently available RAM.
    ///
    /// This budgets deliberately conservatively. A previous, looser version of this formula
    /// (larger usable-memory fraction, flat compute overhead, lower weight-residency
    /// estimate) allowed large-vocab models like Gemma — which run most layers on CPU
    /// because of the correctness cap in `planOffload` — to request a context window that
    /// looked safe on paper but wasn't in practice, causing an out-of-memory failure severe
    /// enough to reboot the device rather than just being killed by iOS. Every margin below
    /// is intentionally wide; a smaller-than-necessary context window is a minor inconvenience,
    /// a device reboot is not an acceptable failure mode.
    private func safeContextTokens(geometry: KVGeometry?,
                                   kvFormat: KVCacheFormat,
                                   plan: OffloadPlan,
                                   availableMemoryGB: Double,
                                   modelSizeGB: Double,
                                   requestedLimit: Int,
                                   trainedCtx: Int,
                                   batchCeiling: Int = 512) -> Int {
        let safeRequestedClamp = contextClamp(requestedLimit: requestedLimit,
                                              trainedCtx: trainedCtx,
                                              modelSizeGB: modelSizeGB)

        guard let geometry else { return safeRequestedClamp }

        let bytesPerTokenAllLayers = geometry.elementsPerToken * kvFormat.bytesPerElement
        guard bytesPerTokenAllLayers > 0 else { return safeRequestedClamp }

        // Budget against the process's real memory *allowance*, with named deductions.
        //
        // `availableMemoryGB` is `os_proc_available_memory()` measured just before the load: the
        // headroom this process has against its own jetsam limit, not the device's free RAM.
        // iOS hands that allowance out dynamically, so it is already the correct quantity to
        // plan against — the previous formula then halved it *and* subtracted the weights, which
        // double-charged for the same memory and left almost nothing for the KV cache. On an
        // 11 GB device with ~6 GB of headroom and a 2.2 GB model, that yielded roughly 0.5 GB of
        // cache — about 4.6k tokens, against a 128k-token-capable model and an 8k user setting.
        //
        // Every reduction below is now something concrete, so the budget can be reasoned about
        // rather than scaled down by a guess.
        //
        // Reserve for the OS and for transient spikes during the load itself (mmap page-in, KV
        // allocation, compute buffer setup), which can briefly exceed steady state.
        let safetyMarginGB = MemoryBudget.safetyMarginGB(for: availableMemoryGB)

        // Weights are mmap'd, but only the tensors left off the GPU actually stay that way.
        // Anything offloaded to Metal gets wired into GPU buffers and costs full price against
        // the process's dirty-memory allowance; the rest are clean file-backed pages that iOS
        // can evict and re-read. `planOffload` is where the placement is decided, so it is also
        // where this is worked out — see `OffloadPlan.residentWeightGB` for how each regime is
        // charged. Measured before the load, so none of it is yet reflected in
        // `availableMemoryGB`.
        let residentWeightGB = plan.residentWeightGB

        let computeOverheadGB = computeOverheadGB(forContext: safeRequestedClamp, nUbatch: batchCeiling)

        var availableForKVGB = availableMemoryGB - safetyMarginGB - residentWeightGB - computeOverheadGB

        // The cache for GPU-resident layers lives in Metal buffers, so it competes with the
        // weights for the working set — a second ceiling the process allowance knows nothing
        // about. Leaving it out is the same oversight that let a fully-offloaded model clear the
        // memory budget and then die on its first command buffer; here it would simply do so
        // with a larger cache.
        let metalGB = MemoryBudget.metalWorkingSetGB
        if metalGB > 0 {
            availableForKVGB = min(availableForKVGB,
                                   metalGB - plan.residentWeightGB - computeOverheadGB - 0.25)
        }
        guard availableForKVGB > 0.05 else { return 512 }

        let availableForKVBytes = availableForKVGB * 1024.0 * 1024.0 * 1024.0
        let maxCtxByMemory = Int(availableForKVBytes / bytesPerTokenAllLayers)

        let resolved = max(512, min(safeRequestedClamp, maxCtxByMemory))
        LogManager.shared.log(String(
            format: "Context budget (%@ KV): %.2f GB free − %.2f margin − %.2f weights − %.2f compute = %.2f GB KV → %d tokens (requested %d, trained %d, applied %d)",
            kvFormat.name, availableMemoryGB, safetyMarginGB, residentWeightGB, computeOverheadGB,
            availableForKVGB, maxCtxByMemory, requestedLimit, trainedCtx, resolved
        ))
        return resolved
    }

    /// Set when a decode failed and the model was torn down as a result. `LLMManager` reads and
    /// clears this after a generation so it can move the UI out of the "loaded" state.
    private var decodeFaulted = false

    func consumeDecodeFault() -> Bool {
        defer { decodeFaulted = false }
        return decodeFaulted
    }

    /// Tears the model down after `llama_decode` fails.
    ///
    /// A decode failure on Metal is not a recoverable per-call error. When a command buffer
    /// fails — `kIOGPUCommandBufferCallbackErrorOutOfMemory` being the case that matters here —
    /// the backend latches into an error state and llama.cpp says so explicitly: *recreate the
    /// backend to recover*. Every subsequent decode then fails the same way, which is what
    /// "the model loaded but never answers" actually is from the user's side.
    ///
    /// Worse, the Metal device is shared with the rest of the app (and with
    /// stable-diffusion.cpp, hence the note in `unloadModelOnly`). A process sitting on an
    /// exhausted GPU allocator corrupts SwiftUI's own rendering — the visual glitching that
    /// accompanies this failure is not a separate bug, it is the same one. So the only correct
    /// response is to give the memory back immediately rather than hold a context that can
    /// never produce another token.
    private func handleDecodeFailure(stage: String) {
        LogManager.shared.log("Decode failed during \(stage) — unloading model to release the GPU allocator")
        decodeFaulted = true
        unloadModelOnly()
    }

    /// Unloads only the model+context, leaving the backend alive for the next load.
    func unloadModelOnly() {
        isCancelled = true  // Stop any ongoing generation
        autoreleasepool {
            if let ctx = context { llama_free(ctx) }
            if let mdl = model   { llama_model_free(mdl) }
            // Do NOT call llama_backend_free() here because the metal context is shared globally with stable-diffusion.cpp.
        }
        context = nil
        model   = nil
        // The cache this described no longer exists — see `residentTokens`'s doc comment.
        residentTokens = []
    }

    /// Full teardown — call only when the actor itself is being destroyed.
    func unload() {
        isCancelled = true
        autoreleasepool {
            if let ctx = context { llama_free(ctx) }
            if let mdl = model   { llama_model_free(mdl) }
            llama_backend_free()
        }
        context = nil
        model   = nil
        residentTokens = []
    }

    func requestCancel() {
        isCancelled = true
    }

    /// Tokenizes a string. Returns empty array on failure.
    private func tokenize(_ text: String, addBOS: Bool) -> [llama_token] {
        guard let mdl = model else { return [] }
        let vocab = llama_model_get_vocab(mdl)
        let utf8 = text.utf8
        var nTokensMax = Int32(utf8.count + 8)
        var tokens = [llama_token](repeating: 0, count: Int(nTokensMax))
        // parse_special = true: the chat-templated prompt this is called on contains literal
        // turn-delimiter markup (e.g. Llama 3's "<|eot_id|><|start_header_id|>assistant
        // <|end_header_id|>", Gemma's "<start_of_turn>model") inserted by the model's own
        // chat template. With parse_special = false these were tokenized as broken-up plain
        // text instead of the single atomic special tokens the model was trained on — so the
        // model's own context showed it a corrupted view of the conversation structure, which
        // is exactly what it then reproduced verbatim when generating (printing the literal
        // tag text and continuing into hallucinated extra turns instead of using the real
        // stop token, regardless of model or the EOG-detection fix on the output side).
        var n = llama_tokenize(vocab, text, Int32(utf8.count), &tokens, nTokensMax, addBOS, true)
        if n < 0 {
            // llama.cpp's documented convention: a negative return means the supplied buffer was
            // too small, and its magnitude is the size actually needed. `utf8.count + 8` is
            // generous for ordinary BPE/SPM tokenization, but retry once at the reported size
            // rather than silently dropping the tokenization (and the whole prompt with it) —
            // mirrors `CoreMLTokenizer.encode`'s identical handling of this same return value.
            nTokensMax = -n
            tokens = [llama_token](repeating: 0, count: Int(nTokensMax))
            n = llama_tokenize(vocab, text, Int32(utf8.count), &tokens, nTokensMax, addBOS, true)
        }
        guard n > 0 else { return [] }
        return Array(tokens.prefix(Int(n)))
    }

    /// Checks whether the loaded model advertises any vision/multimodal capability via its metadata.
    func supportsVision() -> Bool {
        guard let mdl = model else { return false }
        let count = llama_model_meta_count(mdl)
        for i in 0..<count {
            var keyBuf = [CChar](repeating: 0, count: 512)
            llama_model_meta_key_by_index(mdl, i, &keyBuf, 512)
            let key = String(cString: keyBuf).lowercased()
            if key.contains("vision") || key.contains("clip") || key.contains("mmproj") || key.contains("multimodal") {
                return true
            }
        }
        return false
    }

    /// Runs autoregressive inference and streams results via the continuation.
    func generateStream(
        messages: [(role: String, content: String)],
        maxTokens: Int,
        temperature: Float,
        continuation: AsyncStream<String>.Continuation,
        onContextTruncated: @escaping @Sendable () -> Void = {},
        onThinkingProgress: @escaping @Sendable (Int) -> Void = { _ in }
    ) async {
        guard let ctx = context, let mdl = model else {
            continuation.finish()
            return
        }
        guard !isBusyGenerating else {
            // See `isBusyGenerating`'s doc comment — a second call while one is already
            // in flight would corrupt the shared KV cache rather than queue politely. Declining
            // outright is correct for both actual callers: the chat UI already reports an empty
            // response usably ("[No response content was generated…]"), and a skipped
            // background analysis pass just waits for the next batch of messages.
            Task { @MainActor in
                LogManager.shared.log("LlamaRunner: generateStream called while another generation is already in flight — declining rather than risk corrupting the shared KV cache.")
            }
            continuation.finish()
            return
        }
        isBusyGenerating = true
        defer { isBusyGenerating = false }

        isCancelled = false
        let genStartTime = CFAbsoluteTimeGetCurrent()

        let vocab = llama_model_get_vocab(mdl)

        // Apply native chat template.
        var chatStructs: [llama_chat_message] = []
        var pointersToFree: [UnsafeMutablePointer<Int8>] = []

        // 1. Properly bolt down the memory (strdup)
        for msg in messages {
            guard let rolePtr = strdup(msg.role),
                  let contentPtr = strdup(msg.content) else { continue }
            
            pointersToFree.append(rolePtr)
            pointersToFree.append(contentPtr)
            
            chatStructs.append(llama_chat_message(
                role: UnsafePointer(rolePtr),
                content: UnsafePointer(contentPtr)
            ))
        }

        let tmpl = llama_model_chat_template(mdl, nil)
        var tmplBuf = [CChar](repeating: 0, count: 32768)
        let formattedLen = llama_chat_apply_template(tmpl, chatStructs, chatStructs.count, true, &tmplBuf, Int32(tmplBuf.count))
        
        // 2. Safely unbolt and clean up the memory (free)
        for ptr in pointersToFree {
            free(ptr)
        }

        let finalPrompt: String
        if formattedLen > 0 && formattedLen < tmplBuf.count {
            finalPrompt = String(cString: tmplBuf)
        } else {
            // Fallback for models missing a template
            finalPrompt = messages.map { "\($0.role): \($0.content)" }.joined(separator: "\n") + "\nassistant:\n"
        }

        // 1. Tokenize prompt — guard against exceeding the context window
        var promptTokens = tokenize(finalPrompt, addBOS: true)
        guard !promptTokens.isEmpty else {
            continuation.yield("[Error: tokenization returned empty]")
            continuation.finish()
            return
        }

        let maxPromptTokens = max(1, nCtxTokens - min(maxTokens, nCtxTokens / 2))
        if promptTokens.count > maxPromptTokens {
            let pCount = promptTokens.count
            Task { @MainActor in
                LogManager.shared.log("LlamaRunner: Warning - Prompt tokens (\(pCount)) exceed safe threshold (\(maxPromptTokens)). Truncating.")
            }
            let bos = promptTokens[0]
            promptTokens = [bos] + Array(promptTokens.suffix(maxPromptTokens - 1))
        }
        lastPromptTokenCount = promptTokens.count

        // 2. Reuse the KV cache across turns when this turn's freshly-tokenized prompt is a
        // verified, exact extension of exactly what's still resident from the previous turn —
        // see `residentTokens`'s doc comment for the full correctness argument and why this is
        // deliberately narrow (all-or-nothing) rather than a general prefix diff. Any other case
        // — first message, edited/regenerated history, a truncated prompt, or simply no match —
        // falls back to the original, always-correct full clear.
        let commonPrefixLen = Self.commonPrefixLength(residentTokens, promptTokens)
        let reusesCache = commonPrefixLen == residentTokens.count
            && commonPrefixLen > 0
            && commonPrefixLen < promptTokens.count

        if reusesCache {
            LogManager.shared.log("LlamaRunner: reusing \(commonPrefixLen)/\(promptTokens.count) cached prompt tokens from the previous turn")
        } else if let mem = llama_get_memory(ctx) {
            // Clear KV cache to prevent crashes across multiple prompts
            llama_memory_clear(mem, true)
        }

        // 3. Prefill (evaluate the prompt, or just the new suffix when reusing) in chunks of n_batch
        let batchSize = Int(llama_n_batch(ctx))
        var batch = llama_batch_init(Int32(batchSize), 0, 1)
        defer { llama_batch_free(batch) }

        var batchStart = reusesCache ? commonPrefixLen : 0
        while batchStart < promptTokens.count {
            let chunkEnd = min(promptTokens.count, batchStart + batchSize)
            let chunkLen = chunkEnd - batchStart

            for i in 0..<chunkLen {
                let tokenIdx = batchStart + i
                batch.token[i] = promptTokens[tokenIdx]
                batch.pos[i] = Int32(tokenIdx)
                batch.n_seq_id[i] = 1
                if let seqIdPtr = batch.seq_id[i] {
                    seqIdPtr.pointee = 0
                }
                batch.logits[i] = (tokenIdx == promptTokens.count - 1) ? 1 : 0
            }
            batch.n_tokens = Int32(chunkLen)

            // Metal's backend (ggml-metal.m) is Objective-C and creates autoreleased command
            // buffers/encoders inside `llama_decode`. Nothing here drains the pool between
            // calls otherwise — this loop runs entirely between Swift concurrency suspension
            // points, so those objects would pile up for the whole prefill instead of being
            // reclaimed chunk by chunk, and a big enough prompt could exhaust the GPU's working
            // set on transient allocations alone before generation even starts.
            let prefillResult = autoreleasepool { llama_decode(ctx, batch) }
            if prefillResult != 0 {
                handleDecodeFailure(stage: "prefill")
                continuation.yield("\n\n[The model ran out of GPU memory and had to be unloaded. Reload it, or pick a smaller one — this device can't run it at this context size.]")
                continuation.finish()
                return
            }
            batchStart += batchSize
        }
        // The cache now holds exactly `promptTokens` end-to-end — whether by the full reprefill
        // above or by keeping the reused prefix and decoding only the new suffix — so this is
        // what the *next* turn's reuse check compares against. Kept in lockstep with `nPos` from
        // here on (see the context-shift and per-token decode sections below).
        residentTokens = promptTokens

        // 4. Autoregressive decoding with temperature + top-p sampling to prevent repetition
        let nVocab = Int(llama_vocab_n_tokens(vocab))

        // IMPORTANT: Reset nPos each generation — previously unbounded growth caused crashes!
        var nPos = Int32(promptTokens.count)
        var generatedCount = 0

        // Sampling parameters — temperature is passed in dynamically
        let topP: Float = 0.9
        let repeatPenalty: Float = 1.1
        var recentTokens = [llama_token]()

        var singleBatch = llama_batch_init(1, 0, 1)
        defer { llama_batch_free(singleBatch) }

        var accumulatedOutput = ""
        /// Characters of `accumulatedOutput` already handed to the UI. Trails the accumulated
        /// text whenever the tail might be the opening of a turn marker.
        var yieldedCharCount = 0

        // MARK: Thinking-Block Suppression
        // Gemma and similar models emit internal reasoning inside thinking tags
        // (<think>, <channel>analysis, <|channel|>analysis<|message|>, etc.) before the
        // actual response. These tokens are silently consumed — they do NOT count against
        // maxTokens and are NOT streamed to the UI. This gives the model its full token
        // budget for the actual response instead of burning it on reasoning.
        //
        // Different models use wildly different delimiters for this, so instead of chasing
        // every model's exact tag string we key off vocabulary inside any bracketed tag
        // (<tag>, </tag>, or <|tag|>) and exit either on a matching close tag or a
        // transition to a non-reasoning channel (final/message/response/answer) — the latter
        // is how Harmony-style "channel" formats (gpt-oss, some Gemma fine-tunes) signal the
        // switch from internal analysis to the visible reply, since they have no close tag.
        var hasEnteredThinkingBlock = false
        var hasExitedThinkingBlock  = false
        var isBareLabelBlock = false  // true when the current block has no reliable close tag
        var thinkingTagBuffer = ""  // Rolling window for tag detection
        let reasoningTagRegex = try? NSRegularExpression(
            pattern: "<\\|?\\s*(think|thought|thinking|reflect|reason|channel|analysis|internal|scratchpad|deliberat)",
            options: [.caseInsensitive]
        )
        let transitionTagRegex = try? NSRegularExpression(
            pattern: "<\\|?/?\\s*(final|message|response|answer)[a-z]*\\s*\\|?>",
            options: [.caseInsensitive]
        )
        // Some fine-tunes echo bare (untagged) scaffolding labels from their own training
        // format instead of (or chained alongside) proper tags — e.g. "/Style Check: ...",
        // "**My internal monologue:** ...", "*** \n **Target Response Vibe:** ...", or
        // "*(Generating response...)*" — with no reliable bracket/tag structure at all, so
        // the detector above never sees them and they print as if they were the actual
        // answer. This model in particular invents new wording for these on nearly every
        // message, so wording alone can't keep up — these branches key off the recurring
        // *shapes* instead: a slash-command header ("/Word Word:"), a decorative "***"
        // separator line, any bold/italic-wrapped header ending in a colon (however it's
        // worded), and known self-referential/stage-direction phrases in other wrappers.
        let bareLabelRegex = try? NSRegularExpression(
            pattern: "(?:^|\\n)\\s*(?:" +
                "\\*{3,}|" +
                "/[A-Za-z][A-Za-z ]{2,29}:|" +
                "\\*{1,2}[A-Za-z][A-Za-z ,]{2,39}:\\*{0,2}|" +
                "[\\*\"'\\[\\(]{1,2}\\s*(?:self[- _]?correction|self[- _]?review|internal monologue|internal reasoning|response generation|chain of thought|style check|tone check|voice check|persona check|vibe check|character check|generating response|generating\\.\\.\\.)" +
                ")",
            options: [.caseInsensitive]
        )
        func regexMatches(_ regex: NSRegularExpression?, in s: String) -> Bool {
            guard let regex else { return false }
            return regex.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil
        }
        // Hard caps on unbudgeted thinking tokens — without these, a model that never emits
        // (or mis-emits) a close tag runs away until one of these limits is hit. Suppressed
        // tokens are never yielded to the UI, so nothing visibly updates while this runs —
        // on a slow device a large token-only cap can look exactly like a hang for minutes
        // at a time. A wall-clock cap bounds that regardless of token throughput.
        var thinkingTokenCount = 0
        let maxThinkingTokens = max(128, min(nCtxTokens / 8, 512))
        let maxThinkingSeconds: Double = 20.0

        // The loop runs while:
        //  • generatedCount < maxTokens (normal response budget), OR
        //  • the model is still inside a thinking block that started before any real content
        //    (so thinking doesn't consume the response budget)
        while generatedCount < maxTokens || (hasEnteredThinkingBlock && !hasExitedThinkingBlock && generatedCount == 0) {
            await Task.yield()
            guard !isCancelled else { break }

            guard let logitsPtr = llama_get_logits_ith(ctx, -1) else { break }

            // Operate directly on llama.cpp's own logits buffer instead of copying all `nVocab`
            // floats into a fresh Swift `Array` on every single token — on a large-vocab model
            // (128K+ tokens for some catalog models) that copy alone was real, measurable
            // per-token cost. Safe to mutate in place: `llama_get_logits_ith` returns a buffer
            // owned by the context that's only valid until the *next* `llama_decode` call, which
            // is exactly this loop iteration's own lifetime for it — nothing else reads it, and
            // the next decode overwrites it wholesale regardless of what's left here.
            //
            // (A native `llama_sampler_chain`/`llama_sampler_init_*` chain was investigated as a
            // further step — the symbols are genuinely linked and callable here, confirmed against
            // llama.swift's vendored headers. It wasn't adopted: `llama_sampler_init_penalties`
            // applies its repeat penalty once per *distinct* token in the window, where this
            // function's penalty below compounds once per *occurrence* — a real behavioural
            // difference, not just an implementation detail — and reproducing this function's
            // exact semantics natively would mean hand-writing a custom `llama_sampler_i` from
            // Swift, which isn't a change to make with no way to verify actual generations here.)
            let logits = UnsafeMutableBufferPointer(start: logitsPtr, count: nVocab)

            // Reject a distribution that carries no information before sampling from it.
            //
            // The NaN/inf guard further down catches a compute buffer full of garbage. It does
            // not catch the other shape this failure takes: a buffer that came back *zeroed*, or
            // otherwise constant. Every value is finite, so nothing downstream objects — but a
            // flat distribution over a 128K vocabulary means top-p keeps essentially the whole
            // vocabulary and sampling returns uniformly random tokens. The output is fluent-
            // looking token salad, URL fragments and stray words, and it looks like the model
            // being bad rather than the backend being broken.
            //
            // A healthy language model is sharply peaked; the gap between its best and worst
            // logit is tens of units, never a rounding error. Measured on the raw values, before
            // the repeat penalty and temperature below reshape them.
            var minRaw = Float.greatestFiniteMagnitude
            var maxRaw = -Float.greatestFiniteMagnitude
            for value in logits where value.isFinite {
                minRaw = min(minRaw, value)
                maxRaw = max(maxRaw, value)
            }
            let rawSpread = maxRaw - minRaw

            if generatedCount == 0 {
                // One line per generation, at the first token: enough to tell a broken backend
                // from a bad model in a diagnostic log, without flooding it.
                //
                // The spread alone doesn't separate the two failures that matter. A backend
                // producing garbage and a model that is simply incoherent can both yield a
                // healthy-looking range; what distinguishes them is the *shape* at the top. A
                // working model concentrates most of its mass in a handful of tokens, so the
                // top few probabilities and what they actually decode to answer the question
                // outright: a confident, sensible top token means the engine is fine and the
                // weights are to blame, while a top token holding a fraction of a percent means
                // the distribution is flat and the tokens coming out are close to uniform noise
                // no matter how sane the sampler is.
                let peak = logits.enumerated()
                    .filter { $0.element.isFinite }
                    .sorted { $0.element > $1.element }
                    .prefix(5)
                let shifted = peak.map { expf($0.element - maxRaw) }
                // Normalised against the whole vocabulary, not just these five, so the figures
                // are true probabilities rather than a ratio among the leaders.
                var total: Float = 0
                for value in logits where value.isFinite { total += expf(value - maxRaw) }
                let summary = zip(peak, shifted).map { entry, weight -> String in
                    var buf = [CChar](repeating: 0, count: 128)
                    let count = llama_token_to_piece(vocab, llama_token(entry.offset), &buf, 128, 0, true)
                    let piece = count > 0
                        ? String(decoding: buf[0..<Int(count)].map { UInt8(bitPattern: $0) }, as: UTF8.self)
                        : "?"
                    let percent = total > 0 ? weight / total * 100 : 0
                    return String(format: "%@ %.2f%%", piece.debugDescription, percent)
                }.joined(separator: ", ")

                // Copied to `let`s rather than capturing `minRaw`/`maxRaw` themselves — those are
                // `var`s that a `Task { @MainActor in }` closure would otherwise capture by
                // reference, which Swift 6's strict concurrency checking rejects outright since
                // nothing here proves they aren't mutated again before the detached task runs.
                let loggedMinRaw = minRaw, loggedMaxRaw = maxRaw
                Task { @MainActor in
                    LogManager.shared.log(String(format: "Sampler health: logit range %.3f … %.3f (spread %.3f) over %d tokens",
                                                 loggedMinRaw, loggedMaxRaw, rawSpread, nVocab))
                    LogManager.shared.log("Sampler health: top tokens — \(summary)")
                }
            }

            if !rawSpread.isFinite || rawSpread < 0.01 {
                Task { @MainActor in
                    LogManager.shared.log(String(format: "LlamaRunner: degenerate logits (spread %.6f) — compute buffer is not producing real output; stopping.", rawSpread))
                }
                continuation.yield("\n\n[Generation stopped — the model's compute output came back empty, which means the GPU/CPU split isn't producing real results rather than the model writing badly. Try reloading it, or a smaller model.]")
                break
            }

            // Apply repeat penalty
            for tok in recentTokens {
                let idx = Int(tok)
                if idx >= 0 && idx < nVocab {
                    if logits[idx] > 0 {
                        logits[idx] /= repeatPenalty
                    } else {
                        logits[idx] *= repeatPenalty
                    }
                }
            }

            // Temperature scaling
            var tempInvApplied: Float = 1.0
            if temperature > 0 && temperature != 1.0 {
                tempInvApplied = 1.0 / Float(temperature)
                for i in 0..<nVocab { logits[i] *= tempInvApplied }
            }

            // `maxRaw` (computed above, before the repeat penalty) is a safe upper bound for the
            // post-penalty, post-temperature max without a second full-vocabulary scan: the
            // repeat-penalty loop just above only ever pulls a `recentTokens` logit *down* toward
            // zero (a positive value divided by `repeatPenalty > 1`, a negative value multiplied
            // further negative), never up, so nothing anywhere can exceed `maxRaw` afterward, and
            // scaling by a positive `tempInvApplied` preserves that bound. Softmax only needs *a*
            // safe upper bound to shift by for numerical stability, not the exact max, so this is
            // correct rather than approximate — it just avoids re-scanning every logit again.
            let maxLogit = maxRaw * tempInvApplied

            // Optimization: Filter out incredibly improbable logits before expf and sorting (O(N log N) -> O(1))
            let logitThreshold = maxLogit - 12.0

            var validLogits: [(Int, Float)] = []
            validLogits.reserveCapacity(1000)
            for i in 0..<nVocab {
                let val = logits[i]
                if val > logitThreshold {
                    validLogits.append((i, val))
                }
            }

            // Softmax + top-p nucleus sampling, over the filtered candidates only.
            //
            // Sorting by raw logit descending yields the same order as sorting by probability —
            // `expf` is monotonic and the normalization below is one shared positive scale factor
            // — so the sort runs once, before softmax, instead of building a full probability
            // array first and sorting *that*. `sumExp` still has to see every filtered candidate
            // (top-p is defined against the whole filtered probability mass), but computing each
            // candidate's actual probability can stop as soon as the nucleus crosses `topP`,
            // which for a peaked distribution is usually a handful of entries, not the whole
            // filtered set.
            validLogits.sort { $0.1 > $1.1 }
            var sumExp: Float = 0
            for (_, val) in validLogits { sumExp += expf(val - maxLogit) }

            var nucleusSum: Float = 0
            var nucleus: [(Int, Float)] = []
            if sumExp > 0 {
                for (idx, val) in validLogits {
                    let prob = expf(val - maxLogit) / sumExp
                    nucleus.append((idx, prob))
                    nucleusSum += prob
                    if nucleusSum >= topP { break }
                }
            }

            // Sample from nucleus.
            //
            // Guarded because `Float.random(in:)` traps on an empty range. When every logit
            // comes back as -inf or NaN — which is what a broken compute buffer looks like,
            // and this backend has produced garbage tensors before (see the flash-attention
            // notes in `SDWrapper`) — the threshold filter above keeps nothing, the nucleus is
            // empty, and `nucleusSum` is 0. That trapped with "Range requires lowerBound <
            // upperBound" and killed the app mid-answer. A model returning unusable numbers has
            // to be a survivable state, not an uncatchable crash. `nucleusSum` already holds the
            // sum of every probability actually appended to `nucleus` above (accumulated in
            // lockstep in that same loop), so it's reused here rather than recomputed via reduce.
            var sampledId: Int
            if nucleusSum.isFinite, nucleusSum > 0 {
                var rand = Float.random(in: 0..<nucleusSum)
                sampledId = nucleus.first?.0 ?? 0
                for (idx, prob) in nucleus {
                    rand -= prob
                    if rand <= 0 {
                        sampledId = idx
                        break
                    }
                }
            } else {
                // Greedy fallback — highest finite logit, skipping NaN/inf entries.
                var bestIdx = -1
                var bestVal = -Float.greatestFiniteMagnitude
                for i in 0..<nVocab {
                    let value = logits[i]
                    if value.isFinite && value > bestVal {
                        bestVal = value
                        bestIdx = i
                    }
                }
                guard bestIdx >= 0 else {
                    // Not a single usable number in the whole distribution. Stop cleanly and
                    // say so rather than emitting noise or dying.
                    Task { @MainActor in
                        LogManager.shared.log("LlamaRunner: sampler got unusable logits (all NaN/inf) — stopping generation.")
                    }
                    continuation.yield("\n\n[Generation stopped — the model returned unusable output. Try reloading the model, or loading a different one.]")
                    break
                }
                sampledId = bestIdx
            }

            let bestId = llama_token(sampledId)

            // Covers every end-of-generation token the model defines — not just the primary
            // EOS (<|end_of_text|>) but also model-specific end-of-turn tokens like Llama 3's
            // <|eot_id|>. Checking only the primary EOS missed <|eot_id|> entirely, so the
            // model's own turn-end signal was ignored and it kept generating past it,
            // hallucinating further fake turns instead of stopping.
            if llama_vocab_is_eog(vocab, bestId) { break }

            // Detokenize
            var tokenBuf = [CChar](repeating: 0, count: 256)
            let nChars = llama_token_to_piece(vocab, bestId, &tokenBuf, 256, 0, false)
            var yieldedRealToken = false
            if nChars > 0 {
                let piece = String(bytes: tokenBuf.prefix(Int(nChars)).map { UInt8(bitPattern: $0) }, encoding: .utf8) ?? ""
                if !piece.isEmpty {
                    accumulatedOutput += piece

                    // Update the rolling tag-detection buffer. Wide enough to hold multi-part
                    // delimiters like "<|channel|>analysis<|message|>" plus surrounding context.
                    thinkingTagBuffer += piece
                    if thinkingTagBuffer.count > 96 { thinkingTagBuffer = String(thinkingTagBuffer.suffix(96)) }
                    let bufLower = thinkingTagBuffer.lowercased()

                    // Detect a preamble block opening. Only checked before any real content
                    // has been produced (generatedCount == 0) — this is re-armable (not a
                    // one-time latch) so a model that chains multiple preamble blocks back
                    // to back (e.g. "/Style Check: ..." immediately followed by
                    // "**My internal monologue:** ...") gets each one caught in turn, rather
                    // than only the first.
                    if !hasEnteredThinkingBlock && generatedCount == 0 {
                        if regexMatches(reasoningTagRegex, in: thinkingTagBuffer) {
                            hasEnteredThinkingBlock = true
                            hasExitedThinkingBlock = false
                            isBareLabelBlock = false
                        } else if regexMatches(bareLabelRegex, in: thinkingTagBuffer) {
                            hasEnteredThinkingBlock = true
                            hasExitedThinkingBlock = false
                            isBareLabelBlock = true
                        }
                    }
                    // Detect the block's end. Tagged blocks only end on a real close/transition
                    // tag — never on a blank line, since genuine multi-paragraph reasoning
                    // inside e.g. <think>...</think> can itself contain blank lines. Bare-label
                    // blocks have no reliable close marker at all, so a blank line is the best
                    // available signal that the label's aside has ended.
                    if hasEnteredThinkingBlock && !hasExitedThinkingBlock {
                        let closedByTag = bufLower.contains("</think") || bufLower.contains("</thought") ||
                            bufLower.contains("</reflect") || bufLower.contains("</reason") ||
                            bufLower.contains("</channel") || bufLower.contains("</analysis") ||
                            bufLower.contains("</internal") || bufLower.contains("</scratchpad") ||
                            regexMatches(transitionTagRegex, in: thinkingTagBuffer)
                        let closedByBlankLine = isBareLabelBlock && thinkingTagBuffer.contains("\n\n")
                        if closedByTag || closedByBlankLine {
                            hasExitedThinkingBlock = true
                        }
                    }

                    let inThinkingBlock = hasEnteredThinkingBlock && !hasExitedThinkingBlock

                    if !inThinkingBlock {
                        // Real content — check stop strings and stream to UI. Properly
                        // converted instruct GGUFs set their turn-end token as the model's
                        // real EOS (handled above via llama_vocab_eos), so this is a text-level
                        // fallback for conversions that don't — covering Mistral ([INST]),
                        // ChatML (<|im_end|>), Gemma (<start_of_turn>user), Llama 3
                        // (<|eot_id|>, <|start_header_id|>user), and a couple of generic forms.
                        let lower = accumulatedOutput.lowercased()

                        // Angle-bracket turn markers, matched anywhere rather than only at the
                        // very end. Two reasons the suffix-only form wasn't enough:
                        //
                        // `<|assistant|>` was missing from the list entirely, which is what let
                        // OLMoE finish its answer, open a fresh turn and then write *both* sides
                        // of the conversation — it emits that marker as ordinary text because
                        // it isn't a single special token in this vocabulary, so
                        // `llama_vocab_is_eog` above never sees it.
                        //
                        // And a marker only lands exactly at the end if the final piece stops
                        // there. Here it arrived as `<|assistant|>\n`, so the suffix test failed
                        // even for the markers that were listed. Searching the text instead, and
                        // cutting at the marker, catches it wherever the tokeniser puts the
                        // boundary.
                        //
                        // These forms never occur in ordinary prose, so a substring match is
                        // safe. `[inst]` and `user:` do occur, so they stay suffix-only below.
                        let turnMarkers = ["<|assistant|>", "<|user|>", "<|system|>",
                                           "<|im_start|>", "<|im_end|>", "<|endoftext|>",
                                           "<|eot_id|>", "<|start_header_id|>", "<|end|>",
                                           "<|return|>", "<|call|>", "<start_of_turn>"]
                        if let cut = turnMarkers.compactMap({ lower.range(of: $0)?.lowerBound }).min() {
                            // Everything before the marker is real content, and some of it may
                            // still be held back (see below), so flush that before stopping and
                            // drop the marker and everything after it.
                            let held = accumulatedOutput.index(accumulatedOutput.startIndex,
                                                               offsetBy: yieldedCharCount)
                            if cut > held {
                                continuation.yield(String(accumulatedOutput[held..<cut]))
                                yieldedRealToken = true
                            }
                            accumulatedOutput = String(accumulatedOutput[..<cut])
                            break
                        }

                        if lower.hasSuffix("[inst]") || lower.hasSuffix("user:") {
                            break
                        }

                        // Hold back any tail that could still turn into a marker.
                        //
                        // A tokeniser rarely hands over `<|assistant|>` in one piece — it comes
                        // as `<`, `|`, `assistant`, `|`, `>`. Yielding each piece the moment it
                        // arrives means the first four are already on screen by the time the
                        // fifth completes the match, so the marker gets cut from the transcript
                        // but its opening still shows. Emitting only up to the last position
                        // that cannot begin a marker defers those characters until they're
                        // proven to be ordinary text.
                        let pendingPrefix = turnMarkers.reduce(0) { longest, marker in
                            var length = min(marker.count - 1, lower.count)
                            while length > longest {
                                if lower.hasSuffix(marker.prefix(length)) { return length }
                                length -= 1
                            }
                            return longest
                        }
                        let safeCount = accumulatedOutput.count - pendingPrefix
                        if safeCount > yieldedCharCount {
                            let from = accumulatedOutput.index(accumulatedOutput.startIndex,
                                                               offsetBy: yieldedCharCount)
                            let to = accumulatedOutput.index(accumulatedOutput.startIndex,
                                                             offsetBy: safeCount)
                            continuation.yield(String(accumulatedOutput[from..<to]))
                            yieldedCharCount = safeCount
                            yieldedRealToken = true
                        }
                    }
                    // Thinking-block tokens: silently consumed — no UI yield, no budget decrement.
                    // Still reported via onThinkingProgress so the UI can show live movement
                    // instead of an indistinguishable-from-hung frozen "Thinking..." state.
                    else {
                        thinkingTokenCount += 1
                        onThinkingProgress(thinkingTokenCount)
                        let elapsedThinking = CFAbsoluteTimeGetCurrent() - genStartTime
                        if thinkingTokenCount >= maxThinkingTokens || elapsedThinking >= maxThinkingSeconds {
                            hasExitedThinkingBlock = true
                            let reason = thinkingTokenCount >= maxThinkingTokens ? "\(maxThinkingTokens) tokens" : "\(Int(elapsedThinking))s"
                            Task { @MainActor in LogManager.shared.log("LlamaRunner: Thinking/preamble suppression exceeded its budget (\(reason)) without closing — forcing exit.") }
                        }
                    }

                    // A block just closed and no real content has been shown yet — re-arm
                    // detection for a possible chained follow-up block, and reset the buffer
                    // so leftover text from the block we just closed can't immediately
                    // re-match the same pattern.
                    if hasEnteredThinkingBlock && hasExitedThinkingBlock && generatedCount == 0 {
                        hasEnteredThinkingBlock = false
                        isBareLabelBlock = false
                        thinkingTagBuffer = ""
                    }
                }
            }

            // Track recent tokens for repeat penalty (sliding window of last 64)
            recentTokens.append(bestId)
            if recentTokens.count > 64 { recentTokens.removeFirst() }

            // Context window management: instead of hard-stopping generation once the KV
            // cache fills up, shift the oldest tokens out and keep going — the standard
            // "context shift" technique llama.cpp's own server/main examples use. This can
            // still be reached even though the prompt was pre-budgeted to leave room for
            // maxTokens, because suppressed thinking/scaffold-label tokens above consume
            // real KV cache space without counting against that budget.
            if nPos >= Int32(nCtxTokens) - 1 {
                let nKeep = Int32(min(64, promptTokens.count))
                let nDiscard = max(Int32(1), (nPos - nKeep) / 2)
                if let mem = llama_get_memory(ctx),
                   llama_memory_can_shift(mem),
                   nKeep + nDiscard < nPos,
                   llama_memory_seq_rm(mem, 0, nKeep, nKeep + nDiscard) {
                    llama_memory_seq_add(mem, 0, nKeep + nDiscard, nPos, -nDiscard)
                    nPos -= nDiscard
                    // Mirror the same eviction into `residentTokens` — see its doc comment. Must
                    // track `nPos` exactly, since that's what the next turn's prefix-reuse check
                    // trusts as "what the cache actually contains right now."
                    residentTokens.removeSubrange(Int(nKeep)..<Int(nKeep + nDiscard))
                    onContextTruncated()
                    Task { @MainActor in LogManager.shared.log("LlamaRunner: Context window full — dropped \(nDiscard) oldest tokens to keep generating.") }
                } else {
                    // Memory doesn't support shifting (or the shift failed) — fall back to
                    // the old safe behavior rather than risk decoding into a full cache.
                    Task { @MainActor in LogManager.shared.log("LlamaRunner: Hard context limit reached and shifting unavailable. Stopping generation.") }
                    continuation.yield("\n\n[System: Context window limit reached. Generation stopped.]")
                    break
                }
            }

            // Advance KV cache
            singleBatch.token[0] = bestId
            singleBatch.pos[0]   = nPos
            singleBatch.n_seq_id[0] = 1
            if let seqIdPtr = singleBatch.seq_id[0] {
                seqIdPtr.pointee = 0
            }
            singleBatch.logits[0] = 1
            singleBatch.n_tokens  = 1

            // Same reasoning as the prefill decode above, and more pressing here: this call
            // runs once per generated token, so on a long response the pool would otherwise
            // never drain for the whole stream. Draining it every token is what keeps a
            // multi-turn conversation from accumulating leftover Metal buffers turn over turn
            // until a later, unrelated decode fails with an out-of-memory command buffer error.
            let decodeResult = autoreleasepool { llama_decode(ctx, singleBatch) }
            if decodeResult != 0 {
                handleDecodeFailure(stage: "generation")
                continuation.yield("\n\n[The model ran out of GPU memory and had to be unloaded. Reload it, or pick a smaller one — this device can't run it at this context size.]")
                break
            }

            nPos += 1
            // `bestId` is now actually resident in the cache at the position that was just
            // decoded — mirror it into `residentTokens` in the same lockstep as `nPos` above.
            residentTokens.append(bestId)
            // Only increment the response budget counter for real (non-thinking) tokens
            if yieldedRealToken { generatedCount += 1 }
        }

        continuation.finish()
    }

    deinit {
        if let ctx = context { llama_free(ctx) }
        if let mdl = model   { llama_model_free(mdl) }
        model = nil
        context = nil
    }
}

// MARK: - LLMManager

@MainActor
class LLMManager: ObservableObject {
    @Published var loadState: ModelLoadState = .unloaded
    @Published var isGenerating: Bool = false
    /// Set by `cancelGeneration()`, read (and expected to be reset) by the caller's `onComplete`
    /// handler in `generateResponse`. Image cancellation already leaves an explicit
    /// "[Image generation cancelled.]" marker in the chat; text cancellation used to just freeze
    /// whatever had streamed so far with nothing distinguishing it from a normal, complete reply
    /// — this is what lets the caller tell the two apart and mark the message accordingly.
    private(set) var wasCancelled: Bool = false
    @Published var systemMemoryGB: Double = 0.0
    @Published var activeModelURL: URL? = nil
    @Published var generationSpeed: Double = 0.0
    /// Context window actually applied to the loaded model (post safeContextTokens clamp),
    /// distinct from `contextTokenLimit` which is just the user's requested setting.
    @Published var loadedContextWindow: Int = 0

    /// Context the loaded model was trained for, 0 when nothing is loaded.
    ///
    /// A hard property of the weights: asking for more than this doesn't extend the model's
    /// reach, it just allocates cache the model cannot use. Kept separate from
    /// `loadedContextWindow` (what was actually applied) so the ceiling can explain *why* it is
    /// where it is.
    @Published var loadedTrainedContext: Int = 0
    /// Only meaningful when `activeBackend == .coreML`. True for a real sliding-window cache
    /// (Llama 3.2 1B, via `ChunkedPipelineCoreMLEngine`) that keeps generating past
    /// `loadedContextWindow` by forgetting the earliest turns; false for a hard-stop fixed window
    /// (OpenELM, via `SingleWindowCoreMLEngine`) that refuses once it's reached. See
    /// `CoreMLRunner.isSlidingWindow`.
    @Published var coreMLContextIsSliding: Bool = false
    /// Prompt tokens + tokens generated so far in the current/most recent turn — the live
    /// "how full is the context window right now" figure shown in the status bar.
    @Published var contextTokensUsed: Int = 0
    /// Tokens generated in the current/most recent response, for verifying against maxTokens.
    @Published var currentResponseTokenCount: Int = 0
    /// Live count of suppressed "thinking"/preamble tokens the model has produced so far in
    /// the current turn but not yet resolved into a real answer. These are never yielded to
    /// the chat stream, so without this the UI has no visibility into that work at all —
    /// shown so "Thinking…" reflects live progress instead of looking frozen/hung.
    @Published var thinkingTokensUsed: Int = 0
    /// True while the current/most recent turn had to drop the oldest conversation history
    /// (or shift the oldest tokens out of the live KV cache) to fit the context window,
    /// rather than ever hard-stopping generation.
    @Published var isContextTruncating: Bool = false
    @Published var modelSupportsVision: Bool = false
    /// Which engine `activeModelURL` is currently running on. Only meaningful while
    /// `isModelLoaded`/`loadState == .loading` — reset to `.llamaCpp` on unload.
    @Published var activeBackend: LLMBackendKind = .llamaCpp

    /// Bumped at the start of every `loadModel`/`loadCoreMLModel` call and on explicit unload.
    /// Each load `Task` captures its own value at launch and checks it against the current one
    /// before applying state after any `await` — two quick taps (or a load racing an unload)
    /// no longer risk a stale, slower call clobbering a newer one's final `loadState`.
    private var loadGeneration: Int = 0

    var isModelLoaded: Bool {
        if case .loaded = loadState { return true }
        return false
    }
    @Published var maxTokens: Int = UserDefaults.standard.object(forKey: "maxTokens") as? Int ?? 512 {
        didSet { UserDefaults.standard.set(maxTokens, forKey: "maxTokens") }
    }
    
    @Published var contextTokenLimit: Int = UserDefaults.standard.object(forKey: "contextTokenLimit") as? Int ?? 8192 {
        didSet { UserDefaults.standard.set(contextTokenLimit, forKey: "contextTokenLimit") }
    }
    
    @Published var temperature: Double = UserDefaults.standard.object(forKey: "temperature") as? Double ?? 0.85 {
        didSet { UserDefaults.standard.set(temperature, forKey: "temperature") }
    }
    
    @Published var highVariabilityEnabled: Bool = UserDefaults.standard.bool(forKey: "highVariabilityEnabled") {
        didSet { UserDefaults.standard.set(highVariabilityEnabled, forKey: "highVariabilityEnabled") }
    }
    
    // Reconstructed from the current Models directory + a stored filename rather than a
    // stored absolute path — the sandbox container's UUID isn't guaranteed stable across an
    // app reinstall, so a remembered absolute path can silently stop resolving (the launch
    // auto-load prompt then fails instantly, before llama.cpp even logs anything, since the
    // file simply isn't at that stale path) while Settings works fine because it discovers
    // models by scanning the current Models directory fresh instead of trusting a saved path.
    @Published var lastUsedModelPath: String? = LLMManager.resolveLastUsedModelPath() {
        didSet {
            if let path = lastUsedModelPath {
                UserDefaults.standard.set(URL(fileURLWithPath: path).lastPathComponent, forKey: "lastUsedModelFileName")
            } else {
                UserDefaults.standard.removeObject(forKey: "lastUsedModelFileName")
            }
        }
    }

    /// `lastUsedModelFileName` is shared by both chat backends (set from both `loadModel`'s GGUF
    /// success path and `loadCoreMLModel`'s), but this used to resolve it only against
    /// `AppFiles.models` — the GGUF directory. A Core ML model's saved filename was never found
    /// there (it installs under `AppFiles.coreMLModels`, as either a `.mlpackage` or a plain-named
    /// directory), so `lastUsedModelPath` silently resolved to `nil` for anyone whose last session
    /// ended on one, and the "load previous model?" prompt never appeared for them at all — no
    /// error, the feature just quietly didn't apply. Checking both install directories is what
    /// actually covers both backends this property is shared between.
    private static func resolveLastUsedModelPath() -> String? {
        func resolved(fileName: String) -> String? {
            for directory in [AppFiles.models, AppFiles.coreMLModels] {
                let url = directory.appendingPathComponent(fileName)
                if FileManager.default.fileExists(atPath: url.path) {
                    return url.path
                }
            }
            return nil
        }

        if let fileName = UserDefaults.standard.string(forKey: "lastUsedModelFileName") {
            return resolved(fileName: fileName)
        }

        // One-time migration from the old absolute-path storage format.
        if let oldPath = UserDefaults.standard.string(forKey: "lastUsedModelPath") {
            let fileName = URL(fileURLWithPath: oldPath).lastPathComponent
            if let path = resolved(fileName: fileName) {
                UserDefaults.standard.set(fileName, forKey: "lastUsedModelFileName")
                UserDefaults.standard.removeObject(forKey: "lastUsedModelPath")
                return path
            }
        }

        return nil
    }
    
    // MARK: - Device context ceiling

    /// The largest context window this *hardware* can be asked for, before any model is loaded.
    ///
    /// Separate from `safeContextLimit`, which is model-aware and only meaningful once something
    /// is loaded. This is the hard ceiling that follows from the device's physical RAM, and it's
    /// what the settings slider is bounded by — a 4 GB iPhone should never be able to request a
    /// 32k context in the first place, rather than accepting the number and having it silently
    /// clamped at load time to something a quarter the size.
    /// Upper bound offered by the Settings slider for this device.
    ///
    /// Only the selectable range — `LlamaRunner.safeContextTokens` still decides what a given
    /// model actually gets at load time, against real memory, and reports when it lowers the
    /// figure. These tiers were raised once that budget stopped double-charging for the weights:
    /// the top tier previously stopped at 32768 even on a 12 GB device, capping a 128k-capable
    /// model well below what its memory could hold.
    var deviceContextCeiling: Int {
        switch systemMemoryGB {
        case ..<3.5:  return 4096    // 2–3 GB devices: iPhone SE, older SoCs
        case ..<5.5:  return 8192    // 4 GB: iPhone 12/13/14 non-Pro
        case ..<7.5:  return 16384   // 6 GB: recent Pro models
        case ..<9.5:  return 32768   // 8 GB
        default:      return 65536   // 12 GB+: iPhone 17 Pro Max and later
        }
    }

    /// The ceiling actually in force right now: hardware bound, tightened by the loaded model's
    /// own memory profile once one exists.
    var effectiveContextCeiling: Int {
        if case .loaded = loadState {
            // Three independent bounds, all of which have to hold: what the hardware can carry,
            // what this model's memory profile leaves room for, and what the model was trained
            // for. The last was missing, so the slider would offer 65,536 tokens against an
            // 8,192-token model, the load would quietly apply 8,192, and Settings would go on
            // reporting a number the user could never actually get.
            var ceiling = min(deviceContextCeiling, safeContextLimit)
            if loadedTrainedContext > 0 { ceiling = min(ceiling, loadedTrainedContext) }
            return ceiling
        }
        return deviceContextCeiling
    }

    /// Set when the stored limit had to be lowered, so Settings can explain the change rather
    /// than leaving the user staring at a number they didn't choose.
    @Published var contextLimitAutoAdjustedTo: Int? = nil

    /// Clamps the persisted context limit down to what the device (and, if loaded, the model)
    /// can actually sustain.
    ///
    /// Only ever lowers. If the user deliberately picked something smaller than the ceiling,
    /// that choice is theirs and is left alone — the point of this is to stop an impossible
    /// setting from surviving, not to push everyone to the maximum.
    @discardableResult
    func applyContextCeiling() -> Bool {
        let ceiling = effectiveContextCeiling
        guard contextTokenLimit > ceiling else {
            contextLimitAutoAdjustedTo = nil
            return false
        }
        let previous = contextTokenLimit
        contextTokenLimit = ceiling
        contextLimitAutoAdjustedTo = ceiling
        LogManager.shared.log("LLMManager: context limit lowered \(previous) → \(ceiling) (device RAM \(String(format: "%.1f", systemMemoryGB)) GB)")
        return true
    }

    /// Advisory ceiling shown in Settings — the point past which the slider warns.
    ///
    /// Budgets from the process's actual allowance (`os_proc_available_memory()`), matching what
    /// `LlamaRunner.safeContextTokens` will really apply at load time. It used to work from
    /// `systemMemoryGB * 0.40`, a fraction of *device* RAM, which bore no relation to what iOS
    /// had actually granted this process and warned far too early on a large-memory device.
    var safeContextLimit: Int {
        let availableGB = MemoryBudget.plannableHeadroomGB()
        let safetyMarginGB = MemoryBudget.safetyMarginGB(for: availableGB)

        // The per-token figures below were calibrated against an f16 KV cache. The runner now
        // applies a Q8_0 cache wherever the architecture allows it (see `KVCacheFormat`), at
        // 1.0625 bytes per element instead of 2.0, so leaving them unscaled would warn at
        // roughly half the context the load will really grant. A model whose head geometry
        // forces the f16 fallback makes this advisory number optimistic, which is harmless:
        // `safeContextTokens` still clamps at load time and `applyContextCeiling` re-clamps
        // afterwards, so the slider corrects itself rather than over-committing the device.
        let kvQuantFactor = (34.0 / 32.0) / 2.0

        switch loadState {
        case .loaded(_, let sizeGB):
            // Weights are already resident, so they're accounted for in `availableGB` — only the
            // compute buffers still need reserving on top of the margin.
            let availableForKV = availableGB - safetyMarginGB - 0.75
            if availableForKV <= 0 { return 2048 }
            let gbPer1kTokens = max(0.04, sizeGB * 0.02) * kvQuantFactor
            return max(2048, min(65536, Int((availableForKV / gbPer1kTokens) * 1000)))
        default:
            // Nothing loaded yet: leave room for a typical model plus its compute buffers.
            let availableForKV = availableGB - safetyMarginGB - 2.5
            if availableForKV <= 0 { return 4096 }
            let gbPer1kTokens = 0.08 * kvQuantFactor
            return max(2048, min(65536, Int((availableForKV / gbPer1kTokens) * 1000)))
        }
    }

    private let runner = LlamaRunner()
    private let coreMLRunner = CoreMLRunner()

    /// Tokens for the lifecycle observers registered below, so `deinit` can remove them —
    /// `NotificationCenter` does not deregister a block-based observer on its own. Mirrors the
    /// same fix on `DiffusionManager`'s memory-warning observer.
    private var willTerminateObserver: NSObjectProtocol?
    private var memoryWarningObserver: NSObjectProtocol?

    init() {
        self.systemMemoryGB = getPhysicalMemory()
        setupAppLifecycleObservers()
        // Runs before anything can read the setting. A limit persisted on one device (or from a
        // build with a higher default) is otherwise carried forward unchanged onto hardware that
        // can't honour it.
        applyContextCeiling()
    }

    deinit {
        if let willTerminateObserver { NotificationCenter.default.removeObserver(willTerminateObserver) }
        if let memoryWarningObserver { NotificationCenter.default.removeObserver(memoryWarningObserver) }
    }

    // MARK: - App Lifecycle - Clean unload on background/termination

    private func setupAppLifecycleObservers() {
        willTerminateObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.willTerminateNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task {
                await self?.runner.unload()
                await self?.coreMLRunner.unload()
            }
        }
        memoryWarningObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            guard let self else { return }
            Task {
                switch await MainActor.run(body: { self.activeBackend }) {
                case .llamaCpp:
                    await self.runner.requestCancel()
                    await self.runner.unloadModelOnly()
                case .coreML:
                    await self.coreMLRunner.requestCancel()
                    await self.coreMLRunner.unload()
                }
                await MainActor.run {
                    self.loadState = .failed(error: "System memory pressure — model unloaded safely. Reload when ready.")
                    self.activeModelURL = nil
                    self.modelSupportsVision = false
                }
            }
        }
    }

    // MARK: - Memory
    func getPhysicalMemory() -> Double {
        return Double(ProcessInfo.processInfo.physicalMemory) / (1024.0 * 1024.0 * 1024.0)
    }

    /// Real, current headroom before this process hits its dirty-memory limit — unlike
    /// getPhysicalMemory() (a constant), this reflects whatever else is using memory right
    /// now. Model loading should always budget against this, not total device RAM.
    func getAvailableMemoryGB() -> Double {
        // Plannable headroom, not the raw instantaneous reading — see `MemoryBudget`. This is
        // what gets passed to `LlamaRunner.safeContextTokens`, so it decides the context window.
        return MemoryBudget.plannableHeadroomGB()
    }

    func getModelSizeGB(at url: URL) -> Double {
        guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize else { return 0 }
        return Double(size) / (1024.0 * 1024.0 * 1024.0)
    }

    /// Process headroom a model of this size needs before `checkMemorySafety` will pass it.
    ///
    /// Derived from the same constants the check itself uses, so callers that need to *wait* for
    /// memory rather than merely test it can target the real threshold instead of guessing.
    /// Reloading the chat model after image generation is the case that matters: a fixed sleep
    /// after unloading a multi-gigabyte diffusion checkpoint was routinely too short, and the
    /// reload then failed its own safety check over memory that was about to come back.
    func memoryHeadroomNeededGB(forModelSizeGB modelSizeGB: Double) -> Double {
        // Clamped to what this device can actually reach. For a model that is going to stream
        // its weights, the fully-resident figure is unreachable by definition, and waiting on it
        // would spend the whole `waitForRelease` timeout before every single load.
        let fullyResident = (modelSizeGB * 1.15 + MemoryBudget.fixedOverheadGB(for: MemoryBudget.plannableHeadroomGB())) / 0.85
        return min(fullyResident, MemoryBudget.ceilingGB() * 0.85)
    }

    /// Largest model this device will attempt at all, streaming included.
    ///
    /// Partial offload takes memory out of the role of hard limit — whatever doesn't fit stays
    /// as evictable mmap pages, so a model well beyond the memory budget still loads. What it
    /// does not remove is the cost of reading those pages back, and past some size the result is
    /// too slow to be worth offering. This cap is what keeps "it loads" from quietly becoming
    /// "it loads and emits a token a minute".
    ///
    /// Where that size falls depends entirely on how much of the file a single token touches. A
    /// dense model re-reads every streamed byte per token, so the ceiling has to sit close to
    /// what the device can nearly hold. A sparse one reads a few experts per block, so a much
    /// larger file stays responsive and the ceiling can be far more generous. Both multipliers
    /// are judgement calls that want measuring on real hardware.
    /// Sparse models used to get a far larger ceiling here, on the grounds that pinning experts
    /// meant only a fraction of the file had to be resident. Expert pinning is gone (see
    /// `planOffload`), so a mixture-of-experts model now costs exactly what a dense one of the
    /// same size costs and is judged on the same terms.
    var streamableSizeCeilingGB: Double { systemMemoryGB * 1.5 }

    /// Resident cost of a load that streams most of its weights: KV cache, compute buffers and
    /// the app itself. The weights are excluded because the ones being streamed are file-backed
    /// and evictable, and so are not charged against the process's dirty-memory allowance.
    private var streamingFloorGB: Double {
        MemoryBudget.fixedOverheadGB(for: MemoryBudget.plannableHeadroomGB())
    }

    /// Whether a model of this size will have to stream part of itself from storage rather than
    /// sit entirely in memory. Lets the UI explain the cost before the load rather than after.
    func willStreamFromStorage(modelSizeGB: Double) -> Bool {
        let headroom = MemoryBudget.plannableHeadroomGB()
        return modelSizeGB * 1.15 + MemoryBudget.fixedOverheadGB(for: headroom) > headroom * 0.85
    }



    /// Convenience for callers holding a file rather than a bare size — profiles it so a sparse
    /// model is judged against the sparse ceiling.
    ///
    /// The profile is only taken when it can change the answer, i.e. when the model is too large
    /// to hold resident. Reading a GGUF header means parsing its full metadata block, tokenizer
    /// vocabulary included, which is milliseconds but not free; the model picker calls this once
    /// per row while building a list on the main thread, and models that comfortably fit — which
    /// is most of them — have no reason to pay for it.
    func checkMemorySafety(at url: URL) -> MemorySafetyStatus {
        checkMemorySafety(modelSizeGB: getModelSizeGB(at: url))
    }

    func checkMemorySafety(modelSizeGB: Double) -> MemorySafetyStatus {
        let total = systemMemoryGB
        let availableNowGB = MemoryBudget.plannableHeadroomGB()
        // Overhead for app context, system tasks, and the KV-cache/compute buffers that get
        // allocated on top of the model weights (this pre-check runs before the model is
        // loaded, so it can't size those precisely the way safeContextTokens does downstream —
        // budget generously here since this is the only check standing between "load" and a
        // process-limit failure) let modelSizeGB scale it (larger models load larger buffers).
        let required = modelSizeGB * 1.15 + MemoryBudget.fixedOverheadGB(for: availableNowGB)

        // Real-time check: how much headroom does THIS process actually have right now,
        // before hitting its dirty-memory limit? Total device RAM is a constant and can't
        // tell "plenty free right now" apart from transient pressure — e.g. right after app
        // launch, while SwiftUI/asset setup is still consuming memory that will be released
        // moments later. That gap is exactly what caused the auto-load-last-model prompt at
        // launch to fail with an out-of-memory error even though the identical load succeeds
        // seconds later via Settings, once that startup churn has settled.
        if required > availableNowGB * 0.85 {
            // Not enough room to hold the whole model resident. That used to end the matter, and
            // it is what kept a 7B Q4 off an 8 GB device entirely: the check weighed the whole
            // file against the memory budget, even though only the layers actually offloaded to
            // Metal are charged against it.
            //
            // `LlamaRunner.planOffload` now sizes the GPU share to fit the budget and leaves the
            // remaining layers mmap'd. So the real question here is narrower — does the
            // *streaming floor* fit (the KV cache and compute buffers, which are unavoidably
            // resident), and is the model small enough to stream at a tolerable speed? If both
            // hold, this is a warning about speed rather than a refusal, and `handleModelSelection`
            // puts the decision to the user.
            guard modelSizeGB <= streamableSizeCeilingGB,
                  streamingFloorGB < availableNowGB * 0.85 else {
                return .dangerous(requiredGB: required, availableGB: availableNowGB)
            }
            return .warning(requiredGB: required, availableGB: availableNowGB)
        }

        if required > total * 0.90 {
            return .dangerous(requiredGB: required, availableGB: total)
        } else if required > total * 0.70 {
            return .warning(requiredGB: required, availableGB: total)
        }
        return .safe
    }

    /// Memory pre-flight for a Core ML model, run before `loadCoreMLModel` the same way
    /// `checkMemorySafety` gates the GGUF path — that path had no equivalent check at all, which
    /// was a reasonable gap while the catalog's only Core ML model was a ~1 GB fixed-size demo,
    /// and stopped being one once it also gained a 3+ GB / 6 GB-minimum-RAM entry.
    ///
    /// Deliberately not a call to `checkMemorySafety`: that function's "streamable" escape hatch
    /// (a GGUF model too big to fit resident can still load, slower, since llama.cpp can mmap the
    /// excess and only pull in what a given token actually touches) doesn't apply here — every
    /// compiled `MLModel` a Core ML load touches has to be fully resident, chunked pipeline or
    /// not, so there's no slower-but-working fallback for something that doesn't fit the way
    /// there is for GGUF.
    func checkCoreMLMemorySafety(modelSizeGB: Double) -> MemorySafetyStatus {
        let total = systemMemoryGB
        let availableNowGB = MemoryBudget.plannableHeadroomGB()
        let required = modelSizeGB * 1.15 + MemoryBudget.fixedOverheadGB(for: availableNowGB)

        if required > availableNowGB * 0.85 {
            return .dangerous(requiredGB: required, availableGB: availableNowGB)
        }
        if required > total * 0.90 {
            return .dangerous(requiredGB: required, availableGB: total)
        } else if required > total * 0.70 {
            return .warning(requiredGB: required, availableGB: total)
        }
        return .safe
    }

    // MARK: - Model Loading

    func loadModel(at url: URL, forceLoad: Bool = false) {
        // Bumped before either branch below, so a GGUF load, a CoreML load, and an explicit
        // unload all invalidate each other consistently through the one counter.
        loadGeneration += 1
        let generation = loadGeneration

        // A `.mlpackage` extension only ever identified `SingleWindowCoreMLEngine`'s own single-
        // file model (OpenELM) — a chunked pipeline model (Llama) installs as a plain-named
        // directory of `.mlmodelc` bundles with no `.mlpackage` anywhere in it, so that check
        // silently routed it into the GGUF/llama.cpp branch below instead. Every Core ML model,
        // whatever its internal shape, installs directly under `AppFiles.coreMLModels` — checking
        // the parent directory instead of the extension is what actually identifies "this is a
        // Core ML model" regardless of which engine ends up running it.
        guard url.deletingLastPathComponent().path != AppFiles.coreMLModels.path else {
            loadCoreMLModel(at: url, forceLoad: forceLoad, generation: generation)
            return
        }
        activeBackend = .llamaCpp
        let sizeGB = getModelSizeGB(at: url)

        // Whether we're replacing a resident model. `LlamaRunner.load` does unload the old one
        // first, but that happens deep inside the load — after the memory pre-flight below has
        // already run and after the context window has been sized. Both were therefore budgeting
        // against memory the outgoing model was still holding: swapping a large model for
        // another could fail the safety check outright, or succeed with a context window sized
        // for the leftovers. Evicting up front, and waiting for the memory to actually return,
        // means the incoming model is measured against a clean process.
        let needsEviction = isModelLoaded

        activeModelURL = url
        self.loadState = .loading(
            progress: 0.05,
            status: needsEviction ? "Unloading previous model…" : "Initialising llama.cpp backend…"
        )

        // NOTE: the load itself below is deliberately a single attempt, not a retry loop. A
        // previous version retried up to 3 times on failure to work around a transient
        // launch-time issue, but Metal/GPU memory from a failed multi-gigabyte allocation
        // isn't guaranteed to be fully reclaimed before the next attempt starts — retrying
        // compounds pressure on large models instead of recovering from it, and for models
        // near the device's memory ceiling this escalated an app-level failure into a full
        // device reboot. The pre-flight check below is the correct fix for the transient-
        // launch case instead: refuse to attempt when there genuinely isn't headroom, rather
        // than attempting and repeatedly retrying a risky allocation.
        Task {
            if needsEviction {
                await self.unloadModelAsync(invalidatesInFlightLoad: false)
                // `unloadModelAsync` clears these, so restore the incoming model's state before
                // the load continues — otherwise the bar reads "No model loaded" mid-swap. Guard
                // against a genuinely newer `loadModel`/explicit-unload call having started while
                // this one awaited — this eviction step itself doesn't bump the generation (it'd
                // invalidate its own token), but an unrelated one racing in during the wait still
                // should win.
                guard await MainActor.run(body: { self.loadGeneration == generation }) else { return }
                await MainActor.run {
                    self.activeModelURL = url
                    self.loadState = .loading(progress: 0.15, status: "Freeing previous model…")
                }
                await MemoryBudget.waitForRelease(
                    atLeastGB: self.memoryHeadroomNeededGB(forModelSizeGB: sizeGB)
                )
                guard await MainActor.run(body: { self.loadGeneration == generation }) else { return }
                await MainActor.run {
                    self.loadState = .loading(progress: 0.25, status: "Initialising llama.cpp backend…")
                }
            }

            if !forceLoad {
                // iOS manages memory dynamically — it can reclaim room from suspended/cached
                // background processes as foreground memory pressure increases, so a single
                // low os_proc_available_memory() reading isn't necessarily a final verdict
                // (this is especially true right after app launch, before iOS has had a
                // chance to react to the new foreground process). Give it one brief window to
                // settle and recheck before treating a marginal reading as a hard failure —
                // this check itself is nearly free (no allocation), unlike retrying the
                // actual model load, so there's no downside to asking twice.
                var safety = await MainActor.run { self.checkMemorySafety(modelSizeGB: sizeGB) }
                if case .dangerous = safety {
                    try? await Task.sleep(nanoseconds: 1_500_000_000)
                    safety = await MainActor.run { self.checkMemorySafety(modelSizeGB: sizeGB) }
                }
                if case .dangerous(let requiredGB, let availableGB) = safety {
                    let req = String(format: "%.1f", requiredGB)
                    let avail = String(format: "%.1f", availableGB)
                    await MainActor.run {
                        guard self.loadGeneration == generation else { return }
                        self.loadState = .failed(error: "Memory Failsafe: Model requires \(req) GB but only \(avail) GB is safely available right now.")
                    }
                    return
                }
            }

            guard await MainActor.run(body: { self.loadGeneration == generation }) else { return }

            do {
                await MainActor.run {
                    self.loadState = .loading(progress: 0.4, status: "Loading GGUF weights…")
                }

                let availMem = await MainActor.run { self.getAvailableMemoryGB() }

                let currentContextLimit = await MainActor.run {
                    min(self.contextTokenLimit, self.effectiveContextCeiling)
                }
                // Checkpoint before the single largest allocation the app makes. If the process
                // is jetsam-killed during this load there is no handler to run, so this
                // breadcrumb plus the memory reading is the only evidence the next launch has.
                await MainActor.run {
                    CrashReporter.note("loading chat model \(url.lastPathComponent) (\(String(format: "%.1f", sizeGB)) GB, ctx \(currentContextLimit))")
                    CrashReporter.noteChatModel(url.lastPathComponent)
                }
                try await runner.load(
                    path: url.path,
                    availableMemoryGB: availMem,
                    modelSizeGB: sizeGB,
                    contextLimit: currentContextLimit
                )

                let hasVision = await runner.supportsVision()
                let appliedContextWindow = await runner.getContextWindowTokens()
                let trainedContextWindow = await runner.getTrainedContextTokens()

                // The single most important check in this function: without it, a slower load
                // that lost the race would still overwrite a newer load's (or an explicit
                // unload's) already-committed state with its own — potentially stale — result,
                // even though the model it just finished loading is no longer the one the runner
                // actor is about to serve for a fresher, still-in-flight call.
                guard await MainActor.run(body: { self.loadGeneration == generation }) else { return }

                await MainActor.run {
                    self.loadState = .loaded(modelName: url.lastPathComponent, sizeGB: sizeGB)
                    self.modelSupportsVision = hasVision
                    self.lastUsedModelPath = url.path
                    self.loadedContextWindow = appliedContextWindow
                    self.loadedTrainedContext = trainedContextWindow
                    self.contextTokensUsed = 0
                    self.currentResponseTokenCount = 0
                    // Now that a model is resident, the ceiling is model-aware and usually
                    // tighter. Re-clamp so the stored setting reflects what will actually be
                    // honoured next time instead of drifting back to an unreachable number.
                    self.applyContextCeiling()
                }
            } catch {
                await MainActor.run {
                    guard self.loadGeneration == generation else { return }
                    self.loadState = .failed(error: error.localizedDescription)
                    self.modelSupportsVision = false
                }
            }
        }
    }

    /// Load path for an installed Core ML model — deliberately not spliced into `loadModel`
    /// above. That path's context-ceiling math and multi-format KV-cache selection are sized for
    /// GGUF weights streamed through llama.cpp and don't apply here; Core ML models (from the
    /// ~1 GB fixed-window OpenELM package up through the ~3 GB chunked Llama pipeline) must be
    /// fully resident, so there's no "streamable" escape hatch the way `loadModel`'s check has.
    /// Memory pre-flight runs before eviction of any currently-loaded model, using
    /// `checkCoreMLMemorySafety` — mirroring `loadModel`'s `.dangerous` hard-block, but with no
    /// `.warning` confirmation tier: a warning simply proceeds, since a wrong guess here just
    /// costs a load-and-fail rather than risking a mid-generation OOM.
    private func loadCoreMLModel(at url: URL, forceLoad: Bool = false, generation: Int) {
        let needsEviction = isModelLoaded
        activeModelURL = url
        activeBackend = .coreML
        loadState = .loading(
            progress: 0.1,
            status: needsEviction ? "Unloading previous model…" : "Compiling Core ML model…"
        )

        Task {
            // Size is known before the model is loaded — either from the install ledger's
            // recorded total, or (for a package the ledger doesn't know about) a live directory
            // scan. Computed up front so the memory pre-flight can run before any eviction of a
            // currently-loaded model, mirroring the GGUF path's check-before-committing order.
            let sizeGB = await MainActor.run { () -> Double in
                if let installed = ModelInventory.shared.installed.first(where: {
                    $0.kind == .coreML && $0.fileName == url.lastPathComponent
                }) {
                    return Double(installed.byteSize) / (1024 * 1024 * 1024)
                }
                return AppFiles.directorySizeGB(at: url)
            }

            if !forceLoad {
                let safety = await MainActor.run { self.checkCoreMLMemorySafety(modelSizeGB: sizeGB) }
                if case .dangerous(let requiredGB, let availableGB) = safety {
                    let req = String(format: "%.1f", requiredGB)
                    let avail = String(format: "%.1f", availableGB)
                    await MainActor.run {
                        guard self.loadGeneration == generation else { return }
                        self.loadState = .failed(error: "Memory Failsafe: Model requires \(req) GB but only \(avail) GB is safely available right now.")
                    }
                    return
                }
            }

            guard await MainActor.run(body: { self.loadGeneration == generation }) else { return }

            if needsEviction {
                await self.unloadModelAsync(invalidatesInFlightLoad: false)
                guard await MainActor.run(body: { self.loadGeneration == generation }) else { return }
                await MainActor.run {
                    self.activeModelURL = url
                    self.activeBackend = .coreML
                    self.loadState = .loading(progress: 0.2, status: "Compiling Core ML model…")
                }
            }

            await MainActor.run {
                CrashReporter.note("loading Core ML model \(url.lastPathComponent)")
                CrashReporter.noteChatModel(url.lastPathComponent)
            }

            do {
                try await coreMLRunner.load(path: url.path)
                let contextWindow = await coreMLRunner.getContextWindowTokens()
                let trainedContext = await coreMLRunner.getTrainedContextTokens()
                let isSliding = await coreMLRunner.isSlidingWindow

                // As in `loadModel`'s GGUF path: the load itself can't be interrupted mid-flight,
                // but a superseded call's result must not overwrite whatever a newer load (or an
                // explicit unload) already committed.
                guard await MainActor.run(body: { self.loadGeneration == generation }) else { return }

                // Prefer the install ledger's recorded total (every file in the manifest, already
                // summed once at download time — see `ModelDownloadManager.finalizeCoreMLDownload`)
                // over re-walking the directory here — falls back to a live scan only for a
                // package the ledger doesn't know about.
                let sizeGB = await MainActor.run { () -> Double in
                    if let installed = ModelInventory.shared.installed.first(where: {
                        $0.kind == .coreML && $0.fileName == url.lastPathComponent
                    }) {
                        return Double(installed.byteSize) / (1024 * 1024 * 1024)
                    }
                    return AppFiles.directorySizeGB(at: url)
                }

                await MainActor.run {
                    self.loadState = .loaded(modelName: url.lastPathComponent, sizeGB: sizeGB)
                    self.modelSupportsVision = false
                    self.lastUsedModelPath = url.path
                    self.loadedContextWindow = contextWindow
                    self.loadedTrainedContext = trainedContext
                    self.coreMLContextIsSliding = isSliding
                    self.contextTokensUsed = 0
                    self.currentResponseTokenCount = 0
                }
            } catch {
                await MainActor.run {
                    guard self.loadGeneration == generation else { return }
                    self.loadState = .failed(error: error.localizedDescription)
                    self.modelSupportsVision = false
                }
            }
        }
    }

    func unloadModel() {
        Task {
            await unloadModelAsync()
        }
    }

    /// - Parameter invalidatesInFlightLoad: Bumps `loadGeneration`, so any `loadModel`/
    ///   `loadCoreMLModel` call still awaiting something can no longer commit its result on
    ///   completion. True for every real caller (an explicit Unload tap, the diffusion handoff
    ///   freeing memory, `ModelRecovery`'s reset) — false only for the one internal case where
    ///   `loadModel`/`loadCoreMLModel` call this as their own eviction-before-load step, since
    ///   that call must not invalidate the very generation token it just captured for itself.
    func unloadModelAsync(invalidatesInFlightLoad: Bool = true) async {
        switch activeBackend {
        case .llamaCpp:
            await runner.requestCancel()
            await runner.unloadModelOnly()
        case .coreML:
            await coreMLRunner.requestCancel()
            await coreMLRunner.unload()
        }
        await MainActor.run {
            if invalidatesInFlightLoad {
                self.loadGeneration += 1
            }
            self.activeModelURL = nil
            self.activeBackend = .llamaCpp
            self.loadState = .unloaded
            self.modelSupportsVision = false
            self.loadedContextWindow = 0
            self.contextTokensUsed = 0
            self.currentResponseTokenCount = 0
            CrashReporter.noteChatModel(nil)
        }
    }

    /// The Task currently driving `generateStream` on whichever actor is active, if any. Let's
    /// `generateResponse` wait for a just-cancelled generation to actually finish unwinding
    /// before starting a new one — see its capture of `previousGenerationTask` below for why.
    private var currentGenerationTask: Task<Void, Never>?

    func cancelGeneration() {
        switch activeBackend {
        case .llamaCpp: Task { await runner.requestCancel() }
        case .coreML: Task { await coreMLRunner.requestCancel() }
        }
        isGenerating = false
        wasCancelled = true
    }

    // MARK: - Inference

    func generateResponse(
        prompt: String,
        history: [ChatMessage],
        systemPrompt: String,
        memoriesContext: String,
        ragContext: String,
        temperatureBoost: Float = 0.0,
        onToken: @escaping (String) -> Void,
        onComplete: @escaping (String) -> Void
    ) {
        guard case .loaded = loadState else {
            let msg = "⚠️ No model loaded. Go to Settings and load a GGUF model first."
            onToken(msg); onComplete(msg)
            return
        }

        isGenerating = true
        wasCancelled = false

        // Build System Context
        //
        // The GGUF backend gets the full treatment below: a date stamp, an instruction against a
        // specific quantization's known meta-commentary habit, plus custom instructions/memories/
        // RAG context — routinely several hundred tokens, which is nothing against a multi-
        // thousand-token GGUF context window. A CoreML model's window is nowhere near that size
        // (128 tokens fixed for OpenELM, ~576 sliding for the Llama pipeline) — that boilerplate
        // alone can consume the *entire* window before the user's own message is even added,
        // which is exactly what was happening: `SingleWindowCoreMLEngine`'s truncation keeps the
        // tail of the combined token stream, so an oversized system block silently ate the actual
        // question too. Memories/RAG in particular assume GGUF-sized budgets and don't degrade
        // gracefully at this scale, so CoreML skips all of it and keeps only the user's own custom
        // instructions, trimmed — the one thing worth spending part of a tiny window on.
        var systemBlock = ""
        if activeBackend == .llamaCpp {
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "EEEE, MMMM d, yyyy 'at' h:mm a"
            // Rounded down to the nearest 15 minutes rather than the exact instant. This string
            // sits at the very front of the system block, which is the very front of the whole
            // prompt — and `LlamaRunner`'s KV-cache prefix reuse (see `residentTokens`) only
            // engages when a turn's tokenized prompt is byte-identical to the previous turn's
            // over its *entire* resident length. At minute precision this changed on essentially
            // every real turn (any gap of a minute or more between messages), silently defeating
            // reuse for ordinary back-and-forth chat every single time. Bucketing keeps it stable
            // across a typical short exchange, which is what actually lets reuse fire, while
            // staying far more precise than an assistant has any real need for.
            let bucketSeconds: TimeInterval = 15 * 60
            let bucketedNow = Date(timeIntervalSinceReferenceDate:
                (Date().timeIntervalSinceReferenceDate / bucketSeconds).rounded(.down) * bucketSeconds)
            let currentDateString = dateFormatter.string(from: bucketedNow)
            systemBlock += "Current Date and Time: \(currentDateString)\n\n"
            // This model (and quantization) has a strong, persistent tendency to open every
            // reply with meta-commentary about how it plans to respond — style/tone/vibe
            // "checks," headers like "My internal monologue:", decorative *** separators, and
            // literal placeholder text like "(Generating response...)". The client-side filter
            // catches most of this after the fact, but instructing against it directly cuts
            // down how often it happens at all (and how much budget gets burned on it).
            systemBlock += "Respond directly in your own voice. Do not include any internal notes, planning, self-analysis, or meta-commentary about how you are going to respond — no headers or asides like 'Style Check:', 'My internal monologue:', 'Target Response Vibe:', decorative '***' separators, or placeholder text like '(Generating response...)'. Output only the actual reply itself.\n\n"

            if !memoriesContext.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                systemBlock += memoriesContext.trimmingCharacters(in: .whitespacesAndNewlines) + "\n\n"
            }
            if !ragContext.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                systemBlock += ragContext.trimmingCharacters(in: .whitespacesAndNewlines) + "\n\n"
            }
        }
        if !systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            systemBlock += systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines) + "\n\n"
        }

        var swiftMessages: [(role: String, content: String)] = []

        // `loadedContextWindow`, not `contextTokenLimit` — the latter is only what the user
        // requested (defaults to 8192) and can be well above what the loaded model actually got
        // clamped to (`safeContextTokens`, e.g. OLMoE's trained context caps it at 4096
        // regardless of the request). Budgeting history against the requested figure let this
        // pick more history/system content than would actually fit; `generateStream` then had to
        // truncate to the real window anyway, and its truncation keeps the *end* of the token
        // stream — dropping the system block, where Custom Instructions live, silently off the
        // front instead of trimming the oldest history turns the way this budgeting intends.
        let contextLimit = loadedContextWindow > 0 ? loadedContextWindow : contextTokenLimit
        let reservedGeneration = min(maxTokens, max(256, contextLimit / 2))
        let systemTokenEstimate = (systemBlock.count / 3) + 64
        let promptTokenEstimate = (prompt.count / 3) + 16
        let historyBudgetTokens = max(0, contextLimit - reservedGeneration - systemTokenEstimate - promptTokenEstimate - 128)

        var pairs: [(user: ChatMessage, assistant: ChatMessage)] = []
        var i = 0
        let msgs = history
        while i < msgs.count - 1 {
            if msgs[i].isUser && !msgs[i + 1].isUser {
                pairs.append((msgs[i], msgs[i + 1]))
                i += 2
            } else {
                i += 1
            }
        }

        var selectedPairs: [(user: ChatMessage, assistant: ChatMessage)] = []
        var usedTokens = 0
        for pair in pairs.reversed() {
            let estimate = (pair.user.text.count + pair.assistant.text.count) / 3 + 8
            if usedTokens + estimate > historyBudgetTokens { break }
            selectedPairs.insert(pair, at: 0)
            usedTokens += estimate
        }

        // Dropping the oldest turns to fit is expected, normal behavior — never a hard stop —
        // but surface it in the status bar so it's visible when it's actually happening.
        self.isContextTruncating = selectedPairs.count < pairs.count

        if !systemBlock.isEmpty {
            swiftMessages.append(("system", systemBlock.trimmingCharacters(in: .whitespacesAndNewlines)))
        }

        for pair in selectedPairs {
            swiftMessages.append(("user", pair.user.text))
            swiftMessages.append(("assistant", pair.assistant.text))
        }

        swiftMessages.append(("user", prompt))

        let tokenLimit = maxTokens

        // Immediate estimate (char-count based, same heuristic used for history budgeting
        // above) so the context status pill has something to show the instant generation
        // starts, before the actor's real tokenizer count is available.
        let estimatedPromptTokens = systemTokenEstimate + promptTokenEstimate + usedTokens
        self.contextTokensUsed = estimatedPromptTokens
        self.currentResponseTokenCount = 0
        self.thinkingTokensUsed = 0

        let backend = activeBackend

        // `cancelGeneration()` resets `isGenerating` (and requests actor cancellation) the
        // instant it's called, but cooperative cancellation on the actor only takes effect at
        // its next loop checkpoint — not instantly. A message sent in that narrow window used to
        // reach the actor while it still considered itself busy from the just-cancelled call,
        // get silently declined by its own `isBusyGenerating` guard, and complete with no
        // response at all. Waiting for the previous generation's Task to actually finish before
        // this one touches the actor closes that window without delaying the UI's own "stopped"
        // state that `cancelGeneration()` already set.
        let previousGenerationTask = currentGenerationTask
        currentGenerationTask = Task {
            await previousGenerationTask?.value

            var accumulated = ""
            var realPromptTokenCount = estimatedPromptTokens

            await MainActor.run { self.generationSpeed = 0.0 }
            let startTime = CFAbsoluteTimeGetCurrent()
            var tokenCount = 0

            let stream = AsyncStream<String> { continuation in
                Task {
                    let effectiveTemperature = self.highVariabilityEnabled ? Float(2.5) : Float(self.temperature) + temperatureBoost
                    switch backend {
                    case .llamaCpp:
                        await runner.generateStream(
                            messages: swiftMessages,
                            maxTokens: tokenLimit,
                            temperature: effectiveTemperature,
                            continuation: continuation,
                            onContextTruncated: {
                                Task { @MainActor in self.isContextTruncating = true }
                            },
                            onThinkingProgress: { count in
                                Task { @MainActor in self.thinkingTokensUsed = count }
                            }
                        )
                    case .coreML:
                        await coreMLRunner.generateStream(
                            messages: swiftMessages,
                            maxTokens: tokenLimit,
                            temperature: effectiveTemperature,
                            continuation: continuation,
                            onContextTruncated: {
                                Task { @MainActor in self.isContextTruncating = true }
                            }
                        )
                    }
                }
            }

            for await piece in stream {
                guard isGenerating else { break }

                tokenCount += 1
                if tokenCount == 1, backend == .llamaCpp {
                    // Prefill just finished — refine the estimate with the actor's real,
                    // post-truncation tokenizer count. Only meaningful for the llama.cpp
                    // backend, which tracks it; CoreMLRunner has no equivalent, so the
                    // char-count estimate stands for the whole turn there.
                    let real = await runner.getLastPromptTokenCount()
                    if real > 0 { realPromptTokenCount = real }
                }
                let elapsed = max(0.01, CFAbsoluteTimeGetCurrent() - startTime)
                let tps = Double(tokenCount) / elapsed

                accumulated += piece
                
                // Infinite Loop Prevention — detect dot/space loops
                let trimmed = accumulated.trimmingCharacters(in: CharacterSet(charactersIn: ". \n"))
                if trimmed.isEmpty && elapsed > 5.0 && accumulated.count > 10 {
                    switch backend {
                    case .llamaCpp: await self.runner.requestCancel()
                    case .coreML: await self.coreMLRunner.requestCancel()
                    }
                    await MainActor.run {
                        onToken("\n[Loop detected. Stopping generation.]")
                        self.isGenerating = false
                    }
                    return
                }

                // General repetition-loop detector — catches a short phrase/word repeating
                // verbatim (e.g. "I understand I understand I understand..."), which is what
                // a model "stuck in a loop" usually looks like from the outside, beyond the
                // narrow dot/space case above. Throttled since it's O(period) per check.
                if tokenCount % 8 == 0 && accumulated.count > 60 && elapsed > 3.0 {
                    if self.hasTrailingRepetition(accumulated) {
                        switch backend {
                        case .llamaCpp: await self.runner.requestCancel()
                        case .coreML: await self.coreMLRunner.requestCancel()
                        }
                        await MainActor.run {
                            onToken("\n[Repetition loop detected. Stopping generation.]")
                            self.isGenerating = false
                        }
                        return
                    }
                }

                await MainActor.run {
                    self.generationSpeed = tps
                    self.currentResponseTokenCount = tokenCount
                    self.contextTokensUsed = realPromptTokenCount + tokenCount
                    onToken(piece)
                }
            }

            // A decode/prediction failure tears the model down inside its own actor (see
            // `LlamaRunner.handleDecodeFailure` and both Core ML engines' `consumeDecodeFault()`).
            // Without this the app would keep showing a loaded model that can never answer again —
            // which is exactly how this surfaced: "the model loaded but does not respond".
            let faulted: Bool
            switch backend {
            case .llamaCpp: faulted = await runner.consumeDecodeFault()
            case .coreML:   faulted = await coreMLRunner.consumeDecodeFault()
            }

            await MainActor.run {
                isGenerating = false
                if faulted {
                    self.loadState = .failed(error: "The model ran out of GPU memory while generating and was unloaded. Try a smaller model, or lower the context size in Settings.")
                    self.activeModelURL = nil
                    self.modelSupportsVision = false
                    self.loadedContextWindow = 0
                    self.loadedTrainedContext = 0
                    self.contextTokensUsed = 0
                }
                onComplete(accumulated)
            }
        }
    }

    /// Detects whether the tail of `text` consists of a short pattern (3–60 chars)
    /// repeated verbatim at least 4 times in a row — the shape a stuck/looping model
    /// takes regardless of whether it's whitespace, a word, or a whole phrase.
    /// `nonisolated` since it's a pure string check with no shared state, so it can be
    /// called from the background generation Task without hopping onto the MainActor.
    nonisolated private func hasTrailingRepetition(_ text: String) -> Bool {
        let chars = Array(text.suffix(240))
        let n = chars.count
        let repeats = 4
        for period in 3...60 {
            let needed = period * repeats
            guard n >= needed else { break }
            let window = chars.suffix(needed)
            let lastChunk = window.suffix(period)
            var matches = true
            for r in 1..<repeats {
                let start = window.count - period * (r + 1)
                let chunk = window[window.startIndex.advanced(by: start)..<window.startIndex.advanced(by: start + period)]
                if !chunk.elementsEqual(lastChunk) {
                    matches = false
                    break
                }
            }
            if matches { return true }
        }
        return false
    }

    func generateBackgroundAnalysis(prompt: String) async -> String? {
        guard case .loaded = loadState else { return nil }
        // Auxiliary, non-chat generation stays on the llama.cpp backend only for v1 — the Core
        // ML backend's 128-token-total window has no room to spare for a background pass on top
        // of a real chat turn.
        guard activeBackend == .llamaCpp else { return nil }
        // LlamaRunner is a single serialized actor — a background analysis call that
        // races the main chat response for the actor silently delays the user's real
        // answer by the analysis's entire generation time (no UI indication why).
        // Never let it enter if a real chat turn is in flight or starts before us.
        guard !isGenerating else { return nil }

        // System message ensures the analysis result is plain text, not markdown-polluted
        let systemMsg = (role: "system", content: "Respond in plain text only. No asterisks, no bullet points, no markdown headers, no thinking tags. Write each observation as a plain sentence on its own line.")
        let userMsg = (role: "user", content: prompt)
        
        var accumulated = ""
        
        let stream = AsyncStream<String> { continuation in
            Task {
                await runner.generateStream(
                    messages: [systemMsg, userMsg],
                    maxTokens: 512,
                    temperature: 0.3,
                    continuation: continuation
                )
            }
        }
        
        for await piece in stream {
            accumulated += piece
        }
        
        return accumulated.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Uses the currently loaded chat model to expand a short/casual image request into
    /// a richer, more descriptive Stable Diffusion prompt (concrete subject, setting,
    /// lighting, style/quality tags) — PromptClassifier's keyword-stripping alone only
    /// removes the triggering verb ("draw me a dragon" → "a dragon"), it doesn't add any
    /// visual detail SD benefits from. Must be called *before* the LLM is unloaded for
    /// the diffusion handoff, since it needs the model resident to run.
    /// Returns nil (caller should fall back to the raw request) if no model is loaded, a
    /// real chat turn is already in flight, or the model's output doesn't look usable.
    func generateImagePrompt(from userRequest: String) async -> String? {
        guard case .loaded = loadState else { return nil }
        // Same reasoning as `generateBackgroundAnalysis` — llama.cpp only for v1.
        guard activeBackend == .llamaCpp else { return nil }
        // Same reasoning as generateBackgroundAnalysis — LlamaRunner is a single
        // serialized actor, never enter if a real chat generation owns it right now.
        guard !isGenerating else { return nil }

        let systemMsg = (role: "system", content: "You are a Stable Diffusion prompt writer. Rewrite the user's image request as a single vivid, comma-separated prompt describing concrete subject, setting, composition, lighting, and style/quality tags. Output ONLY the prompt itself on one line — no explanation, no quotation marks, no markdown, no leading label like 'Prompt:'.")
        let userMsg = (role: "user", content: userRequest)

        var accumulated = ""
        let stream = AsyncStream<String> { continuation in
            Task {
                await runner.generateStream(
                    messages: [systemMsg, userMsg],
                    // Raised from 120. A reasoning model spends its whole budget inside a
                    // `<think>` block before writing a single word of the actual prompt, so a
                    // tight cap meant the reply was *entirely* deliberation and there was nothing
                    // left to use. Still short enough to read as a beat, not a second wait.
                    maxTokens: 220,
                    temperature: 0.6,
                    continuation: continuation
                )
            }
        }
        for await piece in stream { accumulated += piece }

        return Self.usableImagePrompt(from: accumulated, matching: userRequest)
    }

    /// Extracts a usable Stable Diffusion prompt from raw model output, or `nil` if what came back
    /// can't be trusted to describe what the user asked for.
    ///
    /// This is the fix for images that had nothing to do with the request. The old version trimmed
    /// whitespace and quotes, checked the length, and handed the result to the text encoder — so
    /// three separate kinds of non-prompt sailed through and became the image:
    ///
    /// * **Reasoning blocks.** `<think>Okay, the user wants a barn…</think>` is what a reasoning
    ///   model emits first, and at the old token budget it was frequently the entire response.
    ///   CLIP was handed the model's private deliberation verbatim.
    /// * **Conversational preamble.** "Sure! Here's a prompt for you:" is longer than three
    ///   characters, so it passed, and it describes nothing.
    /// * **Refusals and meta-commentary.** "I can't help with that" likewise.
    ///
    /// Filtering handles the first two. The subject check handles what filtering cannot: a model
    /// that ignored the instruction and answered a different question entirely. Requiring one
    /// content word from the request to survive into the expansion is deliberately loose — a good
    /// expansion elaborates on the subject rather than replacing it, so this rejects drift without
    /// punishing the paraphrasing the expansion exists to do.
    ///
    /// `internal` and `static` so it can be exercised directly; nothing about it needs the actor.
    nonisolated static func usableImagePrompt(from raw: String, matching userRequest: String) -> String? {
        // The same cleanup the chat bubble applies, which is what removes `<think>` blocks,
        // channel markers, bare scaffolding headers, and role echoes.
        var cleaned = ModelOutput.filterThoughts(from: raw)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "\"'`"))

        // A leading label survives `filterThoughts` because "Prompt:" is a legitimate thing for a
        // reply to contain — it only reads as scaffolding here, where the whole output is supposed
        // to *be* the prompt.
        for label in ["prompt:", "image prompt:", "sd prompt:", "here is the prompt:",
                      "here's the prompt:", "output:", "final prompt:"] {
            if cleaned.lowercased().hasPrefix(label) {
                cleaned = String(cleaned.dropFirst(label.count)).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }

        // One line. Models that add a trailing "Let me know if you'd like…" put it on its own
        // line, and the first line is the prompt in every well-formed case.
        if let firstLine = cleaned.split(separator: "\n").first(where: {
            !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }) {
            cleaned = String(firstLine).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        cleaned = cleaned.trimmingCharacters(in: CharacterSet(charactersIn: "\"'`*"))

        guard cleaned.count > 3, cleaned.count < 500 else { return nil }

        // Reject the shapes that are unmistakably the model talking *about* the task rather than
        // performing it. Checked as a prefix so a prompt that legitimately contains "sorry" (a
        // sorrowful scene) isn't caught by the word appearing anywhere.
        let lower = cleaned.lowercased()
        let refusalOpeners = ["i can't", "i cannot", "i'm sorry", "i am sorry", "sorry,",
                              "as an ai", "i'm unable", "i am unable", "sure!", "sure,",
                              "certainly", "of course", "here is", "here's", "okay, ", "ok, ",
                              "the user "]
        if refusalOpeners.contains(where: { lower.hasPrefix($0) }) { return nil }

        // Subject check. If the request had nothing concrete in it ("draw something nice"), there
        // is nothing to verify against and the expansion is accepted as-is.
        let requestWords = contentWords(in: userRequest)
        guard !requestWords.isEmpty else { return cleaned }
        let promptWords = contentWords(in: cleaned)
        let survived = requestWords.contains { word in
            promptWords.contains(word) || promptWords.contains { $0.hasPrefix(word) || word.hasPrefix($0) }
        }
        guard survived else {
            LogManager.shared.log("Image prompt expansion discarded — it dropped the subject of the request")
            return nil
        }

        return cleaned
    }

    /// Lowercased words worth matching on: long enough to be a subject rather than grammar, and
    /// not one of the words every image request contains regardless of what it depicts.
    ///
    /// The vague-adjective entries at the end matter as much as the verbs. Without them, "draw
    /// something nice" yielded the content words {something, nice} and then rejected every
    /// expansion that didn't literally repeat them — throwing away a perfectly good prompt because
    /// the request had named nothing for it to keep. A request made only of filler should leave
    /// nothing to check against, which is exactly what the caller treats as "accept as-is".
    private nonisolated static func contentWords(in text: String) -> Set<String> {
        let ignored: Set<String> = [
            "image", "picture", "photo", "photograph", "draw", "drawing", "paint", "painting",
            "generate", "create", "make", "render", "show", "illustration", "illustrate",
            "please", "with", "that", "this", "some", "very", "into", "from", "your", "have",
            "want", "would", "could", "about", "like", "give", "using", "style", "quality",
            "something", "anything", "everything", "someone", "somebody", "stuff", "thing",
            "things", "nice", "cool", "good", "great", "pretty", "beautiful", "awesome",
            "amazing", "interesting", "random", "please", "maybe", "really"
        ]
        let words = text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 3 && !ignored.contains($0) }
        return Set(words)
    }

}
