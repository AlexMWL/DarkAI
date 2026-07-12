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


    private var model: OpaquePointer? = nil
    private var context: OpaquePointer? = nil
    private var nCtxTokens: Int = 2048  // actual context window in tokens
    private var isCancelled = false
    /// Actual tokenized prompt length (post-truncation) from the most recent generation —
    /// the real figure, as opposed to the char-count estimate used for UI budgeting.
    private var lastPromptTokenCount: Int = 0

    var isLoaded: Bool { model != nil && context != nil }
    /// The context window actually applied to the loaded model, which can differ from the
    /// user's requested setting once `safeContextTokens` clamps it to available RAM.
    func getContextWindowTokens() -> Int { nCtxTokens }
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

        // Cheap vocab-only probe to pick a GPU-offload strategy before the real load. Very
        // large vocabularies (Gemma's 256K-token vocab, in particular) have been observed to
        // produce zeroed Metal compute buffers when the output projection layer — which scales
        // with vocab size — is fully GPU-offloaded on this device; capping GPU layers works
        // around it. Queried from the model itself (not matched against the file name) so this
        // correctly protects any large-vocab model — e.g. Llama 3's 128K vocab stays comfortably
        // under the threshold and gets full offload, without needing a per-model name allowlist.
        var nGpuLayers: Int32 = 99
        var probeParams = llama_model_default_params()
        probeParams.vocab_only = true
        if let probeModel = llama_model_load_from_file(path, probeParams) {
            let probeVocab = llama_model_get_vocab(probeModel)
            if llama_vocab_n_tokens(probeVocab) >= 150_000 {
                nGpuLayers = 15
            }
            llama_model_free(probeModel)
        }

        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = nGpuLayers
        modelParams.use_mmap = true
        modelParams.use_mlock = false

        guard let mdl = llama_model_load_from_file(path, modelParams) else {
            throw NSError(domain: "LlamaRunner", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to load GGUF. Check the file is a valid model and fits in RAM."])
        }
        self.model = mdl

        let trainedCtx = Int(llama_model_n_ctx_train(mdl))
        let nCtx = Int32(safeContextTokens(model: mdl,
                                           availableMemoryGB: availableMemoryGB,
                                           modelSizeGB: modelSizeGB,
                                           requestedLimit: contextLimit,
                                           trainedCtx: trainedCtx))

        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx   = UInt32(nCtx)
        ctxParams.n_batch = UInt32(min(nCtx, 512))
        
        // Optimized for Apple Silicon: Using all cores (including E-cores) degrades performance.
        let optimalThreads = Int32(max(2, min(4, ProcessInfo.processInfo.activeProcessorCount / 2)))
        ctxParams.n_threads       = optimalThreads
        ctxParams.n_threads_batch = optimalThreads
        ctxParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED

        guard let ctx = llama_init_from_model(mdl, ctxParams) else {
            llama_model_free(mdl)
            self.model = nil
            throw NSError(domain: "LlamaRunner", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to create inference context. The model may require more RAM."])
        }
        self.context = ctx
        self.nCtxTokens = Int(nCtx)
    }

    /// Computes a safe context window using the model's actual KV-cache geometry
    /// (layer count, KV head count, per-head K/V dims read from GGUF metadata) rather
    /// than a generic size-based guess, so the limit tracks the true per-token memory
    /// cost for *this* model on *this* device's real, currently available RAM.
    ///
    /// This budgets deliberately conservatively. A previous, looser version of this formula
    /// (larger usable-memory fraction, flat compute overhead, lower weight-residency
    /// estimate) allowed large-vocab models like Gemma — which run most layers on CPU
    /// because of the GPU-offload restriction above — to request a context window that
    /// looked safe on paper but wasn't in practice, causing an out-of-memory failure severe
    /// enough to reboot the device rather than just being killed by iOS. Every margin below
    /// is intentionally wide; a smaller-than-necessary context window is a minor inconvenience,
    /// a device reboot is not an acceptable failure mode.
    private func safeContextTokens(model: OpaquePointer,
                                   availableMemoryGB: Double,
                                   modelSizeGB: Double,
                                   requestedLimit: Int,
                                   trainedCtx: Int) -> Int {
        let trainedClamp = max(512, trainedCtx > 0 ? trainedCtx : requestedLimit)
        let requestedClamp = max(512, min(requestedLimit, trainedClamp))
        // Backstop applied to every return path below, independent of whether the detailed
        // formula even runs — never request more than this for a model of this size, in case
        // a given architecture's real memory behavior (e.g. Gemma's mixed local/global
        // attention layers) doesn't match this generic per-layer estimate.
        let hardCeiling = modelSizeGB > 4.0 ? 8192 : (modelSizeGB > 2.0 ? 16384 : 32768)
        let safeRequestedClamp = min(requestedClamp, hardCeiling)

        let nLayer = Int(llama_model_n_layer(model))
        let nHeadKV = Int(llama_model_n_head_kv(model))
        guard nLayer > 0, nHeadKV > 0 else { return safeRequestedClamp }

        func metaString(_ key: String) -> String? {
            var buf = [CChar](repeating: 0, count: 128)
            let n = llama_model_meta_val_str(model, key, &buf, buf.count)
            guard n > 0 else { return nil }
            return String(cString: buf)
        }

        let arch = metaString("general.architecture") ?? ""
        let nHead = Int(llama_model_n_head(model))
        let nEmbd = Int(llama_model_n_embd(model))
        let fallbackHeadDim = nHead > 0 ? nEmbd / nHead : 128

        let headDimK = (arch.isEmpty ? nil : metaString("\(arch).attention.key_length").flatMap { Int($0) }) ?? fallbackHeadDim
        let headDimV = (arch.isEmpty ? nil : metaString("\(arch).attention.value_length").flatMap { Int($0) }) ?? headDimK
        guard headDimK > 0, headDimV > 0 else { return safeRequestedClamp }

        // Default KV cache dtype is f16 (2 bytes/element) unless explicitly overridden elsewhere.
        let bytesPerTokenAllLayers = Double(nLayer * nHeadKV) * Double(headDimK + headDimV) * 2.0
        guard bytesPerTokenAllLayers > 0 else { return safeRequestedClamp }

        // `availableMemoryGB` is the process's real current headroom (os_proc_available_memory),
        // already netting out whatever else is using memory right now — not total device RAM.
        // Still only budget a fraction of it: loading is not instantaneous, and peak memory
        // during the load (mmap page-in, KV cache allocation, compute buffer setup) can exceed
        // steady-state usage, so headroom measured before the load starts needs real margin.
        let usableGB = availableMemoryGB * 0.5

        // Weights are mmap'd (evictable) but with GPU offload restricted on large-vocab models
        // (see the vocab probe above), most layers run on CPU and stay actively resident across
        // every forward pass — budget close to the model's full file size, not a token discount.
        let residentWeightGB = modelSizeGB * 0.8

        // Compute/activation buffers (attention scratch space, batch buffers) scale with
        // context size, not a flat constant — a large requested context needs meaningfully
        // more scratch space than a small one.
        let computeOverheadGB = max(0.5, Double(safeRequestedClamp) / 8192.0 * 0.75)

        let availableForKVGB = usableGB - residentWeightGB - computeOverheadGB
        guard availableForKVGB > 0.05 else { return 512 }

        let availableForKVBytes = availableForKVGB * 1024.0 * 1024.0 * 1024.0
        let maxCtxByMemory = Int(availableForKVBytes / bytesPerTokenAllLayers)

        return max(512, min(safeRequestedClamp, maxCtxByMemory))
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
    }

    func requestCancel() {
        isCancelled = true
    }

    /// Tokenizes a string. Returns empty array on failure.
    private func tokenize(_ text: String, addBOS: Bool) -> [llama_token] {
        guard let mdl = model else { return [] }
        let vocab = llama_model_get_vocab(mdl)
        let utf8 = text.utf8
        let nTokensMax = Int32(utf8.count + 8)
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
        let n = llama_tokenize(vocab, text, Int32(utf8.count), &tokens, nTokensMax, addBOS, true)
        guard n > 0 else { return [] }
        return Array(tokens.prefix(Int(n)))
    }

    /// Checks whether the loaded model advertises any vision/multimodal capability via its metadata.
    func supportsVision() -> Bool {
        guard let mdl = model else { return false }
        // Check model metadata key for projector or mmproj type
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

        isCancelled = false
        let genStartTime = CFAbsoluteTimeGetCurrent()

        let vocab = llama_model_get_vocab(mdl)

        // --- Apply Native Chat Template ---
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

        // 2. Clear KV cache to prevent crashes across multiple prompts
        if let mem = llama_get_memory(ctx) {
            llama_memory_clear(mem, true)
        }

        // 3. Prefill (evaluate the prompt) in chunks of n_batch
        let batchSize = Int(llama_n_batch(ctx))
        var batch = llama_batch_init(Int32(batchSize), 0, 1)
        defer { llama_batch_free(batch) }
        
        var batchStart = 0
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
            
            if llama_decode(ctx, batch) != 0 {
                continuation.yield("[Error: prefill decode failed]")
                continuation.finish()
                return
            }
            batchStart += batchSize
        }

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

            // Copy logits to a Swift array for manipulation
            var logits = Array(UnsafeBufferPointer(start: logitsPtr, count: nVocab))

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
            if temperature > 0 && temperature != 1.0 {
                let invTemp = 1.0 / Float(temperature)
                for i in 0..<nVocab { logits[i] *= invTemp }
            }

            let maxLogit = logits.max() ?? 0

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

            // Softmax on filtered logits
            var expLogits = validLogits.map { ($0.0, expf($0.1 - maxLogit)) }
            let sumExp = expLogits.reduce(0) { $0 + $1.1 }
            if sumExp > 0 {
                for i in 0..<expLogits.count { expLogits[i].1 /= sumExp }
            }

            // Top-p nucleus sampling
            let sorted = expLogits.sorted { $0.1 > $1.1 }
            var cumulative: Float = 0
            var nucleus: [(Int, Float)] = []
            for item in sorted {
                nucleus.append(item)
                cumulative += item.1
                if cumulative >= topP { break }
            }

            // Sample from nucleus
            let nucleusSum = nucleus.reduce(0) { $0 + $1.1 }
            var rand = Float.random(in: 0..<nucleusSum)
            var sampledId = nucleus.first?.0 ?? 0
            for (idx, prob) in nucleus {
                rand -= prob
                if rand <= 0 {
                    sampledId = idx
                    break
                }
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
                        if lower.hasSuffix("[inst]") || lower.hasSuffix("user:") || lower.hasSuffix("<|im_end|>") ||
                            lower.hasSuffix("<start_of_turn>user") || lower.hasSuffix("<|user|>") ||
                            lower.hasSuffix("<|eot_id|>") || lower.hasSuffix("<|start_header_id|>user") {
                            break
                        }
                        continuation.yield(piece)
                        yieldedRealToken = true
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

            if llama_decode(ctx, singleBatch) != 0 { break }

            nPos += 1
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
    @Published var systemMemoryGB: Double = 0.0
    @Published var activeModelURL: URL? = nil
    @Published var generationSpeed: Double = 0.0
    /// Context window actually applied to the loaded model (post safeContextTokens clamp),
    /// distinct from `contextTokenLimit` which is just the user's requested setting.
    @Published var loadedContextWindow: Int = 0
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
    
    @Published var chaosModeEnabled: Bool = UserDefaults.standard.bool(forKey: "chaosModeEnabled") {
        didSet { UserDefaults.standard.set(chaosModeEnabled, forKey: "chaosModeEnabled") }
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

    private static func resolveLastUsedModelPath() -> String? {
        guard let docsUrl = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return nil
        }
        let modelsDir = docsUrl.appendingPathComponent("Models")

        if let fileName = UserDefaults.standard.string(forKey: "lastUsedModelFileName") {
            let url = modelsDir.appendingPathComponent(fileName)
            return FileManager.default.fileExists(atPath: url.path) ? url.path : nil
        }

        // One-time migration from the old absolute-path storage format.
        if let oldPath = UserDefaults.standard.string(forKey: "lastUsedModelPath") {
            let fileName = URL(fileURLWithPath: oldPath).lastPathComponent
            let url = modelsDir.appendingPathComponent(fileName)
            if FileManager.default.fileExists(atPath: url.path) {
                UserDefaults.standard.set(fileName, forKey: "lastUsedModelFileName")
                UserDefaults.standard.removeObject(forKey: "lastUsedModelPath")
                return url.path
            }
        }

        return nil
    }
    
    var safeContextLimit: Int {
        let maxMemory = systemMemoryGB * 0.40
        switch loadState {
        case .loaded(_, let sizeGB):
            let activeModelWeightRAM = sizeGB * 0.10
            let availableForKV = maxMemory - activeModelWeightRAM
            if availableForKV <= 0 { return 2048 }
            let gbPer1kTokens = max(0.04, sizeGB * 0.02)
            let safeLimit = (availableForKV / gbPer1kTokens) * 1000
            return max(2048, min(32768, Int(safeLimit)))
        default:
            let availableForKV = maxMemory - 1.5
            if availableForKV <= 0 { return 4096 }
            let gbPer1kTokens = 0.08
            let safeLimit = (availableForKV / gbPer1kTokens) * 1000
            return max(2048, min(32768, Int(safeLimit)))
        }
    }

    private let runner = LlamaRunner()

    init() {
        self.systemMemoryGB = getPhysicalMemory()
        setupAppLifecycleObservers()
    }

    // MARK: - App Lifecycle - Clean unload on background/termination

    private func setupAppLifecycleObservers() {
        NotificationCenter.default.addObserver(
            forName: UIApplication.willTerminateNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { await self?.runner.unload() }
        }
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            guard let self else { return }
            Task {
                await self.runner.requestCancel()
                await self.runner.unloadModelOnly()
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
        return Double(os_proc_available_memory()) / (1024.0 * 1024.0 * 1024.0)
    }

    func getModelSizeGB(at url: URL) -> Double {
        guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize else { return 0 }
        return Double(size) / (1024.0 * 1024.0 * 1024.0)
    }

    func checkMemorySafety(modelSizeGB: Double) -> MemorySafetyStatus {
        let total = systemMemoryGB
        let availableNowGB = Double(os_proc_available_memory()) / (1024.0 * 1024.0 * 1024.0)
        // Overhead for app context, system tasks, and the KV-cache/compute buffers that get
        // allocated on top of the model weights (this pre-check runs before the model is
        // loaded, so it can't size those precisely the way safeContextTokens does downstream —
        // budget generously here since this is the only check standing between "load" and a
        // process-limit failure) let modelSizeGB scale it (larger models load larger buffers).
        let required = modelSizeGB * 1.15 + 2.0

        // Real-time check: how much headroom does THIS process actually have right now,
        // before hitting its dirty-memory limit? Total device RAM is a constant and can't
        // tell "plenty free right now" apart from transient pressure — e.g. right after app
        // launch, while SwiftUI/asset setup is still consuming memory that will be released
        // moments later. That gap is exactly what caused the auto-load-last-model prompt at
        // launch to fail with an out-of-memory error even though the identical load succeeds
        // seconds later via Settings, once that startup churn has settled.
        if required > availableNowGB * 0.85 {
            return .dangerous(requiredGB: required, availableGB: availableNowGB)
        }

        // If the required memory takes up more than 90% of total device RAM, it's a no-go
        if required > total * 0.90 {
            return .dangerous(requiredGB: required, availableGB: total)
        } else if required > total * 0.70 {
            return .warning(requiredGB: required, availableGB: total)
        }
        return .safe
    }

    // MARK: - Model Loading

    func loadModel(at url: URL, forceLoad: Bool = false) {
        let sizeGB = getModelSizeGB(at: url)

        activeModelURL = url
        self.loadState = .loading(progress: 0.1, status: "Initialising llama.cpp backend…")

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
                        self.loadState = .failed(error: "Memory Failsafe: Model requires \(req) GB but only \(avail) GB is safely available right now.")
                    }
                    return
                }
            }

            do {
                await MainActor.run {
                    self.loadState = .loading(progress: 0.4, status: "Loading GGUF weights…")
                }

                let availMem = await MainActor.run { self.getAvailableMemoryGB() }

                let currentContextLimit = min(self.contextTokenLimit, self.safeContextLimit)
                try await runner.load(
                    path: url.path,
                    availableMemoryGB: availMem,
                    modelSizeGB: sizeGB,
                    contextLimit: currentContextLimit
                )

                let hasVision = await runner.supportsVision()
                let appliedContextWindow = await runner.getContextWindowTokens()

                await MainActor.run {
                    self.loadState = .loaded(modelName: url.lastPathComponent, sizeGB: sizeGB)
                    self.modelSupportsVision = hasVision
                    self.lastUsedModelPath = url.path
                    self.loadedContextWindow = appliedContextWindow
                    self.contextTokensUsed = 0
                    self.currentResponseTokenCount = 0
                }
            } catch {
                await MainActor.run {
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
    
    func unloadModelAsync() async {
        await runner.requestCancel()
        await runner.unloadModelOnly()
        await MainActor.run {
            self.activeModelURL = nil
            self.loadState = .unloaded
            self.modelSupportsVision = false
            self.loadedContextWindow = 0
            self.contextTokensUsed = 0
            self.currentResponseTokenCount = 0
        }
    }

    func cancelGeneration() {
        Task { await runner.requestCancel() }
        isGenerating = false
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

        // Build System Context
        var systemBlock = ""
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "EEEE, MMMM d, yyyy 'at' h:mm a"
        let currentDateString = dateFormatter.string(from: Date())
        systemBlock += "Current Date and Time: \(currentDateString)\n\n"
        // This model (and quantization) has a strong, persistent tendency to open every
        // reply with meta-commentary about how it plans to respond — style/tone/vibe
        // "checks," headers like "My internal monologue:", decorative *** separators, and
        // literal placeholder text like "(Generating response...)". The client-side filter
        // catches most of this after the fact, but instructing against it directly cuts
        // down how often it happens at all (and how much budget gets burned on it).
        systemBlock += "Respond directly in your own voice. Do not include any internal notes, planning, self-analysis, or meta-commentary about how you are going to respond — no headers or asides like 'Style Check:', 'My internal monologue:', 'Target Response Vibe:', decorative '***' separators, or placeholder text like '(Generating response...)'. Output only the actual reply itself.\n\n"

        if !systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            systemBlock += systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines) + "\n\n"
        }
        if !memoriesContext.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            systemBlock += memoriesContext.trimmingCharacters(in: .whitespacesAndNewlines) + "\n\n"
        }
        if !ragContext.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            systemBlock += ragContext.trimmingCharacters(in: .whitespacesAndNewlines) + "\n\n"
        }

        var swiftMessages: [(role: String, content: String)] = []

        let contextLimit = contextTokenLimit
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

        Task {
            var accumulated = ""
            var realPromptTokenCount = estimatedPromptTokens

            await MainActor.run { self.generationSpeed = 0.0 }
            let startTime = CFAbsoluteTimeGetCurrent()
            var tokenCount = 0

            let stream = AsyncStream<String> { continuation in
                Task {
                    await runner.generateStream(
                        messages: swiftMessages,
                        maxTokens: tokenLimit,
                        temperature: self.chaosModeEnabled ? 2.5 : Float(self.temperature) + temperatureBoost,
                        continuation: continuation,
                        onContextTruncated: {
                            Task { @MainActor in self.isContextTruncating = true }
                        },
                        onThinkingProgress: { count in
                            Task { @MainActor in self.thinkingTokensUsed = count }
                        }
                    )
                }
            }

            for await piece in stream {
                guard isGenerating else { break }

                tokenCount += 1
                if tokenCount == 1 {
                    // Prefill just finished — refine the estimate with the actor's real,
                    // post-truncation tokenizer count.
                    let real = await runner.getLastPromptTokenCount()
                    if real > 0 { realPromptTokenCount = real }
                }
                let elapsed = max(0.01, CFAbsoluteTimeGetCurrent() - startTime)
                let tps = Double(tokenCount) / elapsed

                accumulated += piece
                
                // Infinite Loop Prevention — detect dot/space loops
                let trimmed = accumulated.trimmingCharacters(in: CharacterSet(charactersIn: ". \n"))
                if trimmed.isEmpty && elapsed > 5.0 && accumulated.count > 10 {
                    await self.runner.requestCancel()
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
                        await self.runner.requestCancel()
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

            await MainActor.run {
                isGenerating = false
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
                    // Short and cheap by design — this delays the start of image generation,
                    // so it should read as "a beat of extra thinking," not a second wait.
                    maxTokens: 120,
                    temperature: 0.6,
                    continuation: continuation
                )
            }
        }
        for await piece in stream { accumulated += piece }

        let cleaned = accumulated
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
        // Sanity-guard the output — a degenerate/empty/runaway result is worse than just
        // falling back to the caller's own rule-based prompt.
        guard cleaned.count > 3, cleaned.count < 500 else { return nil }
        return cleaned
    }

}
