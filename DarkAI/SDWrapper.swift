import Foundation
import UIKit
import CoreGraphics

/// A pure Swift wrapper around the C++ stable-diffusion.cpp library.
/// Explicitly `nonisolated` — it's always driven from the `DiffusionRunner` background
/// actor (off the main actor, which is this module's default isolation) so its heavy,
/// blocking C++ calls never run on the main thread; `@unchecked Sendable` reflects that
/// its internal state is only ever touched serially from that single background actor.
nonisolated class SDWrapper: @unchecked Sendable {
    private var sd_ctx: OpaquePointer?
    private var isLoaded: Bool = false

    /// Guards every read/write of `sd_ctx`/`isLoaded`/`isCurrentlyGenerating`.
    ///
    /// `DiffusionRunner` (the actor that owns this class) dispatches the actual blocking C++
    /// calls onto `DispatchQueue.global()` and awaits a continuation — which suspends the actor,
    /// and a suspended actor is reentrant. That let `unloadModel()` run on the actor, on a
    /// *different* thread, while `generateImage` was still mid-flight on the dispatch queue: with
    /// no lock, `unload()`'s `!isCurrentlyGenerating` check could read `false` (not yet set) in
    /// the gap between `generateImage`'s guard capturing `ctx` and it setting that flag, free the
    /// context out from under the in-flight call, and hand `generate_image` a dangling pointer.
    /// The lock is only ever held across the short check/capture/flag steps below, never across
    /// the slow native calls themselves, so it costs nothing beyond what it protects.
    private let stateLock = NSLock()

    init() {
        sd_set_log_callback({ level, text, user_data in
            guard let text = text, let str = String(cString: text, encoding: .utf8) else { return }
            let cleanStr = str.trimmingCharacters(in: .whitespacesAndNewlines)
            if !cleanStr.isEmpty {
                DispatchQueue.main.async {
                    LogManager.shared.log("SD: \(cleanStr)")
                }
            }
        }, nil)
    }

    /// Loads the diffusion model.
    /// Note: `modelPath` should point to a standard SD1.5 or SDXL GGUF model.
    /// - Parameter availableMemoryGB: the process's real current headroom
    ///   (`os_proc_available_memory()`) measured by the caller right before this call — not
    ///   total device RAM. Recorded in the log alongside the load so a later failure can be
    ///   read against what was actually free at the time.
    /// - Parameter modelSizeGB: the checkpoint's own file size, logged for the same reason.
    func loadModel(modelPath: String, availableMemoryGB: Double, modelSizeGB: Double) throws {
        // A refused unload means a generation is still in flight on the current context — if we
        // pressed ahead here, the next line below would overwrite `sd_ctx` with the new model's
        // pointer while that in-flight call still holds the old one, leaking it (nothing would
        // ever call `free_sd_ctx` on it again) and leaving the app's idea of "loaded model"
        // silently out of sync with what's actually still generating.
        guard unload() else {
            throw NSError(domain: "SDWrapper", code: 8, userInfo: [NSLocalizedDescriptionKey: "Can't switch models while an image is still generating."])
        }

        // `strdup` returns nil on allocation failure — unchecked, that would flow into
        // `ctxParams.model_path` as a null C string. See the matching guard in `generateImage`.
        guard let pathCStr = strdup(modelPath) else {
            throw NSError(domain: "SDWrapper", code: 12, userInfo: [NSLocalizedDescriptionKey: "Out of memory while preparing to load the model."])
        }
        defer { free(pathCStr) }

        var ctxParams = sd_ctx_params_t()
        sd_ctx_params_init(&ctxParams)
        // Assign a stable copy; sd_ctx owns nothing, so we can free after new_sd_ctx returns
        ctxParams.model_path = UnsafePointer(pathCStr)

        // `max_vram` is deliberately left unset (nullptr, the library's default).
        //
        // This used to be set to a computed "weight VRAM budget". That was based on a
        // misreading of the parameter. In stable-diffusion.cpp `max_vram` does not cap weight
        // residency at all — it parses into `MaxVramAssignment` and becomes
        // `max_graph_vram_bytes`, a budget for *compute graph* segmentation. Worse, a non-zero
        // value is the sole gate that switches the runner onto the graph-cut segmented compute
        // path (`should_use_graph_cut_segmented_compute` requires `max_graph_vram_bytes > 0`
        // plus a non-CPU, single-device backend — exactly this configuration).
        //
        // On device that path returned an empty result from the CLIP text encoder, tripping
        // `GGML_ASSERT(!chunk_hidden_states.empty())` in `get_learned_condition_common` and
        // calling `abort()` — an uncatchable SIGABRT partway into image generation, before the
        // first denoising step. Leaving the parameter unset keeps the library on its ordinary
        // unsegmented compute path.
        //
        // Nothing is lost by removing it: the memory controls that actually matter here are the
        // library's lazy per-module weight preparation (see the `enable_mmap` note below) and
        // `DiffusionManager.checkMemorySafety`, which refuses the load up front if the model
        // genuinely can't fit. `availableMemoryGB` and `modelSizeGB` are still taken as
        // parameters because both are worth logging next to a failure.
        LogManager.shared.log(String(
            format: "SD: loading (mmap off, lazy prep), %.1f GB model, %.1f GB process headroom",
            modelSizeGB, availableMemoryGB
        ))
        // SD_TYPE_COUNT = use the model's native quantized types; do NOT re-quantize on load
        ctxParams.wtype = SD_TYPE_COUNT

        // mmap is off, and that is not the memory regression it looks like.
        //
        // `ModelManager::mmap_params` points `tensor->data` at the mapped region but never
        // assigns `tensor->buffer` — the mmap storage block carries no backend buffer at all.
        // `can_mmap_storage` still allows it here because Metal reports host-buffer support on
        // Apple's unified memory. The result is a tensor with data and no Metal buffer, so the
        // first CLIP kernel dispatch hit `ggml_metal_buffer_get_id: tensor '...' buffer is nil`
        // and Metal killed the process on `missing Buffer binding at index 1 for src0[0]`.
        //
        // Peak memory is held down by the library's lazy per-module preparation instead, which
        // is doing the real work here regardless of this flag: it reported
        // "weights will be prepared lazily" and loaded 178 of 2513 tensors to build the
        // conditioning graph, preparing the text encoders, UNet, and VAE as each is needed
        // rather than making all 2.7 GB resident at once.
        ctxParams.enable_mmap = false
        // Scale thread count with the headroom `checkMemorySafety` already cleared this load
        // against, instead of a flat 2 for every device. Each extra thread costs additional
        // working memory for intermediate ggml tensors, so this stays conservative — it only
        // grants more concurrency when `availableMemoryGB` shows real room for it — rather than
        // maxing out the core count outright.
        let threadBudget: Int
        switch availableMemoryGB {
        case ..<1.5: threadBudget = 2
        case ..<3.0: threadBudget = 3
        case ..<5.0: threadBudget = 4
        default: threadBudget = 6
        }
        // `sd_get_num_physical_cores()` asks the library itself, rather than reimplementing the
        // heuristic in Swift. It matters here specifically: `ProcessInfo.processorCount` counts
        // logical cores, which on Apple Silicon includes efficiency cores that are real but much
        // slower for this workload, so a naive processorCount-based budget over-threads relative
        // to what actually helps. Falls back to `processorCount` only in the (unexpected) case
        // the library reports nothing usable.
        let physicalCores = Int(sd_get_num_physical_cores())
        let coreCount = physicalCores > 0 ? physicalCores : ProcessInfo.processInfo.processorCount
        ctxParams.n_threads = Int32(min(coreCount, threadBudget))
        
        // Disabled for both the text encoder AND the diffusion UNet. Flash Attention on this
        // Apple Silicon Metal backend can silently misalign/corrupt memory rather than throwing
        // — the library reports a fully successful, error-free generate_image completion (no
        // NaN/overflow warnings, no truncated tensors) while the actual output is structureless
        // colorful noise. This was already known and worked around for CLIP; the UNet path
        // (diffusion_flash_attn) was left enabled and hit the same class of bug.
        ctxParams.flash_attn = false
        ctxParams.diffusion_flash_attn = false
        
        let newCtx = new_sd_ctx(&ctxParams)

        guard let newCtx else {
            throw NSError(domain: "SDWrapper", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to initialize sd_ctx. The model may be unsupported or the device may be out of memory."])
        }

        // A cheap, authoritative second opinion from the library itself — on top of
        // `GGUFValidator.validateDiffusionCheckpoint`'s own hand-rolled architecture check —
        // that a loaded checkpoint can actually do image generation, catching e.g. a
        // video/audio-only model this app's own check didn't anticipate.
        guard sd_ctx_supports_image_generation(newCtx) else {
            free_sd_ctx(newCtx)
            throw NSError(domain: "SDWrapper", code: 9, userInfo: [NSLocalizedDescriptionKey: "This checkpoint doesn't support image generation."])
        }

        stateLock.lock()
        sd_ctx = newCtx
        isLoaded = true
        stateLock.unlock()
    }
    private var isCurrentlyGenerating: Bool = false

    /// Frees the diffusion context.
    ///
    /// Returns `false` when the request was refused because generation is in flight — freeing
    /// `sd_ctx` while the C++ denoising loop is running on another thread is a use-after-free.
    /// The return value matters: `DiffusionRunner` used to clear its `loadedPath` unconditionally
    /// after calling this, so a refused unload left the actor believing no model was loaded while
    /// the context was in fact still alive and generating.
    ///
    /// The check-and-clear below runs entirely under `stateLock`, so it can't interleave with
    /// `generateImage`'s own check-and-mark-busy under the same lock — whichever of the two
    /// actually acquires the lock first is the one that determines the outcome. The native free
    /// call itself happens after unlocking, on a local copy of the pointer: `sd_ctx` is already
    /// nil by then, so nothing new can be handed that same pointer, and nothing already past the
    /// guard in `generateImage` (`isCurrentlyGenerating` already `true`) can reach here at all.
    @discardableResult
    func unload() -> Bool {
        stateLock.lock()
        guard !isCurrentlyGenerating else {
            stateLock.unlock()
            return false
        }
        let ctxToFree = sd_ctx
        sd_ctx = nil
        isLoaded = false
        stateLock.unlock()

        if let ctxToFree {
            autoreleasepool {
                free_sd_ctx(ctxToFree)
            }
        }
        return true
    }

    /// Requests that an in-flight `generate_image` stop as soon as possible.
    ///
    /// stable-diffusion.cpp checks the cancellation flag between sampler steps, so this does not
    /// interrupt whichever step is currently mid-flight — the call already running finishes
    /// first, and the loop exits at the next checkpoint rather than instantly. `SD_CANCEL_ALL`
    /// (stop immediately) is used rather than `SD_CANCEL_NEW_LATENTS` (finish the current image,
    /// only skip the rest of the batch) because every call from this app uses `batch_count = 1`,
    /// so there is no "rest of the batch" to skip and the two modes would behave identically here
    /// anyway. The library resets this flag itself at the top of every `generate_image` call, so
    /// there is no matching "reset" to issue from the Swift side before the next generation.
    ///
    /// Safe to call whether or not a generation is actually running — `sd_ctx` is read under
    /// `stateLock` exactly like every other access in this class, and if there is no context
    /// loaded this is simply a no-op.
    func cancel() {
        stateLock.lock()
        let ctx = sd_ctx
        stateLock.unlock()
        guard let ctx else { return }
        sd_cancel_generation(ctx, SD_CANCEL_ALL)
    }

    func generateImage(
        prompt: String,
        negativePrompt: String = "",
        steps: Int = 20,
        cfgScale: Float = 7.0,
        width: Int = 512,
        height: Int = 512,
        seed: Int = -1,
        progressHandler: ((Double) -> Void)? = nil,
        previewHandler: ((UIImage) -> Void)? = nil
    ) throws -> Data {
        // Capturing `ctx` and marking generation busy happen together, under the lock, so
        // `unload()` can never observe "not generating yet" in the gap between them — see the
        // lock's own doc comment above `stateLock`'s declaration.
        stateLock.lock()
        guard isLoaded, let ctx = sd_ctx else {
            stateLock.unlock()
            throw NSError(domain: "SDWrapper", code: 2, userInfo: [NSLocalizedDescriptionKey: "Diffusion model is not loaded."])
        }
        guard !isCurrentlyGenerating else {
            stateLock.unlock()
            throw NSError(domain: "SDWrapper", code: 9, userInfo: [NSLocalizedDescriptionKey: "A generation is already in progress."])
        }
        isCurrentlyGenerating = true
        stateLock.unlock()
        defer {
            stateLock.lock()
            isCurrentlyGenerating = false
            stateLock.unlock()
        }

        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else {
            throw NSError(domain: "SDWrapper", code: 7, userInfo: [NSLocalizedDescriptionKey: "Empty prompt sent to text encoder."])
        }

        // CLIP works in 77-token chunks and every extra chunk is another full text-encoder
        // graph. Keeping both prompts inside roughly two chunks bounds that work and keeps the
        // conditioning stage well away from the size at which it has been seen to fail. The
        // limit is generous — SD ignores most of what lands past the first couple of chunks
        // anyway, so truncating here costs nothing a user would notice.
        let boundedPrompt = String(trimmedPrompt.prefix(600))
        // The negative prompt is boilerplate + `ContentSafety.diffusionNegativePromptSuffix`, not
        // user text — safe under 600 characters today, but nothing enforces that as the suffix
        // list grows, and a silent truncation here could quietly drop safety terms off the end.
        // Logged rather than silent so that regression is visible instead of invisible.
        if negativePrompt.count > 600 {
            LogManager.shared.log("SDWrapper: negative prompt truncated from \(negativePrompt.count) to 600 characters — check whether ContentSafety.diffusionNegativePromptSuffix has grown too long.")
        }
        let boundedNegative = String(negativePrompt.prefix(600))

        class ProgressContext {
            var handler: ((Double) -> Void)?
            /// Sampler step count this generation asked for. Used to tell denoising progress
            /// apart from weight-loading progress — see the callback below.
            var samplerSteps: Int32 = 0
        }
        let progressCtx = ProgressContext()
        progressCtx.handler = progressHandler
        progressCtx.samplerSteps = Int32(steps)
        
        // Use passRetained so it stays alive during generation
        let unmanagedCtx = Unmanaged.passRetained(progressCtx).toOpaque()
        
        // stable-diffusion.cpp routes two unrelated things through this one callback:
        // `pretty_progress` reports denoising steps, and `pretty_bytes_progress` reports tensor
        // loading (`model_loader.cpp`). Because weights are prepared lazily *inside*
        // `generate_image`, both fire during a single generation — which is why the bar used to
        // fill to 100%, reset, and fill again.
        //
        // The callback carries no phase identifier, but the two differ in their denominator:
        // sampling reports out of the step count this call requested, while loading reports out
        // of a tensor count. Matching on that keeps the bar showing generation progress only.
        // Loading is still visible to the user — it's the "Loading diffusion model…" stage,
        // which shows an indeterminate spinner rather than a bar that lies about being done.
        sd_set_progress_callback({ step, steps, time, data in
            guard let data = data else { return }
            let pCtx = Unmanaged<ProgressContext>.fromOpaque(data).takeUnretainedValue()
            guard steps == pCtx.samplerSteps, steps > 0 else { return }
            pCtx.handler?(Double(step) / Double(steps))
        }, unmanagedCtx)

        class PreviewContext {
            var handler: ((UIImage) -> Void)?
            /// Guards against doing preview-frame decode work faster than it can be consumed.
            /// `sd_set_preview_callback` fires synchronously on this same background thread as
            /// the sampler advances — cheap for the library to emit, but turning a raw pixel
            /// buffer into a `CGImage`/`UIImage` is real work, and there's no reason to pay for a
            /// second decode before the first one has even finished being handed off. A callback
            /// invocation that arrives while the previous frame is still being decoded just drops
            /// its frame rather than queuing behind it — a slightly-stale preview being skipped
            /// in favor of the next one is invisible to the user; falling behind and decoding a
            /// backlog of stale frames would not be.
            private let busyLock = NSLock()
            private var isBusy = false
            func beginFrame() -> Bool {
                busyLock.lock()
                defer { busyLock.unlock() }
                guard !isBusy else { return false }
                isBusy = true
                return true
            }
            func endFrame() {
                busyLock.lock()
                isBusy = false
                busyLock.unlock()
            }
        }
        let previewCtx = PreviewContext()
        previewCtx.handler = previewHandler
        let unmanagedPreviewCtx = Unmanaged.passRetained(previewCtx).toOpaque()

        // `PREVIEW_PROJ` reports a live preview via a cheap linear latent->RGB projection —
        // pure arithmetic on the latent tensor, no VAE decode involved. `PREVIEW_VAE`/
        // `PREVIEW_TAE` would look sharper, but both fall back to a full VAE decode on every
        // preview frame here (no dedicated TAE model is loaded — see `taesd_path`, left unset
        // above), which is exactly the per-decode memory spike `vae_tiling_params` was just
        // enabled to bound; reporting a preview that way would mean paying that cost dozens of
        // times per generation instead of once. `interval: 1`, `denoised: true`, `noisy: false`
        // are the library's own defaults (`sd_preview_interval`/`sd_preview_denoised`/
        // `sd_preview_noisy` in util.cpp) — only `mode` is changed from its default `PREVIEW_NONE`.
        sd_set_preview_callback({ step, frameCount, frames, isNoisy, data in
            guard let data = data, let frames = frames, frameCount > 0 else { return }
            let pCtx = Unmanaged<PreviewContext>.fromOpaque(data).takeUnretainedValue()
            guard pCtx.beginFrame() else { return }
            defer { pCtx.endFrame() }

            // Image generation never reports more than one frame per callback (`frame_count` > 1
            // is a video-generation concept); the first frame is the only one that exists here.
            let frame = frames.pointee
            guard let framePtr = frame.data else { return }
            let byteCount = Int(frame.width * frame.height * frame.channel)
            guard byteCount > 0 else { return }
            let pixelData = Data(bytes: framePtr, count: byteCount)
            guard let image = try? SDWrapper.makeUIImage(
                from: pixelData, width: Int(frame.width), height: Int(frame.height), channels: Int(frame.channel)
            ) else { return }
            pCtx.handler?(image)
        }, PREVIEW_PROJ, 1, true, false, unmanagedPreviewCtx)

        let promptCStr = strdup(boundedPrompt)
        let negPromptCStr = strdup(boundedNegative)

        defer {
            sd_set_progress_callback(nil, nil)
            sd_set_preview_callback(nil, PREVIEW_NONE, 1, true, false, nil)
            Unmanaged<ProgressContext>.fromOpaque(unmanagedCtx).release()
            Unmanaged<PreviewContext>.fromOpaque(unmanagedPreviewCtx).release()
            // `free` on a nil pointer is a harmless no-op in C, so this defer is already safe to
            // run even when the guard below throws before either `strdup` result is used.
            free(promptCStr)
            free(negPromptCStr)
        }

        // `strdup` returns nil on allocation failure. Unchecked, a nil pointer here would flow
        // straight into `imgParams.prompt`/`negative_prompt` below and on into `generate_image`,
        // which has no reason to expect a null C string — likely an uncatchable native crash
        // (e.g. `strlen(NULL)`) instead of the clean Swift throw every other failure path in this
        // function goes to the trouble of guaranteeing.
        guard let promptCStr, let negPromptCStr else {
            throw NSError(domain: "SDWrapper", code: 11, userInfo: [NSLocalizedDescriptionKey: "Out of memory while preparing the prompt."])
        }

        var imgParams = sd_img_gen_params_t()
        sd_img_gen_params_init(&imgParams)
        imgParams.prompt = UnsafePointer(promptCStr)
        imgParams.negative_prompt = UnsafePointer(negPromptCStr)
        imgParams.width = Int32(width)
        imgParams.height = Int32(height)
        imgParams.seed = Int64(seed)
        imgParams.batch_count = 1
        
        imgParams.sample_params.sample_steps = Int32(steps)
        imgParams.sample_params.guidance.txt_cfg = cfgScale
        // Use the library's own default sampler for the loaded model architecture.
        // Hard-coding EULER_A was wrong for SDXL — the library knows the correct default.
        let defaultMethod = sd_get_default_sample_method(ctx)
        imgParams.sample_params.sample_method = defaultMethod
        imgParams.sample_params.scheduler = sd_get_default_scheduler(ctx, defaultMethod)

        // The VAE decode at the end of every generation is the pixel-count-scaling memory spike
        // `DiffusionManager.checkMemorySafety` budgets for but nothing here previously guarded
        // against directly — decoding the full latent in one pass needs its activation buffers
        // for the whole image at once. Tiling decodes it patch by patch instead, trading a small
        // amount of extra compute for a much smaller peak.
        //
        // Only `enabled` is set. `tile_size_x`/`tile_size_y` are left at
        // `sd_img_gen_params_init`'s default of 0, which the library's own tile-sizing
        // (`VAE::get_tile_sizes` in vae.hpp) turns into an automatic 32-latent-pixel tile, and
        // `target_overlap` stays at that same default of 0.5 (already the maximum the library
        // clamps it to). These are exactly the values stable-diffusion.cpp's own CLI falls back
        // to for `--vae-tiling` with no explicit tile size — there's no local reason to second-
        // guess them with hand-picked pixel counts.
        imgParams.vae_tiling_params.enabled = true

        var imgOut: UnsafeMutablePointer<sd_image_t>? = nil
        var numImages: Int32 = 0
        
        guard generate_image(ctx, &imgParams, &imgOut, &numImages), let imgPtr = imgOut, numImages > 0 else {
            throw NSError(domain: "SDWrapper", code: 3, userInfo: [NSLocalizedDescriptionKey: "Image generation failed. Try again, or try a different prompt or checkpoint."])
        }
        // The library hands ownership of `imgPtr` (and each image's pixel buffer) to the caller —
        // must be released via `free_sd_images` on every exit path, including the throws below.
        defer { free_sd_images(imgPtr, numImages) }

        // imgPtr points to an array of sd_image_t.
        let imageStruct = imgPtr.pointee
        let widthU32 = imageStruct.width
        let heightU32 = imageStruct.height
        let channelU32 = imageStruct.channel

        let totalBytes = Int(widthU32 * heightU32 * channelU32)
        // `generate_image` returning true is not, on its own, a guarantee of a populated buffer —
        // guard the pointer and size before copying rather than crashing on a null/short buffer
        // from an otherwise "successful" native call.
        guard let dataPtr = imageStruct.data, totalBytes > 0 else {
            throw NSError(domain: "SDWrapper", code: 10, userInfo: [NSLocalizedDescriptionKey: "Generated image had no pixel data."])
        }
        let pixelData = Data(bytes: dataPtr, count: totalBytes)
        
        return try Self.makeJPEG(from: pixelData, width: Int(widthU32), height: Int(heightU32), channels: Int(channelU32))
    }

    /// Builds a `UIImage` from a raw RGB buffer as `sd_image_t` reports it.
    ///
    /// `static` — and deliberately touches no instance state — so it can also be called from
    /// the non-capturing preview callback above, which as a C function pointer cannot capture
    /// `self`. Both the final-output path (`makeJPEG`) and the live-preview path share this exact
    /// conversion rather than each doing their own, since the preview frame and the final image
    /// arrive from the library in the same raw-RGB shape.
    private static func makeUIImage(from rawData: Data, width: Int, height: Int, channels: Int) throws -> UIImage {
        // SD outputs RGB (3 channels) by default.
        guard channels == 3 else {
            throw NSError(domain: "SDWrapper", code: 4, userInfo: [NSLocalizedDescriptionKey: "Expected 3 channels (RGB)"])
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)

        guard let provider = CGDataProvider(data: rawData as CFData),
              let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 24,
                bytesPerRow: width * 3,
                space: colorSpace,
                bitmapInfo: bitmapInfo,
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              ) else {
            throw NSError(domain: "SDWrapper", code: 5, userInfo: [NSLocalizedDescriptionKey: "Failed to create CGImage from raw data"])
        }

        return UIImage(cgImage: cgImage)
    }

    private static func makeJPEG(from rawData: Data, width: Int, height: Int, channels: Int) throws -> Data {
        let uiImage = try makeUIImage(from: rawData, width: width, height: height, channels: channels)
        guard let jpegData = uiImage.jpegData(compressionQuality: 0.9) else {
            throw NSError(domain: "SDWrapper", code: 6, userInfo: [NSLocalizedDescriptionKey: "Failed to compress to JPEG"])
        }

        return jpegData
    }

    deinit {
        // `deinit` can't await or block, so there's no way to actually wait out an in-flight
        // generation here — this is purely about visibility. Every other call site that can be
        // refused (`loadModel`, `DiffusionRunner.unloadModel`, `DiffusionManager.unloadDiffusionModelAsync`)
        // checks this same return value and logs; deinit discarding it silently was the one place
        // a leaked context (never freed, because nothing holds this `SDWrapper` anymore to retry
        // the unload) could happen with no trace anywhere.
        if !unload() {
            LogManager.shared.log("SDWrapper: deinit couldn't free sd_ctx — a generation was still in flight, so the context leaked")
        }
    }
}
