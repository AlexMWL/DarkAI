import Foundation
import Combine
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
        unload()

        let pathCStr = strdup(modelPath)
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
        // Use 2 threads on iOS devices to keep peak RAM within limits during the denoising loop.
        // Each extra thread requires additional working memory for intermediate ggml tensors.
        // Reducing to 1 strictly limits the intermediate compute buffers.
        ctxParams.n_threads = 2
        
        // Disabled for both the text encoder AND the diffusion UNet. Flash Attention on this
        // Apple Silicon Metal backend can silently misalign/corrupt memory rather than throwing
        // — the library reports a fully successful, error-free generate_image completion (no
        // NaN/overflow warnings, no truncated tensors) while the actual output is structureless
        // colorful noise. This was already known and worked around for CLIP; the UNet path
        // (diffusion_flash_attn) was left enabled and hit the same class of bug.
        ctxParams.flash_attn = false
        ctxParams.diffusion_flash_attn = false
        
        sd_ctx = new_sd_ctx(&ctxParams)
        
        if sd_ctx == nil {
            throw NSError(domain: "SDWrapper", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to initialize sd_ctx. The model may be unsupported or the device may be out of memory."])
        }
        
        isLoaded = true
    }
    private var isCurrentlyGenerating: Bool = false
    
    /// Frees the diffusion context.
    ///
    /// Returns `false` when the request was refused because generation is in flight — freeing
    /// `sd_ctx` while the C++ denoising loop is running on another thread is a use-after-free.
    /// The return value matters: `DiffusionRunner` used to clear its `loadedPath` unconditionally
    /// after calling this, so a refused unload left the actor believing no model was loaded while
    /// the context was in fact still alive and generating.
    @discardableResult
    func unload() -> Bool {
        guard !isCurrentlyGenerating else {
            return false
        }
        if let ctx = sd_ctx {
            autoreleasepool {
                free_sd_ctx(ctx)
            }
            sd_ctx = nil
        }
        isLoaded = false
        return true
    }
    
    func generateImage(
        prompt: String,
        negativePrompt: String = "",
        steps: Int = 20,
        cfgScale: Float = 7.0,
        width: Int = 512,
        height: Int = 512,
        seed: Int = -1,
        progressHandler: ((Double) -> Void)? = nil
    ) throws -> Data {
        guard isLoaded, let ctx = sd_ctx else {
            throw NSError(domain: "SDWrapper", code: 2, userInfo: [NSLocalizedDescriptionKey: "Diffusion model is not loaded."])
        }
        
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else {
            throw NSError(domain: "SDWrapper", code: 7, userInfo: [NSLocalizedDescriptionKey: "Empty prompt sent to text encoder."])
        }
        
        isCurrentlyGenerating = true
        defer { isCurrentlyGenerating = false }

        // CLIP works in 77-token chunks and every extra chunk is another full text-encoder
        // graph. Keeping both prompts inside roughly two chunks bounds that work and keeps the
        // conditioning stage well away from the size at which it has been seen to fail. The
        // limit is generous — SD ignores most of what lands past the first couple of chunks
        // anyway, so truncating here costs nothing a user would notice.
        let boundedPrompt = String(trimmedPrompt.prefix(600))
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
        
        let promptCStr = strdup(boundedPrompt)
        let negPromptCStr = strdup(boundedNegative)
        
        defer {
            sd_set_progress_callback(nil, nil)
            Unmanaged<ProgressContext>.fromOpaque(unmanagedCtx).release()
            free(promptCStr)
            free(negPromptCStr)
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
        
        var imgOut: UnsafeMutablePointer<sd_image_t>? = nil
        var numImages: Int32 = 0
        
        guard generate_image(ctx, &imgParams, &imgOut, &numImages), let imgPtr = imgOut, numImages > 0 else {
            throw NSError(domain: "SDWrapper", code: 3, userInfo: [NSLocalizedDescriptionKey: "txt2img returned nil or failed."])
        }
        
        // imgPtr points to an array of sd_image_t.
        let imageStruct = imgPtr.pointee
        let widthU32 = imageStruct.width
        let heightU32 = imageStruct.height
        let channelU32 = imageStruct.channel
        
        let totalBytes = Int(widthU32 * heightU32 * channelU32)
        let pixelData = Data(bytes: imageStruct.data, count: totalBytes)
        
        return try makeJPEG(from: pixelData, width: Int(widthU32), height: Int(heightU32), channels: Int(channelU32))
    }
    
    private func makeJPEG(from rawData: Data, width: Int, height: Int, channels: Int) throws -> Data {
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
        
        let uiImage = UIImage(cgImage: cgImage)
        guard let jpegData = uiImage.jpegData(compressionQuality: 0.9) else {
            throw NSError(domain: "SDWrapper", code: 6, userInfo: [NSLocalizedDescriptionKey: "Failed to compress to JPEG"])
        }
        
        return jpegData
    }
    
    deinit {
        unload()
    }
}
