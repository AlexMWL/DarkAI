import Foundation
import Combine
import UIKit
import CoreGraphics

/// A pure Swift wrapper around the C++ stable-diffusion.cpp library
class SDWrapper {
    private var sd_ctx: OpaquePointer?
    private var isLoaded: Bool = false

    /// Loads the diffusion model. 
    /// Note: `modelPath` should point to a standard SD1.5 or SDXL GGUF model.
    func loadModel(modelPath: String) throws {
        unload()
        
        let pathCStr = strdup(modelPath)
        defer { free(pathCStr) }
        
        var ctxParams = sd_ctx_params_t()
        sd_ctx_params_init(&ctxParams)
        // Assign a stable copy; sd_ctx owns nothing, so we can free after new_sd_ctx returns
        ctxParams.model_path = UnsafePointer(pathCStr)
        
        let physicalMemory = ProcessInfo.processInfo.physicalMemory
        let totalGiB = Double(physicalMemory) / 1024 / 1024 / 1024
        
        // iOS limits Metal memory to ~60% of total physical RAM (before increased-memory-limit).
        // We use 55% as a safe upper limit for all Metal allocations.
        let metalLimitGiB = totalGiB * 0.55
        // We must subtract ~0.6 GiB to leave room for the required intermediate compute buffers,
        // leaving the remainder purely for the model weights.
        var weightVRAM = metalLimitGiB - 0.6
        if weightVRAM < 1.0 { weightVRAM = 1.0 } // Minimum fallback to prevent weird edge cases
        
        let vramString = String(format: "%.1f", weightVRAM)
        let maxVRAM = strdup(vramString)
        defer { free(maxVRAM) }
        ctxParams.max_vram = UnsafePointer(maxVRAM)
        // SD_TYPE_COUNT = use the model's native quantized types; do NOT re-quantize on load
        ctxParams.wtype = SD_TYPE_COUNT
        // mmap = true: weights are paged in from flash on demand rather than fully read into RAM.
        // This is CRITICAL on iOS — without it a 3.7 GB SDXL model will exceed the OS memory limit
        // and jetsam will kill the process before the first denoising step.
        ctxParams.enable_mmap = true
        // Use 2 threads on iOS devices to keep peak RAM within limits during the denoising loop.
        // Each extra thread requires additional working memory for intermediate ggml tensors.
        // Reducing to 1 strictly limits the intermediate compute buffers.
        ctxParams.n_threads = 1
        
        // Enable Flash Attention to aggressively reduce VRAM usage during the denoising loop
        ctxParams.flash_attn = true
        ctxParams.diffusion_flash_attn = true
        
        sd_ctx = new_sd_ctx(&ctxParams)
        
        if sd_ctx == nil {
            throw NSError(domain: "SDWrapper", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to initialize sd_ctx. The model may be unsupported or the device may be out of memory."])
        }
        
        isLoaded = true
    }
    private var isCurrentlyGenerating: Bool = false
    
    func unload() {
        guard !isCurrentlyGenerating else {
            // Cannot unload the model while it's actively generating on a background thread!
            return
        }
        if let ctx = sd_ctx {
            autoreleasepool {
                free_sd_ctx(ctx)
            }
            sd_ctx = nil
        }
        isLoaded = false
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
        
        isCurrentlyGenerating = true
        defer { isCurrentlyGenerating = false }
        
        class ProgressContext {
            var handler: ((Double) -> Void)?
        }
        let progressCtx = ProgressContext()
        progressCtx.handler = progressHandler
        
        // Use passRetained so it stays alive during generation
        let unmanagedCtx = Unmanaged.passRetained(progressCtx).toOpaque()
        
        sd_set_progress_callback({ step, steps, time, data in
            guard let data = data else { return }
            let pCtx = Unmanaged<ProgressContext>.fromOpaque(data).takeUnretainedValue()
            let progress = steps > 0 ? Double(step) / Double(steps) : 0.0
            pCtx.handler?(progress)
        }, unmanagedCtx)
        
        // Properly manage C strings
        let promptCStr = strdup(prompt)
        let negPromptCStr = strdup(negativePrompt)
        
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
        
        // Copy pixel data
        let totalBytes = Int(widthU32 * heightU32 * channelU32)
        let pixelData = Data(bytes: imageStruct.data, count: totalBytes)
        
        return try makeJPEG(from: pixelData, width: Int(widthU32), height: Int(heightU32), channels: Int(channelU32))
    }
    
    private func makeJPEG(from rawData: Data, width: Int, height: Int, channels: Int) throws -> Data {
        // SD outputs RGB (3 channels) by default.
        // We'll use CoreGraphics to create a CGImage
        
        
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
