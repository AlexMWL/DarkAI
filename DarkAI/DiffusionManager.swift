import Foundation
import Combine
import UIKit
import Darwin
// LlamaSwift is imported for future llama_generate_image() integration.
// The current lumina2 architecture is NOT registered in llama.cpp b9837,
// so we use a pure-Swift GGUF validator instead of llama_model_load_from_file.
import LlamaSwift

// MARK: - Diffusion Load State

enum DiffusionLoadState: Equatable {
    case unloaded
    case loading(progress: Double, status: String)
    case loaded(modelName: String, sizeGB: Double)
    case failed(error: String)

    static func == (lhs: DiffusionLoadState, rhs: DiffusionLoadState) -> Bool {
        switch (lhs, rhs) {
        case (.unloaded, .unloaded):                          return true
        case (.loaded(let a, let b), .loaded(let c, let d)): return a == c && b == d
        case (.failed(let a), .failed(let b)):                return a == b
        default:                                              return false
        }
    }

    var isLoaded: Bool {
        if case .loaded = self { return true }
        return false
    }

    var isLoading: Bool {
        if case .loading = self { return true }
        return false
    }
}

// MARK: - Diffusion Runner (Background Actor)

actor DiffusionRunner {
    private let sdWrapper = SDWrapper()
    var loadedPath: String?

    func loadModel(at url: URL, availableMemoryGB: Double, modelSizeGB: Double) async throws {
        let path = url.path
        let wrapper = sdWrapper  // Capture actor-isolated property before leaving actor context
        // Run the heavy blocking C++ load on a background thread.
        // This keeps the Swift concurrency runtime healthy and the UI responsive during the load.
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try wrapper.loadModel(modelPath: path, availableMemoryGB: availableMemoryGB, modelSizeGB: modelSizeGB)
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
        loadedPath = path
    }

    func generate(prompt: String, steps: Int, cfgScale: Float, width: Int, height: Int, seed: Int, progressHandler: @escaping (Double) -> Void) async throws -> Data {
        guard loadedPath != nil else {
            throw NSError(domain: "DiffusionRunner", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "No model loaded"])
        }
        
        let wrapper = sdWrapper  // Capture before leaving actor context
        let p = prompt, s = steps, cfg = cfgScale, w = width, h = height, sd = seed, ph = progressHandler

        // Run the blocking denoising loop on a background thread.
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Data, Error>) in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let data = try wrapper.generateImage(
                        prompt: p,
                        negativePrompt: "ugly, blurry, lowres, bad anatomy, bad hands, cropped, worst quality",
                        steps: s,
                        cfgScale: cfg,
                        width: w,
                        height: h,
                        seed: sd,
                        progressHandler: ph
                    )
                    continuation.resume(returning: data)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func unloadModel() {
        sdWrapper.unload()
        loadedPath = nil
    }
}

// MARK: - DiffusionManager

@MainActor
class DiffusionManager: ObservableObject {

    // MARK: Published State
    @Published var diffusionLoadState: DiffusionLoadState = .unloaded
    @Published var isGenerating: Bool  = false
    @Published var generationProgress: Double = 0.0
    @Published var activeDiffusionURL: URL? = nil

    // MARK: Persisted Settings
    @Published var steps: Int = UserDefaults.standard.object(forKey: "diffusionSteps") as? Int ?? 20 {
        didSet { UserDefaults.standard.set(steps, forKey: "diffusionSteps") }
    }
    @Published var cfgScale: Double = UserDefaults.standard.object(forKey: "diffusionCFG") as? Double ?? 7.0 {
        didSet { UserDefaults.standard.set(cfgScale, forKey: "diffusionCFG") }
    }
    @Published var outputSize: Int = UserDefaults.standard.object(forKey: "diffusionSize") as? Int ?? 512 {
        didSet { UserDefaults.standard.set(outputSize, forKey: "diffusionSize") }
    }
    // Reconstructed from the current DiffusionModels directory + a stored filename rather
    // than a stored absolute path — see the matching comment on LLMManager.lastUsedModelPath
    // for why a remembered absolute path silently stops resolving across app reinstalls.
    @Published var lastDiffusionModelPath: String? = DiffusionManager.resolveLastDiffusionModelPath() {
        didSet {
            if let p = lastDiffusionModelPath {
                UserDefaults.standard.set(URL(fileURLWithPath: p).lastPathComponent, forKey: "lastDiffusionModelFileName")
            } else {
                UserDefaults.standard.removeObject(forKey: "lastDiffusionModelFileName")
            }
        }
    }

    private static func resolveLastDiffusionModelPath() -> String? {
        guard let docsUrl = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return nil
        }
        let modelsDir = docsUrl.appendingPathComponent("DiffusionModels")

        if let fileName = UserDefaults.standard.string(forKey: "lastDiffusionModelFileName") {
            let url = modelsDir.appendingPathComponent(fileName)
            return FileManager.default.fileExists(atPath: url.path) ? url.path : nil
        }

        // One-time migration from the old absolute-path storage format.
        if let oldPath = UserDefaults.standard.string(forKey: "lastDiffusionModelPath") {
            let fileName = URL(fileURLWithPath: oldPath).lastPathComponent
            let url = modelsDir.appendingPathComponent(fileName)
            if FileManager.default.fileExists(atPath: url.path) {
                UserDefaults.standard.set(fileName, forKey: "lastDiffusionModelFileName")
                UserDefaults.standard.removeObject(forKey: "lastDiffusionModelPath")
                return url.path
            }
        }

        return nil
    }

    private let runner = DiffusionRunner()

    // MARK: Init
    init() {
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil, queue: .main
        ) { [weak self] _ in
            guard let self else { return }
            Task {
                await self.runner.unloadModel()
                await MainActor.run {
                    self.diffusionLoadState = .failed(error: "Memory pressure — diffusion model unloaded.")
                    self.activeDiffusionURL = nil
                }
            }
        }
    }

    // MARK: Actions

    func loadDiffusionModel(at url: URL) {
        Task {
            try? await loadDiffusionModelAsync(at: url)
        }
    }
    
    /// Real, current headroom before this process hits its dirty-memory limit — see the
    /// matching comment on `LLMManager.getAvailableMemoryGB()`. Diffusion model loading
    /// must budget against this, not total device RAM, for the same reason the LLM path
    /// does: a snapshot of total RAM can't tell "plenty free right now" apart from
    /// whatever else (a not-yet-fully-released LLM, RAG/conversation state) is currently
    /// using memory.
    func getAvailableMemoryGB() -> Double {
        return Double(os_proc_available_memory()) / (1024.0 * 1024.0 * 1024.0)
    }

    /// Pre-flight check mirroring `LLMManager.checkMemorySafety` — diffusion models are
    /// frequently the single largest allocation in the app (SDXL weights plus CLIP/VAE plus
    /// the denoising compute graph), so attempting a load with genuinely insufficient
    /// headroom risked a hard jetsam kill instead of a recoverable, user-visible failure.
    ///
    /// This is deliberately *not* as paranoid as it first looks it should be: the actual
    /// weight-residency budgeting already happens one layer down, in
    /// `SDWrapper.loadModel`'s `max_vram` calculation, which caps itself to real available
    /// memory. This check's job is only to reject genuinely hopeless attempts before ever
    /// reaching the C++ load — not to re-derive that budget. An earlier version used a 1.3x
    /// weight-size multiplier plus a flat 1.0 GB and an 0.85 available-memory margin, which
    /// rejected legitimate loads with real headroom to spare (observed: "requires ~4.8 GB
    /// but only 5.6 GB is safely available" — a load that in practice had plenty of room).
    func checkMemorySafety(modelSizeGB: Double) -> MemorySafetyStatus {
        let availableNowGB = getAvailableMemoryGB()
        // Flat compute overhead for CLIP text encoders, VAE, and UNet activation buffers —
        // not multiplicative with weight size, since `max_vram` downstream already caps
        // weight residency rather than assuming the full file size stays resident at once.
        let required = modelSizeGB + 0.8
        if required > availableNowGB * 0.9 {
            return .dangerous(requiredGB: required, availableGB: availableNowGB)
        }
        let total = ProcessInfo.processInfo.physicalMemory
        let totalGiB = Double(total) / (1024.0 * 1024.0 * 1024.0)
        if required > totalGiB * 0.90 {
            return .dangerous(requiredGB: required, availableGB: totalGiB)
        } else if required > totalGiB * 0.70 {
            return .warning(requiredGB: required, availableGB: totalGiB)
        }
        return .safe
    }

    func loadDiffusionModelAsync(at url: URL) async throws {
        await MainActor.run {
            diffusionLoadState = .loading(progress: 0.1, status: "Validating GGUF...")
            activeDiffusionURL = url
            lastDiffusionModelPath = url.path
        }

        let sizeGB = getFileSizeGB(at: url)
        let safety = checkMemorySafety(modelSizeGB: sizeGB)
        if case .dangerous(let requiredGB, let availableGB) = safety {
            let req = String(format: "%.1f", requiredGB)
            let avail = String(format: "%.1f", availableGB)
            let error = NSError(domain: "DiffusionManager", code: 10, userInfo: [
                NSLocalizedDescriptionKey: "Memory Failsafe: Diffusion model requires ~\(req) GB but only \(avail) GB is safely available right now."
            ])
            await MainActor.run {
                self.diffusionLoadState = .failed(error: error.localizedDescription)
                self.activeDiffusionURL = nil
            }
            throw error
        }

        do {
            let availMem = getAvailableMemoryGB()
            try await runner.loadModel(at: url, availableMemoryGB: availMem, modelSizeGB: sizeGB)

            await MainActor.run {
                self.diffusionLoadState = .loaded(modelName: url.lastPathComponent, sizeGB: sizeGB)
            }
        } catch {
            await MainActor.run {
                self.diffusionLoadState = .failed(error: error.localizedDescription)
                self.activeDiffusionURL = nil
            }
            throw error
        }
    }

    func unloadDiffusionModel() {
        Task {
            await unloadDiffusionModelAsync()
        }
    }
    
    func unloadDiffusionModelAsync() async {
        await runner.unloadModel()
        await MainActor.run {
            self.activeDiffusionURL = nil
            self.diffusionLoadState = .unloaded
        }
    }

    // MARK: Image Generation

    /// Generates an image off the main thread. Call with `await` from a Task that already
    /// owns the MainActor — each internal `await` properly suspends and releases the
    /// MainActor so the UI stays fully responsive throughout the multi-minute denoising loop.
    func generateImageAsync(prompt: String,
                            seed: Int = Int.random(in: 0..<Int.max)) async -> Data? {
        guard diffusionLoadState.isLoaded else { return nil }

        isGenerating = true
        generationProgress = 0.0
        let s = steps; let cfg = Float(cfgScale); let sz = outputSize

        // `await runner.generate(...)` suspends this @MainActor function and hops to the
        // DiffusionRunner actor, which further offloads the C++ work onto DispatchQueue.global().
        // The MainActor is therefore FREE to process touch events, animations, etc.
        do {
            let data = try await runner.generate(
                prompt: prompt, steps: s, cfgScale: cfg, width: sz, height: sz, seed: seed,
                progressHandler: { [weak self] p in
                    // Schedule a tiny MainActor task just to update the progress bar.
                    // Creating Task{@MainActor} from a background GCD thread is safe.
                    Task { @MainActor [weak self] in
                        self?.generationProgress = p
                    }
                }
            )
            isGenerating = false
            generationProgress = 1.0
            return data
        } catch {
            isGenerating = false
            generationProgress = 0.0
            LogManager.shared.log("DiffusionManager: Generation error — \(error.localizedDescription)")
            return nil
        }
    }

    // MARK: Helpers
    func getFileSizeGB(at url: URL) -> Double {
        guard let sz = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize else { return 0 }
        return Double(sz) / (1024.0 * 1024.0 * 1024.0)
    }
}
