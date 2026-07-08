import Foundation
import Combine
import UIKit
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

    func loadModel(at url: URL) async throws {
        let path = url.path
        let wrapper = sdWrapper  // Capture actor-isolated property before leaving actor context
        // Run the heavy blocking C++ load on a background thread.
        // This keeps the Swift concurrency runtime healthy and the UI responsive during the load.
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try wrapper.loadModel(modelPath: path)
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
                        negativePrompt: "",
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
    @Published var cfgScale: Double = UserDefaults.standard.object(forKey: "diffusionCFG") as? Double ?? 4.0 {
        didSet { UserDefaults.standard.set(cfgScale, forKey: "diffusionCFG") }
    }
    @Published var outputSize: Int = UserDefaults.standard.object(forKey: "diffusionSize") as? Int ?? 512 {
        didSet { UserDefaults.standard.set(outputSize, forKey: "diffusionSize") }
    }
    @Published var lastDiffusionModelPath: String? = UserDefaults.standard.string(forKey: "lastDiffusionModelPath") {
        didSet {
            if let p = lastDiffusionModelPath { UserDefaults.standard.set(p, forKey: "lastDiffusionModelPath") }
            else { UserDefaults.standard.removeObject(forKey: "lastDiffusionModelPath") }
        }
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
    
    func loadDiffusionModelAsync(at url: URL) async throws {
        await MainActor.run {
            diffusionLoadState = .loading(progress: 0.1, status: "Validating GGUF...")
            activeDiffusionURL = url
            lastDiffusionModelPath = url.path
        }
        
        do {
            try await runner.loadModel(at: url)
            
            let sizeGB = getFileSizeGB(at: url)
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
