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
                    // The safety terms are appended unconditionally and are not user-editable.
                    // Prompt screening upstream in `ContentSafety` catches what the user asked
                    // for; this catches where the *model* would otherwise drift, which is the
                    // real failure mode with community fine-tunes whose training data skews
                    // explicit regardless of how innocuous the prompt was.
                    let data = try wrapper.generateImage(
                        prompt: p,
                        negativePrompt: "ugly, blurry, lowres, bad anatomy, bad hands, cropped, worst quality, "
                            + ContentSafety.diffusionNegativePromptSuffix,
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

    /// Returns `false` if the unload was refused because generation is still running, so the
    /// caller doesn't record the model as gone while its context is still alive.
    @discardableResult
    func unloadModel() -> Bool {
        guard sdWrapper.unload() else { return false }
        loadedPath = nil
        return true
    }
}

// MARK: - DiffusionManager

@MainActor
class DiffusionManager: ObservableObject {

    // MARK: Published State
    @Published var diffusionLoadState: DiffusionLoadState = .unloaded

    /// True for the *entire* user-visible operation — evicting the chat model, loading the
    /// checkpoint, denoising, and unloading again — not just the sampler loop.
    ///
    /// `private(set)` on purpose. This used to be publicly settable and `ContentView` turned it
    /// on by hand before starting its Task so the spinner would appear during the load. That is
    /// exactly how the app hung: `generateImageAsync` bailed out through an early `guard` that
    /// never reset it, and the caller's failure branch didn't either, so the flag stayed true
    /// forever — freezing the progress bubble on screen and making `sendMessage` silently
    /// discard every message the user typed afterwards. The session is now owned here and
    /// closed by the caller's `defer`, so there is no path that leaves it stuck.
    @Published private(set) var isGenerating: Bool = false

    /// What the operation is doing right now. Loading a 3 GB checkpoint and denoising 30 steps
    /// both take minutes; without this the UI is indistinguishable from a hang.
    @Published private(set) var generationStage: String = ""

    @Published var generationProgress: Double = 0.0
    @Published var activeDiffusionURL: URL? = nil

    /// Set when the user backs out. Generation itself can't be interrupted — stable-diffusion.cpp
    /// offers no abort hook — so this releases the UI immediately and the result is discarded
    /// when the sampler eventually finishes.
    private(set) var isCancelled = false

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
        let modelsDir = AppFiles.diffusionModels

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
            Task { @MainActor in
                // Don't tear down a running generation. The unload would be refused anyway
                // (the C++ loop owns the context), but the old code marked the state `.failed`
                // regardless — so an in-flight generation would then fail its own
                // `isLoaded` guard and, before the session fix above, hang the app outright.
                // A memory warning during a multi-minute SDXL run is not rare.
                guard !self.isGenerating else {
                    CrashReporter.note("memory warning during image generation")
                    LogManager.shared.log("DiffusionManager: memory warning ignored — generation in flight")
                    return
                }
                let didUnload = await self.runner.unloadModel()
                guard didUnload else { return }
                self.diffusionLoadState = .failed(error: "Memory pressure — diffusion model unloaded.")
                self.activeDiffusionURL = nil
            }
        }
    }

    // MARK: Generation session

    /// Opens the user-visible operation. Pair with `endGenerationSession()` from a `defer` in the
    /// caller so every exit path — success, thrown error, early return, cancellation — closes it.
    func beginGenerationSession(stage: String) {
        isGenerating = true
        isCancelled = false
        generationProgress = 0
        generationStage = stage
    }

    func updateGenerationStage(_ stage: String) {
        guard isGenerating else { return }
        generationStage = stage
    }

    func endGenerationSession() {
        isGenerating = false
        generationProgress = 0
        generationStage = ""
        // Reached via the caller's `defer`, i.e. once the run has genuinely unwound — including
        // a cancelled one, which is the point at which the chat model has been restored.
        isFinishingCancelledRun = false
    }

    /// Releases the UI now. The in-flight sampler keeps running to completion on its background
    /// thread — there's no safe way to tear its context down mid-loop — but its output is thrown
    /// away and the user gets their input back immediately.
    /// True from the moment the user cancels until the in-flight run actually unwinds.
    ///
    /// stable-diffusion.cpp has no abort hook, so a cancellation during denoising can't stop the
    /// sampler — the task stays parked on that call until it returns, and only then can the
    /// checkpoint be freed and the chat model reloaded. Without this the app looked broken in
    /// that window: input was released, but the model bar read "No model loaded" with nothing
    /// indicating the chat model was on its way back.
    @Published private(set) var isFinishingCancelledRun = false

    func cancelGeneration() {
        guard isGenerating else { return }
        isCancelled = true
        LogManager.shared.log("DiffusionManager: generation cancelled by user")

        // Release the UI immediately, but don't run the full session teardown here — the task's
        // own `defer` does that when it finally unwinds, which is also when the chat model is
        // back. Clearing everything now would erase the only signal that work is still pending.
        isGenerating = false
        generationProgress = 0
        generationStage = ""
        isFinishingCancelledRun = true
    }

    // MARK: Actions

    /// Real, current headroom before this process hits its dirty-memory limit — see the
    /// matching comment on `LLMManager.getAvailableMemoryGB()`. Diffusion model loading
    /// must budget against this, not total device RAM, for the same reason the LLM path
    /// does: a snapshot of total RAM can't tell "plenty free right now" apart from
    /// whatever else (a not-yet-fully-released LLM, RAG/conversation state) is currently
    /// using memory.
    func getAvailableMemoryGB() -> Double {
        // Same basis as the chat model's budgeting — see `MemoryBudget` for why the raw
        // instantaneous reading understates what the app can claim.
        return MemoryBudget.plannableHeadroomGB()
    }

    /// Pre-flight check mirroring `LLMManager.checkMemorySafety` — diffusion models are
    /// frequently the single largest allocation in the app (SDXL weights plus CLIP/VAE plus
    /// the denoising compute graph), so attempting a load with genuinely insufficient
    /// headroom risked a hard jetsam kill instead of a recoverable, user-visible failure.
    ///
    /// Unlike the LLM path, there is no library-level backstop underneath this one: `max_vram`
    /// is deliberately left unset in `SDWrapper.loadModel` (setting it previously caused a
    /// different, uncatchable crash — see the comment there), so stable-diffusion.cpp applies
    /// no memory cap of its own during generation. This check is the *only* thing standing
    /// between a load attempt and a jetsam kill — not a second layer on top of one. An earlier
    /// version used a 1.3x weight-size multiplier plus a flat 1.0 GB and an 0.85
    /// available-memory margin, which rejected legitimate loads with real headroom to spare
    /// (observed: "requires ~4.8 GB but only 5.6 GB is safely available" — a load that in
    /// practice had plenty of room).
    ///
    /// - Parameter outputSize: the square output resolution generation will actually run at.
    ///   Weights are prepared *lazily* (see `SDWrapper.loadModel`) — only a fraction of the
    ///   checkpoint's tensors are resident when this check runs, with the rest materializing
    ///   during `generate_image()` itself, alongside UNet activation buffers and a VAE-decode
    ///   step that both scale with pixel count. A flat compute-overhead constant here was
    ///   blind to that: it judged a 768×768 generation exactly as safe as a 256×256 one from
    ///   the same checkpoint, when the former needs meaningfully more headroom to actually
    ///   finish without the OS reclaiming memory mid-sampler — a kill this library has no way
    ///   to recover from (`generate_image` has no abort hook).
    func checkMemorySafety(modelSizeGB: Double, outputSize: Int) -> MemorySafetyStatus {
        let availableNowGB = getAvailableMemoryGB()
        // Compute overhead for CLIP text encoders, VAE, and UNet activation buffers — not
        // multiplicative with weight size, since `max_vram` downstream already caps weight
        // residency rather than assuming the full file size stays resident at once. Scaled
        // against the 512×512 baseline this constant was tuned for, floored there so behavior
        // at the app's default resolution is unchanged.
        let pixelRatio = Double(outputSize * outputSize) / (512.0 * 512.0)
        let computeOverheadGB = max(0.8, 0.8 * pixelRatio)
        let required = modelSizeGB + computeOverheadGB
        // Margin matches `LLMManager.checkMemorySafety`'s 0.85 — diffusion's lazy weight
        // loading and end-of-run VAE spike make its real memory curve less predictable from a
        // single pre-load snapshot than the LLM's, so it doesn't warrant a looser margin.
        if required > availableNowGB * 0.85 {
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

    /// Process headroom a diffusion load of this size/resolution needs before
    /// `checkMemorySafety` will pass it. Mirrors `LLMManager.memoryHeadroomNeededGB(forModelSizeGB:)`
    /// — derived from the same constants the check above uses, so callers that need to *wait*
    /// for memory rather than merely test it (the LLM→diffusion handoff in `ContentView`) can
    /// target the real threshold instead of guessing with a flat sleep.
    func memoryHeadroomNeededGB(forModelSizeGB modelSizeGB: Double, outputSize: Int) -> Double {
        let pixelRatio = Double(outputSize * outputSize) / (512.0 * 512.0)
        let computeOverheadGB = max(0.8, 0.8 * pixelRatio)
        return (modelSizeGB + computeOverheadGB) / 0.85
    }

    func loadDiffusionModelAsync(at url: URL) async throws {
        await MainActor.run {
            diffusionLoadState = .loading(progress: 0.1, status: "Validating GGUF...")
            activeDiffusionURL = url
            lastDiffusionModelPath = url.path
        }

        // Compatibility gate before anything is mapped. A checkpoint that fails this would
        // abort the process from inside C++ partway through generation — see
        // `GGUFValidator.validateDiffusionCheckpoint` for the mechanism. Checking here is what
        // turns that into a message the user can read.
        do {
            try GGUFValidator.validateDiffusionCheckpoint(path: url.path)
        } catch {
            await MainActor.run {
                self.diffusionLoadState = .failed(error: error.localizedDescription)
                self.activeDiffusionURL = nil
            }
            LogManager.shared.log("Diffusion checkpoint rejected: \(error.localizedDescription)")
            throw error
        }

        let sizeGB = getFileSizeGB(at: url)
        let safety = checkMemorySafety(modelSizeGB: sizeGB, outputSize: outputSize)
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
            // Same reasoning as the chat-model checkpoint: diffusion weights are the other
            // allocation big enough to get the process killed outright.
            CrashReporter.note("loading diffusion model \(url.lastPathComponent) (\(String(format: "%.1f", sizeGB)) GB)")
            CrashReporter.noteDiffusionModel(url.lastPathComponent)
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
        let didUnload = await runner.unloadModel()
        await MainActor.run {
            // Only claim the model is gone if it actually is. Reporting `.unloaded` after a
            // refused teardown is what let a subsequent load believe it was starting from a
            // clean slate while several gigabytes of context were still resident.
            guard didUnload else {
                LogManager.shared.log("DiffusionManager: unload deferred — generation still in flight")
                return
            }
            self.activeDiffusionURL = nil
            self.diffusionLoadState = .unloaded
            CrashReporter.noteDiffusionModel(nil)
        }
    }

    // MARK: Image Generation

    /// Generates an image off the main thread. Call with `await` from a Task that already
    /// owns the MainActor — each internal `await` properly suspends and releases the
    /// MainActor so the UI stays fully responsive throughout the multi-minute denoising loop.
    /// Throws rather than returning `nil`. The old optional return collapsed "no model loaded",
    /// "the sampler failed", and "the output couldn't be encoded" into one silent `nil`, which
    /// the caller could only report as a generic failure — and which hid the state bug above.
    func generateImageAsync(prompt: String,
                            seed: Int = Int.random(in: 0..<Int.max)) async throws -> Data {
        guard diffusionLoadState.isLoaded else {
            throw NSError(domain: "DiffusionManager", code: 20, userInfo: [
                NSLocalizedDescriptionKey: "The diffusion model was unloaded before generation could start — the device may have run low on memory. Try a smaller checkpoint or a lower output resolution."
            ])
        }

        generationProgress = 0.0
        let s = steps; let cfg = Float(cfgScale); let sz = outputSize
        CrashReporter.note("generating image at \(sz)×\(sz), \(s) steps")

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
            generationProgress = 1.0
            return data
        } catch {
            generationProgress = 0.0
            LogManager.shared.log("DiffusionManager: Generation error — \(error.localizedDescription)")
            throw error
        }
    }

    // MARK: Helpers
    func getFileSizeGB(at url: URL) -> Double {
        guard let sz = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize else { return 0 }
        return Double(sz) / (1024.0 * 1024.0 * 1024.0)
    }
}
