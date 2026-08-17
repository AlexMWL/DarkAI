import Foundation

/// Thin dispatcher `LLMManager` talks to for the Core ML backend — picks which `CoreMLEngine`
/// actually runs a given installed model and forwards every call to it.
///
/// The two engines' on-disk shapes are easy to tell apart without any catalog bookkeeping: a
/// chunked ANE pipeline (`ChunkedPipelineCoreMLEngine`) always installs a `cache-processor.mlmodelc`
/// alongside its chunks, and a single-window model (`SingleWindowCoreMLEngine`) never does — so
/// `load(path:)` decides by looking at what's actually on disk, the same way the rest of the app
/// prefers reading real state over trusting a label. `LLMManager` never needs to know there are
/// two flavors of Core ML model; its `coreMLRunner.load/generateStream/unload` calls are unchanged
/// from when `CoreMLRunner` was the single-window engine itself.
actor CoreMLRunner {

    private var engine: (any CoreMLEngine)?

    var isLoaded: Bool {
        get async { await engine?.isLoaded ?? false }
    }

    /// Whether the currently-loaded model is a real sliding-window cache (`ChunkedPipelineCoreMLEngine`,
    /// e.g. Llama 3.2) rather than a hard-stop fixed window (`SingleWindowCoreMLEngine`, e.g.
    /// OpenELM). `LLMManager` surfaces this so Settings can describe `getContextWindowTokens()`'s
    /// number accurately — the two engines mean very different things by it: one refuses once hit,
    /// the other keeps generating and gradually forgets the earliest turns instead.
    var isSlidingWindow: Bool {
        engine is ChunkedPipelineCoreMLEngine
    }

    func load(path: String) async throws {
        let directory = URL(fileURLWithPath: path)
        let isPipeline = FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("cache-processor.mlmodelc").path
        )

        let chosen: any CoreMLEngine = isPipeline ? ChunkedPipelineCoreMLEngine() : SingleWindowCoreMLEngine()
        try await chosen.load(path: path)
        // Only swap in the newly-loaded engine once it has actually finished loading — on
        // failure, whatever was loaded before (if anything) is left in place rather than torn
        // down for nothing.
        await engine?.unload()
        engine = chosen
    }

    func unload() async {
        let previous = engine
        engine = nil
        await previous?.unload()
    }

    func requestCancel() async {
        await engine?.requestCancel()
    }

    func generateStream(
        messages: [(role: String, content: String)],
        maxTokens: Int,
        temperature: Float,
        continuation: AsyncStream<String>.Continuation,
        onContextTruncated: @escaping @Sendable () -> Void = {},
        onThinkingProgress: @escaping @Sendable (Int) -> Void = { _ in }
    ) async {
        guard let engine else {
            continuation.finish()
            return
        }
        await engine.generateStream(
            messages: messages,
            maxTokens: maxTokens,
            temperature: temperature,
            continuation: continuation,
            onContextTruncated: onContextTruncated,
            onThinkingProgress: onThinkingProgress
        )
    }

    func getContextWindowTokens() async -> Int {
        await engine?.getContextWindowTokens() ?? 0
    }

    func getTrainedContextTokens() async -> Int {
        await engine?.getTrainedContextTokens() ?? 0
    }
}
