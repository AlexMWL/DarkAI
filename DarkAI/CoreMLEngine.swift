import Foundation

/// Shared shape for the two ways `CoreMLRunner` can actually run a Core ML model.
///
/// Unlike the `LlamaRunner`/`CoreMLRunner` split (deliberately *not* a shared protocol — see
/// `LLMBackend.swift`), a protocol is the right call here: both engines genuinely share the same
/// `load(path:)`/`generateStream(...)` shape DarkAI needs, with no lowest-common-denominator
/// compromise required. What differs is entirely internal to each conformer.
///
/// - `SingleWindowCoreMLEngine`: one `.mlpackage`, no cache, a fixed-size window recomputed every
///   step (OpenELM-270M-Instruct).
/// - `ChunkedPipelineCoreMLEngine`: several already-compiled `.mlmodelc` models plus a KV-cache
///   shift model, orchestrated together (Llama 3.2 1B Instruct).
protocol CoreMLEngine: Actor {
    var isLoaded: Bool { get }

    func load(path: String) async throws
    func unload()
    func requestCancel()

    func generateStream(
        messages: [(role: String, content: String)],
        maxTokens: Int,
        temperature: Float,
        continuation: AsyncStream<String>.Continuation,
        onContextTruncated: @escaping @Sendable () -> Void,
        onThinkingProgress: @escaping @Sendable (Int) -> Void
    ) async

    func getContextWindowTokens() -> Int
    func getTrainedContextTokens() -> Int
}
