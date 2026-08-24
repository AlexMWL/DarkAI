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
    func load(path: String) async throws
    func unload()
    func requestCancel()

    /// `onThinkingProgress` mirrors `LlamaRunner`'s reasoning-token callback so `CoreMLRunner` can
    /// dispatch to either backend family with one uniform call shape. As of this writing neither
    /// Core ML conformer actually invokes it (confirmed by grepping the whole project — only
    /// `LlamaRunner`'s llama.cpp path calls it); it stays part of the protocol anyway because
    /// `CoreMLRunner.generateStream` forwards it by name to whichever engine is loaded, so
    /// dropping the parameter here would also require changing `CoreMLRunner.swift`.
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

    /// Read-and-clear: `true` if a prediction failed since the last call, mirroring
    /// `LlamaRunner.consumeDecodeFault()`. Without this, a Core ML backend that hits a prediction
    /// failure kept reporting itself as loaded even though it could never produce another token —
    /// the same "loaded but never responds" symptom `LlamaRunner`'s fault-tracking exists for.
    func consumeDecodeFault() -> Bool
}
