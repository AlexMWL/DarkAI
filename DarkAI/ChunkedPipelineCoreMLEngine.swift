import Foundation
import CoreML

/// Runs a chunked, multi-model ANE pipeline — Llama 3.2 1B Instruct, via
/// `smpanaro/Llama-3.2-1B-Instruct-CoreML`. Unlike `SingleWindowCoreMLEngine` (one model, no
/// cache, a hard 128-token ceiling), this is several already-*compiled* `.mlmodelc` chunk models
/// plus a small KV-cache-shift model, together implementing a real sliding-window cache — real
/// chat-quality output at ~512 tokens of live context, not a demo.
///
/// This is a faithful Swift port of `smpanaro/coreml-llm-cli`'s `ModelPipeline` (the orchestration
/// below), `MultiArrayStore`/`KVCacheProcessor`/`PipelineInferenceConfiguration`
/// (`CoreMLPipelineSupport.swift`) — real, tuned ANE engineering (IOSurface-backed pixel-buffer
/// tensors, zero-copy output backings, async per-block cache updates) ported structurally rather
/// than reimplemented from a paraphrase, since a subtly-wrong port of arithmetic this specific is
/// far more likely to silently misplace ops or produce incoherent output than to fail loudly.
///
/// Two deliberate deviations from the reference:
/// - **Sampling.** The reference always does greedy argmax, via a dedicated `logit-processor.mlmodelc`
///   (no learned weights, and no temperature control at all). This engine skips downloading/loading
///   that model entirely and instead runs `CoreMLSampling.sample` — the same temperature/top-p
///   sampler `SingleWindowCoreMLEngine` already uses — directly over the raw logit tensors the
///   reference already extracts before handing them off. Keeps this backend consistent with the
///   rest of the app's temperature control, and drops one model from the download.
/// - **`onContextTruncated` wiring.** The cache is a *sliding* window, not a hard stop — once full,
///   the cache-processor's job is eviction of the oldest entries, so generation keeps going rather
///   than refusing the way `SingleWindowCoreMLEngine` does at 128 tokens. `LLMManager` already has
///   a callback built for exactly this situation (used by `LlamaRunner`); it's fired here once
///   eviction starts, unlike `SingleWindowCoreMLEngine` where a hard stop didn't fit its purpose.
actor ChunkedPipelineCoreMLEngine: CoreMLEngine {

    enum PipelineError: Error, LocalizedError {
        case modelChunksNotFound
        case cacheProcessorNotFound
        case unsupportedInferenceConfiguration

        var errorDescription: String? {
            switch self {
            case .modelChunksNotFound: return "No model chunk files were found for this Core ML pipeline."
            case .cacheProcessorNotFound: return "The KV-cache processor model could not be found."
            case .unsupportedInferenceConfiguration: return "This Core ML pipeline's model shapes couldn't be understood."
            }
        }
    }

    private var chunkModels: [MLModel] = []
    private var cacheProcessorModel: MLModel?
    private var config: PipelineInferenceConfiguration?
    private let tokenizer = CoreMLTokenizer(vocabulary: .llama3BytePairEncoding)
    private var isCancelledFlag = false
    private var isBusyGenerating = false
    private var temperature: Float = 0.7

    var isLoaded: Bool { !chunkModels.isEmpty && cacheProcessorModel != nil && config != nil }

    /// The sliding window's total live capacity (prompt + reply + cache), not a hard stop — see
    /// the type-level doc comment.
    func getContextWindowTokens() -> Int { config?.contextLength ?? 0 }
    func getTrainedContextTokens() -> Int { config?.contextLength ?? 0 }

    func requestCancel() {
        isCancelledFlag = true
    }

    // MARK: - Load

    func load(path: String) async throws {
        unload()
        let folder = URL(fileURLWithPath: path)

        let contents = try FileManager.default.contentsOfDirectory(atPath: folder.path)
        let chunkFiles: [(url: URL, chunkNumber: Int)] = contents.compactMap { name in
            guard name.hasSuffix(".mlmodelc") else { return nil }
            let stem = (name as NSString).deletingPathExtension
            guard let last = stem.components(separatedBy: "_").last, last.hasPrefix("chunk"),
                  let number = Int(last.dropFirst("chunk".count)) else { return nil }
            return (folder.appendingPathComponent(name), number)
        }.sorted { $0.chunkNumber < $1.chunkNumber }

        guard !chunkFiles.isEmpty else { throw PipelineError.modelChunksNotFound }

        var loadedChunks: [MLModel] = []
        for (index, chunkFile) in chunkFiles.enumerated() {
            let configuration = MLModelConfiguration()
            // The first chunk has operations the ANE compiler can't place — matches the
            // reference's own compute-unit assignment exactly (see the type-level doc comment on
            // why this isn't relaxed to `.all` the way `SingleWindowCoreMLEngine` is).
            configuration.computeUnits = index == 0 ? .cpuOnly : .cpuAndNeuralEngine
            configuration.modelDisplayName = "Chunk \(chunkFile.chunkNumber)"
            loadedChunks.append(try MLModel(contentsOf: chunkFile.url, configuration: configuration))
        }

        let cacheProcessorURL = folder.appendingPathComponent("cache-processor.mlmodelc")
        guard FileManager.default.fileExists(atPath: cacheProcessorURL.path) else {
            throw PipelineError.cacheProcessorNotFound
        }
        let cacheConfiguration = MLModelConfiguration()
        cacheConfiguration.computeUnits = .cpuAndNeuralEngine
        cacheConfiguration.modelDisplayName = "Cache Processor"
        let loadedCacheProcessor = try MLModel(contentsOf: cacheProcessorURL, configuration: cacheConfiguration)

        guard let inferredConfig = PipelineInferenceConfiguration(chunkModels: loadedChunks) else {
            throw PipelineError.unsupportedInferenceConfiguration
        }

        chunkModels = loadedChunks
        cacheProcessorModel = loadedCacheProcessor
        config = inferredConfig
    }

    func unload() {
        chunkModels = []
        cacheProcessorModel = nil
        config = nil
    }

    // MARK: - Generation

    func generateStream(
        messages: [(role: String, content: String)],
        maxTokens: Int,
        temperature: Float,
        continuation: AsyncStream<String>.Continuation,
        onContextTruncated: @escaping @Sendable () -> Void = {},
        onThinkingProgress: @escaping @Sendable (Int) -> Void = { _ in }
    ) async {
        guard let config, let cacheProcessorModel, !chunkModels.isEmpty else {
            continuation.finish()
            return
        }
        guard !isBusyGenerating else {
            Task { @MainActor in
                LogManager.shared.log("ChunkedPipelineCoreMLEngine: generateStream called while another generation is already in flight — declining.")
            }
            continuation.finish()
            return
        }
        isBusyGenerating = true
        defer { isBusyGenerating = false }
        isCancelledFlag = false
        self.temperature = temperature

        let prompt = Llama3ChatTemplate.render(messages: messages)
        var promptTokens: [Int]
        do {
            promptTokens = try await tokenizer.encode(prompt, addBOS: false).map(Int.init)
        } catch {
            continuation.yield("[Tokenizer error: \(error.localizedDescription)]")
            continuation.finish()
            return
        }
        guard !promptTokens.isEmpty else {
            continuation.yield("[Error: tokenization returned empty]")
            continuation.finish()
            return
        }

        let inputLength = config.inputLength
        var promptChunks = stride(from: 0, to: promptTokens.count, by: inputLength).map {
            Array(promptTokens[$0..<min($0 + inputLength, promptTokens.count)])
        }
        var tokens = promptChunks.removeFirst()

        let store = PipelineArrayStore(chunkModels: chunkModels)
        let cacheProcessor = PipelineCacheProcessor(chunkCount: chunkModels.count, processorModel: cacheProcessorModel)

        var generatedCount = 0
        var hasStartedEvicting = false

        while generatedCount < maxTokens {
            if isCancelledFlag { break }

            var logitChunks: [MLMultiArray] = []
            do {
                for (chunkIndex, model) in chunkModels.enumerated() {
                    // Wait for this chunk's cache to finish any update the *previous* pass
                    // submitted, before reading it as this pass's input.
                    try await cacheProcessor.wait(forChunk: chunkIndex)

                    let inputs = try store.featureProvider(forChunk: chunkIndex, model: model, tokens: tokens)
                    let options = MLPredictionOptions()
                    options.outputBackings = store.outputBackings(forChunk: chunkIndex, model: model)
                    let outputs = try await model.prediction(from: inputs, options: options)

                    // Only update the cache once a full input segment has been consumed — matches
                    // the reference exactly: with inputLength 64, that's every 64/128/192/...
                    // tokens, not every single token.
                    if tokens.count % inputLength == 0 {
                        cacheProcessor.submit(inputs: inputs, outputs: outputs, forChunk: chunkIndex)
                        if !hasStartedEvicting, tokens.count > config.cacheLength {
                            hasStartedEvicting = true
                            onContextTruncated()
                        }
                    }

                    let shards = outputs.orderedLogitChunks()
                    if !shards.isEmpty { logitChunks = shards }
                }
            } catch {
                Task { @MainActor in
                    LogManager.shared.log("ChunkedPipelineCoreMLEngine: prediction failed — \(error.localizedDescription)")
                }
                break
            }

            if !promptChunks.isEmpty {
                // Prefill: these tokens are already known (they're the rest of the prompt) — this
                // pass computed their cache entries, nothing to sample.
                tokens.append(contentsOf: promptChunks.removeFirst())
                continue
            }

            let sampleIndex = tokens.isEmpty ? 0 : (tokens.count - 1) % inputLength
            guard let nextToken = sampleNextToken(logitChunks: logitChunks, index: sampleIndex) else { break }

            if (try? await tokenizer.isEndOfGeneration(nextToken)) == true { break }

            let piece = (try? await tokenizer.decode(nextToken)) ?? ""
            if !piece.isEmpty {
                continuation.yield(piece)
            }
            tokens.append(Int(nextToken))
            generatedCount += 1

            // Same reasoning as `LlamaRunner`/`SingleWindowCoreMLEngine`'s per-token yield: keeps
            // this actor's executor from starving a queued cancel request on a fast device.
            await Task.yield()
        }

        continuation.finish()
    }

    /// Concatenates whatever shards `logitChunks` holds (usually one; occasionally the vocab is
    /// split across several output tensors) into a single row for `index`, then samples over it —
    /// see the type-level doc comment on why this replaces the reference's dedicated
    /// `logit-processor.mlmodelc`.
    private func sampleNextToken(logitChunks: [MLMultiArray], index: Int) -> Int32? {
        guard !logitChunks.isEmpty else { return nil }

        var combined: [Float] = []
        for shard in logitChunks {
            let vocabShard = shard.shape[2].intValue
            let base = index * vocabShard
            combined.append(contentsOf: Self.floatRow(of: shard, range: base..<(base + vocabShard)))
        }
        return combined.withUnsafeBufferPointer { CoreMLSampling.sample(row: $0, temperature: temperature) }
    }

    /// Reads `range` out of `array` as `Float`, regardless of the tensor's actual storage type.
    ///
    /// This is the ANE-tuned half of the pipeline (see the type-level doc comment) — its logits
    /// output is float16, not float32, unlike `SingleWindowCoreMLEngine`'s OpenELM export. Reading
    /// a float16-backed `MLMultiArray` via `withUnsafeBufferPointer(ofType: Float.self)` doesn't
    /// convert, it reinterprets — which is what actually crashed here the first time (`Fatal
    /// error: Range out of bounds`, since a float16 buffer stores half as many usable elements
    /// as its byte count implies under a `Float` stride). Branch on `array.dataType` and convert
    /// explicitly instead of assuming a layout.
    private static func floatRow(of array: MLMultiArray, range: Range<Int>) -> [Float] {
        switch array.dataType {
        case .float16:
            return array.withUnsafeBufferPointer(ofType: Float16.self) { buffer in
                range.map { Float(buffer[$0]) }
            }
        case .double:
            return array.withUnsafeBufferPointer(ofType: Double.self) { buffer in
                range.map { Float(buffer[$0]) }
            }
        default:
            return array.withUnsafeBufferPointer(ofType: Float.self) { buffer in
                Array(buffer[range])
            }
        }
    }
}
