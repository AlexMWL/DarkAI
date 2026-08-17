import Foundation
import CoreML
import CoreVideo

/// Supporting types for `ChunkedPipelineCoreMLEngine`, ported from `smpanaro/coreml-llm-cli`
/// (`MultiArrayStore.swift`, `KVCacheProcessor.swift`, `CoreML+Extensions.swift`,
/// `PipelineInferenceConfiguration.swift` in that repo) — see the engine's own doc comment for why
/// this is a faithful port rather than a reimplementation-from-paraphrase.

// MARK: - Inference configuration

/// Shapes read off the loaded chunk models themselves, rather than hardcoded — a chunked ANE
/// export bakes its window/context sizes into the model graph, so the only reliable source for
/// them is the graph.
nonisolated struct PipelineInferenceConfiguration {
    let vocabSize: Int
    /// Query length — how many new tokens one forward pass through the chunks consumes at once.
    let inputLength: Int
    /// Key + value length the cache holds.
    let contextLength: Int

    var cacheLength: Int { contextLength - inputLength }

    init?(chunkModels: [MLModel]) {
        guard chunkModels.count > 2, let first = chunkModels.first, let last = chunkModels.last else { return nil }

        guard let inputLen = first.modelDescription.inputDescriptionsByName["input_ids"]?
            .multiArrayConstraint?.shape.last?.intValue else { return nil }
        self.inputLength = inputLen

        let logitsDescription = last.modelDescription.outputDescriptionsByName["logits"]
            ?? last.modelDescription.outputDescriptionsByName["logits_0"]
        guard let vocab = logitsDescription?.multiArrayConstraint?.shape.last?.intValue else { return nil }
        self.vocabSize = vocab

        let innerModels = chunkModels[1..<chunkModels.count - 1]
        guard let firstInner = innerModels.first,
              let cacheLen = firstInner.modelDescription.inputDescriptionsByName["k_cache_0"]?
                .multiArrayConstraint?.shape.last?.intValue else { return nil }
        self.contextLength = inputLength + cacheLen
    }
}

// MARK: - IOSurface-backed tensors

nonisolated extension MLMultiArray {
    /// A float16 `MLMultiArray` backed by an IOSurface `CVPixelBuffer` rather than plain heap
    /// memory — what lets Core ML pass a tensor to/from the ANE without a CPU-side copy, and what
    /// makes the zero-copy `outputBackings` trick below actually save anything. Ordinary
    /// `MLMultiArray(shape:dataType:)` arrays are not IOSurface-backed and would silently defeat
    /// this.
    static func emptyIOSurfaceArray(shape: [Int]) -> MLMultiArray? {
        guard shape.count > 0, let width = shape.last else { return nil }
        let height = shape[0..<shape.count - 1].reduce(1, *)

        let attributes = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        guard kCVReturnSuccess == CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent16Half, attributes, &pixelBuffer
        ), let pixelBuffer else { return nil }

        guard kCVReturnSuccess == CVPixelBufferLockBaseAddress(pixelBuffer, .init(rawValue: 0)) else { return nil }
        memset(CVPixelBufferGetBaseAddress(pixelBuffer), 0, CVPixelBufferGetDataSize(pixelBuffer))
        guard kCVReturnSuccess == CVPixelBufferUnlockBaseAddress(pixelBuffer, .init(rawValue: 0)) else { return nil }

        return MLMultiArray(pixelBuffer: pixelBuffer, shape: shape.map { $0 as NSNumber })
    }
}

// MARK: - Array store

/// Owns every reusable float16 tensor the pipeline's chunks read and write across a single
/// generation: RoPE cos/sin and the attention mask (shared — identical across chunks, allocated
/// once), and each chunk's own K/V cache (kept separate per chunk index). Allocated fresh per
/// `generateStream` call — nothing here is retained across generations.
nonisolated final class PipelineArrayStore {
    private let sharedArrays: [String: MLMultiArray]
    private let chunkArrays: [[String: MLMultiArray]]
    /// Maps an output name to a different key to use as its backing array — lets an output
    /// overwrite the input array it was computed from in place instead of allocating a new one.
    private let outputBackingMapping: [String: String]

    init(sharedArrays: [String: MLMultiArray], chunkArrays: [[String: MLMultiArray]], outputBackingMapping: [String: String]) {
        self.sharedArrays = sharedArrays
        self.chunkArrays = chunkArrays
        self.outputBackingMapping = outputBackingMapping
    }

    convenience init(chunkModels: [MLModel]) {
        var sharedArrays = [String: MLMultiArray]()
        var chunkArrays = [[String: MLMultiArray]]()
        let outputBackingMapping = ["new_x": "x"]

        for model in chunkModels {
            var floatShapes = [String: [Int]]()
            let description = model.modelDescription
            let allDescriptions = description.inputDescriptionsByName
                .merging(description.outputDescriptionsByName, uniquingKeysWith: { a, _ in a })

            for (name, desc) in allDescriptions {
                guard let constraint = desc.multiArrayConstraint, constraint.dataType == .float16 else { continue }
                if outputBackingMapping.keys.contains(name) { continue }
                floatShapes[name] = constraint.shape.map(\.intValue)
            }

            let isShared: (String) -> Bool = { !$0.contains("cache") }
            for (name, shape) in floatShapes where isShared(name) {
                if sharedArrays[name] == nil {
                    sharedArrays[name] = MLMultiArray.emptyIOSurfaceArray(shape: shape)!
                }
            }
            var currentChunkArrays = [String: MLMultiArray]()
            for (name, shape) in floatShapes where !isShared(name) {
                currentChunkArrays[name] = MLMultiArray.emptyIOSurfaceArray(shape: shape)!
            }
            chunkArrays.append(currentChunkArrays)
        }

        self.init(sharedArrays: sharedArrays, chunkArrays: chunkArrays, outputBackingMapping: outputBackingMapping)
    }

    private func array(named name: String, for chunkIndex: Int) -> MLMultiArray? {
        sharedArrays[name] ?? chunkArrays[chunkIndex][name]
    }

    func inputFeatures(forChunk chunkIndex: Int, model: MLModel) -> [String: MLMultiArray] {
        Dictionary(uniqueKeysWithValues: model.modelDescription.inputDescriptionsByName.keys.compactMap { name in
            self.array(named: name, for: chunkIndex).map { (name, $0) }
        })
    }

    func outputBackings(forChunk chunkIndex: Int, model: MLModel) -> [String: MLMultiArray] {
        Dictionary(uniqueKeysWithValues: model.modelDescription.outputDescriptionsByName.keys.compactMap { name in
            let backingName = outputBackingMapping[name] ?? name
            return self.array(named: backingName, for: chunkIndex).map { (name, $0) }
        })
    }

    /// Builds one chunk's input feature set for the current token window — including the
    /// sliding-window `input_ids` padding scheme every chunk's cache alignment depends on.
    ///
    /// Ported structurally (not from paraphrase) from the reference's own diagram: the cache
    /// covers everything before the current input segment; the input segment holds up to
    /// `inputLength` fresh tokens, zero-padded on the right when there are fewer; once a segment
    /// fills, the *next* call's cache has shifted to absorb it. `full_sequence_length` (when the
    /// model exposes it) exists so RoPE stays correct for the real tokens in a padded segment —
    /// the causal mask alone keeps the pad positions from influencing anything, but RoPE's
    /// rotation angle for the real positions still needs the true sequence length, not the padded
    /// one.
    func featureProvider(forChunk chunkIndex: Int, model: MLModel, tokens: [Int]) throws -> MLFeatureProvider {
        var features = inputFeatures(forChunk: chunkIndex, model: model)

        var padCount = 0
        let inputDescriptions = model.modelDescription.inputDescriptionsByName
        if let inputIDsConstraint = inputDescriptions["input_ids"]?.multiArrayConstraint {
            let inputShape = inputIDsConstraint.shape.map(\.intValue)
            let inputLength = inputShape.last!

            let suffixLength = tokens.isEmpty ? 0 : (tokens.count - 1) % inputLength + 1
            let inputTokens = tokens.suffix(suffixLength).map { Int32($0) }
            padCount = max(0, inputLength - inputTokens.count)
            let paddedInputTokens = inputTokens + Array(repeating: Int32(0), count: padCount)
            let inputIDs = MLShapedArray<Int32>(scalars: paddedInputTokens, shape: inputShape)
            features["input_ids"] = MLMultiArray(inputIDs)
        }

        if inputDescriptions["full_sequence_length"]?.multiArrayConstraint != nil {
            let fullSequenceLength = MLShapedArray(repeating: Int32(tokens.count + padCount), shape: [1])
            features["full_sequence_length"] = MLMultiArray(fullSequenceLength)
        }

        return try MLDictionaryFeatureProvider(dictionary: features)
    }
}

// MARK: - Async KV-cache shifting

/// Runs the cache-shift model (`cache-processor.mlmodelc`) asynchronously against a chunk's just-
/// produced cache outputs, so the *next* chunk's forward pass isn't blocked waiting for this one's
/// cache to update — only that chunk's own *next* prediction needs to wait, via `wait(forChunk:)`.
nonisolated final class PipelineCacheProcessor {
    private var chunkTasks: [Task<Void, Error>?]
    private let processorModel: MLModel

    init(chunkCount: Int, processorModel: MLModel) {
        self.chunkTasks = Array(repeating: nil, count: chunkCount)
        self.processorModel = processorModel
    }

    /// Submits one async update per attention block in this chunk, run concurrently — each block's
    /// K/V cache is independent, so there's no reason to serialize them.
    func submit(inputs: MLFeatureProvider, outputs: MLFeatureProvider, forChunk chunkIndex: Int) {
        let pair = PipelineCacheUpdate(inputs: inputs, outputs: outputs)
        let processorModel = processorModel
        chunkTasks[chunkIndex] = Task<Void, Error> {
            try await withThrowingTaskGroup(of: Void.self) { group in
                for blockIndex in 0..<inputs.cacheBlockCount() {
                    group.addTask {
                        let blockInputs = try pair.cacheProcessorInputs(forBlock: blockIndex)
                        let options = pair.cacheProcessorOptions(forBlock: blockIndex)
                        _ = try await processorModel.prediction(from: blockInputs, options: options)
                    }
                }
                try await group.waitForAll()
            }
        }
    }

    /// Blocks until any pending cache update for this chunk has finished — called before that
    /// chunk's *next* prediction reads its (by-then-updated-in-place) cache arrays.
    func wait(forChunk chunkIndex: Int) async throws {
        guard let task = chunkTasks[chunkIndex] else { return }
        try await task.value
    }
}

private nonisolated extension MLFeatureProvider {
    /// How many attention blocks this chunk has, inferred from its highest `k_cache_<N>` input —
    /// there's no other single source for "block count" on a loaded `MLFeatureProvider`.
    func cacheBlockCount() -> Int {
        let maxIndex = featureNames
            .filter { $0.contains("k_cache") }
            .compactMap { $0.components(separatedBy: "_").last }
            .compactMap { Int($0) }
            .max()
        return (maxIndex ?? -1) + 1
    }
}

private nonisolated struct PipelineCacheUpdate {
    let inputs: MLFeatureProvider
    let outputs: MLFeatureProvider

    func cacheProcessorInputs(forBlock blockIndex: Int) throws -> MLFeatureProvider {
        let dict: [String: MLMultiArray] = [
            "old_k_cache": inputs.featureValue(for: "k_cache_\(blockIndex)")!.multiArrayValue!,
            "new_k_cache": outputs.featureValue(for: "new_k_cache_\(blockIndex)")!.multiArrayValue!,
            "old_v_cache": inputs.featureValue(for: "v_cache_\(blockIndex)")!.multiArrayValue!,
            "new_v_cache": outputs.featureValue(for: "new_v_cache_\(blockIndex)")!.multiArrayValue!,
        ]
        return try MLDictionaryFeatureProvider(dictionary: dict)
    }

    /// The updated cache is written back into the *existing* `k_cache_N`/`v_cache_N` input arrays
    /// in place — the same arrays `PipelineArrayStore` keeps handing to this chunk on every
    /// subsequent call, so "updating the cache" and "the chunk's next input already reflecting
    /// that update" are the same write.
    func cacheProcessorOptions(forBlock blockIndex: Int) -> MLPredictionOptions {
        let options = MLPredictionOptions()
        options.outputBackings = [
            "updated_k_cache": inputs.featureValue(for: "k_cache_\(blockIndex)")!.multiArrayValue!,
            "updated_v_cache": inputs.featureValue(for: "v_cache_\(blockIndex)")!.multiArrayValue!,
        ]
        return options
    }
}

private nonisolated extension String {
    func trailingNumberSuffix() -> Int? {
        guard let number = self.split(separator: "_").last else { return nil }
        return Int(number)
    }
}

nonisolated extension MLFeatureProvider {
    /// Every `logit`-prefixed output, in shard order — a vocab this large is sometimes split
    /// across several output tensors rather than one, and the caller needs them concatenated in
    /// the right order to reconstruct a full row.
    func orderedLogitChunks() -> [MLMultiArray] {
        featureNames
            .filter { $0.hasPrefix("logit") }
            .sorted { ($0.trailingNumberSuffix() ?? -1) < ($1.trailingNumberSuffix() ?? -1) }
            .compactMap { featureValue(for: $0)?.multiArrayValue }
    }
}
