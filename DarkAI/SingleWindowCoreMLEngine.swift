import Foundation
import CoreML

/// Runs a Core ML text-generation model shaped like OpenELM-270M-Instruct: one `.mlpackage`, no
/// cache, a single fixed-size input/output window recomputed from scratch every step.
///
/// The published `.mlpackage` for this model (verified directly against its spec — see
/// `Manifest.json`/`model.mlmodel` under `ModelCatalog.coreMLModels`) has a single input,
/// `input_ids` — an Int32 `MLMultiArray` of fixed shape `[1, 128]` — and a single output,
/// `logits` — a Float32 `MLMultiArray` of shape `[1, 128, 32000]`. There is no `MLState`/KV-cache
/// input anywhere in the graph: every step re-runs the full 128-token window from scratch, and
/// causal attention means whatever sits in the window past the real sequence length never
/// influences the logits at an earlier position, so padding the tail with zeros is safe. That
/// architecture — not a growing KV cache — is what makes this backend's total capacity a hard
/// 128 tokens (prompt + reply combined), and is why this actor's shape is much simpler than
/// `LlamaRunner`'s: there is no context-window planning, no GPU-layer offload, no quantized KV
/// cache format to choose between.
actor SingleWindowCoreMLEngine: CoreMLEngine {

    /// The fixed sequence length every input/output tensor is shaped for. Not a setting — it's
    /// baked into the model graph.
    static let fixedSequenceLength = 128

    enum RunnerError: Error, LocalizedError {
        case packageNotFound
        case unexpectedModelShape(String)
        case predictionFailed

        var errorDescription: String? {
            switch self {
            case .packageNotFound: return "The Core ML model package could not be found."
            case .unexpectedModelShape(let detail): return "This Core ML model doesn't match what CoreMLRunner expects: \(detail)"
            case .predictionFailed: return "Core ML prediction failed."
            }
        }
    }

    private var model: MLModel?
    private let tokenizer = CoreMLTokenizer()
    private var isCancelledFlag = false

    /// Mirrors `LlamaRunner.isBusyGenerating` — a second overlapping call here wouldn't corrupt
    /// shared native state the way it would for llama.cpp's KV cache (there's no persistent
    /// per-call state at all), but it would still let two generations race on `isCancelledFlag`,
    /// so the same "decline rather than interleave" rule applies.
    private var isBusyGenerating = false

    var isLoaded: Bool { model != nil }

    /// The model's own trained/usable context, in the sense `LLMManager` asks `LlamaRunner` for
    /// it — here it's simply the fixed window, since there's no separate "trained vs allocated"
    /// distinction for a non-stateful model.
    func getContextWindowTokens() -> Int { Self.fixedSequenceLength }
    func getTrainedContextTokens() -> Int { Self.fixedSequenceLength }

    func requestCancel() {
        isCancelledFlag = true
    }

    // MARK: - Load

    func load(path: String) async throws {
        unload()
        let packageURL = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: packageURL.path) else {
            throw RunnerError.packageNotFound
        }

        let compiledURL = try await compiledModelURL(for: packageURL)

        let configuration = MLModelConfiguration()
        // `.all` rather than `.cpuAndNeuralEngine`: it lets Core ML fall back to GPU per-op if
        // the ANE compiler rejects part of the graph, instead of forcing CPU for the whole model
        // the way `.cpuAndNeuralEngine` would on a rejected op. There is no public API that
        // reports which engine actually ran a given op, so this backend doesn't attempt to
        // detect or surface "ran on ANE" — Core ML's own placement is trusted as-is.
        configuration.computeUnits = .all

        let loaded = try MLModel(contentsOf: compiledURL, configuration: configuration)

        // Fail loudly and specifically here rather than downstream in `predict`, where a shape
        // mismatch would otherwise surface as an opaque Core ML error with no context about what
        // this backend actually expected.
        guard let inputDescription = loaded.modelDescription.inputDescriptionsByName["input_ids"],
              let inputConstraint = inputDescription.multiArrayConstraint,
              inputConstraint.shape.map(\.intValue) == [1, Self.fixedSequenceLength] else {
            throw RunnerError.unexpectedModelShape("expected an \"input_ids\" input shaped [1, \(Self.fixedSequenceLength)]")
        }
        guard let outputDescription = loaded.modelDescription.outputDescriptionsByName["logits"],
              let outputConstraint = outputDescription.multiArrayConstraint,
              outputConstraint.shape.count == 3,
              outputConstraint.shape[0].intValue == 1,
              outputConstraint.shape[1].intValue == Self.fixedSequenceLength else {
            throw RunnerError.unexpectedModelShape("expected a \"logits\" output shaped [1, \(Self.fixedSequenceLength), vocabSize]")
        }

        model = loaded
    }

    /// Locates (or produces) the compiled `.mlmodelc` Xcode would normally build automatically
    /// for a *bundled* `.mlpackage` — this one arrived via download, so nothing has compiled it
    /// yet. Cached next to the package so this only costs real time once per install, not once
    /// per app launch.
    private func compiledModelURL(for packageURL: URL) async throws -> URL {
        let compiledURL = packageURL.deletingLastPathComponent()
            .appendingPathComponent(packageURL.deletingPathExtension().lastPathComponent)
            .appendingPathExtension("mlmodelc")
        let fm = FileManager.default

        if fm.fileExists(atPath: compiledURL.path),
           let compiledModified = try? fm.attributesOfItem(atPath: compiledURL.path)[.modificationDate] as? Date,
           let packageModified = try? fm.attributesOfItem(atPath: packageURL.path)[.modificationDate] as? Date,
           compiledModified >= packageModified {
            return compiledURL
        }

        let freshlyCompiled = try await MLModel.compileModel(at: packageURL)
        if fm.fileExists(atPath: compiledURL.path) {
            try fm.removeItem(at: compiledURL)
        }
        try fm.moveItem(at: freshlyCompiled, to: compiledURL)
        AppFiles.excludeFromBackup(compiledURL)
        return compiledURL
    }

    func unload() {
        model = nil
    }

    // MARK: - Generation

    /// Same public shape as `LlamaRunner.generateStream` so `LLMManager` can dispatch to either
    /// backend uniformly. `onContextTruncated`/`onThinkingProgress` are accepted but unused —
    /// this model isn't a "thinking" model, and hitting the 128-token ceiling is reported
    /// in-stream as plain text rather than through the truncation-callback UX `LlamaRunner` uses
    /// for its (much larger, sliding) context window.
    func generateStream(
        messages: [(role: String, content: String)],
        maxTokens: Int,
        temperature: Float,
        continuation: AsyncStream<String>.Continuation,
        onContextTruncated: @escaping @Sendable () -> Void = {},
        onThinkingProgress: @escaping @Sendable (Int) -> Void = { _ in }
    ) async {
        guard let model else {
            continuation.finish()
            return
        }
        guard !isBusyGenerating else {
            Task { @MainActor in
                LogManager.shared.log("SingleWindowCoreMLEngine: generateStream called while another generation is already in flight — declining.")
            }
            continuation.finish()
            return
        }
        isBusyGenerating = true
        defer { isBusyGenerating = false }
        isCancelledFlag = false
        self.temperature = temperature

        // No chat template to apply — OpenELM ships none of its own, so this mirrors
        // `LlamaRunner`'s own no-template fallback shape exactly (see LLMManager.swift, the
        // `formattedLen <= 0` branch of `generateStream`).
        let prompt = messages.map { "\($0.role): \($0.content)" }.joined(separator: "\n") + "\nassistant:\n"

        var tokens: [Int32]
        do {
            tokens = try await tokenizer.encode(prompt, addBOS: true)
        } catch {
            continuation.yield("[Tokenizer error: \(error.localizedDescription)]")
            continuation.finish()
            return
        }
        guard !tokens.isEmpty else {
            continuation.yield("[Error: tokenization returned empty]")
            continuation.finish()
            return
        }

        let windowSize = Self.fixedSequenceLength
        // Leave room for at least one generated token. A prompt that alone fills the window has
        // nowhere to put a reply — truncate from the front, keeping the BOS token, the same way
        // `LlamaRunner` truncates an over-long prompt.
        if tokens.count > windowSize - 1 {
            let bos = tokens[0]
            tokens = [bos] + Array(tokens.suffix(windowSize - 2))
            onContextTruncated()
        }

        // Allocated once and reused for every step below, instead of a fresh `MLMultiArray` (plus
        // 128 individually NSNumber-boxed subscript writes) on every single generated token — the
        // window is the same fixed shape for the whole call, so there's nothing per-token about
        // this allocation except its contents.
        let inputArray: MLMultiArray
        do {
            inputArray = try MLMultiArray(shape: [1, NSNumber(value: windowSize)], dataType: .int32)
        } catch {
            continuation.yield("[Error: \(error.localizedDescription)]")
            continuation.finish()
            return
        }

        var generatedCount = 0
        var hitWindowCeiling = false
        while generatedCount < maxTokens {
            if isCancelledFlag { break }
            if tokens.count >= windowSize { hitWindowCeiling = true; break }

            let nextTokenId: Int32
            do {
                nextTokenId = try await predict(model: model, tokens: tokens, inputArray: inputArray)
            } catch {
                Task { @MainActor in
                    LogManager.shared.log("SingleWindowCoreMLEngine: prediction failed — \(error.localizedDescription)")
                }
                break
            }

            if (try? await tokenizer.isEndOfGeneration(nextTokenId)) == true { break }

            let piece = (try? await tokenizer.decode(nextTokenId)) ?? ""
            if !piece.isEmpty {
                continuation.yield(piece)
            }
            tokens.append(nextTokenId)
            generatedCount += 1

            // Same reasoning as `LlamaRunner`'s per-token yield: without it, a fast model on a
            // fast device can hold the actor's executor continuously enough to starve other work
            // queued on it (like a cancel request) until generation ends on its own.
            await Task.yield()
        }

        if hitWindowCeiling {
            continuation.yield("\n\n[Reached this model's 128-token limit — start a new conversation to continue.]")
        }
        continuation.finish()
    }

    /// One forward pass over the current (padded-to-128) window, returning the sampled next
    /// token. Every call recomputes the whole window — there is no cache to extend — which is
    /// expected and fine at this model's size; see the type-level doc comment.
    ///
    /// `inputArray` is caller-owned and reused across every step of one `generateStream` call
    /// (see there) — written here via a raw buffer pointer rather than 128 per-call NSNumber
    /// subscript writes. Uses the async `prediction(from:options:)` rather than the synchronous
    /// `prediction(from:)`, matching `ChunkedPipelineCoreMLEngine`: the synchronous call blocks
    /// this actor's executor for the full inference duration on every token, which also delays a
    /// queued `requestCancel()` until the in-flight prediction returns.
    private func predict(model: MLModel, tokens: [Int32], inputArray: MLMultiArray) async throws -> Int32 {
        let windowSize = Self.fixedSequenceLength
        inputArray.withUnsafeMutableBufferPointer(ofType: Int32.self) { buffer, _ in
            for i in 0..<windowSize {
                buffer[i] = i < tokens.count ? tokens[i] : 0
            }
        }

        let input = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputArray])
        let output = try await model.prediction(from: input, options: MLPredictionOptions())
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw RunnerError.predictionFailed
        }

        let vocabSize = logits.shape[2].intValue
        let position = tokens.count - 1
        let sampled = logits.withUnsafeBufferPointer(ofType: Float.self) { buffer -> Int32 in
            let base = position * vocabSize
            let row = UnsafeBufferPointer(rebasing: buffer[base..<(base + vocabSize)])
            return CoreMLSampling.sample(row: row, temperature: temperature)
        }
        return sampled
    }

    /// Set from `generateStream` at the top of each call so `predict` (a nonisolated-in-spirit
    /// helper called from inside the actor) can read it without threading it through every call.
    private var temperature: Float = 0.7
}
