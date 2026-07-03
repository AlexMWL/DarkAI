import Foundation
import Combine
import LlamaSwift
import os
import UIKit
import Darwin

// MARK: - State Types

enum ModelLoadState {
    case unloaded
    case loading(progress: Double, status: String)
    case loaded(modelName: String, sizeGB: Double)
    case failed(error: String)
}

enum MemorySafetyStatus: Equatable {
    case safe
    case warning(requiredGB: Double, availableGB: Double)
    case dangerous(requiredGB: Double, availableGB: Double)
}

// MARK: - LlamaRunner Actor
// Wraps the raw llama.cpp C-API in a background actor to keep inference off the main thread.



actor LlamaRunner {


    private var model: OpaquePointer? = nil
    private var context: OpaquePointer? = nil
    private var nCtxTokens: Int = 2048  // actual context window in tokens
    private var isCancelled = false

    var isLoaded: Bool { model != nil && context != nil }

    init() {
        // Initialize the backend once for the lifetime of this actor
        llama_log_set({ level, text, user_data in
            guard let text = text, let str = String(cString: text, encoding: .utf8) else { return }
            let cleanStr = str.trimmingCharacters(in: .whitespacesAndNewlines)
            if !cleanStr.isEmpty {
                DispatchQueue.main.async {
                    LogManager.shared.log("LLAMA: \(cleanStr)")
                }
            }
        }, nil)
        llama_backend_init()
    }

    func load(path: String, availableMemoryGB: Double, modelSizeGB: Double, contextLimit: Int) throws {
        unloadModelOnly()

        var modelParams = llama_model_default_params()
        // Prevent Metal buffer zeros ("...") on Gemma's massive vocab by restricting GPU layers
        modelParams.n_gpu_layers = path.lowercased().contains("gemma") ? 15 : 99
        modelParams.use_mmap = true
        modelParams.use_mlock = false

        guard let mdl = llama_model_load_from_file(path, modelParams) else {
            throw NSError(domain: "LlamaRunner", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to load GGUF. Check the file is a valid model and fits in RAM."])
        }
        self.model = mdl

        let trainedCtx = llama_model_n_ctx_train(mdl)
        let nCtx = min(Int32(contextLimit), Int32(trainedCtx))

        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx   = UInt32(nCtx)
        ctxParams.n_batch = UInt32(min(nCtx, 512))
        
        // Optimized for Apple Silicon: Using all cores (including E-cores) degrades performance.
        let optimalThreads = Int32(max(2, min(4, ProcessInfo.processInfo.activeProcessorCount / 2)))
        ctxParams.n_threads       = optimalThreads
        ctxParams.n_threads_batch = optimalThreads
        ctxParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED

        guard let ctx = llama_init_from_model(mdl, ctxParams) else {
            llama_model_free(mdl)
            self.model = nil
            throw NSError(domain: "LlamaRunner", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to create inference context. The model may require more RAM."])
        }
        self.context = ctx
        self.nCtxTokens = Int(nCtx)
    }

    /// Unloads only the model+context, leaving the backend alive for the next load.
    func unloadModelOnly() {
        isCancelled = true  // Stop any ongoing generation
        if let ctx = context { llama_free(ctx) }
        if let mdl = model   { llama_model_free(mdl) }
        context = nil
        model   = nil
    }

    /// Full teardown — call only when the actor itself is being destroyed.
    func unload() {
        isCancelled = true
        if let ctx = context { llama_free(ctx) }
        if let mdl = model   { llama_model_free(mdl) }
        context = nil
        model   = nil
        llama_backend_free()
    }

    func requestCancel() {
        isCancelled = true
    }

    /// Tokenizes a string. Returns empty array on failure.
    private func tokenize(_ text: String, addBOS: Bool) -> [llama_token] {
        guard let mdl = model else { return [] }
        let vocab = llama_model_get_vocab(mdl)
        let utf8 = text.utf8
        let nTokensMax = Int32(utf8.count + 8)
        var tokens = [llama_token](repeating: 0, count: Int(nTokensMax))
        let n = llama_tokenize(vocab, text, Int32(utf8.count), &tokens, nTokensMax, addBOS, false)
        guard n > 0 else { return [] }
        return Array(tokens.prefix(Int(n)))
    }

    /// Checks whether the loaded model advertises any vision/multimodal capability via its metadata.
    func supportsVision() -> Bool {
        guard let mdl = model else { return false }
        // Check model metadata key for projector or mmproj type
        let count = llama_model_meta_count(mdl)
        for i in 0..<count {
            var keyBuf = [CChar](repeating: 0, count: 512)
            llama_model_meta_key_by_index(mdl, i, &keyBuf, 512)
            let key = String(cString: keyBuf).lowercased()
            if key.contains("vision") || key.contains("clip") || key.contains("mmproj") || key.contains("multimodal") {
                return true
            }
        }
        return false
    }

    /// Runs autoregressive inference and streams results via the continuation.
    func generateStream(
        messages: [(role: String, content: String)],
        maxTokens: Int,
        temperature: Float,
        continuation: AsyncStream<String>.Continuation
    ) async {
        guard let ctx = context, let mdl = model else {
            continuation.finish()
            return
        }

        isCancelled = false

        let vocab = llama_model_get_vocab(mdl)

        // --- Apply Native Chat Template ---
        var chatStructs: [llama_chat_message] = []
        for msg in messages {
            let rolePtr = strdup(msg.role)
            let contentPtr = strdup(msg.content)
            chatStructs.append(llama_chat_message(role: rolePtr, content: contentPtr))
        }

        let tmpl = llama_model_chat_template(mdl, nil)
        var tmplBuf = [CChar](repeating: 0, count: 32768)
        let formattedLen = llama_chat_apply_template(tmpl, chatStructs, chatStructs.count, true, &tmplBuf, Int32(tmplBuf.count))
        
        // Free the allocated strings
        for chat in chatStructs {
            free(UnsafeMutableRawPointer(mutating: chat.role))
            free(UnsafeMutableRawPointer(mutating: chat.content))
        }

        let finalPrompt: String
        if formattedLen > 0 && formattedLen < tmplBuf.count {
            finalPrompt = String(cString: tmplBuf)
        } else {
            // Fallback for models missing a template
            finalPrompt = messages.map { "\($0.role): \($0.content)" }.joined(separator: "\n") + "\nassistant:\n"
        }

        // 1. Tokenize prompt — guard against exceeding the context window
        var promptTokens = tokenize(finalPrompt, addBOS: true)
        guard !promptTokens.isEmpty else {
            continuation.yield("[Error: tokenization returned empty]")
            continuation.finish()
            return
        }

        let maxPromptTokens = max(1, nCtxTokens - min(maxTokens, nCtxTokens / 2))
        if promptTokens.count > maxPromptTokens {
            let pCount = promptTokens.count
            Task { @MainActor in
                LogManager.shared.log("LlamaRunner: Warning - Prompt tokens (\(pCount)) exceed safe threshold (\(maxPromptTokens)). Truncating.")
            }
            let bos = promptTokens[0]
            promptTokens = [bos] + Array(promptTokens.suffix(maxPromptTokens - 1))
        }

        // 2. Clear KV cache to prevent crashes across multiple prompts
        if let mem = llama_get_memory(ctx) {
            llama_memory_clear(mem, true)
        }

        // 3. Prefill (evaluate the prompt) in chunks of n_batch
        let batchSize = Int(llama_n_batch(ctx))
        var batch = llama_batch_init(Int32(batchSize), 0, 1)
        defer { llama_batch_free(batch) }
        
        var batchStart = 0
        while batchStart < promptTokens.count {
            let chunkEnd = min(promptTokens.count, batchStart + batchSize)
            let chunkLen = chunkEnd - batchStart
            
            for i in 0..<chunkLen {
                let tokenIdx = batchStart + i
                batch.token[i] = promptTokens[tokenIdx]
                batch.pos[i] = Int32(tokenIdx)
                batch.n_seq_id[i] = 1
                if let seqIdPtr = batch.seq_id[i] {
                    seqIdPtr.pointee = 0
                }
                batch.logits[i] = (tokenIdx == promptTokens.count - 1) ? 1 : 0
            }
            batch.n_tokens = Int32(chunkLen)
            
            if llama_decode(ctx, batch) != 0 {
                continuation.yield("[Error: prefill decode failed]")
                continuation.finish()
                return
            }
            batchStart += batchSize
        }

        // 4. Autoregressive decoding with temperature + top-p sampling to prevent repetition
        let eosToken = llama_vocab_eos(vocab)
        let nVocab = Int(llama_vocab_n_tokens(vocab))

        // IMPORTANT: Reset nPos each generation — previously unbounded growth caused crashes!
        var nPos = Int32(promptTokens.count)
        var generatedCount = 0

        // Sampling parameters — temperature is passed in dynamically
        let topP: Float = 0.9
        let repeatPenalty: Float = 1.1
        var recentTokens = [llama_token]()

        var singleBatch = llama_batch_init(1, 0, 1)
        defer { llama_batch_free(singleBatch) }

        var accumulatedOutput = ""

        while generatedCount < maxTokens {
            await Task.yield()
            guard !isCancelled else { break }

            guard let logitsPtr = llama_get_logits_ith(ctx, -1) else { break }

            // Copy logits to a Swift array for manipulation
            var logits = Array(UnsafeBufferPointer(start: logitsPtr, count: nVocab))

            // Apply repeat penalty
            for tok in recentTokens {
                let idx = Int(tok)
                if idx >= 0 && idx < nVocab {
                    if logits[idx] > 0 {
                        logits[idx] /= repeatPenalty
                    } else {
                        logits[idx] *= repeatPenalty
                    }
                }
            }

            // Temperature scaling
            if temperature > 0 && temperature != 1.0 {
                let invTemp = 1.0 / Float(temperature)
                for i in 0..<nVocab { logits[i] *= invTemp }
            }

            let maxLogit = logits.max() ?? 0
            
            // Optimization: Filter out incredibly improbable logits before expf and sorting (O(N log N) -> O(1))
            let logitThreshold = maxLogit - 12.0
            
            var validLogits: [(Int, Float)] = []
            validLogits.reserveCapacity(1000)
            for i in 0..<nVocab {
                let val = logits[i]
                if val > logitThreshold {
                    validLogits.append((i, val))
                }
            }

            // Softmax on filtered logits
            var expLogits = validLogits.map { ($0.0, expf($0.1 - maxLogit)) }
            let sumExp = expLogits.reduce(0) { $0 + $1.1 }
            if sumExp > 0 {
                for i in 0..<expLogits.count { expLogits[i].1 /= sumExp }
            }

            // Top-p nucleus sampling
            let sorted = expLogits.sorted { $0.1 > $1.1 }
            var cumulative: Float = 0
            var nucleus: [(Int, Float)] = []
            for item in sorted {
                nucleus.append(item)
                cumulative += item.1
                if cumulative >= topP { break }
            }

            // Sample from nucleus
            let nucleusSum = nucleus.reduce(0) { $0 + $1.1 }
            var rand = Float.random(in: 0..<nucleusSum)
            var sampledId = nucleus.first?.0 ?? 0
            for (idx, prob) in nucleus {
                rand -= prob
                if rand <= 0 {
                    sampledId = idx
                    break
                }
            }

            let bestId = llama_token(sampledId)

            if bestId == eosToken { break }

            // Detokenize
            var tokenBuf = [CChar](repeating: 0, count: 256)
            let nChars = llama_token_to_piece(vocab, bestId, &tokenBuf, 256, 0, false)
            if nChars > 0 {
                let piece = String(bytes: tokenBuf.prefix(Int(nChars)).map { UInt8(bitPattern: $0) }, encoding: .utf8) ?? ""
                if !piece.isEmpty {
                    accumulatedOutput += piece

                    // Stop String check — prevents models talking to themselves
                    let lower = accumulatedOutput.lowercased()
                    if lower.hasSuffix("[inst]") || lower.hasSuffix("user:") || lower.hasSuffix("<|im_end|>") || lower.hasSuffix("<start_of_turn>user") || lower.hasSuffix("<|user|>") {
                        break
                    }

                    continuation.yield(piece)
                }
            }

            // Track recent tokens for repeat penalty (sliding window of last 64)
            recentTokens.append(bestId)
            if recentTokens.count > 64 { recentTokens.removeFirst() }

            // Advance
            singleBatch.token[0] = bestId
            singleBatch.pos[0] = nPos
            singleBatch.n_seq_id[0] = 1
            if let seqIdPtr = singleBatch.seq_id[0] {
                seqIdPtr.pointee = 0
            }
            singleBatch.logits[0] = 1
            singleBatch.n_tokens = 1

            // Guard against context overflow gracefully
            if nPos >= Int32(nCtxTokens) - 1 {
                Task { @MainActor in LogManager.shared.log("LlamaRunner: Hard context limit reached. Stopping generation.") }
                continuation.yield("\n\n[System: Context window limit reached. Generation stopped.]")
                break
            }

            if llama_decode(ctx, singleBatch) != 0 { break }

            nPos += 1
            generatedCount += 1
        }

        continuation.finish()
    }

    deinit {
        if let ctx = context { llama_free(ctx) }
        if let mdl = model   { llama_model_free(mdl) }
        llama_backend_free()
    }
}

// MARK: - LLMManager

@MainActor
class LLMManager: ObservableObject {
    @Published var loadState: ModelLoadState = .unloaded
    @Published var isGenerating: Bool = false
    @Published var systemMemoryGB: Double = 0.0
    @Published var activeModelURL: URL? = nil
    @Published var generationSpeed: Double = 0.0
    @Published var modelSupportsVision: Bool = false

    @Published var maxTokens: Int = UserDefaults.standard.object(forKey: "maxTokens") as? Int ?? 512 {
        didSet { UserDefaults.standard.set(maxTokens, forKey: "maxTokens") }
    }
    
    @Published var contextTokenLimit: Int = UserDefaults.standard.object(forKey: "contextTokenLimit") as? Int ?? 8192 {
        didSet { UserDefaults.standard.set(contextTokenLimit, forKey: "contextTokenLimit") }
    }
    
    @Published var temperature: Double = UserDefaults.standard.object(forKey: "temperature") as? Double ?? 0.85 {
        didSet { UserDefaults.standard.set(temperature, forKey: "temperature") }
    }
    
    @Published var chaosModeEnabled: Bool = UserDefaults.standard.bool(forKey: "chaosModeEnabled") {
        didSet { UserDefaults.standard.set(chaosModeEnabled, forKey: "chaosModeEnabled") }
    }
    
    @Published var lastUsedModelPath: String? = UserDefaults.standard.string(forKey: "lastUsedModelPath") {
        didSet {
            if let path = lastUsedModelPath {
                UserDefaults.standard.set(path, forKey: "lastUsedModelPath")
            } else {
                UserDefaults.standard.removeObject(forKey: "lastUsedModelPath")
            }
        }
    }
    
    var safeContextLimit: Int {
        let maxMemory = systemMemoryGB * 0.40
        switch loadState {
        case .loaded(_, let sizeGB):
            let activeModelWeightRAM = sizeGB * 0.10
            let availableForKV = maxMemory - activeModelWeightRAM
            if availableForKV <= 0 { return 2048 }
            let gbPer1kTokens = max(0.04, sizeGB * 0.02)
            let safeLimit = (availableForKV / gbPer1kTokens) * 1000
            return max(2048, min(32768, Int(safeLimit)))
        default:
            let availableForKV = maxMemory - 1.5
            if availableForKV <= 0 { return 4096 }
            let gbPer1kTokens = 0.08
            let safeLimit = (availableForKV / gbPer1kTokens) * 1000
            return max(2048, min(32768, Int(safeLimit)))
        }
    }

    private let runner = LlamaRunner()

    init() {
        self.systemMemoryGB = getPhysicalMemory()
        setupAppLifecycleObservers()
        freeSwapStorage()
    }

    // MARK: - App Lifecycle - Clean unload on background/termination

    private func setupAppLifecycleObservers() {
        NotificationCenter.default.addObserver(
            forName: UIApplication.willTerminateNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { await self?.runner.unload() }
        }
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            guard let self else { return }
            Task {
                await self.runner.requestCancel()
                await self.runner.unloadModelOnly()
                await MainActor.run {
                    self.loadState = .failed(error: "System memory pressure — model unloaded safely. Reload when ready.")
                    self.activeModelURL = nil
                    self.modelSupportsVision = false
                }
            }
        }
    }

    // MARK: - Memory
    func freeSwapStorage() {
        Task.detached(priority: .background) {
            let fileManager = FileManager.default
            let urls = fileManager.urls(for: .documentDirectory, in: .userDomainMask)
            guard let docDir = urls.first else { return }
            
            // Delete old chunk files if they exist
            if let existingFiles = try? fileManager.contentsOfDirectory(atPath: docDir.path) {
                for file in existingFiles where file.hasPrefix("swap_chunk_") {
                    try? fileManager.removeItem(atPath: docDir.appendingPathComponent(file).path)
                }
            }
            
            let filePath = docDir.appendingPathComponent("swap_reserve.bin").path
            if fileManager.fileExists(atPath: filePath) {
                try? fileManager.removeItem(atPath: filePath)
            }
        }
    }

    func getPhysicalMemory() -> Double {
        return Double(ProcessInfo.processInfo.physicalMemory) / (1024.0 * 1024.0 * 1024.0)
    }

    func getModelSizeGB(at url: URL) -> Double {
        guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize else { return 0 }
        return Double(size) / (1024.0 * 1024.0 * 1024.0)
    }

    func checkMemorySafety(modelSizeGB: Double) -> MemorySafetyStatus {
        let total = systemMemoryGB
        // We add 1.5 GB overhead for the app context and system tasks
        let required = modelSizeGB + 1.5 

        // If the required memory takes up more than 90% of what's available, it's a no-go
        if required > total * 0.90 {
            return .dangerous(requiredGB: required, availableGB: total)
        } else if required > total * 0.70 {
            return .warning(requiredGB: required, availableGB: total)
        }
        return .safe
    }

    // MARK: - Model Loading

    func loadModel(at url: URL, forceLoad: Bool = false) {
        let sizeGB = getModelSizeGB(at: url)
        let safety = checkMemorySafety(modelSizeGB: sizeGB)

        if case .dangerous = safety, !forceLoad {
            let req = String(format: "%.1f", sizeGB + 1.5)
            let avail = String(format: "%.1f", systemMemoryGB * 0.90)
            self.loadState = .failed(error: "Memory Failsafe: Model requires \(req) GB but only \(avail) GB is safely available.")
            return
        }

        activeModelURL = url
        self.loadState = .loading(progress: 0.1, status: "Initialising llama.cpp backend…")

        Task {
            do {
                await MainActor.run {
                    self.loadState = .loading(progress: 0.4, status: "Loading GGUF weights…")
                }

                let availMem = await MainActor.run { self.getPhysicalMemory() }

                let currentContextLimit = min(self.contextTokenLimit, self.safeContextLimit)
                try await runner.load(
                    path: url.path,
                    availableMemoryGB: availMem,
                    modelSizeGB: sizeGB,
                    contextLimit: currentContextLimit
                )

                let hasVision = await runner.supportsVision()

                await MainActor.run {
                    self.loadState = .loaded(modelName: url.lastPathComponent, sizeGB: sizeGB)
                    self.modelSupportsVision = hasVision
                    self.lastUsedModelPath = url.path
                }
            } catch {
                await MainActor.run {
                    self.loadState = .failed(error: error.localizedDescription)
                    self.modelSupportsVision = false
                }
            }
        }
    }

    func unloadModel() {
        Task {
            await runner.requestCancel()
            await runner.unloadModelOnly()
        }
        activeModelURL = nil
        loadState = .unloaded
        modelSupportsVision = false
    }

    func cancelGeneration() {
        Task { await runner.requestCancel() }
        isGenerating = false
    }

    // MARK: - Inference

    func generateResponse(
        prompt: String,
        history: [ChatMessage],
        systemPrompt: String,
        memoriesContext: String,
        ragContext: String,
        onToken: @escaping (String) -> Void,
        onComplete: @escaping (String) -> Void
    ) {
        guard case .loaded = loadState else {
            let msg = "⚠️ No model loaded. Go to Settings and load a GGUF model first."
            onToken(msg); onComplete(msg)
            return
        }

        isGenerating = true

        // Build System Context
        var systemBlock = ""
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "EEEE, MMMM d, yyyy 'at' h:mm a"
        let currentDateString = dateFormatter.string(from: Date())
        systemBlock += "Current Date and Time: \(currentDateString)\n\n"
        
        if !systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            systemBlock += systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines) + "\n\n"
        }
        if !memoriesContext.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            systemBlock += memoriesContext.trimmingCharacters(in: .whitespacesAndNewlines) + "\n\n"
        }
        if !ragContext.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            systemBlock += ragContext.trimmingCharacters(in: .whitespacesAndNewlines) + "\n\n"
        }

        var swiftMessages: [(role: String, content: String)] = []

        let contextLimit = contextTokenLimit
        let reservedGeneration = min(maxTokens, max(256, contextLimit / 2))
        let systemTokenEstimate = (systemBlock.count / 3) + 64
        let promptTokenEstimate = (prompt.count / 3) + 16
        let historyBudgetTokens = max(0, contextLimit - reservedGeneration - systemTokenEstimate - promptTokenEstimate - 128)

        var pairs: [(user: ChatMessage, assistant: ChatMessage)] = []
        var i = 0
        let msgs = history
        while i < msgs.count - 1 {
            if msgs[i].isUser && !msgs[i + 1].isUser {
                pairs.append((msgs[i], msgs[i + 1]))
                i += 2
            } else {
                i += 1
            }
        }

        var selectedPairs: [(user: ChatMessage, assistant: ChatMessage)] = []
        var usedTokens = 0
        for pair in pairs.reversed() {
            let estimate = (pair.user.text.count + pair.assistant.text.count) / 3 + 8
            if usedTokens + estimate > historyBudgetTokens { break }
            selectedPairs.insert(pair, at: 0)
            usedTokens += estimate
        }

        if !systemBlock.isEmpty {
            swiftMessages.append(("system", systemBlock.trimmingCharacters(in: .whitespacesAndNewlines)))
        }

        for pair in selectedPairs {
            swiftMessages.append(("user", pair.user.text))
            swiftMessages.append(("assistant", pair.assistant.text))
        }

        swiftMessages.append(("user", prompt))

        let tokenLimit = maxTokens

        Task {
            var accumulated = ""

            await MainActor.run { self.generationSpeed = 0.0 }
            let startTime = CFAbsoluteTimeGetCurrent()
            var tokenCount = 0

            let stream = AsyncStream<String> { continuation in
                Task {
                    await runner.generateStream(
                        messages: swiftMessages,
                        maxTokens: tokenLimit,
                        temperature: self.chaosModeEnabled ? 2.5 : Float(self.temperature),
                        continuation: continuation
                    )
                }
            }

            for await piece in stream {
                guard isGenerating else { break }

                tokenCount += 1
                let elapsed = max(0.01, CFAbsoluteTimeGetCurrent() - startTime)
                let tps = Double(tokenCount) / elapsed

                await MainActor.run {
                    self.generationSpeed = tps
                    onToken(piece)
                }

                accumulated += piece
            }

            await MainActor.run {
                isGenerating = false
                onComplete(accumulated)
            }
        }
    }

    func generateBackgroundAnalysis(prompt: String) async -> String? {
        guard case .loaded = loadState else { return nil }
        
        let msg = (role: "user", content: prompt)
        
        var accumulated = ""
        
        let stream = AsyncStream<String> { continuation in
            Task {
                await runner.generateStream(
                    messages: [msg],
                    maxTokens: 512, // Keep analysis reasonably bounded
                    temperature: 0.3, // Low temp for analytical tasks
                    continuation: continuation
                )
            }
        }
        
        for await piece in stream {
            accumulated += piece
        }
        
        return accumulated
    }

}
