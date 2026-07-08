import SwiftUI
import UniformTypeIdentifiers
import Photos

struct ContentView: View {
    @StateObject private var llmManager = LLMManager()
    @StateObject private var memoryManager = MemoryManager()
    @StateObject private var ragManager = RAGManager()
    @StateObject private var conversationManager = ConversationManager()
    @StateObject private var personalityManager = PersonalityManager()
    @StateObject private var diffusionManager = DiffusionManager()

    @AppStorage("customInstructions") private var customInstructions: String = "You are a local assistant. Respond with precise answers."
    @State private var enableRAG = true
    @State private var enableMemories = true

    @State private var showFileImporter = false
    @State private var showAutoLoadAlert = false
    @State private var pendingAttachmentName: String? = nil
    @State private var pendingAttachmentText: String? = nil

    @State private var inputText: String = ""
    @State private var showSettings = false
    @State private var showDrawer = false

    // Image generation state
    @State private var showDiffusionNotLoadedBanner = false
    @State private var diffusionBannerTask: Task<Void, Never>? = nil

    // Pulse animation state
    @State private var pulseActive = false
    
    var body: some View {
        ZStack {
            // MAIN VIEW
            VStack(spacing: 0) {
                
                // Custom Premium Navigation Header
                customHeaderView
                
                // Active status banner
                modelBanner
                
                // Messages board
                if let activeConv = conversationManager.activeConversation, !activeConv.messages.isEmpty {
                    ScrollViewReader { proxy in
                        ScrollView {
                            LazyVStack(spacing: 16) {
                                ForEach(activeConv.messages) { message in
                                    messageBubble(for: message)
                                        .id(message.id)
                                }
                            }
                            .padding()
                        }
                        // Keyboard Dismissal by Dragging List
                        .gesture(
                            DragGesture().onChanged { _ in
                                endEditing()
                            }
                        )
                        // Keyboard Dismissal by Tapping message area
                        .onTapGesture {
                            endEditing()
                        }
                        .onChange(of: activeConv.messages) { _ in
                            if let last = activeConv.messages.last {
                                withAnimation {
                                    proxy.scrollTo(last.id, anchor: .bottom)
                                }
                            }
                        }
                    }
                } else {
                    emptyStateView
                }
                
                // Active parameter indicators
                activeParametersIndicator
                
                // Chat input area
                inputArea
            }
            .background(GlitchBackgroundView().ignoresSafeArea())
            .blur(radius: showDrawer ? 4 : 0)
            .disabled(showDrawer)
            
            // DRAWER DIMMER OVERLAY
            if showDrawer {
                Color.black.opacity(0.6)
                    .ignoresSafeArea()
                    .onTapGesture {
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                            showDrawer = false
                        }
                    }
                    .transition(.opacity)
            }
            
            // CONVERSATIONS SIDEBAR DRAWER
            sidebarDrawer

            // DIFFUSION NOT-LOADED WARNING BANNER
            if showDiffusionNotLoadedBanner {
                VStack {
                    HStack(spacing: 10) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        VStack(alignment: .leading, spacing: 2) {
                            Text("No Diffusion Model Loaded")
                                .font(.system(size: 13, weight: .bold))
                                .foregroundColor(.white)
                            Text("Load a diffusion model in Settings to generate images.")
                                .font(.system(size: 11))
                                .foregroundColor(Color.white.opacity(0.7))
                        }
                        Spacer()
                        Button {
                            withAnimation { showDiffusionNotLoadedBanner = false }
                        } label: {
                            Image(systemName: "xmark")
                                .foregroundColor(.white.opacity(0.6))
                                .font(.system(size: 12, weight: .bold))
                        }
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.orange.opacity(0.18))
                            .overlay(RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.orange.opacity(0.5), lineWidth: 1))
                    )
                    .padding(.horizontal, 16)
                    .padding(.top, 12)
                    Spacer()
                }
                .transition(.move(edge: .top).combined(with: .opacity))
                .zIndex(10)
            }
        } // end ZStack
        .sheet(isPresented: $showSettings) {
            SettingsView(
                llmManager: llmManager,
                memoryManager: memoryManager,
                ragManager: ragManager,
                personalityManager: personalityManager,
                diffusionManager: diffusionManager,
                customInstructions: $customInstructions,
                enableRAG: $enableRAG,
                enableMemories: $enableMemories
            )
        }
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [
                .image, .pdf, .plainText
            ],
            allowsMultipleSelection: false
        ) { result in
            handleFileImport(result)
        }
        .onAppear {
            // Open to an existing empty chat, or create one if all chats have messages
            let emptyChat = conversationManager.conversations.first(where: { $0.messages.isEmpty })
            if let empty = emptyChat {
                conversationManager.selectConversation(id: empty.id)
            } else {
                conversationManager.createConversation()
            }
            
            if llmManager.lastUsedModelPath != nil && llmManager.activeModelURL == nil {
                showAutoLoadAlert = true
            }
        }
        .alert(isPresented: $showAutoLoadAlert) {
            Alert(
                title: Text("Load Previous Model?"),
                message: Text("Would you like to auto-load the last used model?"),
                primaryButton: .default(Text("Load")) {
                    if let path = llmManager.lastUsedModelPath {
                        llmManager.loadModel(at: URL(fileURLWithPath: path))
                    }
                },
                secondaryButton: .cancel(Text("No"))
            )
        }
    }
    
    // Custom Navigation Header View
    private var customHeaderView: some View {
        HStack(spacing: 16) {
            // Left: Sidebar Toggle
            Button(action: {
                withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                    showDrawer.toggle()
                }
            }) {
                Image(systemName: "line.horizontal.3")
                    .foregroundColor(Theme.accent)
                    .font(.system(size: 20, weight: .semibold))
                    .frame(width: 44, height: 44)
                    .contentShape(Rectangle())
            }
            
            // Center: Futuristic title
            Spacer()
            HStack(spacing: 4) {
                Text("DARK")
                    .font(.system(size: 20, weight: .black))
                    .foregroundColor(.white)
                Text("AI")
                    .font(.system(size: 20, weight: .black))
                    .foregroundColor(Theme.accent)
                    .neonGlow(color: Theme.accent, radius: 4)
            }
            .kerning(1.5)
            Spacer()
            
            // Right: Status indicator pill + Gear icon
            HStack(spacing: 12) {
                // Connection Status Pill
                HStack(spacing: 6) {
                    Circle()
                        .fill(isModelActive ? Color.green : Theme.accent)
                        .frame(width: 6, height: 6)
                        .scaleEffect(pulseActive ? 1.4 : 1.0)
                        .onAppear {
                            withAnimation(.easeInOut(duration: 1.2).repeatForever(autoreverses: true)) {
                                pulseActive = true
                            }
                        }
                    
                    Text(isModelActive ? "READY" : "OFFLINE")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundColor(isModelActive ? .green : Theme.textSecondary)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Theme.border.opacity(0.3))
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(isModelActive ? Color.green.opacity(0.3) : Theme.border, lineWidth: 1)
                        )
                )
                
                // Settings button
                Button(action: { showSettings = true }) {
                    Image(systemName: "gearshape.fill")
                        .foregroundColor(.white)
                        .font(.system(size: 18))
                        .frame(width: 32, height: 32)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 48) // Account for safe area
        .padding(.bottom, 12)
        .background(
            Color.black.opacity(0.9)
                .overlay(
                    Rectangle()
                        .frame(height: 1)
                        .foregroundColor(Theme.border),
                    alignment: .bottom
                )
        )
    }
    
    private var isModelActive: Bool {
        if case .loaded = llmManager.loadState {
            return true
        }
        return false
    }
    
    // Model detail banner
    @ViewBuilder
    private var modelBanner: some View {
        HStack {
            Image(systemName: "cpu.fill")
                .foregroundColor(Theme.accent)
            
            switch llmManager.loadState {
            case .unloaded:
                Text("Bypass active. Select a GGUF model in settings.")
                    .foregroundColor(Theme.textSecondary)
            case .loading(_, let status):
                Text(status)
                    .foregroundColor(Theme.textPrimary)
            case let .loaded(name, size):
                HStack(spacing: 8) {
                    Text("\(name) [\(String(format: "%.2f GB", size)) weights]")
                        .foregroundColor(Theme.textPrimary)
                    
                    if llmManager.isGenerating && llmManager.generationSpeed > 0 {
                        Text("\(String(format: "%.1f", llmManager.generationSpeed)) t/s")
                            .foregroundColor(Theme.accentCyan)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Theme.accentCyan.opacity(0.1))
                            .cornerRadius(4)
                    }
                }
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
            case .failed(let error):
                Text("Crash prevention: \(error)")
                    .foregroundColor(Theme.accent)
            }
            
            Spacer()
        }
        .font(.system(size: 12))
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Theme.background)
        .overlay(
            Rectangle()
                .frame(height: 1)
                .foregroundColor(Theme.border),
            alignment: .bottom
        )
    }
    
    // Welcome Instructions empty view
    @ViewBuilder
    private var emptyStateView: some View {
        VStack(spacing: 20) {
            Spacer()
            
            Image(systemName: "shield.fill")
                .font(.system(size: 60))
                .foregroundColor(Theme.accent)
                .neonGlow(color: Theme.accent, radius: 10)
            
            Text("LOCAL RUNNER v5.7.7")
                .font(.system(size: 15, weight: .bold))
                .foregroundColor(.white)
                .kerning(1.5)
            
            VStack(alignment: .leading, spacing: 10) {
                Text("• Local & offline text generation active.")
                Text("• RAG document storage active.")
                Text("• Memory safety limits validator active.")
            }
            .font(.system(size: 12))
            .foregroundColor(Theme.textSecondary)
            .padding()
            .glassCard(cornerRadius: 14)
            .frame(width: 290)
            
            Spacer()
        }
    }
    
    // Indicator pills
    @ViewBuilder
    private var activeParametersIndicator: some View {
        HStack(spacing: 12) {
            if conversationManager.stealthMode {
                HStack(spacing: 4) {
                    Image(systemName: "eye.slash.fill")
                    Text("STEALTH ACTIVE")
                }
                .font(.system(size: 9, weight: .bold))
                .foregroundColor(Theme.accent)
                .neonGlow(color: Theme.accent, radius: 4)
            } else {
                HStack(spacing: 4) {
                    Image(systemName: "tray.full.fill")
                    Text("LOGGING ACTIVE")
                }
                .font(.system(size: 9, weight: .bold))
                .foregroundColor(.green)
            }
            
            Spacer()
            
            if enableRAG {
                Text("RAG")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundColor(Theme.accentCyan)
            }
            if enableMemories {
                Text("MEMORIES")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundColor(Theme.accent)
            }
            Text("MAX: \(llmManager.maxTokens)T")
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .foregroundColor(Theme.textSecondary)
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 16)
        .background(Color.black.opacity(0.8))
        .overlay(
            Rectangle()
                .frame(height: 1)
                .foregroundColor(Theme.border),
            alignment: .top
        )
    }
    
    // MARK: - Output Filtering
    private func filterThoughts(from text: String, stripMarkdown: Bool = false) -> String {
        var filtered = text
        
        // --- 1. Strip XML-style thinking/correction/reflection tags ---
        // Covers Gemma, Qwen, DeepSeek, Llama, and other instruct model variants
        let xmlTags = [
            "think", "thinking", "thought", "thoughts",
            "channel thought", "channel thoughts", "channel>thought",
            "|channel>thought", "self-correction", "self_correction",
            "correction", "reflection", "reasoning", "internal"
        ]
        for tag in xmlTags {
            while let startRange = filtered.range(of: "<\(tag)", options: .caseInsensitive) {
                if let endRange = filtered.range(of: "</\(tag)>", options: .caseInsensitive) {
                    filtered.removeSubrange(startRange.lowerBound..<endRange.upperBound)
                } else {
                    filtered.removeSubrange(startRange.lowerBound..<filtered.endIndex)
                    break
                }
            }
        }
        
        // Strip pipe-delimited thinking tokens used by some models
        let pipeTokens = ["<|thinking|>", "<|/thinking|>", "<|thought|>", "<|/thought|>"]
        for token in pipeTokens {
            filtered = filtered.replacingOccurrences(of: token, with: "", options: .caseInsensitive)
        }
        
        // --- 2. Strip plaintext preamble headers ---
        let plaintextHeaders = ["Thinking Process:", "Thought Process:", "Internal Reasoning:", "Chain of Thought:"]
        for header in plaintextHeaders {
            while let startRange = filtered.range(of: header, options: .caseInsensitive) {
                if let endRange = filtered.range(of: "Response:", options: .caseInsensitive) {
                    filtered.removeSubrange(startRange.lowerBound..<endRange.upperBound)
                } else {
                    filtered.removeSubrange(startRange.lowerBound..<filtered.endIndex)
                    break
                }
            }
        }
        
        // --- 3. Strip exact known artifact strings ---
        let exactArtifacts = [
            "[Response generation]", "*self-correction/review*",
            "<|im_start|>", "<|im_end|>", "<|start_of_turn|>", "<|end_of_turn|>"
        ]
        for artifact in exactArtifacts {
            filtered = filtered.replacingOccurrences(of: artifact, with: "", options: .caseInsensitive)
        }
        
        // --- 4. Strip leading role-echo preamble (model parroting its own role prefix) ---
        let leadingPreambles = ["assistant:", "response:", "answer:", "a:"]
        var didTrimLeading = true
        while didTrimLeading {
            didTrimLeading = false
            let trimmedLower = filtered.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            for preamble in leadingPreambles {
                if trimmedLower.hasPrefix(preamble) {
                    filtered = String(filtered.trimmingCharacters(in: .whitespacesAndNewlines).dropFirst(preamble.count))
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    didTrimLeading = true
                    break
                }
            }
        }
        
        // --- 5. Strip personality system-prompt leak via regex ---
        if let regex = try? NSRegularExpression(
            pattern: "\\(?(?:Critical Instructions?|User Style Matrix|Communication Style Note)[\\s\\S]*?fr\\.?\\)?",
            options: [.caseInsensitive, .dotMatchesLineSeparators]
        ) {
            filtered = regex.stringByReplacingMatches(
                in: filtered,
                options: [],
                range: NSRange(location: 0, length: filtered.utf16.count),
                withTemplate: ""
            ).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        
        // --- 6. Markdown stripping (mature personality mode only, always preserves fenced code blocks) ---
        if stripMarkdown {
            var codeBlocks: [String] = []
            var protected = filtered
            if let codeBlockRegex = try? NSRegularExpression(pattern: "```[\\s\\S]*?```", options: []) {
                let matches = codeBlockRegex.matches(in: protected, range: NSRange(protected.startIndex..., in: protected)).reversed()
                for match in matches {
                    if let range = Range(match.range, in: protected) {
                        let block = String(protected[range])
                        let placeholder = "CODEBLOCK_\(codeBlocks.count)_PLACEHOLDER"
                        codeBlocks.append(block)
                        protected.replaceSubrange(range, with: placeholder)
                    }
                }
            }
            protected = protected.replacingOccurrences(of: "**", with: "")
            protected = protected.replacingOccurrences(of: "__", with: "")
            let mdLines = protected.components(separatedBy: "\n").map { line -> String in
                var l = line
                if l.hasPrefix("### ") { l = String(l.dropFirst(4)) }
                else if l.hasPrefix("## ") { l = String(l.dropFirst(3)) }
                else if l.hasPrefix("# ") { l = String(l.dropFirst(2)) }
                return l
            }
            protected = mdLines.joined(separator: "\n")
            for (i, block) in codeBlocks.enumerated() {
                protected = protected.replacingOccurrences(of: "CODEBLOCK_\(i)_PLACEHOLDER", with: block)
            }
            filtered = protected
        }
        
        // --- 7. Strip trailing role-echo stop tokens ---
        var finalFiltered = filtered.trimmingCharacters(in: .whitespacesAndNewlines)
        let trailingStops = ["user", "user:", "<|im_end|>", "<start_of_turn>user", "<|user|>", "<|eot_id|>"]
        var didTrimTrailing = true
        while didTrimTrailing {
            didTrimTrailing = false
            for stop in trailingStops {
                if finalFiltered.lowercased().hasSuffix(stop) {
                    finalFiltered.removeLast(stop.count)
                    finalFiltered = finalFiltered.trimmingCharacters(in: .whitespacesAndNewlines)
                    didTrimTrailing = true
                }
            }
        }
        
        return finalFiltered
    }

    private func messageBubble(for message: ChatMessage) -> some View {
        HStack(alignment: .top, spacing: 12) {
            if message.isUser {
                Spacer()
                Text(message.text)
                    .font(.system(size: 14))
                    .foregroundColor(.white)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: 14)
                            .fill(LinearGradient(colors: [Theme.accent, Theme.accentRose], startPoint: .topLeading, endPoint: .bottomTrailing))
                    )
                    .contextMenu {
                        Button {
                            UIPasteboard.general.string = message.text
                        } label: {
                            Label("Copy", systemImage: "doc.on.doc")
                        }
                    }
            } else if let imgData = message.imageData, let uiImg = UIImage(data: imgData) {
                // ── AI-Generated Image Bubble ───────────────────────────────
                Image(systemName: "sparkles")
                    .foregroundColor(Color.purple)
                    .font(.system(size: 12))
                    .padding(8)
                    .background(Theme.border.opacity(0.4))
                    .clipShape(Circle())

                VStack(alignment: .leading, spacing: 8) {
                    // Image
                    Image(uiImage: uiImg)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: 280)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .overlay(RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.purple.opacity(0.35), lineWidth: 1))
                        .contextMenu {
                            Button {
                                UIImageWriteToSavedPhotosAlbum(uiImg, nil, nil, nil)
                            } label: {
                                Label("Save to Photos", systemImage: "square.and.arrow.down")
                            }
                            Button {
                                UIPasteboard.general.image = uiImg
                            } label: {
                                Label("Copy Image", systemImage: "doc.on.doc")
                            }
                        }

                    // Always-visible action row
                    HStack(spacing: 10) {
                        // Save button
                        Button {
                            PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
                                if status == .authorized || status == .limited {
                                    UIImageWriteToSavedPhotosAlbum(uiImg, nil, nil, nil)
                                }
                            }
                        } label: {
                            Label("Save", systemImage: "square.and.arrow.down")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundColor(.white)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 5)
                                .background(
                                    RoundedRectangle(cornerRadius: 7)
                                        .fill(Color.purple.opacity(0.75))
                                )
                        }

                        // Copy button
                        Button {
                            UIPasteboard.general.image = uiImg
                        } label: {
                            Label("Copy", systemImage: "doc.on.doc")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundColor(Theme.textSecondary)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 5)
                                .background(
                                    RoundedRectangle(cornerRadius: 7)
                                        .fill(Theme.border.opacity(0.5))
                                )
                        }

                        // RAG badge
                        HStack(spacing: 3) {
                            Image(systemName: "brain")
                                .font(.system(size: 9))
                            Text("In RAG")
                                .font(.system(size: 10, weight: .medium))
                        }
                        .foregroundColor(Color.purple.opacity(0.6))
                    }

                    if !message.text.isEmpty {
                        Text(message.text)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(Theme.textSecondary)
                            .lineLimit(2)
                    }
                }
                Spacer()
                // ───────────────────────────────────────────────────────────
            } else if message.imageData == nil && !message.isUser && diffusionManager.isGenerating && conversationManager.activeConversation?.messages.last?.id == message.id {
                // ── In-progress image generation spinner ───────────────────
                Image(systemName: "sparkles")
                    .foregroundColor(Color.purple)
                    .font(.system(size: 12))
                    .padding(8)
                    .background(Theme.border.opacity(0.4))
                    .clipShape(Circle())

                VStack(alignment: .leading, spacing: 6) {
                    Text("Generating image... \(Int(diffusionManager.generationProgress * 100))%")
                        .font(.system(size: 13, design: .monospaced))
                        .foregroundColor(Color.purple.opacity(0.8))
                    
                    ProgressView(value: diffusionManager.generationProgress)
                        .progressViewStyle(LinearProgressViewStyle(tint: Color.purple))
                        .frame(width: 150)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(Theme.cardBackground)
                .cornerRadius(14)
                .overlay(RoundedRectangle(cornerRadius: 14)
                    .stroke(Color.purple.opacity(0.3), lineWidth: 1))
                Spacer()
                // ───────────────────────────────────────────────────────────
            } else {
                // ── Standard text bubble ────────────────────────────────────
                Image(systemName: "terminal.fill")
                    .foregroundColor(Theme.accentCyan)
                    .font(.system(size: 12))
                    .padding(8)
                    .background(Theme.border.opacity(0.4))
                    .clipShape(Circle())

                let filteredText = filterThoughts(from: message.text, stripMarkdown: personalityManager.isMature)
                let isThinking = filteredText.isEmpty && llmManager.isGenerating

                Text(isThinking ? "Thinking..." : filteredText)
                    .font(.system(size: 14, design: .monospaced))
                    .foregroundColor(isThinking ? Theme.textSecondary : Theme.textPrimary)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(Theme.cardBackground)
                    .cornerRadius(14)
                    .overlay(
                        RoundedRectangle(cornerRadius: 14)
                            .stroke(Theme.border, lineWidth: 1)
                    )
                    .contextMenu {
                        Button {
                            UIPasteboard.general.string = isThinking ? message.text : filteredText
                        } label: {
                            Label("Copy", systemImage: "doc.on.doc")
                        }
                    }
                Spacer()
                // ───────────────────────────────────────────────────────────
            }
        }
    }
    
    // Input Area View
    @ViewBuilder
    private var inputArea: some View {
        VStack(alignment: .leading, spacing: 8) {
            
            // Pending Attachment Pill
            if let attName = pendingAttachmentName {
                HStack {
                    Image(systemName: "doc.text.image")
                        .foregroundColor(Theme.accentCyan)
                    Text(attName)
                        .font(.system(size: 12, weight: .bold))
                        .foregroundColor(.white)
                        .lineLimit(1)
                    Spacer()
                    Button(action: {
                        pendingAttachmentName = nil
                        pendingAttachmentText = nil
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.gray)
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Theme.cardBackground)
                .cornerRadius(12)
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(Theme.border, lineWidth: 1))
            }
            
            HStack(spacing: 10) {
            
            // Universal OCR / Vision Upload Button (Always available for all models via iOS Native Vision)
            Button(action: { showFileImporter = true }) {
                Image(systemName: "paperclip")
                    .foregroundColor(Theme.accentCyan)
                    .font(.system(size: 18))
                    .frame(width: 40, height: 40)
                    .background(Theme.accentCyan.opacity(0.15))
                    .clipShape(Circle())
                    .overlay(Circle().stroke(Theme.accentCyan.opacity(0.4), lineWidth: 1))
            }
            
            // Stealth Mode Toggle
            Button(action: {
                conversationManager.stealthMode.toggle()
            }) {
                Image(systemName: conversationManager.stealthMode ? "eye.slash.fill" : "eye.fill")
                    .foregroundColor(conversationManager.stealthMode ? Theme.accent : Theme.textSecondary)
                    .font(.system(size: 18))
                    .frame(width: 40, height: 40)
                    .background(Theme.border.opacity(0.3))
                    .clipShape(Circle())
                    .overlay(
                        Circle()
                            .stroke(conversationManager.stealthMode ? Theme.accent : Color.clear, lineWidth: 1.5)
                    )
            }
            
            // Text Entry
            TextField(isModelActive ? "Execute prompt..." : "Model unloaded...", text: $inputText, axis: .vertical)
                .lineLimit(1...8)
                .font(.system(size: 14))
                .foregroundColor(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(Theme.background)
                .cornerRadius(20)
                .overlay(
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(Theme.border, lineWidth: 1.5)
                )
                .disabled(!isModelActive)
            
            // Send / Stop button
            Button(action: sendMessage) {
                Image(systemName: llmManager.isGenerating ? "stop.fill" : "arrow.up")
                    .foregroundColor(.white)
                    .font(.system(size: 14, weight: .bold))
                    .padding(10)
                    .background(
                        Circle()
                            .fill(isModelActive ? Theme.accent : Theme.border)
                    )
                    .neonGlow(color: isModelActive ? Theme.accent : .clear, radius: 4)
            }
            .disabled(!isModelActive && !llmManager.isGenerating)
            }
        }
        .padding()
        .background(Color.black.opacity(0.9))
        .overlay(
            Rectangle()
                .frame(height: 1)
                .foregroundColor(Theme.border),
            alignment: .top
        )
    }
    
    // Left-side Collapsible Sidebar Drawer
    private var sidebarDrawer: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 20) {
                
                // Drawer Header
                HStack {
                    Text("CHATS")
                        .font(.system(size: 13, weight: .black))
                        .foregroundColor(.white)
                        .kerning(2.0)
                    
                    Spacer()

                    // New chat compact plus button
                    Button(action: {
                        conversationManager.createConversation()
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                            showDrawer = false
                        }
                    }) {
                        Image(systemName: "plus")
                            .foregroundColor(Theme.accent)
                            .font(.system(size: 16, weight: .bold))
                    }
                    
                    Button(action: {
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                            showDrawer = false
                        }
                    }) {
                        Image(systemName: "xmark")
                            .foregroundColor(Theme.textSecondary)
                            .font(.system(size: 14, weight: .bold))
                    }
                }
                .padding(.top, 56)
                .padding(.horizontal)
                
                
                // Scrollable Chat log list
                ScrollView {
                    VStack(spacing: 12) {
                        ForEach(conversationManager.conversations) { conversation in
                            HStack {
                                // Conversation Switcher Button
                                Button(action: {
                                    conversationManager.selectConversation(id: conversation.id)
                                    withAnimation(.spring()) {
                                        showDrawer = false
                                    }
                                }) {
                                    HStack(spacing: 10) {
                                        Image(systemName: "chevron.right.square.fill")
                                            .foregroundColor(conversationManager.activeConversationId == conversation.id ? Theme.accent : Theme.textMuted)
                                        Text(conversation.title)
                                            .font(.system(size: 13, weight: .semibold))
                                            .foregroundColor(conversationManager.activeConversationId == conversation.id ? .white : Theme.textSecondary)
                                            .lineLimit(1)
                                    }
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                }
                                
                                // Delete conversation
                                Button(action: {
                                    conversationManager.deleteConversation(id: conversation.id)
                                }) {
                                    Image(systemName: "trash")
                                        .foregroundColor(Theme.textMuted)
                                        .font(.system(size: 12))
                                }
                            }
                            .padding(10)
                            .background(
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(conversationManager.activeConversationId == conversation.id ? Theme.border.opacity(0.3) : Color.clear)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(conversationManager.activeConversationId == conversation.id ? Theme.border : Color.clear, lineWidth: 1)
                            )
                        }
                    }
                    .padding(.horizontal)
                }
                
                Spacer()
                
                // Drawer Footer status
                VStack(alignment: .leading, spacing: 6) {
                    Text("DarkAI Local OS bypass")
                        .font(.system(size: 10))
                        .foregroundColor(Theme.textMuted)
                }
                .padding(.horizontal)
                .padding(.bottom, 24)
            }
            .frame(width: 270)
            .background(Color(hex: "06060c"))
            .overlay(
                Rectangle()
                    .frame(width: 1)
                    .foregroundColor(Theme.border),
                alignment: .trailing
            )
            .offset(x: showDrawer ? 0 : -270)
            
            Spacer()
        }
        .ignoresSafeArea(.container, edges: [.leading, .trailing])
    }
    
    private func sendMessage() {
        // Cancel in-progress text generation
        if llmManager.isGenerating {
            llmManager.cancelGeneration()
            return
        }

        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty || pendingAttachmentText != nil else { return }

        inputText = ""

        // ── Prompt Intent Classification ──────────────────────────────────────
        // Check whether the user is requesting image generation BEFORE attaching
        // files so that file context doesn't accidentally override intent.
        let intent = PromptClassifier.classify(text)
        if case .imageGeneration(let refinedPrompt) = intent, pendingAttachmentText == nil {
            // Add the user message bubble
            conversationManager.addMessageToActive(isUser: true, text: text)

            // Guard: a diffusion model must be selected
            guard let diffPath = diffusionManager.lastDiffusionModelPath else {
                // Show banner and do NOT send to the LLM
                diffusionBannerTask?.cancel()
                withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                    showDiffusionNotLoadedBanner = true
                }
                diffusionBannerTask = Task {
                    try? await Task.sleep(nanoseconds: 5_000_000_000)
                    await MainActor.run {
                        withAnimation { showDiffusionNotLoadedBanner = false }
                    }
                }
                return
            }

            // Add a placeholder bubble (shows the spinner while generating)
            conversationManager.addMessageToActive(isUser: false, text: refinedPrompt)

            // SET UI STATE IMMEDIATELY so the progress bar shows while loading
            diffusionManager.isGenerating = true
            diffusionManager.generationProgress = 0.0

            Task {
                let savedLLMUrl = llmManager.activeModelURL
                await llmManager.unloadModelAsync()
                
                // Allow the OS time to flush Metal memory buffers released by llama.cpp
                // before we attempt to map SDXL's massive weights into RAM.
                try? await Task.sleep(nanoseconds: 1_500_000_000)

                do {
                    // Load the diffusion model — suspends here (MainActor is free during this)
                    try await diffusionManager.loadDiffusionModelAsync(at: URL(fileURLWithPath: diffPath))

                    // Run image generation — each `await` inside releases the MainActor, so
                    // the UI stays responsive throughout the multi-minute denoising loop.
                    let imageData = await diffusionManager.generateImageAsync(prompt: refinedPrompt)

                    // Cleanup: unload diffusion model then reload the LLM
                    await diffusionManager.unloadDiffusionModelAsync()
                    if let llm = savedLLMUrl {
                        llmManager.loadModel(at: llm)
                    }

                    // Update the chat bubble with the result
                    if let data = imageData {
                        conversationManager.updateLastMessageImage(imageData: data)
                        conversationManager.saveConversations()

                        // ── Auto-ingest to RAG ────────────────────────────────
                        ragManager.ingestGeneratedImage(prompt: refinedPrompt, imageData: data)

                    } else {
                        conversationManager.updateLastMessage(text: "[Image generation failed. Check the diffusion model and try again.]")
                        conversationManager.saveConversations()
                    }
                } catch {
                    conversationManager.updateLastMessage(text: "[Failed to load diffusion model: \(error.localizedDescription)]")
                    conversationManager.saveConversations()
                    diffusionManager.isGenerating = false

                    if let llm = savedLLMUrl {
                        llmManager.loadModel(at: llm)
                    }
                }
            }
            return
        }
        // ─────────────────────────────────────────────────────────────────────

        let history = conversationManager.activeConversation?.messages ?? []

        var promptText = text
        if let attachmentName = pendingAttachmentName, let attachmentText = pendingAttachmentText {
            let fileInfo = "\n\n[ATTACHED FILE: \(attachmentName)]\n\(attachmentText)\n[/ATTACHED FILE]"
            if text.isEmpty {
                promptText = "Please analyze the attached file." + fileInfo
            } else {
                promptText = text + fileInfo
            }

            conversationManager.addMessageToActive(isUser: true, text: text.isEmpty ? "[Sent Attachment: \(attachmentName)]" : text + "\n[Sent Attachment: \(attachmentName)]")

            pendingAttachmentName = nil
            pendingAttachmentText = nil
        } else {
            conversationManager.addMessageToActive(isUser: true, text: text)
        }

        if enableMemories && !text.isEmpty {
            memoryManager.extractMemories(from: text)
            if let activeModel = llmManager.activeModelURL?.lastPathComponent {
                personalityManager.analyzeUserMessage(text, modelName: activeModel, llmManager: llmManager)
            }
        }

        let ragContext = enableRAG ? ragManager.retrieveRelevantContext(query: text) : ""
        let memoriesContext = enableMemories ? memoryManager.getFormattedMemoriesForContext() : ""

        conversationManager.addMessageToActive(isUser: false, text: "")

        var finalSystemPrompt = customInstructions
        if let activeModel = llmManager.activeModelURL?.lastPathComponent {
            let personality = personalityManager.getPersonality(for: activeModel)
            if !personality.isEmpty {
                let score = personalityManager.maturityScore
                if score < 0.4 {
                    finalSystemPrompt += "\n\n[Communication Style Note — adapt naturally to user's style]:\n" + personality
                } else if score < 0.7 {
                    finalSystemPrompt += "\n\n" + personality
                } else {
                    let identityAnchor = customInstructions
                        .components(separatedBy: CharacterSet(charactersIn: ".!?\n"))
                        .first?
                        .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                    if identityAnchor.isEmpty {
                        finalSystemPrompt = personality
                    } else {
                        finalSystemPrompt = identityAnchor + ".\n\n" + personality
                    }
                }
            }
        }

        llmManager.generateResponse(
            prompt: promptText,
            history: history,
            systemPrompt: finalSystemPrompt,
            memoriesContext: memoriesContext,
            ragContext: ragContext
        ) { token in
            conversationManager.updateLastMessage(text: (conversationManager.activeConversation?.messages.last?.text ?? "") + token)
        } onComplete: { finalText in
            var cleanedText = self.filterThoughts(from: finalText, stripMarkdown: self.personalityManager.isMature)
            if cleanedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                cleanedText = "[Context exhausted during reasoning. Try a shorter prompt or reduce the context window in Settings.]"
            }
            conversationManager.updateLastMessage(text: cleanedText)
            conversationManager.saveConversations()
        }
    }
    
    // MARK: - File Import Handler

    private func handleFileImport(_ result: Result<[URL], Error>) {
        guard let url = try? result.get().first else { return }

        Task {
            guard url.startAccessingSecurityScopedResource() else { return }
            defer { url.stopAccessingSecurityScopedResource() }

            do {
                let extractedText = try await DocumentProcessor.extractText(from: url)
                let fileName = url.lastPathComponent
                let ext = url.pathExtension.lowercased()
                let isImage = ["jpg", "jpeg", "png", "gif", "heic"].contains(ext)

                await MainActor.run {
                    // Ingest into RAG
                    ragManager.ingestDocument(name: fileName, content: extractedText)

                    // Show upload notification in chat
                    let uploadNote = isImage
                        ? "[Image uploaded: \(fileName) — \(extractedText.count) characters extracted via OCR]"
                        : "[Document uploaded: \(fileName) — \(extractedText.count) characters extracted]"
                    conversationManager.addMessageToActive(isUser: true, text: uploadNote)

                    // Create empty assistant message to stream into
                    conversationManager.addMessageToActive(isUser: false, text: "")
                }

                // Build a description prompt — the model reads the extracted text and describes it
                let truncatedText = String(extractedText.prefix(1200)) // Limit context to avoid overflow
                let describePrompt: String
                if ["jpg", "jpeg", "png", "gif", "heic"].contains(ext) {
                    describePrompt = "The following text was extracted from an uploaded image using OCR. Based on this text, describe what this image appears to be. Start your response with 'I see you have uploaded an image that looks like...' then describe it. OCR content:\n\n\(truncatedText)"
                } else {
                    describePrompt = "The following text was extracted from an uploaded document called '\(url.lastPathComponent)'. Based on this content, identify what kind of document this is (e.g. receipt, resume, code, article, invoice, etc.) and summarize it briefly. Start your response with 'I see you uploaded a document that looks like...' then describe it. Document content:\n\n\(truncatedText)"
                }

                let history = await MainActor.run { conversationManager.activeConversation?.messages ?? [] }
                let memoriesContext = await MainActor.run { enableMemories ? memoryManager.getFormattedMemoriesForContext() : "" }

                await MainActor.run {
                    llmManager.generateResponse(
                        prompt: describePrompt,
                        history: history,
                        systemPrompt: customInstructions,
                        memoriesContext: memoriesContext,
                        ragContext: ""
                    ) { token in
                        conversationManager.updateLastMessage(
                            text: (conversationManager.activeConversation?.messages.last?.text ?? "") + token
                        )
                    } onComplete: { finalText in
                        var cleanedText = self.filterThoughts(from: finalText, stripMarkdown: self.personalityManager.isMature)
                        if cleanedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            cleanedText = "[Context exhausted. Could not produce a description.]"
                        }
                        conversationManager.updateLastMessage(text: cleanedText)
                        conversationManager.saveConversations()
                    }
                }

            } catch {
                await MainActor.run {
                    conversationManager.addMessageToActive(isUser: false, text: "[System] Failed to process file: \(error.localizedDescription)")
                }
            }
        }
    }
}

extension View {
    func endEditing() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}
