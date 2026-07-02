import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var llmManager = LLMManager()
    @StateObject private var memoryManager = MemoryManager()
    @StateObject private var ragManager = RAGManager()
    @StateObject private var conversationManager = ConversationManager()
    @StateObject private var personalityManager = PersonalityManager()
    
    @AppStorage("customInstructions") private var customInstructions: String = "You are DarkAI, a local assistant. Respond with precise, helpful answers."
    @State private var enableRAG = true
    @State private var enableMemories = true
    
    @State private var showFileImporter = false
    @State private var showAutoLoadAlert = false
    @State private var pendingAttachmentName: String? = nil
    @State private var pendingAttachmentText: String? = nil
    
    @State private var inputText: String = ""
    @State private var showSettings = false
    @State private var showDrawer = false
    
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
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(
                llmManager: llmManager,
                memoryManager: memoryManager,
                ragManager: ragManager,
                personalityManager: personalityManager,
                customInstructions: $customInstructions,
                enableRAG: $enableRAG,
                enableMemories: $enableMemories
            )
        }
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [
                .image, .pdf, .plainText, .rtf,
                UTType(filenameExtension: "doc")!,
                UTType(filenameExtension: "docx")!
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
            
            Text("LOCAL RUNNER v5.7.3")
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
    
    // Bubble UI
    @ViewBuilder
    private func filterThoughts(from text: String) -> String {
        var filtered = text
        let tags = ["channel thoughts", "think", "thought", "thinking", "|channel>thought", "channel>thought", "self-correction", "self_correction", "correction"]
        for tag in tags {
            while let startRange = filtered.range(of: "<\(tag)", options: .caseInsensitive) {
                if let endRange = filtered.range(of: "</\(tag)>", options: .caseInsensitive) {
                    filtered.removeSubrange(startRange.lowerBound..<endRange.upperBound)
                } else {
                    filtered.removeSubrange(startRange.lowerBound..<filtered.endIndex)
                    break
                }
            }
        }
        
        let plaintextTags = ["Thinking Process:", "Thought Process:"]
        for pt in plaintextTags {
            while let startRange = filtered.range(of: pt, options: .caseInsensitive) {
                if let endRange = filtered.range(of: "Response:", options: .caseInsensitive) {
                    filtered.removeSubrange(startRange.lowerBound..<endRange.upperBound)
                } else {
                    filtered.removeSubrange(startRange.lowerBound..<filtered.endIndex)
                    break
                }
            }
        }
        
        return filtered.trimmingCharacters(in: .whitespacesAndNewlines)
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
            } else {
                Image(systemName: "terminal.fill")
                    .foregroundColor(Theme.accentCyan)
                    .font(.system(size: 12))
                    .padding(8)
                    .background(Theme.border.opacity(0.4))
                    .clipShape(Circle())
                
                let filteredText = filterThoughts(from: message.text)
                let isThinking = filteredText.isEmpty && llmManager.isGenerating
                
                Text(isThinking ? "Thinking..." : (filteredText.isEmpty ? "..." : filteredText))
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
                Spacer()
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
        if llmManager.isGenerating {
            llmManager.cancelGeneration()
            return
        }
        
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty || pendingAttachmentText != nil else { return }
        
        inputText = ""
        
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
                personalityManager.analyzeUserMessage(text, modelName: activeModel)
            }
        }
        
        let ragContext = enableRAG ? ragManager.retrieveRelevantContext(query: text) : ""
        let memoriesContext = enableMemories ? memoryManager.getFormattedMemoriesForContext() : ""
        
        conversationManager.addMessageToActive(isUser: false, text: "")
        
        var finalSystemPrompt = customInstructions
        if let activeModel = llmManager.activeModelURL?.lastPathComponent {
            let personality = personalityManager.getPersonality(for: activeModel)
            if !personality.isEmpty {
                finalSystemPrompt += "\n\nYour Unique Personality: " + personality
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
            conversationManager.updateLastMessage(text: finalText)
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
                let ext2 = url.pathExtension.lowercased()
                let describePrompt: String
                if ["jpg", "jpeg", "png", "gif", "heic"].contains(ext2) {
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
                        conversationManager.updateLastMessage(text: finalText)
                    }
                }

            } catch {
                await MainActor.run {
                    conversationManager.addMessageToActive(isUser: false, text: "[System] Failed to process file: \(error.localizedDescription)")
                }
            }
        }
    }
    
    private func endEditing() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}

extension View {
    func endEditing() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}
