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
    @StateObject private var webSearchManager = WebSearchManager()

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

    // Content safety + one-off notices (blocked prompts, Photos permission, save results)
    @State private var notice: Notice? = nil
    @State private var reportTarget: ChatMessage? = nil

    // Crash recovered from the previous run, offered once on launch
    /// Crash recovered from the previous run, shown as a dismissible banner.
    @State private var recoveredCrash: CrashReporter.Report? = nil
    /// Set only when the user taps through from that banner.
    @State private var crashToInspect: CrashReporter.Report? = nil

    struct Notice: Identifiable {
        let id = UUID()
        let title: String
        let message: String
    }

    // Pulse animation state
    @State private var pulseActive = false
    
    var body: some View {
        ZStack {
            // MAIN VIEW
            VStack(spacing: 0) {
                
                // Custom Premium Navigation Header
                customHeaderView

                // Recovered-crash notice, if the previous run ended badly
                crashRecoveryBanner

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
                        .onChange(of: activeConv.messages) {
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
                webSearchManager: webSearchManager,
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
            
            // A recovered crash is surfaced as a banner, never as a modal at launch.
            //
            // `takePendingReport()` consumes the file immediately, so even if the banner or the
            // detail sheet fails to render, the next launch starts clean instead of replaying
            // the same failure forever. The report is held in memory for this session and is
            // still reachable from Settings → Safety & Legal.
            recoveredCrash = CrashReporter.takePendingReport()

            if let queued = OnboardingHandoff.takeAutoLoadURL(), llmManager.activeModelURL == nil {
                // Straight to loading, no prompt: this model was chosen moments ago during setup.
                // The delay is the same one the manual auto-load path uses — competing with the
                // app's own launch work for memory was a reliable way to fail a load that would
                // succeed fine a second later.
                Task {
                    try? await Task.sleep(nanoseconds: 1_500_000_000)
                    llmManager.loadModel(at: queued)
                }
            } else if recoveredCrash == nil,
                      llmManager.lastUsedModelPath != nil,
                      llmManager.activeModelURL == nil {
                showAutoLoadAlert = true
            }
        }
        .sheet(item: $reportTarget) { message in
            ReportContentView(
                content: message.imageData != nil ? "" : message.text,
                modelName: llmManager.activeModelURL?.lastPathComponent ?? "none",
                wasImage: message.imageData != nil
            ) {
                conversationManager.deleteMessage(id: message.id)
            }
        }
        .alert(
            notice?.title ?? "",
            isPresented: Binding(get: { notice != nil }, set: { if !$0 { notice = nil } })
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(notice?.message ?? "")
        }
        // Uses the iOS 15+ alert API rather than the older `Alert`-returning form so it can
        // coexist with the safety and report presentations above — mixing the two generations
        // of alert modifier on one view lets the last one silently win.
        .alert("Load Previous Model?", isPresented: $showAutoLoadAlert) {
            Button("Load") {
                if let path = llmManager.lastUsedModelPath {
                    // Give the app's own launch sequence (SwiftUI setup, asset decoding, etc.)
                    // a moment to settle before competing with it for memory — loading
                    // immediately on first launch was a common cause of an avoidable
                    // out-of-memory failure that wouldn't reproduce moments later loading the
                    // identical model from Settings.
                    Task {
                        try? await Task.sleep(nanoseconds: 1_500_000_000)
                        llmManager.loadModel(at: URL(fileURLWithPath: path))
                    }
                }
            }
            Button("Not Now", role: .cancel) {}
        } message: {
            Text("Would you like to load the model you used last?")
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
                    .foregroundColor(Theme.textPrimary)
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
                        .foregroundColor(Theme.textPrimary)
                        .font(.system(size: 18))
                        .frame(width: 32, height: 32)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 48) // Account for safe area
        .padding(.bottom, 12)
        .background(
            Theme.chrome
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
    
    /// Non-modal replacement for the launch-time crash dialog. Inline, dismissible, and it
    /// cannot block the app if anything about it fails to draw.
    @ViewBuilder
    private var crashRecoveryBanner: some View {
        if let crash = recoveredCrash {
            HStack(spacing: 10) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.orange)
                    .font(.system(size: 14))

                VStack(alignment: .leading, spacing: 2) {
                    Text("\(AppInfo.displayName) closed unexpectedly last time")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundColor(Theme.textPrimary)
                    Text(crash.reason)
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textSecondary)
                        .lineLimit(2)
                }

                Spacer(minLength: 4)

                Button("View") { crashToInspect = crash }
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color.orange.opacity(0.85)))

                Button {
                    withAnimation { recoveredCrash = nil }
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(Theme.textSecondary)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Color.orange.opacity(0.12))
            .overlay(
                Rectangle().frame(height: 1).foregroundColor(Color.orange.opacity(0.4)),
                alignment: .bottom
            )
            // Attached here rather than to the root ZStack so it isn't competing with the
            // settings and content-report sheets during launch layout.
            .sheet(item: $crashToInspect) { report in
                CrashReportView(report: report) { }
            }
        }
    }

    /// Either actively generating, or unwinding a cancelled run that hasn't released the
    /// checkpoint yet. Both are states where the bar should describe the image pipeline rather
    /// than the chat model, which is unloaded throughout.
    private var diffusionBusy: Bool {
        diffusionManager.isGenerating || diffusionManager.isFinishingCancelledRun
    }

    /// Diffusion checkpoint driving the current generation.
    ///
    /// Prefers the name the loader reported, since that is the checkpoint actually resident.
    /// Falls back to the selected file's name for the stretch before the load completes — the
    /// banner needs something to say from the moment generation starts, not just once the
    /// weights are up.
    private var activeDiffusionModelName: String? {
        if case let .loaded(name, _) = diffusionManager.diffusionLoadState {
            return name
        }
        if let url = diffusionManager.activeDiffusionURL {
            return url.lastPathComponent
        }
        if let path = diffusionManager.lastDiffusionModelPath {
            return URL(fileURLWithPath: path).lastPathComponent
        }
        return nil
    }

    // Model detail banner
    @ViewBuilder
    private var modelBanner: some View {
        HStack {
            Image(systemName: diffusionBusy ? "photo.fill" : "cpu.fill")
                .foregroundColor(diffusionBusy ? .purple : Theme.accent)

            // Image generation evicts the chat model to make room, so `llmManager.loadState`
            // reads `.unloaded` for the whole operation — the bar used to say "No model loaded.
            // Choose one in Settings to start." while the app was visibly busy generating.
            // Show what is actually running instead, and let it fall back to the LLM state on
            // its own once the session ends and the chat model is reloaded.
            if diffusionManager.isFinishingCancelledRun {
                // The sampler can't be interrupted, so this covers the gap between the user
                // cancelling and the run unwinding far enough to put the chat model back.
                Text("Cancelling — the chat model will reload when the current step finishes.")
                    .foregroundColor(Theme.textSecondary)
                    .lineLimit(2)
            } else if diffusionManager.isGenerating {
                HStack(spacing: 8) {
                    Text("Generating with \(activeDiffusionModelName ?? "diffusion model")")
                        .foregroundColor(Theme.textPrimary)
                        .lineLimit(1)
                        .truncationMode(.middle)

                    if diffusionManager.generationProgress > 0 {
                        Text("\(Int(diffusionManager.generationProgress * 100))%")
                            .foregroundColor(.purple)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.purple.opacity(0.15))
                            .cornerRadius(4)
                    }
                }
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
            } else {
            switch llmManager.loadState {
            case .unloaded:
                Text("No model loaded. Choose one in Settings to start.")
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
                Text(error)
                    .foregroundColor(Theme.accent)
                    .lineLimit(2)
            }
            }

            Spacer()
        }
        .font(.system(size: 12))
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(diffusionBusy ? Color.purple.opacity(0.10) : Theme.background)
        .animation(.easeInOut(duration: 0.25), value: diffusionBusy)
        .overlay(
            Rectangle()
                .frame(height: 1)
                .foregroundColor(Theme.border),
            alignment: .bottom
        )
    }
    
    // Welcome / empty conversation view
    @ViewBuilder
    private var emptyStateView: some View {
        VStack(spacing: 18) {
            Spacer()

            welcomePanel

            Spacer()
        }
        .padding(.horizontal, 20)
    }

    /// The welcome copy and its shade panel. Split out from `emptyStateView` so the panel wraps
    /// only the content — wrapping the enclosing `VStack` would stretch it across the spacers
    /// and shade the entire screen, which is the thing this is meant to avoid.
    @ViewBuilder
    private var welcomePanel: some View {
        VStack(spacing: 18) {
            Image(systemName: isModelActive ? "bubble.left.and.bubble.right.fill" : "square.and.arrow.down.fill")
                .font(.system(size: 52))
                .foregroundColor(Theme.accent)
                .neonGlow(color: Theme.accent, radius: 10)

            Text(isModelActive ? "Ask me anything" : "Add a model to begin")
                .font(.system(size: 19, weight: .bold))
                .foregroundColor(Theme.textPrimary)

            Text(isModelActive
                 ? "Everything runs on this device. Try a question, or describe a picture to generate one."
                 : "\(AppInfo.displayName) needs a model to run. Open Settings to download one — it only takes a minute.")
                .font(.system(size: 13))
                .foregroundColor(Theme.textSecondary)
                .multilineTextAlignment(.center)
                .lineSpacing(3)
                .frame(maxWidth: 300)

            if !isModelActive {
                Button {
                    showSettings = true
                } label: {
                    Text("Open Settings")
                        .font(.system(size: 14, weight: .bold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 26)
                        .padding(.vertical, 11)
                        .background(RoundedRectangle(cornerRadius: 12).fill(Theme.accent))
                }
            }

            // Sets expectations before the first response rather than after — a reviewer
            // opening a fresh install sees this, and so does a user about to trust an answer.
            Text("Responses are generated by a model on this device and can be inaccurate. Check anything important.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textMuted)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 290)
                .padding(.top, 6)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 26)
        // Shade behind the copy rather than over the whole screen, so the backdrop stays part
        // of the design while the text sitting on it keeps its contrast.
        .readablePanel(cornerRadius: 28)
    }
    
    // Indicator pills
    @ViewBuilder
    private var activeParametersIndicator: some View {
        HStack(spacing: 12) {
            if conversationManager.privateMode {
                HStack(spacing: 4) {
                    Image(systemName: "eye.slash.fill")
                    Text("PRIVATE — NOT SAVED")
                }
                .font(.system(size: 9, weight: .bold))
                .foregroundColor(Theme.accent)
                .neonGlow(color: Theme.accent, radius: 4)
            } else {
                HStack(spacing: 4) {
                    Image(systemName: "tray.full.fill")
                    Text("SAVED ON DEVICE")
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

            // Shown whenever the oldest conversation history (or, mid-generation, the
            // oldest tokens in the live KV cache) had to be dropped to fit the context
            // window — the conversation keeps going, this just makes it visible when it's
            // happening rather than a silent, confusing drop in what the model can recall.
            if llmManager.isContextTruncating {
                HStack(spacing: 3) {
                    Image(systemName: "arrow.trianglehead.2.clockwise")
                    Text("TRUNCATING")
                }
                .font(.system(size: 9, weight: .bold))
                .foregroundColor(.orange)
            }

            // Live context-window usage vs. the limit actually applied to the loaded
            // model (which can be lower than the requested setting once clamped to
            // available RAM). Turns orange near the limit as an early warning.
            let ctxLimit = llmManager.loadedContextWindow > 0 ? llmManager.loadedContextWindow : llmManager.contextTokenLimit
            if ctxLimit > 0 {
                let ctxFraction = Double(llmManager.contextTokensUsed) / Double(ctxLimit)
                Text("CTX: \(llmManager.contextTokensUsed)/\(ctxLimit)T")
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundColor(ctxFraction >= 0.9 ? .orange : Theme.textSecondary)
            }

            // Shows live generated-token progress against the cap while streaming, so the
            // max-response-size setting is directly observable being enforced in the UI.
            if llmManager.isGenerating {
                Text("GEN: \(llmManager.currentResponseTokenCount)/\(llmManager.maxTokens)T")
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundColor(Theme.accentCyan)
            } else {
                Text("MAX: \(llmManager.maxTokens)T")
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundColor(Theme.textSecondary)
            }
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 16)
        .background(Theme.chrome)
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

        // --- 0. Strip format-agnostic reasoning/channel blocks ---
        // Some models (gpt-oss, certain Gemma fine-tunes) use a "channel" convention with
        // no matching close tag at all — e.g. <|channel|>analysis<|message|> ... reasoning ...
        // <|channel|>final<|message|> — where analysis only ends when a new non-reasoning
        // channel begins. The fixed open/close tag pairs in step 1 below can't express that,
        // so this keys off vocabulary inside any bracketed tag (<tag>, </tag>, <|tag|>) and
        // exits on whichever comes first: a real close tag, or a transition to a
        // final/message/response/answer channel.
        let reasoningOpenPattern = "<\\|?\\s*(think|thought|thinking|reflect|reason|channel|analysis|internal|scratchpad|deliberat)"
        let reasoningClosePattern = "</\\s*(think|thought|thinking|reflect|reflection|reason|reasoning|channel|analysis|internal|scratchpad|deliberation)\\s*>"
        let transitionPattern = "<\\|?/?\\s*(final|message|response|answer)[a-z]*\\s*\\|?>"
        if let openRegex = try? NSRegularExpression(pattern: reasoningOpenPattern, options: [.caseInsensitive]),
           let closeRegex = try? NSRegularExpression(pattern: reasoningClosePattern, options: [.caseInsensitive]),
           let transitionRegex = try? NSRegularExpression(pattern: transitionPattern, options: [.caseInsensitive]) {
            while let openMatch = openRegex.firstMatch(in: filtered, range: NSRange(filtered.startIndex..., in: filtered)),
                  let openRange = Range(openMatch.range, in: filtered) {
                let searchRange = NSRange(openRange.upperBound..., in: filtered)
                let closeEnd = closeRegex.firstMatch(in: filtered, range: searchRange).flatMap { Range($0.range, in: filtered)?.upperBound }
                let transEnd = transitionRegex.firstMatch(in: filtered, range: searchRange).flatMap { Range($0.range, in: filtered)?.upperBound }

                let end: String.Index?
                switch (closeEnd, transEnd) {
                case let (c?, t?): end = min(c, t)
                case let (c?, nil): end = c
                case let (nil, t?): end = t
                default: end = nil
                }

                if let end {
                    filtered.removeSubrange(openRange.lowerBound..<end)
                } else {
                    filtered.removeSubrange(openRange.lowerBound..<filtered.endIndex)
                    break
                }
            }
        }

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
            "<|im_start|>", "<|im_end|>", "<|start_of_turn|>", "<|end_of_turn|>"
        ]
        for artifact in exactArtifacts {
            filtered = filtered.replacingOccurrences(of: artifact, with: "", options: .caseInsensitive)
        }

        // --- 3b. Strip bare (untagged) scaffolding/preamble blocks ---
        // Some fine-tunes open every reply with meta-commentary about how they plan to
        // respond — style/tone "checks," headers like "My internal monologue:", decorative
        // "***" separators, or literal placeholder text like "(Generating response...)" —
        // with no tag structure at all, and wording that varies message to message. Match
        // the recurring *shapes* instead of specific wording: a slash-command header, a
        // decorative "***" line, any bold/italic header ending in a colon, or known
        // self-referential/stage-direction phrases in other wrappers. These have no
        // reliable close tag, so the next blank line is treated as the block's end —
        // scoped to near the start of the message only, so a legitimate bolded header
        // later in a well-formed answer is never touched.
        let bareLabelPattern = "(?:^|\\n)\\s*(?:" +
            "\\*{3,}|" +
            "/[A-Za-z][A-Za-z ]{2,29}:|" +
            "\\*{1,2}[A-Za-z][A-Za-z ,]{2,39}:\\*{0,2}|" +
            "[\\*\"'\\[\\(]{1,2}\\s*(?:self[- _]?correction|self[- _]?review|internal monologue|internal reasoning|response generation|chain of thought|style check|tone check|voice check|persona check|vibe check|character check|generating response|generating\\.\\.\\.)" +
            ")"
        if let bareLabelRegex = try? NSRegularExpression(pattern: bareLabelPattern, options: [.caseInsensitive]) {
            var guardIterations = 0
            while guardIterations < 20,
                  let openMatch = bareLabelRegex.firstMatch(in: filtered, range: NSRange(filtered.startIndex..., in: filtered)),
                  let openRange = Range(openMatch.range, in: filtered),
                  filtered.distance(from: filtered.startIndex, to: openRange.lowerBound) < 600 {
                guardIterations += 1
                if let blankRange = filtered.range(of: "\n\n", range: openRange.upperBound..<filtered.endIndex) {
                    filtered.removeSubrange(openRange.lowerBound..<blankRange.upperBound)
                } else {
                    filtered.removeSubrange(openRange.lowerBound..<filtered.endIndex)
                    break
                }
            }
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
                                saveToPhotos(uiImg)
                            } label: {
                                Label("Save to Photos", systemImage: "square.and.arrow.down")
                            }
                            Button {
                                UIPasteboard.general.image = uiImg
                            } label: {
                                Label("Copy Image", systemImage: "doc.on.doc")
                            }
                            Divider()
                            Button(role: .destructive) {
                                reportTarget = message
                            } label: {
                                Label("Report a Concern", systemImage: "flag")
                            }
                        }

                    // Always-visible action row
                    HStack(spacing: 10) {
                        // Save button
                        Button {
                            saveToPhotos(uiImg)
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

                        // Report button — a visible, one-tap path to flag generated imagery,
                        // not just a hidden long-press affordance.
                        Button {
                            reportTarget = message
                        } label: {
                            Label("Report", systemImage: "flag")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundColor(Theme.textSecondary)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 5)
                                .background(
                                    RoundedRectangle(cornerRadius: 7)
                                        .fill(Theme.border.opacity(0.5))
                                )
                        }

                        Spacer(minLength: 0)
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

                VStack(alignment: .leading, spacing: 7) {
                    // Naming the stage matters here: loading a 3 GB checkpoint shows no sampler
                    // progress at all, so a bare "0%" for two minutes reads as a frozen app.
                    Text(diffusionManager.generationStage.isEmpty ? "Working…" : diffusionManager.generationStage)
                        .font(.system(size: 13, design: .monospaced))
                        .foregroundColor(Theme.textPrimary)

                    if diffusionManager.generationProgress > 0 {
                        ProgressView(value: diffusionManager.generationProgress)
                            .progressViewStyle(LinearProgressViewStyle(tint: Color.purple))
                            .frame(width: 170)
                        Text("\(Int(diffusionManager.generationProgress * 100))%")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(Theme.textSecondary)
                    } else {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: Color.purple))
                            .scaleEffect(0.7)
                            .frame(height: 18, alignment: .leading)
                    }

                    // An escape hatch that always works. Generation itself can't be aborted, but
                    // the user should never be stuck staring at a bar with no way back.
                    Button {
                        diffusionManager.cancelGeneration()
                        conversationManager.updateLastMessage(text: "[Image generation cancelled.]")
                        conversationManager.saveConversations()
                    } label: {
                        Text("Cancel")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundColor(Theme.accent)
                    }
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
                // Suppressed thinking/preamble tokens are never streamed as visible text, so
                // without this the bubble would show a bare "Thinking..." with no indication
                // whether it's actively working or has stalled — indistinguishable from a hang.
                let thinkingLabel = llmManager.thinkingTokensUsed > 0
                    ? "Thinking... (\(llmManager.thinkingTokensUsed)T)"
                    : "Thinking..."

                VStack(alignment: .leading, spacing: 8) {
                    Text(isThinking ? thinkingLabel : filteredText)
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
                            Divider()
                            Button(role: .destructive) {
                                reportTarget = message
                            } label: {
                                Label("Report a Concern", systemImage: "flag")
                            }
                        }

                    // ── Web search offer buttons ────────────────────────────
                    // Only ever set on an assistant message asking whether to search — see
                    // `respondToSearchOffer`. Cleared the moment either button is tapped, so
                    // this never lingers once the user has answered.
                    if let query = message.pendingSearchQuery {
                        HStack(spacing: 8) {
                            Button {
                                respondToSearchOffer(accepted: true, query: query, offerMessageId: message.id)
                            } label: {
                                Label("Search", systemImage: "magnifyingglass")
                                    .font(.system(size: 12, weight: .semibold))
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(RoundedRectangle(cornerRadius: 8).fill(Theme.accentCyan))
                            }
                            Button {
                                respondToSearchOffer(accepted: false, query: query, offerMessageId: message.id)
                            } label: {
                                Text("No, just answer")
                                    .font(.system(size: 12, weight: .semibold))
                                    .foregroundColor(Theme.textSecondary)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(RoundedRectangle(cornerRadius: 8).fill(Theme.border.opacity(0.5)))
                            }
                        }
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
            
            // Private-chat toggle — this conversation is kept in memory only
            Button(action: {
                conversationManager.privateMode.toggle()
            }) {
                Image(systemName: conversationManager.privateMode ? "eye.slash.fill" : "eye.fill")
                    .foregroundColor(conversationManager.privateMode ? Theme.accent : Theme.textSecondary)
                    .font(.system(size: 18))
                    .frame(width: 40, height: 40)
                    .background(Theme.border.opacity(0.3))
                    .clipShape(Circle())
                    .overlay(
                        Circle()
                            .stroke(conversationManager.privateMode ? Theme.accent : Color.clear, lineWidth: 1.5)
                    )
            }
            .accessibilityLabel(conversationManager.privateMode ? "Private chat on" : "Private chat off")
            
            // Text Entry
            TextField(isModelActive ? "Execute prompt..." : "Model unloaded...", text: $inputText, axis: .vertical)
                .lineLimit(1...8)
                .font(.system(size: 14))
                .foregroundColor(Theme.textPrimary)
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
        .background(Theme.chrome)
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
                    Text("\(AppInfo.displayName) · runs entirely on device")
                        .font(.system(size: 10))
                        .foregroundColor(Theme.textMuted)
                }
                .padding(.horizontal)
                .padding(.bottom, 24)
            }
            .frame(width: 270)
            .background(Theme.background)
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
        // Only one heavy model can be resident at a time, so a second request during image
        // generation would put two multi-gigabyte models in memory at once and get the app
        // killed. Say so rather than dropping the message on the floor — silently discarding
        // input is what made the earlier stuck-flag bug present as a dead app rather than a
        // visible error.
        if diffusionManager.isGenerating {
            notice = Notice(
                title: "Still Generating",
                message: "Wait for the current image to finish, or tap Cancel on it, before sending another message."
            )
            return
        }

        // Cancel in-progress text generation
        if llmManager.isGenerating {
            llmManager.cancelGeneration()
            return
        }

        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty || pendingAttachmentText != nil else { return }

        inputText = ""
        // Clear again on the next main-queue turn.
        //
        // The field is a `TextField(axis: .vertical)`, which is UIKit underneath. When a send
        // happens while the keyboard still has work in flight — an autocorrection, a predictive
        // suggestion, dictation, a marked composition region — UIKit delivers that pending text
        // to the binding *after* this synchronous block finishes, which put the prompt straight
        // back into a box the user had just watched empty. The message itself always sent, so it
        // looked like a duplicate waiting to happen.
        //
        // Guarded on the value so it only undoes that specific late delivery: if the user has
        // genuinely started typing something new by the time this runs, it leaves it alone.
        DispatchQueue.main.async {
            if inputText == text { inputText = "" }
        }

        // ── Prompt Intent Classification ──────────────────────────────────────
        // Check whether the user is requesting image generation BEFORE attaching
        // files so that file context doesn't accidentally override intent.
        let intent = PromptClassifier.classify(text)
        let isImageRequest: Bool = {
            if case .imageGeneration = intent { return true }
            return false
        }()

        // ── Content policy ────────────────────────────────────────────────────
        // Runs before the prompt reaches any model, and before the message is even added to
        // the conversation, so blocked material is never persisted. The surface matters: an
        // image request is held to the stricter standard, since Guideline 1.1.4 is about what
        // the app *renders*.
        let promptDecision = ContentSafety.review(text, surface: isImageRequest ? .imagePrompt : .chatPrompt)
        guard promptDecision.isAllowed else {
            // Hand the text back rather than swallowing it. Any lexical filter has false
            // positives, and silently destroying what someone typed is the wrong way to be
            // wrong about one.
            inputText = text
            notice = Notice(
                title: "Request Blocked",
                message: promptDecision.message ?? "This request isn't allowed under the app's content policy."
            )
            LogManager.shared.log("ContentSafety: blocked prompt — \(promptDecision.category?.rawValue ?? "unknown")")
            return
        }

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

            diffusionManager.beginGenerationSession(stage: "Preparing…")

            Task {
                // Closes the session on every exit path — success, thrown error, early return.
                // The previous version reset the flag by hand at some exit points and missed
                // others, which is what left the app permanently refusing new messages.
                defer { diffusionManager.endGenerationSession() }

                // Falls back to the last-used model rather than only the currently resident one.
                //
                // `activeModelURL` is nil whenever no chat model happens to be loaded, and that
                // is exactly the state a previous cancellation leaves behind — so the failure
                // compounded: one cancelled generation unloaded the model, and every attempt
                // afterwards captured nil and had nothing to put back. Resolving through
                // `lastUsedModelPath` means "restore the last used LLM" holds even when the run
                // started with nothing loaded.
                let savedLLMUrl = llmManager.activeModelURL
                    ?? llmManager.lastUsedModelPath.map { URL(fileURLWithPath: $0) }

                // Ask the currently loaded chat model to expand the request into a
                // richer SD prompt — must happen BEFORE the unload below, since it needs
                // the model resident. Falls back to the rule-based `refinedPrompt` if no
                // model is loaded, the actor is busy, or the model's output isn't usable.
                var finalPrompt = refinedPrompt
                if llmManager.isModelLoaded {
                    diffusionManager.updateGenerationStage("Writing the image prompt…")
                    if let llmPrompt = await llmManager.generateImagePrompt(from: refinedPrompt) {
                        // The expansion is model output, and an imported model can embellish an
                        // innocuous request into something that would never have passed the
                        // original screen. Re-check what actually reaches the sampler; fall back
                        // to the rule-based prompt, which has already been cleared, rather than
                        // failing the whole request over the model's phrasing.
                        if ContentSafety.review(llmPrompt, surface: .imagePrompt).isAllowed {
                            finalPrompt = llmPrompt
                            conversationManager.updateLastMessage(text: finalPrompt)
                        } else {
                            LogManager.shared.log("ContentSafety: rejected LLM-expanded image prompt, using original")
                        }
                    }
                }
                guard !diffusionManager.isCancelled else { return }

                diffusionManager.updateGenerationStage("Freeing memory…")
                await llmManager.unloadModelAsync()
                // Wait for the LLM's memory to actually come back before attempting to map
                // SDXL's massive weights into RAM. A flat sleep here was the same mistake
                // already fixed on the reverse handoff below (see `MemoryBudget.waitForRelease`):
                // Metal returns buffers to the system on its own schedule, so a fixed delay is
                // routinely too short and makes the diffusion safety check read less headroom
                // than is really about to be available.
                let diffSizeGB = diffusionManager.getFileSizeGB(at: URL(fileURLWithPath: diffPath))
                await MemoryBudget.waitForRelease(atLeastGB: diffusionManager.memoryHeadroomNeededGB(
                    forModelSizeGB: diffSizeGB, outputSize: diffusionManager.outputSize
                ))

                var resultData: Data? = nil
                var failureMessage: String? = nil

                // Cancelling here skips the work but must NOT skip the teardown below. The chat
                // model has already been evicted at this point, so returning early — which is
                // what this path used to do — left the user with no model loaded at all and a
                // bar reading "No model loaded", as though cancelling had unloaded their model
                // on purpose. Falling through restores it exactly as a completed run does.
                if !diffusionManager.isCancelled {
                    do {
                        diffusionManager.updateGenerationStage("Loading diffusion model…")
                        try await diffusionManager.loadDiffusionModelAsync(at: URL(fileURLWithPath: diffPath))

                        diffusionManager.updateGenerationStage("Generating…")
                        // Each `await` inside releases the MainActor, so the UI stays responsive
                        // throughout the multi-minute denoising loop.
                        resultData = try await diffusionManager.generateImageAsync(prompt: finalPrompt)
                    } catch {
                        failureMessage = error.localizedDescription
                        LogManager.shared.log("Image generation failed — \(error.localizedDescription)")
                    }
                }

                // Teardown runs whether or not generation succeeded, so a failure never strands
                // several gigabytes of checkpoint resident with no chat model loaded.
                diffusionManager.updateGenerationStage("Freeing memory…")
                await diffusionManager.unloadDiffusionModelAsync()

                if let llm = savedLLMUrl {
                    // Wait for the checkpoint's memory to actually come back before asking the
                    // chat model to load. A flat sleep here meant the reload ran its safety
                    // check while Metal was still handing several gigabytes back to the system,
                    // so a successful generation could still end with an out-of-memory error
                    // sitting in the model bar.
                    let needed = llmManager.memoryHeadroomNeededGB(
                        forModelSizeGB: llmManager.getModelSizeGB(at: llm)
                    )
                    await MemoryBudget.waitForRelease(atLeastGB: needed)

                    diffusionManager.updateGenerationStage("Restoring chat model…")
                    llmManager.loadModel(at: llm)
                }

                // A cancelled run has already had its bubble replaced and its session closed;
                // dropping the result here is the whole point of cancellation.
                guard !diffusionManager.isCancelled else { return }

                if let data = resultData {
                    // Screen the pixels, not just the prompt. Prompt filtering only governs what
                    // was asked for; an uncensored checkpoint can produce explicit output from a
                    // request that contained nothing to object to. This is the only point where
                    // that can actually be caught, so a flagged image is discarded here — never
                    // shown in the conversation, never written to RAG, never saved to Photos.
                    diffusionManager.updateGenerationStage("Checking result…")
                    let verdict = await ImageSafetyAnalyzer.screen(imageData: data)
                    if case .blocked(let reason) = verdict {
                        conversationManager.updateLastMessage(
                            text: "[The generated image was blocked by the content filter (\(reason)) and has been discarded.]"
                        )
                        conversationManager.saveConversations()
                        LogManager.shared.log("ContentSafety: discarded generated image — \(reason)")
                        return
                    }

                    conversationManager.updateLastMessageImage(imageData: data)
                    ragManager.ingestGeneratedImage(prompt: finalPrompt, imageData: data)
                } else {
                    conversationManager.updateLastMessage(
                        text: "[Couldn't generate the image. \(failureMessage ?? "The diffusion model failed to run.")]"
                    )
                }
                conversationManager.saveConversations()
            }
            return
        }
        // ─────────────────────────────────────────────────────────────────────

        // Captured before the attachment merge below clears these — needed to keep the web
        // search offer from firing on a message that's really "analyze this attachment."
        let hadAttachment = pendingAttachmentText != nil

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

        // Distress signals get resources attached, not a refusal. The model still answers
        // normally below — this appears alongside the conversation rather than replacing it,
        // because cutting someone off mid-sentence is not help.
        if promptDecision.attachesCrisisResources {
            conversationManager.addMessageToActive(isUser: false, text: LegalText.crisisResources)
        }

        // ── Web search offer ──────────────────────────────────────────────────
        // Gated entirely on the user having turned internet access on in Settings — when it's
        // off, nothing here runs at all, so behavior is byte-for-byte unchanged for anyone who
        // hasn't opted in. Never fires on an attachment turn (that's "analyze this," not "look
        // this up"), and never auto-searches — this only ever offers, the app never decides on
        // the user's behalf. See `respondToSearchOffer` for what happens when they answer.
        if webSearchManager.isEnabled, !hadAttachment,
           WebSearchClassifier.classify(text) != nil {
            conversationManager.addSearchOfferToActive(
                text: "This looks like it might need current information from the internet. Want me to search, or should I just answer from what I already know?",
                query: text
            )
            return
        }

        generateTextResponse(text: text, promptText: promptText, history: history)
    }

    /// Runs a normal (no web search) response for a captured user turn. This is the exact body
    /// that used to live inline in `sendMessage()`'s `Task`, factored out so both the everyday
    /// path and "user declined the search offer" (`respondToSearchOffer`) share one
    /// implementation instead of two copies that could drift.
    ///
    /// `extraRagContext` / `responseSuffix` / `createsNewBubble` exist only for
    /// `performWebSearchAndRespond`: folding search results into context, appending a Sources
    /// footer once the answer clears content review, and continuing to stream into the
    /// "Searching the interwebs…" bubble that already exists instead of creating a second one.
    /// All three default to a no-op, so the everyday call site behaves exactly as the original
    /// inline code did.
    private func generateTextResponse(
        text: String,
        promptText: String,
        history: [ChatMessage],
        extraRagContext: String = "",
        responseSuffix: String = "",
        createsNewBubble: Bool = true
    ) {
        let capturedPromptText = promptText
        let scanner = StreamSafetyScanner()

        Task {
            // Evict Stable Diffusion from RAM before running the LLM — only if something is
            // actually resident. This used to unload (a harmless no-op when nothing was
            // loaded) and then sleep a flat second unconditionally on *every* chat message,
            // adding a full second of dead latency to the app's most common interaction
            // regardless of whether there was ever anything to free.
            let wasDiffusionLoaded = await MainActor.run { diffusionManager.diffusionLoadState.isLoaded }
            if wasDiffusionLoaded {
                await diffusionManager.unloadDiffusionModelAsync()
                // Wait for the freed memory to actually come back before reloading the LLM
                // below, same reasoning as the image-generation handoff: a flat sleep here
                // was routinely too short and raced Metal's own buffer-release schedule.
                // Skipped when there's no LLM to reload — nothing downstream needs the
                // memory back immediately in that case.
                if let llm = await MainActor.run(body: { llmManager.activeModelURL }) {
                    let neededGB = await MainActor.run {
                        llmManager.memoryHeadroomNeededGB(forModelSizeGB: llmManager.getModelSizeGB(at: llm))
                    }
                    await MemoryBudget.waitForRelease(atLeastGB: neededGB)
                }
            }

            await MainActor.run {
                if !llmManager.isModelLoaded, let llm = llmManager.activeModelURL {
                    llmManager.loadModel(at: llm)
                }

                if enableMemories && !text.isEmpty {
                    memoryManager.extractMemories(from: text)
                    // NOTE: personality analysis can trigger its own background LLM call
                    // (every 3rd message). LlamaRunner is a single serialized actor, so
                    // firing that here would race the chat response below for the model —
                    // whichever reaches the actor first runs, silently delaying replies.
                    // It's fired from onComplete instead, strictly after this turn finishes.
                }

                let ragContext = (enableRAG ? ragManager.retrieveRelevantContext(query: text) : "") + extraRagContext
                let memoriesContext = enableMemories ? memoryManager.getFormattedMemoriesForContext() : ""

                if createsNewBubble {
                    conversationManager.addMessageToActive(isUser: false, text: "")
                }

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
                    prompt: capturedPromptText,
                    history: history,
                    systemPrompt: finalSystemPrompt,
                    memoriesContext: memoriesContext,
                    ragContext: ragContext
                ) { token in
                    let updated = (conversationManager.activeConversation?.messages.last?.text ?? "") + token
                    conversationManager.updateLastMessage(text: updated)

                    // Abort mid-stream rather than rendering violating tokens and retracting
                    // them a second later. Only screens the non-negotiable category, and only
                    // every few hundred characters, so it costs nothing per token.
                    if scanner.shouldScan(updated),
                       let violation = ContentSafety.streamingViolation(in: updated) {
                        llmManager.cancelGeneration()
                        conversationManager.updateLastMessage(
                            text: "[Response stopped by the content filter — \(violation.reportLabel).]"
                        )
                        LogManager.shared.log("ContentSafety: cancelled stream — \(violation.rawValue)")
                    }
                } onComplete: { finalText in
                    var cleanedText = self.filterThoughts(from: finalText, stripMarkdown: self.personalityManager.isMature)
                    if cleanedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        cleanedText = "[No response content was generated — the model may have only produced internal notes for this turn. Try regenerating or rephrasing your prompt.]"
                    }

                    // Final screen of the completed response. The streaming check runs on
                    // partial text and can't see a violation that only forms across the last
                    // few tokens, so this is the one that actually has to hold.
                    let outputDecision = ContentSafety.review(cleanedText, surface: .modelOutput)
                    if !outputDecision.isAllowed {
                        cleanedText = outputDecision.message ?? "[This response was withheld by the content filter.]"
                        LogManager.shared.log("ContentSafety: withheld response — \(outputDecision.category?.rawValue ?? "unknown")")
                    } else if !responseSuffix.isEmpty {
                        cleanedText += responseSuffix
                    }

                    conversationManager.updateLastMessage(text: cleanedText)
                    conversationManager.saveConversations()

                    // Run personality analysis only after the real response is fully done,
                    // so its occasional background LLM call never delays a chat reply.
                    if enableMemories && !text.isEmpty, let activeModel = llmManager.activeModelURL?.lastPathComponent {
                        personalityManager.analyzeUserMessage(text, modelName: activeModel, llmManager: llmManager)
                    }
                    // Capture the AI's own stated likes/dislikes/favorites from its reply
                    // (e.g. "my favorite car is a Tesla") so it stays consistent if asked
                    // again, rather than borrowing an answer from the user's own memories.
                    if enableMemories, let activeModel = llmManager.activeModelURL?.lastPathComponent {
                        personalityManager.analyzeAssistantMessage(cleanedText, modelName: activeModel)
                    }
                }
            }
        }
    }

    // MARK: - Web search

    /// Handles the user tapping Search or No on a pending offer bubble — see
    /// `messageBubble(for:)` and `ChatMessage.pendingSearchQuery`. `query` is the original user
    /// message the offer was generated from; re-classified here rather than persisted as a
    /// `WebSearchQueryType`, since classification is cheap and deterministic and this keeps
    /// `ChatMessage` from needing to know that type exists at all.
    private func respondToSearchOffer(accepted: Bool, query: String, offerMessageId: UUID) {
        guard let messages = conversationManager.activeConversation?.messages,
              let offerIndex = messages.firstIndex(where: { $0.id == offerMessageId }) else { return }

        conversationManager.clearPendingSearch(messageId: offerMessageId)

        // History is everything before the user message this offer answers — excluding both
        // that message and the offer bubble itself, the same split `sendMessage` already draws
        // between `history` and the turn's own `prompt`. Indexed off the offer's actual position
        // rather than assumed to be "the last two messages": nothing stops the user from sending
        // something newer before getting back to an older, still-pending offer.
        let history = Array(messages[0..<max(0, offerIndex - 1)])

        guard accepted else {
            generateTextResponse(text: query, promptText: query, history: history)
            return
        }
        performWebSearchAndRespond(query: query, history: history)
    }

    /// Runs the actual search, then generation, for an accepted offer. Owns its own placeholder
    /// bubble ("🔍 Searching the interwebs…") so there's always visible feedback during the
    /// network round trip — replaced in place by the streamed answer once the search resolves,
    /// or by a plain failure message if it doesn't.
    private func performWebSearchAndRespond(query: String, history: [ChatMessage]) {
        conversationManager.addMessageToActive(isUser: false, text: "🔍 Searching the interwebs…")

        Task {
            let queryType = WebSearchClassifier.classify(query) ?? .general(query: query)
            do {
                let result = try await webSearchManager.search(queryType)
                let extraContext = "\n\nWeb search results for \"\(query)\":\n\(result.answer)\n\nUse this information to help answer the question, and mention that it came from a web search."
                let sourcesFooter = result.sources.isEmpty ? "" :
                    "\n\nSources:\n" + result.sources.map { "• \($0.title) — \($0.url)" }.joined(separator: "\n")

                await MainActor.run {
                    // Clears the "Searching…" placeholder text so the token-streaming callback's
                    // accumulate-by-appending logic starts from empty rather than appending onto
                    // the stage text.
                    conversationManager.updateLastMessage(text: "")
                    generateTextResponse(
                        text: query, promptText: query, history: history,
                        extraRagContext: extraContext, responseSuffix: sourcesFooter,
                        createsNewBubble: false
                    )
                }
            } catch WebSearchError.blocked(let message) {
                // A deliberate content-policy refusal — this one stays a hard stop, not
                // something to paper over with a normal answer.
                await MainActor.run {
                    conversationManager.updateLastMessage(text: "[\(message)]")
                    conversationManager.saveConversations()
                }
            } catch {
                // Any technical failure — no results, a network hiccup, an unparseable
                // response — degrades to a normal answer instead of a dead end. The free
                // keyless providers (DuckDuckGo's instant-answer API especially) often have
                // nothing for a given query even though the search itself worked correctly;
                // the model can still be useful from what it already knows, honestly framed
                // as not having found live results, rather than the user getting nothing.
                LogManager.shared.log("WebSearch: search failed, answering without results — \(error.localizedDescription)")
                let extraContext = "\n\n(A web search for \"\(query)\" was attempted but didn't return usable results. Briefly say you couldn't find live results for this, then answer as well as you can from what you already know.)"
                await MainActor.run {
                    conversationManager.updateLastMessage(text: "")
                    generateTextResponse(
                        text: query, promptText: query, history: history,
                        extraRagContext: extraContext, createsNewBubble: false
                    )
                }
            }
        }
    }

    // MARK: - Photos

    /// Saves a generated image to the user's library, asking for add-only access first.
    ///
    /// The previous version called `UIImageWriteToSavedPhotosAlbum` directly from the context
    /// menu with no authorization check at all, which on a device that had never been asked
    /// simply did nothing — no prompt, no error, no image. Add-only is the narrowest scope that
    /// does the job: the app never gains the ability to read the user's photos.
    private func saveToPhotos(_ image: UIImage) {
        let status = PHPhotoLibrary.authorizationStatus(for: .addOnly)
        switch status {
        case .authorized, .limited:
            performPhotoSave(image)
        case .notDetermined:
            PHPhotoLibrary.requestAuthorization(for: .addOnly) { newStatus in
                DispatchQueue.main.async {
                    if newStatus == .authorized || newStatus == .limited {
                        performPhotoSave(image)
                    } else {
                        notice = Notice(
                            title: "Photos Access Needed",
                            message: "To save images, allow \(AppInfo.displayName) to add photos in Settings › Privacy & Security › Photos."
                        )
                    }
                }
            }
        default:
            notice = Notice(
                title: "Photos Access Needed",
                message: "To save images, allow \(AppInfo.displayName) to add photos in Settings › Privacy & Security › Photos."
            )
        }
    }

    private func performPhotoSave(_ image: UIImage) {
        PHPhotoLibrary.shared().performChanges {
            PHAssetChangeRequest.creationRequestForAsset(from: image)
        } completionHandler: { success, error in
            DispatchQueue.main.async {
                if success {
                    notice = Notice(title: "Saved", message: "The image was added to your photo library.")
                } else {
                    notice = Notice(
                        title: "Couldn't Save",
                        message: error?.localizedDescription ?? "The image couldn't be added to your photo library."
                    )
                }
            }
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
                        // Attached files are an untrusted input path into the model just like a
                        // typed prompt is, so the description it produces gets the same screen.
                        let outputDecision = ContentSafety.review(cleanedText, surface: .modelOutput)
                        if !outputDecision.isAllowed {
                            cleanedText = outputDecision.message ?? "[This response was withheld by the content filter.]"
                        }
                        conversationManager.updateLastMessage(text: cleanedText)
                        conversationManager.saveConversations()

                        if enableMemories, let activeModel = llmManager.activeModelURL?.lastPathComponent {
                            personalityManager.analyzeAssistantMessage(cleanedText, modelName: activeModel)
                        }
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

/// Throttles the mid-stream safety scan.
///
/// A reference type because the token callback is an escaping closure captured from a SwiftUI
/// `View` struct, which has nowhere to put mutable per-generation state. Scanning the whole
/// accumulated buffer on every token would be quadratic in response length; scanning it every
/// few hundred characters bounds that to something that doesn't show up in a profile, and the
/// completed response is screened in full afterwards regardless.
final class StreamSafetyScanner {
    private var lastScannedLength = 0
    private let interval = 240

    func shouldScan(_ text: String) -> Bool {
        guard text.count - lastScannedLength >= interval else { return false }
        lastScannedLength = text.count
        return true
    }
}
