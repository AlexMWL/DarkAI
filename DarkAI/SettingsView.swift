import SwiftUI
import UniformTypeIdentifiers

struct SettingsView: View {
    @Environment(\.dismiss) var dismiss
    @ObservedObject var llmManager: LLMManager
    @ObservedObject var memoryManager: MemoryManager
    @ObservedObject var ragManager: RAGManager
    @ObservedObject var personalityManager: PersonalityManager
    
    @Binding var customInstructions: String
    @Binding var enableRAG: Bool
    @Binding var enableMemories: Bool
    
    @State private var importedModels: [URL] = []
    @State private var showModelImporter = false
    @State private var showDocImporter = false
    @State private var showResetPersonalityAlert = false
    
    // Context Warning
    @State private var showContextWarningPopup = false
    
    // Failsafe Modal States
    @State private var showFailsafePopup = false
    @State private var selectedModelToLoad: URL? = nil
    @State private var failsafeMessage = ""
    @State private var failsafeRequiredRAM = 0.0
    @State private var isFailsafeWarningOnly = false
    
    var body: some View {
        NavigationView {
            ZStack {
                Theme.background
                    .ignoresSafeArea()
                
                ScrollView {
                    VStack(spacing: 24) {
                        
                        // SECTION 1: Local Model Manager (.gguf)
                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                Image(systemName: "cpu")
                                    .foregroundColor(Theme.accentCyan)
                                    .font(.headline)
                                Text("LOCAL LLM MODELS (.GGUF)")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(Theme.textPrimary)
                                    .kerning(1.2)
                                Spacer()
                                Button(action: { showModelImporter = true }) {
                                    HStack(spacing: 6) {
                                        Image(systemName: "plus")
                                        Text("Import")
                                    }
                                    .font(.system(size: 12, weight: .bold))
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(
                                        RoundedRectangle(cornerRadius: 8)
                                            .fill(LinearGradient(colors: [Theme.accent, Theme.accentCyan], startPoint: .leading, endPoint: .trailing))
                                    )
                                }
                            }
                            
                            if importedModels.isEmpty {
                                Text("No models imported yet. Tap 'Import' to copy a .gguf model from your Files storage.")
                                    .font(.system(size: 13))
                                    .foregroundColor(Theme.textSecondary)
                                    .padding()
                                    .frame(maxWidth: .infinity, alignment: .center)
                                    .glassCard(cornerRadius: 12)
                            } else {
                                VStack(spacing: 12) {
                                    ForEach(importedModels, id: \.self) { url in
                                        modelRow(for: url)
                                    }
                                }
                            }
                        }
                        .glassCard(cornerRadius: 16)
                        
                        // SECTION 2: Custom Prompt Instructions
                        VStack(alignment: .leading, spacing: 12) {
                            HStack {
                                Image(systemName: "slider.horizontal.3")
                                    .foregroundColor(Theme.accent)
                                Text("CUSTOM SYSTEM INSTRUCTIONS")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(Theme.textPrimary)
                                    .kerning(1.2)
                            }
                            
                            TextEditor(text: $customInstructions)
                                .scrollContentBackground(.hidden) // Fix for iOS 16+ white background washout
                                .font(.system(size: 13, design: .monospaced))
                                .foregroundColor(Theme.textPrimary)
                                .padding(4)
                                .frame(minHeight: 80)
                                .background(Theme.background.opacity(0.5))
                                .cornerRadius(8)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(Theme.border, lineWidth: 1)
                                )
                        }
                        .glassCard(cornerRadius: 16)
                        
                        // SECTION: Model Parameters (Token Limit)
                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                Image(systemName: "slider.horizontal.below.rectangle")
                                    .foregroundColor(Theme.accentCyan)
                                Text("MODEL PARAMETERS")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(Theme.textPrimary)
                                    .kerning(1.2)
                            }
                            
                            Divider().background(Theme.border)
                            
                            VStack(alignment: .leading, spacing: 8) {
                                HStack {
                                    Text("Context Window Limit:")
                                        .font(.system(size: 13))
                                        .foregroundColor(Theme.textSecondary)
                                    Spacer()
                                    Text("\(llmManager.contextTokenLimit) tokens")
                                        .font(.system(size: 13, weight: .bold, design: .monospaced))
                                        .foregroundColor(Theme.accent)
                                }
                                
                                GeometryReader { geo in
                                    ZStack(alignment: .leading) {
                                        Slider(value: Binding(
                                            get: { Double(llmManager.contextTokenLimit) },
                                            set: { llmManager.contextTokenLimit = Int($0) }
                                        ), in: 2048...32768, step: 512, onEditingChanged: { editing in
                                            if !editing {
                                                if llmManager.contextTokenLimit > llmManager.safeContextLimit {
                                                    showContextWarningPopup = true
                                                }
                                            }
                                        })
                                        .accentColor(Theme.accent)
                                        
                                        let totalRange = 32768.0 - 2048.0
                                        let safeValue = Double(llmManager.safeContextLimit)
                                        let percentage = max(0, min(1, (safeValue - 2048.0) / totalRange))
                                        let thumbWidth: CGFloat = 28
                                        let trackWidth = geo.size.width - thumbWidth
                                        let xOffset = (trackWidth * percentage) + (thumbWidth / 2)
                                        
                                        Rectangle()
                                            .fill(Color.orange)
                                            .frame(width: 2, height: 12)
                                            .offset(x: xOffset - 1, y: (geo.size.height - 12) / 2)
                                            .allowsHitTesting(false)
                                    }
                                }
                                .frame(height: 28)
                                
                                if llmManager.contextTokenLimit > llmManager.safeContextLimit {
                                    Text("⚠️ Warning: Based on your device memory and the loaded model size, setting the context limit above \(llmManager.safeContextLimit) tokens risks an Out-Of-Memory crash.")
                                        .font(.system(size: 11))
                                        .foregroundColor(.orange)
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                            }
                            
                            Divider().background(Theme.border)
                            
                            VStack(alignment: .leading, spacing: 8) {
                                HStack {
                                    Text("Max Output Limit:")
                                        .font(.system(size: 13))
                                        .foregroundColor(Theme.textSecondary)
                                    Spacer()
                                    Text("\(llmManager.maxTokens) tokens")
                                        .font(.system(size: 13, weight: .bold, design: .monospaced))
                                        .foregroundColor(Theme.accent)
                                }
                                
                                Slider(value: Binding(
                                    get: { Double(llmManager.maxTokens) },
                                    set: { llmManager.maxTokens = Int($0) }
                                ), in: 64...8192, step: 128)
                                .accentColor(Theme.accent)
                            }
                            
                            Divider().background(Theme.border)
                            
                            VStack(alignment: .leading, spacing: 8) {
                                HStack {
                                    Text("Temperature (Creativity):")
                                        .font(.system(size: 13))
                                        .foregroundColor(Theme.textSecondary)
                                    Spacer()
                                    if llmManager.chaosModeEnabled {
                                        Text("CHAOS (2.50)")
                                            .font(.system(size: 13, weight: .bold, design: .monospaced))
                                            .foregroundColor(.red)
                                    } else {
                                        Text(String(format: "%.2f", llmManager.temperature))
                                            .font(.system(size: 13, weight: .bold, design: .monospaced))
                                            .foregroundColor(Theme.accent)
                                    }
                                }
                                
                                Slider(value: $llmManager.temperature, in: 0.0...2.0, step: 0.05)
                                    .accentColor(llmManager.chaosModeEnabled ? .gray : Theme.accent)
                                    .disabled(llmManager.chaosModeEnabled)
                                    .opacity(llmManager.chaosModeEnabled ? 0.5 : 1.0)
                                    
                                Toggle(isOn: $llmManager.chaosModeEnabled) {
                                    Text("Chaos Mode (Max Creativity)")
                                        .font(.system(size: 13))
                                        .foregroundColor(llmManager.chaosModeEnabled ? .red : Theme.textSecondary)
                                }
                                .toggleStyle(SwitchToggleStyle(tint: .red))
                            }
                        }
                        .glassCard(cornerRadius: 16)
                        
                        // SECTION 3: RAG Document Manager
                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                Image(systemName: "doc.text.magnifyingglass")
                                    .foregroundColor(Theme.accentCyan)
                                Text("RAG CONFIGURATION")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(Theme.textPrimary)
                                    .kerning(1.2)
                                Spacer()
                                Toggle("", isOn: $enableRAG)
                                    .toggleStyle(SwitchToggleStyle(tint: Theme.accent))
                                    .labelsHidden()
                            }
                            
                            if enableRAG {
                                Divider().background(Theme.border)
                                
                                HStack {
                                    Text("DOCUMENT INDEX")
                                        .font(.system(size: 11, weight: .semibold))
                                        .foregroundColor(Theme.textSecondary)
                                    Spacer()
                                    Button(action: { showDocImporter = true }) {
                                        HStack(spacing: 4) {
                                            Image(systemName: "doc.badge.plus")
                                            Text("Add File")
                                        }
                                        .font(.system(size: 12, weight: .semibold))
                                        .foregroundColor(Theme.accentCyan)
                                    }
                                }
                                
                                if ragManager.documents.isEmpty {
                                    Text("No documents indexed. Add text files (.txt) to query local knowledge.")
                                        .font(.system(size: 13))
                                        .foregroundColor(Theme.textMuted)
                                        .padding(.vertical, 8)
                                } else {
                                    VStack(alignment: .leading, spacing: 8) {
                                        ForEach(ragManager.documents) { doc in
                                            HStack {
                                                Image(systemName: "doc.plaintext")
                                                    .foregroundColor(Theme.textSecondary)
                                                VStack(alignment: .leading, spacing: 2) {
                                                    Text(doc.name)
                                                        .font(.system(size: 13, weight: .medium))
                                                        .foregroundColor(Theme.textPrimary)
                                                    Text("\(doc.chunks.count) chunks indexed")
                                                        .font(.system(size: 11))
                                                        .foregroundColor(Theme.textMuted)
                                                }
                                                Spacer()
                                                Button(action: {
                                                    if let idx = ragManager.documents.firstIndex(where: { $0.id == doc.id }) {
                                                        ragManager.deleteDocument(at: IndexSet(integer: idx))
                                                    }
                                                }) {
                                                    Image(systemName: "trash")
                                                        .foregroundColor(.red.opacity(0.8))
                                                        .font(.system(size: 13))
                                                }
                                            }
                                            .padding(8)
                                            .background(Theme.background.opacity(0.4))
                                            .cornerRadius(8)
                                        }
                                    }
                                }
                            }
                        }
                        .glassCard(cornerRadius: 16)
                        
                        // SECTION 4: Long-Term Memories
                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                Image(systemName: "brain.head.profile")
                                    .foregroundColor(Theme.accentRose)
                                Text("CONVERSATIONAL MEMORIES")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(Theme.textPrimary)
                                    .kerning(1.2)
                                Spacer()
                                Toggle("", isOn: $enableMemories)
                                    .toggleStyle(SwitchToggleStyle(tint: Theme.accentRose))
                                    .labelsHidden()
                            }
                            
                            if enableMemories {
                                Divider().background(Theme.border)
                                
                                HStack {
                                    Text("EXTRACTED PREFERENCES")
                                        .font(.system(size: 11, weight: .semibold))
                                        .foregroundColor(Theme.textSecondary)
                                    Spacer()
                                    Button(action: { memoryManager.clearAllMemories() }) {
                                        Text("Clear All")
                                            .font(.system(size: 12, weight: .semibold))
                                            .foregroundColor(.red.opacity(0.8))
                                    }
                                }
                                
                                if memoryManager.memories.isEmpty {
                                    Text("No memories extracted yet. Tell the chatbot things like 'I prefer Python' or 'My name is John' to build long-term memory.")
                                        .font(.system(size: 13))
                                        .foregroundColor(Theme.textMuted)
                                        .padding(.vertical, 8)
                                } else {
                                    VStack(alignment: .leading, spacing: 8) {
                                        ForEach(Array(memoryManager.memories.enumerated()), id: \.offset) { index, memory in
                                            HStack {
                                                Text(memory)
                                                    .font(.system(size: 13))
                                                    .foregroundColor(Theme.textPrimary)
                                                    .lineLimit(2)
                                                Spacer()
                                                Button(action: { memoryManager.removeMemory(at: index) }) {
                                                    Image(systemName: "xmark.circle")
                                                        .foregroundColor(Theme.textSecondary)
                                                        .font(.system(size: 14))
                                                }
                                            }
                                            .padding(8)
                                            .background(Theme.background.opacity(0.4))
                                            .cornerRadius(8)
                                        }
                                    }
                                }
                            }
                        }
                        .glassCard(cornerRadius: 16)
                        
                        // Personality Reset Section
                        VStack(alignment: .leading, spacing: 12) {
                            HStack {
                                Image(systemName: "person.text.rectangle")
                                    .foregroundColor(Theme.accentRose)
                                Text("Personality Matrix")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(.white)
                                Spacer()
                                Text(personalityManager.databaseSizeString)
                                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                                    .foregroundColor(Theme.accentRose)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(Theme.accentRose.opacity(0.15))
                                    .cornerRadius(6)
                            }
                            
                            Text("DarkAI slowly learns your speech patterns and builds a unique persona over time. Resetting will erase all learned personality traits for the currently loaded model.")
                                .font(.system(size: 13))
                                .foregroundColor(Theme.textSecondary)
                                .lineSpacing(4)
                            
                            Button(action: {
                                showResetPersonalityAlert = true
                            }) {
                                HStack {
                                    Image(systemName: "trash")
                                    Text("Reset Current Model's Personality")
                                }
                                .font(.system(size: 13, weight: .bold))
                                .foregroundColor(.white)
                                .padding(.vertical, 10)
                                .frame(maxWidth: .infinity)
                                .background(Theme.accentRose.opacity(0.2))
                                .cornerRadius(8)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(Theme.accentRose.opacity(0.5), lineWidth: 1)
                                )
                            }
                            .alert("Reset Personality?", isPresented: $showResetPersonalityAlert) {
                                Button("Cancel", role: .cancel) { }
                                Button("Reset", role: .destructive) {
                                    if let currentModel = llmManager.activeModelURL?.lastPathComponent {
                                        personalityManager.resetPersonality(for: currentModel)
                                    }
                                }
                            } message: {
                                Text("This will erase all learned speech patterns for the currently loaded model. This action cannot be undone.")
                            }
                            .disabled(llmManager.activeModelURL == nil)
                            .opacity(llmManager.activeModelURL == nil ? 0.5 : 1.0)
                        }
                        .glassCard(cornerRadius: 16)
                        
                        // Sideload info section
                        VStack(spacing: 8) {
                            Text("SIDELOAD STATUS: SYSTEM BYPASS ACTIVE")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundColor(Theme.accentCyan)
                                .kerning(1.5)
                            
                            Text("This application has been compiled with increased memory limit entitlement, enabling addressable RAM access beyond standard App Store caps.")
                                .font(.system(size: 11))
                                .foregroundColor(Theme.textSecondary)
                                .multilineTextAlignment(.center)
                                .lineSpacing(3)
                        }
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Theme.accentCyan.opacity(0.05))
                        .cornerRadius(12)
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Theme.accentCyan.opacity(0.2), lineWidth: 1)
                        )
                        
                    }
                    .padding()
                }
                
                // Loading Model Overlay
                if case let .loading(progress, status) = llmManager.loadState {
                    ZStack {
                        Color.black.opacity(0.75)
                            .ignoresSafeArea()
                        
                        VStack(spacing: 20) {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: Theme.accent))
                                .scaleEffect(1.5)
                            
                            Text(status)
                                .font(.system(size: 16, weight: .semibold))
                                .foregroundColor(Theme.textPrimary)
                            
                            VStack(spacing: 6) {
                                Text("\(Int(progress * 100))%")
                                    .font(.system(size: 12))
                                    .foregroundColor(Theme.textSecondary)
                                
                                GeometryReader { geo in
                                    ZStack(alignment: .leading) {
                                        RoundedRectangle(cornerRadius: 4)
                                            .fill(Theme.border)
                                            .frame(height: 8)
                                        
                                        RoundedRectangle(cornerRadius: 4)
                                            .fill(LinearGradient(colors: [Theme.accent, Theme.accentCyan], startPoint: .leading, endPoint: .trailing))
                                            .frame(width: geo.size.width * progress, height: 8)
                                            .shadow(color: Theme.accent.opacity(0.5), radius: 4)
                                    }
                                }
                                .frame(height: 8)
                            }
                            .frame(width: 200)
                        }
                        .padding(30)
                        .background(Theme.cardBackground)
                        .cornerRadius(20)
                        .overlay(
                            RoundedRectangle(cornerRadius: 20)
                                .stroke(Theme.border, lineWidth: 1)
                        )
                    }
                }
                
                // Memory Failsafe Alert Overlay
                if showFailsafePopup {
                    ZStack {
                        Color.black.opacity(0.8)
                            .ignoresSafeArea()
                        
                        VStack(spacing: 20) {
                            Image(systemName: isFailsafeWarningOnly ? "exclamationmark.triangle" : "xmark.octagon")
                                .font(.system(size: 48))
                                .foregroundColor(isFailsafeWarningOnly ? .yellow : .red)
                                .neonGlow(color: isFailsafeWarningOnly ? .yellow : .red, radius: 10)
                            
                            Text(isFailsafeWarningOnly ? "MEMORY ALLOCATION WARNING" : "MEMORY FAILSAFE TRIGGERED")
                                .font(.system(size: 16, weight: .bold))
                                .foregroundColor(.white)
                                .kerning(1.2)
                            
                            Text(failsafeMessage)
                                .font(.system(size: 13))
                                .foregroundColor(Theme.textSecondary)
                                .multilineTextAlignment(.center)
                                .lineSpacing(4)
                            
                            VStack(spacing: 8) {
                                HStack {
                                    Text("iPhone 17 Pro Max RAM:")
                                        .foregroundColor(Theme.textMuted)
                                    Spacer()
                                    Text(String(format: "%.1f GB", llmManager.systemMemoryGB))
                                        .foregroundColor(.white)
                                }
                                HStack {
                                    Text("Estimated Model Footprint:")
                                        .foregroundColor(Theme.textMuted)
                                    Spacer()
                                    Text(String(format: "%.1f GB", failsafeRequiredRAM))
                                        .foregroundColor(isFailsafeWarningOnly ? .yellow : .red)
                                }
                            }
                            .font(.system(size: 12, design: .monospaced))
                            .padding()
                            .background(Theme.background)
                            .cornerRadius(10)
                            
                            HStack(spacing: 16) {
                                Button(action: {
                                    showFailsafePopup = false
                                    selectedModelToLoad = nil
                                }) {
                                    Text("Cancel")
                                        .font(.system(size: 14, weight: .bold))
                                        .foregroundColor(.white)
                                        .frame(maxWidth: .infinity)
                                        .padding(.vertical, 12)
                                        .background(
                                            RoundedRectangle(cornerRadius: 10)
                                                .stroke(Theme.border, lineWidth: 1.5)
                                        )
                                }
                                
                                Button(action: {
                                    if let url = selectedModelToLoad {
                                        llmManager.loadModel(at: url, forceLoad: true)
                                    }
                                    showFailsafePopup = false
                                }) {
                                    Text(isFailsafeWarningOnly ? "Load Model" : "Force Sideload")
                                        .font(.system(size: 14, weight: .bold))
                                        .foregroundColor(.white)
                                        .frame(maxWidth: .infinity)
                                        .padding(.vertical, 12)
                                        .background(
                                            RoundedRectangle(cornerRadius: 10)
                                                .fill(isFailsafeWarningOnly ? Color.yellow.opacity(0.8) : Color.red.opacity(0.8))
                                        )
                                }
                            }
                        }
                        .padding(26)
                        .frame(width: 320)
                        .background(Theme.cardBackground)
                        .cornerRadius(20)
                        .overlay(
                            RoundedRectangle(cornerRadius: 20)
                                .stroke(isFailsafeWarningOnly ? Color.yellow.opacity(0.4) : Color.red.opacity(0.4), lineWidth: 1.5)
                        )
                    }
                }
            }
            .navigationTitle("DarkAI Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                    .foregroundColor(Theme.accentCyan)
                    .font(.system(size: 16, weight: .bold))
                }
            }
            .alert(isPresented: $showContextWarningPopup) {
                Alert(
                    title: Text("High Context Warning"),
                    message: Text("Based on your device memory and the loaded model size, setting the context limit above \(llmManager.safeContextLimit) tokens risks an Out-Of-Memory crash. Please consider lowering it."),
                    primaryButton: .default(Text("Revert to Safe Limit")) {
                        llmManager.contextTokenLimit = llmManager.safeContextLimit
                    },
                    secondaryButton: .destructive(Text("Ignore & Keep")) {
                        // User chooses to keep the limit
                    }
                )
            }
            .fileImporter(
                isPresented: $showModelImporter,
                allowedContentTypes: [.item],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    if let firstUrl = urls.first {
                        copyModelToAppDocuments(from: firstUrl)
                    }
                case .failure(let error):
                    print("Model import error: \(error.localizedDescription)")
                }
            }
            .fileImporter(
                isPresented: $showDocImporter,
                allowedContentTypes: [.item],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    if let firstUrl = urls.first {
                        ingestRAGDocument(from: firstUrl)
                    }
                case .failure(let error):
                    print("Document import error: \(error.localizedDescription)")
                }
            }
            .onAppear {
                refreshModelList()
            }
        }
    }
    
    // Custom View for rows of imported GGUF models
    @ViewBuilder
    private func modelRow(for url: URL) -> some View {
        let sizeGB = llmManager.getModelSizeGB(at: url)
        let safety = llmManager.checkMemorySafety(modelSizeGB: sizeGB)
        let isLoaded = isCurrentModel(url: url)
        
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(url.lastPathComponent)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(Theme.textPrimary)
                    .lineLimit(1)
                
                HStack(spacing: 8) {
                    Text(String(format: "%.2f GB", sizeGB))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(Theme.textSecondary)
                    
                    // Safety Label
                    safetyTag(for: safety)
                }
            }
            
            Spacer()
            
            if isLoaded {
                // Unload button when model is currently active
                Button(action: {
                    llmManager.unloadModel()
                }) {
                    HStack(spacing: 4) {
                        Image(systemName: "eject.fill")
                            .font(.system(size: 10))
                        Text("UNLOAD")
                            .font(.system(size: 11, weight: .bold))
                    }
                    .foregroundColor(.red.opacity(0.9))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Color.red.opacity(0.08))
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.red.opacity(0.35), lineWidth: 1)
                    )
                }
            } else {
                Button(action: {
                    handleModelSelection(url: url, sizeGB: sizeGB, safety: safety)
                }) {
                    Text("Load")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 6)
                        .background(Theme.border)
                        .cornerRadius(8)
                }
            }
        }
        .padding(10)
        .background(Theme.background.opacity(0.4))
        .cornerRadius(10)
    }
    
    @ViewBuilder
    private func safetyTag(for safety: MemorySafetyStatus) -> some View {
        switch safety {
        case .safe:
            Text("SAFE")
                .font(.system(size: 9, weight: .bold))
                .foregroundColor(.green)
                .padding(.horizontal, 5)
                .padding(.vertical, 2)
                .background(Color.green.opacity(0.1))
                .cornerRadius(4)
        case .warning:
            Text("WARNING")
                .font(.system(size: 9, weight: .bold))
                .foregroundColor(.yellow)
                .padding(.horizontal, 5)
                .padding(.vertical, 2)
                .background(Color.yellow.opacity(0.1))
                .cornerRadius(4)
        case .dangerous:
            Text("OOM DANGER")
                .font(.system(size: 9, weight: .bold))
                .foregroundColor(.red)
                .padding(.horizontal, 5)
                .padding(.vertical, 2)
                .background(Color.red.opacity(0.1))
                .cornerRadius(4)
        }
    }
    
    private func isCurrentModel(url: URL) -> Bool {
        if case let .loaded(name, _) = llmManager.loadState {
            return name == url.lastPathComponent
        }
        return false
    }
    
    private func handleModelSelection(url: URL, sizeGB: Double, safety: MemorySafetyStatus) {
        let totalRequired = sizeGB + 1.5 // model + context overhead
        
        switch safety {
        case .safe:
            llmManager.loadModel(at: url)
        case .warning:
            selectedModelToLoad = url
            failsafeRequiredRAM = totalRequired
            isFailsafeWarningOnly = true
            failsafeMessage = "The model '\(url.lastPathComponent)' is very large. Memory allocation is tight on this device. Sideloading entitlements allow high memory limits, but loading might cause background apps to close."
            showFailsafePopup = true
        case .dangerous:
            selectedModelToLoad = url
            failsafeRequiredRAM = totalRequired
            isFailsafeWarningOnly = false
            failsafeMessage = "The model '\(url.lastPathComponent)' exceeds the safety limits of this iPhone. Loading is highly likely to trigger iOS out-of-memory kernel termination."
            showFailsafePopup = true
        }
    }
    
    private func refreshModelList() {
        guard let docsUrl = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let modelsDir = docsUrl.appendingPathComponent("Models")
        
        // Ensure Models folder exists
        try? FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        
        do {
            let files = try FileManager.default.contentsOfDirectory(at: modelsDir, includingPropertiesForKeys: nil)
            // Only list real GGUF files — no mock seeds
            self.importedModels = files.filter { $0.pathExtension.lowercased() == "gguf" }
        } catch {
            print("Failed listing models: \(error.localizedDescription)")
        }
    }
    
    private func copyModelToAppDocuments(from sourceURL: URL) {
        // Must start accessing security scoped resource
        guard sourceURL.startAccessingSecurityScopedResource() else { return }
        defer { sourceURL.stopAccessingSecurityScopedResource() }
        
        guard let docsUrl = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let modelsDir = docsUrl.appendingPathComponent("Models")
        let destinationURL = modelsDir.appendingPathComponent(sourceURL.lastPathComponent)
        
        do {
            if FileManager.default.fileExists(atPath: destinationURL.path) {
                try FileManager.default.removeItem(at: destinationURL)
            }
            try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
            refreshModelList()
        } catch {
            print("Failed copying model file: \(error)")
        }
    }
    
    private func ingestRAGDocument(from sourceURL: URL) {
        guard sourceURL.startAccessingSecurityScopedResource() else { return }
        defer { sourceURL.stopAccessingSecurityScopedResource() }
        
        do {
            let content = try String(contentsOf: sourceURL, encoding: .utf8)
            ragManager.ingestDocument(name: sourceURL.lastPathComponent, content: content)
        } catch {
            print("Failed ingesting RAG document: \(error)")
        }
    }
}
