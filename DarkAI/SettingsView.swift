import SwiftUI
import UniformTypeIdentifiers

struct SettingsView: View {
    // stable-diffusion.cpp loads GGUF, safetensors, and legacy .ckpt checkpoints through the
    // same model_path — unlike the LLM chat path (LlamaSwift/llama.cpp), which is GGUF-only.
    static let acceptedDiffusionExtensions: Set<String> = ["gguf", "safetensors", "ckpt"]

    @Environment(\.dismiss) var dismiss
    @ObservedObject var llmManager: LLMManager
    @ObservedObject var memoryManager: MemoryManager
    @ObservedObject var ragManager: RAGManager
    @ObservedObject var personalityManager: PersonalityManager
    @ObservedObject var diffusionManager: DiffusionManager
    @ObservedObject var webSearchManager: WebSearchManager

    @Binding var customInstructions: String
    @Binding var enableRAG: Bool
    @Binding var enableMemories: Bool

    @StateObject private var downloads = ModelDownloadManager.shared
    @StateObject private var appearance = AppearanceManager.shared

    @State private var importedModels: [URL] = []
    @State private var storageUsedGB: Double = 0
    @State private var showModelImporter = false
    @State private var isImporting = false
    @State private var importProgress = ""
    @State private var showResetPersonalityAlert = false

    // Diffusion model import
    @State private var importedDiffusionModels: [URL] = []
    @State private var showDiffusionImporter = false
    @State private var isDiffusionImporting = false
    @State private var diffusionImportProgress = ""
    /// Why the last import was refused. Separate from `diffusionImportProgress` because that
    /// row is only shown while an import is in flight — see `handleDiffusionImport`.
    @State private var diffusionImportError: String? = nil

    // Context Warning
    @State private var showContextWarningPopup = false
    @State private var showInvalidFileTypeAlert = false

    // Failsafe Modal States
    // Shared by both pickers below — exactly one of `selectedModelToLoad` /
    // `pendingDiffusionModelToLoad` is set at a time, matching whichever picker triggered the
    // popup. The "Load Anyway" button checks both so it doesn't need to know which.
    @State private var showFailsafePopup = false
    @State private var selectedModelToLoad: URL? = nil
    @State private var pendingDiffusionModelToLoad: URL? = nil
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
                                Text("LOCAL LLM MODELS")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(Theme.textPrimary)
                                    .kerning(1.2)
                                Spacer()
                                Button(action: { showModelImporter = true }) {
                                    HStack(spacing: 6) {
                                        Image(systemName: "plus")
                                        Text("Import .gguf")
                                    }
                                    .font(.system(size: 12, weight: .bold))
                                    .foregroundColor(Theme.onAccent)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(
                                        RoundedRectangle(cornerRadius: 8)
                                            .fill(LinearGradient(colors: [Theme.accent, Theme.accentCyan], startPoint: .leading, endPoint: .trailing))
                                    )
                                }
                            }
                            
                            if importedModels.isEmpty {
                                Text("No models installed yet. Download one below, or import a .gguf file you already have.")
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

                            Divider().background(Theme.border)

                            downloadCatalogSection(for: .chat)

                            if isImporting {
                                HStack {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle(tint: Theme.accentCyan))
                                    Text(importProgress)
                                        .font(.system(size: 13, weight: .medium))
                                        .foregroundColor(Theme.textPrimary)
                                        .padding(.leading, 8)
                                    Spacer()
                                }
                                .padding()
                                .background(Theme.background)
                                .cornerRadius(12)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(Theme.accentCyan.opacity(0.3), lineWidth: 1)
                                )
                            }
                        }
                        .glassCard(cornerRadius: 16)

                        // SECTION 1b: Diffusion Model Manager
                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                Image(systemName: "photo.badge.plus")
                                    .foregroundColor(Color.purple)
                                    .font(.headline)
                                Text("DIFFUSION MODEL")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(Theme.textPrimary)
                                    .kerning(1.2)
                                Spacer()
                                Button(action: { showDiffusionImporter = true }) {
                                    HStack(spacing: 6) {
                                        Image(systemName: "plus")
                                        Text("Import Model")
                                    }
                                    .font(.system(size: 12, weight: .bold))
                                    .foregroundColor(Theme.onAccent)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(
                                        RoundedRectangle(cornerRadius: 8)
                                            .fill(LinearGradient(
                                                colors: [Color.purple, Color.purple.opacity(0.6)],
                                                startPoint: .leading, endPoint: .trailing))
                                    )
                                }
                            }

                            // Diffusion model list
                            if importedDiffusionModels.isEmpty {
                                Text("No diffusion model installed. Download one below, or import a .gguf or .safetensors checkpoint you already have.")
                                    .font(.system(size: 13))
                                    .foregroundColor(Theme.textSecondary)
                                    .padding()
                                    .frame(maxWidth: .infinity, alignment: .center)
                                    .glassCard(cornerRadius: 12)
                            } else {
                                VStack(spacing: 12) {
                                    ForEach(importedDiffusionModels, id: \.self) { url in
                                        diffusionModelRow(for: url)
                                    }
                                }
                            }

                            Divider().background(Theme.border)

                            downloadCatalogSection(for: .diffusion)

                            if isDiffusionImporting {
                                HStack {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle(tint: Color.purple))
                                    Text(diffusionImportProgress)
                                        .font(.system(size: 13, weight: .medium))
                                        .foregroundColor(Theme.textPrimary)
                                        .padding(.leading, 8)
                                    Spacer()
                                }
                                .padding()
                                .background(Theme.background)
                                .cornerRadius(12)
                                .overlay(RoundedRectangle(cornerRadius: 12)
                                    .stroke(Color.purple.opacity(0.3), lineWidth: 1))
                            }

                            // Import refusal. These messages explain *what kind* of file was
                            // picked (a LoRA, a bare VAE, a checkpoint with no VAE baked in),
                            // so they're given room to wrap rather than being truncated to a
                            // line — the explanation is the whole value.
                            if let importError = diffusionImportError {
                                VStack(alignment: .leading, spacing: 10) {
                                    HStack(alignment: .top, spacing: 8) {
                                        Image(systemName: "exclamationmark.triangle.fill")
                                            .foregroundColor(.orange)
                                            .font(.system(size: 14))
                                        Text("Can't use this file")
                                            .font(.system(size: 13, weight: .bold))
                                            .foregroundColor(Theme.textPrimary)
                                        Spacer()
                                        Button {
                                            diffusionImportError = nil
                                        } label: {
                                            Image(systemName: "xmark")
                                                .font(.system(size: 12, weight: .bold))
                                                .foregroundColor(Theme.textSecondary)
                                        }
                                    }
                                    Text(importError)
                                        .font(.system(size: 12))
                                        .foregroundColor(Theme.textSecondary)
                                        .fixedSize(horizontal: false, vertical: true)
                                        .lineSpacing(3)
                                }
                                .padding()
                                .background(Color.orange.opacity(0.12))
                                .cornerRadius(12)
                                .overlay(RoundedRectangle(cornerRadius: 12)
                                    .stroke(Color.orange.opacity(0.45), lineWidth: 1))
                            }

                            // Current diffusion load state
                            switch diffusionManager.diffusionLoadState {
                            case .loading(_, let status):
                                HStack(spacing: 8) {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle(tint: Color.purple))
                                        .scaleEffect(0.8)
                                    Text(status)
                                        .font(.system(size: 12, design: .monospaced))
                                        .foregroundColor(Theme.textSecondary)
                                }
                            case .failed(let err):
                                Text("Error: \(err)")
                                    .font(.system(size: 12))
                                    .foregroundColor(.red.opacity(0.8))
                            default:
                                EmptyView()
                            }

                            Divider().background(Theme.border)

                            // Generation settings sub-section
                            Text("IMAGE GEN SETTINGS")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundColor(Theme.textSecondary)
                                .kerning(1.0)

                            // Steps
                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Text("Inference Steps:")
                                        .font(.system(size: 13))
                                        .foregroundColor(Theme.textSecondary)
                                    Spacer()
                                    Text("\(diffusionManager.steps)")
                                        .font(.system(size: 13, weight: .bold, design: .monospaced))
                                        .foregroundColor(Color.purple)
                                }
                                Slider(value: Binding(
                                    get: { Double(diffusionManager.steps) },
                                    set: { diffusionManager.steps = Int($0) }
                                ), in: 4...50, step: 1)
                                .accentColor(Color.purple)
                            }

                            Divider().background(Theme.border)

                            // CFG Scale
                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Text("CFG Scale (Prompt Strength):")
                                        .font(.system(size: 13))
                                        .foregroundColor(Theme.textSecondary)
                                    Spacer()
                                    Text(String(format: "%.1f", diffusionManager.cfgScale))
                                        .font(.system(size: 13, weight: .bold, design: .monospaced))
                                        .foregroundColor(Color.purple)
                                }
                                Slider(value: $diffusionManager.cfgScale, in: 1.0...12.0, step: 0.5)
                                    .accentColor(Color.purple)
                            }

                            Divider().background(Theme.border)

                            // Output resolution
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Output Resolution:")
                                    .font(.system(size: 13))
                                    .foregroundColor(Theme.textSecondary)
                                HStack(spacing: 8) {
                                    ForEach([256, 512, 768], id: \.self) { size in
                                        Button(action: { diffusionManager.outputSize = size }) {
                                            Text("\(size)")
                                                .font(.system(size: 12, weight: .bold, design: .monospaced))
                                                .foregroundColor(diffusionManager.outputSize == size ? .white : Theme.textSecondary)
                                                .padding(.horizontal, 12)
                                                .padding(.vertical, 6)
                                                .background(
                                                    RoundedRectangle(cornerRadius: 8)
                                                        .fill(diffusionManager.outputSize == size
                                                            ? Color.purple
                                                            : Theme.border.opacity(0.4))
                                                )
                                        }
                                    }
                                }
                            }
                        }
                        .glassCard(cornerRadius: 16)
                        .fileImporter(
                            isPresented: $showDiffusionImporter,
                            allowedContentTypes: [UTType.data],
                            allowsMultipleSelection: false
                        ) { result in
                            handleDiffusionImport(result)
                        }
                        .onAppear { loadDiffusionModels() }

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
                                Text("LLM SETTINGS")
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

                                // The slider's upper bound is the device ceiling, not a fixed
                                // 32768. On a 4 GB phone the old slider let you dial in a number
                                // that was quietly cut to a quarter of itself at load time,
                                // which just made the setting a lie.
                                let ceiling = llmManager.deviceContextCeiling
                                Slider(value: Binding(
                                    get: { Double(min(llmManager.contextTokenLimit, ceiling)) },
                                    set: { newValue in
                                        let val = min(Int(newValue), ceiling)
                                        llmManager.contextTokenLimit = val
                                        llmManager.contextLimitAutoAdjustedTo = nil
                                        if val > llmManager.safeContextLimit {
                                            showContextWarningPopup = true
                                        }
                                    }
                                ), in: 512...Double(ceiling), step: 256)
                                .accentColor(Theme.accent)

                                if let adjusted = llmManager.contextLimitAutoAdjustedTo {
                                    HStack(alignment: .top, spacing: 6) {
                                        Image(systemName: "info.circle.fill")
                                            .font(.system(size: 11))
                                            .foregroundColor(.orange)
                                        Text("Lowered to \(adjusted) tokens to fit this device's \(String(format: "%.0f", llmManager.systemMemoryGB)) GB of memory. You can still set it lower.")
                                            .font(.system(size: 11))
                                            .foregroundColor(.orange)
                                            .fixedSize(horizontal: false, vertical: true)
                                    }
                                } else {
                                    Text("Maximum for this device: \(ceiling) tokens.")
                                        .font(.system(size: 11))
                                        .foregroundColor(Theme.textMuted)
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
                                    if llmManager.highVariabilityEnabled {
                                        Text("HIGH (2.50)")
                                            .font(.system(size: 13, weight: .bold, design: .monospaced))
                                            .foregroundColor(.orange)
                                    } else {
                                        Text(String(format: "%.2f", llmManager.temperature))
                                            .font(.system(size: 13, weight: .bold, design: .monospaced))
                                            .foregroundColor(Theme.accent)
                                    }
                                }
                                
                                Slider(value: $llmManager.temperature, in: 0.0...2.0, step: 0.05)
                                    .accentColor(llmManager.highVariabilityEnabled ? .gray : Theme.accent)
                                    .disabled(llmManager.highVariabilityEnabled)
                                    .opacity(llmManager.highVariabilityEnabled ? 0.5 : 1.0)
                                    
                                Toggle(isOn: $llmManager.highVariabilityEnabled) {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text("High Variability")
                                            .font(.system(size: 13))
                                            .foregroundColor(llmManager.highVariabilityEnabled ? .orange : Theme.textSecondary)
                                        Text("Overrides temperature with a very high value. Output becomes far less predictable.")
                                            .font(.system(size: 11))
                                            .foregroundColor(Theme.textMuted)
                                    }
                                }
                                .toggleStyle(SwitchToggleStyle(tint: .orange))
                            }
                        }
                        .glassCard(cornerRadius: 16)
                        
                        // SECTION 3: RAG Document Manager
                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                Image(systemName: "doc.text.magnifyingglass")
                                    .foregroundColor(Theme.accentCyan)
                                Text("RAG CONFIG")
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
                                NavigationLink(destination: MindscapeView(ragManager: ragManager)) {
                                    HStack {
                                        Image(systemName: "brain.filled.head.profile")
                                            .foregroundColor(Theme.accentCyan)
                                        Text("Open Mindscape")
                                            .font(.system(size: 14, weight: .bold))
                                            .foregroundColor(Theme.textPrimary)
                                        Spacer()
                                        Image(systemName: "chevron.right")
                                            .foregroundColor(Theme.textSecondary)
                                            .font(.system(size: 12))
                                    }
                                    .padding(.vertical, 8)
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
                                    Text("No memories extracted yet. Tell DarkAI things like 'I prefer Python' or 'My name is John' to build long-term memory.")
                                        .font(.system(size: 13))
                                        .foregroundColor(Theme.textMuted)
                                        .padding(.vertical, 8)
                                } else {
                                    VStack(alignment: .leading, spacing: 8) {
                                        ForEach(Array(memoryManager.memories.enumerated()), id: \.element.id) { index, memory in
                                            memoryRow(memory, index: index)
                                        }
                                    }
                                }
                            }
                        }
                        .glassCard(cornerRadius: 16)

                        // SECTION 5: Internet Access
                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                Image(systemName: "globe")
                                    .foregroundColor(Theme.accentCyan)
                                Text("INTERNET ACCESS")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(Theme.textPrimary)
                                    .kerning(1.2)
                                Spacer()
                                Toggle("", isOn: $webSearchManager.isEnabled)
                                    .toggleStyle(SwitchToggleStyle(tint: Theme.accentCyan))
                                    .labelsHidden()
                            }

                            Text("Off by default — \(AppInfo.displayName) never uses the internet on its own. When on, the assistant will *ask* before searching for anything (like current weather or recent events); it never searches automatically. Only your search text is sent out, to Open-Meteo, DuckDuckGo, and — if you add a key below — Brave. No account, no identifiers.")
                                .font(.system(size: 12))
                                .foregroundColor(Theme.textSecondary)
                                .lineSpacing(3)

                            if webSearchManager.isEnabled {
                                Divider().background(Theme.border)

                                Text("SEARCH API KEY (OPTIONAL)")
                                    .font(.system(size: 11, weight: .semibold))
                                    .foregroundColor(Theme.textSecondary)

                                Text("Weather, and factual questions Wikipedia can answer, already work for free with no setup. A Brave Search API key adds real web search — needed for recent news, live scores, prices, and anything else that changes day to day.")
                                    .font(.system(size: 12))
                                    .foregroundColor(Theme.textMuted)

                                SecureField("Brave Search API key", text: $webSearchManager.braveAPIKey)
                                    .font(.system(size: 13, design: .monospaced))
                                    .padding(10)
                                    .background(Theme.background.opacity(0.4))
                                    .cornerRadius(8)
                                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Theme.border, lineWidth: 1))
                                    .autocorrectionDisabled()
                                    #if os(iOS)
                                    .textInputAutocapitalization(.never)
                                    #endif

                                HStack(spacing: 6) {
                                    Image(systemName: webSearchManager.hasBraveKey ? "checkmark.circle.fill" : "info.circle")
                                        .foregroundColor(webSearchManager.hasBraveKey ? .green : Theme.textMuted)
                                        .font(.system(size: 11))
                                    Text(webSearchManager.hasBraveKey
                                         ? "Key saved on this device (Keychain) — full web search is active."
                                         : "No key set — searches use the free weather and Wikipedia lookups, which can't answer questions about recent events.")
                                        .font(.system(size: 11))
                                        .foregroundColor(Theme.textMuted)
                                }

                                Link(destination: URL(string: "https://brave.com/search/api/")!) {
                                    HStack(spacing: 4) {
                                        Text("Get a Brave Search API key")
                                        Image(systemName: "arrow.up.right")
                                    }
                                    .font(.system(size: 12, weight: .semibold))
                                    .foregroundColor(Theme.accentCyan)
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
                                    .foregroundColor(Theme.textPrimary)
                                Spacer()
                                if personalityManager.isMature {
                                    Text("[ADAPTED]")
                                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                                        .foregroundColor(Theme.onAccent)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 4)
                                        .background(Theme.accentRose)
                                        .cornerRadius(6)
                                } else {
                                    Text("[LEARNING...]")
                                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                                        .foregroundColor(Theme.textSecondary)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 4)
                                        .background(Theme.textSecondary.opacity(0.2))
                                        .cornerRadius(6)
                                }
                                
                                Text(personalityManager.databaseSizeString)
                                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                                    .foregroundColor(Theme.accentRose)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(Theme.accentRose.opacity(0.15))
                                    .cornerRadius(6)
                            }
                            
                            Text("\(AppInfo.displayName) gradually adapts its tone to how you write. Everything it learns stays on this device. Resetting erases all learned traits for the currently loaded model.")
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
                                .foregroundColor(Theme.textPrimary)
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
                        
                        // SECTION 6: Diagnostics
                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                Image(systemName: "ladybug")
                                    .foregroundColor(.yellow)
                                Text("DIAGNOSTICS LOGS")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(Theme.textPrimary)
                                    .kerning(1.2)
                                Spacer()
                            }
                            
                            Divider().background(Theme.border)
                            
                            NavigationLink {
                                LogExportView()
                            } label: {
                                HStack {
                                    Text("View & Export Logs")
                                        .font(.system(size: 13, weight: .medium))
                                        .foregroundColor(Theme.textPrimary)
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .font(.system(size: 12))
                                        .foregroundColor(Theme.textSecondary)
                                }
                                .padding()
                                .background(Theme.cardBackground)
                                .cornerRadius(12)
                            }
                        }
                        .glassCard(cornerRadius: 16)
                        
                        // SECTION: Appearance
                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                Image(systemName: "paintbrush.fill")
                                    .foregroundColor(Theme.accentCyan)
                                Text("APPEARANCE")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(Theme.textPrimary)
                                    .kerning(1.2)
                                Spacer()
                            }

                            VStack(spacing: 0) {
                                ForEach(AppearanceMode.allCases) { mode in
                                    Button {
                                        withAnimation(.easeInOut(duration: 0.2)) {
                                            appearance.mode = mode
                                        }
                                    } label: {
                                        HStack(spacing: 12) {
                                            Image(systemName: mode.icon)
                                                .foregroundColor(appearance.mode == mode ? Theme.accent : Theme.textSecondary)
                                                .frame(width: 22)
                                            VStack(alignment: .leading, spacing: 2) {
                                                Text(mode.title)
                                                    .font(.system(size: 14, weight: .medium))
                                                    .foregroundColor(Theme.textPrimary)
                                                Text(mode.subtitle)
                                                    .font(.system(size: 11))
                                                    .foregroundColor(Theme.textMuted)
                                            }
                                            Spacer()
                                            Image(systemName: appearance.mode == mode ? "largecircle.fill.circle" : "circle")
                                                .foregroundColor(appearance.mode == mode ? Theme.accent : Theme.textMuted)
                                        }
                                        .padding(.vertical, 11)
                                        .contentShape(Rectangle())
                                    }
                                    if mode != AppearanceMode.allCases.last {
                                        Divider().background(Theme.border)
                                    }
                                }
                            }

                            Text("The app icon changes to match — light mode uses a light icon on your Home Screen.")
                                .font(.system(size: 11))
                                .foregroundColor(Theme.textMuted)
                                .lineSpacing(3)
                        }
                        .glassCard(cornerRadius: 16)

                        // SECTION 7: Safety, legal & storage
                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                Image(systemName: "checkmark.shield.fill")
                                    .foregroundColor(.green)
                                Text("SAFETY & LEGAL")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(Theme.textPrimary)
                                    .kerning(1.2)
                                Spacer()
                            }

                            Divider().background(Theme.border)

                            NavigationLink {
                                SafetyLegalView()
                            } label: {
                                HStack {
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text("Content Policy, Terms & Reporting")
                                            .font(.system(size: 13, weight: .medium))
                                            .foregroundColor(Theme.textPrimary)
                                        Text("Filter is always on · Report generated content")
                                            .font(.system(size: 11))
                                            .foregroundColor(Theme.textMuted)
                                    }
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .font(.system(size: 12))
                                        .foregroundColor(Theme.textSecondary)
                                }
                                .padding()
                                .background(Theme.cardBackground)
                                .cornerRadius(12)
                            }

                            Divider().background(Theme.border)

                            HStack {
                                Text("Storage used by \(AppInfo.displayName)")
                                    .font(.system(size: 13))
                                    .foregroundColor(Theme.textSecondary)
                                Spacer()
                                Text(String(format: "%.2f GB", storageUsedGB))
                                    .font(.system(size: 13, weight: .bold, design: .monospaced))
                                    .foregroundColor(Theme.accentCyan)
                            }

                            Text("Models, generated images, and logs are stored on this device only and are excluded from iCloud backup. Deleting the app removes all of it.")
                                .font(.system(size: 11))
                                .foregroundColor(Theme.textMuted)
                                .lineSpacing(3)
                        }
                        .glassCard(cornerRadius: 16)

                    }
                    .padding()
                }
                
                // Loading Model Overlay
                if case let .loading(progress, status) = llmManager.loadState {
                    ZStack {
                        Color.black.opacity(0.45)
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
                        Color.black.opacity(0.5)
                            .ignoresSafeArea()
                        
                        VStack(spacing: 20) {
                            Image(systemName: isFailsafeWarningOnly ? "exclamationmark.triangle" : "xmark.octagon")
                                .font(.system(size: 48))
                                .foregroundColor(isFailsafeWarningOnly ? .yellow : .red)
                                .neonGlow(color: isFailsafeWarningOnly ? .yellow : .red, radius: 10)
                            
                            Text(isFailsafeWarningOnly ? "MEMORY ALLOCATION WARNING" : "MEMORY FAILSAFE TRIGGERED")
                                .font(.system(size: 16, weight: .bold))
                                .foregroundColor(Theme.textPrimary)
                                .kerning(1.2)
                            
                            Text(failsafeMessage)
                                .font(.system(size: 13))
                                .foregroundColor(Theme.textSecondary)
                                .multilineTextAlignment(.center)
                                .lineSpacing(4)
                            
                            VStack(spacing: 8) {
                                HStack {
                                    Text("This device's RAM:")
                                        .foregroundColor(Theme.textMuted)
                                    Spacer()
                                    Text(String(format: "%.1f GB", llmManager.systemMemoryGB))
                                        .foregroundColor(Theme.textPrimary)
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
                            
                            // A model in the `.dangerous` band has no "load anyway" button.
                            // Overriding it reliably ends in a jetsam kill, and an app that
                            // hands the user a button whose documented outcome is termination
                            // is failing Guideline 2.1 — the correct answer is to refuse and
                            // point at a model that fits.
                            HStack(spacing: 16) {
                                Button(action: {
                                    showFailsafePopup = false
                                    selectedModelToLoad = nil
                                    pendingDiffusionModelToLoad = nil
                                }) {
                                    Text(isFailsafeWarningOnly ? "Cancel" : "OK")
                                        .font(.system(size: 14, weight: .bold))
                                        .foregroundColor(Theme.textPrimary)
                                        .frame(maxWidth: .infinity)
                                        .padding(.vertical, 12)
                                        .background(
                                            RoundedRectangle(cornerRadius: 10)
                                                .stroke(Theme.border, lineWidth: 1.5)
                                        )
                                }

                                if isFailsafeWarningOnly {
                                    Button(action: {
                                        if let url = selectedModelToLoad {
                                            llmManager.loadModel(at: url, forceLoad: true)
                                        } else if let url = pendingDiffusionModelToLoad {
                                            diffusionManager.lastDiffusionModelPath = url.path
                                        }
                                        showFailsafePopup = false
                                        selectedModelToLoad = nil
                                        pendingDiffusionModelToLoad = nil
                                    }) {
                                        Text("Load Anyway")
                                            .font(.system(size: 14, weight: .bold))
                                            .foregroundColor(.black)
                                            .frame(maxWidth: .infinity)
                                            .padding(.vertical, 12)
                                            .background(
                                                RoundedRectangle(cornerRadius: 10)
                                                    .fill(Color.yellow.opacity(0.8))
                                            )
                                    }
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
                    title: Text("High RAM Usage Warning"),
                    message: Text("You have set the context window higher than the safe limit for your device's available memory.\n\nThis dramatically increases the chance iOS will kill the app (crash) due to memory pressure.\n\nAre you sure you want to proceed?"),
                    primaryButton: .default(Text("Revert to Safe Limit")) {
                        llmManager.contextTokenLimit = llmManager.safeContextLimit
                    },
                    secondaryButton: .destructive(Text("Ignore & Keep")) {
                        // User chooses to keep the limit
                    }
                )
            }
            .alert("Invalid File", isPresented: $showInvalidFileTypeAlert) {
                Button("OK", role: .cancel) { }
            } message: {
                Text("Please select a valid .gguf model file.")
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

            .onAppear {
                refreshModelList()
            }
        }
    }
    
    /// One remembered fact. Split out of the enclosing view because the whole settings body was
    /// a single expression the type-checker gave up on once this row grew a badge.
    @ViewBuilder
    private func memoryRow(_ memory: Memory, index: Int) -> some View {
        HStack(spacing: 8) {
            Text(memoryKindLabel(memory.kind))
                .font(.system(size: 9, weight: .bold))
                .foregroundColor(memoryKindColor(memory.kind))
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .background(memoryKindColor(memory.kind).opacity(0.15))
                .cornerRadius(4)

            Text(memory.text)
                .font(.system(size: 13))
                .foregroundColor(Theme.textPrimary)
                .lineLimit(2)

            Spacer(minLength: 4)

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

    private func memoryKindLabel(_ kind: Memory.Kind) -> String {
        switch kind {
        case .identity:   return "YOU"
        case .preference: return "PREF"
        case .intent:     return "PLAN"
        case .event:      return "EVENT"
        }
    }

    private func memoryKindColor(_ kind: Memory.Kind) -> Color {
        switch kind {
        case .identity:   return Theme.accent
        case .preference: return Theme.accentCyan
        case .intent:     return Theme.accentRose
        case .event:      return Theme.textMuted
        }
    }

    // MARK: - Download catalog

    /// Curated, developer-vetted models the user can fetch without leaving the app.
    ///
    /// This is what keeps the app usable on a fresh install. Before it existed the only way to
    /// get a model in was to find a `.gguf` elsewhere and side-load it through Files — which is
    /// fine for the person who built the app and useless to everyone else, reviewers included.
    @ViewBuilder
    private func downloadCatalogSection(for kind: ModelKind) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(kind == .chat ? "DOWNLOAD A CHAT MODEL" : "DOWNLOAD A DIFFUSION MODEL")
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(Theme.textSecondary)
                .kerning(1.0)

            // A single download runs at a time, so the in-progress card is shown by whichever
            // section owns the active transfer — showing it in both would imply two downloads.
            if let progress = downloads.active, downloads.activeKind == kind {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Downloading…")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(Theme.textPrimary)
                    ProgressView(value: progress.fractionCompleted)
                        .progressViewStyle(LinearProgressViewStyle(tint: Theme.accent))
                    HStack {
                        Text("\(progress.writtenDescription) of \(progress.totalDescription)")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(Theme.textSecondary)
                        Spacer()
                        Button("Cancel") { downloads.cancel() }
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundColor(.red.opacity(0.9))
                    }
                }
                .padding(12)
                .background(Theme.cardBackground)
                .cornerRadius(10)
            } else {
                ForEach(ModelCatalog.models(for: kind)) { model in
                    catalogRow(model)
                }

                if let error = downloads.lastError, downloads.active == nil {
                    Text(error)
                        .font(.system(size: 12))
                        .foregroundColor(.orange)
                }

                Toggle(isOn: $downloads.allowsCellularDownload) {
                    Text("Allow downloads over cellular")
                        .font(.system(size: 12))
                        .foregroundColor(Theme.textSecondary)
                }
                .toggleStyle(SwitchToggleStyle(tint: Theme.accent))
            }
        }
        .onChange(of: downloads.lastCompletedModelID) { _, newValue in
            guard newValue != nil else { return }
            refreshModelList()
            loadDiffusionModels()
        }
    }

    @ViewBuilder
    private func catalogRow(_ model: CatalogModel) -> some View {
        let installed = downloads.isInstalled(model)
        // Warn before spending a gigabyte of data on something this device can't run. The
        // check is against total RAM rather than current headroom, since that's the ceiling
        // that won't change by closing something.
        let fitsOnDevice = model.approxRuntimeGB < llmManager.systemMemoryGB * 0.75

        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top, spacing: 10) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(model.displayName)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(Theme.textPrimary)
                    Text("\(model.sizeDescription) · \(model.parameterCount) · \(model.quantization) · \(model.license)")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(Theme.textMuted)
                    Text(model.summary)
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                    if let attribution = model.attribution {
                        Text(attribution)
                            .font(.system(size: 9))
                            .foregroundColor(Theme.textMuted)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    if !fitsOnDevice {
                        Text("May not run on this device — needs about \(String(format: "%.1f", model.approxRuntimeGB)) GB of memory.")
                            .font(.system(size: 10))
                            .foregroundColor(.orange)
                    }
                }
                Spacer(minLength: 6)

                if installed {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                        .font(.system(size: 18))
                } else {
                    Button {
                        downloads.download(model)
                    } label: {
                        Text("Get")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(Theme.onAccent)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 6)
                            .background(RoundedRectangle(cornerRadius: 8).fill(Theme.accent))
                    }
                    .disabled(downloads.isDownloading)
                    .opacity(downloads.isDownloading ? 0.4 : 1)
                }
            }
        }
        .padding(11)
        .background(Theme.background.opacity(0.4))
        .cornerRadius(10)
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Theme.border, lineWidth: 1))
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
            
            HStack(spacing: 8) {
                if !isLoaded {
                    Button(action: { deleteModel(at: url, isDiffusion: false) }) {
                        Image(systemName: "trash")
                            .font(.system(size: 12))
                            .foregroundColor(Theme.textSecondary)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(Theme.border)
                            .cornerRadius(8)
                    }
                }
            
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
                    handleModelSelection(url: url, safety: safety)
                }) {
                    Text("Load")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundColor(Theme.textPrimary)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 6)
                        .background(Theme.border)
                        .cornerRadius(8)
                }
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
    
    private func handleModelSelection(url: URL, safety: MemorySafetyStatus) {
        switch safety {
        case .safe:
            llmManager.loadModel(at: url)
        case .warning(let requiredGB, _):
            selectedModelToLoad = url
            pendingDiffusionModelToLoad = nil
            failsafeRequiredRAM = requiredGB
            isFailsafeWarningOnly = true
            failsafeMessage = "'\(url.lastPathComponent)' is large for this device. It should load, but memory will be tight and other apps may be closed in the background to make room."
            showFailsafePopup = true
        case .dangerous(let requiredGB, _):
            selectedModelToLoad = nil
            pendingDiffusionModelToLoad = nil
            failsafeRequiredRAM = requiredGB
            isFailsafeWarningOnly = false
            failsafeMessage = "'\(url.lastPathComponent)' needs more memory than this device can give it, so it can't be loaded — attempting it would end with iOS terminating the app. Try a smaller model, or a more compressed version of this one."
            showFailsafePopup = true
        }
    }

    /// Mirrors `handleModelSelection` above for the diffusion picker. Selecting a diffusion
    /// model only ever records a path — the actual load (and its own, independent hard gate)
    /// happens lazily from the image-generation flow in `ContentView` — so this is an advisory
    /// warning at selection time, same as the LLM picker's is advisory at "Load" time.
    private func handleDiffusionModelSelection(url: URL, safety: MemorySafetyStatus) {
        let res = diffusionManager.outputSize
        switch safety {
        case .safe:
            diffusionManager.lastDiffusionModelPath = url.path
        case .warning(let requiredGB, _):
            selectedModelToLoad = nil
            pendingDiffusionModelToLoad = url
            failsafeRequiredRAM = requiredGB
            isFailsafeWarningOnly = true
            failsafeMessage = "'\(url.lastPathComponent)' is large for this device at the current output resolution (\(res)×\(res)px). It should work, but memory will be tight and other apps may be closed in the background to make room."
            showFailsafePopup = true
        case .dangerous(let requiredGB, _):
            selectedModelToLoad = nil
            pendingDiffusionModelToLoad = nil
            failsafeRequiredRAM = requiredGB
            isFailsafeWarningOnly = false
            failsafeMessage = "'\(url.lastPathComponent)' needs more memory than this device can give it at the current output resolution (\(res)×\(res)px), so it can't be used — attempting it would end with iOS terminating the app. Try a smaller checkpoint, or a lower Output Resolution."
            showFailsafePopup = true
        }
    }
    
    private func refreshModelList() {
        importedModels = AppFiles.contents(of: AppFiles.models, matchingExtensions: ["gguf"])
        storageUsedGB = AppFiles.totalUsedGB()
    }
    
    private func deleteModel(at url: URL, isDiffusion: Bool) {
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
            if isDiffusion {
                loadDiffusionModels()
            } else {
                refreshModelList()
            }
        } catch {
            print("Failed to delete model: \(error.localizedDescription)")
        }
    }
    
    private func copyModelToAppDocuments(from sourceURL: URL) {
        guard sourceURL.startAccessingSecurityScopedResource() else { return }

        guard sourceURL.pathExtension.lowercased() == "gguf" else {
            sourceURL.stopAccessingSecurityScopedResource()
            showInvalidFileTypeAlert = true
            return
        }

        AppFiles.prepare()
        let destinationURL = AppFiles.models.appendingPathComponent(sourceURL.lastPathComponent)

        isImporting = true
        importProgress = "Checking space..."

        Task.detached(priority: .background) {
            defer {
                sourceURL.stopAccessingSecurityScopedResource()
            }

            do {
                let fileSizeGB = AppFiles.fileSizeGB(at: sourceURL)
                let freeGB = AppFiles.availableDiskGB()

                if fileSizeGB + 0.5 > freeGB {
                    await MainActor.run {
                        importProgress = String(format: "Not enough storage — needs %.1f GB, %.1f GB free.", fileSizeGB + 0.5, freeGB)
                    }
                    return
                }

                await MainActor.run { importProgress = "Copying (keep the app open)..." }

                if FileManager.default.fileExists(atPath: destinationURL.path) {
                    try FileManager.default.removeItem(at: destinationURL)
                }
                try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
                AppFiles.excludeFromBackup(destinationURL)

                await MainActor.run {
                    refreshModelList()
                    self.isImporting = false
                }
            } catch {
                await MainActor.run {
                    // Left visible rather than cleared, so the failure doesn't vanish before
                    // the user can read it.
                    importProgress = "Import failed: \(error.localizedDescription)"
                }
                LogManager.shared.log("Model import failed: \(error.localizedDescription)")
            }
        }
    }
    

    // MARK: - Diffusion Model Helpers

    private func diffusionModelRow(for url: URL) -> some View {
        let sizeGB = diffusionManager.getFileSizeGB(at: url)
        let isSelected = diffusionManager.lastDiffusionModelPath == url.path
        // Live, resolution-aware — re-evaluates if the user changes Output Resolution above,
        // same as `modelRow`'s safety tag re-evaluates against the LLM's context setting.
        // Judged on resident size, which for an FP8 checkpoint is double the file size.
        let residentGB = diffusionManager.effectiveWeightSizeGB(at: url)
        let expands = diffusionManager.weightsExpandOnLoad(at: url)
        let safety = diffusionManager.checkMemorySafety(modelSizeGB: residentGB, outputSize: diffusionManager.outputSize)

        return HStack(spacing: 12) {
            Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                .font(.system(size: 14))
                .foregroundColor(isSelected ? .purple : Theme.textSecondary)
                .frame(width: 24)

            VStack(alignment: .leading, spacing: 3) {
                Text(url.lastPathComponent)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(Theme.textPrimary)
                    .lineLimit(1)
                HStack(spacing: 8) {
                    Text(String(format: "%.2f GB", sizeGB))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(Theme.textSecondary)

                    safetyTag(for: safety)
                }
                // Without this a 4 GB file refused on a device showing 8 GB free looks like a
                // bug in the check rather than a property of the file.
                if expands {
                    Text(String(format: "8-bit weights — needs ~%.1f GB in memory", residentGB))
                        .font(.system(size: 10))
                        .foregroundColor(.orange)
                }
            }

            Spacer()

            HStack(spacing: 8) {
                if !isSelected {
                    Button(action: { deleteModel(at: url, isDiffusion: true) }) {
                        Image(systemName: "trash")
                            .font(.system(size: 12))
                            .foregroundColor(Theme.textSecondary)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(Theme.cardBackground)
                            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.border, lineWidth: 1))
                    }
                }
            
                if isSelected {
                    Text("SELECTED")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(.purple)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .overlay(RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.purple.opacity(0.5), lineWidth: 1))
                } else {
                    Button(action: { handleDiffusionModelSelection(url: url, safety: safety) }) {
                        HStack(spacing: 4) {
                            Text("SELECT")
                        }
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(Theme.onAccent)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(LinearGradient(
                                    colors: [Color.purple, Color.purple.opacity(0.7)],
                                    startPoint: .leading, endPoint: .trailing))
                        )
                    }
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .background(Theme.cardBackground)
        .cornerRadius(10)
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Theme.border, lineWidth: 1))
    }

    private func handleDiffusionImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let sourceURL = urls.first else { return }
            // stable-diffusion.cpp auto-detects the actual weight format (GGUF, safetensors,
            // or legacy checkpoint) from file content, not extension — SDWrapper.loadModel
            // passes whatever path it's given straight through. This is just a sanity filter
            // on what the picker will offer to import in the first place.
            guard Self.acceptedDiffusionExtensions.contains(sourceURL.pathExtension.lowercased()) else { return }
            isDiffusionImporting = true
            diffusionImportError = nil
            diffusionImportProgress = "Checking \(sourceURL.lastPathComponent)…"
            Task {
                let isSecured = sourceURL.startAccessingSecurityScopedResource()
                defer { if isSecured { sourceURL.stopAccessingSecurityScopedResource() } }
                AppFiles.prepare()
                let destURL = AppFiles.diffusionModels.appendingPathComponent(sourceURL.lastPathComponent)
                do {
                    // Reject an unusable checkpoint before spending a multi-gigabyte copy on it,
                    // and before it can reach the loader and abort the process.
                    try GGUFValidator.validateDiffusionCheckpoint(path: sourceURL.path)

                    if FileManager.default.fileExists(atPath: destURL.path) {
                        try FileManager.default.removeItem(at: destURL)
                    }
                    try FileManager.default.copyItem(at: sourceURL, to: destURL)
                    AppFiles.excludeFromBackup(destURL)
                    await MainActor.run { isDiffusionImporting = false; loadDiffusionModels() }
                } catch {
                    await MainActor.run {
                        isDiffusionImporting = false
                        // Surfaced through its own state, not `diffusionImportProgress` — that
                        // row only renders while `isDiffusionImporting` is true, so writing the
                        // failure there and clearing the flag in the same breath meant a
                        // rejected import showed the user nothing at all and looked like the
                        // button simply hadn't worked.
                        diffusionImportError = error.localizedDescription
                        LogManager.shared.log("Diffusion import rejected: \(error.localizedDescription)")
                    }
                }
            }
        case .failure(let error):
            diffusionImportError = error.localizedDescription
            LogManager.shared.log("Diffusion import failed: \(error.localizedDescription)")
        }
    }

    private func loadDiffusionModels() {
        importedDiffusionModels = AppFiles.contents(
            of: AppFiles.diffusionModels,
            matchingExtensions: Self.acceptedDiffusionExtensions
        )
        storageUsedGB = AppFiles.totalUsedGB()
    }

}

struct LogExportView: View {
    @ObservedObject var logManager = LogManager.shared

    var body: some View {
        VStack {
            ScrollView {
                Text(logManager.logs.joined(separator: "\n"))
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(.green)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
            .background(Theme.cardBackground)
            .cornerRadius(10)
            
            Spacer()
            
            HStack {
                Button(action: {
                    logManager.clearLogs()
                }) {
                    Label("Clear Logs", systemImage: "trash")
                        .font(.headline)
                        .foregroundColor(Theme.onAccent)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.red)
                        .cornerRadius(15)
                }
                
                ShareLink(item: logManager.getLogFileURL()) {
                    Label("Export Logs", systemImage: "square.and.arrow.up")
                        .font(.headline)
                        .foregroundColor(Theme.onAccent)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Theme.accent)
                        .cornerRadius(15)
                }
            }
            .padding()
        }
        .padding()
        .background(Theme.background.ignoresSafeArea())
        .navigationTitle("Diagnostic Logs")
        .navigationBarTitleDisplayMode(.inline)
    }
}

/// Shared "Export" pill used by both Mindscape entry types.
private struct MindscapeExportLabel: View {
    var body: some View {
        HStack {
            Image(systemName: "square.and.arrow.up")
            Text("Export")
        }
        .font(.system(size: 12, weight: .bold))
        .foregroundColor(Theme.onAccent)
        .padding(.vertical, 6)
        .padding(.horizontal, 12)
        .background(Theme.accent)
        .cornerRadius(8)
    }
}

/// Content for a Mindscape entry backed by an actual generated image.
private struct MindscapeImageContent: View {
    let doc: RAGDocument
    let uiImage: UIImage
    let imageURL: URL

    var body: some View {
        Image(uiImage: uiImage)
            .resizable()
            .scaledToFit()
            .frame(maxWidth: .infinity)
            .frame(height: 220)
            .clipShape(RoundedRectangle(cornerRadius: 8))

        Text(doc.content)
            .font(.system(.caption, design: .monospaced))
            .foregroundColor(Theme.textSecondary)
            .frame(maxWidth: .infinity, alignment: .leading)

        let preview = SharePreview(doc.name, image: Image(uiImage: uiImage))
        ShareLink(item: imageURL, preview: preview) {
            MindscapeExportLabel()
        }
    }
}

/// Content for a plain-text Mindscape entry.
private struct MindscapeTextContent: View {
    let doc: RAGDocument

    private var joinedText: String { doc.chunks.joined(separator: "\n\n---\n\n") }

    var body: some View {
        ScrollView(.vertical) {
            Text(joinedText)
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(Theme.textPrimary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(height: 150)
        .padding(8)
        .background(Theme.cardBackground)
        .cornerRadius(8)

        ShareLink(item: joinedText) {
            MindscapeExportLabel()
        }
    }
}

/// A single Mindscape entry row — an image entry if the document has a backing
/// generated image, otherwise the plain-text chunk view.
private struct MindscapeDocumentRow: View {
    let doc: RAGDocument
    @ObservedObject var ragManager: RAGManager

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(doc.name)
                    .font(.headline)
                    .foregroundColor(Theme.accentCyan)
                Spacer()
                Button(action: {
                    if let idx = ragManager.documents.firstIndex(where: { $0.id == doc.id }) {
                        ragManager.deleteDocument(at: IndexSet(integer: idx))
                    }
                }) {
                    Image(systemName: "trash")
                        .foregroundColor(.red)
                }
            }

            if doc.imageFileName != nil,
               let imgData = ragManager.imageData(for: doc),
               let uiImage = UIImage(data: imgData),
               let imageURL = ragManager.imageURL(for: doc) {
                MindscapeImageContent(doc: doc, uiImage: uiImage, imageURL: imageURL)
            } else {
                MindscapeTextContent(doc: doc)
            }
        }
        .padding()
        .background(Color.black)
        .cornerRadius(12)
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Theme.border, lineWidth: 1))
    }
}

struct MindscapeView: View {
    @ObservedObject var ragManager: RAGManager
    @State private var showDocImporter = false

    var body: some View {
        VStack(spacing: 0) {
            // In-view header. The navigation bar title alone was the problem the user hit: it's
            // drawn by UIKit, so before `AppAppearance.configure()` it rendered in near-black on
            // this near-black background. That's fixed, but an explicit header also gives the
            // screen a readable title regardless of how the bar is styled, and room to say what
            // the Mindscape actually is.
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Image(systemName: "brain.filled.head.profile")
                        .foregroundColor(Theme.accentCyan)
                    Text("MINDSCAPE")
                        .font(.system(size: 15, weight: .black))
                        .foregroundColor(Theme.textPrimary)
                        .kerning(1.5)
                    Spacer()
                    Text("\(ragManager.documents.count)")
                        .font(.system(size: 12, weight: .bold, design: .monospaced))
                        .foregroundColor(Theme.accentCyan)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(Theme.accentCyan.opacity(0.15))
                        .cornerRadius(6)
                }
                Text("Documents and generated images the assistant can draw on.")
                    .font(.system(size: 12))
                    .foregroundColor(Theme.textSecondary)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.background)
            .overlay(
                Rectangle().frame(height: 1).foregroundColor(Theme.border),
                alignment: .bottom
            )

            ScrollView {
                VStack(spacing: 12) {
                    if ragManager.documents.isEmpty {
                        Text("No entries yet. Import a text document below, or generate an image — generated images are added here automatically.")
                            .font(.system(size: 13))
                            .foregroundColor(Theme.textSecondary)
                            .multilineTextAlignment(.center)
                            .lineSpacing(3)
                            .padding()
                    } else {
                        ForEach(ragManager.documents) { doc in
                            MindscapeDocumentRow(doc: doc, ragManager: ragManager)
                        }
                    }
                }
                .padding()
            }
            
            Button(action: { showDocImporter = true }) {
                HStack {
                    Image(systemName: "doc.badge.plus")
                    Text("Import RAG Document")
                }
                .font(.headline)
                .foregroundColor(Theme.onAccent)
                .padding()
                .frame(maxWidth: .infinity)
                .background(Theme.accentCyan)
                .cornerRadius(15)
            }
            .padding()
        }
        .background(Theme.background.ignoresSafeArea())
        .navigationTitle("Mindscape")
        .navigationBarTitleDisplayMode(.inline)
        .fileImporter(
            isPresented: $showDocImporter,
            allowedContentTypes: [.plainText],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let firstUrl = urls.first else { return }
                ingestRAGDocument(from: firstUrl)
            case .failure(let error):
                print("Error importing: \(error)")
            }
        }
    }
    
    private func ingestRAGDocument(from sourceURL: URL) {
        let isSecured = sourceURL.startAccessingSecurityScopedResource()
        defer { if isSecured { sourceURL.stopAccessingSecurityScopedResource() } }
        
        do {
            let content = try String(contentsOf: sourceURL, encoding: .utf8)
            ragManager.ingestDocument(name: sourceURL.lastPathComponent, content: content)
        } catch {
            print("Failed ingesting RAG document: \(error)")
        }
    }
}
