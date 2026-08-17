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
    @StateObject private var inventory = ModelInventory.shared

    @State private var importedModels: [URL] = []
    @State private var importedCoreMLModels: [URL] = []
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

                        missingModelsBanner

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

                            VStack(alignment: .leading, spacing: 10) {
                                Text("DOWNLOAD A CHAT MODEL")
                                    .font(.system(size: 11, weight: .bold))
                                    .foregroundColor(Theme.textSecondary)
                                    .kerning(1.0)
                                downloadCatalogButton(for: .chat)
                            }

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

                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                Image(systemName: "cpu.fill")
                                    .foregroundColor(Theme.accentCyan)
                                    .font(.headline)
                                Text("CORE ML MODELS (ANE)")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(Theme.textPrimary)
                                    .kerning(1.2)
                                Spacer()
                            }

                            Text("Runs on the Apple Neural Engine instead of the CPU/GPU path the models above use. No file import — catalog only.")
                                .font(.system(size: 11))
                                .foregroundColor(Theme.textMuted)
                                .fixedSize(horizontal: false, vertical: true)

                            if importedCoreMLModels.isEmpty {
                                Text("No Core ML model installed. Download the one below to try it.")
                                    .font(.system(size: 13))
                                    .foregroundColor(Theme.textSecondary)
                                    .padding()
                                    .frame(maxWidth: .infinity, alignment: .center)
                                    .glassCard(cornerRadius: 12)
                            } else {
                                VStack(spacing: 12) {
                                    ForEach(importedCoreMLModels, id: \.self) { url in
                                        coreMLModelRow(for: url)
                                    }
                                }
                            }

                            Divider().background(Theme.border)

                            VStack(alignment: .leading, spacing: 10) {
                                Text("DOWNLOAD A CORE ML MODEL")
                                    .font(.system(size: 11, weight: .bold))
                                    .foregroundColor(Theme.textSecondary)
                                    .kerning(1.0)
                                downloadCatalogButton(for: .coreML)
                            }
                        }
                        .glassCard(cornerRadius: 16)

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

                            VStack(alignment: .leading, spacing: 10) {
                                Text("DOWNLOAD A DIFFUSION MODEL")
                                    .font(.system(size: 11, weight: .bold))
                                    .foregroundColor(Theme.textSecondary)
                                    .kerning(1.0)
                                downloadCatalogButton(for: .diffusion)
                            }

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

                            Text("IMAGE GEN SETTINGS")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundColor(Theme.textSecondary)
                                .kerning(1.0)

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

                            // The Core ML backend has no adjustable context window — whatever
                            // `loadedContextWindow` reports is baked into the model graph, not
                            // something `contextTokenLimit` governs. Showing the GGUF slider here
                            // would offer a control that does nothing for the model that's
                            // actually loaded. The two Core ML engines mean different things by
                            // that number, though (`coreMLContextIsSliding` — see its doc comment
                            // on `LLMManager`), so the description can't just hardcode either
                            // engine's specific behavior the way this used to hardcode OpenELM's.
                            if llmManager.activeBackend == .coreML {
                                VStack(alignment: .leading, spacing: 6) {
                                    HStack {
                                        Text("Context Window:")
                                            .font(.system(size: 13))
                                            .foregroundColor(Theme.textSecondary)
                                        Spacer()
                                        Text(llmManager.coreMLContextIsSliding
                                             ? "\(llmManager.loadedContextWindow) tokens (sliding)"
                                             : "\(llmManager.loadedContextWindow) tokens (fixed)")
                                            .font(.system(size: 13, weight: .bold, design: .monospaced))
                                            .foregroundColor(Theme.accent)
                                    }
                                    Text(llmManager.coreMLContextIsSliding
                                         ? "This Core ML model has no adjustable context. Once a conversation passes \(llmManager.loadedContextWindow) tokens, the earliest turns are gradually forgotten rather than the reply being cut off."
                                         : "This Core ML model has no adjustable context — \(llmManager.loadedContextWindow) tokens total, prompt and reply combined.")
                                        .font(.system(size: 11))
                                        .foregroundColor(Theme.textMuted)
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                            } else {
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
                                    // Naming which bound is binding. "Maximum for this device"
                                    // was wrong whenever the model, not the hardware, was the
                                    // limit — and that is the common case, since most models are
                                    // trained well below what a recent iPhone could allocate.
                                    Text(llmManager.loadedTrainedContext > 0
                                         && llmManager.loadedTrainedContext <= ceiling
                                         ? "Maximum for this model: \(ceiling) tokens — it was trained for \(llmManager.loadedTrainedContext)."
                                         : "Maximum for this device: \(ceiling) tokens.")
                                        .font(.system(size: 11))
                                        .foregroundColor(Theme.textMuted)
                                }
                            }
                            }


                            Divider().background(Theme.border)

                            // Same reasoning as the Context Window gate above: a Core ML model's
                            // reply is always cut off by its own context window (128 tokens total
                            // for OpenELM; the sliding cache's own bookkeeping for the Llama
                            // pipeline) long before this slider's value could ever bind — showing
                            // it un-gated offered a number the model could structurally never
                            // reach, e.g. "512 tokens" sitting next to a model capped at 128 total.
                            if llmManager.activeBackend == .coreML {
                                Text("Reply length for this Core ML model is bounded by its context window above, not by a separate output limit.")
                                    .font(.system(size: 11))
                                    .foregroundColor(Theme.textMuted)
                                    .fixedSize(horizontal: false, vertical: true)
                            } else {
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
                            
                            Text("\(AppInfo.displayName) gradually adapts its tone to how you write. It builds one profile shared by every model, and everything it learns stays on this device. Resetting erases it completely.")
                                .font(.system(size: 13))
                                .foregroundColor(Theme.textSecondary)
                                .lineSpacing(4)
                            
                            Button(action: {
                                showResetPersonalityAlert = true
                            }) {
                                HStack {
                                    Image(systemName: "trash")
                                    Text("Reset Personality")
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
                                    personalityManager.resetPersonality()
                                }
                            } message: {
                                Text("This will erase every learned speech pattern, for all models. This action cannot be undone.")
                            }
                        }
                        .glassCard(cornerRadius: 16)
                        
                        // Recovery. Sits ahead of Diagnostics on purpose: someone scrolling here
                        // after an out-of-memory error wants the fix before the log of it.
                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                Image(systemName: "arrow.clockwise.circle")
                                    .foregroundColor(Theme.accentCyan)
                                Text("TROUBLESHOOTING")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(Theme.textPrimary)
                                    .kerning(1.2)
                                Spacer()
                            }

                            ResetModelsButton(llmManager: llmManager, diffusionManager: diffusionManager)
                        }
                        .glassCard(cornerRadius: 16)

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
                    secondaryButton: .destructive(Text("Ignore & Keep")) {}
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
                refreshCoreMLModelList()
            }
            // Lives at the root rather than on the download screen itself, since that screen is
            // torn down the moment the user navigates back — a download that finishes while
            // they're back on the main Settings screen (or with the app backgrounded) still
            // needs to move from "downloading" to "installed" in the list either way.
            .onChange(of: downloads.lastCompletedModelID) { _, newValue in
                guard newValue != nil else { return }
                refreshModelList()
                refreshCoreMLModelList()
                loadDiffusionModels()
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
    /// Shown when a model the app recorded as installed is no longer on disk.
    ///
    /// Without this the same situation renders as an empty model list, which reads as "the update
    /// wiped my downloads" no matter what actually happened. Naming the file, saying why it can go
    /// missing while everything else survives, and putting the re-download one tap away is the
    /// whole fix — the app cannot prevent the loss (excluding multi-gigabyte weights from iCloud
    /// backup is deliberate) but it can stop it being a mystery.
    @ViewBuilder
    private var missingModelsBanner: some View {
        if !inventory.missing.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: "externaldrive.badge.questionmark")
                        .foregroundColor(.orange)
                        .font(.system(size: 16))
                    VStack(alignment: .leading, spacing: 4) {
                        Text(inventory.missing.count == 1 ? "A model is missing" : "\(inventory.missing.count) models are missing")
                            .font(.system(size: 14, weight: .bold))
                            .foregroundColor(Theme.textPrimary)
                        Text("These were installed on this device and are no longer here. App updates don't remove them — but model files are kept out of iCloud backup (they're far too large for it), so restoring from a backup, transferring to a new device, or reinstalling the app leaves them behind.")
                            .font(.system(size: 11))
                            .foregroundColor(Theme.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer(minLength: 4)
                }

                ForEach(inventory.missing) { entry in
                    HStack(spacing: 10) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(entry.fileName)
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundColor(Theme.textPrimary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Text(entry.catalogModel == nil
                                 ? "Imported by you — re-import the file to get it back."
                                 : "\(entry.sizeDescription) · available in the catalog below")
                                .font(.system(size: 10))
                                .foregroundColor(Theme.textMuted)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        Spacer(minLength: 4)

                        if let model = entry.catalogModel {
                            Button {
                                downloads.download(model)
                                inventory.dismiss(entry)
                            } label: {
                                Text("Download")
                                    .font(.system(size: 11, weight: .bold))
                                    .foregroundColor(Theme.onAccent)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(RoundedRectangle(cornerRadius: 8).fill(Theme.accent))
                            }
                            .disabled(downloads.isDownloading)
                            .opacity(downloads.isDownloading ? 0.4 : 1)
                        }

                        Button {
                            inventory.dismiss(entry)
                        } label: {
                            Image(systemName: "xmark")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundColor(Theme.textMuted)
                                .frame(width: 28, height: 28)
                        }
                    }
                    .padding(10)
                    .background(Theme.cardBackground)
                    .cornerRadius(10)
                }
            }
            .padding(14)
            .background(Color.orange.opacity(0.12))
            .cornerRadius(14)
            .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.orange.opacity(0.45), lineWidth: 1))
        }
    }

    /// Compact entry point shown inline in each "LOCAL LLM MODELS" / "DIFFUSION MODEL" card —
    /// pushes to `modelDownloadScreen`, which holds the catalog list itself. Reflects an
    /// in-progress download for this `kind` even while collapsed, so leaving the download
    /// screen doesn't hide the fact that a transfer is still running.
    private func tint(for kind: ModelKind) -> Color {
        switch kind {
        case .chat: return Theme.accentCyan
        case .diffusion: return Color.purple
        case .coreML: return Theme.accentCyan
        }
    }

    private func browseIcon(for kind: ModelKind) -> String {
        switch kind {
        case .chat: return "arrow.down.circle.fill"
        case .diffusion: return "photo.badge.arrow.down.fill"
        case .coreML: return "cpu.fill"
        }
    }

    private func displayName(for kind: ModelKind) -> String {
        switch kind {
        case .chat: return "Chat"
        case .diffusion: return "Diffusion"
        case .coreML: return "Core ML"
        }
    }

    @ViewBuilder
    private func downloadCatalogButton(for kind: ModelKind) -> some View {
        // Downloads run concurrently, so more than one of this `kind` can be active at once —
        // the collapsed row aggregates them by bytes rather than trying to name just one.
        let activeForKind = downloads.activeDownloads.values.filter { $0.kind == kind }
        let tint = tint(for: kind)
        let downloadableCount = ModelCatalog.models(for: kind).filter { !downloads.isInstalled($0) }.count

        NavigationLink {
            // `.id(kind)` matters here, not just style: `NavigationView` (this file predates
            // `NavigationStack`) identifies a closure-built destination by its view *shape*, and
            // the chat and diffusion buttons build one from the exact same shape with only
            // `kind` differing inside a closure. Without an explicit identity tied to `kind`,
            // tapping either button could push whichever destination `NavigationView` last
            // resolved for that shape — observed in testing as the chat button opening the
            // diffusion screen.
            modelDownloadScreen(for: kind)
                .id(kind)
        } label: {
            HStack(spacing: 10) {
                Image(systemName: browseIcon(for: kind))
                    .foregroundColor(tint)
                    .font(.system(size: 16))
                VStack(alignment: .leading, spacing: 2) {
                    Text(activeForKind.isEmpty
                         ? "Browse \(displayName(for: kind)) Models"
                         : (activeForKind.count == 1 ? "Downloading…" : "Downloading \(activeForKind.count)…"))
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(Theme.textPrimary)
                    if !activeForKind.isEmpty {
                        let writtenTotal = activeForKind.reduce(0) { $0 + $1.bytesWritten }
                        let expectedTotal = activeForKind.reduce(0) { $0 + $1.totalBytes }
                        let fraction = expectedTotal > 0 ? Double(writtenTotal) / Double(expectedTotal) : 0
                        Text("\(Int(fraction * 100))% overall · \(ByteCountFormatter.string(fromByteCount: writtenTotal, countStyle: .file)) of \(ByteCountFormatter.string(fromByteCount: expectedTotal, countStyle: .file))")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundColor(Theme.textSecondary)
                    } else {
                        Text(downloadableCount > 0
                             ? "\(downloadableCount) available to download"
                             : "All caught up — every catalog model is installed")
                            .font(.system(size: 10))
                            .foregroundColor(Theme.textMuted)
                    }
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(Theme.textMuted)
            }
            .padding(12)
            .background(Theme.background.opacity(0.4))
            .cornerRadius(10)
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(Theme.border, lineWidth: 1))
        }
        .buttonStyle(.plain)
    }

    /// The screen `downloadCatalogButton` pushes to — the catalog list itself, moved out of the
    /// main Settings screen so it isn't rendered (and its rows re-evaluated) until asked for.
    @ViewBuilder
    private func modelDownloadScreen(for kind: ModelKind) -> some View {
        ZStack {
            Theme.background.ignoresSafeArea()
            ScrollView {
                downloadCatalogSection(for: kind)
                    .padding()
            }
        }
        .navigationTitle("\(displayName(for: kind)) Models")
        .navigationBarTitleDisplayMode(.inline)
    }

    @ViewBuilder
    private func downloadCatalogSection(for kind: ModelKind) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            // Every catalog model is listed here, installed or not — this is also where anyone
            // reads a model's details (size, license, memory tier), not only where they download
            // one, so hiding a row the moment it's installed took that reference away right when
            // there'd otherwise be a reason to check it again. `catalogRow` shows the install
            // state, its own Get/Resume button, or its own progress bar, per model.
            ForEach(ModelCatalog.models(for: kind)) { model in
                catalogRow(model)
            }

            if let error = downloads.lastError {
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

    /// A row for a catalog model not yet installed — the list this appears in already filters
    /// Lists every catalog model, whether installed or not, so its details stay readable either
    /// way — only the trailing control (checkmark / Get / Resume / progress) changes per state.
    @ViewBuilder
    private func catalogRow(_ model: CatalogModel) -> some View {
        let installed = downloads.isInstalled(model)
        // Warn before spending a gigabyte of data on something this device can't run. Checked
        // against the ceiling rather than current headroom, since that's the bound that won't
        // change by closing something.
        //
        // That ceiling is `MemoryBudget.entitledGB`, the same figure every other memory decision
        // in the app is made against, rather than a flat fraction of physical RAM. A fraction
        // scales the wrong way — iOS reserves a roughly fixed amount for itself — so a flat 75%
        // was optimistic on exactly the small devices where the warning matters most, promising
        // a 4 GB iPhone it could run models it cannot. Sharing the figure also means the badge
        // here and the failsafe at load time can no longer disagree with each other.
        let fitsOnDevice = model.approxRuntimeGB < MemoryBudget.entitledGB
        // Physical RAM, not the app's share of it, because that is what the device tag names.
        // A device at or above the tier runs the model as described; below it, the tag says
        // which iPhone to expect it on rather than leaving "may not run" as the only signal.
        let meetsTier = MemoryBudget.physicalGB + 0.6 >= model.minimumRAMGB

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
                    Text(meetsTier
                         ? "Runs on this iPhone · recommended \(model.minimumDevice)"
                         : "Recommended for \(model.minimumDevice)")
                        .font(.system(size: 10, weight: meetsTier ? .regular : .semibold))
                        .foregroundColor(meetsTier ? Theme.textMuted : .orange)
                        .fixedSize(horizontal: false, vertical: true)
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
                } else if downloads.activeDownloads[model.id] == nil {
                    // Downloads run concurrently now, so this button is only ever gated on
                    // *this* model's own state — starting one no longer disables every other
                    // row's.
                    let isResumable = downloads.resumableModelIDs.contains(model.id)
                    Button {
                        downloads.download(model)
                    } label: {
                        Text(isResumable ? "Resume" : "Get")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(Theme.onAccent)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 6)
                            .background(RoundedRectangle(cornerRadius: 8).fill(isResumable ? Color.orange : Theme.accent))
                    }
                }
            }

            if !installed, let progress = downloads.activeDownloads[model.id] {
                VStack(alignment: .leading, spacing: 6) {
                    ProgressView(value: progress.fractionCompleted)
                        .progressViewStyle(LinearProgressViewStyle(tint: Theme.accent))
                    HStack {
                        Text("\(progress.isResumed ? "Resuming" : "Downloading") — \(progress.writtenDescription) of \(progress.totalDescription)")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundColor(Theme.textSecondary)
                        Spacer()
                        // Named for what it now does. Stopping keeps the bytes already fetched, so
                        // calling it "Cancel" would understate it and push people into waiting out
                        // a transfer they could safely interrupt.
                        Button("Pause") { downloads.cancel(model) }
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundColor(.orange)
                    }
                }
            } else if downloads.resumableModelIDs.contains(model.id) {
                HStack(spacing: 10) {
                    Text("Partly downloaded — resuming will continue rather than start over.")
                        .font(.system(size: 10))
                        .foregroundColor(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 4)
                    Button("Discard") { downloads.discardPartial(for: model) }
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(Theme.textMuted)
                }
            }
        }
        .padding(11)
        .background(Theme.background.opacity(0.4))
        .cornerRadius(10)
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Theme.border, lineWidth: 1))
    }

    @ViewBuilder
    private func modelRow(for url: URL) -> some View {
        let sizeGB = llmManager.getModelSizeGB(at: url)
        let safety = llmManager.checkMemorySafety(at: url)
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

    /// Simplified counterpart to `modelRow` for a Core ML package. Load skips straight to
    /// `llmManager.loadModel`, which routes anything installed under `AppFiles.coreMLModels` to
    /// `loadCoreMLModel` internally by checking the parent directory — not, as this used to say,
    /// by checking for a `.mlpackage` extension, which only ever matched `SingleWindowCoreMLEngine`'s
    /// own single-file install shape and silently missed the newer chunked-pipeline models that
    /// install as a plain-named directory instead. `loadCoreMLModel` has its own memory pre-flight
    /// now too (see its doc comment) — no separate tag needed here the way `modelRow` has one.
    @ViewBuilder
    private func coreMLModelRow(for url: URL) -> some View {
        let sizeGB = AppFiles.directorySizeGB(at: url)
        let isLoaded = isCurrentModel(url: url)

        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(url.lastPathComponent)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(Theme.textPrimary)
                    .lineLimit(1)
                Text(String(format: "%.2f GB · ANE", sizeGB))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(Theme.textSecondary)
            }

            Spacer()

            HStack(spacing: 8) {
                if !isLoaded {
                    Button(action: { deleteCoreMLModel(at: url) }) {
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
                    Button(action: { llmManager.unloadModel() }) {
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
                    Button(action: { llmManager.loadModel(at: url) }) {
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
            // Two situations share this case: a model that merely leaves memory tight, and one
            // too large to hold resident, which runs by streaming the layers that don't fit from
            // storage on every token. The second is a speed trade-off rather than a memory risk,
            // and saying so is what stops a model that loads fine from just seeming slow.
            let sizeGB = llmManager.getModelSizeGB(at: url)
            failsafeMessage = llmManager.willStreamFromStorage(modelSizeGB: sizeGB)
                ? "'\(url.lastPathComponent)' is larger than this device can hold in memory at once. It will still run — the layers that don't fit are read from storage as they're needed — but expect much slower replies."
                : "'\(url.lastPathComponent)' is large for this device. It should load, but memory will be tight and other apps may be closed in the background to make room."
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

    /// Unlike `.chat`/`.diffusion`, Core ML models are catalog-only — there is no "import an
    /// arbitrary file" path for them (see `ModelKind`'s doc comment) — so the installed list is
    /// derived from the catalog plus `ModelDownloadManager.isInstalled`, not a filesystem scan.
    /// A blind extension filter (the previous approach: `matchingExtensions: ["mlpackage"]`)
    /// silently dropped any installed model whose root isn't literally a `.mlpackage` directory —
    /// which every chunked pipeline model is, since it installs as a plain-named folder of several
    /// `.mlmodelc` bundles, not a single `.mlpackage`.
    private func refreshCoreMLModelList() {
        let installDirectory = ModelDownloadManager.installDirectory(for: .coreML)
        importedCoreMLModels = ModelCatalog.coreMLModels
            .filter { ModelDownloadManager.shared.isInstalled($0) }
            .map { installDirectory.appendingPathComponent($0.fileName) }
    }

    private func deleteCoreMLModel(at url: URL) {
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
            ModelInventory.shared.forget(fileName: url.lastPathComponent, kind: .coreML)
            refreshCoreMLModelList()
        } catch {
            print("Failed to delete Core ML model: \(error.localizedDescription)")
        }
    }

    private func deleteModel(at url: URL, isDiffusion: Bool) {
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
            // Drop the ledger entry too, so a deliberate deletion is never reported back as a
            // model that went missing on its own.
            ModelInventory.shared.forget(fileName: url.lastPathComponent, kind: isDiffusion ? .diffusion : .chat)
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
                    ModelInventory.shared.record(
                        fileName: destinationURL.lastPathComponent,
                        kind: .chat,
                        catalogID: ModelCatalog.model(withFileName: destinationURL.lastPathComponent)?.id,
                        byteSize: Int64(fileSizeGB * 1024 * 1024 * 1024)
                    )
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
                    await MainActor.run {
                        ModelInventory.shared.record(
                            fileName: destURL.lastPathComponent,
                            kind: .diffusion,
                            catalogID: ModelCatalog.model(withFileName: destURL.lastPathComponent)?.id,
                            byteSize: (try? destURL.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init) ?? 0
                        )
                        isDiffusionImporting = false
                        loadDiffusionModels()
                    }
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
    @State private var showJSONImporter = false
    @State private var showImportHelp = false
    /// True while a JSON file is being parsed and screened off the main actor.
    @State private var isParsingJSON = false
    /// Result of the last structured import — success summary or refusal, shown inline rather than
    /// as an alert so the text stays on screen while the user looks at what changed below it.
    @State private var importNote: ImportNote?

    private struct ImportNote: Identifiable {
        let id = UUID()
        let text: String
        let isError: Bool
        let icon: String
    }

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
                        Text("No entries yet. Import a text document or a JSON study guide / journal below, or generate an image — generated images are added here automatically.")
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
            
            importControls
        }
        .background(Theme.background.ignoresSafeArea())
        .navigationTitle("Mindscape")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showImportHelp) {
            StructuredImportHelpView()
        }
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
                importNote = ImportNote(text: error.localizedDescription, isError: true, icon: "exclamationmark.triangle.fill")
            }
        }
        .fileImporter(
            isPresented: $showJSONImporter,
            allowedContentTypes: [.json],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let firstUrl = urls.first else { return }
                ingestStructuredJSON(from: firstUrl)
            case .failure(let error):
                importNote = ImportNote(text: error.localizedDescription, isError: true, icon: "exclamationmark.triangle.fill")
            }
        }
    }

    // MARK: - Import controls

    @ViewBuilder
    private var importControls: some View {
        VStack(spacing: 10) {
            if isParsingJSON {
                HStack(spacing: 10) {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: Theme.accentCyan))
                    Text("Reading the file…")
                        .font(.system(size: 12))
                        .foregroundColor(Theme.textSecondary)
                    Spacer()
                }
                .padding(12)
                .background(Theme.cardBackground)
                .cornerRadius(10)
            }

            if let importNote {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: importNote.icon)
                        .foregroundColor(importNote.isError ? .orange : .green)
                        .font(.system(size: 13))
                    Text(importNote.text)
                        .font(.system(size: 12))
                        .foregroundColor(Theme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 4)
                    Button {
                        withAnimation { self.importNote = nil }
                    } label: {
                        Image(systemName: "xmark")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(Theme.textMuted)
                    }
                }
                .padding(12)
                .background((importNote.isError ? Color.orange : Color.green).opacity(0.12))
                .cornerRadius(10)
            }

            // Two buttons rather than one picker that accepts both: the JSON path does something
            // materially different (it reads the file's structure and can create many entries from
            // one file), and labelling that up front is what makes it discoverable at all.
            HStack(spacing: 10) {
                Button(action: { showDocImporter = true }) {
                    importButtonLabel("Text", icon: "doc.badge.plus", fill: Theme.accentCyan)
                }
                Button(action: { showJSONImporter = true }) {
                    importButtonLabel("JSON", icon: "curlybraces", fill: Theme.accent)
                }
                .disabled(isParsingJSON)
                .opacity(isParsingJSON ? 0.5 : 1)
            }

            Button {
                showImportHelp = true
            } label: {
                Text("What can I import?")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(Theme.accentCyan)
            }
        }
        .padding()
    }

    private func importButtonLabel(_ title: String, icon: String, fill: Color) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
            Text(title)
        }
        .font(.system(size: 15, weight: .bold))
        .foregroundColor(Theme.onAccent)
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity)
        .background(fill)
        .cornerRadius(15)
    }

    // MARK: - Ingest

    private func ingestRAGDocument(from sourceURL: URL) {
        let isSecured = sourceURL.startAccessingSecurityScopedResource()
        let fileName = sourceURL.lastPathComponent

        // Same size cap `StructuredImport` applies on its own (JSON) ingestion path — a plain-text
        // file has no structure to sniff, so size is the only guard available before reading it.
        let sizeBytes = (try? sourceURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        guard sizeBytes <= StructuredImport.maxFileBytes else {
            if isSecured { sourceURL.stopAccessingSecurityScopedResource() }
            let cap = ByteCountFormatter.string(fromByteCount: Int64(StructuredImport.maxFileBytes), countStyle: .file)
            importNote = ImportNote(text: "That file is too large to import (limit \(cap)).", isError: true, icon: "exclamationmark.triangle.fill")
            return
        }

        Task {
            // Reading is backgrounded for the same reason `ingestStructuredJSON` backgrounds its
            // parsing below: a large text file read synchronously on the main thread — which this
            // path used to do unconditionally, with no size limit at all — visibly freezes the app
            // mid-tap.
            let result = await Task.detached(priority: .userInitiated) {
                Result { try String(contentsOf: sourceURL, encoding: .utf8) }
            }.value
            if isSecured { sourceURL.stopAccessingSecurityScopedResource() }

            switch result {
            case .success(let content):
                let wasTruncated = ragManager.ingestDocument(name: fileName, content: content)
                importNote = ImportNote(
                    text: wasTruncated ? "Added \(fileName) (large file — only part of it was kept)." : "Added \(fileName).",
                    isError: false,
                    icon: "checkmark.circle.fill"
                )
            case .failure(let error):
                importNote = ImportNote(text: "Couldn't read that file: \(error.localizedDescription)", isError: true, icon: "exclamationmark.triangle.fill")
                LogManager.shared.log("Mindscape text import failed: \(error.localizedDescription)")
            }
        }
    }

    /// Parses a study guide / journal / transcript JSON file and commits it in one step.
    ///
    /// Parsing and ingesting are separate calls (see `StructuredImport.parse`) precisely so a file
    /// that is too large, unreadable, or refused by the content filter leaves the Mindscape exactly
    /// as it was — a half-imported journal would be worse than none.
    private func ingestStructuredJSON(from sourceURL: URL) {
        let isSecured = sourceURL.startAccessingSecurityScopedResource()
        let fileName = sourceURL.lastPathComponent
        let data: Data
        do {
            data = try Data(contentsOf: sourceURL)
        } catch {
            if isSecured { sourceURL.stopAccessingSecurityScopedResource() }
            importNote = ImportNote(text: "Couldn't read that file: \(error.localizedDescription)", isError: true, icon: "exclamationmark.triangle.fill")
            return
        }
        if isSecured { sourceURL.stopAccessingSecurityScopedResource() }

        isParsingJSON = true
        importNote = nil

        Task {
            // Parsed off the main actor. Both halves of the work scale with file size — decoding
            // several megabytes of JSON, and then screening every extracted character through
            // `ContentSafety`, which walks its term lists across the whole string. On the largest
            // file the caps allow, doing that inline would visibly freeze the app mid-tap.
            let outcome = await Task.detached(priority: .userInitiated) {
                Result { try StructuredImport.parse(data: data, fileName: fileName) }
            }.value

            isParsingJSON = false
            switch outcome {
            case .success(let result):
                for document in result.documents {
                    ragManager.ingestDocument(name: document.name, content: document.content)
                }
                withAnimation {
                    importNote = ImportNote(text: result.summary, isError: false, icon: result.kind.icon)
                }
                LogManager.shared.log("Mindscape JSON import: \(result.summary)")
            case .failure(let error):
                withAnimation {
                    importNote = ImportNote(
                        text: error.localizedDescription,
                        isError: true,
                        icon: "exclamationmark.triangle.fill"
                    )
                }
                LogManager.shared.log("Mindscape JSON import failed: \(error.localizedDescription)")
            }
        }
    }
}
