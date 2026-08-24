import SwiftUI
import UniformTypeIdentifiers

/// First-run flow. Three jobs, all of them things App Review looks for specifically:
///
/// 1. **Agreement gate** (Guideline 1.2) — an affirmative, un-prechecked acceptance of the
///    acceptable-use policy and terms, plus a 17+ age confirmation, before any generation is
///    possible. Re-shown whenever `AppInfo.policyVersion` increases.
/// 2. **A working app out of the box** (Guideline 2.1) — the app used to open to a disabled text
///    field and the word "unloaded", which reads to a reviewer as a broken or incomplete app.
///    The setup step gets a real model onto the device before the user ever reaches the chat.
/// 3. **Setting expectations** — that output is machine-generated and can be wrong, and that
///    nothing leaves the device.
struct OnboardingView: View {

    @AppStorage("onboardingCompleted") private var onboardingCompleted = false
    @AppStorage("acceptedPolicyVersion") private var acceptedPolicyVersion = 0
    @AppStorage("ageConfirmed") private var ageConfirmed = false

    // `@ObservedObject`, not `@StateObject`: this view doesn't own `ModelDownloadManager`'s
    // lifecycle — `.shared` always returns the same instance regardless of which wrapper reads
    // it, and `@StateObject` implies ownership that isn't real here.
    @ObservedObject private var downloads = ModelDownloadManager.shared

    @State private var page = 0
    @State private var agreedToPolicies = false
    @State private var confirmedAge = false
    @State private var showModelImporter = false
    @State private var installedAModel = false
    /// Status line for the "import a .gguf I already have" path — `nil` when idle, an in-progress
    /// message while `handleImport` runs off the main actor, or an error left visible on failure.
    @State private var importStatus: String? = nil
    @State private var isImportingModel = false

    /// Skips straight to the agreement page when the only reason we're here is a policy update —
    /// re-running the whole introduction for an existing user would be noise.
    private let startsAtAgreement: Bool

    init(startsAtAgreement: Bool = false) {
        self.startsAtAgreement = startsAtAgreement
        _page = State(initialValue: startsAtAgreement ? 2 : 0)
    }

    var body: some View {
        ZStack {
            // Modest scrim — the copy on these pages sits on its own panel, so the backdrop
            // stays visible instead of being washed out to keep the text legible.
            GlitchBackgroundView(scrimOpacity: 0.35).ignoresSafeArea()

            VStack(spacing: 0) {
                TabView(selection: $page) {
                    welcomePage.tag(0)
                    privacyPage.tag(1)
                    agreementPage.tag(2)
                    modelSetupPage.tag(3)
                }
                .tabViewStyle(.page(indexDisplayMode: .never))

                pageIndicator
                    .padding(.bottom, 12)

                footer
                    .padding(.horizontal, 24)
                    .padding(.bottom, 28)
            }
        }
        .fileImporter(
            isPresented: $showModelImporter,
            allowedContentTypes: [.item],
            allowsMultipleSelection: false
        ) { result in
            handleImport(result)
        }
        .onChange(of: downloads.lastCompletedModelID) { _, newValue in
            guard let id = newValue else { return }
            installedAModel = true
            // Same reasoning as the import path: a model the user just chose and waited on
            // should be ready when they reach the chat, not gated behind a prompt.
            if let model = ModelCatalog.all.first(where: { $0.id == id }), model.kind == .chat {
                OnboardingHandoff.requestAutoLoad(fileName: model.fileName)
            }
        }
        .onChange(of: page) { _, newValue in
            // The primary "Continue" button already gates page 2 → 3 behind `canAdvance`, but a
            // `.page`-style `TabView` lets an ordinary swipe land on page 3 directly, bypassing
            // the age/policy consent gate this screen exists to enforce (Guideline 1.2) — exactly
            // the kind of interaction App Store review performs when skimming onboarding. Snap
            // back rather than let an unattended swipe substitute for a real agreement.
            if newValue == 3, !(confirmedAge && agreedToPolicies) {
                page = 2
            }
        }
        .onAppear {
            installedAModel = !AppFiles.contents(of: AppFiles.models, matchingExtensions: ["gguf"]).isEmpty
        }
    }

    // MARK: - Pages

    private var welcomePage: some View {
        onboardingScaffold(
            icon: "cpu.fill",
            iconColor: Theme.accent,
            title: "Welcome to \(AppInfo.displayName)",
            subtitle: "An AI assistant that runs entirely on your iPhone."
        ) {
            VStack(alignment: .leading, spacing: 16) {
                bullet("bolt.fill", "Works offline",
                       "Text and images are always generated by a model stored on this device — no account, no server. There's an optional, off-by-default internet search feature you can turn on later; the app will always ask before using it.")
                bullet("sparkles", "Chat and images in one place",
                       "Ask a question, or describe a picture and the app will generate it.")
                bullet("exclamationmark.triangle.fill", "Machine-generated output",
                       "Responses come from a statistical model. They can be wrong, outdated, or made up. Check anything that matters.")
            }
        }
    }

    private var privacyPage: some View {
        onboardingScaffold(
            icon: "lock.shield.fill",
            iconColor: Theme.accentCyan,
            title: "Your data stays here",
            subtitle: "Nothing you type is uploaded, logged, or sold — unless you explicitly ask this app to search the internet for it."
        ) {
            VStack(alignment: .leading, spacing: 16) {
                bullet("iphone", "Stored on device only",
                       "Conversations, memories, attachments, and generated images live in this app's private storage and are deleted with the app.")
                bullet("antenna.radiowaves.left.and.right.slash", "No analytics, no tracking",
                       "There is no advertising SDK, no analytics SDK, and no third-party tracking of any kind. There is no account and no login.")
                bullet("ladybug", "Crash reports stay local",
                       "If the app closes unexpectedly, it writes a technical report — crash type, stack trace, memory, and which models were loaded — to this device. It contains no prompts or conversations, and it is only ever sent if you choose to send it from your own mail app.")
                bullet("arrow.down.circle", "Downloading a model",
                       "Contacts the file's host to fetch it. No prompt or personal data is ever sent — only a standard file download.")
                bullet("globe", "Internet search (off by default)",
                       "Turn it on in Settings if you want it. Even then, the app asks before every search and only sends that search's text — nothing else about you or your conversation.")
            }
        }
    }

    private var agreementPage: some View {
        onboardingScaffold(
            icon: "hand.raised.fill",
            iconColor: Theme.accentRose,
            title: "Ground rules",
            subtitle: "Required before you can generate anything."
        ) {
            VStack(alignment: .leading, spacing: 18) {
                Text("There is zero tolerance for objectionable content. Prompts and responses are screened on this device, and requests that sexualize minors are blocked outright — that filter cannot be turned off, including for models you import yourself.")
                    .font(.system(size: 13))
                    .foregroundColor(Theme.textSecondary)
                    .lineSpacing(4)

                consentToggle(isOn: $confirmedAge,
                              text: "I am 17 years of age or older.")

                consentToggle(isOn: $agreedToPolicies,
                              text: "I agree to the Acceptable Use Policy and Terms of Use.")

                HStack(spacing: 16) {
                    NavigationLinkButton("Acceptable Use", text: LegalText.acceptableUse)
                    NavigationLinkButton("Terms", text: LegalText.termsOfUse)
                    NavigationLinkButton("Privacy", text: LegalText.privacyPolicy)
                }
            }
        }
    }

    private var modelSetupPage: some View {
        onboardingScaffold(
            icon: installedAModel ? "checkmark.circle.fill" : "arrow.down.circle.fill",
            iconColor: installedAModel ? .green : Theme.accent,
            title: installedAModel ? "You're ready" : "Add a model",
            subtitle: installedAModel
                ? "A model is installed. You can add more or swap it any time in Settings."
                : "\(AppInfo.displayName) needs a model to think with. Pick one to download — this is a one-time step."
        ) {
            VStack(spacing: 14) {
                // Onboarding only ever offers one model at a time, so taking whichever download
                // happens to be active is equivalent to taking "the" one here — concurrent
                // downloads are a Settings-only affordance.
                if let progress = downloads.activeDownloads.values.first {
                    downloadProgressCard(progress)
                } else {
                    // Chat models only. Onboarding's job is to get the app to a working state
                    // in one step; a 3 GB diffusion checkpoint is a deliberate choice the user
                    // can make later in Settings, not something to put in front of them here.
                    ForEach(ModelCatalog.chatModels) { model in
                        catalogRow(model)
                    }

                    if let error = downloads.lastError {
                        Text(error)
                            .font(.system(size: 12))
                            .foregroundColor(.orange)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    Toggle(isOn: $downloads.allowsCellularDownload) {
                        Text("Allow downloads over cellular")
                            .font(.system(size: 13))
                            .foregroundColor(Theme.textSecondary)
                    }
                    .toggleStyle(SwitchToggleStyle(tint: Theme.accent))

                    Button {
                        importStatus = nil
                        showModelImporter = true
                    } label: {
                        Text("Or import a .gguf file I already have")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundColor(Theme.accentCyan)
                    }
                    .disabled(isImportingModel)
                    .padding(.top, 4)

                    if let importStatus {
                        Text(importStatus)
                            .font(.system(size: 12))
                            .foregroundColor(Theme.textSecondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
        }
    }

    // MARK: - Components

    private func catalogRow(_ model: CatalogModel) -> some View {
        let installed = downloads.isInstalled(model)
        // Same device-fit badges Settings' own catalog row shows (see its `catalogRow` for the
        // full reasoning on both figures) — this is the first, highest-stakes model choice, so it
        // shouldn't get less guidance than the same choice made later from Settings.
        let fitsOnDevice = model.approxRuntimeGB < MemoryBudget.entitledGB
        let meetsTier = MemoryBudget.physicalGB + 0.6 >= model.minimumRAMGB
        return HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(model.displayName)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(Theme.textPrimary)
                Text("\(model.sizeDescription) · \(model.parameterCount) · \(model.license)")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(Theme.textMuted)
                Text(model.summary)
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
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
            Spacer(minLength: 8)

            if installed {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
                    .font(.system(size: 20))
            } else {
                // A partial transfer survives being interrupted, including by the app being killed
                // mid-download during setup — offer to continue it rather than silently restarting.
                let isResumable = downloads.resumableModelIDs.contains(model.id)
                Button {
                    downloads.download(model)
                } label: {
                    Text(isResumable ? "Resume" : "Get")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundColor(Theme.onAccent)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 7)
                        .background(RoundedRectangle(cornerRadius: 8).fill(isResumable ? Color.orange : Theme.accent))
                }
            }
        }
        .padding(12)
        .background(Theme.cardBackground)
        .cornerRadius(12)
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Theme.border, lineWidth: 1))
    }

    private func downloadProgressCard(_ progress: ModelDownloadManager.Progress) -> some View {
        VStack(spacing: 12) {
            Text(progress.isResumed ? "Resuming…" : "Downloading…")
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(Theme.textPrimary)

            ProgressView(value: progress.fractionCompleted)
                .progressViewStyle(LinearProgressViewStyle(tint: Theme.accent))

            Text("\(progress.writtenDescription) of \(progress.totalDescription)")
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(Theme.textSecondary)

            Text("You can leave the app — the download continues in the background, and picks up where it left off if it's interrupted.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textMuted)
                .multilineTextAlignment(.center)

            if let model = ModelCatalog.model(withID: progress.modelID) {
                Button("Pause Download") { downloads.cancel(model) }
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.orange)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity)
        .background(Theme.cardBackground)
        .cornerRadius(12)
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Theme.border, lineWidth: 1))
    }

    private func consentToggle(isOn: Binding<Bool>, text: String) -> some View {
        Button {
            isOn.wrappedValue.toggle()
        } label: {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: isOn.wrappedValue ? "checkmark.square.fill" : "square")
                    .font(.system(size: 20))
                    .foregroundColor(isOn.wrappedValue ? Theme.accent : Theme.textMuted)
                Text(text)
                    .font(.system(size: 14))
                    .foregroundColor(Theme.textPrimary)
                    .multilineTextAlignment(.leading)
                Spacer()
            }
            .contentShape(Rectangle())
        }
    }

    private func bullet(_ icon: String, _ title: String, _ body: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(Theme.accentCyan)
                .font(.system(size: 15))
                .frame(width: 24)
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(Theme.textPrimary)
                Text(body)
                    .font(.system(size: 12))
                    .foregroundColor(Theme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .lineSpacing(3)
            }
        }
    }

    private var pageIndicator: some View {
        HStack(spacing: 7) {
            ForEach(0..<4, id: \.self) { index in
                Circle()
                    .fill(index == page ? Theme.accent : Theme.border)
                    .frame(width: 6, height: 6)
            }
        }
    }

    private var footer: some View {
        VStack(spacing: 10) {
            Button {
                advance()
            } label: {
                Text(primaryButtonTitle)
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(canAdvance ? Theme.onAccent : Theme.textSecondary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 15)
                    .background(
                        RoundedRectangle(cornerRadius: 14)
                            .fill(canAdvance ? Theme.accent : Theme.border)
                    )
            }
            .disabled(!canAdvance)

            if page == 3 && !installedAModel {
                // Padding plus an explicit hit shape, because a bare `Button(_:)` is tappable only
                // across the glyphs themselves — a 16 pt-tall target sitting directly under a
                // paged `TabView`, which reliably swallowed the first tap. This is the escape
                // hatch from a required setup step, so it has to work the first time.
                Button("Set this up later") { complete() }
                    .font(.system(size: 13))
                    .foregroundColor(Theme.textMuted)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 10)
                    .contentShape(Rectangle())
            }
        }
    }

    // MARK: - Navigation

    private var canAdvance: Bool {
        switch page {
        case 2:  return confirmedAge && agreedToPolicies
        case 3:  return !downloads.isDownloading && installedAModel
        default: return true
        }
    }

    private var primaryButtonTitle: String {
        switch page {
        case 2:  return "Agree & Continue"
        case 3:  return installedAModel ? "Start Using \(AppInfo.displayName)" : "Choose a model above"
        default: return "Continue"
        }
    }

    private func advance() {
        if page == 3 {
            complete()
        } else {
            withAnimation { page += 1 }
        }
    }

    private func complete() {
        // Defense in depth alongside the `onChange(of: page)` swipe guard above: this is the
        // one gate that actually flips `onboardingCompleted`, so it shouldn't trust that every
        // path leading here already enforced consent.
        guard confirmedAge, agreedToPolicies else { return }
        acceptedPolicyVersion = AppInfo.policyVersion
        ageConfirmed = confirmedAge
        onboardingCompleted = true
    }

    private func handleImport(_ result: Result<[URL], Error>) {
        guard let source = try? result.get().first else { return }
        guard source.pathExtension.lowercased() == "gguf" else { return }
        guard source.startAccessingSecurityScopedResource() else { return }

        AppFiles.prepare()
        let destination = AppFiles.models.appendingPathComponent(source.lastPathComponent)

        isImportingModel = true
        importStatus = "Checking file…"

        // Backgrounded and validated the same way `SettingsView.copyModelToAppDocuments` is —
        // this path never got either fix: a multi-GB copy run synchronously on the main thread
        // (during onboarding specifically, the one screen whose own doc comment says its job is
        // to avoid looking like a broken app) freezes the UI for the whole copy, and skipping
        // `GGUFValidator.validate` let a mislabeled/corrupt file install successfully and only
        // fail later, less diagnostically, inside `LlamaRunner`.
        Task.detached(priority: .background) {
            defer { source.stopAccessingSecurityScopedResource() }
            do {
                try GGUFValidator.validate(path: source.path)

                await MainActor.run { importStatus = "Copying (keep the app open)…" }

                if FileManager.default.fileExists(atPath: destination.path) {
                    try FileManager.default.removeItem(at: destination)
                }
                try FileManager.default.copyItem(at: source, to: destination)
                AppFiles.excludeFromBackup(destination)

                await MainActor.run {
                    ModelInventory.shared.record(
                        fileName: destination.lastPathComponent,
                        kind: .chat,
                        catalogID: ModelCatalog.model(withFileName: destination.lastPathComponent)?.id,
                        byteSize: (try? destination.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init) ?? 0
                    )
                    installedAModel = true
                    isImportingModel = false
                    importStatus = nil
                    OnboardingHandoff.requestAutoLoad(fileName: destination.lastPathComponent)
                    LogManager.shared.log("Onboarding: imported \(destination.lastPathComponent), queued for auto-load")
                }
            } catch {
                await MainActor.run {
                    isImportingModel = false
                    // Left visible rather than cleared, so the failure doesn't vanish before the
                    // user can read it — mirrors `copyModelToAppDocuments`.
                    importStatus = "Import failed: \(error.localizedDescription)"
                }
                LogManager.shared.log("Onboarding: import failed — \(error.localizedDescription)")
            }
        }
    }
}

/// One-shot handoff from onboarding to the main screen.
///
/// Someone who has just picked a model file has unambiguously told the app which model they
/// want; making them then dismiss a "Load previous model?" prompt is asking a question that was
/// already answered. This records the choice so `ContentView` can load it directly on first
/// appearance, and clears itself after one read so it never fires twice.
enum OnboardingHandoff {
    private static let key = "DarkAI_PendingAutoLoadModel"

    static func requestAutoLoad(fileName: String) {
        UserDefaults.standard.set(fileName, forKey: key)
    }

    /// Returns the queued model's URL if one is waiting and still exists, consuming the request.
    static func takeAutoLoadURL() -> URL? {
        guard let fileName = UserDefaults.standard.string(forKey: key) else { return nil }
        UserDefaults.standard.removeObject(forKey: key)
        let url = AppFiles.models.appendingPathComponent(fileName)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }
}

/// Small helper so the agreement page can push full policy text without wrapping the whole
/// onboarding flow in a `NavigationView` (which would fight the paged `TabView`).
private struct NavigationLinkButton: View {
    let title: String
    let text: String
    @State private var isPresented = false

    init(_ title: String, text: String) {
        self.title = title
        self.text = text
    }

    var body: some View {
        Button(title) { isPresented = true }
            .font(.system(size: 12, weight: .semibold))
            .foregroundColor(Theme.accentCyan)
            .sheet(isPresented: $isPresented) {
                NavigationView {
                    PolicyTextView(title: title, text: text)
                        .toolbar {
                            ToolbarItem(placement: .navigationBarTrailing) {
                                Button("Done") { isPresented = false }
                                    .foregroundColor(Theme.accentCyan)
                            }
                        }
                }
            }
    }
}

// MARK: - Shared page scaffold

extension OnboardingView {
    @ViewBuilder
    fileprivate func onboardingScaffold<Content: View>(
        icon: String,
        iconColor: Color,
        title: String,
        subtitle: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        ScrollView {
            VStack(spacing: 22) {
                Spacer(minLength: 40)

                Image(systemName: icon)
                    .font(.system(size: 52))
                    .foregroundColor(iconColor)
                    .neonGlow(color: iconColor, radius: 10)

                VStack(spacing: 8) {
                    Text(title)
                        .font(.system(size: 24, weight: .black))
                        .foregroundColor(Theme.textPrimary)
                        .multilineTextAlignment(.center)
                    Text(subtitle)
                        .font(.system(size: 14))
                        .foregroundColor(Theme.textSecondary)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                }

                content()
                    .padding(.top, 6)
            }
            .padding(.horizontal, 22)
            .padding(.vertical, 26)
            // Same treatment as the main screen's welcome copy: shade behind the text, not
            // across the whole backdrop.
            .readablePanel(cornerRadius: 26)
            .padding(.horizontal, 18)
            .padding(.bottom, 20)
        }
    }
}
