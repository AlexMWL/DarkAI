import SwiftUI
import Combine
import MessageUI

// MARK: - Content reports

/// A report the user filed about something the app generated.
///
/// Guideline 1.2 requires a mechanism for users to flag objectionable content and for the
/// developer to act on it. For an offline app there is no server to escalate to, so the flow is:
/// record it locally (so the user can see their report was taken), remove the offending message
/// from the conversation immediately, and offer to mail the developer the details.
struct ContentReport: Identifiable, Codable {
    var id: UUID = UUID()
    var date: Date = Date()
    var reason: String
    var note: String
    /// Truncated copy of what was reported. Kept short on purpose — a report is evidence, not
    /// an archive, and the user has already told us they don't want this content.
    var excerpt: String
    var modelName: String
    var wasImage: Bool
}

@MainActor
final class ContentReportManager: ObservableObject {

    static let shared = ContentReportManager()

    @Published private(set) var reports: [ContentReport] = []

    private let storageKey = "DarkAI_ContentReports"

    static let reasons = [
        "Sexual content involving a minor",
        "Sexually explicit content",
        "Graphic violence or gore",
        "Harassment or hate speech",
        "Depicts a real person without consent",
        "Encourages self-harm",
        "Dangerous or illegal instructions",
        "Something else"
    ]

    private init() { load() }

    private func load() {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let decoded = try? JSONDecoder().decode([ContentReport].self, from: data) else { return }
        reports = decoded
    }

    private func save() {
        do {
            let encoded = try JSONEncoder().encode(reports)
            UserDefaults.standard.set(encoded, forKey: storageKey)
        } catch {
            LogManager.shared.log("ContentReportManager: failed to encode \(reports.count) reports for save — \(error.localizedDescription)")
        }
    }

    func file(reason: String, note: String, content: String, modelName: String, wasImage: Bool) -> ContentReport {
        let report = ContentReport(
            reason: reason,
            note: note,
            excerpt: String(content.prefix(500)),
            modelName: modelName,
            wasImage: wasImage
        )
        reports.insert(report, at: 0)
        // Cap the local log. Reports are a user-facing receipt, not a growing liability.
        if reports.count > 50 { reports = Array(reports.prefix(50)) }
        save()
        LogManager.shared.log("ContentReport filed: \(reason)")
        return report
    }

    func clearAll() {
        reports.removeAll()
        save()
    }

    // MARK: Outgoing mail

    func mailSubject(for report: ContentReport) -> String {
        "[\(AppInfo.displayName)] Content report — \(report.reason)"
    }

    func mailBody(for report: ContentReport) -> String {
        """
        A user filed a content report from inside \(AppInfo.displayName).

        Reason: \(report.reason)
        Date: \(report.date.formatted(date: .abbreviated, time: .shortened))
        Model in use: \(report.modelName)
        Type: \(report.wasImage ? "Generated image" : "Generated text")
        App: \(AppInfo.versionString)
        Device: \(UIDevice.current.systemName) \(UIDevice.current.systemVersion)

        User's note:
        \(report.note.isEmpty ? "(none)" : report.note)

        Content excerpt:
        \(report.wasImage ? "(image — not included)" : report.excerpt)
        """
    }
}

// MARK: - Diagnostic log attachment

/// Builds the optional diagnostic-log attachment shared by the content-report and crash-report
/// flows.
///
/// Always opt-in and always previewable before sending. The log holds technical events only —
/// model loads, memory decisions, filter *categories* but never the text that triggered them —
/// and that boundary is worth preserving deliberately rather than by accident, since this is the
/// one thing in the app that can travel off-device.
enum DiagnosticAttachment {

    static let fileName = "darkai-diagnostics.txt"

    static var isAvailable: Bool {
        FileManager.default.fileExists(atPath: AppFiles.diagnosticLogFile.path)
    }

    /// The tail of the log. Capped so a long-running install doesn't produce an attachment the
    /// mail composer chokes on.
    static func contents(maxLines: Int = 500) -> String? {
        guard let raw = try? String(contentsOf: AppFiles.diagnosticLogFile, encoding: .utf8) else { return nil }
        let lines = raw.components(separatedBy: "\n").filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        guard !lines.isEmpty else { return nil }
        let header = """
        \(AppInfo.displayName) diagnostic log
        \(AppInfo.versionString) · \(CrashReporter.deviceIdentifier()) · \(UIDevice.current.systemName) \(UIDevice.current.systemVersion)
        Last \(min(maxLines, lines.count)) entries

        """
        return header + lines.suffix(maxLines).joined(separator: "\n")
    }

    static func attachment() -> (data: Data, mimeType: String, fileName: String)? {
        guard let text = contents(), let data = text.data(using: .utf8) else { return nil }
        return (data, "text/plain", fileName)
    }
}

// MARK: - Report composer

struct ReportContentView: View {
    let content: String
    let modelName: String
    let wasImage: Bool
    /// Called when the report is filed, so the caller can remove the offending message.
    var onFiled: () -> Void

    @Environment(\.dismiss) private var dismiss
    @StateObject private var reports = ContentReportManager.shared

    @State private var selectedReason = ContentReportManager.reasons[0]
    @State private var note = ""
    @State private var attachDiagnostics = false
    @State private var showLogPreview = false
    @State private var filedReport: ContentReport?
    @State private var showMailComposer = false
    @State private var showMailUnavailable = false

    var body: some View {
        NavigationView {
            ZStack {
                Theme.background.ignoresSafeArea()

                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {

                        Text("What's wrong with this content?")
                            .font(.system(size: 15, weight: .bold))
                            .foregroundColor(Theme.textPrimary)

                        VStack(spacing: 0) {
                            ForEach(ContentReportManager.reasons, id: \.self) { reason in
                                Button {
                                    selectedReason = reason
                                } label: {
                                    HStack(spacing: 12) {
                                        Image(systemName: selectedReason == reason ? "largecircle.fill.circle" : "circle")
                                            .foregroundColor(selectedReason == reason ? Theme.accent : Theme.textMuted)
                                        Text(reason)
                                            .font(.system(size: 14))
                                            .foregroundColor(Theme.textPrimary)
                                            .multilineTextAlignment(.leading)
                                        Spacer()
                                    }
                                    .padding(.vertical, 11)
                                    .contentShape(Rectangle())
                                }
                                if reason != ContentReportManager.reasons.last {
                                    Divider().background(Theme.border)
                                }
                            }
                        }
                        .padding(.horizontal, 14)
                        .glassCard(cornerRadius: 14)

                        VStack(alignment: .leading, spacing: 8) {
                            Text("Anything else we should know? (optional)")
                                .font(.system(size: 13))
                                .foregroundColor(Theme.textSecondary)
                            TextEditor(text: $note)
                                .scrollContentBackground(.hidden)
                                .font(.system(size: 13))
                                .foregroundColor(Theme.textPrimary)
                                .frame(height: 90)
                                .padding(6)
                                .background(Theme.background.opacity(0.5))
                                .cornerRadius(8)
                                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Theme.border, lineWidth: 1))
                        }

                        if DiagnosticAttachment.isAvailable {
                            VStack(alignment: .leading, spacing: 8) {
                                Toggle(isOn: $attachDiagnostics) {
                                    Text("Attach diagnostic log")
                                        .font(.system(size: 13))
                                        .foregroundColor(Theme.textPrimary)
                                }
                                .toggleStyle(SwitchToggleStyle(tint: Theme.accent))

                                Text("Technical events only — model loads, memory decisions, and which filter category fired. Never your prompts or conversations.")
                                    .font(.system(size: 11))
                                    .foregroundColor(Theme.textMuted)
                                    .fixedSize(horizontal: false, vertical: true)

                                Button("Review what would be sent") { showLogPreview = true }
                                    .font(.system(size: 12, weight: .semibold))
                                    .foregroundColor(Theme.accentCyan)
                            }
                            .padding(12)
                            .glassCard(cornerRadius: 12)
                        }

                        Text("Filing this report removes the content from your conversation. Reports go to \(AppInfo.supportEmail) and are reviewed within 24 hours.")
                            .font(.system(size: 12))
                            .foregroundColor(Theme.textMuted)
                            .lineSpacing(3)

                        Button {
                            submit()
                        } label: {
                            Text("Submit Report")
                                .font(.system(size: 15, weight: .bold))
                                .foregroundColor(Theme.onAccent)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 14)
                                .background(RoundedRectangle(cornerRadius: 12).fill(Theme.accent))
                        }
                    }
                    .padding()
                }
            }
            .navigationTitle("Report a Concern")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") { dismiss() }
                        .foregroundColor(Theme.textSecondary)
                }
            }
            .sheet(isPresented: $showMailComposer) {
                if let report = filedReport {
                    MailComposeView(
                        recipient: AppInfo.supportEmail,
                        subject: reports.mailSubject(for: report),
                        body: reports.mailBody(for: report),
                        attachments: attachDiagnostics
                            ? [DiagnosticAttachment.attachment()].compactMap { $0 }
                            : []
                    ) { _ in
                        dismiss()
                    }
                }
            }
            .sheet(isPresented: $showLogPreview) {
                NavigationView {
                    PolicyTextView(
                        title: "Diagnostic Log",
                        text: DiagnosticAttachment.contents() ?? "The log is empty."
                    )
                    .toolbar {
                        ToolbarItem(placement: .navigationBarTrailing) {
                            Button("Done") { showLogPreview = false }
                                .foregroundColor(Theme.accentCyan)
                        }
                    }
                }
            }
            .alert("Report Saved", isPresented: $showMailUnavailable) {
                Button("Done") { dismiss() }
                Button("Copy Details") {
                    if let report = filedReport {
                        UIPasteboard.general.string = reports.mailBody(for: report)
                    }
                    dismiss()
                }
            } message: {
                Text("The content was removed and your report is saved in Settings → Safety & Legal. This device has no Mail account set up, so it couldn't be sent automatically — you can email \(AppInfo.supportEmail) directly.")
            }
        }
    }

    private func submit() {
        let report = reports.file(
            reason: selectedReason,
            note: note,
            content: content,
            modelName: modelName,
            wasImage: wasImage
        )
        filedReport = report
        onFiled()

        if MFMailComposeViewController.canSendMail() {
            showMailComposer = true
        } else {
            showMailUnavailable = true
        }
    }
}

// MARK: - Mail compose bridge

struct MailComposeView: UIViewControllerRepresentable {
    let recipient: String
    let subject: String
    let body: String
    /// Optional file attachments, as (data, mimeType, filename).
    var attachments: [(data: Data, mimeType: String, fileName: String)] = []
    var onFinish: (MFMailComposeResult) -> Void

    func makeUIViewController(context: Context) -> MFMailComposeViewController {
        let controller = MFMailComposeViewController()
        controller.mailComposeDelegate = context.coordinator
        controller.setToRecipients([recipient])
        controller.setSubject(subject)
        controller.setMessageBody(body, isHTML: false)
        for attachment in attachments {
            controller.addAttachmentData(attachment.data,
                                         mimeType: attachment.mimeType,
                                         fileName: attachment.fileName)
        }
        return controller
    }

    func updateUIViewController(_ uiViewController: MFMailComposeViewController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(onFinish: onFinish) }

    final class Coordinator: NSObject, MFMailComposeViewControllerDelegate {
        let onFinish: (MFMailComposeResult) -> Void
        init(onFinish: @escaping (MFMailComposeResult) -> Void) { self.onFinish = onFinish }

        func mailComposeController(_ controller: MFMailComposeViewController,
                                   didFinishWith result: MFMailComposeResult,
                                   error: Error?) {
            controller.dismiss(animated: true) { self.onFinish(result) }
        }
    }
}

// MARK: - Crash report

/// Shown when the previous run ended badly. Presents the whole report verbatim before anything
/// can be sent — for an app whose pitch is "nothing leaves your device", the user has to be able
/// to see exactly what would leave it.
struct CrashReportView: View {
    let report: CrashReporter.Report
    /// Called when the user is finished with this report and it should be cleared.
    var onDismissReport: () -> Void

    @Environment(\.dismiss) private var dismiss

    @State private var attachDiagnostics = false
    @State private var showMailComposer = false
    @State private var showMailUnavailable = false

    var body: some View {
        NavigationView {
            ZStack {
                Theme.background.ignoresSafeArea()

                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {

                        HStack(spacing: 10) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundColor(.orange)
                                .font(.system(size: 26))
                            VStack(alignment: .leading, spacing: 3) {
                                Text(report.kind)
                                    .font(.system(size: 16, weight: .bold))
                                    .foregroundColor(Theme.textPrimary)
                                Text(report.date.formatted(date: .abbreviated, time: .shortened))
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundColor(Theme.textMuted)
                            }
                        }

                        Text(report.reason)
                            .font(.system(size: 14))
                            .foregroundColor(Theme.textSecondary)
                            .lineSpacing(4)
                            .fixedSize(horizontal: false, vertical: true)

                        // The memory reading is the actionable part for this app specifically:
                        // a low number here means the model was too big for the device, which is
                        // something the user can fix themselves.
                        if report.availableMemoryMB > 0 && report.availableMemoryMB < 600 {
                            HStack(alignment: .top, spacing: 8) {
                                Image(systemName: "memorychip")
                                    .foregroundColor(.orange)
                                Text("Only \(report.availableMemoryMB) MB of memory was free beforehand. Try a smaller model, or lower the context window in Settings.")
                                    .font(.system(size: 12))
                                    .foregroundColor(.orange)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            .padding(12)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.orange.opacity(0.1))
                            .cornerRadius(10)
                        }

                        VStack(alignment: .leading, spacing: 8) {
                            Text("REPORT CONTENTS")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundColor(Theme.textSecondary)
                                .kerning(1.0)
                            ScrollView(.horizontal) {
                                Text(report.formattedForEmail)
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundColor(Theme.textSecondary)
                                    .textSelection(.enabled)
                            }
                            .frame(maxHeight: 240)
                        }
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(white: 0.08))
                        .cornerRadius(10)

                        if DiagnosticAttachment.isAvailable {
                            Toggle(isOn: $attachDiagnostics) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("Attach diagnostic log")
                                        .font(.system(size: 13))
                                        .foregroundColor(Theme.textPrimary)
                                    Text("Adds the technical event log. No prompts or conversations.")
                                        .font(.system(size: 11))
                                        .foregroundColor(Theme.textMuted)
                                }
                            }
                            .toggleStyle(SwitchToggleStyle(tint: Theme.accent))
                        }

                        Text("Nothing is sent automatically. Tapping Send opens your mail app with this report filled in, and you can edit or discard it there.")
                            .font(.system(size: 12))
                            .foregroundColor(Theme.textMuted)
                            .lineSpacing(3)

                        Button {
                            send()
                        } label: {
                            Text("Send Report")
                                .font(.system(size: 15, weight: .bold))
                                .foregroundColor(Theme.onAccent)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 14)
                                .background(RoundedRectangle(cornerRadius: 12).fill(Theme.accent))
                        }

                        Button {
                            onDismissReport()
                            dismiss()
                        } label: {
                            Text("Don't Send")
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundColor(Theme.textSecondary)
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .padding()
                }
            }
            .navigationTitle("Crash Report")
            .navigationBarTitleDisplayMode(.inline)
            .sheet(isPresented: $showMailComposer) {
                MailComposeView(
                    recipient: AppInfo.supportEmail,
                    subject: "[\(AppInfo.displayName)] Crash report — \(report.kind)",
                    body: report.formattedForEmail,
                    attachments: attachDiagnostics
                        ? [DiagnosticAttachment.attachment()].compactMap { $0 }
                        : []
                ) { _ in
                    onDismissReport()
                    dismiss()
                }
            }
            .alert("No Mail Account", isPresented: $showMailUnavailable) {
                Button("Copy Report") {
                    UIPasteboard.general.string = report.formattedForEmail
                    onDismissReport()
                    dismiss()
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This device has no Mail account set up. Copy the report and email it to \(AppInfo.supportEmail).")
            }
        }
    }

    private func send() {
        if MFMailComposeViewController.canSendMail() {
            showMailComposer = true
        } else {
            showMailUnavailable = true
        }
    }
}

// MARK: - Safety & legal hub

struct SafetyLegalView: View {
    @StateObject private var reports = ContentReportManager.shared
    @State private var showClearReportsAlert = false
    @State private var crashReportingEnabled = CrashReporter.isEnabled
    @State private var pendingCrash: CrashReporter.Report? = CrashReporter.pendingReport()
    @State private var presentedCrash: CrashReporter.Report? = nil

    var body: some View {
        ZStack {
            Theme.background.ignoresSafeArea()

            ScrollView {
                VStack(spacing: 20) {

                    VStack(alignment: .leading, spacing: 10) {
                        HStack(spacing: 8) {
                            Image(systemName: "checkmark.shield.fill")
                                .foregroundColor(.green)
                            Text("CONTENT FILTER: ALWAYS ON")
                                .font(.system(size: 13, weight: .bold))
                                .foregroundColor(Theme.textPrimary)
                                .kerning(1.0)
                        }
                        Text("Every prompt is screened before it reaches a model, and every response is screened before you see it. Requests that sexualize minors are blocked outright. This filter has no off switch and applies to models you import yourself.")
                            .font(.system(size: 13))
                            .foregroundColor(Theme.textSecondary)
                            .lineSpacing(4)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .glassCard(cornerRadius: 16)

                    VStack(spacing: 0) {
                        policyLink("Acceptable Use Policy", icon: "hand.raised.fill", text: LegalText.acceptableUse)
                        Divider().background(Theme.border)
                        policyLink("Terms of Use", icon: "doc.text.fill", text: LegalText.termsOfUse)
                        Divider().background(Theme.border)
                        policyLink("Privacy Policy", icon: "lock.fill", text: LegalText.privacyPolicy)
                        Divider().background(Theme.border)
                        Link(destination: AppInfo.appleStandardEULA) {
                            rowLabel("Apple Standard EULA", icon: "link", trailing: "arrow.up.right.square")
                        }
                    }
                    .glassCard(cornerRadius: 16)

                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Image(systemName: "flag.fill")
                                .foregroundColor(Theme.accentRose)
                            Text("YOUR REPORTS")
                                .font(.system(size: 13, weight: .bold))
                                .foregroundColor(Theme.textPrimary)
                                .kerning(1.0)
                            Spacer()
                            if !reports.reports.isEmpty {
                                Button("Clear") { showClearReportsAlert = true }
                                    .font(.system(size: 12, weight: .semibold))
                                    .foregroundColor(.red.opacity(0.8))
                            }
                        }

                        if reports.reports.isEmpty {
                            Text("You haven't reported anything. To report something the app generated, touch and hold the message and choose \"Report a Concern\".")
                                .font(.system(size: 13))
                                .foregroundColor(Theme.textMuted)
                                .lineSpacing(4)
                        } else {
                            ForEach(reports.reports) { report in
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(report.reason)
                                        .font(.system(size: 13, weight: .semibold))
                                        .foregroundColor(Theme.textPrimary)
                                    Text(report.date.formatted(date: .abbreviated, time: .shortened))
                                        .font(.system(size: 11, design: .monospaced))
                                        .foregroundColor(Theme.textMuted)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(10)
                                .background(Theme.background.opacity(0.4))
                                .cornerRadius(8)
                            }
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .glassCard(cornerRadius: 16)

                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Image(systemName: "ladybug.fill")
                                .foregroundColor(.orange)
                            Text("CRASH REPORTS")
                                .font(.system(size: 13, weight: .bold))
                                .foregroundColor(Theme.textPrimary)
                                .kerning(1.0)
                            Spacer()
                        }

                        Toggle(isOn: $crashReportingEnabled) {
                            Text("Detect unexpected closures")
                                .font(.system(size: 13))
                                .foregroundColor(Theme.textPrimary)
                        }
                        .toggleStyle(SwitchToggleStyle(tint: Theme.accent))
                        .onChange(of: crashReportingEnabled) { _, newValue in
                            CrashReporter.isEnabled = newValue
                        }

                        Text("Writes a technical report to this device when the app closes unexpectedly — crash type, stack trace, and free memory. No prompts or conversations. Nothing is ever sent unless you choose to send it.")
                            .font(.system(size: 12))
                            .foregroundColor(Theme.textMuted)
                            .lineSpacing(3)

                        if let crash = pendingCrash {
                            Button {
                                presentedCrash = crash
                            } label: {
                                HStack {
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text("Report waiting to be sent")
                                            .font(.system(size: 13, weight: .semibold))
                                            .foregroundColor(.orange)
                                        Text(crash.summary)
                                            .font(.system(size: 11))
                                            .foregroundColor(Theme.textMuted)
                                            .lineLimit(2)
                                    }
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .font(.system(size: 12))
                                        .foregroundColor(Theme.textSecondary)
                                }
                                .padding(12)
                                .background(Color.orange.opacity(0.1))
                                .cornerRadius(10)
                            }
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .glassCard(cornerRadius: 16)

                    VStack(alignment: .leading, spacing: 10) {
                        Text("SUPPORT")
                            .font(.system(size: 13, weight: .bold))
                            .foregroundColor(Theme.textPrimary)
                            .kerning(1.0)
                        Text("Questions, bug reports, or content concerns:")
                            .font(.system(size: 13))
                            .foregroundColor(Theme.textSecondary)
                        if let url = URL(string: "mailto:\(AppInfo.supportEmail)") {
                            Link(AppInfo.supportEmail, destination: url)
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundColor(Theme.accentCyan)
                        }
                        Text(AppInfo.versionString)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(Theme.textMuted)
                            .padding(.top, 4)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .glassCard(cornerRadius: 16)
                }
                .padding()
            }
        }
        .navigationTitle("Safety & Legal")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(item: $presentedCrash) { crash in
            CrashReportView(report: crash) {
                CrashReporter.clearPendingReport()
                pendingCrash = nil
            }
        }
        .alert("Clear all reports?", isPresented: $showClearReportsAlert) {
            Button("Cancel", role: .cancel) {}
            Button("Clear", role: .destructive) { reports.clearAll() }
        } message: {
            Text("This removes your local copy of the reports you've filed. Reports you already emailed are unaffected.")
        }
    }

    @ViewBuilder
    private func policyLink(_ title: String, icon: String, text: String) -> some View {
        NavigationLink {
            PolicyTextView(title: title, text: text)
        } label: {
            rowLabel(title, icon: icon, trailing: "chevron.right")
        }
    }

    private func rowLabel(_ title: String, icon: String, trailing: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(Theme.accentCyan)
                .frame(width: 20)
            Text(title)
                .font(.system(size: 14, weight: .medium))
                .foregroundColor(Theme.textPrimary)
            Spacer()
            Image(systemName: trailing)
                .font(.system(size: 12))
                .foregroundColor(Theme.textSecondary)
        }
        .padding(.vertical, 14)
        .contentShape(Rectangle())
    }
}

struct PolicyTextView: View {
    let title: String
    let text: String

    var body: some View {
        ZStack {
            Theme.background.ignoresSafeArea()
            ScrollView {
                Text(text)
                    .font(.system(size: 13))
                    .foregroundColor(Theme.textSecondary)
                    .lineSpacing(5)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding()
            }
        }
        .navigationTitle(title)
        .navigationBarTitleDisplayMode(.inline)
    }
}
