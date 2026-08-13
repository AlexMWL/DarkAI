import Foundation
import SwiftUI
import UniformTypeIdentifiers

/// Turns a conversation (or a single message) into a file the user can send somewhere else.
///
/// Everything this app produces lives in its own container and is deleted with the app — which is
/// the privacy promise, but it also means a conversation worth keeping has no way out. Export is
/// that way out, and it is deliberately the *only* thing here that moves content off the device:
/// nothing is uploaded, the file is written to the app's temporary directory and handed to the
/// system share sheet, and where it goes from there is the user's choice.
///
/// Explicitly `nonisolated` for the same reason `AppFiles` is: this module builds with
/// `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`, and encoding a long transcript with embedded
/// image data is not work that belongs on the UI thread.
nonisolated enum ConversationExport {

    // MARK: - Formats

    enum Format: String, CaseIterable, Identifiable {
        /// Headings and speaker labels. What you want if a human is going to read it.
        case markdown
        /// No syntax at all. Pastes cleanly into anything.
        case plainText
        /// Lossless and re-importable. See `StructuredImport` for the read side.
        case json

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .markdown:  return "Markdown"
            case .plainText: return "Plain Text"
            case .json:      return "JSON"
            }
        }

        var fileExtension: String {
            switch self {
            case .markdown:  return "md"
            case .plainText: return "txt"
            case .json:      return "json"
            }
        }

        var icon: String {
            switch self {
            case .markdown:  return "doc.richtext"
            case .plainText: return "doc.plaintext"
            case .json:      return "curlybraces"
            }
        }

        var summary: String {
            switch self {
            case .markdown:  return "Headings, timestamps, and speaker labels. Best for reading or pasting into notes."
            case .plainText: return "No formatting at all. Pastes cleanly into anything."
            case .json:      return "Every field, machine-readable — and the only format that can be imported back into the Mindscape."
            }
        }

        /// Only JSON can carry the image bytes; the text formats reference generated images by
        /// their prompt instead, so offering the toggle there would promise something it can't do.
        var supportsEmbeddedImages: Bool { self == .json }
    }

    /// Schema tag written into every JSON export and checked on the way back in. Versioned so a
    /// future change to the message model can be recognised rather than guessed at.
    static let jsonSchemaIdentifier = "darkai.conversation.v1"

    // MARK: - Errors

    enum ExportError: LocalizedError {
        case emptyConversation
        case encodingFailed

        var errorDescription: String? {
            switch self {
            case .emptyConversation: return "This chat has no messages to export yet."
            case .encodingFailed:    return "The transcript couldn't be encoded. Please try a different format."
            }
        }
    }

    // MARK: - Entry point

    /// Writes `conversation` to a file in the app's temporary directory and returns its URL.
    ///
    /// - Parameters:
    ///   - sanitize: applied to assistant text before it is written. The chat view strips
    ///     model reasoning tags (`<think>…`) before display, and an export that reintroduced them
    ///     would not match what the user actually read. Passed in rather than reimplemented so
    ///     there is exactly one copy of that logic.
    ///   - includeImages: embeds generated images as base64. Ignored by the text formats, which
    ///     have nowhere to put them.
    static func makeFile(
        for conversation: Conversation,
        format: Format,
        includeImages: Bool = true,
        sanitize: (String) -> String = { $0 }
    ) throws -> URL {
        guard !conversation.messages.isEmpty else { throw ExportError.emptyConversation }

        let data: Data
        switch format {
        case .markdown:
            data = Data(markdown(for: conversation, sanitize: sanitize).utf8)
        case .plainText:
            data = Data(plainText(for: conversation, sanitize: sanitize).utf8)
        case .json:
            data = try json(for: conversation, includeImages: includeImages, sanitize: sanitize)
        }

        let directory = exportDirectory()
        let url = directory.appendingPathComponent("\(fileNameStem(for: conversation)).\(format.fileExtension)")
        try? FileManager.default.removeItem(at: url)
        try data.write(to: url, options: .atomic)
        return url
    }

    /// A single message rendered for the share sheet. Used by the per-message action, where a
    /// whole file would be overkill — the share sheet takes the string directly.
    static func shareText(for message: ChatMessage, sanitize: (String) -> String = { $0 }) -> String {
        let body = message.isUser ? message.text : sanitize(message.text)
        if message.isImageMessage {
            let prompt = body.isEmpty ? "(no prompt recorded)" : body
            return "Image generated by \(AppInfo.displayName) on \(Self.readableDate.string(from: message.timestamp))\nPrompt: \(prompt)"
        }
        return body
    }

    /// Estimated size of the finished file, for the confirmation UI. Embedded images dominate it
    /// completely, and a user about to mail a transcript deserves to know it is 40 MB before they
    /// pick a recipient rather than after.
    static func estimatedByteCount(
        for conversation: Conversation,
        format: Format,
        includeImages: Bool
    ) -> Int {
        let textBytes = conversation.messages.reduce(0) { $0 + $1.text.utf8.count + 64 }
        guard format.supportsEmbeddedImages, includeImages else { return textBytes }
        // base64 is 4 bytes out for every 3 in.
        let imageBytes = conversation.messages.reduce(0) { $0 + (($1.imageData?.count ?? 0) * 4 / 3) }
        return textBytes + imageBytes
    }

    // MARK: - Renderers

    private static func markdown(for conversation: Conversation, sanitize: (String) -> String) -> String {
        var out = "# \(conversation.title)\n\n"
        out += "*Exported from \(AppInfo.displayName) \(AppInfo.version) on \(readableDate.string(from: Date()))*\n\n"
        out += "> Responses in this transcript were generated by a language model running on the "
        out += "exporting device. They can be inaccurate or fabricated.\n\n"
        if conversation.isPrivate {
            // The privacy guarantee is "never written to disk by the app", not "can never be
            // exported" — but a transcript that was deliberately kept out of storage should say so
            // wherever it ends up.
            out += "> This was a private chat. It was never saved to the device; this file is the only copy.\n\n"
        }
        out += "---\n\n"

        for message in conversation.messages {
            let speaker = message.isUser ? "You" : AppInfo.displayName
            out += "### \(speaker) — \(readableDate.string(from: message.timestamp))\n\n"

            if message.isImageMessage {
                let prompt = message.text.isEmpty ? "(no prompt recorded)" : message.text
                out += "*[Generated image — not included in this format. Export as JSON to keep the image data.]*\n\n"
                out += "Prompt: \(prompt)\n\n"
            } else {
                let body = message.isUser ? message.text : sanitize(message.text)
                out += body.trimmingCharacters(in: .whitespacesAndNewlines) + "\n\n"
            }
        }
        return out
    }

    private static func plainText(for conversation: Conversation, sanitize: (String) -> String) -> String {
        var out = "\(conversation.title)\n"
        out += String(repeating: "=", count: max(3, conversation.title.count)) + "\n"
        out += "Exported from \(AppInfo.displayName) \(AppInfo.version) on \(readableDate.string(from: Date()))\n"
        out += "Responses were generated by a language model on-device and can be inaccurate.\n\n"

        for message in conversation.messages {
            let speaker = message.isUser ? "YOU" : AppInfo.displayName.uppercased()
            out += "[\(readableDate.string(from: message.timestamp))] \(speaker):\n"
            if message.isImageMessage {
                let prompt = message.text.isEmpty ? "(no prompt recorded)" : message.text
                out += "(generated image — prompt: \(prompt))\n\n"
            } else {
                let body = message.isUser ? message.text : sanitize(message.text)
                out += body.trimmingCharacters(in: .whitespacesAndNewlines) + "\n\n"
            }
        }
        return out
    }

    private static func json(
        for conversation: Conversation,
        includeImages: Bool,
        sanitize: (String) -> String
    ) throws -> Data {
        let payload = TranscriptFile(
            schema: jsonSchemaIdentifier,
            exportedAt: Date(),
            app: .init(name: AppInfo.displayName, version: AppInfo.version, build: AppInfo.build),
            conversation: .init(
                id: conversation.id.uuidString,
                title: conversation.title,
                createdAt: conversation.createdAt,
                wasPrivate: conversation.isPrivate,
                messages: conversation.messages.map { message in
                    TranscriptFile.Message(
                        id: message.id.uuidString,
                        role: message.isUser ? "user" : "assistant",
                        timestamp: message.timestamp,
                        text: message.isUser ? message.text : sanitize(message.text),
                        image: (includeImages ? message.imageData : nil).map {
                            .init(format: "jpeg", base64: $0.base64EncodedString())
                        }
                    )
                }
            )
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(payload) else { throw ExportError.encodingFailed }
        return data
    }

    // MARK: - JSON shape

    /// The on-disk shape of a `.json` export. Also the shape `StructuredImport` recognises when a
    /// transcript is brought back in, so the two must stay in step — hence one declaration rather
    /// than a dictionary literal on each side.
    struct TranscriptFile: Codable {
        struct App: Codable {
            let name: String
            let version: String
            let build: String
        }

        struct Image: Codable {
            let format: String
            let base64: String
        }

        struct Message: Codable {
            let id: String
            /// `"user"` or `"assistant"`.
            let role: String
            let timestamp: Date
            let text: String
            let image: Image?
        }

        struct Body: Codable {
            let id: String
            let title: String
            let createdAt: Date
            let wasPrivate: Bool
            let messages: [Message]
        }

        let schema: String
        let exportedAt: Date
        let app: App
        let conversation: Body
    }

    // MARK: - Files

    /// Exports land in their own subdirectory of `tmp/`, wiped on every export.
    ///
    /// `tmp/` is not backed up and iOS reclaims it under storage pressure, which is the right home
    /// for a file whose only job is to survive long enough to reach the share sheet. Wiping first
    /// matters more than it looks: without it, a transcript the user exported once stays readable
    /// in the container until iOS decides otherwise, which is not what "this app keeps nothing"
    /// should mean.
    private static func exportDirectory() -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("Exports", isDirectory: true)
        try? FileManager.default.removeItem(at: directory)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    /// Filename without extension. The conversation title is user-authored text that reaches the
    /// filesystem and then a share sheet, so it is reduced to a conservative character set rather
    /// than merely having `/` swapped out.
    private static func fileNameStem(for conversation: Conversation) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: " -_"))
        let cleaned = conversation.title.unicodeScalars
            .map { allowed.contains($0) ? Character($0) : "-" }
            .reduce(into: "") { $0.append($1) }
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let base = cleaned.isEmpty ? "Chat" : String(cleaned.prefix(48))
        return "\(AppInfo.displayName) - \(base) - \(fileStamp.string(from: Date()))"
    }

    private static let readableDate: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter
    }()

    private static let fileStamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HHmm"
        return formatter
    }()
}

// MARK: - Export sheet

/// Format picker plus the share button. Presented from the chat drawer.
///
/// The file is written when the user picks a format rather than up front, so opening this sheet on
/// a long conversation costs nothing and backing out of it leaves nothing behind.
struct ConversationExportSheet: View {

    let conversation: Conversation
    /// Assistant-text cleanup, supplied by the chat view. See `ConversationExport.makeFile`.
    let sanitize: (String) -> String

    @Environment(\.dismiss) private var dismiss

    @State private var format: ConversationExport.Format = .markdown
    @State private var includeImages = true
    @State private var preparedFile: URL?
    @State private var failure: String?

    private var hasImages: Bool { conversation.messages.contains { $0.isImageMessage } }

    var body: some View {
        NavigationView {
            ZStack {
                Theme.background.ignoresSafeArea()

                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        header

                        VStack(spacing: 10) {
                            ForEach(ConversationExport.Format.allCases) { option in
                                formatRow(option)
                            }
                        }

                        if hasImages && format.supportsEmbeddedImages {
                            Toggle(isOn: $includeImages) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("Include generated images")
                                        .font(.system(size: 13, weight: .semibold))
                                        .foregroundColor(Theme.textPrimary)
                                    Text("Embeds the image data in the file. Turning this off keeps only the prompts.")
                                        .font(.system(size: 11))
                                        .foregroundColor(Theme.textSecondary)
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                            }
                            .toggleStyle(SwitchToggleStyle(tint: Theme.accent))
                        } else if hasImages {
                            Label(
                                "Generated images can't be carried by \(format.displayName). Their prompts are kept; export as JSON to keep the images themselves.",
                                systemImage: "info.circle"
                            )
                            .font(.system(size: 11))
                            .foregroundColor(Theme.textMuted)
                            .fixedSize(horizontal: false, vertical: true)
                        }

                        if let failure {
                            Text(failure)
                                .font(.system(size: 12))
                                .foregroundColor(.orange)
                                .fixedSize(horizontal: false, vertical: true)
                        }

                        shareButton

                        Text("The file is written to this app's temporary storage and handed to the share sheet. Nothing is uploaded — where it goes next is up to you.")
                            .font(.system(size: 11))
                            .foregroundColor(Theme.textMuted)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(20)
                }
            }
            .navigationTitle("Export Chat")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                        .foregroundColor(Theme.accentCyan)
                }
            }
        }
        .onChange(of: format) { _, _ in preparedFile = nil; failure = nil }
        .onChange(of: includeImages) { _, _ in preparedFile = nil; failure = nil }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(conversation.title)
                .font(.system(size: 18, weight: .bold))
                .foregroundColor(Theme.textPrimary)
                .lineLimit(2)
            Text("\(conversation.messages.count) message\(conversation.messages.count == 1 ? "" : "s") · about \(sizeDescription)")
                .font(.system(size: 12, design: .monospaced))
                .foregroundColor(Theme.textSecondary)
        }
    }

    private var sizeDescription: String {
        let bytes = ConversationExport.estimatedByteCount(
            for: conversation, format: format, includeImages: includeImages
        )
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }

    private func formatRow(_ option: ConversationExport.Format) -> some View {
        Button {
            format = option
        } label: {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: option.icon)
                    .font(.system(size: 16))
                    .foregroundColor(format == option ? Theme.accent : Theme.textMuted)
                    .frame(width: 24)

                VStack(alignment: .leading, spacing: 3) {
                    Text(option.displayName)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(Theme.textPrimary)
                    Text(option.summary)
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .multilineTextAlignment(.leading)
                }

                Spacer(minLength: 4)

                Image(systemName: format == option ? "checkmark.circle.fill" : "circle")
                    .foregroundColor(format == option ? Theme.accent : Theme.textMuted)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.cardBackground)
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(format == option ? Theme.accent : Theme.border, lineWidth: 1)
            )
        }
    }

    @ViewBuilder
    private var shareButton: some View {
        if let preparedFile {
            ShareLink(item: preparedFile) {
                actionLabel("Share \(preparedFile.lastPathComponent)", icon: "square.and.arrow.up")
            }
        } else {
            Button {
                prepare()
            } label: {
                actionLabel("Prepare \(format.displayName) File", icon: format.icon)
            }
        }
    }

    private func actionLabel(_ title: String, icon: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
            Text(title)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .font(.system(size: 15, weight: .bold))
        .foregroundColor(Theme.onAccent)
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity)
        .background(RoundedRectangle(cornerRadius: 14).fill(Theme.accent))
    }

    private func prepare() {
        failure = nil
        do {
            preparedFile = try ConversationExport.makeFile(
                for: conversation,
                format: format,
                includeImages: includeImages,
                sanitize: sanitize
            )
        } catch {
            preparedFile = nil
            failure = error.localizedDescription
            LogManager.shared.log("Export failed: \(error.localizedDescription)")
        }
    }
}
