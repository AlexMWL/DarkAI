import Foundation
import SwiftUI

/// Reads a JSON file into Mindscape documents.
///
/// The Mindscape's existing importer takes plain text, which works for prose and is useless for
/// anything with structure: a study guide dumped as raw JSON retrieves as a soup of braces and
/// key names, and a journal loses the one thing that makes its entries findable — their dates.
/// This parses the structure first and writes readable text second, so retrieval sees
/// "Date: 2026-03-04 … Mood: tired" rather than `{"d":"2026-03-04","m":"tired"}`.
///
/// **Deliberately tolerant about shape.** Nobody is going to hand-author a schema to feed their
/// own notes into a phone app; the file is going to come out of Anki, Notion, Day One, Obsidian,
/// a spreadsheet export, or a model that was asked to "put this in JSON". So the parser sniffs
/// the shape rather than demanding one, accepts the common key spellings for each field, and
/// tells the user exactly what it decided the file was — see `Outcome.summary`.
///
/// `nonisolated` for the same reason as `AppFiles` and `ConversationExport`: parsing a
/// multi-megabyte file is background work, not UI work.
nonisolated enum StructuredImport {

    // MARK: - Limits
    //
    // Mindscape documents are persisted in `UserDefaults` (see `RAGManager.saveDocuments`), which
    // is a property list read into memory in full on every launch. It is the wrong store for bulk
    // text and these caps are what keep a 40 MB journal export from turning into a launch that
    // never finishes. They are enforced with a message rather than silently, because a journal
    // that imported "successfully" minus half its entries is worse than one that refused.

    /// Largest file accepted at all, before parsing.
    static let maxFileBytes = 8 * 1024 * 1024
    /// Most documents a single import may create.
    static let maxDocuments = 400
    /// Total characters of extracted text a single import may add.
    static let maxTotalCharacters = 750_000

    // MARK: - Results

    enum Kind: Sendable {
        case studyGuide
        case journal
        case transcript

        var label: String {
            switch self {
            case .studyGuide: return "Study guide"
            case .journal:    return "Journal"
            case .transcript: return "Chat transcript"
            }
        }

        var icon: String {
            switch self {
            case .studyGuide: return "graduationcap.fill"
            case .journal:    return "book.closed.fill"
            case .transcript: return "bubble.left.and.bubble.right.fill"
            }
        }
    }

    /// One Mindscape document waiting to be ingested.
    struct Document: Sendable {
        let name: String
        let content: String
    }

    struct Outcome: Sendable {
        let kind: Kind
        let title: String
        let documents: [Document]
        /// Entries that were present but unusable — no question, no body, not an object.
        let skippedEntries: Int
        /// Entries dropped because a cap above was hit, rather than because they were malformed.
        let truncatedEntries: Int

        var summary: String {
            var parts = ["\(kind.label) · \(documents.count) entr\(documents.count == 1 ? "y" : "ies") added"]
            if skippedEntries > 0 { parts.append("\(skippedEntries) skipped (no readable content)") }
            if truncatedEntries > 0 { parts.append("\(truncatedEntries) not imported (file too large)") }
            return parts.joined(separator: " · ")
        }
    }

    enum ImportError: LocalizedError, Sendable {
        case tooLarge(Int)
        case notJSON
        case unrecognizedShape
        case nothingUsable
        case blockedByContentFilter(String)

        var errorDescription: String? {
            switch self {
            case .tooLarge(let bytes):
                let size = ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
                let cap = ByteCountFormatter.string(fromByteCount: Int64(maxFileBytes), countStyle: .file)
                return "That file is \(size). The largest that can be imported is \(cap) — try splitting it up."
            case .notJSON:
                return "That file isn't valid JSON. Check it opens in a text editor and starts with { or [."
            case .unrecognizedShape:
                return """
                The JSON parsed, but nothing in it looked like study-guide cards or journal entries. \
                Tap "What can I import?" for the shapes this accepts.
                """
            case .nothingUsable:
                return "The file was read, but every entry in it was empty. Nothing was imported."
            case .blockedByContentFilter(let reason):
                return "This file wasn't imported: \(reason)"
            }
        }
    }

    // MARK: - Entry point

    /// Parses `data` and returns the documents it should become. Does not ingest anything —
    /// the caller decides whether to commit, so a refused import leaves the Mindscape untouched.
    static func parse(data: Data, fileName: String) throws -> Outcome {
        guard data.count <= maxFileBytes else { throw ImportError.tooLarge(data.count) }

        guard let root = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) else {
            throw ImportError.notJSON
        }

        let sourceName = (fileName as NSString).deletingPathExtension
        let outcome = try interpret(root, sourceName: sourceName)

        guard !outcome.documents.isEmpty else { throw ImportError.nothingUsable }

        // Imported text becomes retrieval context — it is fed to the model on future turns exactly
        // like a typed prompt is, so it gets the same screening a typed prompt gets. Reviewed as a
        // whole rather than per entry: a file split across entries to slip past a per-entry check
        // would otherwise be the obvious way through.
        let combined = outcome.documents.map(\.content).joined(separator: "\n\n")
        let decision = ContentSafety.review(combined, surface: .chatPrompt)
        if !decision.isAllowed {
            throw ImportError.blockedByContentFilter(decision.message ?? "It contains content this app does not allow.")
        }

        return outcome
    }

    // MARK: - Shape detection

    private static func interpret(_ root: Any, sourceName: String) throws -> Outcome {
        // 1. The app's own transcript export. Recognised by its schema tag, so a re-import is
        //    never mistaken for a journal that happens to have `text` fields.
        if let object = root as? [String: Any],
           let schema = object["schema"] as? String,
           schema == ConversationExport.jsonSchemaIdentifier {
            return transcriptOutcome(object, sourceName: sourceName)
        }

        // 2. An explicit declaration, if the file bothers to make one.
        let declared = declaredKind(root)

        // 3. Find the array of entries, wherever it is.
        let (entries, containerTitle) = locateEntries(root)
        guard !entries.isEmpty else { throw ImportError.unrecognizedShape }

        let title = containerTitle ?? sourceName
        let kind = declared ?? inferKind(from: entries)

        switch kind {
        case .studyGuide: return studyGuideOutcome(entries, title: title)
        case .journal:    return journalOutcome(entries, title: title)
        case .transcript: return studyGuideOutcome(entries, title: title)   // unreachable via sniffing
        }
    }

    /// Honours a `type` / `kind` field if the file carries one. Anything else falls through to
    /// structural inference — a wrong label is more likely than a wrong shape.
    private static func declaredKind(_ root: Any) -> Kind? {
        guard let object = root as? [String: Any] else { return nil }
        let declared = ["type", "kind", "format", "documentType"]
            .compactMap { object[$0] as? String }
            .first?
            .lowercased()
        guard let declared else { return nil }
        if declared.contains("study") || declared.contains("flashcard") || declared.contains("quiz") || declared.contains("deck") {
            return .studyGuide
        }
        if declared.contains("journal") || declared.contains("diary") || declared.contains("log") {
            return .journal
        }
        return nil
    }

    /// Decides between the two shapes by looking at what the entries actually contain.
    ///
    /// Journal wins ties on purpose. A journal entry with a `title` and a `body` also technically
    /// satisfies "has a question-ish key and an answer-ish key", so scoring question/answer first
    /// would misfile every diary as a flashcard deck; requiring a date-like field for journal and
    /// checking it first makes the two tests actually disjoint.
    private static func inferKind(from entries: [[String: Any]]) -> Kind {
        let sample = entries.prefix(20)
        guard !sample.isEmpty else { return .studyGuide }

        let journalish = sample.filter { entry in
            value(in: entry, forAnyOf: dateKeys) != nil && value(in: entry, forAnyOf: bodyKeys) != nil
        }.count
        if Double(journalish) >= Double(sample.count) * 0.5 { return .journal }

        let cardish = sample.filter { entry in
            value(in: entry, forAnyOf: questionKeys) != nil && value(in: entry, forAnyOf: answerKeys) != nil
        }.count
        if Double(cardish) >= Double(sample.count) * 0.5 { return .studyGuide }

        // Body but no date: prose without timestamps. Reads far better as a journal than as
        // flashcards with empty answers.
        return value(in: sample[sample.startIndex], forAnyOf: bodyKeys) != nil ? .journal : .studyGuide
    }

    /// Walks the JSON looking for the first array of objects that could be entries.
    ///
    /// Handles the three layouts that actually turn up: a bare top-level array, an object with a
    /// named array (`{"cards": [...]}`), and a sectioned document (`{"sections":[{"title":…,
    /// "cards":[…]}]}`) — the last flattened, with each section's title carried onto its entries
    /// so a card still says which chapter it came from once it is a standalone document.
    private static func locateEntries(_ root: Any) -> ([[String: Any]], String?) {
        if let array = root as? [Any] {
            return (array.compactMap { $0 as? [String: Any] }, nil)
        }

        guard let object = root as? [String: Any] else { return ([], nil) }
        let title = value(in: object, forAnyOf: titleKeys)

        // A named array of entry objects, in preference order.
        for key in containerKeys {
            guard let array = matchingValue(in: object, forKey: key) as? [Any] else { continue }
            let objects = array.compactMap { $0 as? [String: Any] }
            guard !objects.isEmpty else { continue }

            // Sectioned: the "entries" are themselves containers holding the real entries.
            let flattened = flattenSections(objects)
            return (flattened ?? objects, title)
        }

        // Last resort: any array of objects anywhere in the top level. Sorted by key rather than
        // walked in the dictionary's own iteration order, which Swift leaves unspecified (and
        // varies by process, via hash-seed randomization) — without the sort, importing the exact
        // same file could pick a different array, and so produce different content, on different
        // launches.
        for key in object.keys.sorted() {
            if let array = object[key] as? [Any] {
                let objects = array.compactMap { $0 as? [String: Any] }
                if !objects.isEmpty { return (flattenSections(objects) ?? objects, title) }
            }
        }
        return ([], title)
    }

    /// Returns flattened entries if `objects` are section wrappers, `nil` if they are entries.
    private static func flattenSections(_ objects: [[String: Any]]) -> [[String: Any]]? {
        var flattened: [[String: Any]] = []
        for section in objects {
            guard let nested = containerKeys.lazy
                .compactMap({ matchingValue(in: section, forKey: $0) as? [Any] })
                .first else { return nil }
            let sectionTitle = value(in: section, forAnyOf: titleKeys)
            for case let entry as [String: Any] in nested {
                var carried = entry
                if let sectionTitle, carried["__section"] == nil { carried["__section"] = sectionTitle }
                flattened.append(carried)
            }
        }
        return flattened.isEmpty ? nil : flattened
    }

    // MARK: - Study guide

    private static func studyGuideOutcome(_ entries: [[String: Any]], title: String) -> Outcome {
        var lines: [String] = []
        var skipped = 0
        var index = 0
        var truncated = 0
        var characters = 0

        for entry in entries {
            let question = value(in: entry, forAnyOf: questionKeys)
            let answer = value(in: entry, forAnyOf: answerKeys) ?? value(in: entry, forAnyOf: bodyKeys)
            guard let question, !question.isEmpty else { skipped += 1; continue }

            index += 1
            var block = "\(index). \(question)"
            if let answer, !answer.isEmpty { block += "\n   Answer: \(answer)" }
            if let section = entry["__section"] as? String { block += "\n   Section: \(section)" }
            if let hint = value(in: entry, forAnyOf: hintKeys), !hint.isEmpty { block += "\n   Hint: \(hint)" }
            if let tags = tagList(in: entry) { block += "\n   Tags: \(tags)" }

            guard characters + block.count <= maxTotalCharacters else { truncated += 1; continue }
            characters += block.count
            lines.append(block)
        }

        guard !lines.isEmpty else {
            return Outcome(kind: .studyGuide, title: title, documents: [], skippedEntries: skipped, truncatedEntries: truncated)
        }

        // One document rather than one per card: a flashcard is a couple of sentences, and 200 of
        // them as 200 Mindscape entries would bury everything else in the list while retrieving no
        // better — `RAGManager` chunks long documents anyway.
        let content = """
        [Study Guide] \(title)
        Imported: \(readableDate.string(from: Date()))
        \(lines.count) item\(lines.count == 1 ? "" : "s").

        \(lines.joined(separator: "\n\n"))
        """

        return Outcome(
            kind: .studyGuide,
            title: title,
            documents: [Document(name: "\(title) (Study Guide).txt", content: content)],
            skippedEntries: skipped,
            truncatedEntries: truncated
        )
    }

    // MARK: - Journal

    private static func journalOutcome(_ entries: [[String: Any]], title: String) -> Outcome {
        var documents: [Document] = []
        var skipped = 0
        var truncated = 0
        var characters = 0

        for entry in entries {
            let body = value(in: entry, forAnyOf: bodyKeys) ?? value(in: entry, forAnyOf: answerKeys)
            guard let body, !body.isEmpty else { skipped += 1; continue }

            guard documents.count < maxDocuments else { truncated += 1; continue }

            let rawDate = value(in: entry, forAnyOf: dateKeys)
            let day = rawDate.flatMap(normalizedDay) ?? rawDate
            let entryTitle = value(in: entry, forAnyOf: titleKeys)

            var header = "[Journal Entry]"
            if let day { header += "\nDate: \(day)" }
            if let entryTitle, !entryTitle.isEmpty { header += "\nTitle: \(entryTitle)" }
            if let mood = value(in: entry, forAnyOf: moodKeys), !mood.isEmpty { header += "\nMood: \(mood)" }
            if let location = value(in: entry, forAnyOf: locationKeys), !location.isEmpty { header += "\nLocation: \(location)" }
            if let tags = tagList(in: entry) { header += "\nTags: \(tags)" }
            if let section = entry["__section"] as? String { header += "\nSection: \(section)" }

            let content = "\(header)\n\n\(body)"
            guard characters + content.count <= maxTotalCharacters else { truncated += 1; continue }
            characters += content.count

            // One document per entry, unlike the study guide. Dates and moods are exactly what a
            // user asks the assistant about ("what was I doing in March"), and that only retrieves
            // well if each entry is separately named and separately scored.
            let label = [day, entryTitle].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " – ")
            let name = label.isEmpty
                ? "Journal – entry \(documents.count + 1).txt"
                : "Journal – \(String(label.prefix(70))).txt"
            documents.append(Document(name: name, content: content))
        }

        return Outcome(
            kind: .journal,
            title: title,
            documents: documents,
            skippedEntries: skipped,
            truncatedEntries: truncated
        )
    }

    // MARK: - Transcript re-import

    private static func transcriptOutcome(_ object: [String: Any], sourceName: String) -> Outcome {
        guard let conversation = object["conversation"] as? [String: Any],
              let messages = conversation["messages"] as? [Any] else {
            return Outcome(kind: .transcript, title: sourceName, documents: [], skippedEntries: 0, truncatedEntries: 0)
        }

        let title = (conversation["title"] as? String) ?? sourceName
        var lines: [String] = []
        var characters = 0
        var truncated = 0

        for case let message as [String: Any] in messages {
            let role = (message["role"] as? String) == "user" ? "You" : AppInfo.displayName
            let text = (message["text"] as? String) ?? ""
            guard !text.isEmpty else { continue }
            let block = "\(role): \(text)"
            guard characters + block.count <= maxTotalCharacters else { truncated += 1; continue }
            characters += block.count
            lines.append(block)
        }

        guard !lines.isEmpty else {
            return Outcome(kind: .transcript, title: title, documents: [], skippedEntries: 0, truncatedEntries: truncated)
        }

        let content = """
        [Chat Transcript] \(title)
        Imported: \(readableDate.string(from: Date()))

        \(lines.joined(separator: "\n\n"))
        """
        return Outcome(
            kind: .transcript,
            title: title,
            documents: [Document(name: "\(title) (Transcript).txt", content: content)],
            skippedEntries: 0,
            truncatedEntries: truncated
        )
    }

    // MARK: - Key vocabulary
    //
    // Matched case- and separator-insensitively (see `matchingValue`), so "createdAt",
    // "created_at", and "Created At" all hit the same entry here.

    private static let containerKeys = [
        "cards", "flashcards", "questions", "items", "terms", "vocabulary",
        "entries", "journal", "logs", "records", "notes", "days", "posts",
        "sections", "chapters", "topics", "data", "content", "results"
    ]
    private static let titleKeys    = ["title", "name", "heading", "subject", "topic", "term", "front"]
    private static let questionKeys = ["question", "q", "prompt", "front", "term", "word", "title", "key"]
    private static let answerKeys   = ["answer", "a", "definition", "back", "response", "explanation", "meaning", "value"]
    private static let bodyKeys     = ["body", "entry", "text", "content", "note", "notes", "description", "detail", "message"]
    private static let dateKeys     = ["date", "day", "created", "createdat", "timestamp", "when", "datetime", "time", "modified"]
    private static let moodKeys     = ["mood", "feeling", "emotion", "sentiment"]
    private static let locationKeys = ["location", "place", "where", "city"]
    private static let hintKeys     = ["hint", "clue", "example", "mnemonic"]
    private static let tagKeys      = ["tags", "labels", "topics", "categories", "keywords"]

    // MARK: - Field access

    private static func value(in entry: [String: Any], forAnyOf keys: [String]) -> String? {
        for key in keys {
            guard let raw = matchingValue(in: entry, forKey: key) else { continue }
            guard let string = stringify(raw), !string.isEmpty else { continue }
            return string
        }
        return nil
    }

    /// Case- and separator-insensitive key lookup. Exact match is tried first so a file that has
    /// both `note` and `notes` resolves to the one that was actually asked for.
    private static func matchingValue(in entry: [String: Any], forKey key: String) -> Any? {
        if let exact = entry[key] { return exact }
        let normalizedKey = normalizeKey(key)
        for (candidate, value) in entry where normalizeKey(candidate) == normalizedKey {
            return value
        }
        return nil
    }

    private static func normalizeKey(_ key: String) -> String {
        key.lowercased().filter { $0.isLetter || $0.isNumber }
    }

    /// Flattens a JSON value to displayable text. Arrays and nested objects are kept rather than
    /// dropped — an answer written as `["first", "second"]` is still an answer, and a file that
    /// silently lost half its content would be worse than one that reads a little flatly.
    private static func stringify(_ raw: Any, depth: Int = 0) -> String? {
        guard depth < 3 else { return nil }
        switch raw {
        case let string as String:
            return string.trimmingCharacters(in: .whitespacesAndNewlines)
        case let number as NSNumber:
            // `Bool` bridges to NSNumber, so it has to be distinguished explicitly or `true`
            // renders as `1`.
            if CFGetTypeID(number) == CFBooleanGetTypeID() { return number.boolValue ? "yes" : "no" }
            return number.stringValue
        case let array as [Any]:
            let parts = array.compactMap { stringify($0, depth: depth + 1) }.filter { !$0.isEmpty }
            return parts.isEmpty ? nil : parts.joined(separator: "; ")
        case let object as [String: Any]:
            let parts = object.keys.sorted().compactMap { key -> String? in
                guard let value = object[key], let rendered = stringify(value, depth: depth + 1),
                      !rendered.isEmpty else { return nil }
                return "\(key): \(rendered)"
            }
            return parts.isEmpty ? nil : parts.joined(separator: ", ")
        case is NSNull:
            return nil
        default:
            return nil
        }
    }

    private static func tagList(in entry: [String: Any]) -> String? {
        guard let raw = value(in: entry, forAnyOf: tagKeys) else { return nil }
        return raw.isEmpty ? nil : raw
    }

    /// Reduces whatever the file calls a date to `yyyy-MM-dd`, so entry names sort chronologically
    /// in the Mindscape list. Falls back to the raw string when nothing parses — an unrecognised
    /// date is still worth showing.
    private static func normalizedDay(_ raw: String) -> String? {
        for parser in isoParsers {
            if let parsed = parser.date(from: raw) { return dayFormatter.string(from: parsed) }
        }
        if let isoDate = isoDateOnlyParser.date(from: raw) { return dayFormatter.string(from: isoDate) }
        // Epoch seconds or milliseconds, which is what most app exports emit.
        if let number = Double(raw), number > 100_000 {
            let seconds = number > 100_000_000_000 ? number / 1000 : number
            return dayFormatter.string(from: Date(timeIntervalSince1970: seconds))
        }
        return raw.isEmpty ? nil : raw
    }

    /// Both ISO-8601 flavours, plain first.
    ///
    /// `ISO8601DateFormatter` treats `.withFractionalSeconds` as a requirement rather than a
    /// tolerance, so a single formatter carrying that option rejects `2026-03-04T08:12:00Z` —
    /// which is the exact spelling most journalling apps export. One formatter is not enough.
    private static let isoParsers: [ISO8601DateFormatter] = {
        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return [plain, fractional]
    }()

    private static let isoDateOnlyParser: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter
    }()

    private static let dayFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter
    }()

    private static let readableDate: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter
    }()
}

// MARK: - Format reference

/// Shown from the import button. The parser is tolerant, but a user staring at an error needs to
/// see a shape that definitely works — and one they can hand to a model to have their own notes
/// converted into.
struct StructuredImportHelpView: View {

    @Environment(\.dismiss) private var dismiss

    private let studyGuideExample = """
    {
      "type": "study_guide",
      "title": "Cell Biology — Midterm",
      "cards": [
        {
          "question": "What does the mitochondrion do?",
          "answer": "Produces ATP through cellular respiration.",
          "tags": ["organelles"]
        },
        {
          "question": "Define osmosis.",
          "answer": "Diffusion of water across a semipermeable membrane."
        }
      ]
    }
    """

    private let journalExample = """
    {
      "type": "journal",
      "entries": [
        {
          "date": "2026-03-04",
          "title": "Long run",
          "mood": "tired but good",
          "body": "Managed 10k without stopping for the first time."
        },
        {
          "date": "2026-03-05",
          "body": "Quiet day. Read most of the afternoon."
        }
      ]
    }
    """

    var body: some View {
        NavigationView {
            ZStack {
                Theme.background.ignoresSafeArea()

                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        Text("Import a JSON file to turn it into Mindscape entries the assistant can draw on. Two shapes are recognised, and the app works out which one it is looking at.")
                            .font(.system(size: 13))
                            .foregroundColor(Theme.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)

                        example(
                            icon: "graduationcap.fill",
                            title: "Study guide",
                            blurb: "Becomes a single numbered entry you can quiz yourself against. `answer`, `tags`, and `hint` are all optional.",
                            code: studyGuideExample
                        )

                        example(
                            icon: "book.closed.fill",
                            title: "Journal",
                            blurb: "Each entry becomes its own Mindscape document, named by date, so you can ask about a particular day or stretch of time.",
                            code: journalExample
                        )

                        VStack(alignment: .leading, spacing: 8) {
                            Text("IF YOUR FILE LOOKS DIFFERENT")
                                .font(.system(size: 11, weight: .black))
                                .foregroundColor(Theme.textPrimary)
                                .kerning(1.2)
                            bullet("Key names are flexible — `q`/`a`, `front`/`back`, `term`/`definition`, `text`, `content`, `createdAt` and similar are all understood.")
                            bullet("A bare top-level array works: `[{ \"question\": …, \"answer\": … }]`.")
                            bullet("Files grouped into `sections` or `chapters` are flattened, and each entry keeps its section name.")
                            bullet("A chat transcript exported from this app as JSON can be imported back in.")
                            bullet("Files are capped at \(ByteCountFormatter.string(fromByteCount: Int64(StructuredImport.maxFileBytes), countStyle: .file)) and \(StructuredImport.maxDocuments) entries. Everything is parsed on this device — nothing is uploaded.")
                        }
                        .padding(14)
                        .background(Theme.cardBackground)
                        .cornerRadius(12)
                        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Theme.border, lineWidth: 1))
                    }
                    .padding(20)
                }
            }
            .navigationTitle("What can I import?")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                        .foregroundColor(Theme.accentCyan)
                }
            }
        }
    }

    private func example(icon: String, title: String, blurb: String, code: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .foregroundColor(Theme.accentCyan)
                Text(title)
                    .font(.system(size: 14, weight: .bold))
                    .foregroundColor(Theme.textPrimary)
                Spacer()
                Button {
                    UIPasteboard.general.string = code
                } label: {
                    Text("Copy")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(Theme.accentCyan)
                }
            }

            // `.init(_:)` routes the string through `LocalizedStringKey`, which is what makes the
            // inline markdown render — without it the backticks around key names print literally.
            Text(.init(blurb))
                .font(.system(size: 12))
                .foregroundColor(Theme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)

            // Horizontally scrollable so long lines don't force the whole sheet wide on a small
            // screen — the same rule the rest of the app follows for monospaced content.
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(Theme.textPrimary)
                    .padding(10)
            }
            .background(Theme.background)
            .cornerRadius(8)
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Theme.border, lineWidth: 1))
        }
        .padding(14)
        .background(Theme.cardBackground)
        .cornerRadius(12)
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Theme.border, lineWidth: 1))
    }

    private func bullet(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text("•").foregroundColor(Theme.accent)
            Text(.init(text))
                .font(.system(size: 12))
                .foregroundColor(Theme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
