import Foundation
import Combine

/// A single remembered fact about the user.
///
/// Stored structured rather than as a bare string so memories can supersede each other and be
/// evicted by value. A flat `[String]` list couldn't do either: moving house produced two
/// contradictory "User lives in…" entries that were both fed to the model forever, and the
/// 60-entry cap dropped the oldest fact regardless of whether it was the user's name or a
/// throwaway note about what they had for lunch.
struct Memory: Codable, Identifiable, Equatable {
    enum Kind: String, Codable {
        /// Stable facts about who the user is. Never evicted before anything else.
        case identity
        /// Durable preferences and traits.
        case preference
        /// Plans and intentions — meaningful, but they expire in practice.
        case intent
        /// Time-anchored things that happened. First to go when space runs short.
        case event

        /// Higher survives eviction longer.
        var priority: Int {
            switch self {
            case .identity:   return 3
            case .preference: return 2
            case .intent:     return 1
            case .event:      return 0
            }
        }
    }

    var id: UUID = UUID()
    var text: String
    var kind: Kind
    var date: Date = Date()
    /// Facts sharing a slot replace each other — "home" holds one value, so a new one wins
    /// instead of contradicting the old. `nil` means the fact simply accumulates.
    var slot: String?
}

class MemoryManager: ObservableObject {
    @Published var memories: [Memory] = []

    private let storageKey = "DarkAI_MemoriesV2"
    private let legacyStorageKey = "DarkAI_Memories"

    private static let eventDateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "MMMM d, yyyy"
        return f
    }()

    /// Every stored memory is re-injected into every future prompt, so unbounded growth quietly
    /// eats the context budget and crowds out actual conversation.
    private let maxMemories = 80

    init() {
        loadMemories()
    }

    // MARK: - Persistence

    func loadMemories() {
        if let data = UserDefaults.standard.data(forKey: storageKey) {
            if let decoded = try? JSONDecoder().decode([Memory].self, from: data) {
                memories = decoded
                return
            }
            // Data existed but couldn't be decoded, e.g. a future non-additive schema change.
            // Falling through to an empty list is still the only real option, but doing so with
            // no trace at all is what turns a legitimate schema change into what reads as
            // inexplicable, silent data loss. Raw bytes left in place at `storageKey` rather than
            // cleared, in case they're worth inspecting later — the legacy migration below is
            // skipped either way, since `data` being present at all means this device has already
            // migrated past the legacy key once.
            LogManager.shared.log("MemoryManager: found \(data.count) bytes of stored memories but failed to decode them — starting fresh rather than losing them silently")
            return
        }
        // One-time migration from the flat string list. Everything comes across as a preference
        // with no slot — the old format carried no way to tell identity from small talk.
        if let legacy = UserDefaults.standard.stringArray(forKey: legacyStorageKey) {
            memories = legacy.map { Memory(text: $0, kind: .preference, slot: nil) }
            saveMemories()
            UserDefaults.standard.removeObject(forKey: legacyStorageKey)
        }
    }

    func saveMemories() {
        guard let encoded = try? JSONEncoder().encode(memories) else { return }
        UserDefaults.standard.set(encoded, forKey: storageKey)
    }

    // MARK: - Mutation

    func addMemory(_ text: String, kind: Memory.Kind = .preference, slot: String? = nil) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count > 5 else { return }

        // Exact duplicate: refresh its timestamp so repetition reinforces rather than duplicates.
        if let index = memories.firstIndex(where: { $0.text.caseInsensitiveCompare(trimmed) == .orderedSame }) {
            memories[index].date = Date()
            saveMemories()
            return
        }

        // Slotted facts supersede. "User lives in Seattle" replaces "User lives in Portland"
        // rather than sitting alongside it and leaving the model to guess which is current.
        if let slot {
            memories.removeAll { $0.slot == slot }
        }

        memories.append(Memory(text: trimmed, kind: kind, slot: slot))
        enforceCap()
        saveMemories()
    }

    /// Drops the least valuable memories first: events before intents, intents before
    /// preferences, and identity only as a last resort. Within a tier, oldest goes first.
    private func enforceCap() {
        guard memories.count > maxMemories else { return }
        let overflow = memories.count - maxMemories
        let doomed = memories
            .enumerated()
            .sorted { lhs, rhs in
                if lhs.element.kind.priority != rhs.element.kind.priority {
                    return lhs.element.kind.priority < rhs.element.kind.priority
                }
                return lhs.element.date < rhs.element.date
            }
            .prefix(overflow)
            .map(\.offset)
        let doomedSet = Set(doomed)
        memories = memories.enumerated().filter { !doomedSet.contains($0.offset) }.map(\.element)
    }

    func removeMemory(at index: Int) {
        guard index >= 0 && index < memories.count else { return }
        memories.remove(at: index)
        saveMemories()
    }

    func clearAllMemories() {
        memories.removeAll()
        saveMemories()
    }

    // MARK: - Extraction

    /// Pulls facts out of a user message.
    ///
    /// The significant change from the original: patterns are matched against each *clause*, not
    /// against the start of the whole message. The old version used `hasPrefix` on the entire
    /// string, so it only ever caught a fact if the message opened with it — "I live in Seattle"
    /// was remembered, "Yeah, I live in Seattle" and "I'm tired. I live in Seattle now" were both
    /// missed. Splitting on sentence and clause boundaries first is most of the improvement here.
    func extractMemories(from userMessage: String) {
        let today = Self.eventDateFormatter.string(from: Date())

        for clause in Self.clauses(in: userMessage) {
            let lower = clause.lowercased()

            // Negations are skipped outright — "I don't live in Seattle" must not be stored as
            // "User lives in Seattle", which is what prefix matching on the positive form did.
            if Self.isNegated(lower) { continue }

            // — Identity — slotted, so each supersedes rather than accumulates.
            match(lower, clause, ["my name is ", "i am called ", "call me "],
                  kind: .identity, slot: "name") { "User's name is \($0)." }
            match(lower, clause, ["i live in ", "i'm based in ", "i am based in "],
                  kind: .identity, slot: "home") { "User lives in \($0)." }
            match(lower, clause, ["i'm from ", "i am from ", "i grew up in "],
                  kind: .identity, slot: "origin") { "User is from \($0)." }
            match(lower, clause, ["i work as ", "my job is ", "i work at "],
                  kind: .identity, slot: "job") { "User works as \($0)." }
            match(lower, clause, ["i am a ", "i'm a ", "i am an ", "i'm an "],
                  kind: .identity, slot: "role") { "User is a \($0)." }
            match(lower, clause, ["i am allergic to ", "i'm allergic to "],
                  kind: .identity, slot: "allergy") { "User is allergic to \($0)." }
            match(lower, clause, ["my pronouns are "],
                  kind: .identity, slot: "pronouns") { "User's pronouns are \($0)." }

            // — People & pets —
            for (prefix, label) in [("my wife is ", "wife"), ("my husband is ", "husband"),
                                    ("my partner is ", "partner"), ("my girlfriend is ", "girlfriend"),
                                    ("my boyfriend is ", "boyfriend"), ("my son is ", "son"),
                                    ("my daughter is ", "daughter"), ("my dog is ", "dog"),
                                    ("my cat is ", "cat"), ("my pet is ", "pet")] {
                match(lower, clause, [prefix], kind: .identity, slot: label) {
                    "User's \(label) is \($0)."
                }
            }

            // — Preferences —
            match(lower, clause, ["i love "], kind: .preference) { "User loves \($0)." }
            match(lower, clause, ["i really like ", "i like "], kind: .preference) { "User likes \($0)." }
            match(lower, clause, ["i enjoy "], kind: .preference) { "User enjoys \($0)." }
            match(lower, clause, ["i prefer "], kind: .preference) { "User prefers \($0)." }
            match(lower, clause, ["i hate "], kind: .preference) { "User hates \($0)." }
            match(lower, clause, ["i dislike ", "i don't like ", "i do not like "],
                  kind: .preference) { "User dislikes \($0)." }
            match(lower, clause, ["i use ", "i mostly use "], kind: .preference) { "User uses \($0)." }
            match(lower, clause, ["i collect "], kind: .preference) { "User collects \($0)." }
            match(lower, clause, ["my hobby is "], kind: .preference, slot: "hobby") { "User's hobby is \($0)." }
            match(lower, clause, ["i am learning ", "i'm learning ", "i am studying ", "i'm studying "],
                  kind: .preference) { "User is learning \($0)." }
            match(lower, clause, ["i know how to "], kind: .preference) { "User knows how to \($0)." }
            match(lower, clause, ["i always "], kind: .preference) { "User always \($0)." }
            match(lower, clause, ["i usually "], kind: .preference) { "User usually \($0)." }
            match(lower, clause, ["i never "], kind: .preference) { "User never \($0)." }
            match(lower, clause, ["i think ", "i believe "], kind: .preference) { "User believes \($0)." }

            // — Ownership —
            match(lower, clause, ["i own a ", "i have a ", "i have an "], kind: .preference) { "User has \($0)." }
            match(lower, clause, ["my car is a "], kind: .preference, slot: "car") { "User's car is a \($0)." }
            match(lower, clause, ["my phone is a "], kind: .preference, slot: "phone") { "User's phone is a \($0)." }
            match(lower, clause, ["my computer is a "], kind: .preference, slot: "computer") { "User's computer is a \($0)." }

            // — Intents —
            match(lower, clause, ["i plan to ", "i'm planning to ", "i am planning to "],
                  kind: .intent) { "User plans to \($0)." }
            match(lower, clause, ["i want to "], kind: .intent) { "User wants to \($0)." }
            match(lower, clause, ["i need to ", "i have to ", "i must ", "i've got to "],
                  kind: .intent) { "User needs to \($0)." }
            match(lower, clause, ["i'm going to ", "i am going to "], kind: .intent) { "User is going to \($0)." }

            // — Events — time-anchored, so the model can weigh how recent they are.
            for prefix in ["yesterday i ", "today i ", "this morning i ", "this afternoon i ",
                           "last night i ", "last week i ", "earlier today i ", "i just ",
                           "i recently ", "i finished ", "i started ", "i got ", "i lost ",
                           "i found ", "i broke ", "i passed ", "i failed ", "i missed "] {
                match(lower, clause, [prefix], kind: .event) { value in
                    "On \(today), user mentioned they \(prefix.replacingOccurrences(of: "i ", with: "")) \(value)."
                        .replacingOccurrences(of: "  ", with: " ")
                }
            }

            // — "My favourite X is Y" —
            for marker in ["my favorite ", "my favourite "] {
                guard let range = lower.range(of: marker) else { continue }
                let rest = clause[range.upperBound...]
                guard let isRange = rest.lowercased().range(of: " is ") else { continue }
                let subject = rest[..<isRange.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
                let value = rest[isRange.upperBound...]
                    .trimmingCharacters(in: CharacterSet(charactersIn: " .!?,;"))
                if !subject.isEmpty, !value.isEmpty, value.count < 100 {
                    addMemory("User's favorite \(subject) is \(value).",
                              kind: .preference, slot: "favorite-\(subject.lowercased())")
                }
            }
        }
    }

    // MARK: - Extraction helpers

    /// Splits a message into sentences, then into comma/conjunction clauses, so a fact stated
    /// anywhere in a long message is still reachable by a prefix pattern.
    private static func clauses(in message: String) -> [String] {
        let sentences = message.components(separatedBy: CharacterSet(charactersIn: ".!?\n"))
        var result: [String] = []
        for sentence in sentences {
            let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            result.append(trimmed)

            // Also consider the text after common clause breaks, so "Yeah, I live in Seattle"
            // and "I'm tired but I love hiking" both expose their fact to the prefix patterns.
            for separator in [", ", " but ", " and ", " so ", " because ", " though ", " although "] {
                var searchStart = trimmed.startIndex
                while let range = trimmed.range(of: separator, options: .caseInsensitive,
                                                range: searchStart..<trimmed.endIndex) {
                    let tail = trimmed[range.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines)
                    if tail.count > 5 { result.append(tail) }
                    searchStart = range.upperBound
                    if result.count > 40 { break }   // pathological input guard
                }
            }
        }
        return result
    }

    private static let negationMarkers = [
        "i don't ", "i do not ", "i didn't ", "i did not ", "i'm not ", "i am not ",
        "i never ", "i wouldn't ", "i would not ", "i can't ", "i cannot ", "i won't ",
        "not really", "no longer", "used to but"
    ]

    /// True when a clause negates the pattern that would otherwise match it. Checked before
    /// extraction so a denial is never stored as an affirmation. `i never`/`i don't like` have
    /// their own positive handlers above, which run on the un-negated forms.
    private static func isNegated(_ lower: String) -> Bool {
        // These are handled explicitly as their own memories, so don't treat them as negations.
        if lower.hasPrefix("i never ") || lower.hasPrefix("i don't like ") || lower.hasPrefix("i do not like ") {
            return false
        }
        return negationMarkers.contains { lower.hasPrefix($0) }
    }

    private func match(_ lower: String,
                       _ original: String,
                       _ prefixes: [String],
                       kind: Memory.Kind,
                       slot: String? = nil,
                       format: (String) -> String?) {
        for prefix in prefixes {
            guard lower.hasPrefix(prefix) else { continue }
            let remainder = String(original.dropFirst(prefix.count))
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .trimmingCharacters(in: CharacterSet(charactersIn: ".!?,;"))
            guard !remainder.isEmpty, remainder.count < 120 else { return }
            if let formatted = format(remainder) {
                addMemory(formatted, kind: kind, slot: slot)
            }
            return
        }
    }

    // MARK: - Prompt rendering

    /// Renders memories for the system prompt, most valuable first and grouped by kind so the
    /// model can tell a stable fact from something that happened last Tuesday.
    func getFormattedMemoriesForContext() -> String {
        guard !memories.isEmpty else { return "" }

        var result = """
        ### What you know about the user:
        (Background only. Do not recite or list these unless the user directly asks what you \
        remember. Use them silently to inform tone and answers.)

        """

        let groups: [(Memory.Kind, String)] = [
            (.identity, "About them"),
            (.preference, "Preferences"),
            (.intent, "Plans"),
            (.event, "Recent")
        ]
        for (kind, heading) in groups {
            let items = memories.filter { $0.kind == kind }.sorted { $0.date > $1.date }
            guard !items.isEmpty else { continue }
            result += "\n\(heading):\n"
            for item in items { result += "- \(item.text)\n" }
        }
        return result
    }
}
