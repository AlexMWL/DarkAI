import Foundation

// MARK: - Web Search Query Type

/// What kind of external lookup a message looks like it wants, if any.
enum WebSearchQueryType: Equatable {
    /// `place` is `nil` when a weather word was detected but no location could be pulled out of
    /// the text — routed to the general path instead of guessing a location.
    case weather(place: String?)
    case general(query: String)
}

// MARK: - Web Search Classifier

/// Rule-based, zero-latency classifier for "does this message look like it needs information
/// from the internet." Same shape as `PromptClassifier` (deterministic keyword matching, no LLM
/// call), but orthogonal to it rather than a competing case of the same enum — a message can be
/// an ordinary question *or* look search-worthy, it isn't mutually exclusive the way image intent
/// is. Only ever consulted when the user has turned internet access on; see `WebSearchManager`.
///
/// This is a trigger, not a guarantee. Like `ContentSafety`'s lexical screening, it will miss
/// some genuinely current questions and occasionally flag one that didn't need it — false
/// negatives just mean the user has to ask explicitly ("search the web for..."), which is always
/// honored.
struct WebSearchClassifier {

    // MARK: Trigger Lists

    /// Explicit requests to search, where the grammar guarantees what follows the trigger is a
    /// clean subject — "search the web for X" → "X" is safe to strip.
    private static let strongTriggersWithCleanStrip: [String] = [
        "search the internet for", "search the web for", "search online for", "search for "
    ]

    /// Explicit requests to search where the trigger is a conversational lead-in, not a phrase
    /// with a clean "everything after this is the query" boundary — stripping "can you search"
    /// from "can you search and tell me the latest news" leaves the broken fragment "and tell me
    /// the latest news". These use the whole original message as the query instead.
    private static let strongTriggersUseFullText: [String] = [
        "search the internet", "search the web", "search online",
        "look up online", "look that up online", "look it up online",
        "google ", "can you search", "could you search", "please search"
    ]

    /// Weather words. Checked before the general pattern list — a weather question is routed to
    /// the dedicated (free, precise) weather provider rather than general search.
    private static let weatherTriggers: [String] = [
        "weather", "forecast", "temperature", "humidity",
        "is it raining", "is it going to rain", "is it snowing", "is it going to snow",
        "how hot is it", "how cold is it", "how hot will it be", "how cold will it be"
    ]

    /// Preposition words that introduce a location in a weather question. Matched as whole
    /// words, and on the *last* occurrence in the message — "what's the weather like in Santa
    /// Clara" has "like" sitting between "weather" and "in", which a literal "weather in "
    /// marker can't see past; searching for the preposition itself instead of a fixed phrase
    /// survives whatever filler words come before it.
    private static let placePrepositions: Set<String> = ["in", "at", "near", "for"]

    /// Current-events / live-data phrasing — news, prices, scores, standings. Medium confidence:
    /// checked against the exclusion list below before firing, since several of these phrases
    /// also show up inside purely explanatory questions.
    private static let currentInfoTriggers: [String] = [
        // "news" alone (not just "latest/breaking news about X") — the question-shape guard
        // above is what makes a bare keyword safe here, the same way it covers "weather" on its
        // own. Enumerating every "news in/for/about/near X" preposition combination would still
        // miss real phrasings like "what's been in the news lately for X"; requiring a question
        // shape and matching on the topic word itself covers all of them at once.
        "news", "latest news", "breaking news", "recent news about", "news about",
        "what happened today", "what's happening", "whats happening",
        "who won", "score of the", "final score", "standings",
        "current price of", "stock price of", "share price of", "exchange rate",
        "current population of", "current ceo of", "current president of",
        "current prime minister of", "who is the current",
        "as of today", "as of this week", "right now", "this week's",
        "release date of", "come out yet", "coming out this"
    ]

    /// Overrides the current-info tier (not the strong-trigger tier — an explicit "search the
    /// web for" is always honored). Catches explanatory/conceptual phrasing that happens to
    /// contain a trigger word without actually wanting live data.
    private static let exclusionPrefixes: [String] = [
        "explain", "describe", "define", "definition of", "what does",
        "how does", "how do", "how can", "how is", "how are",
        "why does", "why is", "why are", "what causes",
        "write a", "write an", "summarize", "summarise",
        "compare", "what's the difference", "what is the difference"
    ]

    /// Words/phrases that mark a message as actually asking or requesting something, as opposed
    /// to just mentioning a trigger word in passing. Required before a weather or current-info
    /// trigger fires — without this, "hell yeah, I love this weather" matched on the bare word
    /// "weather" and offered to search the internet for a remark that wasn't a question at all.
    /// Not required for `strongTriggers*` — "search the web for X" is unambiguous on its own.
    private static let questionStarters: [String] = [
        "what", "whats", "what's", "how", "hows", "how's", "when", "whens", "when's",
        "where", "wheres", "where's", "who", "whos", "who's", "which", "is it", "will it",
        "does it", "did it", "can you", "could you", "would you", "will you",
        "tell me", "let me know", "check the", "check if"
    ]

    private static func looksLikeQuestionOrRequest(_ lower: String) -> Bool {
        if lower.hasSuffix("?") { return true }
        return questionStarters.contains { lower.hasPrefix($0) }
    }

    // MARK: Classification

    static func classify(_ input: String) -> WebSearchQueryType? {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let lower = trimmed.lowercased()

        // 1. Explicit request — always wins, no exclusion check and no question-shape check:
        // asking outright to search is unambiguous regardless of how it's phrased.
        for trigger in strongTriggersWithCleanStrip where lower.contains(trigger) {
            let stripped = strip(trigger, from: trimmed)
            return .general(query: stripped.isEmpty ? trimmed : stripped)
        }
        for trigger in strongTriggersUseFullText where lower.contains(trigger) {
            return .general(query: trimmed)
        }

        // Everything below this point merely *mentions* a topic that could be weather- or
        // current-events-related — "weather", "who won", "latest news" all show up in ordinary
        // remarks as often as in real questions ("hell yeah, I love this weather" is not a
        // request for a forecast). Require the message to actually look like a question or
        // request before treating a bare keyword match as one.
        guard looksLikeQuestionOrRequest(lower) else { return nil }

        // 2. Weather — checked before the exclusion list fires on "how hot is it" style phrasing
        // that would otherwise look like an excluded "how" question.
        for trigger in weatherTriggers where lower.contains(trigger) {
            return .weather(place: extractPlace(from: lower))
        }

        // 3. Exclusion check — only guards the tier below.
        for exclusion in exclusionPrefixes where lower.hasPrefix(exclusion) {
            return nil
        }

        // 4. General current-info patterns.
        for trigger in currentInfoTriggers where lower.contains(trigger) {
            return .general(query: trimmed)
        }

        return nil
    }

    // MARK: Helpers

    private static func strip(_ trigger: String, from original: String) -> String {
        let lower = original.lowercased()
        guard let range = lower.range(of: trigger) else {
            return original.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        var cleaned = original
        cleaned.removeSubrange(range)
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func extractPlace(from lower: String) -> String? {
        let words = lower.split(separator: " ").map(String.init)

        // Last occurrence, not first — "for the rest of the week, what's the weather in
        // Boston" has an earlier "for" that isn't the one introducing the place.
        guard let prepositionIndex = words.lastIndex(where: {
            placePrepositions.contains($0.trimmingCharacters(in: .punctuationCharacters))
        }) else {
            return nil
        }

        var placeWords = Array(words[(prepositionIndex + 1)...])
        if placeWords.first == "the" { placeWords.removeFirst() }
        guard !placeWords.isEmpty else { return nil }

        var place = placeWords.joined(separator: " ")
        for trailing in ["today", "right now", "tomorrow", "this weekend", "this week", "now", "area", "region"] {
            if place.hasSuffix(trailing) {
                place = String(place.dropLast(trailing.count)).trimmingCharacters(in: .whitespaces)
            }
        }
        place = place.trimmingCharacters(in: CharacterSet(charactersIn: "?.!,; "))
        return place.isEmpty ? nil : place
    }
}
