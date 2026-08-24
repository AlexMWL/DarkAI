import Foundation

/// Whether `needle` occurs in `haystack` at a word boundary — not merely as a substring. A bare
/// `.contains` matches "art" inside "chart", "render" inside "surrender", "search for " inside
/// "research for ideas": real trigger phrases sitting mid-word or mid-phrase in text that was
/// never asking for what they'd otherwise imply. Treats any non-letter/non-number as a boundary —
/// punctuation included, not just whitespace — so a trigger at the end of a sentence still
/// matches. Callers pass already-lowercased/normalized text; this does no normalization itself.
///
/// Consolidates three previously-independent copies of this same logic —
/// `ContentSafety.wordBoundaryMatch`, `PromptClassifier.wordBoundaryContains`, and
/// `WebSearchClassifier.containsAsWord` — which had started drifting out of sync with each other
/// (only two of the three had been fixed for the trailing-space case below before this existed).
/// `nonisolated`: pure string logic with no shared state, called from `ContentSafety` (which is
/// itself `nonisolated` and screens content off the main actor) as well as from the two
/// main-actor-isolated classifiers.
nonisolated func sharedWordBoundaryContains(_ needle: String, in haystack: String) -> Bool {
    guard !needle.isEmpty else { return false }
    // Many callers' triggers already end in a trailing space or other separator ("draw a ",
    // "search for ") specifically so the phrase reads as a clean prefix to strip — that trailing
    // character already *is* the boundary after the match, so requiring the character following
    // it (the first letter of whatever subject comes next, in the overwhelmingly common case) to
    // ALSO be non-alphanumeric made the match impossible for exactly the ordinary phrasing this
    // exists to catch: "please draw a cat" stopped matching "draw a " because 'c' (of "cat")
    // isn't a boundary character.
    let needleEndsAtBoundary: Bool = {
        guard let last = needle.last else { return false }
        return !last.isLetter && !last.isNumber
    }()
    var searchStart = haystack.startIndex
    while searchStart < haystack.endIndex,
          let range = haystack.range(of: needle, range: searchStart..<haystack.endIndex) {
        let beforeOK: Bool = {
            guard range.lowerBound > haystack.startIndex else { return true }
            let prev = haystack[haystack.index(before: range.lowerBound)]
            return !prev.isLetter && !prev.isNumber
        }()
        let afterOK: Bool = needleEndsAtBoundary || {
            guard range.upperBound < haystack.endIndex else { return true }
            let next = haystack[range.upperBound]
            return !next.isLetter && !next.isNumber
        }()
        if beforeOK && afterOK { return true }
        searchStart = haystack.index(after: range.lowerBound)
    }
    return false
}
