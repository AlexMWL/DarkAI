import Foundation
import Combine

enum WebSearchError: LocalizedError {
    case blocked(String)

    var errorDescription: String? {
        switch self {
        case .blocked(let message): return message
        }
    }
}

/// Orchestrates the whole "does this need the internet" feature: the master on/off switch, the
/// optional user-supplied Brave key, provider routing, and the two content-safety checkpoints
/// (query going out, result coming back) that run unconditionally whenever a search actually
/// happens — the toggle controls *whether* search happens at all, not whether it's filtered when
/// it does, matching `ContentSafety`'s "no off switch for the filter" stance elsewhere.
@MainActor
final class WebSearchManager: ObservableObject {

    // MARK: Published state

    /// Off by default. Internet access is a deliberate, narrow exception to this app's
    /// no-server/offline design, so nothing about this feature activates — not classification,
    /// not the offer bubble, nothing — until the user turns it on explicitly.
    @Published var isEnabled: Bool = UserDefaults.standard.bool(forKey: "webSearchEnabled") {
        didSet { UserDefaults.standard.set(isEnabled, forKey: "webSearchEnabled") }
    }

    @Published private(set) var isSearching: Bool = false
    /// What the search is doing right now, shown in place of the assistant's message while a
    /// search is in flight — same role as `DiffusionManager.generationStage`.
    @Published private(set) var searchStage: String = ""

    /// User-supplied Brave Search API key. Kept in the Keychain, not UserDefaults — this is a
    /// real secret, unlike every other persisted setting in the app. Empty string means "not
    /// configured," which is also what a fresh Keychain read returns, so the two states don't
    /// need to be told apart anywhere else.
    @Published var braveAPIKey: String = KeychainStore.get(forKey: WebSearchManager.braveKeyKeychainKey) ?? "" {
        didSet {
            let trimmed = braveAPIKey.trimmingCharacters(in: .whitespacesAndNewlines)
            KeychainStore.set(trimmed.isEmpty ? nil : trimmed, forKey: WebSearchManager.braveKeyKeychainKey)
        }
    }

    private static let braveKeyKeychainKey = "braveSearchAPIKey"

    var hasBraveKey: Bool { !braveAPIKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    private let openMeteo = OpenMeteoProvider()
    private let duckDuckGo = DuckDuckGoInstantAnswerProvider()

    // MARK: Search

    /// Runs one search end to end: screens the query, routes to a provider, screens the result.
    /// Throws rather than returning an optional — the caller needs "nothing came back" (a
    /// provider's `.noResults`), "that wasn't allowed" (`WebSearchError.blocked`), and a plain
    /// network failure told apart, so it can say something honest instead of a generic failure.
    func search(_ queryType: WebSearchQueryType) async throws -> WebSearchResult {
        let queryText: String
        switch queryType {
        case .weather(let place): queryText = place ?? "weather"
        case .general(let query):  queryText = query
        }

        let queryDecision = ContentSafety.review(queryText, surface: .webSearchQuery)
        guard queryDecision.isAllowed else {
            throw WebSearchError.blocked(queryDecision.message ?? "That search isn't allowed under the app's content policy.")
        }

        isSearching = true
        defer { isSearching = false; searchStage = "" }

        let result: WebSearchResult
        if case .weather(let place) = queryType, let place, !place.isEmpty {
            searchStage = "Checking the weather…"
            result = try await openMeteo.search(query: place)
        } else {
            // Either a general query, or a weather-shaped one with no location the classifier
            // could pull out — falls through to general search rather than guessing a place.
            searchStage = "Searching the interwebs…"
            result = try await generalSearch(queryText)
        }

        let resultDecision = ContentSafety.review(result.answer, surface: .webSearchResult)
        guard resultDecision.isAllowed else {
            throw WebSearchError.blocked(resultDecision.message ?? "The results weren't something I could show you.")
        }

        return result
    }

    private func generalSearch(_ query: String) async throws -> WebSearchResult {
        if hasBraveKey {
            return try await BraveSearchProvider(apiKey: braveAPIKey).search(query: query)
        }
        return try await duckDuckGo.search(query: query)
    }
}
