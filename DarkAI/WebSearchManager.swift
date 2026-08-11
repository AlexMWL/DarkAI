import Foundation
import Combine

enum WebSearchError: LocalizedError {
    case blocked(String)
    /// A weather question that never named a place. The app deliberately asks no location
    /// permission (see `WebSearchClassifier` — the place comes from the text itself), so the
    /// honest move is to ask which city rather than run a general search that cannot succeed.
    case needsLocation

    var errorDescription: String? {
        switch self {
        case .blocked(let message): return message
        case .needsLocation:        return "Which city should I check the weather for?"
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
    private let wikipedia = WikipediaProvider()
    private let googleNews = GoogleNewsProvider()

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
        if case .weather(let place) = queryType {
            // No place in the message means there's nothing to look up — a general search for
            // the bare word "weather" returns nothing useful from any provider. Ask instead.
            guard let place, !place.isEmpty else { throw WebSearchError.needsLocation }
            searchStage = "Checking the weather…"
            result = try await openMeteo.search(query: place)
        } else {
            searchStage = "Searching the interwebs…"
            result = try await generalSearch(queryText)
        }

        let resultDecision = ContentSafety.review(result.answer, surface: .webSearchResult)
        guard resultDecision.isAllowed else {
            throw WebSearchError.blocked(resultDecision.message ?? "The results weren't something I could show you.")
        }

        return result
    }

    /// Tries providers in order and returns the first real answer.
    ///
    /// Chained rather than single-shot because each keyless provider covers a different, narrow
    /// slice and any one of them comes back empty most of the time: DuckDuckGo's instant-answer
    /// API only responds for well-known named entities, and Wikipedia only for things it has an
    /// article about. Running one provider alone was why general searches almost always failed —
    /// the request succeeded, there was simply nothing in that particular source.
    ///
    /// Brave goes first when a key is set, since it's the only one that's a real web index. It
    /// still falls through on failure so a rate-limited or mistyped key degrades to the free
    /// sources instead of failing the whole search.
    private func generalSearch(_ query: String) async throws -> WebSearchResult {
        var providers: [WebSearchProvider] = []
        if hasBraveKey {
            providers.append(BraveSearchProvider(apiKey: braveAPIKey))
        }
        // Current-events questions go to the news feed before the encyclopedia sources, which
        // can describe a subject accurately while saying nothing about what happened to it this
        // week — the exact case where a confident, stale answer is worse than none.
        if WebSearchClassifier.isNewsQuery(query) {
            providers.append(googleNews)
        }
        providers.append(duckDuckGo)
        providers.append(wikipedia)

        var lastError: Error = WebSearchProviderError.noResults
        for provider in providers {
            do {
                return try await provider.search(query: query)
            } catch {
                LogManager.shared.log("WebSearch: \(type(of: provider)) returned nothing — \(error.localizedDescription)")
                lastError = error
            }
        }
        throw lastError
    }
}
