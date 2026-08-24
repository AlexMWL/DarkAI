import Foundation

// MARK: - Result Types

struct WebSearchSource {
    var title: String
    var url: String
}

struct WebSearchResult {
    /// Plain-text snippet(s) meant to be dropped into the LLM's context, not shown verbatim as
    /// the final answer — the model still composes the reply from this, same as it does with
    /// RAG/memories context.
    var answer: String
    var sources: [WebSearchSource]
}

enum WebSearchProviderError: LocalizedError {
    case noResults
    case invalidResponse
    /// A single provider took longer than `WebSearchManager`'s per-provider timeout to respond.
    case timedOut

    var errorDescription: String? {
        switch self {
        case .noResults:       return "No results were found for that search."
        case .invalidResponse: return "The search provider returned something unexpected."
        case .timedOut:        return "The search provider took too long to respond."
        }
    }
}

// MARK: - Provider protocol

/// `Sendable` so a provider can be captured into the `@Sendable` closure
/// `WebSearchManager.withTimeout` hands to `withThrowingTaskGroup` — every conforming type below
/// is already a plain value type with no non-Sendable state (an empty struct, or one holding only
/// a `String` API key), so this only formalizes what was already true rather than constraining
/// anything.
protocol WebSearchProvider: Sendable {
    /// `query` is a place name for `OpenMeteoProvider`, free text for the others — routing
    /// which provider gets called is `WebSearchManager`'s job, not this protocol's.
    func search(query: String) async throws -> WebSearchResult
}

/// Validates an HTTP response's status code — matches the check `BraveSearchProvider` already
/// did on its own. The other providers below called straight through to `JSONDecoder`/
/// `XMLParser` without this, so a 4xx/5xx or rate-limited response that happened to decode (or
/// simply failed to decode) was indistinguishable in the log from an honest "no results," masking
/// real outages/throttling.
private nonisolated func validateHTTPStatus(_ response: URLResponse) throws {
    guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
        throw WebSearchProviderError.invalidResponse
    }
}

// MARK: - Open-Meteo (weather, keyless)

/// Free, keyless weather lookup. Two calls against the same free service: Open-Meteo's own
/// geocoding endpoint to resolve a place name to coordinates, then the forecast endpoint for
/// current conditions. No API key, no account — but its CC-BY 4.0 data license does require
/// attribution, which is why the source below always reads "Weather data by Open-Meteo.com"
/// rather than being omitted like a typical search result would be.
nonisolated struct OpenMeteoProvider: WebSearchProvider {

    private struct GeocodeResponse: Decodable {
        struct Result: Decodable {
            var name: String
            var latitude: Double
            var longitude: Double
            var country: String?
            var country_code: String?
            var admin1: String?
        }
        var results: [Result]?
    }

    private struct ForecastResponse: Decodable {
        struct Current: Decodable {
            var temperature_2m: Double?
            var relative_humidity_2m: Double?
            var apparent_temperature: Double?
            var weather_code: Int?
            var wind_speed_10m: Double?
        }
        var current: Current?
    }

    func search(query place: String) async throws -> WebSearchResult {
        let match = try await resolveLocation(place)

        var forecastComponents = URLComponents(string: "https://api.open-meteo.com/v1/forecast")!
        forecastComponents.queryItems = [
            URLQueryItem(name: "latitude", value: String(match.latitude)),
            URLQueryItem(name: "longitude", value: String(match.longitude)),
            URLQueryItem(name: "current", value: "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"),
            URLQueryItem(name: "temperature_unit", value: "fahrenheit"),
            URLQueryItem(name: "wind_speed_unit", value: "mph")
        ]
        guard let forecastURL = forecastComponents.url else { throw WebSearchProviderError.invalidResponse }
        let (forecastData, forecastResponse) = try await URLSession.shared.data(from: forecastURL)
        try validateHTTPStatus(forecastResponse)
        let forecast = try JSONDecoder().decode(ForecastResponse.self, from: forecastData)
        guard let current = forecast.current else {
            throw WebSearchProviderError.invalidResponse
        }

        let locationLabel = [match.name, match.admin1, match.country]
            .compactMap { $0 }.joined(separator: ", ")

        var parts: [String] = []
        if let temp = current.temperature_2m { parts.append("\(Int(temp.rounded()))°F") }
        if let feels = current.apparent_temperature { parts.append("feels like \(Int(feels.rounded()))°F") }
        parts.append(Self.description(forWeatherCode: current.weather_code))
        if let humidity = current.relative_humidity_2m { parts.append("\(Int(humidity))% humidity") }
        if let wind = current.wind_speed_10m { parts.append("wind \(Int(wind.rounded())) mph") }

        let answer = "Current weather in \(locationLabel): " + parts.joined(separator: ", ") + "."
        return WebSearchResult(
            answer: answer,
            sources: [WebSearchSource(title: "Weather data by Open-Meteo.com", url: "https://open-meteo.com")]
        )
    }

    /// Open-Meteo's geocoder matches against a bare place name, not a "city, region" phrase —
    /// a query like "san jose california" (the extracted place still carries the trailing state
    /// name) returns zero results even though "san jose" alone matches immediately. Rather than
    /// try to guess which trailing word is a state/country and strip only that, this tries the
    /// full phrase first and then progressively drops trailing words until something resolves —
    /// simple, and it degrades correctly for "san jose california usa" too.
    private func resolveLocation(_ place: String) async throws -> GeocodeResponse.Result {
        let words = place.split(separator: " ").map(String.init)
        guard !words.isEmpty else { throw WebSearchProviderError.noResults }

        var wordCount = words.count
        while wordCount >= 1 {
            let candidate = words.prefix(wordCount).joined(separator: " ")
            if let match = try? await geocode(candidate) {
                return match
            }
            wordCount -= 1
        }
        throw WebSearchProviderError.noResults
    }

    private func geocode(_ name: String) async throws -> GeocodeResponse.Result {
        var components = URLComponents(string: "https://geocoding-api.open-meteo.com/v1/search")!
        components.queryItems = [
            URLQueryItem(name: "name", value: name),
            URLQueryItem(name: "count", value: "5")
        ]
        guard let url = components.url else { throw WebSearchProviderError.invalidResponse }
        let (data, response) = try await URLSession.shared.data(from: url)
        try validateHTTPStatus(response)
        let decoded = try JSONDecoder().decode(GeocodeResponse.self, from: data)
        guard let results = decoded.results, !results.isEmpty else {
            throw WebSearchProviderError.noResults
        }

        // Open-Meteo ranks candidates by global significance (population), which picks the
        // wrong "Santa Clara" for a Bay Area user — Santa Clara, Cuba (pop. ~250k) outranks
        // Santa Clara, California (pop. ~126k) with nothing else to disambiguate on. Preferring
        // a same-named match in the device's own region over a foreign one is right far more
        // often than not, without hard-filtering out international results entirely — a bare
        // "Paris" still resolves to France, not some unrelated same-named town, whenever nothing
        // in the device's own country matches at all.
        if let deviceRegion = Locale.current.region?.identifier,
           let domesticMatch = results.first(where: { $0.country_code == deviceRegion }) {
            return domesticMatch
        }
        return results[0]
    }

    private static func description(forWeatherCode code: Int?) -> String {
        // WMO weather interpretation codes, per Open-Meteo's own documented mapping.
        guard let code else { return "conditions unknown" }
        switch code {
        case 0:            return "clear sky"
        case 1, 2, 3:       return "partly cloudy"
        case 45, 48:        return "foggy"
        case 51, 53, 55:    return "drizzle"
        case 56, 57:        return "freezing drizzle"
        case 61, 63, 65:    return "rain"
        case 66, 67:        return "freezing rain"
        case 71, 73, 75:    return "snow"
        case 77:            return "snow grains"
        case 80, 81, 82:    return "rain showers"
        case 85, 86:        return "snow showers"
        case 95:            return "thunderstorm"
        case 96, 99:        return "thunderstorm with hail"
        default:            return "conditions unknown"
        }
    }
}

// MARK: - DuckDuckGo Instant Answer (general fallback, keyless)

/// Free, keyless. This is an *instant answer* API — definitions, disambiguation, quick facts —
/// not full web search, so it frequently comes back empty for anything past a simple factual
/// question. That's surfaced as `.noResults` rather than a synthesized-sounding empty answer, so
/// the caller can be honest with the user about the gap instead of pretending nothing was asked.
nonisolated struct DuckDuckGoInstantAnswerProvider: WebSearchProvider {

    private struct Response: Decodable {
        var AbstractText: String?
        var AbstractURL: String?
        var AbstractSource: String?
        struct RelatedTopic: Decodable {
            var Text: String?
            var FirstURL: String?
        }
        var RelatedTopics: [RelatedTopic]?
    }

    func search(query: String) async throws -> WebSearchResult {
        var components = URLComponents(string: "https://api.duckduckgo.com/")!
        components.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "format", value: "json"),
            URLQueryItem(name: "no_html", value: "1"),
            URLQueryItem(name: "skip_disambig", value: "1")
        ]
        guard let url = components.url else { throw WebSearchProviderError.invalidResponse }
        let (data, response) = try await URLSession.shared.data(from: url)
        try validateHTTPStatus(response)
        let decoded = try JSONDecoder().decode(Response.self, from: data)

        if let abstract = decoded.AbstractText, !abstract.isEmpty {
            var sources: [WebSearchSource] = []
            if let url = decoded.AbstractURL, !url.isEmpty {
                sources.append(WebSearchSource(title: decoded.AbstractSource ?? "Source", url: url))
            }
            return WebSearchResult(answer: abstract, sources: sources)
        }

        // Fall back to the first related topic with real text — often a disambiguation-list
        // entry, still more useful than nothing. Held to the same relevance bar as the
        // Wikipedia provider: a disambiguation list can wander a long way from what was asked,
        // and a tangential entry presented as a search result is worse than no result.
        let subjectTerms = WebSearchClassifier.subjectTerms(in: query)
        if let related = decoded.RelatedTopics?.first(where: { topic in
            guard let text = topic.Text, !text.isEmpty else { return false }
            return WebSearchClassifier.isPlausiblyRelevant(text, toSubjectTerms: subjectTerms)
        }), let text = related.Text {
            let sources = related.FirstURL.map { [WebSearchSource(title: "DuckDuckGo", url: $0)] } ?? []
            return WebSearchResult(answer: text, sources: sources)
        }

        throw WebSearchProviderError.noResults
    }
}

// MARK: - Google News RSS (current events, keyless)

/// Free, keyless source for the one thing an encyclopedia structurally cannot answer: what
/// happened recently. Reads Google News' public RSS feed — a standard syndication endpoint meant
/// to be consumed by feed readers, not scraped HTML, so there's no bot-detection gate and no
/// terms problem.
///
/// Two feeds, picked by whether the question named a topic: a topic search for "what's in the
/// news for Santa Clara", and the general top-stories feed for a bare "tell me the latest news".
/// Headlines and publishers only — this returns what was reported and by whom, and the model
/// summarises from that rather than from article bodies it never sees.
nonisolated struct GoogleNewsProvider: WebSearchProvider {

    /// Collects `<item><title>` / `<link>` pairs. `XMLParser` is Foundation's own parser, so
    /// CDATA sections (which every title in this feed uses) are handled correctly — regex over
    /// XML would not be.
    private final class FeedParser: NSObject, XMLParserDelegate {
        var items: [(title: String, link: String)] = []
        private var insideItem = false
        private var currentElement = ""
        private var title = ""
        private var link = ""

        func parser(_ parser: XMLParser, didStartElement elementName: String,
                    namespaceURI: String?, qualifiedName: String?, attributes: [String: String]) {
            currentElement = elementName
            if elementName == "item" {
                insideItem = true
                title = ""
                link = ""
            }
        }

        func parser(_ parser: XMLParser, foundCharacters string: String) {
            guard insideItem else { return }
            if currentElement == "title" { title += string }
            if currentElement == "link" { link += string }
        }

        func parser(_ parser: XMLParser, foundCDATA CDATABlock: Data) {
            guard insideItem, currentElement == "title",
                  let text = String(data: CDATABlock, encoding: .utf8) else { return }
            title += text
        }

        func parser(_ parser: XMLParser, didEndElement elementName: String,
                    namespaceURI: String?, qualifiedName: String?) {
            if elementName == "item" {
                insideItem = false
                let cleanTitle = title.trimmingCharacters(in: .whitespacesAndNewlines)
                if !cleanTitle.isEmpty {
                    items.append((cleanTitle, link.trimmingCharacters(in: .whitespacesAndNewlines)))
                }
            }
            currentElement = ""
        }
    }

    func search(query: String) async throws -> WebSearchResult {
        // `subjectPhrase` strips the conversational wrapper — feeding the raw
        // "what's been in the news lately for Santa Clara?" to a search feed matches on the
        // filler words as much as the topic.
        let topic = WebSearchClassifier.subjectPhrase(in: query)

        var components: URLComponents
        if topic.isEmpty {
            // No topic named — "tell me the latest news". Top stories is the right answer.
            components = URLComponents(string: "https://news.google.com/rss")!
            components.queryItems = []
        } else {
            components = URLComponents(string: "https://news.google.com/rss/search")!
            components.queryItems = [URLQueryItem(name: "q", value: topic)]
        }
        components.queryItems?.append(contentsOf: [
            URLQueryItem(name: "hl", value: "en-US"),
            URLQueryItem(name: "gl", value: "US"),
            URLQueryItem(name: "ceid", value: "US:en")
        ])

        guard let url = components.url else { throw WebSearchProviderError.invalidResponse }
        let (data, response) = try await URLSession.shared.data(from: url)
        try validateHTTPStatus(response)

        let parserDelegate = FeedParser()
        let parser = XMLParser(data: data)
        parser.delegate = parserDelegate
        guard parser.parse() else { throw WebSearchProviderError.invalidResponse }

        // The first "item" in this feed is the channel description, not a story.
        var items = parserDelegate.items.filter { !$0.title.lowercased().hasSuffix("google news") }

        // A topic search still returns loosely-related stories; keep the ones that actually
        // mention what was asked about. Skipped for top-stories, where there's no topic to be
        // relevant to.
        if !topic.isEmpty {
            let terms = WebSearchClassifier.subjectTerms(in: query)
            let onTopic = items.filter { WebSearchClassifier.isPlausiblyRelevant($0.title, toSubjectTerms: terms) }
            if !onTopic.isEmpty { items = onTopic }
        }

        let top = Array(items.prefix(5))
        guard !top.isEmpty else { throw WebSearchProviderError.noResults }

        let heading = topic.isEmpty ? "Current top news headlines:" : "Recent news headlines about \(topic):"
        let answer = heading + "\n" + top.map { "• \($0.title)" }.joined(separator: "\n")
        let sources = top.prefix(3).compactMap { item -> WebSearchSource? in
            guard !item.link.isEmpty else { return nil }
            return WebSearchSource(title: item.title, url: item.link)
        }
        return WebSearchResult(answer: answer, sources: sources)
    }
}

// MARK: - Wikipedia (general fallback, keyless)

/// Free, keyless, and — unlike the instant-answer API above — it actually returns something for
/// most real questions. One request does both halves of the job (`generator=search` finds the
/// best-matching articles, `prop=extracts` returns their opening paragraphs), so a general
/// lookup costs a single round trip.
///
/// It is an encyclopedia, not a search index: it answers "what/who/where is X" well and cannot
/// answer "what happened today" at all. That gap is real and is why the optional Brave key
/// exists — see `WebSearchManager.generalSearch` for the order these are tried in.
nonisolated struct WikipediaProvider: WebSearchProvider {

    private struct Response: Decodable {
        struct Page: Decodable {
            var pageid: Int?
            /// Search rank. The `pages` object is keyed by page ID, so iterating it yields an
            /// arbitrary order — this is what puts the best match first again.
            var index: Int?
            var title: String?
            var extract: String?
        }
        struct Query: Decodable {
            var pages: [String: Page]?
        }
        var query: Query?
    }

    func search(query: String) async throws -> WebSearchResult {
        var components = URLComponents(string: "https://en.wikipedia.org/w/api.php")!
        components.queryItems = [
            URLQueryItem(name: "action", value: "query"),
            URLQueryItem(name: "generator", value: "search"),
            URLQueryItem(name: "gsrsearch", value: query),
            URLQueryItem(name: "gsrlimit", value: "2"),
            URLQueryItem(name: "prop", value: "extracts"),
            URLQueryItem(name: "exintro", value: "1"),
            URLQueryItem(name: "explaintext", value: "1"),
            URLQueryItem(name: "redirects", value: "1"),
            URLQueryItem(name: "format", value: "json")
        ]
        guard let url = components.url else { throw WebSearchProviderError.invalidResponse }
        var request = URLRequest(url: url)
        // Wikimedia's API etiquette policy requires a descriptive User-Agent; requests with a
        // generic one are liable to be throttled or refused.
        request.setValue("\(AppInfo.displayName)/\(AppInfo.version) (on-device iOS app)",
                         forHTTPHeaderField: "User-Agent")

        let (data, response) = try await URLSession.shared.data(for: request)
        try validateHTTPStatus(response)
        let decoded = try JSONDecoder().decode(Response.self, from: data)

        let pages = (decoded.query?.pages?.values).map { Array($0) } ?? []
        let ranked = pages.sorted { ($0.index ?? .max) < ($1.index ?? .max) }

        // Full-text search always returns *something* — it will happily answer "Can you search
        // and tell me the latest news?" with the TV series "Can This Love Be Translated?",
        // matching only on the conversational wrapper. Requiring the article to share a real
        // subject word with the question turns that into an honest "nothing found" instead of
        // handing the model a confident, unrelated article to answer from.
        let subjectTerms = WebSearchClassifier.subjectTerms(in: query)
        let usable = ranked.filter { page in
            guard !(page.extract ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  let title = page.title else { return false }
            return WebSearchClassifier.isPlausiblyRelevant(title, toSubjectTerms: subjectTerms)
        }
        guard !usable.isEmpty else { throw WebSearchProviderError.noResults }

        let answer = usable.compactMap { page -> String? in
            guard let title = page.title, let extract = page.extract else { return nil }
            // Intros run long; the model only needs enough to answer from.
            return "\(title): \(String(extract.prefix(700)))"
        }.joined(separator: "\n\n")
        guard !answer.isEmpty else { throw WebSearchProviderError.noResults }

        let sources = usable.compactMap { page -> WebSearchSource? in
            guard let title = page.title, let id = page.pageid else { return nil }
            return WebSearchSource(title: "Wikipedia — \(title)",
                                   url: "https://en.wikipedia.org/?curid=\(id)")
        }
        return WebSearchResult(answer: answer, sources: sources)
    }
}

// MARK: - Brave Search (general, requires the user's own key)

/// Real general web search. Used only when the user has supplied their own Brave Search API key
/// in Settings — no key is ever bundled with the app; see `KeychainStore`. REST/JSON with a
/// single header for auth, which is why this is the one "bring your own key" integration this
/// app ships rather than a provider-agnostic plugin system.
nonisolated struct BraveSearchProvider: WebSearchProvider {
    let apiKey: String

    private struct Response: Decodable {
        struct WebResults: Decodable {
            struct Result: Decodable {
                var title: String?
                var url: String?
                var description: String?
            }
            var results: [Result]?
        }
        var web: WebResults?
    }

    func search(query: String) async throws -> WebSearchResult {
        var components = URLComponents(string: "https://api.search.brave.com/res/v1/web/search")!
        components.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "count", value: "5")
        ]
        guard let url = components.url else { throw WebSearchProviderError.invalidResponse }
        var request = URLRequest(url: url)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue(apiKey, forHTTPHeaderField: "X-Subscription-Token")

        let (data, response) = try await URLSession.shared.data(for: request)
        try validateHTTPStatus(response)
        let decoded = try JSONDecoder().decode(Response.self, from: data)
        guard let results = decoded.web?.results, !results.isEmpty else {
            throw WebSearchProviderError.noResults
        }

        // Snippets, not a synthesized answer — Brave returns search results, not a single
        // instant answer, so the LLM gets raw material to summarize, same as a person skimming
        // a results page would.
        let top = results.prefix(3)
        let answer = top.compactMap { result -> String? in
            guard let title = result.title, let description = result.description else { return nil }
            return "\(title): \(description)"
        }.joined(separator: "\n")
        guard !answer.isEmpty else { throw WebSearchProviderError.noResults }

        let sources = top.compactMap { result -> WebSearchSource? in
            guard let title = result.title, let url = result.url else { return nil }
            return WebSearchSource(title: title, url: url)
        }
        return WebSearchResult(answer: answer, sources: sources)
    }
}
