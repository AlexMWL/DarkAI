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

    var errorDescription: String? {
        switch self {
        case .noResults:       return "No results were found for that search."
        case .invalidResponse: return "The search provider returned something unexpected."
        }
    }
}

// MARK: - Provider protocol

protocol WebSearchProvider {
    /// `query` is a place name for `OpenMeteoProvider`, free text for the others — routing
    /// which provider gets called is `WebSearchManager`'s job, not this protocol's.
    func search(query: String) async throws -> WebSearchResult
}

// MARK: - Open-Meteo (weather, keyless)

/// Free, keyless weather lookup. Two calls against the same free service: Open-Meteo's own
/// geocoding endpoint to resolve a place name to coordinates, then the forecast endpoint for
/// current conditions. No API key, no account — but its CC-BY 4.0 data license does require
/// attribution, which is why the source below always reads "Weather data by Open-Meteo.com"
/// rather than being omitted like a typical search result would be.
struct OpenMeteoProvider: WebSearchProvider {

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
        let (forecastData, _) = try await URLSession.shared.data(from: forecastComponents.url!)
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
        let (data, _) = try await URLSession.shared.data(from: components.url!)
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
struct DuckDuckGoInstantAnswerProvider: WebSearchProvider {

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
        let (data, _) = try await URLSession.shared.data(from: components.url!)
        let decoded = try JSONDecoder().decode(Response.self, from: data)

        if let abstract = decoded.AbstractText, !abstract.isEmpty {
            var sources: [WebSearchSource] = []
            if let url = decoded.AbstractURL, !url.isEmpty {
                sources.append(WebSearchSource(title: decoded.AbstractSource ?? "Source", url: url))
            }
            return WebSearchResult(answer: abstract, sources: sources)
        }

        // Fall back to the first related topic with real text — often a disambiguation-list
        // entry, still more useful than nothing.
        if let related = decoded.RelatedTopics?.first(where: { ($0.Text?.isEmpty == false) }),
           let text = related.Text {
            let sources = related.FirstURL.map { [WebSearchSource(title: "DuckDuckGo", url: $0)] } ?? []
            return WebSearchResult(answer: text, sources: sources)
        }

        throw WebSearchProviderError.noResults
    }
}

// MARK: - Brave Search (general, requires the user's own key)

/// Real general web search. Used only when the user has supplied their own Brave Search API key
/// in Settings — no key is ever bundled with the app; see `KeychainStore`. REST/JSON with a
/// single header for auth, which is why this is the one "bring your own key" integration this
/// app ships rather than a provider-agnostic plugin system.
struct BraveSearchProvider: WebSearchProvider {
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
        var request = URLRequest(url: components.url!)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue(apiKey, forHTTPHeaderField: "X-Subscription-Token")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw WebSearchProviderError.invalidResponse
        }
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
