import Foundation
import Combine

/// Running measurements of how the user actually writes.
///
/// The original personality system learned style by looking for words from a fixed slang list.
/// That only fires when someone happens to type "lmao", says nothing about the far more visible
/// traits — sentence length, whether they capitalise, whether they punctuate at all — and learns
/// nothing from a user whose voice is distinctive but doesn't use listed slang.
///
/// These are cumulative rates over every message, so the profile sharpens continuously instead of
/// waiting for keyword hits, and it describes the user in terms the model can actually imitate.
struct StyleMetrics: Codable {
    var messageCount: Int = 0
    var totalWords: Int = 0
    var lowercaseStarts: Int = 0
    var missingTerminalPunctuation: Int = 0
    var questions: Int = 0
    var exclamations: Int = 0
    var emojiMessages: Int = 0
    var contractions: Int = 0
    var slangHits: Int = 0
    var profanityHits: Int = 0
    var multiSentence: Int = 0
    var totalWordLength: Int = 0

    var averageWords: Double { messageCount > 0 ? Double(totalWords) / Double(messageCount) : 0 }
    var averageWordLength: Double { totalWords > 0 ? Double(totalWordLength) / Double(totalWords) : 0 }

    private func rate(_ count: Int) -> Double {
        messageCount > 0 ? Double(count) / Double(messageCount) : 0
    }

    var lowercaseRate: Double { rate(lowercaseStarts) }
    var noPunctuationRate: Double { rate(missingTerminalPunctuation) }
    var questionRate: Double { rate(questions) }
    var exclamationRate: Double { rate(exclamations) }
    var emojiRate: Double { rate(emojiMessages) }
    var contractionRate: Double { rate(contractions) }
    var slangRate: Double { rate(slangHits) }
    var profanityRate: Double { rate(profanityHits) }
    var multiSentenceRate: Double { rate(multiSentence) }

    /// Enough signal to be worth acting on. Deliberately low — a handful of messages is plenty
    /// to tell a terse all-lowercase writer from someone who writes full paragraphs.
    var hasSignal: Bool { messageCount >= 4 }

    /// Turns the measurements into instructions the model can follow.
    ///
    /// Only traits that are clearly present are described. Emitting every metric on every turn
    /// would bury the strong signals among noise and waste context.
    func directives() -> [String] {
        guard hasSignal else { return [] }
        var lines: [String] = []

        switch averageWords {
        case ..<8:    lines.append("The user writes very short messages. Keep replies brief and direct — a few sentences at most.")
        case ..<25:   lines.append("The user writes fairly short messages. Favour concise replies over long ones.")
        case 60...:   lines.append("The user writes long, detailed messages. Match that depth rather than answering in one line.")
        default:      break
        }

        if lowercaseRate > 0.6 {
            lines.append("The user rarely capitalises. Write casually and don't over-format.")
        }
        if noPunctuationRate > 0.5 {
            lines.append("The user often skips ending punctuation. Keep a relaxed, unpolished tone.")
        }
        if slangRate > 0.3 {
            lines.append("The user uses casual slang. Match that register — informal, never stiff or corporate.")
        }
        if contractionRate > 0.5 {
            lines.append("The user speaks in contractions. Use them too; avoid formal phrasing.")
        }
        if emojiRate > 0.25 {
            lines.append("The user uses emoji. An occasional one in reply is welcome.")
        } else if emojiRate < 0.02 && messageCount > 10 {
            lines.append("The user never uses emoji. Don't use them.")
        }
        if questionRate > 0.6 {
            lines.append("The user mostly asks questions. Lead with the answer, then any detail.")
        }
        if exclamationRate > 0.35 {
            lines.append("The user writes with energy and exclamation. Be warm rather than flat.")
        }
        if profanityRate > 0.2 {
            lines.append("The user swears casually. Don't moralise about language or sanitise your tone.")
        }
        if averageWordLength > 5.6 && messageCount > 8 {
            lines.append("The user's vocabulary is precise and technical. Don't over-explain basics.")
        }
        if multiSentenceRate < 0.2 && messageCount > 8 {
            lines.append("The user writes in single thoughts, not paragraphs. Keep structure flat — avoid headers and long bullet lists.")
        }
        return lines
    }

    /// Sums several sets of measurements into one.
    ///
    /// Every counter here is a tally over the user's own messages, and there is only ever one
    /// user — so measurements gathered while different models were loaded describe the same
    /// person and add together cleanly. Used once, to fold the old per-model records into the
    /// single shared profile.
    static func merged(_ all: [StyleMetrics]) -> StyleMetrics {
        var out = StyleMetrics()
        for m in all {
            out.messageCount += m.messageCount
            out.totalWords += m.totalWords
            out.lowercaseStarts += m.lowercaseStarts
            out.missingTerminalPunctuation += m.missingTerminalPunctuation
            out.questions += m.questions
            out.exclamations += m.exclamations
            out.emojiMessages += m.emojiMessages
            out.contractions += m.contractions
            out.slangHits += m.slangHits
            out.profanityHits += m.profanityHits
            out.multiSentence += m.multiSentence
            out.totalWordLength += m.totalWordLength
        }
        return out
    }

    /// Folds one message into the running totals.
    mutating func record(_ message: String) {
        let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        messageCount += 1

        let words = trimmed
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
        totalWords += words.count
        totalWordLength += words.reduce(0) { $0 + $1.count }

        if let first = trimmed.first, first.isLetter, first.isLowercase { lowercaseStarts += 1 }
        if let last = trimmed.last, !".!?".contains(last) { missingTerminalPunctuation += 1 }
        if trimmed.contains("?") { questions += 1 }
        if trimmed.contains("!") { exclamations += 1 }
        if trimmed.unicodeScalars.contains(where: { $0.properties.isEmojiPresentation }) { emojiMessages += 1 }
        if trimmed.contains("'") || trimmed.lowercased().contains("dont") || trimmed.lowercased().contains("cant") {
            contractions += 1
        }

        let sentenceCount = trimmed
            .components(separatedBy: CharacterSet(charactersIn: ".!?"))
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            .count
        if sentenceCount > 1 { multiSentence += 1 }

        let lowerWords = Set(words.map {
            $0.lowercased().trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        })
        if !lowerWords.isDisjoint(with: StyleMetrics.slangWords) { slangHits += 1 }
        if !lowerWords.isDisjoint(with: StyleMetrics.profanityWords) { profanityHits += 1 }
    }

    private static let slangWords: Set<String> = [
        "lol", "lmao", "lmfao", "bruh", "tbh", "fr", "frfr", "ngl", "idk", "idc", "rn",
        "gotcha", "yep", "yeah", "yup", "nah", "dude", "hey", "yo", "bet", "cap", "vibes",
        "sus", "lowkey", "highkey", "deadass", "bruv", "mate", "innit", "imo", "imho",
        "afaik", "btw", "omg", "wtf", "smh", "ikr", "ty", "thx", "pls", "plz", "u", "ur",
        "rly", "kinda", "sorta", "gonna", "wanna", "gimme", "dunno", "aight", "sup"
    ]

    private static let profanityWords: Set<String> = [
        "damn", "hell", "crap", "shit", "fuck", "fucking", "fuckin", "ass", "bastard",
        "bitch", "piss", "bollocks", "bloody", "wtf", "af"
    ]
}

class PersonalityManager: ObservableObject {
    /// The one personality, shared by every model.
    ///
    /// This was keyed by model, which meant switching models started the adaptation over and the
    /// same user was learned several times in parallel. There is only one user, so there is one
    /// profile: whatever the app has worked out about how they write applies whichever model is
    /// answering.
    @Published var personality: String = ""
    @Published var maturityScore: Double = 0.0
    @Published var isMature: Bool = false
    
    private let storageKey = "DarkAI_Personality"
    private let maturityKey = "DarkAI_MaturityScore"
    private let metricsKey = "DarkAI_StyleMetricsUnified"
    /// Pre-unification keys, read once at launch and then deleted.
    private let legacyProfilesKey = "DarkAI_ModelPersonalities"
    private let legacyMetricsKey = "DarkAI_StyleMetrics"
    private var messageBatch: [String] = []

    @Published private(set) var styleMetrics = StyleMetrics()

    init() {
        loadPersonalities()
        loadMetrics()
        migrateLegacyPerModelData()
    }

    /// Folds the old per-model records into the single shared profile, once.
    ///
    /// Rather than discarding what the app had already learned about the user. The measurements
    /// are summed — they are counts of that one user's messages, so they add. The prose profiles
    /// cannot be merged that way without producing a contradictory blob, so the longest is kept
    /// as the most developed. The old keys are removed afterwards so this never runs twice.
    private func migrateLegacyPerModelData() {
        let defaults = UserDefaults.standard
        if personality.isEmpty,
           let data = defaults.data(forKey: legacyProfilesKey),
           let old = try? JSONDecoder().decode([String: String].self, from: data),
           let longest = old.values.max(by: { $0.count < $1.count }) {
            personality = longest
            savePersonalities()
        }
        if styleMetrics.messageCount == 0,
           let data = defaults.data(forKey: legacyMetricsKey),
           let old = try? JSONDecoder().decode([String: StyleMetrics].self, from: data),
           !old.isEmpty {
            styleMetrics = StyleMetrics.merged(Array(old.values))
            saveMetrics()
        }
        defaults.removeObject(forKey: legacyProfilesKey)
        defaults.removeObject(forKey: legacyMetricsKey)
    }

    private func loadMetrics() {
        guard let data = UserDefaults.standard.data(forKey: metricsKey),
              let decoded = try? JSONDecoder().decode(StyleMetrics.self, from: data) else { return }
        styleMetrics = decoded
    }

    private func saveMetrics() {
        guard let encoded = try? JSONEncoder().encode(styleMetrics) else { return }
        UserDefaults.standard.set(encoded, forKey: metricsKey)
    }

    /// Messages observed for a model — drives the maturity readout in Settings.
    var observedMessageCount: Int { styleMetrics.messageCount }
    
    private func loadPersonalities() {
        personality = UserDefaults.standard.string(forKey: storageKey) ?? ""
        maturityScore = UserDefaults.standard.double(forKey: maturityKey)
        isMature = maturityScore >= 0.7
    }
    
    private func savePersonalities() {
        UserDefaults.standard.set(personality, forKey: storageKey)
        UserDefaults.standard.set(maturityScore, forKey: maturityKey)
    }
    
    /// Bytes held by the shared profile: the prose and the measurements together.
    func databaseSizeBytes() -> Int {
        let prose = personality.utf8.count
        let metrics = (try? JSONEncoder().encode(styleMetrics))?.count ?? 0
        return prose == 0 && styleMetrics.messageCount == 0 ? 0 : prose + metrics
    }

    /// Size of what the reset button clears — which is now everything, so it reaches 0 B.
    var databaseSizeString: String {
        let bytes = Double(databaseSizeBytes())
        if bytes < 1024 {
            return "\(Int(bytes)) B"
        } else if bytes < 1024 * 1024 {
            return String(format: "%.1f KB", bytes / 1024)
        } else if bytes < 1024 * 1024 * 1024 {
            return String(format: "%.2f MB", bytes / (1024 * 1024))
        } else {
            return String(format: "%.2f GB", bytes / (1024 * 1024 * 1024))
        }
    }

    /// Erases the profile and the measurements behind it.
    func resetPersonality() {
        personality = ""
        styleMetrics = StyleMetrics()
        maturityScore = 0.0
        isMature = false
        savePersonalities()
        saveMetrics()
    }
    
    func getPersonality() -> String {
        let profile = personality
        let styleDirectives = (styleMetrics).directives()

        // Measured style stands on its own. It's available after a handful of messages, well
        // before the LLM-written prose profile has anything in it, so the assistant starts
        // adapting early rather than waiting for the slower path to accumulate.
        if profile.isEmpty {
            guard !styleDirectives.isEmpty else { return "" }
            return "[How the user writes — match this]\n" + styleDirectives.joined(separator: "\n")
        }

        // The prose profile needs a couple of entries before it's worth applying — but the
        // measured style is independent of it and shouldn't be discarded just because the
        // slower LLM-written half hasn't filled in yet.
        let entryCount = profile.components(separatedBy: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .count
        guard entryCount >= 2 else {
            return styleDirectives.isEmpty
                ? ""
                : "[How the user writes — match this]\n" + styleDirectives.joined(separator: "\n")
        }

        // CRITICAL: lines starting "The AI has previously said..." are the model's OWN
        // established opinions/preferences — separate from the "Long-Term Conversational
        // Memories" block elsewhere in the prompt, which is about the USER. Without this
        // explicit split the model tends to answer "what's your favorite X" by echoing back
        // something it saw stored about the user instead of forming/recalling its own answer.
        let stylePreamble = styleDirectives.isEmpty
            ? ""
            : "[How the user writes — match this]\n" + styleDirectives.joined(separator: "\n") + "\n\n"

        return stylePreamble + """
        [AI Personality Matrix — two distinct kinds of entries below, do not mix them up:
        • Lines starting "The AI has previously said..." are YOUR OWN opinions and preferences. \
        If asked again, give the SAME answer for consistency. Never state something from the \
        user's memories as if it were your own preference — those are two separate people.
        • All other lines describe the user's communication style/facts — adapt your tone to \
        match them, but they belong to the user, not you.]
        \(profile)
        """
    }
    
    func analyzeUserMessage(_ message: String, llmManager: LLMManager?) {

        // 1. Style measurement (every message, no keyword required).
        // This runs unconditionally rather than only when a listed slang word appears, which is
        // what makes the profile sharpen from ordinary messages instead of stalling on users
        // whose voice is distinctive but doesn't happen to use the vocabulary in a fixed list.
        var metrics = styleMetrics
        metrics.record(message)
        styleMetrics = metrics
        saveMetrics()

        var currentProfile = personality
        var newTraits: [String] = []
        
        let sentences = message.components(separatedBy: CharacterSet(charactersIn: ".!?\n"))
        let aggressiveTriggers = ["i like ", "i am ", "i plan to ", "i need to ", "i want to ", "my favorite ", "i have ", "my goal is ", "i love ", "i hate ", "im ", "i'm "]
        
        for sentence in sentences {
            let cleanSentence = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            let lowerSentence = cleanSentence.lowercased()

            // This is the most common of the three fact-extraction paths in this file — twelve
            // everyday English openers ("i am", "i have", "im"/"i'm"...) — so it's also the one
            // most likely to catch a long run-on message when a `.!?\n` separator doesn't show up
            // for a while. Its two siblings (`addFact` below, and the style-analysis block above)
            // both cap what they add before it ever reaches `personality`; this path used to add
            // whatever the message contained verbatim, unbounded, straight into a string injected
            // into every future system prompt.
            guard cleanSentence.count < 140 else { continue }

            for trigger in aggressiveTriggers {
                if lowerSentence.hasPrefix(trigger) && cleanSentence.count > trigger.count + 2 {
                    let fact = "The user stated: '\(cleanSentence)'."
                    if !currentProfile.contains(fact) {
                        newTraits.append("Remember: \(fact)")
                    }
                    break
                }
            }
        }

        if !newTraits.isEmpty {
            if currentProfile.isEmpty {
                currentProfile = newTraits.joined(separator: "\n")
            } else {
                currentProfile += "\n" + newTraits.joined(separator: "\n")
            }

            // Cap growth the same way `analyzeAssistantMessage` and the style-analysis pass below
            // both already do.
            let lines = currentProfile.components(separatedBy: "\n")
            if lines.count > 100 {
                currentProfile = ([lines[0]] + lines.suffix(99)).joined(separator: "\n")
            }
            personality = currentProfile

            maturityScore = min(1.0, maturityScore + (Double(newTraits.count) * 0.08))
            isMature = maturityScore >= 0.7

            savePersonalities()
        }
        
        // 2. Batching & background style analysis.
        messageBatch.append(message)
        
        if messageBatch.count >= 2 {
            let batchedText = messageBatch.joined(separator: "\n\n")
            messageBatch.removeAll() // Always clear batch to prevent unbounded growth
            
            guard let llm = llmManager else { return }
            
            Task {
                let analysisPrompt = """
                Analyze the following user messages strictly for their communication style. Look specifically for:
                1. Typos and misspellings
                2. Lack of capitalization or punctuation
                3. Run-on sentences
                4. Distinct slang or vocabulary
                
                Summarize the communication style as a short list of plain-text observations. Do not use bullet points, asterisks, or markdown formatting.
                
                User Messages:
                \(batchedText)
                """
                
                if let rawAnalysis = await llm.generateBackgroundAnalysis(prompt: analysisPrompt) {
                    // Strip any markdown formatting before embedding in the personality profile
                    var analysis = rawAnalysis
                    analysis = analysis.replacingOccurrences(of: "**", with: "")
                    analysis = analysis.replacingOccurrences(of: "__", with: "")
                    let cleanedLines = analysis.components(separatedBy: "\n").map { line -> String in
                        var l = line.trimmingCharacters(in: .whitespacesAndNewlines)
                        if l.hasPrefix("### ") { l = String(l.dropFirst(4)) }
                        else if l.hasPrefix("## ") { l = String(l.dropFirst(3)) }
                        else if l.hasPrefix("# ") { l = String(l.dropFirst(2)) }
                        if l.hasPrefix("- ") { l = String(l.dropFirst(2)) }
                        else if l.hasPrefix("• ") { l = String(l.dropFirst(2)) }
                        else if l.hasPrefix("* ") { l = String(l.dropFirst(2)) }
                        return l
                    }.filter { !$0.isEmpty }
                    analysis = cleanedLines.joined(separator: "\n")
                    
                    await MainActor.run {
                        var updatedProfile = self.personality
                        if !analysis.isEmpty && !updatedProfile.contains(analysis) {
                            updatedProfile += "\n\n[STYLE ANALYSIS]\n" + analysis
                            
                            // Limit size to prevent unbounded growth
                            let lines = updatedProfile.components(separatedBy: "\n")
                            if lines.count > 100 {
                                updatedProfile = ([lines[0]] + lines.suffix(99)).joined(separator: "\n")
                            }
                            
                            self.personality = updatedProfile
                            
                            self.maturityScore = min(1.0, self.maturityScore + 0.15)
                            self.isMature = self.maturityScore >= 0.7
                            
                            self.savePersonalities()
                        }
                    }
                }
            }
        }
    }

    // MARK: - AI Self-Identity Extraction
    // Captures the model's OWN stated likes/dislikes/favorites from its responses — e.g.
    // "I love jazz" or "my favorite car is a Tesla" — so it can stay consistent when asked
    // again later, instead of forming a fresh (or worse, borrowing the user's) answer every
    // time. Purely local string matching, no LLM call, safe to run after every response.
    func analyzeAssistantMessage(_ message: String) {
        let text = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        var currentProfile = personality
        var newTraits: [String] = []

        func addFact(_ fact: String) {
            guard !fact.isEmpty, fact.count < 140, !currentProfile.contains(fact) else { return }
            newTraits.append(fact)
        }

        let likeTriggers: [(String, (String) -> String)] = [
            ("i love ", { "The AI has previously said it loves \($0)." }),
            ("i really like ", { "The AI has previously said it likes \($0)." }),
            ("i like ", { "The AI has previously said it likes \($0)." }),
            ("i enjoy ", { "The AI has previously said it enjoys \($0)." }),
            ("i prefer ", { "The AI has previously said it prefers \($0)." }),
            ("i'm a fan of ", { "The AI has previously said it's a fan of \($0)." }),
            ("i am a fan of ", { "The AI has previously said it's a fan of \($0)." }),
        ]
        let dislikeTriggers: [(String, (String) -> String)] = [
            ("i hate ", { "The AI has previously said it dislikes \($0)." }),
            ("i dislike ", { "The AI has previously said it dislikes \($0)." }),
            ("i don't like ", { "The AI has previously said it dislikes \($0)." }),
            ("i do not like ", { "The AI has previously said it dislikes \($0)." }),
        ]

        let sentences = text.components(separatedBy: CharacterSet(charactersIn: ".!?\n"))
        for sentence in sentences {
            let cleanSentence = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !cleanSentence.isEmpty else { continue }
            let lowerSentence = cleanSentence.lowercased()

            for (prefix, format) in likeTriggers + dislikeTriggers {
                guard lowerSentence.hasPrefix(prefix), cleanSentence.count > prefix.count + 2 else { continue }
                let value = String(cleanSentence.dropFirst(prefix.count))
                    .trimmingCharacters(in: CharacterSet(charactersIn: " .!?,;"))
                guard !value.isEmpty, value.count < 100 else { continue }
                addFact(format(value))
                break
            }

            // "My favorite X is Y" / "My favourite X is Y"
            for favoritePrefix in ["my favorite ", "my favourite "] {
                guard lowerSentence.hasPrefix(favoritePrefix) else { continue }
                let rest = cleanSentence.dropFirst(favoritePrefix.count)
                guard let isRange = rest.lowercased().range(of: " is ") else { break }
                let subject = rest[..<isRange.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
                let value = rest[isRange.upperBound...].trimmingCharacters(in: CharacterSet(charactersIn: " .!?,;"))
                if !subject.isEmpty, !value.isEmpty, value.count < 100 {
                    addFact("The AI has previously said its favorite \(subject) is \(value).")
                }
                break
            }
        }

        guard !newTraits.isEmpty else { return }

        if currentProfile.isEmpty {
            currentProfile = newTraits.joined(separator: "\n")
        } else {
            currentProfile += "\n" + newTraits.joined(separator: "\n")
        }

        // Cap growth the same way the background style-analysis pass does.
        let lines = currentProfile.components(separatedBy: "\n")
        if lines.count > 100 {
            currentProfile = ([lines[0]] + lines.suffix(99)).joined(separator: "\n")
        }
        personality = currentProfile

        maturityScore = min(1.0, maturityScore + (Double(newTraits.count) * 0.08))
        isMature = maturityScore >= 0.7

        savePersonalities()
    }
}
