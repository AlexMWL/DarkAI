import Foundation
import Combine

class PersonalityManager: ObservableObject {
    @Published var modelPersonalities: [String: String] = [:]
    @Published var maturityScore: Double = 0.0
    @Published var isMature: Bool = false
    
    private let storageKey = "DarkAI_ModelPersonalities"
    private let maturityKey = "DarkAI_MaturityScore"
    private var messageBatch: [String] = []
    
    init() {
        loadPersonalities()
    }
    
    private func loadPersonalities() {
        if let data = UserDefaults.standard.data(forKey: storageKey),
           let decoded = try? JSONDecoder().decode([String: String].self, from: data) {
            self.modelPersonalities = decoded
        }
        
        let savedMaturity = UserDefaults.standard.double(forKey: maturityKey)
        self.maturityScore = savedMaturity
        self.isMature = savedMaturity >= 0.7
    }
    
    private func savePersonalities() {
        if let encoded = try? JSONEncoder().encode(modelPersonalities) {
            UserDefaults.standard.set(encoded, forKey: storageKey)
        }
        UserDefaults.standard.set(maturityScore, forKey: maturityKey)
    }
    
    var databaseSizeString: String {
        guard let data = UserDefaults.standard.data(forKey: storageKey) else {
            return "0 B"
        }
        let bytes = Double(data.count)
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
    
    func resetPersonality(for modelName: String) {
        modelPersonalities.removeValue(forKey: modelName)
        maturityScore = 0.0
        isMature = false
        savePersonalities()
    }
    
    func getPersonality(for modelName: String) -> String {
        let profile = modelPersonalities[modelName] ?? ""
        guard !profile.isEmpty else { return "" }

        // Require at least 2 distinct trait lines before personality has enough signal to apply
        let entryCount = profile.components(separatedBy: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .count
        guard entryCount >= 2 else { return "" }

        // CRITICAL: lines starting "The AI has previously said..." are the model's OWN
        // established opinions/preferences — separate from the "Long-Term Conversational
        // Memories" block elsewhere in the prompt, which is about the USER. Without this
        // explicit split the model tends to answer "what's your favorite X" by echoing back
        // something it saw stored about the user instead of forming/recalling its own answer.
        return """
        [AI Personality Matrix — two distinct kinds of entries below, do not mix them up:
        • Lines starting "The AI has previously said..." are YOUR OWN opinions and preferences. \
        If asked again, give the SAME answer for consistency. Never state something from the \
        user's memories as if it were your own preference — those are two separate people.
        • All other lines describe the user's communication style/facts — adapt your tone to \
        match them, but they belong to the user, not you.]
        \(profile)
        """
    }
    
    func analyzeUserMessage(_ message: String, modelName: String, llmManager: LLMManager?) {
        guard !modelName.isEmpty else { return }
        
        // --- 1. Basic Fast Extraction (Main Thread) ---
        var currentProfile = modelPersonalities[modelName] ?? ""
        let lower = message.lowercased()
        let words = lower.components(separatedBy: .punctuationCharacters).joined().components(separatedBy: .whitespaces)
        
        let slangList = ["lol", "lmao", "bruh", "tbh", "fr", "frfr", "ngl", "idk", "rn", "gotcha", "yep", "yeah", "dude", "hey", "yo", "bet", "cap", "vibes"]
        var usedSlang = Set<String>()
        for word in words {
            if slangList.contains(word) {
                usedSlang.insert(word)
            }
        }
        
        var newTraits: [String] = []
        for slang in usedSlang {
            if !currentProfile.contains("'\(slang)'") {
                newTraits.append("Occasionally use casual slang like '\(slang)'.")
            }
        }
        
        let sentences = message.components(separatedBy: CharacterSet(charactersIn: ".!?\n"))
        let aggressiveTriggers = ["i like ", "i am ", "i plan to ", "i need to ", "i want to ", "my favorite ", "i have ", "my goal is ", "i love ", "i hate ", "im ", "i'm "]
        
        for sentence in sentences {
            let cleanSentence = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            let lowerSentence = cleanSentence.lowercased()
            
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
            modelPersonalities[modelName] = currentProfile
            
            maturityScore = min(1.0, maturityScore + (Double(newTraits.count) * 0.05))
            isMature = maturityScore >= 0.7
            
            savePersonalities()
        }
        
        // --- 2. Batching & Background Style Analysis ---
        messageBatch.append(message)
        
        if messageBatch.count >= 3 {
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
                        var updatedProfile = self.modelPersonalities[modelName] ?? ""
                        if !analysis.isEmpty && !updatedProfile.contains(analysis) {
                            updatedProfile += "\n\n[STYLE ANALYSIS]\n" + analysis
                            
                            // Limit size to prevent unbounded growth
                            let lines = updatedProfile.components(separatedBy: "\n")
                            if lines.count > 100 {
                                updatedProfile = ([lines[0]] + lines.suffix(99)).joined(separator: "\n")
                            }
                            
                            self.modelPersonalities[modelName] = updatedProfile
                            
                            self.maturityScore = min(1.0, self.maturityScore + 0.10)
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
    func analyzeAssistantMessage(_ message: String, modelName: String) {
        guard !modelName.isEmpty else { return }
        let text = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        var currentProfile = modelPersonalities[modelName] ?? ""
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
        modelPersonalities[modelName] = currentProfile

        maturityScore = min(1.0, maturityScore + (Double(newTraits.count) * 0.05))
        isMature = maturityScore >= 0.7

        savePersonalities()
    }
}
