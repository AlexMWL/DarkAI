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
        if profile.isEmpty {
            return "You are DarkAI, a highly capable assistant. Provide concise, helpful answers."
        }
        
        return """
        critical instruction: drop all ai formalities. forget perfect grammar. u must explicitly mimic the user's exact typos, lowercase letters, missing punctuation, and slang. respond exactly as the user texts fr.

        [user style matrix]:
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
                currentProfile = newTraits.joined(separator: "\\n- ")
            } else {
                currentProfile += "\\n- " + newTraits.joined(separator: "\\n- ")
            }
            modelPersonalities[modelName] = currentProfile
            
            maturityScore = min(1.0, maturityScore + (Double(newTraits.count) * 0.05))
            isMature = maturityScore >= 0.7
            
            savePersonalities()
        }
        
        // --- 2. Batching & Background Style Analysis ---
        messageBatch.append(message)
        
        if messageBatch.count >= 3, let llm = llmManager {
            let batchedText = messageBatch.joined(separator: "\n\n")
            messageBatch.removeAll() // Clear batch
            
            Task {
                let analysisPrompt = """
                Analyze the following user messages strictly for their communication style. Look specifically for:
                1. Typos and misspellings
                2. Lack of capitalization or punctuation
                3. Run-on sentences
                4. Distinct slang or vocabulary
                
                Respond ONLY with a bulleted list of strict instructions on how to mimic this style exactly. Do not include any filler text.
                
                User Messages:
                \(batchedText)
                """
                
                if let analysis = await llm.generateBackgroundAnalysis(prompt: analysisPrompt) {
                    await MainActor.run {
                        var updatedProfile = self.modelPersonalities[modelName] ?? ""
                        if !updatedProfile.contains(analysis) {
                            updatedProfile += "\n\n[STYLE ANALYSIS]\n" + analysis.trimmingCharacters(in: .whitespacesAndNewlines)
                            
                            // Limit size
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
}
