import Foundation
import Combine

class PersonalityManager: ObservableObject {
    @Published var modelPersonalities: [String: String] = [:]
    
    private let storageKey = "DarkAI_ModelPersonalities"
    
    init() {
        loadPersonalities()
    }
    
    private func loadPersonalities() {
        if let data = UserDefaults.standard.data(forKey: storageKey),
           let decoded = try? JSONDecoder().decode([String: String].self, from: data) {
            self.modelPersonalities = decoded
        }
    }
    
    private func savePersonalities() {
        if let encoded = try? JSONEncoder().encode(modelPersonalities) {
            UserDefaults.standard.set(encoded, forKey: storageKey)
        }
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
        savePersonalities()
    }
    
    func getPersonality(for modelName: String) -> String {
        return modelPersonalities[modelName] ?? "You are DarkAI, a highly capable assistant. Provide concise, helpful answers."
    }
    
    func analyzeUserMessage(_ message: String, modelName: String) {
        guard !modelName.isEmpty else { return }
        
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
        
        // If the user's message is entirely lowercase and decently long, they are a casual typer.
        let isAllLowercase = (message == lower) && message.trimmingCharacters(in: .whitespacesAndNewlines).count > 8
        let hasEmojis = message.unicodeScalars.contains { $0.properties.isEmojiPresentation }
        let isVerbose = message.count > 200
        
        var newTraits: [String] = []
        
        if isAllLowercase && !currentProfile.contains("all-lowercase") {
            newTraits.append("Type in a casual, all-lowercase style without strict capitalization.")
        }
        
        if hasEmojis && !currentProfile.contains("emojis") {
            newTraits.append("Use emojis occasionally to match the user's energy.")
        }
        
        if isVerbose && !currentProfile.contains("verbose") && !currentProfile.contains("detailed") {
            newTraits.append("Provide highly detailed and verbose answers since the user likes long explanations.")
        }
        
        for slang in usedSlang {
            if !currentProfile.contains("'\(slang)'") {
                newTraits.append("Occasionally use casual slang like '\(slang)'.")
            }
        }
        
        // Aggressively extract facts, plans, and preferences
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
                currentProfile = "You are an adaptive AI. Mirror the user's speech patterns and remember these facts:\n- " + newTraits.joined(separator: "\n- ")
            } else {
                currentProfile += "\n- " + newTraits.joined(separator: "\n- ")
            }
            
            // Limit the persona block so it doesn't consume the entire context window over months of chatting
            // Increased from 15 to 50 for more aggressive retention
            let lines = currentProfile.components(separatedBy: "\n")
            if lines.count > 50 {
                currentProfile = ([lines[0]] + lines.suffix(49)).joined(separator: "\n")
            }
            
            modelPersonalities[modelName] = currentProfile
            savePersonalities()
        }
    }
}
