import Foundation
import Combine

class MemoryManager: ObservableObject {
    @Published var memories: [String] = []

    private let storageKey = "DarkAI_Memories"

    init() {
        loadMemories()
    }

    func loadMemories() {
        if let stored = UserDefaults.standard.stringArray(forKey: storageKey) {
            self.memories = stored
        }
    }

    func saveMemories() {
        UserDefaults.standard.set(memories, forKey: storageKey)
    }

    func addMemory(_ memory: String) {
        let trimmed = memory.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed.count > 5 else { return }
        if !memories.contains(where: { $0.caseInsensitiveCompare(trimmed) == .orderedSame }) {
            memories.append(trimmed)
            saveMemories()
        }
    }

    func removeMemory(at index: Int) {
        guard index >= 0 && index < memories.count else { return }
        memories.remove(at: index)
        saveMemories()
    }

    func clearAllMemories() {
        memories.removeAll()
        saveMemories()
    }

    // MARK: - Broad Passive Memory Extraction
    // Extracts preferences, likes, dislikes, and general user info from any message.
    func extractMemories(from userMessage: String) {
        let message = userMessage.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = message.lowercased()

        // — Identity —
        matchPrefix(lower, message, prefix: "my name is ", format: { "User's name is \($0)." })
        matchPrefix(lower, message, prefix: "i am called ", format: { "User's name is \($0)." })
        matchPrefix(lower, message, prefix: "call me ", format: { "User goes by \($0)." })
        matchPrefix(lower, message, prefix: "i am allergic to ", format: { "User is allergic to \($0)." })
        matchPrefix(lower, message, prefix: "i'm allergic to ", format: { "User is allergic to \($0)." })
        matchPrefix(lower, message, prefix: "i am interested in ", format: { "User is interested in \($0)." })
        matchPrefix(lower, message, prefix: "i'm interested in ", format: { "User is interested in \($0)." })
        matchPrefix(lower, message, prefix: "i work as ", format: { "User works as \($0)." })
        matchPrefix(lower, message, prefix: "i am a ", format: { "User is a \($0)." })
        matchPrefix(lower, message, prefix: "i'm a ", format: { "User is a \($0)." })
        matchPrefix(lower, message, prefix: "i live in ", format: { "User lives in \($0)." })
        matchPrefix(lower, message, prefix: "i'm from ", format: { "User is from \($0)." })
        matchPrefix(lower, message, prefix: "i am from ", format: { "User is from \($0)." })
        matchPrefix(lower, message, prefix: "i'm based in ", format: { "User is based in \($0)." })
        
        // — Family & Pets —
        matchPrefix(lower, message, prefix: "my wife is ", format: { "User's wife is \($0)." })
        matchPrefix(lower, message, prefix: "my husband is ", format: { "User's husband is \($0)." })
        matchPrefix(lower, message, prefix: "my partner is ", format: { "User's partner is \($0)." })
        matchPrefix(lower, message, prefix: "my girlfriend is ", format: { "User's girlfriend is \($0)." })
        matchPrefix(lower, message, prefix: "my boyfriend is ", format: { "User's boyfriend is \($0)." })
        matchPrefix(lower, message, prefix: "my son is ", format: { "User's son is \($0)." })
        matchPrefix(lower, message, prefix: "my daughter is ", format: { "User's daughter is \($0)." })
        matchPrefix(lower, message, prefix: "my dog is ", format: { "User's dog is \($0)." })
        matchPrefix(lower, message, prefix: "my cat is ", format: { "User's cat is \($0)." })
        matchPrefix(lower, message, prefix: "my pet is ", format: { "User's pet is \($0)." })

        // — Ownership & Tech —
        matchPrefix(lower, message, prefix: "i have a ", format: { "User has a \($0)." })
        matchPrefix(lower, message, prefix: "i have an ", format: { "User has an \($0)." })
        matchPrefix(lower, message, prefix: "i own a ", format: { "User owns a \($0)." })
        matchPrefix(lower, message, prefix: "my car is a ", format: { "User's car is a \($0)." })
        matchPrefix(lower, message, prefix: "my computer is a ", format: { "User's computer is a \($0)." })
        matchPrefix(lower, message, prefix: "my phone is a ", format: { "User's phone is a \($0)." })

        // — Skills & Education —
        matchPrefix(lower, message, prefix: "i know how to ", format: { "User knows how to \($0)." })
        matchPrefix(lower, message, prefix: "i am learning ", format: { "User is learning \($0)." })
        matchPrefix(lower, message, prefix: "i'm learning ", format: { "User is learning \($0)." })
        matchPrefix(lower, message, prefix: "i am studying ", format: { "User is studying \($0)." })
        matchPrefix(lower, message, prefix: "i'm studying ", format: { "User is studying \($0)." })
        matchPrefix(lower, message, prefix: "my job is ", format: { "User's job is \($0)." })
        
        // — Health & Traits —
        matchPrefix(lower, message, prefix: "i am allergic to ", format: { "User is allergic to \($0)." })
        matchPrefix(lower, message, prefix: "i'm allergic to ", format: { "User is allergic to \($0)." })

        // — Likes / Dislikes —
        matchPrefix(lower, message, prefix: "i love ", format: { "User loves \($0)." })
        matchPrefix(lower, message, prefix: "i like ", format: { "User likes \($0)." })
        matchPrefix(lower, message, prefix: "i enjoy ", format: { "User enjoys \($0)." })
        matchPrefix(lower, message, prefix: "i prefer ", format: { "User prefers \($0)." })
        matchPrefix(lower, message, prefix: "i use ", format: { "User uses \($0)." })
        matchPrefix(lower, message, prefix: "i hate ", format: { "User hates \($0)." })
        matchPrefix(lower, message, prefix: "i don't like ", format: { "User dislikes \($0)." })
        matchPrefix(lower, message, prefix: "i do not like ", format: { "User dislikes \($0)." })
        matchPrefix(lower, message, prefix: "i dislike ", format: { "User dislikes \($0)." })
        matchPrefix(lower, message, prefix: "i want ", format: { val in val.count < 80 ? "User wants \(val)." : nil })
        matchPrefix(lower, message, prefix: "i need ", format: { val in val.count < 80 ? "User needs \(val)." : nil })
        matchPrefix(lower, message, prefix: "i always ", format: { val in val.count < 80 ? "User always \(val)." : nil })
        matchPrefix(lower, message, prefix: "i usually ", format: { val in val.count < 80 ? "User usually \(val)." : nil })
        matchPrefix(lower, message, prefix: "i never ", format: { val in val.count < 80 ? "User never \(val)." : nil })

        // — Plans & To-Dos —
        matchPrefix(lower, message, prefix: "i plan to ", format: { "User plans to \($0)." })
        matchPrefix(lower, message, prefix: "i'm planning to ", format: { "User is planning to \($0)." })
        matchPrefix(lower, message, prefix: "i am planning to ", format: { "User is planning to \($0)." })
        matchPrefix(lower, message, prefix: "i want to ", format: { "User wants to \($0)." })
        matchPrefix(lower, message, prefix: "i'm going to ", format: { "User is going to \($0)." })
        matchPrefix(lower, message, prefix: "i am going to ", format: { "User is going to \($0)." })
        matchPrefix(lower, message, prefix: "i have to ", format: { "User has to \($0)." })
        matchPrefix(lower, message, prefix: "i need to ", format: { "User needs to \($0)." })
        matchPrefix(lower, message, prefix: "i must ", format: { "User must \($0)." })
        matchPrefix(lower, message, prefix: "i've got to ", format: { "User has got to \($0)." })
        matchPrefix(lower, message, prefix: "i should ", format: { "User should \($0)." })
        
        // — Interests & Hobbies —
        matchPrefix(lower, message, prefix: "i'm interested in ", format: { "User is interested in \($0)." })
        matchPrefix(lower, message, prefix: "i am interested in ", format: { "User is interested in \($0)." })
        matchPrefix(lower, message, prefix: "i'm fascinated by ", format: { "User is fascinated by \($0)." })
        matchPrefix(lower, message, prefix: "i am fascinated by ", format: { "User is fascinated by \($0)." })
        matchPrefix(lower, message, prefix: "my hobby is ", format: { "User's hobby is \($0)." })
        matchPrefix(lower, message, prefix: "i collect ", format: { "User collects \($0)." })
        
        // — Past & History —
        matchPrefix(lower, message, prefix: "i went to ", format: { "User went to \($0)." })
        matchPrefix(lower, message, prefix: "i used to ", format: { "User used to \($0)." })
        matchPrefix(lower, message, prefix: "i grew up in ", format: { "User grew up in \($0)." })
        matchPrefix(lower, message, prefix: "i studied at ", format: { "User studied at \($0)." })
        matchPrefix(lower, message, prefix: "i graduated from ", format: { "User graduated from \($0)." })

        // — Favorite X is Y pattern —
        if let range = lower.range(of: "my favorite ") {
            let rest = message[range.upperBound...]
            if let isRange = rest.lowercased().range(of: " is ") {
                let subject = rest[..<isRange.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
                let value = rest[isRange.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines)
                if !subject.isEmpty && !value.isEmpty && value.count < 100 {
                    addMemory("User's favorite \(subject) is \(value).")
                }
            }
        }

        // — Opinions / Preferences —
        matchContains(lower, message, contains: "i think ", format: { val in val.count < 100 ? "User thinks \(val)." : nil })
        matchContains(lower, message, contains: "i believe ", format: { val in val.count < 100 ? "User believes \(val)." : nil })
    }

    // Helper — matches if the lowercased message starts with `prefix`, extracts the remainder, and formats it.
    private func matchPrefix(_ lower: String, _ original: String, prefix: String, format: (String) -> String?) {
        guard lower.hasPrefix(prefix) else { return }
        let remainder = String(original.dropFirst(prefix.count)).trimmingCharacters(in: .whitespacesAndNewlines)
        // Strip trailing periods/exclamations
        let cleaned = remainder.trimmingCharacters(in: CharacterSet(charactersIn: ".!?,;"))
        guard !cleaned.isEmpty, cleaned.count < 120 else { return }
        if let formatted = format(cleaned) {
            addMemory(formatted)
        }
    }

    // Helper — matches if the lowercased message contains `contains`, extracts what follows.
    private func matchContains(_ lower: String, _ original: String, contains pattern: String, format: (String) -> String?) {
        guard let range = lower.range(of: pattern) else { return }
        let remainder = String(original[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        let cleaned = remainder.trimmingCharacters(in: CharacterSet(charactersIn: ".!?,;"))
        guard !cleaned.isEmpty, cleaned.count < 120 else { return }
        if let formatted = format(cleaned) {
            addMemory(formatted)
        }
    }

    func getFormattedMemoriesForContext() -> String {
        guard !memories.isEmpty else { return "" }
        var result = "### Long-Term Conversational Memories:\n(CRITICAL INSTRUCTION: The following is background context about the user. Do NOT mention or list these memories in your response unless the user explicitly asks 'what are your memories?'. Use them silently to inform your answers.)\n"
        for memory in memories {
            result += "- \(memory)\n"
        }
        return result
    }
}
