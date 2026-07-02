//
//  MemoryManager.swift
//  DarkAI
//

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
        matchPrefix(lower, message, prefix: "i am ", format: { "User is \($0)." })
        matchPrefix(lower, message, prefix: "i'm ", format: { "User is \($0)." })
        matchPrefix(lower, message, prefix: "i work as ", format: { "User works as \($0)." })
        matchPrefix(lower, message, prefix: "i am a ", format: { "User is a \($0)." })
        matchPrefix(lower, message, prefix: "i'm a ", format: { "User is a \($0)." })
        matchPrefix(lower, message, prefix: "i live in ", format: { "User lives in \($0)." })
        matchPrefix(lower, message, prefix: "i'm from ", format: { "User is from \($0)." })
        matchPrefix(lower, message, prefix: "i am from ", format: { "User is from \($0)." })
        matchPrefix(lower, message, prefix: "i'm based in ", format: { "User is based in \($0)." })
        matchPrefix(lower, message, prefix: "i am ", format: { val in val.count < 60 ? "User is \(val)." : nil })

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
