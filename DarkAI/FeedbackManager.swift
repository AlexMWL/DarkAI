import Foundation
import Combine

/// A user's up/down rating on one assistant reply, plus (for a down-vote) their own explanation
/// of what was wrong with it. `userPrompt`/`assistantResponse` are captured at rating time and
/// capped in length — they exist so `FeedbackManager` can synthesize an "avoid" directive later
/// without having to reach back into `ConversationManager`'s history (which may have moved on, or
/// belong to a conversation the user has since deleted).
struct ResponseFeedback: Codable, Identifiable, Equatable {
    let id: UUID
    let conversationId: UUID
    let messageId: UUID
    let userPrompt: String
    let assistantResponse: String
    let rating: Rating
    let reason: String?
    let date: Date

    enum Rating: String, Codable {
        case up, down
    }
}

/// Lets the user mark an assistant reply as appropriate (up) or not (down), and — for down-votes
/// only — turns the accumulated feedback into a short, standing "avoid this" directive injected
/// into every future system prompt, the same way `PersonalityManager` already injects a learned
/// writing-style profile.
///
/// This is *not* model fine-tuning. Nothing in this app trains or modifies model weights — GGUF
/// models run through llama.cpp as fixed, pre-trained weights, and Core ML models are compiled,
/// static graphs. "Learning" here means in-context steering: a background LLM pass summarizes
/// what went wrong across recent down-votes into plain-text directives, which get prepended to
/// the system prompt on every subsequent turn. It's the same mechanism `PersonalityManager`
/// already uses for style, just aimed at suppression instead of imitation.
///
/// Up-votes are recorded and shown in Settings for the user's own reference, but deliberately do
/// not feed anything back into generation — `analyzeAssistantMessage` already runs on every
/// assistant reply regardless of rating, so a separate positive-reinforcement path here would
/// double-count without a clear way to weight it correctly.
@MainActor
final class FeedbackManager: ObservableObject {
    @Published private(set) var feedback: [ResponseFeedback] = []
    @Published private(set) var avoidDirectives: String = ""

    private let storageKey = "response_feedback_v1"
    private let directivesKey = "response_feedback_avoid_directives_v1"

    /// Same order-of-magnitude cap as `MemoryManager`/`PersonalityManager`'s stored history —
    /// large enough that a user reviewing Settings never hits it in practice, small enough that
    /// `UserDefaults` never has to hold an unbounded blob of past feedback.
    private let maxStoredFeedback = 300

    /// How many recent down-votes get fed into one synthesis pass. Bounded independently of
    /// `maxStoredFeedback` — the directive synthesis prompt below sends full response text for
    /// each entry, so this caps that prompt's size regardless of how much feedback history exists.
    private let maxDirectiveInputs = 12

    /// Guards `regenerateAvoidDirectives`'s background `Task` from overlapping itself — rapid
    /// down-votes could otherwise fire several synthesis calls at once, competing with each other
    /// and with the user's own foreground generation for the same model instance.
    private var isRegeneratingDirectives = false

    init() {
        load()
    }

    func rating(for messageId: UUID) -> ResponseFeedback.Rating? {
        feedback.first { $0.messageId == messageId }?.rating
    }

    func rateUp(conversationId: UUID, messageId: UUID, userPrompt: String, assistantResponse: String) {
        setRating(conversationId: conversationId, messageId: messageId, userPrompt: userPrompt,
                  assistantResponse: assistantResponse, rating: .up, reason: nil, llmManager: nil)
    }

    /// `reason` is optional — the caller offers the user a text field but doesn't require it.
    /// `llmManager` is optional too: passing `nil` (e.g. no model loaded) just skips the directive
    /// resynthesis for this vote rather than failing the vote itself.
    func rateDown(conversationId: UUID, messageId: UUID, userPrompt: String, assistantResponse: String,
                  reason: String?, llmManager: LLMManager?) {
        setRating(conversationId: conversationId, messageId: messageId, userPrompt: userPrompt,
                  assistantResponse: assistantResponse, rating: .down, reason: reason, llmManager: llmManager)
    }

    /// Removes any rating on this message — lets a rate button act as a toggle (tap the same
    /// rating again to clear it) rather than only ever being able to switch or add one.
    func clearRating(for messageId: UUID) {
        feedback.removeAll { $0.messageId == messageId }
        save()
    }

    func delete(_ id: UUID) {
        let wasDown = feedback.first(where: { $0.id == id })?.rating == .down
        feedback.removeAll { $0.id == id }
        save()
        // A deleted down-vote might be the entry that shaped the current directive text — rather
        // than leave a stale directive nothing in `feedback` still supports, or re-run a background
        // LLM pass just to remove one entry's influence, clear it here; the next down-vote (if any)
        // regenerates it from whatever remains.
        if wasDown {
            avoidDirectives = ""
            UserDefaults.standard.removeObject(forKey: directivesKey)
        }
    }

    func clearAll() {
        feedback.removeAll()
        avoidDirectives = ""
        UserDefaults.standard.removeObject(forKey: directivesKey)
        save()
    }

    /// Empty when there's nothing to add — callers should treat that as "omit this block
    /// entirely" rather than injecting an empty-but-labeled section.
    func getFormattedAvoidDirectivesForContext() -> String {
        guard !avoidDirectives.isEmpty else { return "" }
        return "[Response feedback — the user down-voted replies shaped like these; do not repeat these patterns]\n" + avoidDirectives
    }

    private func setRating(conversationId: UUID, messageId: UUID, userPrompt: String, assistantResponse: String,
                            rating: ResponseFeedback.Rating, reason: String?, llmManager: LLMManager?) {
        feedback.removeAll { $0.messageId == messageId }
        let entry = ResponseFeedback(
            id: UUID(),
            conversationId: conversationId,
            messageId: messageId,
            userPrompt: String(userPrompt.prefix(500)),
            assistantResponse: String(assistantResponse.prefix(1500)),
            rating: rating,
            reason: reason?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false ? reason : nil,
            date: Date()
        )
        feedback.append(entry)
        if feedback.count > maxStoredFeedback {
            feedback.removeFirst(feedback.count - maxStoredFeedback)
        }
        save()
        if rating == .down, let llmManager {
            regenerateAvoidDirectives(llmManager: llmManager)
        }
    }

    /// Mirrors `PersonalityManager.analyzeUserMessage`'s background style-analysis pass: same
    /// `generateBackgroundAnalysis` call, same markdown/heading/bullet stripping on the result,
    /// since that cleanup has nothing to do with what's being analyzed.
    private func regenerateAvoidDirectives(llmManager: LLMManager) {
        let downVotes = feedback.filter { $0.rating == .down }.suffix(maxDirectiveInputs)
        guard !downVotes.isEmpty, !isRegeneratingDirectives else { return }
        isRegeneratingDirectives = true

        let examples = downVotes.map { entry -> String in
            var block = "User asked: \(entry.userPrompt)\nAssistant replied: \(entry.assistantResponse)"
            if let reason = entry.reason {
                block += "\nUser said this was wrong because: \(reason)"
            }
            return block
        }.joined(separator: "\n\n---\n\n")

        Task {
            let analysisPrompt = """
            The user down-voted each of the following assistant replies as inappropriate or unwanted. \
            For each one, work out what specifically made it unwanted, then summarize the underlying \
            pattern to avoid going forward as a short list of plain-text directives — one imperative \
            instruction per line, no markdown, no numbering, no bullet characters. Generalize the \
            behavior to avoid ("avoid unsolicited safety disclaimers on creative writing requests") \
            rather than restating the specific topic of any one example.

            \(examples)
            """

            guard let rawAnalysis = await llmManager.generateBackgroundAnalysis(prompt: analysisPrompt) else {
                await MainActor.run { self.isRegeneratingDirectives = false }
                return
            }

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
            let result = cleanedLines.joined(separator: "\n")

            await MainActor.run {
                defer { self.isRegeneratingDirectives = false }
                guard !result.isEmpty else { return }
                self.avoidDirectives = result
                UserDefaults.standard.set(result, forKey: self.directivesKey)
            }
        }
    }

    private func load() {
        feedback = loadOrLog(key: storageKey, itemDescription: "FeedbackManager: response feedback") ?? []
        avoidDirectives = UserDefaults.standard.string(forKey: directivesKey) ?? ""
    }

    private func save() {
        do {
            let encoded = try JSONEncoder().encode(feedback)
            UserDefaults.standard.set(encoded, forKey: storageKey)
        } catch {
            LogManager.shared.log("FeedbackManager: failed to encode \(feedback.count) feedback entries for save — \(error.localizedDescription)")
        }
    }
}
