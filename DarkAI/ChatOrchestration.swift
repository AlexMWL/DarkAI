import Foundation

/// The single place that builds a personality-and-directive-layered system prompt for a chat
/// turn. Extracted out of `ContentView.systemPromptWithPersonality` so `WebPortalManager` can
/// drive the exact same generation pipeline as the native chat UI does, rather than a second,
/// independently-maintained copy of this logic that could quietly drift from it.
enum ChatOrchestration {
    /// Layers the personality matrix and measured writing style onto a base system prompt.
    ///
    /// Shared by every path that calls `llmManager.generateResponse` — the ordinary chat turn,
    /// the file-upload auto-description, and the Web Portal's own send handler — so none of them
    /// resets the assistant's voice to a bare, unadapted default for that one reply.
    @MainActor
    static func systemPromptWithPersonality(
        base: String,
        llmManager: LLMManager,
        personalityManager: PersonalityManager,
        feedbackManager: FeedbackManager
    ) -> String {
        guard llmManager.activeModelURL != nil else { return base }

        // Avoid-directives go on unconditionally (once there are any) regardless of the
        // personality-maturity branching below — they're an explicit correction the user asked
        // for, not a style preference that should wait for the profile to "mature" first.
        let avoidDirectives = feedbackManager.getFormattedAvoidDirectivesForContext()
        let withAvoidDirectives: (String) -> String = { text in
            avoidDirectives.isEmpty ? text : text + "\n\n" + avoidDirectives
        }

        let personality = personalityManager.getPersonality()
        guard !personality.isEmpty else { return withAvoidDirectives(base) }

        let score = personalityManager.maturityScore
        if score < 0.4 {
            return withAvoidDirectives(base + "\n\n[Communication Style Note — adapt naturally to user's style]:\n" + personality)
        } else if score < 0.7 {
            return withAvoidDirectives(base + "\n\n" + personality)
        } else {
            let identityAnchor = base
                .components(separatedBy: CharacterSet(charactersIn: ".!?\n"))
                .first?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return withAvoidDirectives(identityAnchor.isEmpty ? personality : identityAnchor + ".\n\n" + personality)
        }
    }
}
