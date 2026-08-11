import Foundation

// MARK: - Prompt Intent

/// The classified intent of a user message — either normal conversation or image generation.
enum PromptIntent {
    case text
    case imageGeneration(refinedPrompt: String)
}

// MARK: - Prompt Classifier

/// Rule-based, zero-latency prompt classifier.
/// No LLM call needed — deterministic keyword matching with ordered priority.
struct PromptClassifier {

    // MARK: Trigger Lists

    /// High-confidence prefixes / phrases — model is almost certainly asking for an image.
    private static let strongTriggers: [String] = [
        // Slash commands
        "/imagine", "/image", "/img", "/gen", "/draw",
        // "generate" prefix — catches any "generate X" request
        "generate ",
        // Explicit image-creation verbs
        "generate an image", "generate a image", "generate image of",
        "create an image", "create a image", "create an illustration",
        "make an image", "make a image", "make a picture of",
        "draw me", "draw a ", "draw an ", "draw the ",
        "paint me", "paint a ", "paint an ", "paint the ",
        "render a ", "render an ", "render the ",
        "show me a picture of", "show me an image of",
        "create a photo of", "generate a photo of",
        "illustrate ", "visualize ", "visualise ",
        "generate art", "create art of", "make art of",
        "generate a portrait", "create a portrait",
        "produce an image", "produce a picture",
    ]

    /// Medium-confidence subject patterns — "X of Y" image request structures.
    private static let patternTriggers: [String] = [
        "image of ", "photo of ", "picture of ",
        "portrait of ", "artwork of ", "scene of ",
        "digital art of ", "painting of ", "illustration of ",
        "photograph of ", "sketch of ", "rendering of ",
        "concept art of ", "anime drawing of ",
    ]

    /// Style/quality descriptors — only fire when the message is short and descriptive.
    private static let styleTriggers: [String] = [
        "photorealistic", "hyperrealistic",
        "8k uhd", "4k resolution", "cinematic lighting",
        "watercolor style", "oil painting style", "anime style",
        "in the style of", "trending on artstation",
        "unreal engine render", "octane render",
        "highly detailed, 4k",
    ]

    /// Exclusion prefixes — override image classification.
    /// Also guards against text-generation requests that start with "generate":
    /// e.g. "generate code", "generate text", "generate a list", "generate a story".
    private static let exclusionPrefixes: [String] = [
        "what is", "what are", "explain", "describe",
        "how does", "how do", "how can",
        "tell me about", "can you explain", "definition of",
        "what's the difference", "compare", "why does", "why is",
        "write a", "write an", "summarize", "summarise",
        "translate", "fix", "debug", "help me",
        // Text-generation guards (intercept "generate X" where X is text-like)
        "generate code", "generate text", "generate a list",
        "generate a story", "generate an essay", "generate a report",
        "generate a script", "generate a poem", "generate a song",
        "generate a table", "generate a summary", "generate a response",
        "generate a plan", "generate ideas", "generate questions",
        "generate a description",
    ]

    // MARK: Classification

    static func classify(_ input: String) -> PromptIntent {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return .text }

        let lower = trimmed.lowercased()

        // 1. Exclusion check — wins over everything else
        for exclusion in exclusionPrefixes {
            if lower.hasPrefix(exclusion) { return .text }
        }

        // 2. Strong triggers (highest confidence)
        for trigger in strongTriggers {
            if lower.hasPrefix(trigger) || lower.contains(trigger) {
                let refined = stripTrigger(trigger, from: trimmed)
                return .imageGeneration(refinedPrompt: refined.isEmpty ? trimmed : refined)
            }
        }

        // 3. Pattern triggers (medium confidence)
        for trigger in patternTriggers {
            if lower.contains(trigger) {
                return .imageGeneration(refinedPrompt: trimmed)
            }
        }

        // 4. Style keyword triggers (lower confidence — only for short descriptive prompts)
        if trimmed.count < 250 {
            for trigger in styleTriggers {
                if lower.contains(trigger) {
                    return .imageGeneration(refinedPrompt: trimmed)
                }
            }
        }

        return .text
    }

    // MARK: Prompt Refinement

    /// Strips the triggering verb/prefix from the prompt so the diffusion model
    /// receives a clean subject description.
    ///
    /// e.g. "generate a sunset over the ocean" → "sunset over the ocean"
    ///      "draw me a dragon" → "a dragon"
    private static func stripTrigger(_ trigger: String, from original: String) -> String {
        // Searched case-insensitively on `original` rather than on a lowercased copy, so the
        // range belongs to the string it's used on.
        //
        // The previous version took the range from `original.lowercased()` and applied it to
        // `original`. Swift string indices carry UTF-8 offsets, and lowercasing can change the
        // byte length — "İ" (U+0130, 2 bytes) lowercases to "i̇" (3 bytes) — so a few of those
        // ahead of the trigger pushed the range past the end of `original` and the removal
        // trapped with "String index range is out of bounds". That is an uncatchable crash on
        // a path that runs on every message the user sends.
        guard let range = original.range(of: trigger, options: [.caseInsensitive]) else {
            return original.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        var cleaned = original
        cleaned.removeSubrange(range)
        cleaned = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
        // Remove leftover "of" or ":" artefacts
        if cleaned.lowercased().hasPrefix("of ") { cleaned = String(cleaned.dropFirst(3)) }
        if cleaned.hasPrefix(":") { cleaned = String(cleaned.dropFirst(1)).trimmingCharacters(in: .whitespaces) }
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
