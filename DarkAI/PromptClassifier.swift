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
    ///
    /// Deliberately does *not* include a bare "generate " catch-all — see `generateVisualNouns`'s
    /// doc comment for why that was a bug, not a feature. Every entry here needs a trailing space
    /// (or to already be a whole word/phrase) so it can only ever match at an actual word
    /// boundary — a fix for the same bug class "draw me"/"paint me" used to have without one:
    /// "draw me" (no boundary) matched the start of "draw meaningful conclusions from this data".
    private static let strongTriggers: [String] = [
        "/imagine", "/image", "/img", "/gen", "/draw",
        "generate an image", "generate a image", "generate image of",
        "create an image", "create a image", "create an illustration",
        "make an image", "make a image", "make a picture of",
        "draw me ", "draw a ", "draw an ", "draw the ",
        "paint me ", "paint a ", "paint an ", "paint the ",
        "render a ", "render an ", "render the ",
        "show me a picture of", "show me an image of",
        "create a photo of", "generate a photo of",
        "illustrate ", "visualize ", "visualise ",
        "generate art", "create art of", "make art of",
        "generate a portrait", "create a portrait",
        "produce an image", "produce a picture",
    ]

    /// Visual nouns that make a bare "generate ..." request unambiguously about an image. This
    /// gates the check added in `classify` for that prefix specifically, replacing what used to
    /// be an unconditional `"generate "` entry in `strongTriggers` above.
    ///
    /// That entry treated *every* "generate X" as an image request unless `X` matched something
    /// on `exclusionPrefixes` — a blacklist that has to enumerate every non-image use of the word
    /// "generate" in English to be complete, and silently misroutes whatever isn't yet on it:
    /// "generate a password", "generate a name for my cat", "generate a random number" all landed
    /// on the diffusion model, mangled by `stripLeadingTrigger`, instead of the LLM. Requiring a
    /// visual noun to actually be present inverts the burden of proof — "generate ..." is text
    /// unless it's actually asking for one of these — which is right for a word this generic,
    /// and needs no exhaustive enumeration of what it *isn't*.
    private static let generateVisualNouns: [String] = [
        "image", "picture", "photo", "photograph", "art", "artwork", "illustration",
        "drawing", "painting", "portrait", "sketch", "wallpaper", "logo", "icon",
        "avatar", "scene", "landscape", "concept art", "render", "graphic"
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

        // 2. Bare "generate " — handled separately from the strong triggers below because it
        // needs the visual-noun gate; see `generateVisualNouns`'s doc comment.
        if lower.hasPrefix("generate "), generateVisualNouns.contains(where: { lower.contains($0) }) {
            let refined = stripLeadingTrigger("generate ", from: trimmed)
            return .imageGeneration(refinedPrompt: refined.isEmpty ? trimmed : refined)
        }

        // 3. Strong triggers (highest confidence).
        //
        // Two passes, and the split matters for what the diffusion model ends up being asked for.
        //
        // Leading triggers are stripped, because "generate an image of a red barn" is a request
        // wrapped around a subject and only the subject should reach the sampler. Longest match
        // first, so the most specific phrasing present wins — e.g. "generate a photo of" over a
        // shorter, less specific entry that happens to also be a prefix of the message.
        //
        // A trigger found *mid-sentence* is left alone entirely. It is part of a description
        // there, not a wrapper around one, and the old code cut it out wherever it appeared:
        // "a mural that could illustrate a children's book" came out as "a mural that could a
        // children's book". Classifying the message as an image request is still right; butchering
        // it on the way to the sampler is not.
        for trigger in strongTriggersLongestFirst where lower.hasPrefix(trigger) {
            let refined = stripLeadingTrigger(trigger, from: trimmed)
            return .imageGeneration(refinedPrompt: refined.isEmpty ? trimmed : refined)
        }
        for trigger in strongTriggersLongestFirst where lower.contains(trigger) {
            return .imageGeneration(refinedPrompt: trimmed)
        }

        // 4. Pattern triggers (medium confidence)
        for trigger in patternTriggers {
            if lower.contains(trigger) {
                return .imageGeneration(refinedPrompt: trimmed)
            }
        }

        // 5. Style keyword triggers (lower confidence — only for short descriptive prompts)
        if trimmed.count < 250 {
            for trigger in styleTriggers {
                if lower.contains(trigger) {
                    return .imageGeneration(refinedPrompt: trimmed)
                }
            }
        }

        return .text
    }

    /// `strongTriggers` ordered longest-first, so the most specific phrasing in a message is the
    /// one that gets matched and stripped. Computed once.
    private static let strongTriggersLongestFirst: [String] =
        strongTriggers.sorted { $0.count > $1.count }

    /// Leftover framing to peel off after the trigger itself is gone. "generate an image of a
    /// dragon" leaves "an image of a dragon"; the subject is "a dragon", and the rest is
    /// instruction that would otherwise be encoded as if it were part of the picture.
    private static let residualFraming: [String] = [
        "me an image of ", "me a picture of ", "me a photo of ", "me an ", "me a ", "me ",
        "an image of ", "a image of ", "image of ",
        "a picture of ", "picture of ",
        "a photo of ", "photo of ", "a photograph of ", "photograph of ",
        "an illustration of ", "illustration of ",
        "a drawing of ", "drawing of ",
        "a painting of ", "painting of ",
        "for me ", "of "
    ]

    // MARK: Prompt Refinement

    /// Strips a leading trigger, plus any framing left behind, so the diffusion model receives a
    /// subject description rather than the instruction that wrapped it.
    ///
    /// e.g. "generate a sunset over the ocean" → "a sunset over the ocean"
    ///      "generate an image of a dragon"    → "a dragon"
    ///      "draw me a dragon"                 → "a dragon"
    ///
    /// Only ever called with a trigger that is genuinely a prefix — see `classify`.
    private static func stripLeadingTrigger(_ trigger: String, from original: String) -> String {
        // Removal by prefix length on `original` itself, never via a range taken from a lowercased
        // copy. Swift string indices carry UTF-8 offsets and lowercasing can change byte length —
        // "İ" (U+0130, 2 bytes) lowercases to "i̇" (3 bytes) — so a range derived from the
        // lowercased string could run past the end of `original` and trap with "String index range
        // is out of bounds", an uncatchable crash on a path that runs on every message sent.
        // Counting characters off the front is immune to that.
        var cleaned = String(original.dropFirst(trigger.count))
            .trimmingCharacters(in: .whitespacesAndNewlines)

        // Peel repeatedly: "draw me an image of a dragon" needs two passes.
        var didStrip = true
        while didStrip {
            didStrip = false
            if cleaned.hasPrefix(":") {
                cleaned = String(cleaned.dropFirst()).trimmingCharacters(in: .whitespaces)
                didStrip = true
            }
            let lower = cleaned.lowercased()
            for framing in residualFraming where lower.hasPrefix(framing) {
                cleaned = String(cleaned.dropFirst(framing.count))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                didStrip = true
                break
            }
        }
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
