import Foundation

/// Llama 3.2's instruct chat format, hand-built rather than rendered from a Jinja template — this
/// app has no Jinja engine, and the format itself is fixed and well-documented (Meta's model
/// card): a `<|begin_of_text|>` header, then one `<|start_header_id|>{role}<|end_header_id|>`
/// block per turn ending in `<|eot_id|>`, closed with an empty assistant turn to prompt a reply.
///
/// These are literal special-token strings, not text the model should ever see as plain words —
/// they only resolve to their real token IDs because `CoreMLTokenizer.encode` tokenizes with
/// `parse_special: true` (matching `SingleWindowCoreMLEngine`'s own tokenizer call), which is what
/// lets a bundled vocab-only GGUF recognize them as special tokens by their literal text rather
/// than tokenizing them as ordinary characters.
nonisolated enum Llama3ChatTemplate {
    /// `messages` already carries a `("system", ...)` entry first when `LLMManager` has a system
    /// prompt configured (see `LLMManager.generateResponse`'s `swiftMessages` construction) — this
    /// just renders whatever roles/content it's given, in order, the same way
    /// `SingleWindowCoreMLEngine.generateStream` treats `messages` uniformly rather than special-
    /// casing a system prompt separately.
    static func render(messages: [(role: String, content: String)]) -> String {
        var text = "<|begin_of_text|>"
        for message in messages {
            text += "<|start_header_id|>\(message.role)<|end_header_id|>\n\n\(message.content)<|eot_id|>"
        }
        text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return text
    }
}
