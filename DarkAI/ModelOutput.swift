import Foundation

/// Cleans up raw language-model output before anything else sees it.
///
/// This lived as a private method on `ContentView` and had exactly one caller: the chat bubble.
/// It moved here because it now has three — the bubble, the transcript exporter, and the
/// image-prompt writer — and the last of those is why it matters. `LLMManager.generateImagePrompt`
/// asks the chat model to expand a request into a Stable Diffusion prompt and previously fed the
/// model's *raw* output straight to the text encoder. On a reasoning model that output opens with
/// a `<think>` block, and with a 120-token budget the block is often the entire response, so what
/// reached CLIP was the model's private deliberation rather than a description of anything the
/// user asked for. That is a picture of the wrong thing, generated from a prompt the user never
/// wrote and never saw.
///
/// `nonisolated` because the image-prompt path runs off the main actor. Pure functions over their
/// arguments, no state.
nonisolated enum ModelOutput {

    /// Strips reasoning blocks, scaffolding preambles, role echoes, and chat-template artifacts.
    ///
    /// - Parameter stripMarkdown: also flattens markdown emphasis and headings, preserving fenced
    ///   code blocks. Used by the "mature" personality mode, which writes in plain prose.
    static func filterThoughts(from text: String, stripMarkdown: Bool = false) -> String {
        var filtered = text

        // --- 0. Strip format-agnostic reasoning/channel blocks ---
        // Some models (gpt-oss, certain Gemma fine-tunes) use a "channel" convention with
        // no matching close tag at all — e.g. <|channel|>analysis<|message|> ... reasoning ...
        // <|channel|>final<|message|> — where analysis only ends when a new non-reasoning
        // channel begins. The fixed open/close tag pairs in step 1 below can't express that,
        // so this keys off vocabulary inside any bracketed tag (<tag>, </tag>, <|tag|>) and
        // exits on whichever comes first: a real close tag, or a transition to a
        // final/message/response/answer channel.
        let reasoningOpenPattern = "<\\|?\\s*(think|thought|thinking|reflect|reason|channel|analysis|internal|scratchpad|deliberat)"
        let reasoningClosePattern = "</\\s*(think|thought|thinking|reflect|reflection|reason|reasoning|channel|analysis|internal|scratchpad|deliberation)\\s*>"
        let transitionPattern = "<\\|?/?\\s*(final|message|response|answer)[a-z]*\\s*\\|?>"
        if let openRegex = try? NSRegularExpression(pattern: reasoningOpenPattern, options: [.caseInsensitive]),
           let closeRegex = try? NSRegularExpression(pattern: reasoningClosePattern, options: [.caseInsensitive]),
           let transitionRegex = try? NSRegularExpression(pattern: transitionPattern, options: [.caseInsensitive]) {
            while let openMatch = openRegex.firstMatch(in: filtered, range: NSRange(filtered.startIndex..., in: filtered)),
                  let openRange = Range(openMatch.range, in: filtered) {
                let searchRange = NSRange(openRange.upperBound..., in: filtered)
                let closeEnd = closeRegex.firstMatch(in: filtered, range: searchRange).flatMap { Range($0.range, in: filtered)?.upperBound }
                let transEnd = transitionRegex.firstMatch(in: filtered, range: searchRange).flatMap { Range($0.range, in: filtered)?.upperBound }

                let end: String.Index?
                switch (closeEnd, transEnd) {
                case let (c?, t?): end = min(c, t)
                case let (c?, nil): end = c
                case let (nil, t?): end = t
                default: end = nil
                }

                if let end {
                    filtered.removeSubrange(openRange.lowerBound..<end)
                } else {
                    filtered.removeSubrange(openRange.lowerBound..<filtered.endIndex)
                    break
                }
            }
        }

        // --- 1. Strip XML-style thinking/correction/reflection tags ---
        // Covers Gemma, Qwen, DeepSeek, Llama, and other instruct model variants
        let xmlTags = [
            "think", "thinking", "thought", "thoughts",
            "channel thought", "channel thoughts", "channel>thought",
            "|channel>thought", "self-correction", "self_correction",
            "correction", "reflection", "reasoning", "internal"
        ]
        // Longest tag first: "<think" is a prefix of "<thinking", so checking "think" first
        // matched the opening of a `<thinking>` block and then failed to find its `</think>`
        // close, deleting the entire rest of the message.
        for tag in xmlTags.sorted(by: { $0.count > $1.count }) {
            while let startRange = filtered.range(of: "<\(tag)", options: .caseInsensitive) {
                // The close tag is searched for strictly *after* the open tag.
                //
                // Searching the whole string was a crash: a stray "</think>" ahead of the
                // "<think>" (models do emit unbalanced tags) produced an end bound lower than
                // the start bound, and `lowerBound..<upperBound` traps when inverted —
                // "Fatal error: Range requires lowerBound <= upperBound", an uncatchable
                // SIGTRAP on a function that runs over every response and every streamed token.
                if let endRange = filtered.range(of: "</\(tag)>",
                                                 options: .caseInsensitive,
                                                 range: startRange.upperBound..<filtered.endIndex) {
                    filtered.removeSubrange(startRange.lowerBound..<endRange.upperBound)
                } else {
                    filtered.removeSubrange(startRange.lowerBound..<filtered.endIndex)
                    break
                }
            }
        }
        
        // Strip pipe-delimited thinking tokens used by some models
        let pipeTokens = ["<|thinking|>", "<|/thinking|>", "<|thought|>", "<|/thought|>"]
        for token in pipeTokens {
            filtered = filtered.replacingOccurrences(of: token, with: "", options: .caseInsensitive)
        }
        
        // --- 2. Strip plaintext preamble headers ---
        let plaintextHeaders = ["Thinking Process:", "Thought Process:", "Internal Reasoning:", "Chain of Thought:"]
        for header in plaintextHeaders {
            while let startRange = filtered.range(of: header, options: .caseInsensitive) {
                // Same inverted-range crash as the tag loop above, and far easier to hit here:
                // any reply that says "Response:" before "Thinking Process:" — entirely
                // ordinary phrasing — produced an inverted range and trapped. Bound the search
                // to the text after the header so the end can never precede the start.
                if let endRange = filtered.range(of: "Response:",
                                                 options: .caseInsensitive,
                                                 range: startRange.upperBound..<filtered.endIndex) {
                    filtered.removeSubrange(startRange.lowerBound..<endRange.upperBound)
                } else {
                    filtered.removeSubrange(startRange.lowerBound..<filtered.endIndex)
                    break
                }
            }
        }
        
        // --- 3. Strip exact known artifact strings ---
        let exactArtifacts = [
            "<|im_start|>", "<|im_end|>", "<|start_of_turn|>", "<|end_of_turn|>"
        ]
        for artifact in exactArtifacts {
            filtered = filtered.replacingOccurrences(of: artifact, with: "", options: .caseInsensitive)
        }

        // --- 3b. Strip bare (untagged) scaffolding/preamble blocks ---
        // Some fine-tunes open every reply with meta-commentary about how they plan to
        // respond — style/tone "checks," headers like "My internal monologue:", decorative
        // "***" separators, or literal placeholder text like "(Generating response...)" —
        // with no tag structure at all, and wording that varies message to message. Match
        // the recurring *shapes* instead of specific wording: a slash-command header, a
        // decorative "***" line, any bold/italic header ending in a colon, or known
        // self-referential/stage-direction phrases in other wrappers. These have no
        // reliable close tag, so the next blank line is treated as the block's end —
        // scoped to near the start of the message only, so a legitimate bolded header
        // later in a well-formed answer is never touched.
        let bareLabelPattern = "(?:^|\\n)\\s*(?:" +
            "\\*{3,}|" +
            "/[A-Za-z][A-Za-z ]{2,29}:|" +
            "\\*{1,2}[A-Za-z][A-Za-z ,]{2,39}:\\*{0,2}|" +
            "[\\*\"'\\[\\(]{1,2}\\s*(?:self[- _]?correction|self[- _]?review|internal monologue|internal reasoning|response generation|chain of thought|style check|tone check|voice check|persona check|vibe check|character check|generating response|generating\\.\\.\\.)" +
            ")"
        if let bareLabelRegex = try? NSRegularExpression(pattern: bareLabelPattern, options: [.caseInsensitive]) {
            var guardIterations = 0
            while guardIterations < 20,
                  let openMatch = bareLabelRegex.firstMatch(in: filtered, range: NSRange(filtered.startIndex..., in: filtered)),
                  let openRange = Range(openMatch.range, in: filtered),
                  filtered.distance(from: filtered.startIndex, to: openRange.lowerBound) < 600 {
                guardIterations += 1
                if let blankRange = filtered.range(of: "\n\n", range: openRange.upperBound..<filtered.endIndex) {
                    filtered.removeSubrange(openRange.lowerBound..<blankRange.upperBound)
                } else {
                    filtered.removeSubrange(openRange.lowerBound..<filtered.endIndex)
                    break
                }
            }
        }
        
        // --- 4. Strip leading role-echo preamble (model parroting its own role prefix) ---
        let leadingPreambles = ["assistant:", "response:", "answer:", "a:"]
        var didTrimLeading = true
        while didTrimLeading {
            didTrimLeading = false
            let trimmedLower = filtered.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            for preamble in leadingPreambles {
                if trimmedLower.hasPrefix(preamble) {
                    filtered = String(filtered.trimmingCharacters(in: .whitespacesAndNewlines).dropFirst(preamble.count))
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    didTrimLeading = true
                    break
                }
            }
        }
        
        // --- 5. Strip personality system-prompt leak via regex ---
        if let regex = try? NSRegularExpression(
            pattern: "\\(?(?:Critical Instructions?|User Style Matrix|Communication Style Note)[\\s\\S]*?fr\\.?\\)?",
            options: [.caseInsensitive, .dotMatchesLineSeparators]
        ) {
            filtered = regex.stringByReplacingMatches(
                in: filtered,
                options: [],
                range: NSRange(location: 0, length: filtered.utf16.count),
                withTemplate: ""
            ).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        
        // --- 6. Markdown stripping (mature personality mode only, always preserves fenced code blocks) ---
        if stripMarkdown {
            var codeBlocks: [String] = []
            var protected = filtered
            if let codeBlockRegex = try? NSRegularExpression(pattern: "```[\\s\\S]*?```", options: []) {
                let matches = codeBlockRegex.matches(in: protected, range: NSRange(protected.startIndex..., in: protected)).reversed()
                for match in matches {
                    if let range = Range(match.range, in: protected) {
                        let block = String(protected[range])
                        let placeholder = "CODEBLOCK_\(codeBlocks.count)_PLACEHOLDER"
                        codeBlocks.append(block)
                        protected.replaceSubrange(range, with: placeholder)
                    }
                }
            }
            protected = protected.replacingOccurrences(of: "**", with: "")
            protected = protected.replacingOccurrences(of: "__", with: "")
            let mdLines = protected.components(separatedBy: "\n").map { line -> String in
                var l = line
                if l.hasPrefix("### ") { l = String(l.dropFirst(4)) }
                else if l.hasPrefix("## ") { l = String(l.dropFirst(3)) }
                else if l.hasPrefix("# ") { l = String(l.dropFirst(2)) }
                return l
            }
            protected = mdLines.joined(separator: "\n")
            for (i, block) in codeBlocks.enumerated() {
                protected = protected.replacingOccurrences(of: "CODEBLOCK_\(i)_PLACEHOLDER", with: block)
            }
            filtered = protected
        }
        
        // --- 7. Strip trailing role-echo stop tokens ---
        var finalFiltered = filtered.trimmingCharacters(in: .whitespacesAndNewlines)
        let trailingStops = ["user", "user:", "<|im_end|>", "<start_of_turn>user", "<|user|>", "<|eot_id|>"]
        var didTrimTrailing = true
        while didTrimTrailing {
            didTrimTrailing = false
            for stop in trailingStops {
                if finalFiltered.lowercased().hasSuffix(stop) {
                    finalFiltered.removeLast(stop.count)
                    finalFiltered = finalFiltered.trimmingCharacters(in: .whitespacesAndNewlines)
                    didTrimTrailing = true
                }
            }
        }
        
        return finalFiltered
    }
}
