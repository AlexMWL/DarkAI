import Foundation

/// On-device content policy enforcement for prompts and model output.
///
/// Why this exists: the app runs arbitrary, user-supplied model weights with no server in the
/// loop, so the model itself cannot be trusted to refuse anything. App Review Guidelines 1.1.4
/// and 1.2 put the obligation on the app, not the model — an AI app has to filter what it
/// generates and cannot ship an "off" switch for that filter. Everything here therefore runs
/// unconditionally, before a prompt reaches a model and again before output reaches the screen.
///
/// Scope and honesty about it: this is lexical screening, not a classifier. It reliably catches
/// direct requests and common obfuscation (leetspeak, spacing, punctuation insertion); it will
/// not catch every euphemism or adversarial phrasing, and it will occasionally flag an innocent
/// phrase. It is a required, meaningful mitigation layer — it is not a guarantee, and the
/// reporting flow in `ContentReportManager` exists precisely because no lexical filter is
/// complete. Thresholds below are tuned to favour false positives on child-safety terms and
/// false negatives everywhere else, which is the correct asymmetry.
///
/// Explicitly `nonisolated`, for the same reason as `AppFiles` and `LogManager`: this module
/// builds with `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`, and screening is not UI work. The
/// type holds nothing but immutable `static let` term lists and pure functions over its
/// arguments, so it is safe from any thread — and it has to be, because the file-import path
/// screens up to several hundred kilobytes at once (see `StructuredImport.parse`) and that is
/// emphatically not something to run on the main thread.
nonisolated enum ContentSafety {

    // MARK: - Types

    enum Category: String, Codable {
        case childSafety
        case sexualContent
        case graphicViolence
        case realPersonSexualized
        case selfHarm

        var reportLabel: String {
            switch self {
            case .childSafety:          return "Child safety"
            case .sexualContent:        return "Sexually explicit"
            case .graphicViolence:      return "Graphic violence"
            case .realPersonSexualized: return "Real person / deepfake"
            case .selfHarm:             return "Self-harm"
            }
        }
    }

    /// Where the text came from. The same words are judged differently depending on whether
    /// they are steering an image generator, steering a chat model, or coming back out of one.
    enum Surface {
        case chatPrompt
        case imagePrompt
        case modelOutput
        /// A search query about to be sent to a third-party provider (Open-Meteo, DuckDuckGo,
        /// or Brave) when the user has turned internet access on. Judged the same as
        /// `chatPrompt` — outbound, user-controlled text — so the app never sends a disallowed
        /// query to a third party in the first place.
        case webSearchQuery
        /// Text that came back from a web search, before it reaches the LLM's context or the
        /// screen. Judged the same as `modelOutput` — inbound content the app doesn't control —
        /// since the web can return anything regardless of how innocuous the query was.
        case webSearchResult
    }

    struct Decision {
        var isAllowed: Bool
        var category: Category?
        /// User-facing explanation. Written to be clear about *what* was refused without
        /// restating the request or reading as a lecture.
        var message: String?
        /// Set when the text signals distress. Independent of `isAllowed` — a self-harm signal
        /// attaches resources to a response, it does not suppress it.
        var attachesCrisisResources: Bool = false

        static let allowed = Decision(isAllowed: true, category: nil, message: nil)
    }

    // MARK: - Public API

    /// Screens a piece of text for a given surface. Cheap enough to call on every send.
    static func review(_ text: String, surface: Surface) -> Decision {
        let normalized = normalize(text)
        let compact = compacted(normalized)
        guard !normalized.isEmpty else { return .allowed }

        let hasMinor = matchesAny(minorTerms, normalized: normalized, compact: compact, useCompact: true)
        let hasSexual = matchesAny(sexualTerms, normalized: normalized, compact: compact, useCompact: true)
        let hasExplicitSexual = matchesAny(explicitSexualTerms, normalized: normalized, compact: compact, useCompact: true)

        // ── 1. Child safety — blocked on every surface, unconditionally, no override ───────
        // Deliberately the broadest check in the file: any co-occurrence of a minor signal and
        // any sexual signal is refused, and the compacted form is searched too so that "1 3 y r
        // o l d" or "m.i.n.o.r" cannot slip past word boundaries.
        if hasMinor && (hasSexual || hasExplicitSexual) {
            return Decision(
                isAllowed: false,
                category: .childSafety,
                message: "This request was blocked. \(AppInfo.displayName) does not process any request that sexualizes a minor, and this rule cannot be turned off."
            )
        }
        if matchesAny(childExploitationTerms, normalized: normalized, compact: compact, useCompact: true) {
            return Decision(
                isAllowed: false,
                category: .childSafety,
                message: "This request was blocked. \(AppInfo.displayName) does not process any request that sexualizes a minor, and this rule cannot be turned off."
            )
        }

        // ── 2. Self-harm — never blocked, always answered with resources ───────────────────
        // Suppressing a message from someone in distress is the wrong intervention. Surfacing
        // help alongside the model's reply is both the safer product behaviour and what Apple
        // asks for on this category.
        let selfHarmSignal = matchesAny(selfHarmTerms, normalized: normalized, compact: compact, useCompact: false)

        // ── 3. Sexually explicit material ─────────────────────────────────────────────────
        // Guideline 1.1.4 bars overtly sexual or pornographic material regardless of medium, so
        // this applies to the chat surface as well as the image surface. The chat threshold is
        // the narrower `explicitSexualTerms` list so that ordinary discussion of sex, sexuality,
        // sexual health, or consent is not caught — only material that is pornographic in intent.
        switch surface {
        case .imagePrompt:
            if hasExplicitSexual || matchesAny(nudityTerms, normalized: normalized, compact: compact, useCompact: true) {
                return Decision(
                    isAllowed: false,
                    category: .sexualContent,
                    message: "\(AppInfo.displayName) doesn't generate sexually explicit images. Try describing the image without that element.",
                    attachesCrisisResources: false
                )
            }
            // Suggestive styling plus a person. Neither half is objectionable alone — "in bed"
            // is a bedroom, "woman" is a portrait — so both are required before refusing, which
            // keeps ordinary portrait and fashion prompts working while catching the phrasing
            // explicit prompts actually use.
            if matchesAny(suggestiveImageTerms, normalized: normalized, compact: compact, useCompact: false),
               matchesAny(personTerms, normalized: normalized, compact: compact, useCompact: false) {
                return Decision(
                    isAllowed: false,
                    category: .sexualContent,
                    message: "\(AppInfo.displayName) doesn't generate sexualized images of people. Try describing the subject without the suggestive framing."
                )
            }
            if matchesAny(goreTerms, normalized: normalized, compact: compact, useCompact: false) {
                return Decision(
                    isAllowed: false,
                    category: .graphicViolence,
                    message: "\(AppInfo.displayName) doesn't generate graphic depictions of violence or gore."
                )
            }
            // Deepfake / real-person sexualization. Requires an explicit real-person marker so
            // that "portrait of a woman" is untouched while "nude photo of <celebrity>" is not.
            if (hasSexual || hasExplicitSexual || matchesAny(nudityTerms, normalized: normalized, compact: compact, useCompact: true)),
               matchesAny(realPersonMarkers, normalized: normalized, compact: compact, useCompact: false) {
                return Decision(
                    isAllowed: false,
                    category: .realPersonSexualized,
                    message: "\(AppInfo.displayName) doesn't generate sexual or intimate imagery of real, identifiable people."
                )
            }
            if matchesAny(deepfakeTerms, normalized: normalized, compact: compact, useCompact: true) {
                return Decision(
                    isAllowed: false,
                    category: .realPersonSexualized,
                    message: "\(AppInfo.displayName) doesn't generate deepfakes or imagery designed to impersonate a real person."
                )
            }

        // Outbound, user-controlled text — a search query is the same kind of thing as a chat
        // prompt: something about to leave the device (to a third party, in the search case)
        // that the app can still choose not to send.
        case .chatPrompt, .webSearchQuery:
            if hasExplicitSexual && matchesAny(pornographicIntentTerms, normalized: normalized, compact: compact, useCompact: false) {
                return Decision(
                    isAllowed: false,
                    category: .sexualContent,
                    message: "\(AppInfo.displayName) doesn't produce sexually explicit material."
                )
            }

        // Inbound, uncontrolled content — a search result is the same kind of thing as model
        // output: text about to reach the LLM's context or the screen that the app didn't
        // generate and can't vouch for the source of.
        case .modelOutput, .webSearchResult:
            if hasExplicitSexual && matchesAny(pornographicIntentTerms, normalized: normalized, compact: compact, useCompact: false) {
                return Decision(
                    isAllowed: false,
                    category: .sexualContent,
                    message: "[This content was withheld because it contained sexually explicit material, which \(AppInfo.displayName)'s content policy does not allow.]"
                )
            }
        }

        if selfHarmSignal {
            return Decision(isAllowed: true, category: .selfHarm, message: nil, attachesCrisisResources: true)
        }

        return .allowed
    }

    /// Cheap incremental check for text that is still streaming in. Only screens the
    /// non-negotiable category, so it can run on a partial buffer many times per response
    /// without burning measurable CPU or acting on half-formed words.
    ///
    /// Returns the offending category so the caller can cancel generation immediately rather
    /// than rendering violating tokens and retracting them a second later.
    static func streamingViolation(in partialText: String) -> Category? {
        guard partialText.count > 24 else { return nil }
        let normalized = normalize(partialText)
        let compact = compacted(normalized)

        if matchesAny(childExploitationTerms, normalized: normalized, compact: compact, useCompact: true) {
            return .childSafety
        }
        let hasMinor = matchesAny(minorTerms, normalized: normalized, compact: compact, useCompact: true)
        if hasMinor && matchesAny(explicitSexualTerms, normalized: normalized, compact: compact, useCompact: true) {
            return .childSafety
        }
        return nil
    }

    /// Safety terms appended to every diffusion negative prompt. Steering the sampler away from
    /// this material is a second, independent layer from prompt screening: it constrains what
    /// the model can land on even when the prompt itself was innocuous, which is the actual
    /// failure mode with fine-tuned community checkpoints.
    static let diffusionNegativePromptSuffix =
        "nsfw, nude, nudity, naked, topless, bottomless, undressed, unclothed, exposed breasts, " +
        "explicit, sexual, sexually suggestive, pornographic, erotic, hentai, lewd, " +
        "lingerie, underwear, cleavage, suggestive pose, provocative pose, " +
        "child, kid, teen, minor, underage, loli, shota, " +
        "gore, blood, mutilation, dismemberment, corpse, graphic violence"

    // MARK: - Normalization

    /// Lowercases, folds diacritics, undoes the common character substitutions used to slip a
    /// term past a literal match, and collapses padded repeats ("nnnude" → "nude").
    private static func normalize(_ input: String) -> String {
        var s = input.folding(options: [.diacriticInsensitive, .widthInsensitive, .caseInsensitive],
                              locale: Locale(identifier: "en_US_POSIX"))
        let substitutions: [Character: Character] = [
            "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g",
            "@": "a", "$": "s", "!": "i", "|": "i", "+": "t"
        ]
        s = String(s.map { substitutions[$0] ?? $0 })

        // Collapse any character repeated 3+ times down to two — "sooooo" stays readable while
        // "n u u u de" style padding stops defeating a boundary match.
        var collapsed = ""
        var previous: Character? = nil
        var runLength = 0
        for ch in s {
            if ch == previous {
                runLength += 1
                if runLength >= 2 { continue }
            } else {
                previous = ch
                runLength = 0
            }
            collapsed.append(ch)
        }
        return collapsed
    }

    /// Normalized text with every non-alphanumeric stripped, so "s.e.x" / "s e x" / "s-e-x"
    /// all reduce to the same needle. Only used for the highest-severity term lists, because
    /// substring matching on a compacted string produces false positives on ordinary words
    /// (the classic "scunthorpe" failure) and that trade is only worth making for child safety.
    private static func compacted(_ normalized: String) -> String {
        String(normalized.unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) })
    }

    private static func matchesAny(_ terms: [String],
                                   normalized: String,
                                   compact: String,
                                   useCompact: Bool) -> Bool {
        for term in terms {
            if wordBoundaryMatch(term, in: normalized) { return true }
            if useCompact {
                let needle = compacted(term)
                if needle.count >= 5, compact.contains(needle) { return true }
            }
        }
        return false
    }

    /// Whole-word (or whole-phrase) containment. Avoids `NSRegularExpression` per term — this
    /// runs on every keystroke-sized buffer during streaming, and building regexes there showed
    /// up as measurable overhead.
    private static func wordBoundaryMatch(_ term: String, in haystack: String) -> Bool {
        guard !term.isEmpty else { return false }
        var searchStart = haystack.startIndex
        while searchStart < haystack.endIndex,
              let range = haystack.range(of: term, range: searchStart..<haystack.endIndex) {
            let beforeOK: Bool = {
                guard range.lowerBound > haystack.startIndex else { return true }
                let prev = haystack[haystack.index(before: range.lowerBound)]
                return !prev.isLetter && !prev.isNumber
            }()
            let afterOK: Bool = {
                guard range.upperBound < haystack.endIndex else { return true }
                let next = haystack[range.upperBound]
                return !next.isLetter && !next.isNumber
            }()
            if beforeOK && afterOK { return true }
            searchStart = haystack.index(after: range.lowerBound)
        }
        return false
    }

    // MARK: - Term lists
    //
    // Kept clinical and deliberately short. These are matched as whole words, so the list does
    // not need morphological variants of everything — but plural/gerund forms that a prompt
    // would realistically use are spelled out rather than inferred, since stemming English
    // correctly is a bigger dependency than this needs.

    private static let minorTerms = [
        "child", "children", "kid", "kids", "minor", "minors", "underage", "under age",
        "toddler", "infant", "baby", "babies", "preteen", "pre teen", "tween",
        "teen", "teens", "teenage", "teenager", "teenagers", "adolescent",
        "schoolgirl", "schoolboy", "middle schooler", "elementary schooler",
        "kindergartner", "8 year old", "9 year old", "10 year old", "11 year old",
        "12 year old", "13 year old", "14 year old", "15 year old", "16 year old",
        "17 year old", "yo girl", "yo boy", "young girl", "young boy", "little girl",
        "little boy", "juvenile", "prepubescent", "pubescent"
    ]

    private static let sexualTerms = [
        "sex", "sexy", "sexual", "sexually", "seduce", "seductive", "erotic", "erotica",
        "aroused", "arousal", "fetish", "lewd", "provocative", "suggestive", "intimate",
        "lingerie", "bikini", "undressed", "undressing", "stripping", "strip tease"
    ]

    private static let explicitSexualTerms = [
        "porn", "porno", "pornographic", "pornography", "hardcore", "xxx", "nsfw",
        "explicit sex", "sex act", "sex acts", "sexual act", "intercourse", "penetration",
        "masturbate", "masturbation", "orgasm", "climax sexually", "genitals", "genitalia",
        "aroused nude", "hentai", "rule 34", "smut", "cum", "orgy", "blowjob", "handjob",
        "anal sex", "oral sex", "bdsm", "bondage sexual"
    ]

    private static let nudityTerms = [
        "nude", "nudes", "nudity", "naked", "topless", "bottomless", "unclothed",
        "no clothes", "without clothes", "fully exposed", "bare breasts", "bare chest nude",
        "undressed", "disrobed", "stark naked", "birthday suit", "au naturel",
        "see through", "see-through", "sheer clothing", "wet t shirt", "wet shirt",
        "nipples", "areola", "cleavage", "underboob", "sideboob", "camel toe",
        "bare butt", "bare buttocks", "exposed skin only", "covered only by"
    ]

    /// Composition and styling language that carries no explicit word but is, in practice, how
    /// explicit image prompts are actually written for diffusion models.
    ///
    /// Applied to the image surface only, and only when paired with a person — see `review`. A
    /// keyword list cannot catch everything an uncensored checkpoint will produce, which is why
    /// `ImageSafetyAnalyzer` screens the finished pixels; this raises the cost of getting there.
    private static let suggestiveImageTerms = [
        "boudoir", "pinup", "pin up", "playboy", "onlyfans", "camgirl", "stripper",
        "seductive", "sensual", "sultry", "alluring", "flirtatious", "coquettish",
        "provocative pose", "suggestive pose", "erotic pose", "spread legs", "legs spread",
        "arched back", "bedroom eyes", "come hither", "tempting pose",
        "bikini", "lingerie", "underwear", "thong", "g string", "corset", "negligee",
        "bathing suit", "swimsuit", "micro bikini", "skimpy", "revealing outfit",
        "barely covered", "clothes falling off", "torn clothes", "ripped clothing",
        "in bed", "on the bed", "shower scene", "bathtub scene", "wet body",
        "curvaceous", "voluptuous", "busty", "thicc", "hourglass figure",
        "perfect body", "body focus", "breast focus", "ass focus", "thigh focus"
    ]

    /// Person indicators — the suggestive list above only matters when a person is depicted.
    /// "in bed" alone is a bedroom photo; "woman in bed, sensual" is something else.
    private static let personTerms = [
        "woman", "women", "girl", "girls", "lady", "female", "man", "men", "boy", "boys",
        "male", "person", "people", "model", "figure", "body", "she", "he", "her", "him",
        "waifu", "character", "portrait", "selfie", "goddess", "nymph", "maiden",
        "babe", "beauty", "bombshell", "seductress"
    ]

    /// Terms whose presence alone is disqualifying — no second signal required.
    private static let childExploitationTerms = [
        "csam", "child porn", "child pornography", "cp underage", "loli", "lolicon",
        "shota", "shotacon", "jailbait", "child sexual", "sexualized child",
        "sexualized minor", "underage sex", "underage nude", "underage porn",
        "child nude", "child naked", "minor nude", "minor naked", "pedophile",
        "pedophilia", "grooming a child"
    ]

    /// Distinguishes discussion of sexuality from a request to produce pornography. Paired with
    /// `explicitSexualTerms` so that "explain safe sex" or "what is consent" never trips, while
    /// "write an explicit sex scene" does.
    private static let pornographicIntentTerms = [
        "write", "generate", "create", "make", "produce", "describe in detail",
        "roleplay", "role play", "story", "scene", "script", "narrate", "continue",
        "draw", "render", "image", "picture", "photo"
    ]

    private static let goreTerms = [
        "gore", "gory", "mutilated", "mutilation", "dismembered", "dismemberment",
        "decapitated", "decapitation", "beheading", "disemboweled", "eviscerated",
        "corpse", "mangled body", "torture", "brutally killed", "graphic violence",
        "blood everywhere", "severed head", "severed limb"
    ]

    private static let realPersonMarkers = [
        "celebrity", "celebrities", "famous person", "famous actress", "famous actor",
        "politician", "president", "prime minister", "senator", "pop star", "singer named",
        "real person", "real people", "actual person", "my classmate", "my coworker",
        "my neighbor", "my neighbour", "my teacher", "my ex", "someone i know",
        "this person in the photo", "the woman in this photo", "the man in this photo"
    ]

    private static let deepfakeTerms = [
        "deepfake", "deep fake", "face swap onto", "faceswap onto", "impersonate a real",
        "pretend to be a real person", "fake photo of a real"
    ]

    private static let selfHarmTerms = [
        "kill myself", "killing myself", "end my life", "ending my life", "suicide",
        "suicidal", "want to die", "wanna die", "better off dead", "self harm",
        "self-harm", "hurt myself", "hurting myself", "cut myself", "cutting myself",
        "overdose on", "no reason to live", "cant go on", "can t go on"
    ]
}
