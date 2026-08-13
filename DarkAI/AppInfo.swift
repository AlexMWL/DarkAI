import Foundation

/// Single source of truth for the identity, support channel, and legal text that App Review
/// requires an AI-content app to surface in-app (Guidelines 1.2, 5.1.1, 5.1.2).
///
/// ⚠️ SUBMISSION CHECKLIST — every value marked `REQUIRED` must be real before you upload a
/// build. App Review will tap these. A placeholder here is a rejection.
///
/// `nonisolated` because the app's name and version are needed by code that has no business
/// hopping to the main actor to read a constant: `ContentSafety` puts the app name in every
/// refusal message, `ConversationExport` stamps it into exported transcripts, and both run off
/// the UI thread. Everything here is either a `let` or a computed read of `Bundle.main`, which is
/// thread-safe.
nonisolated enum AppInfo {

    // MARK: - Identity

    static let displayName = "DarkAI"

    /// Bumped whenever `LegalText.acceptableUse`, `LegalText.termsOfUse`, or the privacy policy
    /// changes materially — version 2 added on-device crash reporting and the opt-in diagnostic
    /// log attachment; version 3 added the opt-in Internet Access feature (`WebSearchManager`),
    /// which is the app's first code path that can send anything — a search query, and only a
    /// search query — off the device. Both are disclosures a user who agreed to an earlier
    /// version hasn't seen.
    /// Users who accepted an older version are re-prompted on next launch — Apple requires an
    /// affirmative agreement to the *current* terms for UGC/AI apps.
    static let policyVersion = 3

    // MARK: - Support & reporting (REQUIRED)

    /// Receives content reports and crash reports filed from inside the app. Apple checks that
    /// this is monitored and that reports get a response — an unattended address is a 1.2
    /// violation.
    static let supportEmail = "lostsoccerball@icloud.com"

    /// Shown on the App Store listing and in Settings → About. A plain mailto: is acceptable
    /// if you have no support site; leave `nil` to hide the row rather than link somewhere dead.
    static let supportURL: URL? = nil

    /// Apple's standard EULA covers you if you leave this `nil` — the app then displays the
    /// bundled `LegalText.termsOfUse` instead. Set it only if you host your own custom EULA.
    static let termsURL: URL? = nil

    /// Optional hosted copy of the privacy policy. The full text is bundled in-app either way,
    /// but App Store Connect requires a reachable URL in the listing metadata.
    static let privacyPolicyURL: URL? = nil

    /// Apple's standard Licensed Application End User License Agreement.
    static let appleStandardEULA = URL(string: "https://www.apple.com/legal/internet-services/itunes/dev/stdeula/")!

    // MARK: - Derived

    static var version: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0"
    }

    static var build: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "1"
    }

    static var versionString: String { "Version \(version) (\(build))" }

    /// `true` when every REQUIRED field above has been filled in with something plausible.
    /// `OnboardingView` surfaces a debug-only banner when this is false so a placeholder build
    /// can't quietly reach App Review.
    static var isSubmissionReady: Bool {
        supportEmail.contains("@") && !supportEmail.hasPrefix("your-")
    }
}

// MARK: - Legal & policy text

/// Bundled offline copies of everything the user has to be able to read without a network
/// connection. The app is fully offline by design, so linking out to a website for the terms
/// would leave an airplane-mode user unable to read what they agreed to.
///
/// `nonisolated` for the same reason as `AppInfo` above: these are immutable strings, and one of
/// them (`crisisResources`) is attached to model output by a screening path that does not run on
/// the main actor.
nonisolated enum LegalText {

    static let acceptableUse = """
    ACCEPTABLE USE POLICY

    \(AppInfo.displayName) runs AI language and image models entirely on your device. You control \
    which model runs, and the output is produced locally by that model. Because of that, you are \
    responsible for what you ask for and what you do with the result.

    ZERO TOLERANCE FOR OBJECTIONABLE CONTENT

    There is no tolerance for objectionable content or abusive use of this app. You may not use \
    \(AppInfo.displayName) to request, generate, store, or share:

    • Any sexual content involving minors, or any content that sexualizes a person under 18. This \
      is prohibited absolutely and is enforced by an on-device filter that cannot be turned off.
    • Pornographic or sexually explicit imagery.
    • Photorealistic depictions of real, identifiable people in sexual, defamatory, or deceptive \
      contexts, including deepfakes.
    • Content that promotes, encourages, or provides instructions for self-harm, suicide, or \
      eating disorders.
    • Graphic depictions of violence or gore produced for shock value.
    • Content that harasses, defames, or incites violence or hatred against a person or group.
    • Instructions for producing weapons, explosives, or illegal drugs, or for committing crimes.
    • Content that infringes another person's copyright, trademark, or right of publicity.

    ENFORCEMENT

    Prompts are screened on your device before they are sent to a model, and model output is \
    screened before it is shown. If Internet Access is on, the same filter also screens a search \
    query before it is sent out, and screens whatever comes back before it reaches the model or \
    your screen. Requests that violate the child-safety rules above are blocked outright and \
    cannot be overridden by any setting, custom instruction, or imported model.

    REPORTING

    If the app produces content you believe violates this policy, use "Report a concern" from the \
    message menu, or from Settings → Safety & Legal. Reports reach \(AppInfo.supportEmail) and are \
    reviewed within 24 hours. Confirmed policy failures are addressed in the next app update.

    IMPORTED MODELS

    You may import your own model weights. Imported models remain subject to this policy and to \
    the on-device filter. Importing a model that was fine-tuned to produce prohibited content does \
    not exempt you from these rules.

    AGE REQUIREMENT

    You must be 17 or older to use this app.
    """

    static let privacyPolicy = """
    PRIVACY POLICY

    Last updated: with app version \(AppInfo.version)

    SUMMARY: \(AppInfo.displayName) does not sell any personal data, has no account or login, \
    and by default sends nothing off your device. The one exception is Internet Access \
    (Settings → Internet Access), which is off by default: if you turn it on, the app will ask, \
    before every individual search, whether to look something up online — it never decides that \
    on its own. Only if you say yes is that search's text sent out. Everything else — your \
    conversations, everything the app generates, and any search you decline — stays on your \
    device.

    WHAT STAYS ON YOUR DEVICE

    • Conversations, including prompts and model responses.
    • Long-term memories and learned style preferences extracted from your messages.
    • Documents and images you attach, and the text extracted from them.
    • Images generated by the app.
    • Model files you download or import.
    • Diagnostic logs.
    • Crash reports (see below).

    All of the above is stored in this app's private container. It is excluded from iCloud backup \
    to avoid consuming your iCloud storage with multi-gigabyte model files. Deleting the app \
    deletes all of it permanently.

    CRASH REPORTS

    \(AppInfo.displayName) detects when it closes unexpectedly and writes a crash report to this \
    device. A crash report contains only technical information: the crash type and signal, a stack \
    trace of function names, the app version, your iOS version and device model, and how much \
    memory was free at the time. It does not contain your prompts, conversations, memories, \
    documents, or generated images.

    Crash reports are never transmitted automatically. Nothing is sent unless you choose to send \
    one, and when you do, your own mail app opens with the report filled in so you can read, edit, \
    or discard it before sending. You can view, delete, or disable crash reports entirely in \
    Settings → Safety & Legal.

    DIAGNOSTIC LOGS

    The app keeps a rolling local log of technical events — model loads, memory decisions, \
    download results, and whether the content filter blocked something (the category only, never \
    the text). You can attach this log to a report if it would help us diagnose a problem, but \
    that is always your choice and it is off by default. You can read the full log before sending \
    it, and clear it at any time, from Settings → Diagnostics.

    WHAT LEAVES YOUR DEVICE

    • Nothing, unless you initiate it.
    • If you download a model from the built-in catalog, your device contacts that model's host \
      (Hugging Face, or Civitai for some diffusion models) to fetch the file. That request is \
      subject to the host's own privacy policy. No prompt, conversation, or personal data is \
      sent — only a standard file download.
    • If you file a content report or send a crash report, your mail app opens with a pre-filled \
      message. Nothing is sent until you tap send, and you can edit or delete any part of it first.
    • If you use the share or export controls, the content goes wherever you send it.
    • INTERNET ACCESS (off by default). Turned on in Settings → Internet Access. When on, the \
      app will still never search on its own — before every individual search it asks you first, \
      in the conversation, and only proceeds if you say yes. If you say yes, the text of that \
      one search (and nothing else — not your conversation history, not your device, not an \
      account identifier, because there is no account) is sent to Open-Meteo for weather, or to \
      DuckDuckGo for general lookups. If you've added your own Brave Search API key in Settings, \
      general lookups go to Brave instead, directly from your device using your own key — that \
      traffic is between you and Brave, under Brave's privacy policy, the same way a model \
      download is between you and its host. Turning Internet Access off, or simply declining an \
      individual search offer, means nothing about that message ever leaves your device.

    NO ANALYTICS

    The app contains no analytics SDK, no advertising SDK, no third-party crash reporting service, \
    and no tracking of any kind. Crash detection is implemented on device and reports go nowhere \
    unless you send them yourself. There is no account and no login.

    ON-DEVICE PROCESSING

    Text generation, image generation, optical character recognition, and content-safety screening \
    all run locally using Apple frameworks and on-device model weights. No prompt is ever \
    transmitted for processing — the model that writes your response always runs on your device, \
    never in the cloud. The only thing that can leave your device at all is a search query, and \
    only when Internet Access is on and you've said yes to that specific search; see WHAT LEAVES \
    YOUR DEVICE above. A search result that comes back is used only as reference material for the \
    still-local model's response — it is not a substitute for on-device generation.

    CHILDREN

    This app is rated 17+ and is not directed to children. It does not knowingly collect any \
    information from anyone, including children.

    YOUR CONTROL

    Clear conversations, memories, learned personality data, and diagnostic logs at any time from \
    Settings. Deleting the app removes everything.

    CONTACT

    \(AppInfo.supportEmail)
    """

    static let termsOfUse = """
    TERMS OF USE

    By using \(AppInfo.displayName) you agree to these terms and to the Acceptable Use Policy.

    1. LICENSE. Use of this app is governed by Apple's Standard Licensed Application End User \
    License Agreement, available at:
    \(AppInfo.appleStandardEULA.absoluteString)

    2. AI-GENERATED CONTENT. This app runs machine-learning models on your device. Model output is \
    generated automatically and may be inaccurate, incomplete, offensive, or fabricated. It is not \
    reviewed by a human before you see it. Do not rely on it as professional advice of any kind — \
    medical, legal, financial, or otherwise. Verify anything that matters.

    3. YOUR RESPONSIBILITY. You are responsible for the prompts you write, for any model you \
    import, and for how you use the output. You agree not to use the app in violation of the \
    Acceptable Use Policy or of any law.

    4. IMPORTED AND DOWNLOADED MODELS. Model files are provided by third parties under their own \
    licenses. The app does not modify or redistribute them. You are responsible for complying with \
    the license of any model you download or import. The developer makes no warranty about any \
    third-party model's behavior, accuracy, or safety.

    5. NO WARRANTY. The app is provided "as is", without warranty of any kind, to the fullest \
    extent permitted by law.

    6. TERMINATION. Access may be terminated for violation of the Acceptable Use Policy.

    7. CONTACT. \(AppInfo.supportEmail)
    """

    /// Shown next to any response that trips the self-harm signal. Deliberately not a block —
    /// blocking someone in crisis is worse than answering them with resources attached.
    static let crisisResources = """
    If you're going through something, you don't have to handle it alone.

    • US & Canada: call or text 988 (Suicide & Crisis Lifeline)
    • UK & Ireland: call 116 123 (Samaritans)
    • Australia: call 13 11 14 (Lifeline)
    • Elsewhere: findahelpline.com

    If someone is in immediate danger, call your local emergency number.
    """
}
