# App Store Submission Notes

Everything below is either done in code or is something **you** have to do in App Store Connect /
the Developer portal. Work through the checklist before you upload.

---

## 1. What changed in the code

### Removed — automatic rejections

| What it was | Where | Why it had to go |
|---|---|---|
| `"SIDELOAD STATUS: SYSTEM BYPASS ACTIVE"` | Settings footer | Advertised circumventing App Store distribution |
| `"Bypass active"`, `"System bypass active"`, `"DarkAI Local OS bypass"` | Chat banner, seeded welcome message, drawer footer | Same |
| **"Sideloading Guide" RAG document** — a seeded doc walking the user through Developer Mode, AltStore, Sideloadly, and **TrollStore** | `RAGManager.loadDocuments()` | Instructions for installing apps outside the App Store. This alone would have ended the review. Existing installs that already persisted it are cleaned up on upgrade. |
| `"Force Sideload"` button that loaded a model past the OOM failsafe | Settings memory dialog | A button whose documented outcome is the app being killed (Guideline 2.1). Models in the `.dangerous` band are now refused outright; the `.warning` band still offers "Load Anyway". |
| `CameraPicker` + `ImagePicker` (both dead code) | `ImagePicker.swift`, deleted | `UIImagePickerController(sourceType: .camera)` with no `NSCameraUsageDescription`. Unreachable, but referencing camera APIs without a purpose string risks rejection at upload. |
| `"[MATURE: OVERRIDE ACTIVE]"` badge | Settings personality section | It meant "has learned enough", but reads to a reviewer as an adult-content unlock. Now `[ADAPTED]`. |
| `"Chaos Mode"` | Settings | Renamed to **High Variability**, with a line explaining what it actually does. |
| `"Stealth Mode"` | Chat input | Renamed to **Private chat**. See the bug fix in §3. |
| `"LOCAL RUNNER v5.7.7"`, `"iPhone 17 Pro Max RAM:"` | Empty state, memory dialog | A fake version string and a hardcoded device name that was wrong on every other phone. |

### Added — required for an AI-content app

**`ContentSafety.swift` — on-device content policy.** Screens prompts before they reach a model
and output before it reaches the screen. Always on, no setting disables it.

- **Child safety** — any co-occurrence of a minor signal and a sexual signal is blocked on every
  surface, plus a standalone term list that blocks on its own. Matched against both the normalized
  text and a compacted form (punctuation/spacing stripped, leetspeak folded) so `s.e.x` and
  `1 3 y r o l d` don't slip through.
- **Sexually explicit** — blocked for image generation; blocked for chat when paired with a
  generation verb, so discussing sex, sexual health, or consent still works.
- **Graphic violence / gore** — blocked for image generation.
- **Deepfakes / real-person sexualization** — blocked for image generation.
- **Self-harm** — deliberately *not* blocked. Trips a signal that attaches crisis resources
  alongside the model's normal reply.

Four enforcement points: the typed prompt, the LLM-expanded image prompt (an imported chat model
can embellish an innocuous request into something that would never have passed), the streaming
response (cancelled mid-generation, throttled to every ~240 characters so it costs nothing per
token), and the completed response. Plus a fifth, independent layer: safety terms appended to
every diffusion negative prompt, which constrains the sampler even when the prompt was clean —
the actual failure mode with community fine-tunes.

**Be straight with yourself about what this is:** lexical screening, not a classifier. It catches
direct requests and common obfuscation. It will miss novel euphemisms and it will occasionally
flag something innocent. That is why the reporting flow below exists, and why blocked prompts are
handed back to the user rather than deleted.

**`SafetyCenter.swift` — reporting (Guideline 1.2).** "Report a Concern" on every generated
message (long-press, plus a visible button on images). Filing a report removes the content from
the conversation immediately, logs it locally so the user can see it was taken, and opens a
pre-filled mail composer to your support address — with a `mailto:` fallback and a copy-details
option for devices with no Mail account. Reports are listed in Settings → Safety & Legal.

**`OnboardingView.swift` — first-run gate.** Affirmative, un-prechecked acceptance of the
acceptable-use policy and terms, plus a 17+ age confirmation, before anything can be generated.
Re-shown automatically when `AppInfo.policyVersion` increases.

**`ModelCatalog.swift` — the fix for Guideline 2.1.** This was the biggest risk: a reviewer used
to open the app to "Model unloaded" and a disabled text field, which reads as a broken app. There
are now three curated models downloadable in-app, on a background `URLSession` so the transfer
survives leaving the app:

**Chat models:**

| Model | Size | License |
|---|---|---|
| Llama 3.2 1B Instruct (Q4_K_M) | 0.75 GB | Llama 3.2 Community License — attribution shown in-app |
| Qwen2.5 1.5B Instruct (Q4_K_M) | 0.92 GB | Apache 2.0 |
| SmolLM2 1.7B Instruct (Q4_K_M) | 0.98 GB | Apache 2.0 |
| Llama 3.2 3B Instruct (Q4_K_M) | 1.88 GB | Llama 3.2 Community License — attribution shown in-app |

**Diffusion models** (Settings → Diffusion Model):

| Model | Size | License |
|---|---|---|
| Stable Diffusion 1.5 (Q4_K) | 2.96 GB | CreativeML Open RAIL-M |
| Juggernaut XL v9 (Q4_K) | 2.76 GB | CreativeML Open RAIL++-M |

Both RAIL licenses carry use restrictions that prohibit generating illegal or harmful content;
that condition is surfaced in the catalog row rather than buried. Diffusion downloads land in
`DiffusionModels/` and chat weights in `Models/`, and the memory warning fires against each
model's *runtime* peak rather than its file size — the SDXL checkpoint needs roughly 5.5 GB
resident, so it warns off anything under 8 GB.

All six URLs and byte sizes were verified against Hugging Face while writing this. Downloads
are cellular-gated (off by default), disk-space pre-flighted, and verified after the fact by exact
byte count **and** GGUF magic bytes — which catches the captive-portal case where an HTML error
page lands on disk under a `.gguf` name.

There is deliberately **no "paste a URL" field**. An arbitrary-URL downloader reads as a way to
ship content around review, and the app can't check the license of an arbitrary URL. Users who
want a specific model still have the document picker and the Files folder.

**`PrivacyInfo.xcprivacy`.** Mandatory since May 2024 — a build without one is rejected at upload.
Declares `NSPrivacyTracking = false`, no collected data types, and required-reason usage for
UserDefaults (CA92.1), disk space (E174.1, 85F4.1), and file timestamps (C617.1).

**`AppFiles.swift` — storage.** Every directory the app writes to now gets `isExcludedFromBackup`
and explicit file protection. Multi-gigabyte model weights sitting in `Documents/` were being
swept into the user's iCloud backup, which the iOS Data Storage Guidelines call out specifically
and which is a common rejection. Settings now shows total storage used.

**`AppInfo.swift`** — support address, policy version, and the full offline text of the acceptable
use policy, privacy policy, and terms. Bundled rather than linked, because the app is offline by
design and an airplane-mode user should still be able to read what they agreed to.

**`CrashReporter.swift` — on-device crash detection.** No third-party SDK: every crash service is
a network dependency that phones home, which would break the privacy claim and force a "Data
Collected" entry on the privacy label. Two detection paths, because this app has two ways of dying:

- **Signals and uncaught exceptions.** `SIGABRT/SEGV/BUS/ILL/FPE/TRAP` plus
  `NSSetUncaughtExceptionHandler`. The signal handler is written to be async-signal-safe — the
  path is resolved to a C string at startup, and the report is written with `open(2)`/`write(2)`
  and `backtrace_symbols_fd`, with no Swift allocation in the handler. It re-raises with the
  default disposition afterwards so the crash still reaches iOS's own reporting.
- **Unclean termination with no signal**, inferred on the next launch from a clean-shutdown flag.
  This is what a jetsam out-of-memory kill looks like, and for an app that loads multi-gigabyte
  weights it is *the* dominant failure mode — the kernel gives you no chance to run a handler.
  Checkpoints recorded before each model load and image generation mean the report can say what
  the app was doing and how much memory was left, which is the actually actionable part.

Reports are shown in full before anything is sent, never transmitted automatically, and the whole
feature has an off switch in Settings → Safety & Legal. The privacy policy documents it explicitly
and `policyVersion` was bumped to 2, so existing users re-consent on next launch.

**Diagnostic log attachment.** Opt-in, off by default, on both the content-report and crash-report
flows, with a "review what would be sent" screen. The log deliberately records filter *categories*
and never the text that triggered them — worth keeping that boundary on purpose, since this is the
one thing in the app that can travel off-device.

### Build settings

- `INFOPLIST_KEY_ITSAppUsesNonExemptEncryption = NO` — without it every upload stalls waiting on
  an export-compliance answer.
- Photo library usage string rewritten to say when it's asked for and that existing photos are
  never read.
- **Deployment target lowered to iOS 17.0** (from 17.6 on the target and 18.0 at the project
  level). `build_sd_ios.sh` was pinned to iOS **26.6**, so the static library had a *newer*
  minimum than the app linking it — the framework has been rebuilt at 17.0 and verified with
  `otool -l` (`minos 17.0`). Keep those two numbers in sync if you change either. Three
  `if #available(iOS 16.0, *)` blocks around `ShareLink` were unconditionally true and have been
  collapsed. `llama.swift` declares `.iOS(.v16)`, so it's unaffected.

### Context window now follows the device

`contextTokenLimit` used to be a free-floating number up to 32768 that was silently clamped at
load time — on a 4 GB phone you could set 32k and get a quarter of it with no indication. There's
now a `deviceContextCeiling` derived from physical RAM (4k / 8k / 16k / 32k across the 2–3, 4, 6,
and 8 GB+ tiers), the slider is bounded by it, and the stored value is clamped on launch and again
after each model load — at which point the ceiling becomes model-aware via `safeContextLimit`.

It **only ever lowers**. A user who deliberately picked something smaller keeps it; the point is
to stop an impossible setting from surviving, not to push everyone to the maximum. When it does
adjust, Settings says so and gives the reason rather than leaving an unexplained number.

---

## 2. Bugs fixed along the way

**Photos saving was silently broken.** The context-menu "Save to Photos" called
`UIImageWriteToSavedPhotosAlbum` with no authorization check at all — on a device that had never
been asked, it did nothing: no prompt, no error, no image. Now goes through an add-only
authorization request with real success/failure feedback and a route to Settings when denied.

**"Stealth Mode" discarded unrelated data.** It short-circuited `saveConversations()` globally, so
while it was on, *every* pending change was dropped — deleting a different chat or renaming one
appeared to work and then came back on next launch. It's now per-conversation: a private
conversation is excluded from disk, everything else saves normally, and turning it on mid-chat
erases the already-persisted half.

**Alert modifier collision.** The auto-load prompt used the deprecated `Alert`-returning API;
mixing that with the iOS 15+ alerts added for safety notices lets one silently win. Converted.

---

## 3. What you have to do

### Before you upload

- [ ] **Verify the support mailbox is monitored.** `AppInfo.supportEmail` is
      `lostsoccerball@icloud.com`. It appears in the acceptable-use policy, the privacy policy,
      every content report, and every crash report. Apple will test that reports reach someone,
      and the policy text promises a 24-hour review window.
- [ ] **Enable the Increased Memory Limit capability** on the App ID in the Developer portal —
      the entitlement is already in `DarkAI.entitlements`, and the build will fail to install
      without the matching capability.
- [ ] **Add an App Privacy Policy URL** in App Store Connect. It's a required metadata field even
      though the full text is bundled in-app. If you host it, also set `AppInfo.privacyPolicyURL`.

### In App Store Connect

- [ ] **Age rating: 17+.** Answer the questionnaire truthfully — an app that generates open-ended
      text and images from arbitrary weights is 17+, and under-rating it is its own violation.
- [ ] **Privacy nutrition label: no longer "Data Not Collected."** The opt-in Internet Access
      feature (`WebSearchManager`, off by default) means the app now has one declared data type —
      see `PrivacyInfo.xcprivacy`. In App Store Connect's questionnaire, declare **Search History**:
      not linked to identity, not used for tracking, purpose **App Functionality** only. Don't
      leave the old "Data Not Collected" answer in place — a mismatch between the questionnaire
      and the bundled privacy manifest is exactly the kind of thing App Review checks for.
- [ ] **Reviewer notes.** Paste something like:

  > DarkAI runs open-source AI models entirely on device. No account, no server-side data
  > collection. There is one opt-in exception: Settings → Internet Access (off by default) lets
  > the model search the web for things it can't know (e.g. current weather) — it always asks
  > before searching, never on its own, and only the search text is sent, to Open-Meteo,
  > DuckDuckGo, or a user-supplied Brave Search key. See LegalText.privacyPolicy in-app for the
  > full disclosure.
  >
  > To test: on first launch, accept the policies, then tap "Get" on Llama 3.2 1B Instruct
  > (0.75 GB, ~1 min on Wi-Fi). When it finishes, tap "Start Using DarkAI" and type any question.
  > For image generation, go to Settings → Diffusion Model and download Stable Diffusion 1.5
  > (2.96 GB), then type "draw a lighthouse at sunset" in the chat. For internet search, turn on
  > Settings → Internet Access, then ask "what's the weather in Paris" and tap Search when offered.
  >
  > Content safety: all prompts and responses are screened on device by a filter that cannot be
  > disabled, including for user-imported models and for search queries/results. Requests that
  > sexualize minors are blocked unconditionally. Every generated message can be reported via
  > long-press → "Report a Concern", which removes the content and emails us. Policies are in
  > Settings → Safety & Legal.

### Still worth doing

- [ ] **Generated images are only filtered at the prompt and sampler.** iOS 17+ has
      `SensitiveContentAnalysis` (`SCSensitivityAnalyzer`), which does on-device nudity detection
      on the resulting image — a genuinely stronger last line of defense. Not added here because it
      needs the `com.apple.developer.sensitivecontentanalysis.client` entitlement, and adding an
      entitlement your provisioning profile doesn't carry breaks installs. Enable the capability
      first, then wire it in. Note it returns `.disabled` unless the user has Sensitive Content
      Warning turned on, so it supplements the prompt filter rather than replacing it.
- [ ] **Screenshots must not show any of the removed language.** If you have old ones with
      "SYSTEM BYPASS ACTIVE", retake them.
- [ ] **The app name.** "DarkAI" was kept, and it's defensible as a dark-theme aesthetic. Be aware
      that combined with an AI app it can prompt a closer look for "uncensored AI" positioning —
      keep the listing copy focused on privacy and on-device performance, not on the absence of
      restrictions.

---

## 4. Build note

`StableDiffusion.xcframework` is not in the repo (removed in commit `04790ad`) and is gitignored,
so a fresh clone can't build. Regenerate it with `./build_sd_ios.sh` before building. Also note the
script pins `IOS_DEPLOYMENT_TARGET="26.6"` while the app target is `17.6` — worth reconciling.
