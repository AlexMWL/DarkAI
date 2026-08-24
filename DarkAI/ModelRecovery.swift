import SwiftUI

/// Puts the app's models back into a known-good state without the user quitting and relaunching.
///
/// **Why this exists.** Every heavy failure in this app leaves it in a state that is technically
/// recoverable and practically stuck: a memory warning unloads the chat model and parks
/// `loadState` on `.failed`, an image generation that runs out of headroom evicts the chat model
/// and then can't load the checkpoint either, and a refused teardown can leave gigabytes resident
/// with nothing loaded. In each case the app is one unload-and-reload away from working, and the
/// only control the user had was force-quitting — which is both a bad answer and one that looks
/// like the app crashed.
///
/// The sequence matters and is the reason this is a single shared operation rather than a button
/// that calls `loadModel` again: stopping any in-flight work, tearing *both* engines down, and
/// then waiting for the memory to genuinely come back before reloading. Reloading without that
/// wait is what produces "not enough memory" on a device that has plenty — see
/// `MemoryBudget.waitForRelease`.
@MainActor
enum ModelRecovery {

    /// Progress as the reset moves through its phases, for the caller to show.
    enum Stage: Equatable {
        case stopping
        case unloading
        case waitingForMemory
        case reloading(String)
        case done(String)
        case partial(String)

        var message: String {
            switch self {
            case .stopping:            return "Stopping current work…"
            case .unloading:           return "Unloading models…"
            case .waitingForMemory:    return "Waiting for memory to come back…"
            case .reloading(let name): return "Reloading \(name)…"
            case .done(let summary):   return summary
            case .partial(let reason): return reason
            }
        }

        var isTerminal: Bool {
            switch self {
            case .done, .partial: return true
            default:              return false
            }
        }
    }

    /// Tears everything down and brings the last-used chat model back.
    ///
    /// Deliberately does *not* reload the diffusion checkpoint: it is only ever loaded for the
    /// duration of one generation, and putting several gigabytes back for no pending request is
    /// the opposite of what someone hitting a memory error needs.
    static func resetModels(
        llm: LLMManager,
        diffusion: DiffusionManager,
        onStage: @escaping (Stage) -> Void
    ) async {
        LogManager.shared.log("ModelRecovery: reset requested — \(MemoryBudget.describe())")

        onStage(.stopping)
        if diffusion.isGenerating { diffusion.cancelGeneration() }
        if llm.isGenerating { llm.cancelGeneration() }

        onStage(.unloading)
        var diffusionFreed = await diffusion.unloadDiffusionModelAsync()
        await llm.unloadModelAsync()

        // `.stopping` above already called `cancelGeneration()`, which now requests real
        // cancellation via `SDWrapper.cancel()` (`sd_cancel_generation`) rather than only
        // flipping Swift-side flags — but the C++ sampler only checks for that request between
        // steps, not instantly, so the unload attempt right above can easily race a step that
        // was already underway when cancellation was requested. Give it a few short beats to
        // actually land before conceding, rather than the single immediate attempt this used to
        // make back when there was no real cancellation for a retry to wait on.
        if !diffusionFreed {
            for _ in 0..<8 where !diffusionFreed {
                try? await Task.sleep(nanoseconds: 400_000_000)
                diffusionFreed = await diffusion.unloadDiffusionModelAsync()
            }
        }

        // If the sampler still hasn't returned after ~3s of retries, say so plainly rather than
        // reporting a reset that didn't happen — the user would otherwise retry into the same
        // wall. This should be rare now that cancellation is real; it mainly covers a very slow
        // individual step (a large SDXL generation on an older device) rather than the "nothing
        // we can do but wait" case this message used to describe.
        guard diffusionFreed else {
            LogManager.shared.log("ModelRecovery: aborted — diffusion context still in use by a running generation")
            onStage(.partial("The image still being generated is taking a moment to stop — cancellation was requested, but the current step hasn't finished yet. Try Reset again in a few seconds."))
            return
        }

        // Checked before the memory wait, not after: with nothing to reload, there's no reason to
        // make the user sit through up to `MemoryBudget.waitForRelease`'s ~20s timeout for memory
        // that was never going to be used, just to delay the same "Everything was unloaded"
        // message the wait doesn't change.
        guard let path = llm.lastUsedModelPath else {
            LogManager.shared.log("ModelRecovery: reset complete, no chat model to restore")
            onStage(.done("Everything was unloaded. Choose a model in Settings to start again."))
            return
        }

        onStage(.waitingForMemory)
        let target = llm.memoryHeadroomNeededGB(forModelSizeGB: llm.getModelSizeGB(at: URL(fileURLWithPath: path)))
        let reclaimed = await MemoryBudget.waitForRelease(atLeastGB: target)

        let url = URL(fileURLWithPath: path)
        onStage(.reloading(url.lastPathComponent))
        // Not a forced load. If there genuinely isn't room, the normal pre-flight refuses with a
        // reason the user can act on — forcing it here would trade a readable error for a jetsam
        // kill, which is the failure this whole button exists to get someone out of.
        llm.loadModel(at: url)

        LogManager.shared.log(String(
            format: "ModelRecovery: reset complete, %.2f GB plannable, reloading %@",
            reclaimed, url.lastPathComponent
        ))
        onStage(.done("Reloading \(url.lastPathComponent). The model bar will show when it's ready."))
    }
}

// MARK: - Button

/// Reset control, shared by the chat screen and Settings.
///
/// Surfaced automatically next to a failure in the model bar, and available unconditionally in
/// Settings — the states worth resetting from are not all states the app manages to detect, and a
/// recovery control the user can only reach when the app agrees something is wrong is not much of
/// a recovery control.
struct ResetModelsButton: View {

    @ObservedObject var llmManager: LLMManager
    @ObservedObject var diffusionManager: DiffusionManager

    /// `true` for the compact inline treatment used in the model bar.
    var compact: Bool = false

    @State private var isResetting = false
    @State private var stage: ModelRecovery.Stage?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button {
                run()
            } label: {
                HStack(spacing: 6) {
                    if isResetting {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: Theme.onAccent))
                            .scaleEffect(compact ? 0.6 : 0.8)
                    } else {
                        Image(systemName: "arrow.clockwise")
                    }
                    Text(compact ? "Reset" : "Reset Models")
                }
                .font(.system(size: compact ? 11 : 14, weight: .bold))
                .foregroundColor(Theme.onAccent)
                .padding(.horizontal, compact ? 12 : 16)
                .padding(.vertical, compact ? 6 : 12)
                .frame(maxWidth: compact ? nil : .infinity)
                .background(
                    RoundedRectangle(cornerRadius: compact ? 8 : 12)
                        .fill(isResetting ? Theme.border : Theme.accentCyan)
                )
            }
            .disabled(isResetting)
            .accessibilityLabel("Reset models")

            if let stage, !compact || !stage.isTerminal || isResetting {
                Text(stage.message)
                    .font(.system(size: compact ? 10 : 12))
                    .foregroundColor(Theme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if !compact {
                Text("Unloads the chat and image models, waits for the memory to come back, then reloads your last chat model. Use this after an out-of-memory error instead of force-quitting.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func run() {
        isResetting = true
        Task {
            await ModelRecovery.resetModels(llm: llmManager, diffusion: diffusionManager) { newStage in
                withAnimation { stage = newStage }
            }
            isResetting = false
        }
    }
}
