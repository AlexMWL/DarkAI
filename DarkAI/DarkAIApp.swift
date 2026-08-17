import SwiftUI

@main
struct DarkAIApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    @AppStorage("onboardingCompleted") private var onboardingCompleted = false
    @AppStorage("acceptedPolicyVersion") private var acceptedPolicyVersion = 0

    @StateObject private var appearance = AppearanceManager.shared

    init() {
        // First, unconditionally — it installs the signal/exception handlers. Everything below
        // is real file I/O (directory creation, enumerating and validating every model on disk)
        // and a crash inside any of it needs to be caught same as a crash during model loading;
        // running this after that I/O left exactly those crashes invisible to the reporter.
        CrashReporter.start()
        AppFiles.prepare()
        // Compares the record of what was installed against what is actually on disk. Model files
        // are excluded from iCloud backup by design, so a restore or device transfer brings the
        // app's settings back without its weights — this is what turns that from "the update
        // deleted my model" into a named model with a re-download button. See `ModelInventory`.
        ModelInventory.shared.reconcile()
        AppAppearance.configure()
    }

    /// Re-gate an existing user when the acceptable-use policy or terms change materially.
    /// Apple expects agreement to the *current* terms, not to whatever was in force the day the
    /// app was installed.
    private var needsPolicyReacceptance: Bool {
        onboardingCompleted && acceptedPolicyVersion < AppInfo.policyVersion
    }

    var body: some Scene {
        WindowGroup {
            Group {
                if !onboardingCompleted {
                    OnboardingView()
                } else if needsPolicyReacceptance {
                    OnboardingView(startsAtAgreement: true)
                } else {
                    ContentView()
                }
            }
            // Drives the whole palette: `Theme`'s colours are dynamic, so setting the scheme
            // here re-resolves every one of them. `nil` (Automatic) hands the decision back to
            // the system.
            .preferredColorScheme(appearance.mode.colorScheme)
            .environmentObject(appearance)
        }
    }
}

/// Exists for one reason: model downloads run on a background `URLSession`, and iOS relaunches
/// the app in the background to hand the finished transfer back through this callback. Without
/// it, a download that completes while the app is suspended is never moved into place.
final class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        Task { @MainActor in
            ModelDownloadManager.shared.backgroundCompletionHandler = completionHandler
        }
    }
}
