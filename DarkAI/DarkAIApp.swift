import SwiftUI

@main
struct DarkAIApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    @AppStorage("onboardingCompleted") private var onboardingCompleted = false
    @AppStorage("acceptedPolicyVersion") private var acceptedPolicyVersion = 0

    @StateObject private var appearance = AppearanceManager.shared

    init() {
        AppFiles.prepare()
        AppAppearance.configure()
        // Must run before anything heavy: it recovers whatever the previous run left behind and
        // arms the handlers for this one, and a crash during model loading is exactly the case
        // it exists to catch.
        CrashReporter.start()
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
