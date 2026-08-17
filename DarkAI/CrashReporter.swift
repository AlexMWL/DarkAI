import Foundation
import Combine
import UIKit
import Darwin

/// On-device crash detection and reporting.
///
/// Why hand-rolled instead of a third-party SDK: every crash-reporting service is a network
/// dependency that phones home, which would break the app's central privacy claim and force a
/// "Data Collected" entry on the App Store privacy label. This writes a report to local storage
/// and does nothing else — nothing is transmitted unless the user opens it and taps send in their
/// own mail app.
///
/// Two detection paths, because this app has two very different ways of dying:
///
/// 1. **Signals and uncaught exceptions** — the classic crash. Caught by handlers that write a
///    report synchronously from inside the failing process.
/// 2. **Unclean termination with no signal** — which is what a jetsam out-of-memory kill looks
///    like, and for an app that loads multi-gigabyte model weights it is *the* dominant failure
///    mode. The kernel gives no chance to run any handler, so this is inferred on the next launch
///    from a clean-shutdown flag, and the memory reading recorded just before the model load is
///    what makes the report actually diagnostic.
enum CrashReporter {

    // MARK: - Types

    struct Report: Codable, Identifiable {
        var id: UUID = UUID()
        var date: Date
        var kind: String
        var reason: String
        var stack: [String]
        var appVersion: String
        var osVersion: String
        var deviceModel: String
        /// Process memory headroom at the last checkpoint before the crash, in MB. The single
        /// most useful number in the whole report for this app.
        var availableMemoryMB: Int
        var lastActivity: String
        /// Chat model resident when the crash happened. Optional with a default so reports
        /// written by an earlier build still decode.
        var chatModel: String? = nil
        /// Diffusion checkpoint resident when the crash happened.
        var diffusionModel: String? = nil

        var summary: String {
            "\(kind) — \(reason)"
        }

        /// `os_proc_available_memory()` is device-only — it reports 0 on the Simulator. Printing
        /// "0 MB" there would read as "the device was completely out of memory", which is both
        /// wrong and the most alarming possible misreading of the most important field here.
        var memoryDescription: String {
            availableMemoryMB > 0 ? "\(availableMemoryMB) MB" : "not recorded"
        }

        var formattedForEmail: String {
            """
            \(AppInfo.displayName) crash report

            When: \(date.formatted(date: .abbreviated, time: .standard))
            Type: \(kind)
            Reason: \(reason)
            Last activity: \(lastActivity)
            Memory free at last checkpoint: \(memoryDescription)

            Chat model: \(chatModel ?? "none loaded")
            Diffusion model: \(diffusionModel ?? "none loaded")

            App: \(appVersion)
            OS: \(osVersion)
            Device: \(deviceModel)

            Stack:
            \(stack.isEmpty ? "(not available — no signal was raised)" : stack.joined(separator: "\n"))
            """
        }
    }

    // MARK: - Storage keys

    private static let cleanExitKey = "DarkAI_CleanExit"
    private static let lastActivityKey = "DarkAI_LastActivity"
    private static let lastMemoryKey = "DarkAI_LastAvailableMemoryMB"
    private static let enabledKey = "DarkAI_CrashReportingEnabled"
    private static let chatModelKey = "DarkAI_LoadedChatModel"
    private static let diffusionModelKey = "DarkAI_LoadedDiffusionModel"

    private static var signalReportURL: URL { AppFiles.crashReportFile }
    private static var pendingReportURL: URL { AppFiles.pendingCrashReportFile }

    /// User-facing kill switch. On by default — it costs nothing and reports go nowhere on their
    /// own — but a user who doesn't want any crash file written at all can say so.
    static var isEnabled: Bool {
        get {
            guard UserDefaults.standard.object(forKey: enabledKey) != nil else { return true }
            return UserDefaults.standard.bool(forKey: enabledKey)
        }
        set { UserDefaults.standard.set(newValue, forKey: enabledKey) }
    }

    // MARK: - Lifecycle

    /// Call once, as early in launch as possible. Recovers any report left by the previous run,
    /// then arms the handlers for this one.
    static func start() {
        recoverPreviousRun()

        guard isEnabled else { return }

        installExceptionHandler()
        installSignalHandlers()
        observeLifecycle()

        // Mark this run as in-progress. Set back to true when the app backgrounds or terminates
        // normally; if it's still false at next launch, the process died without warning.
        UserDefaults.standard.set(false, forKey: cleanExitKey)
        note("launched")
    }

    /// Records what the app was doing, so an OOM kill has some context attached. Deliberately
    /// coarse — this writes to UserDefaults on every call and shouldn't be used as a trace log.
    static func note(_ activity: String) {
        guard isEnabled else { return }
        UserDefaults.standard.set(activity, forKey: lastActivityKey)
        // Zero on the Simulator, where this API isn't implemented. Stored as-is and rendered as
        // "not recorded" downstream rather than faked, so a report from a real device is never
        // confused with one from a simulator run.
        let availableMB = Int(Double(os_proc_available_memory()) / (1024 * 1024))
        UserDefaults.standard.set(availableMB, forKey: lastMemoryKey)
    }

    /// Records which weights are resident, so a report says *what* was running when it died.
    ///
    /// This is the difference between "SIGABRT in the text encoder" and "SIGABRT in the text
    /// encoder while Juggernaut XL v9 was loaded" — only the second one identifies a checkpoint
    /// worth reproducing against. Persisted rather than captured at crash time because a signal
    /// handler can't safely read Swift state.
    static func noteLoadedModels(chat: String?, diffusion: String?) {
        guard isEnabled else { return }
        UserDefaults.standard.set(chat, forKey: chatModelKey)
        UserDefaults.standard.set(diffusion, forKey: diffusionModelKey)
    }

    static func noteChatModel(_ name: String?) {
        guard isEnabled else { return }
        UserDefaults.standard.set(name, forKey: chatModelKey)
    }

    static func noteDiffusionModel(_ name: String?) {
        guard isEnabled else { return }
        UserDefaults.standard.set(name, forKey: diffusionModelKey)
    }

    private static func observeLifecycle() {
        let center = NotificationCenter.default
        for name in [UIApplication.didEnterBackgroundNotification, UIApplication.willTerminateNotification] {
            center.addObserver(forName: name, object: nil, queue: .main) { _ in
                UserDefaults.standard.set(true, forKey: cleanExitKey)
            }
        }
        center.addObserver(forName: UIApplication.didBecomeActiveNotification, object: nil, queue: .main) { _ in
            UserDefaults.standard.set(false, forKey: cleanExitKey)
        }
        center.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main) { _ in
            note("memory warning received")
        }
    }

    // MARK: - Recovery

    /// Turns whatever the previous run left behind into a `Report`, in priority order: an actual
    /// signal/exception dump first, then the unclean-exit inference.
    private static func recoverPreviousRun() {
        let cleanExit = UserDefaults.standard.object(forKey: cleanExitKey) as? Bool ?? true
        let lastActivity = UserDefaults.standard.string(forKey: lastActivityKey) ?? "unknown"
        let lastMemoryMB = UserDefaults.standard.integer(forKey: lastMemoryKey)
        let chatModel = UserDefaults.standard.string(forKey: chatModelKey)
        let diffusionModel = UserDefaults.standard.string(forKey: diffusionModelKey)

        var report: Report?
        // Only true for the signal/exception-dump branch below — the raw file backing it is only
        // safe to delete once the decoded report has been durably written elsewhere. See the
        // deletion at the bottom of this function.
        var hasSignalFileToClear = false

        if let raw = try? String(contentsOf: signalReportURL, encoding: .utf8), !raw.isEmpty {
            let lines = raw.components(separatedBy: "\n")
            let header = lines.first ?? "Unknown"
            let parts = header.components(separatedBy: "|")
            report = Report(
                date: Date(),
                kind: parts.first ?? "Crash",
                reason: parts.count > 1 ? parts[1] : "The app stopped unexpectedly.",
                stack: Array(lines.dropFirst().filter { !$0.isEmpty }.prefix(40)),
                appVersion: AppInfo.versionString,
                osVersion: "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)",
                deviceModel: deviceIdentifier(),
                availableMemoryMB: lastMemoryMB,
                lastActivity: lastActivity,
                chatModel: chatModel,
                diffusionModel: diffusionModel
            )
            hasSignalFileToClear = true

        } else if !cleanExit {
            // No signal, no exception, but the app never shut down cleanly. On iOS that is
            // overwhelmingly the kernel reclaiming memory.
            report = Report(
                date: Date(),
                kind: "Unexpected termination",
                reason: lastMemoryMB > 0 && lastMemoryMB < 600
                    ? "The app was closed by iOS, most likely because it ran out of memory."
                    : "The app closed without shutting down normally.",
                stack: [],
                appVersion: AppInfo.versionString,
                osVersion: "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)",
                deviceModel: deviceIdentifier(),
                availableMemoryMB: lastMemoryMB,
                lastActivity: lastActivity,
                chatModel: chatModel,
                diffusionModel: diffusionModel
            )
        }

        guard let report else { return }

        // Persist the decoded report before touching its raw source. Deleting the signal file
        // first (as this used to) loses the report outright if the write below fails — disk full
        // is a realistic state for a multi-GB-model app to crash in, which is exactly when this
        // path is most likely to run.
        guard let encoded = try? JSONEncoder().encode(report) else {
            LogManager.shared.log("CrashReporter: failed to encode recovered report — \(report.summary)")
            return
        }
        do {
            try encoded.write(to: pendingReportURL)
        } catch {
            LogManager.shared.log("CrashReporter: failed to persist recovered report (\(error.localizedDescription)) — leaving the raw signal file in place to retry next launch")
            return
        }
        AppFiles.excludeFromBackup(pendingReportURL)
        if hasSignalFileToClear {
            try? FileManager.default.removeItem(at: signalReportURL)
        }
        LogManager.shared.log("CrashReporter: recovered report — \(report.summary)")
    }

    /// Peeks at the stored report without consuming it. Used by Settings, where the user is
    /// deliberately going to look at it and expects it to still be there afterwards.
    static func pendingReport() -> Report? {
        guard let data = try? Data(contentsOf: pendingReportURL),
              let report = try? JSONDecoder().decode(Report.self, from: data) else { return nil }
        return report
    }

    /// Reads the stored report **and deletes it in the same call**, so surfacing one can never
    /// fail in a way that leaves it to be surfaced again.
    ///
    /// This exists because the launch path previously deleted the file only when the user
    /// dismissed the crash sheet. A crash during image generation therefore produced a report
    /// that was presented at launch, and if that presentation failed to render — which it did,
    /// racing three stacked sheet modifiers during initial layout — the user got a blank screen
    /// with nothing to dismiss, the file was never cleared, and *every subsequent launch
    /// reproduced it*. A force-quit didn't help because the trap was on disk, not in memory.
    /// Consuming the file up front means the worst case is one lost report, not a bricked app.
    static func takePendingReport() -> Report? {
        let report = pendingReport()
        clearPendingReport()
        return report
    }

    static func clearPendingReport() {
        try? FileManager.default.removeItem(at: pendingReportURL)
    }

    // MARK: - Handlers

    private static func installExceptionHandler() {
        NSSetUncaughtExceptionHandler { exception in
            // Not in a signal context here, so ordinary Swift is safe to use.
            let name = exception.name.rawValue
            let reason = exception.reason ?? "No reason given"
            let stack = exception.callStackSymbols.prefix(40).joined(separator: "\n")
            let payload = "Uncaught exception (\(name))|\(reason)\n\(stack)"
            try? payload.write(to: CrashReporter.signalReportURL, atomically: true, encoding: .utf8)
        }
    }

    private static func installSignalHandlers() {
        // Only async-signal-safe calls are allowed past this point: no Swift String work, no
        // Foundation, no allocation. The report is assembled with `write(2)` and
        // `backtrace_symbols_fd`, both of which are safe to call from a signal handler.
        let handler: @convention(c) (Int32) -> Void = { signalNumber in
            let path = CrashReporter.signalReportPathBuffer
            let fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
            if fd >= 0 {
                let name = CrashReporter.signalName(signalNumber)
                _ = name.withCString { write(fd, $0, strlen($0)) }

                var callstack = [UnsafeMutableRawPointer?](repeating: nil, count: 64)
                let frames = backtrace(&callstack, 64)
                backtrace_symbols_fd(&callstack, frames, fd)
                close(fd)
            }

            // Restore the default disposition and re-raise, so the crash still reaches the OS
            // and shows up in Settings → Privacy → Analytics like any other. Swallowing it would
            // leave the process wedged in a half-dead state.
            signal(signalNumber, SIG_DFL)
            raise(signalNumber)
        }

        for sig in [SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGTRAP] {
            signal(sig, handler)
        }
    }

    /// Path as a C string, resolved once at startup. Computing it inside the handler would mean
    /// allocating in a signal context, which is exactly what must not happen there.
    private static let signalReportPathBuffer: UnsafePointer<CChar> = {
        let path = AppFiles.crashReportFile.path
        let count = path.utf8.count + 1
        let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: count)
        path.withCString { _ = strcpy(buffer, $0) }
        return UnsafePointer(buffer)
    }()

    /// Signal-safe: returns a compile-time constant string, no formatting or allocation.
    private static func signalName(_ sig: Int32) -> StaticString {
        switch sig {
        case SIGABRT: return "Crash (SIGABRT)|The app aborted. This is usually a failed assertion or an unhandled Swift error.\n"
        case SIGSEGV: return "Crash (SIGSEGV)|The app accessed invalid memory.\n"
        case SIGBUS:  return "Crash (SIGBUS)|The app accessed misaligned or unmapped memory.\n"
        case SIGILL:  return "Crash (SIGILL)|The app executed an illegal instruction.\n"
        case SIGFPE:  return "Crash (SIGFPE)|An arithmetic error occurred.\n"
        case SIGTRAP: return "Crash (SIGTRAP)|The app hit a runtime trap, such as a failed force-unwrap or array bounds check.\n"
        default:      return "Crash|The app stopped unexpectedly.\n"
        }
    }

    // MARK: - Device

    /// Hardware identifier ("iPhone16,2") rather than the marketing name. It's what actually
    /// pins down the memory budget a report needs to be interpreted against.
    static func deviceIdentifier() -> String {
        var info = utsname()
        uname(&info)
        // Copied to a local first: passing `&info.machine` into a closure that also reads
        // `info.machine` for its capacity is an overlapping-access violation.
        let machine = info.machine
        let identifier = withUnsafeBytes(of: machine) { raw -> String in
            let bytes = raw.prefix { $0 != 0 }
            return String(decoding: bytes, as: UTF8.self)
        }
        let ramGB = Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
        return String(format: "%@ (%.1f GB RAM)", identifier, ramGB)
    }
}

extension StaticString {
    /// `withCString` for `StaticString`, which doesn't provide one. Safe in a signal handler:
    /// the pointer is to static storage, so nothing is allocated or copied.
    func withCString<R>(_ body: (UnsafePointer<CChar>) -> R) -> R {
        if hasPointerRepresentation {
            return utf8Start.withMemoryRebound(to: CChar.self, capacity: utf8CodeUnitCount + 1) { body($0) }
        }
        return "".withCString(body)
    }
}
