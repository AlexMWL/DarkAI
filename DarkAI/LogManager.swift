import Foundation
import Combine

/// Diagnostic log, writable from any thread.
///
/// Explicitly `nonisolated`. This module builds with `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`,
/// so without this the whole type would be main-actor bound — and almost everything that needs to
/// log is not: the `LlamaRunner` and `DiffusionRunner` actors, `SDWrapper` on its background
/// queue, `AppFiles` during file setup, and the model-download delegate callbacks. Leaving it
/// main-actor isolated meant every one of those call sites either warned or had to hop to the
/// main actor just to record a line of text, which is the wrong direction: logging should never
/// make the caller wait on the UI thread.
///
/// `@unchecked Sendable` is accurate rather than a shortcut. The only mutable state is `logs`,
/// which is pinned to the main actor below, and the file writes are serialised onto `queue`.
nonisolated final class LogManager: ObservableObject, @unchecked Sendable {
    static let shared = LogManager()

    /// Pinned to the main actor: it drives `LogExportView`, and SwiftUI requires published
    /// mutations on the main thread regardless of where the log call originated.
    @MainActor @Published var logs: [String] = []

    private let logFileURL: URL = AppFiles.diagnosticLogFile

    /// `DateFormatter` is not thread-safe, so it is only ever touched from `queue`. It used to be
    /// formatted on whichever thread called `log()`, which was a latent data race.
    private let formatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return f
    }()

    private let queue = DispatchQueue(label: "com.darkai.logmanager", qos: .background)

    private init() {
        loadLogs()
        log("Diagnostic Logger initialized.")
    }

    func log(_ message: String) {
        // Everything happens on the serial queue: timestamp formatting, the console echo, and
        // the file append. The caller — which may be a background actor mid-inference — returns
        // immediately.
        queue.async {
            let timestamp = self.formatter.string(from: Date())
            let logLine = "[\(timestamp)] \(message)"

            // Console echo, so the same line is visible in Xcode while debugging.
            print(logLine)

            self.appendToFile(logLine)

            Task { @MainActor in
                self.logs.append(logLine)
                if self.logs.count > 1000 {
                    self.logs.removeFirst(self.logs.count - 1000)
                }
            }
        }
    }

    private func loadLogs() {
        guard FileManager.default.fileExists(atPath: logFileURL.path),
              let content = try? String(contentsOf: logFileURL, encoding: .utf8) else { return }
        let lines = content
            .components(separatedBy: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        let recent = Array(lines.suffix(1000))
        Task { @MainActor in
            self.logs = recent
        }
    }

    private func appendToFile(_ line: String) {
        let data = (line + "\n").data(using: .utf8) ?? Data()
        if FileManager.default.fileExists(atPath: logFileURL.path) {
            if let fileHandle = try? FileHandle(forWritingTo: logFileURL) {
                fileHandle.seekToEndOfFile()
                fileHandle.write(data)
                fileHandle.closeFile()
            }
        } else {
            try? (line + "\n").write(to: logFileURL, atomically: true, encoding: .utf8)
        }
    }

    func clearLogs() {
        queue.async {
            try? "".write(to: self.logFileURL, atomically: true, encoding: .utf8)
            Task { @MainActor in
                self.logs.removeAll()
            }
            self.log("Logs cleared by user.")
        }
    }

    func getLogFileURL() -> URL {
        logFileURL
    }
}
