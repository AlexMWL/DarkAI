import Foundation

/// Centralized on-disk layout, plus the two file attributes App Review actually checks.
///
/// Why this replaced ad-hoc `FileManager.default.urls(for:in:)` calls scattered across the
/// managers: model weights are multi-gigabyte, and anything sitting in `Documents/` without
/// `isExcludedFromBackup` gets swept into the user's iCloud backup. Apple's iOS Data Storage
/// Guidelines call that out specifically, and it is one of the most common rejections for apps
/// that store large regenerable payloads — a 4 GB model that the user can re-download does not
/// belong in their 5 GB free iCloud tier.
///
/// The directories stay under `Documents/` rather than moving to `Application Support/` on
/// purpose: `UIFileSharingEnabled` exposes `Documents/` in the Files app, which is how a user
/// drops in their own `.gguf` without going through the in-app picker. Excluding from backup
/// gets the storage behaviour right while keeping that affordance.
/// Explicitly `nonisolated`. These are pure filesystem helpers with no shared mutable state, but
/// this module builds with `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`, so by default they would
/// be main-actor bound. That is the wrong isolation for the work: model imports and downloads run
/// on background tasks precisely so multi-gigabyte copies don't block the UI, and they need to
/// size, move, and flag those files from there.
nonisolated enum AppFiles {

    // MARK: - Locations

    private static var documents: URL {
        // Force-unwrap is safe here in a way it usually isn't: the documents directory is
        // guaranteed to exist for a sandboxed iOS app, and every alternative (silently writing
        // to a temp path, or making every call site optional) is worse than failing loudly if
        // that invariant is ever violated.
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    /// Imported and downloaded chat model weights (`.gguf`).
    static var models: URL { documents.appendingPathComponent("Models", isDirectory: true) }

    /// Imported diffusion checkpoints (`.gguf`, `.safetensors`, `.ckpt`).
    static var diffusionModels: URL { documents.appendingPathComponent("DiffusionModels", isDirectory: true) }

    /// Downloaded Core ML model packages (`.mlpackage` directory bundles) plus their compiled
    /// `.mlmodelc` cache — see `CoreMLRunner.load`.
    static var coreMLModels: URL { documents.appendingPathComponent("CoreMLModels", isDirectory: true) }

    /// Images produced by the diffusion pipeline and retained as RAG entries.
    static var generatedImages: URL { documents.appendingPathComponent("GeneratedImages", isDirectory: true) }

    /// Writes generated-image bytes to `generatedImages` under a fresh UUID filename and returns
    /// that filename, or `nil` on failure.
    ///
    /// The single write point for that directory — `ConversationManager` and `RAGManager` both
    /// need a copy of the same generated image (one to show it in the chat, one to make it
    /// retrievable later), and each used to write its own separate file for what was, in memory,
    /// the exact same `Data`. Writing once here and handing both managers the resulting filename
    /// means a generated image is stored on disk exactly once, not twice.
    @discardableResult
    static func writeGeneratedImage(_ data: Data) -> String? {
        createIfNeeded(generatedImages)
        let fileName = "\(UUID().uuidString).jpg"
        let url = generatedImages.appendingPathComponent(fileName)
        do {
            try data.write(to: url)
        } catch {
            LogManager.shared.log("AppFiles: failed to write generated image \(fileName) — \(error.localizedDescription)")
            return nil
        }
        excludeFromBackup(url)
        return fileName
    }

    /// Partial downloads. Separate from `models` so an interrupted download is never mistaken
    /// for an importable model by the settings list.
    static var downloadsInProgress: URL { documents.appendingPathComponent("Downloads", isDirectory: true) }

    static var diagnosticLogFile: URL { documents.appendingPathComponent("diagnostic_logs.txt") }

    /// Written from inside a signal handler, so its path is resolved once at startup and the
    /// file is opened with raw `open(2)`. Consumed and deleted on the next launch.
    static var crashReportFile: URL { documents.appendingPathComponent("last_crash.txt") }

    /// The recovered, structured report waiting to be shown to the user.
    static var pendingCrashReportFile: URL { documents.appendingPathComponent("pending_crash.json") }

    private static var allDirectories: [URL] {
        [models, diffusionModels, coreMLModels, generatedImages, downloadsInProgress]
    }

    // MARK: - Setup

    /// Creates every directory the app writes to and applies backup exclusion + file protection.
    /// Idempotent; called on launch and cheap enough to call again before a write.
    static func prepare() {
        for directory in allDirectories {
            createIfNeeded(directory)
            excludeFromBackup(directory)
            applyFileProtection(directory)
        }
        // These live directly in Documents rather than a subdirectory, so they need the same
        // treatment applied individually.
        excludeFromBackup(diagnosticLogFile)
        excludeFromBackup(crashReportFile)
        excludeFromBackup(pendingCrashReportFile)
    }

    @discardableResult
    static func createIfNeeded(_ url: URL) -> Bool {
        guard !FileManager.default.fileExists(atPath: url.path) else { return true }
        do {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            return true
        } catch {
            LogManager.shared.log("AppFiles: failed to create \(url.lastPathComponent) — \(error.localizedDescription)")
            return false
        }
    }

    /// Marks a file or directory as excluded from iCloud and iTunes backup.
    ///
    /// Applied to the *directory*, which covers files created inside it later — but call it on
    /// individual large files too when they are created outside these directories, because the
    /// exclusion flag is not inherited retroactively by every filesystem operation on all iOS
    /// versions.
    static func excludeFromBackup(_ url: URL) {
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        var target = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        do {
            try target.setResourceValues(values)
        } catch {
            LogManager.shared.log("AppFiles: backup exclusion failed for \(url.lastPathComponent) — \(error.localizedDescription)")
        }
    }

    /// `.completeUntilFirstUserAuthentication` rather than `.complete`: model loading and
    /// background download completion both need to read these files while the device is locked,
    /// which `.complete` would make impossible. This is the same protection class iOS applies to
    /// its own app containers by default, set explicitly so it survives a container migration.
    private static func applyFileProtection(_ url: URL) {
        do {
            try FileManager.default.setAttributes(
                [.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication],
                ofItemAtPath: url.path
            )
        } catch {
            LogManager.shared.log("AppFiles: file protection failed for \(url.lastPathComponent) — \(error.localizedDescription)")
        }
    }

    // MARK: - Queries

    static func contents(of directory: URL, matchingExtensions extensions: Set<String>) -> [URL] {
        createIfNeeded(directory)
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey],
            options: .skipsHiddenFiles
        ) else { return [] }
        return files
            .filter { extensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent.localizedCaseInsensitiveCompare($1.lastPathComponent) == .orderedAscending }
    }

    static func fileSizeGB(at url: URL) -> Double {
        guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize else { return 0 }
        return Double(size) / (1024 * 1024 * 1024)
    }

    /// Total size of everything under a directory, in gigabytes — `fileSizeKey` alone is only
    /// meaningful for a single file, so a `.mlpackage` (a directory bundle) needs this instead.
    static func directorySizeGB(at url: URL) -> Double {
        var total: Int64 = 0
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey],
            options: .skipsHiddenFiles
        ) else { return 0 }
        for case let fileURL as URL in enumerator {
            total += Int64((try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return Double(total) / (1024 * 1024 * 1024)
    }

    /// Free space on the volume backing the app container, in gigabytes. Uses the
    /// capacity-for-important-usage key rather than raw `systemFreeSize`, which over-reports on
    /// modern iOS by counting purgeable space the app can't actually claim.
    static func availableDiskGB() -> Double {
        guard let values = try? documents.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
              let capacity = values.volumeAvailableCapacityForImportantUsage else {
            return 0
        }
        return Double(capacity) / (1024 * 1024 * 1024)
    }

    /// Total bytes occupied by everything this app has written, for the storage readout in
    /// Settings. Users deserve to see where several gigabytes went before being asked to
    /// download more.
    static func totalUsedGB() -> Double {
        var total: Int64 = 0
        for directory in allDirectories {
            guard let enumerator = FileManager.default.enumerator(
                at: directory,
                includingPropertiesForKeys: [.fileSizeKey],
                options: .skipsHiddenFiles
            ) else { continue }
            for case let url as URL in enumerator {
                total += Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            }
        }
        // These live directly in Documents rather than inside one of `allDirectories` (same
        // reason `prepare()` above excludes them from backup individually) — the diagnostic log
        // in particular grows continuously between manual clears, and used to be invisible here.
        for looseFile in [diagnosticLogFile, crashReportFile, pendingCrashReportFile] {
            total += Int64((try? looseFile.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return Double(total) / (1024 * 1024 * 1024)
    }
}
