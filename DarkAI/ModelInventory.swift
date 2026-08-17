import Foundation
import Combine

/// A model the app put on this device, and where it should still be.
struct InstalledModel: Codable, Identifiable, Hashable {
    var fileName: String
    var kind: ModelKind
    /// Set when the model came from the built-in catalog, which is what makes one-tap recovery
    /// possible. `nil` for a file the user imported themselves — the app has no way to fetch that
    /// back and must say so rather than offering a button that can't work.
    var catalogID: String?
    var byteSize: Int64
    var installedAt: Date

    var id: String { "\(kind.rawValue)/\(fileName)" }

    var directory: URL {
        ModelDownloadManager.installDirectory(for: kind)
    }

    var url: URL { directory.appendingPathComponent(fileName) }

    var sizeDescription: String {
        ByteCountFormatter.string(fromByteCount: byteSize, countStyle: .file)
    }

    var catalogModel: CatalogModel? {
        guard let catalogID else { return nil }
        return ModelCatalog.all.first { $0.id == catalogID }
    }
}

/// Keeps a ledger of every model that was successfully installed, and notices when one goes away
/// without the user having deleted it.
///
/// **Why this exists.** Model weights live in `Documents/`, which iOS preserves across app updates,
/// and every lookup in the app is a fresh directory scan rather than a stored absolute path — so an
/// ordinary update does not lose them. What *does* lose them, silently and indistinguishably:
///
/// * Restoring to a new device, or a device-to-device transfer. `AppFiles` marks the model
///   directories `isExcludedFromBackup` on purpose (a 4 GB regenerable file has no business in a
///   5 GB iCloud tier), so the app and all of its settings come back and the weights do not.
/// * Deleting and reinstalling the app rather than updating it — including every
///   install-from-Xcode during development, which replaces the container outright.
/// * iOS reclaiming space under extreme storage pressure.
///
/// In every one of those cases the app's previous behaviour was identical: the model list is empty,
/// `lastUsedModelFileName` no longer resolves so the "Load previous model?" prompt never appears,
/// and the user is left to conclude that the update ate their download. This ledger is what lets the
/// app say which model is gone, why that happens, and — for catalog models — offer to fetch it back
/// in one tap.
///
/// The ledger is tiny (a few hundred bytes) and lives in `UserDefaults`, which is preserved in
/// exactly the cases the model files are not. That asymmetry is the whole point: it is how the app
/// can tell "you never had a model" apart from "your model is missing".
@MainActor
final class ModelInventory: ObservableObject {

    static let shared = ModelInventory()

    /// Models the ledger says should be on disk.
    @Published private(set) var installed: [InstalledModel] = []

    /// Models that were installed and are no longer present, and that the user hasn't dismissed.
    @Published private(set) var missing: [InstalledModel] = []

    private let installedKey = "DarkAI_InstalledModels"
    private let missingKey = "DarkAI_MissingModels"

    private init() {
        installed = load(installedKey)
        missing = load(missingKey)
    }

    // MARK: - Recording

    /// Called after a model is successfully written into its install directory, by whichever path
    /// put it there — catalog download, Settings import, or onboarding import.
    func record(fileName: String, kind: ModelKind, catalogID: String?, byteSize: Int64) {
        let entry = InstalledModel(
            fileName: fileName,
            kind: kind,
            catalogID: catalogID,
            byteSize: byteSize,
            installedAt: Date()
        )
        installed.removeAll { $0.id == entry.id }
        installed.append(entry)
        // A model that is back is no longer missing, however it got here.
        missing.removeAll { $0.id == entry.id }
        persist()
    }

    /// Called when the user deletes a model themselves. Distinct from it going missing: a
    /// deliberate deletion should never come back as a "this disappeared" notice.
    func forget(fileName: String, kind: ModelKind) {
        let id = "\(kind.rawValue)/\(fileName)"
        installed.removeAll { $0.id == id }
        missing.removeAll { $0.id == id }
        persist()
    }

    // MARK: - Reconciliation

    /// Compares the ledger against the filesystem. Run once at launch.
    ///
    /// Also picks up models installed before this ledger existed, by scanning the install
    /// directories — otherwise the first launch after adding this would report a user's existing
    /// models as untracked and then report them missing the moment they genuinely went away.
    func reconcile() {
        adoptUntrackedFiles()

        var stillInstalled: [InstalledModel] = []
        var newlyMissing: [InstalledModel] = []

        for entry in installed {
            if FileManager.default.fileExists(atPath: entry.url.path) {
                stillInstalled.append(entry)
            } else {
                newlyMissing.append(entry)
            }
        }

        installed = stillInstalled
        for entry in newlyMissing where !missing.contains(where: { $0.id == entry.id }) {
            missing.append(entry)
            LogManager.shared.log("ModelInventory: \(entry.fileName) is recorded as installed but is not on disk")
        }
        persist()
    }

    /// Adds anything sitting in the model directories that the ledger doesn't know about.
    /// `catalogID` is recovered by filename where the catalog recognises it, so a model downloaded
    /// by an older build still gets one-tap recovery.
    private func adoptUntrackedFiles() {
        // Core ML models aren't extension-filtered like the other two kinds: a single-window
        // model (OpenELM) installs as a `.mlpackage` directory, but a chunked pipeline model
        // (Llama) installs as a plain-named directory of `.mlmodelc` bundles with no `.mlpackage`
        // anywhere in it — an extension filter would silently never adopt one of those. Every
        // Core ML model, whatever its internal shape, is a top-level directory entry directly
        // under `AppFiles.coreMLModels` and nothing else lives there, so every entry qualifies.
        let coreMLEntries = (try? FileManager.default.contentsOfDirectory(
            at: AppFiles.coreMLModels, includingPropertiesForKeys: nil, options: .skipsHiddenFiles
        )) ?? []
        let discovered = AppFiles.contents(of: AppFiles.models, matchingExtensions: ["gguf"]).map { ($0, ModelKind.chat) }
            + AppFiles.contents(of: AppFiles.diffusionModels, matchingExtensions: SettingsView.acceptedDiffusionExtensions).map { ($0, ModelKind.diffusion) }
            + coreMLEntries.map { ($0, ModelKind.coreML) }

        for (url, kind) in discovered {
            let id = "\(kind.rawValue)/\(url.lastPathComponent)"
            guard !installed.contains(where: { $0.id == id }) else { continue }
            // `fileSizeKey` only describes a single file — a `.mlpackage` directory needs its
            // contents summed instead, or every adopted-but-untracked Core ML entry would record
            // as 0 bytes.
            let size: Int64
            if kind == .coreML {
                size = Int64(AppFiles.directorySizeGB(at: url) * 1024 * 1024 * 1024)
            } else {
                size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init) ?? 0
            }
            record(
                fileName: url.lastPathComponent,
                kind: kind,
                catalogID: ModelCatalog.model(withFileName: url.lastPathComponent)?.id,
                byteSize: size
            )
        }
    }

    /// Removes one loss notice. The ledger entry goes with it — the user has been told, and a
    /// notice that reappears every launch is noise rather than information.
    func dismiss(_ entry: InstalledModel) {
        missing.removeAll { $0.id == entry.id }
        installed.removeAll { $0.id == entry.id }
        persist()
    }

    func dismissAllLosses() {
        for entry in missing { installed.removeAll { $0.id == entry.id } }
        missing.removeAll()
        persist()
    }

    // MARK: - Storage

    private func load(_ key: String) -> [InstalledModel] {
        guard let data = UserDefaults.standard.data(forKey: key),
              let decoded = try? JSONDecoder().decode([InstalledModel].self, from: data) else { return [] }
        return decoded
    }

    private func persist() {
        let encoder = JSONEncoder()
        if let data = try? encoder.encode(installed) {
            UserDefaults.standard.set(data, forKey: installedKey)
        }
        if let data = try? encoder.encode(missing) {
            UserDefaults.standard.set(data, forKey: missingKey)
        }
    }
}
