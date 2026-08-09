import Foundation
import Combine
import UIKit

// MARK: - Catalog

/// A model the app offers to fetch for the user.
///
/// The catalog is a fixed, developer-curated list — there is deliberately no "paste a URL" field.
/// Two reasons: an arbitrary-URL downloader turns the app into a general-purpose file fetcher,
/// which App Review treats as a way to bypass the review of what the app actually ships; and
/// every entry here has had its license checked, which is not something the app can do for an
/// arbitrary URL. Users who want a specific model can still side-load it into the app's Files
/// folder or use the document picker — that path is theirs, not something the app arranges.
enum ModelKind: String, Codable {
    /// Chat/instruct weights driven by llama.cpp. Land in `AppFiles.models`.
    case chat
    /// Diffusion checkpoints driven by stable-diffusion.cpp. Land in `AppFiles.diffusionModels`.
    case diffusion
}

struct CatalogModel: Identifiable, Hashable {
    let id: String
    let kind: ModelKind
    let displayName: String
    let publisher: String
    /// Exact `Content-Length` of the remote file, verified against what actually lands on disk.
    let byteSize: Int64
    let parameterCount: String
    let quantization: String
    let license: String
    /// Extra notice the license obliges the app to display, if any.
    let attribution: String?
    let summary: String
    let url: URL
    /// Approximate peak RAM to run it, weights plus a default context. Drives the
    /// "may not run on this device" warning before a user spends a gigabyte of cellular data.
    let approxRuntimeGB: Double
    /// Overrides `fileName` when `url` doesn't end in the real filename. Hugging Face URLs are
    /// self-describing (`.../resolve/main/some-model.gguf`), but Civitai's download endpoint is
    /// `/api/download/models/{versionId}` — the actual filename only ever appears in the
    /// response's `Content-Disposition` header, never in the URL itself. Without this, a
    /// Civitai-sourced model would install under a literal numeric ID with no extension: hidden
    /// from every extension-filtered model list in the app and unloadable.
    let fileNameOverride: String?

    init(id: String, kind: ModelKind, displayName: String, publisher: String, byteSize: Int64,
         parameterCount: String, quantization: String, license: String, attribution: String?,
         summary: String, url: URL, approxRuntimeGB: Double, fileNameOverride: String? = nil) {
        self.id = id
        self.kind = kind
        self.displayName = displayName
        self.publisher = publisher
        self.byteSize = byteSize
        self.parameterCount = parameterCount
        self.quantization = quantization
        self.license = license
        self.attribution = attribution
        self.summary = summary
        self.url = url
        self.approxRuntimeGB = approxRuntimeGB
        self.fileNameOverride = fileNameOverride
    }

    var fileName: String { fileNameOverride ?? url.lastPathComponent }
    var sizeGB: Double { Double(byteSize) / (1024 * 1024 * 1024) }
    var sizeDescription: String { String(format: "%.2f GB", sizeGB) }
}

enum ModelCatalog {

    /// Curated starter models. All are instruction-tuned, all are small enough to run on any
    /// device that meets the app's deployment target, and all are redistributable under a
    /// license that permits this use.
    ///
    /// Sizes are the exact byte counts served by the host — they are checked after download, so
    /// if a host ever republishes a file at a different size the verification fails loudly
    /// instead of leaving a truncated model that only misbehaves at inference time.
    static let chatModels: [CatalogModel] = [
        CatalogModel(
            id: "llama-3.2-1b-instruct-q4km",
            kind: .chat,
            displayName: "Llama 3.2 1B Instruct",
            publisher: "Meta (GGUF build by bartowski)",
            byteSize: 807_694_464,
            parameterCount: "1B",
            quantization: "Q4_K_M",
            license: "Llama 3.2 Community License",
            attribution: "Built with Llama. Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.",
            summary: "Smallest and fastest. A good first download and the least demanding on memory.",
            url: URL(string: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf")!,
            approxRuntimeGB: 1.6
        ),
        CatalogModel(
            id: "qwen2.5-1.5b-instruct-q4km",
            kind: .chat,
            displayName: "Qwen2.5 1.5B Instruct",
            publisher: "Alibaba Cloud (GGUF build by bartowski)",
            byteSize: 986_048_768,
            parameterCount: "1.5B",
            quantization: "Q4_K_M",
            license: "Apache 2.0",
            attribution: nil,
            summary: "Stronger at reasoning and code than the 1B models, still comfortable on most devices.",
            url: URL(string: "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")!,
            approxRuntimeGB: 1.9
        ),
        CatalogModel(
            id: "smollm2-1.7b-instruct-q4km",
            kind: .chat,
            displayName: "SmolLM2 1.7B Instruct",
            publisher: "Hugging Face (GGUF build by bartowski)",
            byteSize: 1_055_609_824,
            parameterCount: "1.7B",
            quantization: "Q4_K_M",
            license: "Apache 2.0",
            attribution: nil,
            summary: "Conversational and well-behaved on short prompts. Trained on openly documented data.",
            url: URL(string: "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf")!,
            approxRuntimeGB: 2.0
        ),
        CatalogModel(
            id: "llama-3.2-3b-instruct-q4km",
            kind: .chat,
            displayName: "Llama 3.2 3B Instruct",
            publisher: "Meta (GGUF build by hugging-quants)",
            byteSize: 2_019_373_920,
            parameterCount: "3B",
            quantization: "Q4_K_M",
            license: "Llama 3.2 Community License",
            attribution: "Built with Llama. Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.",
            summary: "The most capable chat model here — noticeably better at reasoning and long answers. Needs a device with plenty of memory.",
            url: URL(string: "https://huggingface.co/hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF/resolve/main/llama-3.2-3b-instruct-q4_k_m.gguf")!,
            approxRuntimeGB: 3.4
        )
    ]

    /// Diffusion checkpoints offered for download.
    ///
    /// One entry, chosen conservatively. Image generation loads the full UNet, the CLIP text
    /// encoder, and the VAE at once, and the app has to evict the chat model to make room — so
    /// the smallest complete checkpoint that produces good 512×512 output is the right default,
    /// not the most capable one.
    ///
    /// Verified before listing, which earlier entries were not: the file is a single-file
    /// checkpoint carrying all three components (686 UNet tensors under `model.diffusion_model`,
    /// 248 VAE under `first_stage_model`, 196 CLIP under `cond_stage_model`), and its longest
    /// tensor name is 83 bytes — within the 160-byte `GGML_MAX_NAME` that stable-diffusion.cpp
    /// is compiled with, now that `build_sd_ios.sh` stops those symbols binding to llama.cpp's
    /// 64-byte build.
    static let diffusionModels: [CatalogModel] = [
        CatalogModel(
            id: "sd-1.5-emaonly-q4_0",
            kind: .diffusion,
            displayName: "Stable Diffusion 1.5",
            publisher: "Runway / Stability AI (GGUF build by Second State)",
            byteSize: 1_566_768_416,
            parameterCount: "860M UNet",
            quantization: "Q4_0",
            license: "CreativeML OpenRAIL-M",
            attribution: "Stable Diffusion v1-5 is licensed under the CreativeML OpenRAIL-M license, which prohibits generating illegal or harmful content.",
            summary: "Generates 512×512 images. The EMA-only pruned build — the smallest complete Stable Diffusion checkpoint, and the most forgiving on memory.",
            url: URL(string: "https://huggingface.co/second-state/stable-diffusion-v1-5-GGUF/resolve/main/stable-diffusion-v1-5-pruned-emaonly-Q4_0.gguf")!,
            approxRuntimeGB: 3.0
        ),
        // Sourced from Civitai rather than Hugging Face — verified before listing:
        //  - Checkpoint (not a LoRA/embedding), base model "SD 1.5 Hyper" (standard SD 1.5
        //    architecture, same UNet/CLIP/VAE shapes the app already runs).
        //  - Model-level flags: nsfw=false, poi=false, minor=false, sfwOnly=false — Civitai's
        //    own moderation classifies it general-audience. It is still an uncensored base
        //    checkpoint like the SD 1.5 entry above, so the app's own ContentSafety /
        //    ImageSafetyAnalyzer layers are what actually bound output, not this flag.
        //  - File passed Civitai's pickle and virus scans ("No Pickle imports" — pure tensor
        //    data, nothing executable).
        //  - Byte size (2,132,625,894) confirmed directly from the CDN's `Content-Range`
        //    header, not just the catalog API's rounded `sizeKB` — matches exactly.
        //  - `fileNameOverride` is required here: Civitai's download endpoint is
        //    `/api/download/models/{versionId}`, which carries no filename of its own — the
        //    real name only ever appears in the response's `Content-Disposition` header. See
        //    the doc comment on `fileNameOverride` above.
        CatalogModel(
            id: "realistic-vision-v6-b1-hyper-vae",
            kind: .diffusion,
            displayName: "Realistic Vision V6.0 B1",
            publisher: "SG_161222 (Civitai)",
            byteSize: 2_132_625_894,
            parameterCount: "860M UNet",
            quantization: "FP16 (pruned)",
            license: "Civitai model license — credit required",
            attribution: "Realistic Vision V6.0 B1 by SG_161222 (civitai.com/models/4201). Credit to the creator is required by the model's license. Commercial use of generated images is permitted; the checkpoint file itself may not be sold or repackaged under a different license.",
            summary: "Photorealistic portraits and scenes. One of the most widely used SD 1.5 checkpoints on Civitai, tuned for good results in as few as 4–6 steps and with a VAE already baked in.",
            url: URL(string: "https://civitai.com/api/download/models/501240?fileId=418901")!,
            approxRuntimeGB: 3.5,
            fileNameOverride: "realisticVisionV60B1_v51HyperVAE.safetensors"
        )
    ]

    static var all: [CatalogModel] { chatModels + diffusionModels }

    static var recommended: CatalogModel { chatModels[0] }

    static func models(for kind: ModelKind) -> [CatalogModel] {
        kind == .chat ? chatModels : diffusionModels
    }

    static func model(withFileName name: String) -> CatalogModel? {
        all.first { $0.fileName == name }
    }
}

// MARK: - Download manager

/// Downloads catalog models into `AppFiles.models`.
///
/// Uses a background `URLSession` so a multi-gigabyte transfer survives the user leaving the
/// app — on a phone, a one-gigabyte download that dies the moment the screen locks is not a
/// feature anyone can actually use. The paired `AppDelegate` hook below is what lets iOS wake
/// the app to finish the handoff.
@MainActor
final class ModelDownloadManager: NSObject, ObservableObject {

    static let shared = ModelDownloadManager()

    struct Progress {
        var modelID: String
        var kind: ModelKind
        var fractionCompleted: Double
        var bytesWritten: Int64
        var totalBytes: Int64

        var writtenDescription: String {
            ByteCountFormatter.string(fromByteCount: bytesWritten, countStyle: .file)
        }
        var totalDescription: String {
            ByteCountFormatter.string(fromByteCount: totalBytes, countStyle: .file)
        }
    }

    @Published private(set) var active: Progress?
    @Published private(set) var lastError: String?
    /// Set when a download finishes so the UI can advance without polling the filesystem.
    @Published private(set) var lastCompletedModelID: String?

    /// Off by default. A ~1 GB download on a metered plan is not something to start on the
    /// user's behalf without an explicit opt-in.
    @Published var allowsCellularDownload: Bool = UserDefaults.standard.bool(forKey: "allowsCellularDownload") {
        didSet { UserDefaults.standard.set(allowsCellularDownload, forKey: "allowsCellularDownload") }
    }

    /// Populated by the app delegate when iOS relaunches the app to hand back a finished
    /// background transfer, and called once the session reports it has drained its queue.
    var backgroundCompletionHandler: (() -> Void)?

    private var session: URLSession!
    private var currentTask: URLSessionDownloadTask?
    private var currentModel: CatalogModel?

    private override init() {
        super.init()
        let configuration = URLSessionConfiguration.background(withIdentifier: "com.DDT.DarkAI.modelDownloads")
        configuration.allowsCellularAccess = true   // gated per-task instead, see `download(_:)`
        configuration.isDiscretionary = false
        configuration.sessionSendsLaunchEvents = true
        configuration.waitsForConnectivity = true
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }

    var isDownloading: Bool { active != nil }

    /// Which catalog section owns the in-flight download, so only that section renders the
    /// progress card.
    var activeKind: ModelKind? { active?.kind }

    /// Chat weights and diffusion checkpoints are both GGUF but are consumed by different
    /// engines and listed by different screens, so they have to land in different directories.
    static func installDirectory(for kind: ModelKind) -> URL {
        kind == .chat ? AppFiles.models : AppFiles.diffusionModels
    }

    func isInstalled(_ model: CatalogModel) -> Bool {
        let path = Self.installDirectory(for: model.kind)
            .appendingPathComponent(model.fileName).path
        return FileManager.default.fileExists(atPath: path)
    }

    // MARK: Actions

    func download(_ model: CatalogModel) {
        guard active == nil else { return }
        lastError = nil
        lastCompletedModelID = nil

        // Disk pre-flight. Failing here with a clear number beats failing 900 MB in with
        // "The operation couldn't be completed."
        let requiredGB = model.sizeGB + 0.5
        let availableGB = AppFiles.availableDiskGB()
        guard availableGB > requiredGB else {
            lastError = String(
                format: "Not enough storage. %@ needs about %.1f GB free and this device has %.1f GB.",
                model.displayName, requiredGB, availableGB
            )
            return
        }

        AppFiles.prepare()

        var request = URLRequest(url: model.url)
        request.allowsCellularAccess = allowsCellularDownload
        request.allowsExpensiveNetworkAccess = allowsCellularDownload
        request.timeoutInterval = 60

        let task = session.downloadTask(with: request)
        task.countOfBytesClientExpectsToReceive = model.byteSize

        currentTask = task
        currentModel = model
        active = Progress(modelID: model.id, kind: model.kind, fractionCompleted: 0, bytesWritten: 0, totalBytes: model.byteSize)

        LogManager.shared.log("ModelDownload: starting \(model.displayName) (\(model.sizeDescription))")
        task.resume()
    }

    func cancel() {
        currentTask?.cancel()
        currentTask = nil
        currentModel = nil
        active = nil
        LogManager.shared.log("ModelDownload: cancelled by user")
    }

    func clearError() { lastError = nil }

    // MARK: Verification

    /// Confirms the bytes on disk are the model we asked for before it is ever handed to
    /// llama.cpp. A truncated or redirected download is otherwise indistinguishable from a
    /// corrupt model, and llama.cpp's failure mode for that is a crash rather than an error.
    private func verify(fileAt url: URL, against model: CatalogModel) throws {
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
        guard size == model.byteSize else {
            throw NSError(domain: "ModelDownload", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "The downloaded file is \(ByteCountFormatter.string(fromByteCount: size, countStyle: .file)) but should be \(model.sizeDescription). The download was incomplete."
            ])
        }
        // Header sanity, format-aware. Cheap, and catches an HTML error page saved under the
        // expected filename — which is exactly what a captive-portal Wi-Fi network produces —
        // regardless of which of the app's supported formats this particular model is in.
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let corruptError = NSError(domain: "ModelDownload", code: 2, userInfo: [
            NSLocalizedDescriptionKey: "The downloaded file is not a valid \(model.kind == .diffusion ? "diffusion" : "GGUF") model. Check your network connection and try again."
        ])
        switch (model.fileName as NSString).pathExtension.lowercased() {
        case "safetensors":
            // Layout: an 8-byte little-endian header length, then that many bytes of JSON
            // tensor metadata. Confirming the header actually parses as JSON is the safetensors
            // equivalent of the GGUF magic-byte check below — safetensors has no fixed magic
            // bytes of its own to check instead.
            guard let lenData = try handle.read(upToCount: 8), lenData.count == 8 else { throw corruptError }
            let headerLen = lenData.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
            guard headerLen > 0, headerLen < 100_000_000,
                  let headerBytes = try handle.read(upToCount: Int(headerLen)),
                  (try? JSONSerialization.jsonObject(with: headerBytes)) != nil else {
                throw corruptError
            }
        default:
            // GGUF magic bytes — covers both chat models and the GGUF-format diffusion entry.
            guard let magic = try handle.read(upToCount: 4), magic == Data("GGUF".utf8) else {
                throw corruptError
            }
        }
    }
}

// MARK: - URLSessionDownloadDelegate

extension ModelDownloadManager: URLSessionDownloadDelegate {

    nonisolated func urlSession(_ session: URLSession,
                                downloadTask: URLSessionDownloadTask,
                                didWriteData bytesWritten: Int64,
                                totalBytesWritten: Int64,
                                totalBytesExpectedToWrite: Int64) {
        Task { @MainActor in
            guard var progress = self.active else { return }
            // `totalBytesExpectedToWrite` is -1 when the server omits Content-Length; the
            // catalog's known size is the better denominator in that case.
            let total = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : progress.totalBytes
            progress.bytesWritten = totalBytesWritten
            progress.totalBytes = total
            progress.fractionCompleted = total > 0 ? min(1.0, Double(totalBytesWritten) / Double(total)) : 0
            self.active = progress
        }
    }

    nonisolated func urlSession(_ session: URLSession,
                                downloadTask: URLSessionDownloadTask,
                                didFinishDownloadingTo location: URL) {
        // This runs on a background queue and `location` is deleted the moment it returns, so
        // the move has to happen synchronously here rather than inside a hop to the main actor.
        let temporaryCopy = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        do {
            try FileManager.default.moveItem(at: location, to: temporaryCopy)
        } catch {
            Task { @MainActor in self.finish(with: error) }
            return
        }

        Task { @MainActor in
            guard let model = self.currentModel else {
                try? FileManager.default.removeItem(at: temporaryCopy)
                return
            }
            do {
                try self.verify(fileAt: temporaryCopy, against: model)

                let installDirectory = Self.installDirectory(for: model.kind)
                AppFiles.createIfNeeded(installDirectory)
                let destination = installDirectory.appendingPathComponent(model.fileName)
                if FileManager.default.fileExists(atPath: destination.path) {
                    try FileManager.default.removeItem(at: destination)
                }
                try FileManager.default.moveItem(at: temporaryCopy, to: destination)
                AppFiles.excludeFromBackup(destination)

                LogManager.shared.log("ModelDownload: installed \(model.fileName)")
                self.lastCompletedModelID = model.id
                self.finish(with: nil)
            } catch {
                try? FileManager.default.removeItem(at: temporaryCopy)
                self.finish(with: error)
            }
        }
    }

    nonisolated func urlSession(_ session: URLSession,
                                task: URLSessionTask,
                                didCompleteWithError error: Error?) {
        guard let error else { return }   // success is handled in didFinishDownloadingTo
        let nsError = error as NSError
        // A user-initiated cancel is not a failure worth surfacing as one.
        guard nsError.code != NSURLErrorCancelled else { return }
        Task { @MainActor in self.finish(with: error) }
    }

    nonisolated func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        Task { @MainActor in
            self.backgroundCompletionHandler?()
            self.backgroundCompletionHandler = nil
        }
    }

    @MainActor
    private func finish(with error: Error?) {
        if let error {
            let nsError = error as NSError
            if nsError.domain == NSURLErrorDomain,
               nsError.code == NSURLErrorDataNotAllowed || nsError.code == NSURLErrorInternationalRoamingOff {
                lastError = "This download needs Wi-Fi. Connect to Wi-Fi, or turn on cellular downloads first."
            } else {
                lastError = error.localizedDescription
            }
            LogManager.shared.log("ModelDownload: failed — \(error.localizedDescription)")
        }
        active = nil
        currentTask = nil
        currentModel = nil
    }
}
