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
    /// Physical RAM below which this model is not worth recommending.
    ///
    /// Derived by running each model's real geometry — layer count, KV head count, head dims and
    /// trained context, all read from its GGUF header — through `LlamaRunner.planOffload` and
    /// `safeContextTokens` at each iPhone memory tier, and taking the lowest tier that still
    /// yields a usable context window rather than merely loading. A model that loads with 512
    /// tokens is not one to recommend.
    let minimumRAMGB: Double
    /// The earliest iPhone at `minimumRAMGB`, for a human-readable tag. iOS 17 is the app's
    /// floor, so the oldest hardware in scope is the 4 GB generation (iPhone 11, SE 3rd gen).
    let minimumDevice: String
    /// Overrides `fileName` when `url` doesn't end in the real filename. Hugging Face URLs are
    /// self-describing (`.../resolve/main/some-model.gguf`), but Civitai's download endpoint is
    /// `/api/download/models/{versionId}` — the actual filename only ever appears in the
    /// response's `Content-Disposition` header, never in the URL itself. Without this, a
    /// Civitai-sourced model would install under a literal numeric ID with no extension: hidden
    /// from every extension-filtered model list in the app and unloadable.
    let fileNameOverride: String?

    init(id: String, kind: ModelKind, displayName: String, publisher: String, byteSize: Int64,
         parameterCount: String, quantization: String, license: String, attribution: String?,
         summary: String, url: URL, approxRuntimeGB: Double,
         minimumRAMGB: Double, minimumDevice: String, fileNameOverride: String? = nil) {
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
        self.minimumRAMGB = minimumRAMGB
        self.minimumDevice = minimumDevice
        self.fileNameOverride = fileNameOverride
    }

    var fileName: String { fileNameOverride ?? url.lastPathComponent }
    var sizeGB: Double { Double(byteSize) / (1024 * 1024 * 1024) }
    var sizeDescription: String { String(format: "%.2f GB", sizeGB) }
}

enum ModelCatalog {

    /// Curated starter models. All are instruction-tuned and all are redistributable under a
    /// license that permits this use.
    ///
    /// Sizes are the exact byte counts served by the host — they are checked after download, so
    /// if a host ever republishes a file at a different size the verification fails loudly
    /// instead of leaving a truncated model that only misbehaves at inference time.
    ///
    /// One entry is a mixture-of-experts model, and it is listed on its merits rather than for
    /// any memory advantage. There was supposed to be one: pin the routed experts to mmap and a
    /// sparse model needs only its dense remainder resident. That does not work on Metal —
    /// llama.cpp gives each device a single mapping spanning its first tensor to its last, and
    /// because MoE files interleave expert and attention tensors, pinning experts still forces
    /// the whole file to be mapped (see `LlamaRunner.planOffload` for the measurements). So a
    /// sparse model here costs what a dense one of the same size costs, and is judged the same
    /// way. What it still offers is quality per *compute*: only a fraction of its weights run
    /// for each token, so a 7B-class model answers at roughly 1B-class speed.
    ///
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
            approxRuntimeGB: 1.6,
            minimumRAMGB: 6.0,
            minimumDevice: "minimum iPhone 12 Pro / 14 or newer"
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
            approxRuntimeGB: 1.9,
            minimumRAMGB: 6.0,
            minimumDevice: "minimum iPhone 12 Pro / 14 or newer"
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
            approxRuntimeGB: 2.0,
            minimumRAMGB: 6.0,
            minimumDevice: "minimum iPhone 12 Pro / 14 or newer"
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
            summary: "Strong at reasoning and long answers, and the best balance of quality against download size here. Needs a device with plenty of memory.",
            url: URL(string: "https://huggingface.co/hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF/resolve/main/llama-3.2-3b-instruct-q4_k_m.gguf")!,
            approxRuntimeGB: 3.4,
            minimumRAMGB: 6.0,
            minimumDevice: "minimum iPhone 13 Pro / 14 or newer"
        ),

        CatalogModel(
            id: "olmoe-1b-7b-0924-instruct-q4km",
            kind: .chat,
            displayName: "OLMoE 1B-7B Instruct",
            publisher: "Ai2",
            byteSize: 4_213_512_672,
            parameterCount: "7B total, 1B active",
            quantization: "Q4_K_M",
            license: "Apache 2.0",
            attribution: nil,
            // 16 blocks, 64 experts each, 8 active per token. Resident cost is the whole file
            // like any other model, so this is sized as such — the earlier 1.6 GB figure assumed
            // an expert-pinning saving that Metal does not permit.
            summary: "A 7B model that only uses about an eighth of itself for each word, so it answers about as fast as a 1B while knowing more.",
            url: URL(string: "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf")!,
            approxRuntimeGB: 4.6,
            minimumRAMGB: 8.0,
            minimumDevice: " minimum iPhone 15 Pro / 16 or newer"
        ),
        CatalogModel(
            id: "lfm2.5-8b-a1b-instruct-q4km",
            kind: .chat,
            displayName: "LFM2.5 8B-A1B",
            publisher: "Liquid AI",
            byteSize: 5_155_564_768,
            parameterCount: "8.3B total, 1.5B active",
            quantization: "Q4_K_M",
            license: "LFM Open License v1.0",
            attribution: "LFM2.5-8B-A1B by Liquid AI, licensed under the LFM Open License v1.0.",
            // Replaces the old dense Llama 3 8B entry, which a background personality-analysis
            // call could race for the model — see the fix in LlamaRunner.generateStream — and
            // which, being dense, offered no way to avoid needing its full weight size resident
            // in memory. This model routes each token through only 4 of 32 experts per
            // mixture-of-experts block, so replies are markedly faster than a dense model this
            // size. That speed does NOT come with a smaller memory footprint, though: per-expert
            // GPU/CPU splitting was tried for a mixture-of-experts model on this engine and found
            // to not work reliably on Metal (see the comment on expert pinning in
            // `LLMManager.planOffload`), so this is still sized as needing its full file size
            // resident, exactly like every other model here.
            summary: "A mixture-of-experts model that only routes each word through a fraction of its weights; the power of an 8B model without the memory demand but replies come noticeably slower than other models.",
            url: URL(string: "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/main/LFM2.5-8B-A1B-Q4_K_M.gguf")!,
            approxRuntimeGB: 5.5,
            minimumRAMGB: 12.0,
            minimumDevice: "iPhone 17 Pro or newer"
        )
    ]

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
            approxRuntimeGB: 3.0,
            minimumRAMGB: 6.0,
            minimumDevice: "minimum iPhone 12 Pro / 14 or newer"
        ),
        
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
            minimumRAMGB: 8.0,
            minimumDevice: "minimum iPhone 15 Pro / 16 or newer",
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

    static func model(withID id: String) -> CatalogModel? {
        all.first { $0.id == id }
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
        /// True when this transfer picked up from a saved partial rather than starting over.
        var isResumed: Bool = false

        var writtenDescription: String {
            ByteCountFormatter.string(fromByteCount: bytesWritten, countStyle: .file)
        }
        var totalDescription: String {
            ByteCountFormatter.string(fromByteCount: totalBytes, countStyle: .file)
        }
    }

    /// Every transfer currently in flight, keyed by catalog model ID — downloads run
    /// concurrently rather than one at a time, so this replaces what used to be a single
    /// optional `active` slot.
    @Published private(set) var activeDownloads: [String: Progress] = [:]
    @Published private(set) var lastError: String?
    /// Set when a download finishes so the UI can advance without polling the filesystem.
    @Published private(set) var lastCompletedModelID: String?

    /// Catalog IDs with a partial transfer saved on disk that can be picked up where it stopped.
    ///
    /// Before this existed, *any* interruption — a dropped connection, the user tapping Cancel, iOS
    /// terminating the app, or an app update landing mid-transfer — threw away every byte and
    /// started the next attempt from zero. On a 2 GB checkpoint over a phone connection that is the
    /// difference between a download that eventually finishes and one that never does, and it is
    /// the most likely thing behind "I have to download the model again after every update".
    @Published private(set) var resumableModelIDs: Set<String> = []

    /// Off by default. A ~1 GB download on a metered plan is not something to start on the
    /// user's behalf without an explicit opt-in.
    @Published var allowsCellularDownload: Bool = UserDefaults.standard.bool(forKey: "allowsCellularDownload") {
        didSet { UserDefaults.standard.set(allowsCellularDownload, forKey: "allowsCellularDownload") }
    }

    /// Populated by the app delegate when iOS relaunches the app to hand back a finished
    /// background transfer, and called once the session reports it has drained its queue.
    var backgroundCompletionHandler: (() -> Void)?

    private var session: URLSession!
    /// The in-flight task for each model currently downloading, keyed by catalog model ID.
    private var tasksByModelID: [String: URLSessionDownloadTask] = [:]
    /// The reverse lookup — delegate callbacks only hand back a task, never the model it
    /// belongs to, so this is how they find out which download they're reporting on.
    private var modelsByTaskID: [Int: CatalogModel] = [:]
    /// Task identifiers whose task was created from saved resume data. Read by `finish(_:with:)`
    /// to tell a stale resume blob apart from an ordinary network failure.
    private var resumedTaskIDs: Set<Int> = []

    private override init() {
        super.init()
        let configuration = URLSessionConfiguration.background(withIdentifier: "com.DDT.DarkAI.modelDownloads")
        configuration.allowsCellularAccess = true   // gated per-task instead, see `download(_:)`
        configuration.isDiscretionary = false
        configuration.sessionSendsLaunchEvents = true
        configuration.waitsForConnectivity = true
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
        resumableModelIDs = Self.savedResumableModelIDs()
    }

    /// Whether anything at all is downloading — onboarding uses this to gate advancing past the
    /// model-setup step; it doesn't need to know which or how many.
    var isDownloading: Bool { !activeDownloads.isEmpty }

    /// Whether this specific model is one of the in-flight downloads.
    func isDownloading(_ model: CatalogModel) -> Bool { activeDownloads[model.id] != nil }

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
        guard activeDownloads[model.id] == nil else { return }
        lastError = nil
        lastCompletedModelID = nil

        // Disk pre-flight. Failing here with a clear number beats failing 900 MB in with
        // "The operation couldn't be completed." Downloads run concurrently now, so this has to
        // reserve space for what every other in-flight transfer still has left to write, not
        // just this one — otherwise three simultaneous downloads that each individually fit
        // could jointly overrun what's actually free.
        let remainingForOthersGB = activeDownloads.values.reduce(0.0) { partial, progress in
            partial + Double(max(0, progress.totalBytes - progress.bytesWritten)) / 1_073_741_824.0
        }
        let requiredGB = model.sizeGB + 0.5 + remainingForOthersGB
        let availableGB = AppFiles.availableDiskGB()
        guard availableGB > requiredGB else {
            lastError = activeDownloads.isEmpty
                ? String(format: "Not enough storage. %@ needs about %.1f GB free and this device has %.1f GB.",
                         model.displayName, requiredGB, availableGB)
                : String(format: "Not enough storage. %@ needs about %.1f GB free (accounting for %d other download%@ in progress) and this device has %.1f GB.",
                         model.displayName, requiredGB, activeDownloads.count, activeDownloads.count == 1 ? "" : "s", availableGB)
            return
        }

        AppFiles.prepare()
        start(model, resuming: Self.savedResumeData(for: model))
    }

    /// Creates and starts the task, from saved resume data when there is any.
    ///
    /// `resumeData` is opaque and can be rejected by the system — it goes stale if the server no
    /// longer supports the byte range, if the temporary file behind it has been reclaimed, or
    /// simply if too much time has passed. That failure arrives as an ordinary task error rather
    /// than a throw, so `finish(_:with:)` handles it by discarding the partial and saying the
    /// download restarted, instead of leaving the user stuck retrying a resume that can never work.
    private func start(_ model: CatalogModel, resuming resumeData: Data?) {
        let task: URLSessionDownloadTask
        if let resumeData {
            task = session.downloadTask(withResumeData: resumeData)
            LogManager.shared.log("ModelDownload: resuming \(model.displayName)")
        } else {
            var request = URLRequest(url: model.url)
            request.allowsCellularAccess = allowsCellularDownload
            request.allowsExpensiveNetworkAccess = allowsCellularDownload
            request.timeoutInterval = 60
            task = session.downloadTask(with: request)
            LogManager.shared.log("ModelDownload: starting \(model.displayName) (\(model.sizeDescription))")
        }
        task.countOfBytesClientExpectsToReceive = model.byteSize

        tasksByModelID[model.id] = task
        modelsByTaskID[task.taskIdentifier] = model
        if resumeData != nil { resumedTaskIDs.insert(task.taskIdentifier) }
        activeDownloads[model.id] = Progress(
            modelID: model.id,
            kind: model.kind,
            fractionCompleted: 0,
            bytesWritten: 0,
            totalBytes: model.byteSize,
            isResumed: resumeData != nil
        )
        task.resume()
    }

    /// Stops one transfer but keeps what has already been fetched, so tapping Get again picks up
    /// where this left off. Other in-flight downloads are untouched. Use `discardPartial(for:)`
    /// to throw the bytes away instead.
    func cancel(_ model: CatalogModel) {
        guard let task = tasksByModelID[model.id] else { return }
        // Goes through the singleton rather than capturing `self`: this callback is delivered on a
        // background queue and outlives the call, and reaching back through `shared` keeps that
        // free of a cross-actor capture instead of relying on one being tolerated.
        task.cancel { resumeData in
            Task { @MainActor in
                let manager = ModelDownloadManager.shared
                if let resumeData {
                    Self.saveResumeData(resumeData, for: model)
                    manager.resumableModelIDs.insert(model.id)
                    LogManager.shared.log("ModelDownload: paused \(model.displayName), \(ByteCountFormatter.string(fromByteCount: Int64(resumeData.count), countStyle: .file)) of resume state kept")
                } else {
                    LogManager.shared.log("ModelDownload: cancelled \(model.displayName) — the server didn't support resuming")
                }
            }
        }
        tasksByModelID.removeValue(forKey: model.id)
        modelsByTaskID.removeValue(forKey: task.taskIdentifier)
        resumedTaskIDs.remove(task.taskIdentifier)
        activeDownloads.removeValue(forKey: model.id)
    }

    /// Throws away a saved partial transfer. Offered next to the resume affordance so a user who
    /// changed their mind isn't stuck carrying a gigabyte of a model they no longer want.
    func discardPartial(for model: CatalogModel) {
        Self.deleteResumeData(for: model)
        resumableModelIDs.remove(model.id)
        LogManager.shared.log("ModelDownload: discarded partial download of \(model.displayName)")
    }

    func clearError() { lastError = nil }

    // MARK: Resume state

    /// Resume blobs live beside partial downloads, in the directory that already exists for
    /// exactly this purpose and is already excluded from backup.
    private static func resumeFileURL(for modelID: String) -> URL {
        AppFiles.downloadsInProgress.appendingPathComponent("\(modelID).resume")
    }

    private static func savedResumeData(for model: CatalogModel) -> Data? {
        try? Data(contentsOf: resumeFileURL(for: model.id))
    }

    private static func saveResumeData(_ data: Data, for model: CatalogModel) {
        AppFiles.createIfNeeded(AppFiles.downloadsInProgress)
        let url = resumeFileURL(for: model.id)
        try? data.write(to: url, options: .atomic)
        AppFiles.excludeFromBackup(url)
    }

    private static func deleteResumeData(for model: CatalogModel) {
        try? FileManager.default.removeItem(at: resumeFileURL(for: model.id))
    }

    private static func savedResumableModelIDs() -> Set<String> {
        let files = AppFiles.contents(of: AppFiles.downloadsInProgress, matchingExtensions: ["resume"])
        let known = Set(ModelCatalog.all.map(\.id))
        return Set(files.map { ($0.lastPathComponent as NSString).deletingPathExtension }.filter(known.contains))
    }

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
        let taskID = downloadTask.taskIdentifier
        Task { @MainActor in
            guard let model = self.modelsByTaskID[taskID],
                  var progress = self.activeDownloads[model.id] else { return }
            // `totalBytesExpectedToWrite` is -1 when the server omits Content-Length; the
            // catalog's known size is the better denominator in that case.
            let total = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : progress.totalBytes
            progress.bytesWritten = totalBytesWritten
            progress.totalBytes = total
            progress.fractionCompleted = total > 0 ? min(1.0, Double(totalBytesWritten) / Double(total)) : 0
            self.activeDownloads[model.id] = progress
        }
    }

    nonisolated func urlSession(_ session: URLSession,
                                downloadTask: URLSessionDownloadTask,
                                didFinishDownloadingTo location: URL) {
        let taskID = downloadTask.taskIdentifier
        // This runs on a background queue and `location` is deleted the moment it returns, so
        // the move has to happen synchronously here rather than inside a hop to the main actor.
        let temporaryCopy = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        do {
            try FileManager.default.moveItem(at: location, to: temporaryCopy)
        } catch {
            Task { @MainActor in
                guard let model = self.modelsByTaskID[taskID] else { return }
                self.finish(model, with: error)
            }
            return
        }

        Task { @MainActor in
            guard let model = self.modelsByTaskID[taskID] else {
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

                // The partial is worthless now, and the ledger entry is what lets the app tell the
                // user *which* model vanished if this device is ever restored without it.
                Self.deleteResumeData(for: model)
                self.resumableModelIDs.remove(model.id)
                ModelInventory.shared.record(
                    fileName: model.fileName,
                    kind: model.kind,
                    catalogID: model.id,
                    byteSize: model.byteSize
                )

                LogManager.shared.log("ModelDownload: installed \(model.fileName)")
                self.lastCompletedModelID = model.id
                self.finish(model, with: nil)
            } catch {
                try? FileManager.default.removeItem(at: temporaryCopy)
                self.finish(model, with: error)
            }
        }
    }

    nonisolated func urlSession(_ session: URLSession,
                                task: URLSessionTask,
                                didCompleteWithError error: Error?) {
        guard let error else { return }   // success is handled in didFinishDownloadingTo
        let nsError = error as NSError
        // A user-initiated cancel is not a failure worth surfacing as one, and its resume data
        // arrives through `cancel(byProducingResumeData:)` rather than here.
        guard nsError.code != NSURLErrorCancelled else { return }

        // The bytes already fetched are salvageable whenever the system hands back resume state —
        // a dropped connection nine-tenths of the way through a 2 GB checkpoint should cost the
        // remaining tenth, not the whole thing.
        let resumeData = nsError.userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        let taskID = task.taskIdentifier

        Task { @MainActor in
            guard let model = self.modelsByTaskID[taskID] else { return }
            if let resumeData {
                Self.saveResumeData(resumeData, for: model)
                self.resumableModelIDs.insert(model.id)
            }
            self.finish(model, with: error, producedResumeData: resumeData != nil)
        }
    }

    nonisolated func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        Task { @MainActor in
            self.backgroundCompletionHandler?()
            self.backgroundCompletionHandler = nil
        }
    }

    @MainActor
    private func finish(_ model: CatalogModel, with error: Error?, producedResumeData: Bool = false) {
        let bytesWritten = activeDownloads[model.id]?.bytesWritten ?? 0
        let taskID = tasksByModelID[model.id]?.taskIdentifier
        let wasResumed = taskID.map { resumedTaskIDs.contains($0) } ?? false

        activeDownloads.removeValue(forKey: model.id)
        tasksByModelID.removeValue(forKey: model.id)
        if let taskID {
            modelsByTaskID.removeValue(forKey: taskID)
            resumedTaskIDs.remove(taskID)
        }

        guard let error else { return }

        // Stale resume state: the task was built from a saved partial, produced nothing, and the
        // system declined to give any back. Retrying would fail identically every time, so drop it
        // and start over once rather than stranding the model behind a permanently broken resume.
        if wasResumed, !producedResumeData, bytesWritten == 0 {
            Self.deleteResumeData(for: model)
            resumableModelIDs.remove(model.id)
            LogManager.shared.log("ModelDownload: saved partial for \(model.displayName) was no longer usable — restarting from the beginning")
            start(model, resuming: nil)
            return
        }

        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain,
           nsError.code == NSURLErrorDataNotAllowed || nsError.code == NSURLErrorInternationalRoamingOff {
            lastError = "This download needs Wi-Fi. Connect to Wi-Fi, or turn on cellular downloads first."
        } else if producedResumeData {
            lastError = "\(error.localizedDescription) Your progress was kept — tap Resume to carry on from where it stopped."
        } else {
            lastError = error.localizedDescription
        }
        LogManager.shared.log("ModelDownload: failed — \(model.displayName) — \(error.localizedDescription)")
    }
}
