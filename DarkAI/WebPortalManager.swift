import Combine
import Foundation
import FlyingFox

/// Serves DarkAI's chat interface to any browser on the same local network — a phone-only app
/// that's inconvenient to type into from a laptop otherwise. Off by default: this is a deliberate,
/// narrow exception to the app's no-server design, exactly like `WebSearchManager`'s internet
/// toggle, and nothing here listens on any socket until the user turns it on in Settings.
///
/// Scope, on purpose: v1 is chat only — send a message, get a streamed reply, switch between
/// conversations. It reuses the exact same generation pipeline `ContentView.sendMessage`/
/// `generateTextResponse` drive (content safety, RAG context, memories, personality layering —
/// see `ChatOrchestration`) rather than a second, independently-maintained copy of that logic.
/// Image generation and the web-search-offer flow are not wired up yet; a message that would
/// trigger either gets an honest "not available here yet" reply instead of silently doing the
/// wrong thing.
///
/// Security: every request past the login screen requires the PIN shown in Settings — this is
/// full read/write access to someone's conversations, so it is not left open on the network with
/// no gate. A browser that unlocks with the PIN is then "remembered" via a persistent, `HttpOnly`
/// cookie (`darkai_device`) so it doesn't need the PIN again on every visit — see `WebPortalDevice`
/// below. Every remembered browser is listed in Settings, where it can be renamed or removed;
/// removing one (or regenerating the PIN, which clears all of them at once) means that browser
/// needs the PIN again next time. The cookie never leaves the local network — the server it's
/// scoped to only exists while the portal is running on this device.
@MainActor
final class WebPortalManager: ObservableObject {

    static let shared = WebPortalManager()

    @Published var isEnabled: Bool {
        didSet {
            UserDefaults.standard.set(isEnabled, forKey: Self.enabledKey)
            updateRunningState()
        }
    }
    @Published private(set) var isRunning = false
    @Published private(set) var pin: String
    /// e.g. "http://192.168.1.42:8642" — `nil` whenever the server isn't actually listening.
    @Published private(set) var addressDescription: String?
    @Published private(set) var lastError: String?
    /// Every browser that has ever unlocked with the PIN and hasn't since been removed. Persisted
    /// (see `saveDevices`) so a phone relaunch doesn't silently forget every remembered browser —
    /// only removing a device, or regenerating the PIN, does that.
    @Published private(set) var devices: [WebPortalDevice] = []

    private static let enabledKey = "webPortalEnabled"
    private static let pinKey = "webPortalPIN"
    private static let devicesKey = "webPortalDevices"
    private static let deviceCookieName = "darkai_device"
    static let port: UInt16 = 8642

    private var server: HTTPServer?
    private var runTask: Task<Void, Never>?
    /// Every currently-open, authenticated WebSocket connection, keyed by a per-connection id —
    /// how `broadcast(_:)` pushes an unsolicited update (model load progress) to every connected
    /// browser at once, not just the one that triggered it. Registered/unregistered by
    /// `WebPortalSocketHandler` as connections open and close.
    private var connections: [UUID: (String) -> Void] = [:]
    private var loadStateCancellable: AnyCancellable?

    /// Consecutive wrong-PIN submissions since the last success or lockout. The PIN is only six
    /// digits (see `generatePin`), so without some throttle a LAN attacker could simply try all
    /// one million combinations against `POST /api/login` — this is what makes that impractical.
    private var failedLoginAttempts = 0
    /// Set once `failedLoginAttempts` crosses the threshold; every login attempt is refused
    /// outright until this passes, regardless of the PIN submitted.
    private var loginLockoutUntil: Date?
    private static let maxFailedLoginAttempts = 5
    private static let loginLockoutSeconds: TimeInterval = 30

    private var conversationManager: ConversationManager?
    private var llmManager: LLMManager?
    private var ragManager: RAGManager?
    private var memoryManager: MemoryManager?
    private var personalityManager: PersonalityManager?
    private var feedbackManager: FeedbackManager?
    private var webSearchManager: WebSearchManager?
    private var isConfigured = false

    private init() {
        let storedPin = UserDefaults.standard.string(forKey: Self.pinKey) ?? Self.generatePin()
        pin = storedPin
        UserDefaults.standard.set(storedPin, forKey: Self.pinKey)
        isEnabled = UserDefaults.standard.bool(forKey: Self.enabledKey)
        if let data = UserDefaults.standard.data(forKey: Self.devicesKey),
           let decoded = try? JSONDecoder().decode([WebPortalDevice].self, from: data) {
            devices = decoded
        }
    }

    /// Called once from `ContentView` (which already owns every manager instance below as
    /// `@StateObject`s) so the portal operates on the exact same conversations/RAG corpus/memory
    /// store the native UI does, rather than a second, disconnected set of state.
    func configure(
        conversationManager: ConversationManager,
        llmManager: LLMManager,
        ragManager: RAGManager,
        memoryManager: MemoryManager,
        personalityManager: PersonalityManager,
        feedbackManager: FeedbackManager,
        webSearchManager: WebSearchManager
    ) {
        self.conversationManager = conversationManager
        self.llmManager = llmManager
        self.ragManager = ragManager
        self.memoryManager = memoryManager
        self.personalityManager = personalityManager
        self.feedbackManager = feedbackManager
        self.webSearchManager = webSearchManager
        isConfigured = true
        updateRunningState()

        // Model loading happens on `LLMManager` regardless of whether it was triggered from the
        // phone or from a browser — this is the one subscription that keeps every connected
        // browser's Settings panel showing real progress either way, rather than only reacting to
        // loads the Web Portal itself started. See `broadcastLoadState`.
        loadStateCancellable = llmManager.$loadState
            .receive(on: DispatchQueue.main)
            .sink { [weak self] state in
                self?.broadcastLoadState(state)
            }
    }

    /// Fires on every `LLMManager.loadState` change. Intermediate `.loading` ticks get a small,
    /// cheap `modelLoadState` push so a progress bar can track them live; the settled states
    /// additionally get a full `settings` broadcast, since that's when `activeModelName` and each
    /// model row's `isLoaded`/`safety` actually change.
    private func broadcastLoadState(_ state: ModelLoadState) {
        switch state {
        case .loading(let progress, let status):
            broadcast(["type": "modelLoadState", "isLoading": true, "progress": progress, "message": status])
        case .loaded, .failed, .unloaded:
            broadcast(["type": "modelLoadState", "isLoading": false])
            broadcast(settingsSnapshot())
        }
    }

    private func broadcast(_ dict: [String: Any]) {
        guard !connections.isEmpty else { return }
        let payload = encode(dict)
        for respond in connections.values { respond(payload) }
    }

    fileprivate func registerConnection(_ id: UUID, respond: @escaping (String) -> Void) {
        connections[id] = respond
    }

    fileprivate func unregisterConnection(_ id: UUID) {
        connections.removeValue(forKey: id)
    }

    /// Invalidates every remembered browser at once — the panic-button reset. Anyone, including
    /// browsers that were previously remembered, needs the new PIN to get back in.
    func regeneratePin() {
        pin = Self.generatePin()
        UserDefaults.standard.set(pin, forKey: Self.pinKey)
        devices.removeAll()
        saveDevices()
    }

    /// Gives a remembered device a user-chosen name — the on-device User-Agent guess
    /// (`friendlyName(forUserAgent:)`) is only ever a starting point.
    func renameDevice(id: String, name: String) {
        guard let index = devices.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        devices[index].name = trimmed
        saveDevices()
    }

    /// Revokes one browser's access. Its cookie stops working immediately — the next request from
    /// that browser finds no matching device and falls back to the PIN screen.
    func removeDevice(id: String) {
        devices.removeAll { $0.id == id }
        saveDevices()
    }

    private func saveDevices() {
        guard let data = try? JSONEncoder().encode(devices) else { return }
        UserDefaults.standard.set(data, forKey: Self.devicesKey)
    }

    private static func generatePin() -> String {
        String(format: "%06d", Int.random(in: 0..<1_000_000))
    }

    /// Best-effort, not exhaustive — User-Agent sniffing never is. It only has to be a reasonable
    /// starting guess; `renameDevice` is the actual answer for anyone who wants a precise name.
    private static func friendlyName(forUserAgent userAgent: String?) -> String {
        guard let ua = userAgent, !ua.isEmpty else { return "Unknown device" }

        let platform: String
        if ua.contains("iPhone") { platform = "iPhone" }
        else if ua.contains("iPad") { platform = "iPad" }
        else if ua.contains("Macintosh") { platform = "Mac" }
        else if ua.contains("Android") { platform = "Android" }
        else if ua.contains("Windows") { platform = "Windows PC" }
        else { platform = "device" }

        let browser: String
        if ua.contains("CriOS") { browser = "Chrome" }
        else if ua.contains("FxiOS") || ua.contains("Firefox") { browser = "Firefox" }
        else if ua.contains("EdgiOS") || ua.contains("Edg/") { browser = "Edge" }
        else if ua.contains("Chrome") { browser = "Chrome" }
        else if ua.contains("Safari") { browser = "Safari" }
        else { browser = "Browser" }

        return "\(browser) on \(platform)"
    }

    /// Parses a raw `Cookie` header ("a=1; b=2") for one named value — FlyingFox has no cookie
    /// jar of its own, so this is the whole implementation.
    private static func cookieValue(named name: String, from cookieHeader: String?) -> String? {
        guard let cookieHeader else { return nil }
        for pair in cookieHeader.split(separator: ";") {
            let parts = pair.split(separator: "=", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
            if parts.count == 2, parts[0] == name { return String(parts[1]) }
        }
        return nil
    }

    private func updateRunningState() {
        if isEnabled, isConfigured {
            start()
        } else {
            stop()
        }
    }

    // MARK: - Server lifecycle

    private func start() {
        guard server == nil else { return }
        lastError = nil

        let httpServer = HTTPServer(port: Self.port)
        server = httpServer

        Task { await registerRoutes(on: httpServer) }

        runTask = Task { [weak self] in
            do {
                try await httpServer.run()
            } catch {
                guard let self else { return }
                self.lastError = "Web Portal stopped: \(error.localizedDescription)"
                self.isRunning = false
                self.server = nil
                self.addressDescription = nil
            }
        }

        isRunning = true
        addressDescription = Self.localIPAddress().map { "http://\($0):\(Self.port)" }
        if addressDescription == nil {
            lastError = "Couldn't determine this device's local network address — make sure Wi-Fi is connected."
        }
    }

    private func stop() {
        guard let server else { return }
        let task = runTask
        Task {
            await server.stop()
            task?.cancel()
        }
        self.server = nil
        self.runTask = nil
        isRunning = false
        addressDescription = nil
        // Deliberately NOT clearing `devices` here — turning the portal off and back on again
        // should not force every remembered browser to re-enter the PIN. Only removing a device,
        // or regenerating the PIN, does that (see those methods).
    }

    private func registerRoutes(on server: HTTPServer) async {
        await server.appendRoute("GET /") { _ in
            HTTPResponse(
                statusCode: .ok,
                headers: [.contentType: "text/html; charset=utf-8"],
                body: Data(WebPortalAssets.indexHTML.utf8)
            )
        }

        await server.appendRoute("POST /api/login") { [weak self] request in
            guard let self else { return HTTPResponse(statusCode: .ok, body: Data("{}".utf8)) }
            return await self.handleLogin(request)
        }

        await server.appendRoute("GET /api/ws") { [weak self] request in
            guard let self else { return HTTPResponse(statusCode: .ok, body: Data()) }
            let token = await Self.cookieValue(named: Self.deviceCookieName, from: request.headers[.cookie])
            let isAuthenticated = await self.checkAndTouchDevice(token)
            let wsHandler = WebSocketHTTPHandler(
                handler: MessageFrameWSHandler(handler: WebPortalSocketHandler(portal: self, initiallyAuthenticated: isAuthenticated))
            )
            return try await wsHandler.handleRequest(request)
        }
    }

    /// Verifies the submitted PIN and, on success, mints a new remembered device and sets the
    /// cookie that remembers it — a real HTTP response, not a WebSocket message, because
    /// `Set-Cookie` is an HTTP response header with no WebSocket equivalent. The client is
    /// expected to open (or reopen) its WebSocket connection right after this succeeds, so the new
    /// cookie is present on that connection's own upgrade request — see `checkAndTouchDevice`.
    private func handleLogin(_ request: HTTPRequest) async -> HTTPResponse {
        if let lockoutUntil = loginLockoutUntil {
            guard Date() >= lockoutUntil else {
                return HTTPResponse(statusCode: .ok, headers: [.contentType: "application/json"], body: Data(encode(["success": false, "message": "Too many incorrect attempts. Try again in a moment."]).utf8))
            }
            loginLockoutUntil = nil
            failedLoginAttempts = 0
        }

        guard let bodyData = try? await request.bodyData,
              let object = try? JSONSerialization.jsonObject(with: bodyData) as? [String: Any],
              let submittedPin = object["pin"] as? String,
              submittedPin == pin else {
            failedLoginAttempts += 1
            if failedLoginAttempts >= Self.maxFailedLoginAttempts {
                loginLockoutUntil = Date().addingTimeInterval(Self.loginLockoutSeconds)
                return HTTPResponse(statusCode: .ok, headers: [.contentType: "application/json"], body: Data(encode(["success": false, "message": "Too many incorrect attempts. Try again in a moment."]).utf8))
            }
            return HTTPResponse(statusCode: .ok, headers: [.contentType: "application/json"], body: Data(encode(["success": false]).utf8))
        }

        failedLoginAttempts = 0
        loginLockoutUntil = nil

        let userAgent = request.headers[HTTPHeader("User-Agent")]
        let now = Date()
        let device = WebPortalDevice(
            id: UUID().uuidString,
            name: Self.friendlyName(forUserAgent: userAgent),
            userAgent: userAgent,
            firstSeen: now,
            lastSeen: now
        )
        devices.append(device)
        saveDevices()

        var response = HTTPResponse(statusCode: .ok, headers: [.contentType: "application/json"], body: Data(encode(["success": true]).utf8))
        // A year, not a session cookie — the whole point is not re-asking for the PIN on every
        // visit. `removeDevice`/`regeneratePin` are the actual revocation path, not expiry.
        response.headers[.setCookie] = "\(Self.deviceCookieName)=\(device.id); Path=/; Max-Age=31536000; SameSite=Strict; HttpOnly"
        return response
    }

    /// The `GET /api/ws` route's auth check: does this connection's cookie match a device that's
    /// still remembered? Also bumps that device's `lastSeen` on every successful reconnect, so the
    /// Settings list reflects actual recent use rather than just the day it was first unlocked.
    private func checkAndTouchDevice(_ token: String?) -> Bool {
        guard let token, let index = devices.firstIndex(where: { $0.id == token }) else { return false }
        devices[index].lastSeen = Date()
        saveDevices()
        return true
    }

    // MARK: - Local address

    /// The device's own local Wi-Fi IP (e.g. "192.168.1.42") — what a browser on another device
    /// needs to type in, since `HTTPServer(port:)` binds every interface, not one specific
    /// address. Walks `en0` first (Wi-Fi on every iPhone), falling back to the first non-loopback
    /// IPv4 interface if that's somehow not it.
    private static func localIPAddress() -> String? {
        var address: String?
        var ifaddrPointer: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddrPointer) == 0, let firstAddr = ifaddrPointer else { return nil }
        defer { freeifaddrs(ifaddrPointer) }

        var fallback: String?
        for cursor in sequence(first: firstAddr, next: { $0.pointee.ifa_next }) {
            let flags = Int32(cursor.pointee.ifa_flags)
            guard (flags & IFF_UP) == IFF_UP, (flags & IFF_LOOPBACK) == 0,
                  let ifaAddr = cursor.pointee.ifa_addr, ifaAddr.pointee.sa_family == UInt8(AF_INET) else { continue }

            let name = String(cString: cursor.pointee.ifa_name)
            var hostBuffer = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            let result = getnameinfo(
                ifaAddr, socklen_t(ifaAddr.pointee.sa_len),
                &hostBuffer, socklen_t(hostBuffer.count),
                nil, 0, NI_NUMERICHOST
            )
            guard result == 0 else { continue }
            let host = String(cString: hostBuffer)

            if name == "en0" {
                address = host
                break
            } else if fallback == nil {
                fallback = host
            }
        }
        return address ?? fallback
    }

    // MARK: - Message handling

    /// Read-and-decode helper shared by every case below.
    private func decode(_ jsonString: String) -> [String: Any]? {
        guard let data = jsonString.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return object
    }

    fileprivate func encode(_ dict: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: dict) else { return "{}" }
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    /// Entry point for one incoming WebSocket text frame. Authentication itself is decided once,
    /// up front, from the connection's cookie (see `checkAndTouchDevice` and
    /// `WebPortalSocketHandler.initiallyAuthenticated`) and never changes for the life of the
    /// connection — there's no PIN message on this protocol anymore, so an unauthenticated
    /// connection simply can't do anything until the browser logs in via `POST /api/login` and
    /// reconnects with the resulting cookie.
    fileprivate func handleIncoming(_ jsonString: String, isAuthenticated: Bool, respond: @escaping (String) -> Void) async {
        guard isAuthenticated, let object = decode(jsonString), let type = object["type"] as? String else { return }
        await handleAuthenticated(type: type, object: object, respond: respond)
    }

    private func handleAuthenticated(type: String, object: [String: Any], respond: @escaping (String) -> Void) async {
        guard let conversationManager else {
            respond(encode(["type": "error", "message": "Web Portal isn't fully connected yet — try again in a moment."]))
            return
        }

        func conversationSummaries() -> [[String: Any]] {
            conversationManager.conversations.map { ["id": $0.id.uuidString, "title": $0.title] }
        }

        switch type {
        case "listConversations":
            respond(encode(["type": "conversations", "items": conversationSummaries()]))

        case "getMessages":
            guard let idString = object["conversationId"] as? String,
                  let id = UUID(uuidString: idString),
                  let conversation = conversationManager.conversation(id: id) else { return }
            let items = conversation.messages.map { message -> [String: Any] in
                var dict: [String: Any] = ["id": message.id.uuidString, "isUser": message.isUser, "text": message.text]
                // Inlined as a data URL rather than a separate authenticated file route — this is
                // a v1 simplicity trade (a large image bloats the JSON payload) that avoids
                // standing up a second, independently-gated way to reach someone's generated
                // images.
                if let imageData = message.resolvedImageData {
                    dict["imageURL"] = "data:image/jpeg;base64,\(imageData.base64EncodedString())"
                }
                return dict
            }
            respond(encode(["type": "messages", "conversationId": idString, "items": items]))

        case "newConversation":
            conversationManager.createConversation()
            guard let newId = conversationManager.activeConversationId else { return }
            respond(encode(["type": "created", "conversationId": newId.uuidString]))

        case "deleteConversation":
            guard let idString = object["conversationId"] as? String, let id = UUID(uuidString: idString) else { return }
            conversationManager.deleteConversation(id: id)
            respond(encode(["type": "conversations", "items": conversationSummaries()]))

        case "send":
            guard let idString = object["conversationId"] as? String,
                  let id = UUID(uuidString: idString),
                  let text = object["text"] as? String,
                  !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
            await handleSend(conversationId: id, text: text, respond: respond)

        case "getSettings":
            respond(encode(settingsSnapshot()))

        case "setCustomInstructions":
            guard let text = object["text"] as? String else { return }
            UserDefaults.standard.set(text, forKey: "customInstructions")
            respond(encode(settingsSnapshot()))

        case "setEnableRAG":
            guard let value = object["value"] as? Bool else { return }
            UserDefaults.standard.set(value, forKey: "enableRAG")
            respond(encode(settingsSnapshot()))

        case "setEnableMemories":
            guard let value = object["value"] as? Bool else { return }
            UserDefaults.standard.set(value, forKey: "enableMemories")
            respond(encode(settingsSnapshot()))

        case "setInternetAccess":
            guard let value = object["value"] as? Bool else { return }
            webSearchManager?.isEnabled = value
            respond(encode(settingsSnapshot()))

        case "setBraveAPIKey":
            guard let value = object["value"] as? String else { return }
            webSearchManager?.braveAPIKey = value
            respond(encode(settingsSnapshot()))

        case "deleteMemory":
            guard let idString = object["id"] as? String, let uuid = UUID(uuidString: idString),
                  let memoryManager, let index = memoryManager.memories.firstIndex(where: { $0.id == uuid }) else { return }
            memoryManager.removeMemory(at: index)
            respond(encode(settingsSnapshot()))

        case "clearMemories":
            memoryManager?.clearAllMemories()
            respond(encode(settingsSnapshot()))

        case "resetPersonality":
            personalityManager?.resetPersonality()
            respond(encode(settingsSnapshot()))

        case "deleteFeedback":
            guard let idString = object["id"] as? String, let uuid = UUID(uuidString: idString) else { return }
            feedbackManager?.delete(uuid)
            respond(encode(settingsSnapshot()))

        case "clearFeedback":
            feedbackManager?.clearAll()
            respond(encode(settingsSnapshot()))

        case "setContextLimit":
            guard let value = object["value"] as? Int, let llmManager else { return }
            llmManager.contextTokenLimit = min(value, llmManager.deviceContextCeiling)
            llmManager.contextLimitAutoAdjustedTo = nil
            respond(encode(settingsSnapshot()))

        case "setMaxTokens":
            guard let value = object["value"] as? Int else { return }
            llmManager?.maxTokens = value
            respond(encode(settingsSnapshot()))

        case "setTemperature":
            guard let value = object["value"] as? Double else { return }
            llmManager?.temperature = value
            respond(encode(settingsSnapshot()))

        case "setHighVariability":
            guard let value = object["value"] as? Bool else { return }
            llmManager?.highVariabilityEnabled = value
            respond(encode(settingsSnapshot()))

        case "unloadModel":
            guard let llmManager else { return }
            guard case .loading = llmManager.loadState else {
                llmManager.unloadModel()
                respond(encode(settingsSnapshot()))
                return
            }
            respond(encode(["type": "settingsNotice", "isError": true, "message": "A model is already loading — wait for it to finish first."]))

        case "selectModel":
            handleSelectModel(object, respond: respond)

        default:
            break
        }
    }

    /// Mirrors `SettingsView.handleModelSelection`'s memory-safety gate: safe models load
    /// immediately, borderline ones ask the browser to confirm first (mapped to
    /// `checkMemorySafety`'s `.warning` case, resolved the same way the native "Memory Allocation
    /// Warning" popup resolves it — `loadModel(at:forceLoad: true)` once confirmed), and models
    /// that would actually risk iOS killing the app are refused outright with no override, exactly
    /// like the phone UI offers none there either. Core ML models skip this gate entirely, same as
    /// `coreMLModelRow` — `loadCoreMLModel` runs its own preflight internally.
    private func handleSelectModel(_ object: [String: Any], respond: @escaping (String) -> Void) {
        guard let path = object["path"] as? String, let kind = object["kind"] as? String, let llmManager else { return }
        let url = URL(fileURLWithPath: path)
        let force = (object["force"] as? Bool) ?? false

        // Belt-and-suspenders alongside the client disabling its own Load buttons while
        // `isLoading` — this is what actually stops two connected browsers (or a browser and the
        // phone) from starting overlapping loads, since the client-side disable only protects one
        // browser's own clicks.
        guard case .loading = llmManager.loadState else {
            handleSelectModel(url: url, path: path, kind: kind, force: force, llmManager: llmManager, respond: respond)
            return
        }
        respond(encode(["type": "settingsNotice", "isError": true, "message": "A model is already loading — wait for it to finish before switching."]))
    }

    private func handleSelectModel(url: URL, path: String, kind: String, force: Bool, llmManager: LLMManager, respond: @escaping (String) -> Void) {
        guard kind == "llm" else {
            llmManager.loadModel(at: url)
            respond(encode(settingsSnapshot()))
            return
        }

        if force {
            llmManager.loadModel(at: url, forceLoad: true)
            respond(encode(settingsSnapshot()))
            return
        }

        switch llmManager.checkMemorySafety(at: url) {
        case .safe:
            llmManager.loadModel(at: url)
            respond(encode(settingsSnapshot()))
        case .warning:
            let sizeGB = llmManager.getModelSizeGB(at: url)
            let message = llmManager.willStreamFromStorage(modelSizeGB: sizeGB)
                ? "'\(url.lastPathComponent)' is larger than this device can hold in memory at once. It will still run — the layers that don't fit are read from storage as needed — but expect much slower replies."
                : "'\(url.lastPathComponent)' is large for this device. It should load, but memory will be tight."
            respond(encode(["type": "modelSafetyConfirm", "path": path, "kind": kind, "message": message]))
        case .dangerous:
            respond(encode(["type": "settingsNotice", "isError": true, "message": "'\(url.lastPathComponent)' needs more memory than this device can give it, so it can't be loaded from here."]))
        }
    }

    /// Full settings snapshot sent after `getSettings` and after every settings-mutating message —
    /// the client just re-renders the whole panel each time rather than tracking per-field deltas,
    /// which keeps this protocol simple and immune to drift between fields that change together
    /// (e.g. a model load changes both `backend` and `activeModelName` at once).
    private func settingsSnapshot() -> [String: Any] {
        var dict: [String: Any] = ["type": "settings"]

        dict["customInstructions"] = UserDefaults.standard.string(forKey: "customInstructions") ?? ""
        dict["enableRAG"] = UserDefaults.standard.object(forKey: "enableRAG") as? Bool ?? true
        dict["enableMemories"] = UserDefaults.standard.object(forKey: "enableMemories") as? Bool ?? true

        if let webSearchManager {
            dict["internetAccess"] = webSearchManager.isEnabled
            dict["hasBraveKey"] = webSearchManager.hasBraveKey
        }

        if let llmManager {
            dict["backend"] = llmManager.activeBackend == .coreML ? "coreML" : "llamaCpp"
            switch llmManager.loadState {
            case .loaded(let name, _):
                dict["activeModelName"] = name
            case .loading(let progress, let status):
                // Lets a Settings panel opened *during* an in-progress load (started from the
                // phone, or from another browser) show the same banner immediately, rather than
                // waiting for the next `modelLoadState` push — see `broadcastLoadState`.
                dict["isLoading"] = true
                dict["loadingProgress"] = progress
                dict["loadingMessage"] = status
            default:
                break
            }
            dict["contextLimit"] = llmManager.contextTokenLimit
            dict["contextCeiling"] = llmManager.deviceContextCeiling
            dict["maxTokens"] = llmManager.maxTokens
            dict["temperature"] = llmManager.temperature
            dict["highVariability"] = llmManager.highVariabilityEnabled
            dict["coreMLContextWindow"] = llmManager.loadedContextWindow
            dict["coreMLContextSliding"] = llmManager.coreMLContextIsSliding

            func isLoaded(_ url: URL) -> Bool {
                if case .loaded(let name, _) = llmManager.loadState { return name == url.lastPathComponent }
                return false
            }

            let llmModels = AppFiles.contents(of: AppFiles.models, matchingExtensions: ["gguf"]).map { url -> [String: Any] in
                let safetyLabel: String
                switch llmManager.checkMemorySafety(at: url) {
                case .safe: safetyLabel = "safe"
                case .warning: safetyLabel = "warning"
                case .dangerous: safetyLabel = "dangerous"
                }
                return [
                    "path": url.path,
                    "name": url.lastPathComponent,
                    "sizeGB": llmManager.getModelSizeGB(at: url),
                    "isLoaded": isLoaded(url),
                    "safety": safetyLabel
                ]
            }

            let installDirectory = ModelDownloadManager.installDirectory(for: .coreML)
            let coreMLModels = ModelCatalog.coreMLModels
                .filter { ModelDownloadManager.shared.isInstalled($0) }
                .map { installDirectory.appendingPathComponent($0.fileName) }
                .map { url -> [String: Any] in
                    ["path": url.path, "name": url.lastPathComponent, "sizeGB": AppFiles.directorySizeGB(at: url), "isLoaded": isLoaded(url)]
                }

            dict["models"] = ["llm": llmModels, "coreml": coreMLModels]
        }

        if let personalityManager {
            dict["personality"] = ["isMature": personalityManager.isMature, "databaseSize": personalityManager.databaseSizeString]
        }

        if let memoryManager {
            dict["memories"] = memoryManager.memories.map { memory -> [String: Any] in
                ["id": memory.id.uuidString, "text": memory.text, "kindLabel": Self.memoryKindLabel(memory.kind)]
            }
        }

        if let feedbackManager {
            let downVotes = feedbackManager.feedback.filter { $0.rating == .down }.reversed().prefix(15).map { entry -> [String: Any] in
                var d: [String: Any] = ["id": entry.id.uuidString, "userPrompt": entry.userPrompt, "assistantResponse": entry.assistantResponse]
                if let reason = entry.reason { d["reason"] = reason }
                return d
            }
            dict["feedback"] = [
                "upCount": feedbackManager.feedback.filter { $0.rating == .up }.count,
                "downCount": feedbackManager.feedback.filter { $0.rating == .down }.count,
                "avoidDirectives": feedbackManager.avoidDirectives,
                "downVotes": Array(downVotes)
            ]
        }

        return dict
    }

    private static func memoryKindLabel(_ kind: Memory.Kind) -> String {
        switch kind {
        case .identity:   return "YOU"
        case .preference: return "PREF"
        case .intent:     return "PLAN"
        case .event:      return "EVENT"
        }
    }

    /// Mirrors `ContentView.sendMessage` + `generateTextResponse` step for step — same classifier,
    /// same content-safety gate, same RAG/memory/personality context, same streaming safety scan
    /// — so a message sent from a browser behaves identically to one typed on the phone. The one
    /// deliberate difference is scope: an image-generation intent or a web-search-worthy message
    /// gets an honest decline instead of silently running (or silently doing nothing).
    private func handleSend(conversationId: UUID, text: String, respond: @escaping (String) -> Void) async {
        guard let conversationManager, let llmManager, let ragManager,
              let memoryManager, let personalityManager, let feedbackManager else {
            respond(encode(["type": "error", "message": "Web Portal isn't fully connected yet — try again in a moment."]))
            return
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        let intent = PromptClassifier.classify(trimmed)
        let isImageRequest: Bool = {
            if case .imageGeneration = intent { return true }
            return false
        }()

        let promptDecision = ContentSafety.review(trimmed, surface: isImageRequest ? .imagePrompt : .chatPrompt)
        guard promptDecision.isAllowed else {
            respond(encode(["type": "error", "message": promptDecision.message ?? "This request isn't allowed under the app's content policy."]))
            return
        }

        // Captured before this turn's own user message is added — "everything before this turn,"
        // exactly matching `ContentView.sendMessage`'s ordering, since `generateResponse` takes
        // the current prompt separately from `history`.
        let history = conversationManager.conversation(id: conversationId)?.messages ?? []

        conversationManager.addMessageToActive(isUser: true, text: trimmed, conversationId: conversationId)

        if isImageRequest {
            let notice = "Image generation isn't available from the Web Portal yet — open the app on your phone to generate images."
            conversationManager.addMessageToActive(isUser: false, text: notice, conversationId: conversationId)
            conversationManager.saveConversations()
            respond(encode(["type": "notice", "text": notice]))
            return
        }

        if promptDecision.attachesCrisisResources {
            conversationManager.addMessageToActive(isUser: false, text: LegalText.crisisResources, conversationId: conversationId)
        }

        if !llmManager.isModelLoaded, let activeURL = llmManager.activeModelURL {
            llmManager.loadModel(at: activeURL)
        }
        guard llmManager.isModelLoaded else {
            respond(encode(["type": "error", "message": "No chat model is loaded right now — load one in the app first."]))
            return
        }

        let enableRAG = UserDefaults.standard.object(forKey: "enableRAG") as? Bool ?? true
        let enableMemories = UserDefaults.standard.object(forKey: "enableMemories") as? Bool ?? true
        let customInstructions = UserDefaults.standard.string(forKey: "customInstructions")
            ?? "You are a local assistant. Respond with precise answers."

        if enableMemories {
            memoryManager.extractMemories(from: trimmed)
        }

        let ragContext = enableRAG ? ragManager.retrieveRelevantContext(query: trimmed) : ""
        let memoriesContext = enableMemories ? memoryManager.getFormattedMemoriesForContext() : ""
        let systemPrompt = ChatOrchestration.systemPromptWithPersonality(
            base: customInstructions,
            llmManager: llmManager,
            personalityManager: personalityManager,
            feedbackManager: feedbackManager
        )

        conversationManager.addMessageToActive(isUser: false, text: "", conversationId: conversationId)

        let scanner = StreamSafetyScanner()

        llmManager.generateResponse(
            prompt: trimmed,
            history: history,
            systemPrompt: systemPrompt,
            memoriesContext: memoriesContext,
            ragContext: ragContext
        ) { token in
            let updated = (conversationManager.conversation(id: conversationId)?.messages.last?.text ?? "") + token
            conversationManager.updateLastMessage(text: updated, conversationId: conversationId)
            respond(self.encode(["type": "token", "conversationId": conversationId.uuidString, "text": token]))

            if scanner.shouldScan(updated), let violation = ContentSafety.streamingViolation(in: updated) {
                llmManager.cancelGeneration()
                let stoppedText = "[Response stopped by the content filter — \(violation.reportLabel).]"
                conversationManager.updateLastMessage(text: stoppedText, conversationId: conversationId)
                respond(self.encode(["type": "notice", "text": stoppedText]))
                LogManager.shared.log("ContentSafety: cancelled Web Portal stream — \(violation.rawValue)")
            }
        } onComplete: { finalText in
            let wasCancelled = llmManager.wasCancelled
            var cleanedText = ModelOutput.filterThoughts(from: finalText, stripMarkdown: personalityManager.isMature)
            if cleanedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                cleanedText = "[No response content was generated — the model may have only produced internal notes for this turn. Try regenerating or rephrasing your prompt.]"
            }

            let outputDecision = ContentSafety.review(cleanedText, surface: .modelOutput)
            if !outputDecision.isAllowed {
                cleanedText = outputDecision.message ?? "[This response was withheld by the content filter.]"
                LogManager.shared.log("ContentSafety: withheld Web Portal response — \(outputDecision.category?.rawValue ?? "unknown")")
            } else if wasCancelled {
                cleanedText += "\n\n[Response stopped.]"
            }

            conversationManager.updateLastMessage(text: cleanedText, conversationId: conversationId)
            conversationManager.saveConversations()

            if enableMemories, llmManager.activeModelURL != nil {
                personalityManager.analyzeUserMessage(trimmed, llmManager: llmManager)
                personalityManager.analyzeAssistantMessage(cleanedText)
            }

            respond(self.encode(["type": "done", "conversationId": conversationId.uuidString]))
        }
    }
}

/// One browser that has unlocked the Web Portal with the PIN and hasn't been removed since.
/// `id` doubles as the value stored in the `darkai_device` cookie — a random, unguessable token
/// minted at login time, not anything derived from the browser itself. Shown in Settings so a
/// remembered browser is never invisible: it can always be renamed or removed from there.
struct WebPortalDevice: Codable, Identifiable {
    let id: String
    var name: String
    let userAgent: String?
    let firstSeen: Date
    var lastSeen: Date
}

/// One WebSocket connection's message loop. A thin, stateless-by-itself adapter: all durable state
/// (devices, conversations, generation) lives on `WebPortalManager`; this only carries the one
/// fact decided at connection time by the `GET /api/ws` route — whether the cookie on *this*
/// specific connection matched a remembered device — since a fresh browser tab or a reconnect
/// always re-derives that from its own cookie, regardless of what any other open connection has
/// done.
private struct WebPortalSocketHandler: WSMessageHandler {
    let portal: WebPortalManager
    let initiallyAuthenticated: Bool

    func makeMessages(for client: AsyncStream<WSMessage>) async throws -> AsyncStream<WSMessage> {
        AsyncStream<WSMessage> { continuation in
            let connectionId = UUID()
            let task = Task { @MainActor in
                // Told upfront, unprompted — the client has no way to read an `HttpOnly` cookie
                // itself, so the server is the only side that knows whether this connection is
                // already trusted.
                continuation.yield(.text(portal.encode(["type": "sessionState", "authenticated": initiallyAuthenticated])))
                if initiallyAuthenticated {
                    // Only authenticated connections get unsolicited pushes (model load progress,
                    // etc.) — see `WebPortalManager.broadcast`.
                    portal.registerConnection(connectionId) { outgoing in continuation.yield(.text(outgoing)) }
                }
                for await message in client {
                    guard case .text(let jsonString) = message else { continue }
                    await portal.handleIncoming(jsonString, isAuthenticated: initiallyAuthenticated) { outgoing in
                        continuation.yield(.text(outgoing))
                    }
                }
                portal.unregisterConnection(connectionId)
                continuation.finish()
            }
            continuation.onTermination = { _ in
                Task { @MainActor in portal.unregisterConnection(connectionId) }
                task.cancel()
            }
        }
    }
}

extension WebPortalManager: @unchecked Sendable {
    // Every mutable property is only ever touched from `@MainActor` methods on this class — the
    // same justification `SDWrapper`/other `@unchecked Sendable` types in this app already use.
    // This conformance exists solely so `WebPortalSocketHandler` (a plain `Sendable` struct, per
    // `WSMessageHandler`'s protocol requirement) can hold a reference to it; every actual call
    // into it below still hops back onto the main actor via `Task { @MainActor in }`.
}
