import Foundation
import Combine
import UIKit // Background/terminate lifecycle notifications for `flushPendingSave()`.

/// `nonisolated` because these are pure value types that have to be readable away from the UI:
/// `ConversationExport` encodes a whole transcript — potentially with embedded image data — off
/// the main actor precisely so a long chat doesn't block it. Nothing here is mutable shared state.
nonisolated struct ChatMessage: Identifiable, Codable, Equatable {
    var id: UUID = UUID()
    var isUser: Bool
    var text: String
    var timestamp: Date = Date()
    /// JPEG data for AI-generated images, inline in this struct. `nil` for every message created
    /// after `imageFileName` was added below — kept only so a conversation persisted before that
    /// still decodes and displays its images. New messages are never written into this field; see
    /// `resolvedImageData`.
    var imageData: Data? = nil
    /// Filename of an AI-generated image under `AppFiles.generatedImages`, written once by
    /// `AppFiles.writeGeneratedImage` and shared with the matching `RAGDocument` — the file this
    /// points at is the *only* on-disk copy of a generated image, not a chat-local copy of it.
    /// This is what `imageData` above used to be: embedding the JPEG bytes directly in this
    /// `Codable` struct meant every image in a conversation was re-serialized to `UserDefaults` in
    /// full on every single message sent anywhere in the app (`ConversationManager.persist()`
    /// encodes every conversation on every save), not just when that image's own message changed.
    var imageFileName: String? = nil
    /// Set on an assistant message that's offering to search the web for the preceding user
    /// message — holds that original user text so the search can be re-classified and run when
    /// the user taps through. Cleared the moment the user picks Search or No; a still-pending
    /// offer surviving to the next launch just harmlessly re-shows the same two buttons rather
    /// than needing any special recovery handling.
    var pendingSearchQuery: String? = nil

    var isImageMessage: Bool { imageData != nil || imageFileName != nil }

    /// This message's image bytes, however they're actually stored: read from disk via
    /// `imageFileName` for anything created after that field existed, or from the legacy inline
    /// `imageData` for messages persisted before it did. Every display/export call site should
    /// read this instead of `imageData` directly.
    var resolvedImageData: Data? {
        if let imageFileName {
            return try? Data(contentsOf: AppFiles.generatedImages.appendingPathComponent(imageFileName))
        }
        return imageData
    }
}

nonisolated struct Conversation: Identifiable, Codable, Equatable {
    var id: UUID = UUID()
    var title: String
    var messages: [ChatMessage]
    var createdAt: Date = Date()
    /// A private conversation is held in memory only and never written to disk. Optional with
    /// a default so conversations persisted before this flag existed still decode.
    var isPrivate: Bool = false
}

/// Loads and decodes an array of `T` from `UserDefaults`, logging through `LogManager` and
/// returning `nil` when the key holds bytes that exist but fail to decode — the shared shape
/// behind `ConversationManager.loadConversations`, `RAGManager.loadDocuments`, and
/// `MemoryManager.loadMemories`, which otherwise each hand-rolled the same "try to decode, and if
/// there's data but it didn't decode, say so instead of silently discarding it" sequence.
///
/// Returns `nil` for both "nothing stored yet" and "stored but undecodable" alike. A caller that
/// needs to tell those two apart — `ConversationManager` picks a different seeding behaviour for
/// each, `MemoryManager` only attempts its legacy-key migration for the former — checks
/// `UserDefaults.standard.data(forKey:)` itself for that, same as it would have to either way.
func loadOrLog<T: Decodable>(key: String, itemDescription: String) -> [T]? {
    guard let data = UserDefaults.standard.data(forKey: key) else { return nil }
    guard let decoded = try? JSONDecoder().decode([T].self, from: data) else {
        LogManager.shared.log("\(itemDescription) (\(data.count) bytes) failed to decode — starting fresh rather than losing it silently")
        return nil
    }
    return decoded
}

class ConversationManager: ObservableObject {
    @Published var conversations: [Conversation] = []
    @Published var activeConversationId: UUID? = nil

    /// True for exactly one check after a fresh install (or unreadable stored data) seeds the
    /// Welcome conversation — see `consumeJustSeededWelcome()`.
    @Published private(set) var justSeededWelcome = false

    /// Marks the active conversation as private — it stays in memory and is dropped when the
    /// app quits.
    ///
    /// This used to be a global "stealth mode" that short-circuited `saveConversations()`
    /// entirely, which silently discarded *every* pending change while it was on: deleting a
    /// different chat, or renaming one, appeared to work and then came back on next launch.
    /// Scoping privacy to the individual conversation makes the guarantee precise (this chat is
    /// never written) without holding the rest of the user's data hostage.
    @Published var privateMode: Bool = false {
        didSet {
            guard let activeId = activeConversationId,
                  let index = conversations.firstIndex(where: { $0.id == activeId }) else { return }
            conversations[index].isPrivate = privateMode
            if privateMode {
                // Remove any already-persisted copy of this conversation the moment it is
                // marked private — otherwise turning privacy on mid-chat leaves the earlier
                // half of it on disk, which is exactly the promise being made here.
                persist()
            } else {
                saveConversations()
            }
        }
    }

    private let storageKey = "DarkAI_Conversations"

    /// Serializes conversation-corpus encode+write work off the main actor, in call order, so two
    /// overlapping `persist()` calls (e.g. a streamed token update followed immediately by another)
    /// can't race and let a stale snapshot's write land after a fresher one's. Mirrors
    /// `RAGManager.saveDocuments()`'s `saveQueue`.
    private static let saveQueue = DispatchQueue(label: "com.darkai.conversationmanager.save", qos: .utility)

    /// Kept alive for the lifetime of `ConversationManager` so `NotificationCenter` doesn't drop
    /// them. Mirrors `RAGManager.lifecycleObservers`.
    private var lifecycleObservers: [NSObjectProtocol] = []

    init() {
        loadConversations()
        observeLifecycleForFlush()
    }

    deinit {
        let center = NotificationCenter.default
        lifecycleObservers.forEach { center.removeObserver($0) }
    }

    /// `persist()` queues its encode+write onto `saveQueue` and returns immediately — right for
    /// the common case, but a message saved right before the app is suspended or killed could
    /// have its write silently dropped if suspension lands in the gap between enqueueing it and
    /// the queue actually running it. Forces any pending write through first in exactly that
    /// window. Mirrors `RAGManager.observeLifecycleForFlush()`/`flushPendingSave()`, added there
    /// for the identical reason.
    private func observeLifecycleForFlush() {
        let center = NotificationCenter.default
        for name in [UIApplication.willResignActiveNotification, UIApplication.willTerminateNotification] {
            let token = center.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                // These notifications are documented to always fire on the main thread — the
                // compiler just has no static way to know that, hence the assist rather than an
                // `await` hop that could let the app finish suspending before this runs.
                MainActor.assumeIsolated {
                    self?.flushPendingSave()
                }
            }
            lifecycleObservers.append(token)
        }
    }

    /// Blocks until every `persist()` write already queued has actually run. `saveQueue` is
    /// serial, so a synchronous no-op submitted after them only returns once they have.
    private func flushPendingSave() {
        Self.saveQueue.sync {}
    }

    func loadConversations() {
        // Checked up front, separately from `loadOrLog`'s own "was there data at all" check,
        // because the two failure paths below need different seeding behaviour: a fresh install
        // gets its Welcome chat persisted immediately, but corrupt existing bytes don't, so the
        // raw data survives on disk in case it's worth inspecting later.
        let hadStoredData = UserDefaults.standard.data(forKey: storageKey) != nil
        if let decoded: [Conversation] = loadOrLog(key: storageKey, itemDescription: "ConversationManager: stored conversations") {
            self.conversations = decoded
            if let first = decoded.first {
                self.activeConversationId = first.id
            }
            return
        }
        // Data existed but couldn't be decoded (already logged by `loadOrLog`), e.g. a future
        // non-additive schema change — or there was never any data to begin with. Seeding a
        // fresh Welcome conversation is the only way to leave the app in a usable state either
        // way, but only the "never had data" case should persist it immediately; corrupt bytes
        // used to be overwritten right away by `saveConversations()`, destroying any chance of
        // recovering them, so that case leaves them on disk until the next real save happens.
        seedWelcomeConversation(persisting: !hadStoredData)
    }

    private func seedWelcomeConversation(persisting: Bool) {
        let firstChat = Conversation(
            title: "Welcome",
            messages: [
                ChatMessage(isUser: false, text: "Welcome to \(AppInfo.displayName). Everything here runs on your device — no account needed. Load a model in Settings and start typing.\n\nResponses are generated by a model on this phone and can be wrong, so check anything important. There's an optional, off-by-default internet search feature in Settings — the app will always ask before using it.")
            ]
        )
        self.conversations = [firstChat]
        self.activeConversationId = firstChat.id
        justSeededWelcome = true
        if persisting { saveConversations() }
    }

    /// One-shot read of `justSeededWelcome`, clearing it in the same call. Without this,
    /// `ContentView.openOrReuseConversation()` — which only ever reuses a conversation with zero
    /// messages — never recognizes the freshly seeded Welcome chat as reusable, since it always
    /// has its one welcome message. Left unchecked, that meant the very first thing that happened
    /// after a clean install was a brand-new blank chat silently replacing Welcome as active,
    /// before the user ever saw it. Consuming the flag (rather than leaving it `true`) matters
    /// too: `openOrReuseConversation()` also runs from the drawer's "+" button, which must still
    /// create a real new chat rather than keep bouncing back to Welcome for the rest of the
    /// session.
    func consumeJustSeededWelcome() -> Bool {
        defer { justSeededWelcome = false }
        return justSeededWelcome
    }

    func saveConversations() {
        persist()
    }

    /// Writes everything except conversations flagged private. This used to `JSONEncoder().encode`
    /// the whole filtered array synchronously, inline — called on nearly every message sent or
    /// streamed, across every saved conversation, with no cap on how large that history grows. A
    /// snapshot is taken here, on whichever actor called this (always main, since `conversations`
    /// is `@Published`), and the actual encode+write happens on `saveQueue` — mirrors
    /// `RAGManager.saveDocuments()`, which got this same treatment for the same reason.
    private func persist() {
        let persistable = conversations.filter { !$0.isPrivate }
        let key = storageKey
        Self.saveQueue.async {
            do {
                let encoded = try JSONEncoder().encode(persistable)
                UserDefaults.standard.set(encoded, forKey: key)
            } catch {
                LogManager.shared.log("ConversationManager: failed to encode \(persistable.count) conversations for save — \(error.localizedDescription)")
            }
        }
    }

    var activeConversation: Conversation? {
        conversations.first(where: { $0.id == activeConversationId })
    }

    func conversation(id: UUID) -> Conversation? {
        conversations.first(where: { $0.id == id })
    }

    func createConversation() {
        let newChat = Conversation(title: "Chat \(conversations.count + 1)", messages: [])
        conversations.insert(newChat, at: 0)
        activeConversationId = newChat.id
        privateMode = false
        saveConversations()
    }
    
    func deleteConversation(id: UUID) {
        conversations.removeAll(where: { $0.id == id })
        if activeConversationId == id {
            activeConversationId = conversations.first?.id
        }
        if conversations.isEmpty {
            createConversation()
        } else {
            saveConversations()
        }
    }
    
    func selectConversation(id: UUID) {
        activeConversationId = id
        // Keep the toggle showing the state of the chat actually on screen, rather than
        // carrying the previous chat's setting over to one it was never applied to.
        let isPrivate = conversations.first(where: { $0.id == id })?.isPrivate ?? false
        if privateMode != isPrivate { privateMode = isPrivate }
    }

    /// Removes a single message. Used by the content-report flow, which has to take the
    /// offending content out of the conversation the moment it is reported.
    func deleteMessage(id: UUID) {
        guard let activeId = activeConversationId,
              let index = conversations.firstIndex(where: { $0.id == activeId }) else { return }
        conversations[index].messages.removeAll { $0.id == id }
        saveConversations()
    }
    
    /// `conversationId` defaults to whatever's currently active, but every call site reached
    /// from inside an async generation `Task` should pass the ID it captured when that task
    /// started instead. Between the `await` that kicks a generation off and the token/completion
    /// callbacks that eventually fire, the user is free to switch chats, delete the one that was
    /// active, or start a new one — "active" is a UI selection, not a stable handle on which
    /// conversation a specific in-flight response belongs to. Defaulting to `nil` keeps every
    /// existing synchronous call site (the user's own just-typed message, added before any `await`
    /// has had a chance to run) behaving exactly as before.
    func addMessageToActive(isUser: Bool, text: String, conversationId: UUID? = nil) {
        guard let activeId = conversationId ?? activeConversationId else { return }

        if let index = conversations.firstIndex(where: { $0.id == activeId }) {
            let msg = ChatMessage(isUser: isUser, text: text)
            conversations[index].messages.append(msg)
            
            // Auto update title if it was default and has messages now
            if conversations[index].title.hasPrefix("Chat ") && isUser {
                let cleanTitle = text.prefix(20).trimmingCharacters(in: .whitespacesAndNewlines)
                conversations[index].title = cleanTitle.isEmpty ? "Chat" : String(cleanTitle) + "..."
            }
            
            saveConversations()
        }
    }

    /// Appends an assistant message offering to search the web, with the original user query
    /// attached via `pendingSearchQuery` so the confirm/decline buttons know what to act on.
    func addSearchOfferToActive(text: String, query: String) {
        guard let activeId = activeConversationId else { return }
        if let index = conversations.firstIndex(where: { $0.id == activeId }) {
            var msg = ChatMessage(isUser: false, text: text)
            msg.pendingSearchQuery = query
            conversations[index].messages.append(msg)
            saveConversations()
        }
    }

    /// Clears the pending-search flag on a specific message, so its buttons stop rendering.
    /// Called the moment the user picks either Search or No — the offer is a one-shot choice.
    /// Keyed by message ID rather than "the last message": a user can send a new message before
    /// answering an older offer (nothing blocks that), so the offer being answered isn't
    /// guaranteed to still be the last one in the conversation.
    func clearPendingSearch(messageId: UUID) {
        guard let activeId = activeConversationId,
              let index = conversations.firstIndex(where: { $0.id == activeId }),
              let messageIndex = conversations[index].messages.firstIndex(where: { $0.id == messageId }) else { return }
        conversations[index].messages[messageIndex].pendingSearchQuery = nil
        saveConversations()
    }

    /// See `addMessageToActive`'s doc comment for why `conversationId` exists and when to pass it.
    func updateLastMessage(text: String, conversationId: UUID? = nil) {
        guard let activeId = conversationId ?? activeConversationId else { return }
        if let index = conversations.firstIndex(where: { $0.id == activeId }),
           !conversations[index].messages.isEmpty {
            let lastIndex = conversations[index].messages.count - 1
            conversations[index].messages[lastIndex].text = text
        }
    }

    /// Updates the imageData of the last message (used while generation completes). See
    /// `addMessageToActive`'s doc comment for why `conversationId` exists and when to pass it.
    func updateLastMessageImage(imageFileName: String, conversationId: UUID? = nil) {
        guard let activeId = conversationId ?? activeConversationId else { return }
        if let index = conversations.firstIndex(where: { $0.id == activeId }),
           !conversations[index].messages.isEmpty {
            let lastIndex = conversations[index].messages.count - 1
            conversations[index].messages[lastIndex].imageFileName = imageFileName
        }
    }
}
