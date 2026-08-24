import Foundation
import Combine
import UIKit // Background/terminate lifecycle notifications for `flushPendingSave()`.
import SwiftUI // `deleteDocument(at:)` uses `Array.remove(atOffsets:)`, a SwiftUI extension on
               // `RangeReplaceableCollection` (not a Foundation API) — confirmed still in use,
               // not dead code, despite first appearing unused by a plain textual grep for
               // "SwiftUI"/"@State"/": View" etc. in this file.

struct RAGDocument: Identifiable {
    var id: UUID
    var name: String
    var content: String
    /// Derived from `content` at construction/decode time — see the custom `Codable`
    /// conformance below. Stored (not computed) so repeated access — `retrieveRelevantContext`
    /// reads every document's `chunks` on every single query — doesn't re-split `content` into
    /// words each time; just not persisted, since it's fully reconstructable from `content`.
    var chunks: [String]
    /// Filename (not full path) of an associated generated image, stored under
    /// Documents/GeneratedImages/. Nil for plain text documents. Optional with a
    /// default so existing persisted documents without this field decode cleanly.
    var imageFileName: String? = nil

    static let chunkWordCount = 300 // words per chunk
    static let chunkWordOverlap = 50

    init(name: String, content: String, imageFileName: String? = nil) {
        self.id = UUID()
        self.name = name
        self.content = content
        self.chunks = Self.computeChunks(from: content)
        self.imageFileName = imageFileName
    }

    static func computeChunks(from content: String) -> [String] {
        let words = content.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        var chunks: [String] = []

        var i = 0
        while i < words.count {
            let end = min(i + chunkWordCount, words.count)
            let chunkWords = words[i..<end]
            chunks.append(chunkWords.joined(separator: " "))

            i += (chunkWordCount - chunkWordOverlap)
            if i >= words.count || end == words.count {
                break
            }
        }

        return chunks
    }
}

extension RAGDocument: Codable {
    private enum CodingKeys: String, CodingKey {
        case id, name, content, imageFileName
        // `chunks` intentionally omitted — see its doc comment above. Persisting it used to
        // roughly double the size of every document in the corpus, since a 300-word/50-word-
        // overlap chunking is ~17% redundant with `content` on its own, for data fully
        // reconstructable from `content` at load time.
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        content = try container.decode(String.self, forKey: .content)
        imageFileName = try container.decodeIfPresent(String.self, forKey: .imageFileName)
        chunks = Self.computeChunks(from: content)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(name, forKey: .name)
        try container.encode(content, forKey: .content)
        try container.encodeIfPresent(imageFileName, forKey: .imageFileName)
    }
}

class RAGManager: ObservableObject {
    @Published var documents: [RAGDocument] = []
    
    private let storageKey = "DarkAI_RAGDocuments"

    /// Largest content a single document may store.
    ///
    /// Documents here are persisted to `UserDefaults` in full on every `saveDocuments()` call — a
    /// property list decoded into memory whole on every launch, not a database. `StructuredImport`
    /// already enforces this exact limit on its own (JSON) ingestion path, with the reasoning that
    /// nothing ingested here should be free to grow without bound. This mirrors that cap at the
    /// point every route into the Mindscape shares — chat file upload and the plain-text Settings
    /// importer both called `ingestDocument` directly with no limit of their own before this.
    static let maxDocumentCharacters = 750_000

    /// Total documents this corpus holds before the oldest are evicted to make room for a new
    /// one. Every route into the Mindscape funnels through `ingestDocument` below, and none of
    /// them previously had any cap on the corpus as a whole — `StructuredImport.maxDocuments`
    /// only bounds a single import's own entry count, not what accumulates over the app's whole
    /// lifetime. `ingestGeneratedImage` in particular runs on every successful image generation
    /// with no dedup, so an active user's corpus — and the `UserDefaults` blob it's re-serialized
    /// into in full on every single ingest — would otherwise grow for as long as the app is used.
    static let maxDocuments = 1000

    /// Serializes document-corpus encode+write work off the main actor, in call order, so two
    /// overlapping `saveDocuments()` calls (e.g. an ingest followed immediately by a delete)
    /// can't race and let a stale snapshot's write land after a fresher one's.
    private static let saveQueue = DispatchQueue(label: "com.darkai.ragmanager.save", qos: .utility)

    /// Token set for each document's chunks, keyed by document id — computed once, at ingest or
    /// load, rather than re-tokenized from scratch on every `retrieveRelevantContext` call. Kept
    /// out of `RAGDocument` itself (as opposed to a stored property there) so it's never
    /// persisted: it's a derived index rebuildable from `chunks` at any time, not data, and has
    /// no reason to bloat the blob `saveDocuments()` writes to `UserDefaults`.
    private var chunkTokenCache: [UUID: [Set<String>]] = [:]

    /// Kept alive for the lifetime of `RAGManager` so `NotificationCenter` doesn't drop them.
    private var lifecycleObservers: [NSObjectProtocol] = []

    init() {
        loadDocuments()
        observeLifecycleForFlush()
    }

    deinit {
        let center = NotificationCenter.default
        lifecycleObservers.forEach { center.removeObserver($0) }
    }

    /// `saveDocuments()` queues its encode+write onto `saveQueue` and returns immediately — right
    /// for the common case, but a document ingested (or deleted) right before the app is
    /// suspended or killed could have its write silently dropped if suspension lands in the gap
    /// between enqueueing it and the queue actually running it. Forces any pending write through
    /// first in exactly that window.
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

    /// Blocks until every `saveDocuments()` write already queued has actually run. `saveQueue` is
    /// serial, so a synchronous no-op submitted after them only returns once they have.
    private func flushPendingSave() {
        Self.saveQueue.sync {}
    }

    func loadDocuments() {
        // Starts empty on a fresh install or on a decode failure alike (`loadOrLog` logs the
        // latter case). A previous build seeded a "Sideloading Guide" document here that walked
        // the user through installing the app outside the App Store; besides being an automatic
        // rejection, it also meant every new user's very first retrieval result was
        // app-distribution trivia rather than anything they had added.
        guard let decoded: [RAGDocument] = loadOrLog(key: storageKey, itemDescription: "RAGManager: stored documents") else {
            documents = []
            chunkTokenCache = [:]
            return
        }
        documents = decoded.filter { !Self.isRetiredSeedDocument($0) }
        rebuildChunkTokenCache()
        if documents.count != decoded.count {
            saveDocuments()
        }
    }

    private func rebuildChunkTokenCache() {
        chunkTokenCache = Dictionary(uniqueKeysWithValues: documents.map { ($0.id, $0.chunks.map(tokenize)) })
    }

    /// Drops the retired seed document from installs that already persisted it, so an upgrade
    /// clears it rather than leaving it retrievable forever.
    private static func isRetiredSeedDocument(_ doc: RAGDocument) -> Bool {
        doc.name == "Sideloading Guide.txt"
    }
    
    /// Encodes and writes the corpus off the main actor. This used to `JSONEncoder().encode`
    /// the whole `documents` array synchronously, inline — cheap for a handful of short text
    /// notes, but this runs on every ingest, including after every generated image, and the
    /// corpus is uncapped up to `maxDocuments` (1000) entries. A snapshot is taken here, on
    /// whichever actor called this, and the actual encode+write happens on `saveQueue`, which
    /// also keeps overlapping calls from racing: `saveQueue` is serial, so writes land in the
    /// same order `saveDocuments()` was called in, in call order.
    func saveDocuments() {
        let snapshot = documents
        let key = storageKey
        Self.saveQueue.async {
            do {
                let encoded = try JSONEncoder().encode(snapshot)
                UserDefaults.standard.set(encoded, forKey: key)
            } catch {
                LogManager.shared.log("RAGManager: failed to encode \(snapshot.count) documents for save — \(error.localizedDescription)")
            }
        }
    }
    
    /// Returns whether `content` had to be truncated to fit `maxDocumentCharacters`, so a caller
    /// that shows the user a character count can say so rather than silently storing less than
    /// what was reported as extracted.
    @discardableResult
    func ingestDocument(name: String, content: String, imageFileName: String? = nil) -> Bool {
        let wasTruncated = appendDocument(name: name, content: content, imageFileName: imageFileName)
        evictOldestIfNeeded()
        saveDocuments()
        return wasTruncated
    }

    /// Bulk variant of `ingestDocument` for importing many entries at once — a structured JSON
    /// import can bring in up to `StructuredImport.maxDocuments` (400) entries in one go. Appends
    /// everything first and evicts/saves once, instead of the up to 400 redundant full-corpus
    /// background saves (each snapshotting a still-growing array, and each risking a copy-on-write
    /// duplication if the previous snapshot's background write hasn't finished) that calling
    /// `ingestDocument` once per entry would otherwise trigger.
    func ingestDocuments(_ entries: [(name: String, content: String)]) {
        guard !entries.isEmpty else { return }
        for entry in entries {
            appendDocument(name: entry.name, content: entry.content, imageFileName: nil)
        }
        evictOldestIfNeeded()
        saveDocuments()
    }

    /// Appends one document to `documents`/`chunkTokenCache` without evicting or saving — the
    /// part `ingestDocument` and `ingestDocuments` share, so eviction and the save itself can
    /// happen once per call (single) or once per batch (bulk) instead of being duplicated here.
    @discardableResult
    private func appendDocument(name: String, content: String, imageFileName: String?) -> Bool {
        let wasTruncated = content.count > Self.maxDocumentCharacters
        let boundedContent = wasTruncated ? String(content.prefix(Self.maxDocumentCharacters)) : content
        let doc = RAGDocument(name: name, content: boundedContent, imageFileName: imageFileName)
        documents.append(doc)
        chunkTokenCache[doc.id] = doc.chunks.map(tokenize)
        if wasTruncated {
            LogManager.shared.log("RAGManager: '\(name)' exceeded \(Self.maxDocumentCharacters) characters — truncated on ingest.")
        }
        return wasTruncated
    }

    /// Drops the oldest documents once the corpus exceeds `maxDocuments`. Does not touch each
    /// one's backing image file — see `deleteDocument`'s doc comment for why: the same file can
    /// still be live in a chat message's transcript, and an automatic, silent eviction is exactly
    /// the wrong place to risk breaking that without the user having asked for anything.
    /// Safe to assume `documents` is oldest-first: every ingestion route only ever appends.
    private func evictOldestIfNeeded() {
        guard documents.count > Self.maxDocuments else { return }
        let overflow = documents.count - Self.maxDocuments
        for doc in documents.prefix(overflow) {
            chunkTokenCache.removeValue(forKey: doc.id)
        }
        documents.removeFirst(overflow)
        LogManager.shared.log("RAGManager: corpus exceeded \(Self.maxDocuments) documents — evicted the oldest \(overflow)")
    }

    /// Directory where generated-image files backing RAG entries are stored.
    var generatedImagesDirectory: URL? { AppFiles.generatedImages }

    /// Loads the raw image data for a document's associated generated image, if any.
    func imageData(for doc: RAGDocument) -> Data? {
        guard let fileName = doc.imageFileName, let dir = generatedImagesDirectory else { return nil }
        return try? Data(contentsOf: dir.appendingPathComponent(fileName))
    }

    /// File URL for a document's associated generated image, if any. `UIImage` doesn't
    /// conform to `Transferable`, so ShareLink needs the on-disk file URL instead.
    func imageURL(for doc: RAGDocument) -> URL? {
        guard let fileName = doc.imageFileName, let dir = generatedImagesDirectory else { return nil }
        return dir.appendingPathComponent(fileName)
    }

    /// Creates a RAG text record for an AI-generated image so future LLM prompts can reference
    /// previously generated images by subject or date. `imageFileName` names a file already
    /// written to `AppFiles.generatedImages` (see `AppFiles.writeGeneratedImage`) — this used to
    /// take raw `Data` and write its own second copy of it, when the call site already had (or
    /// could have) a single on-disk copy the chat message was also going to reference.
    func ingestGeneratedImage(prompt: String, imageFileName: String) {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        let timestamp = formatter.string(from: Date())

        let shortPrompt = String(prompt.prefix(60)).trimmingCharacters(in: .whitespacesAndNewlines)
        let docName = "Generated Image – \(shortPrompt).txt"

        let content = """
        [AI-Generated Image]
        Prompt: \(prompt)
        Generated: \(timestamp)

        This entry records an image generated on this device in response to the above prompt.
        The image has been stored in RAG memory and can be recalled.
        """

        ingestDocument(name: docName, content: content, imageFileName: imageFileName)
    }


    func deleteDocument(at offsets: IndexSet) {
        for index in offsets {
            guard index >= 0 && index < documents.count else { continue }
            chunkTokenCache.removeValue(forKey: documents[index].id)
            // Deliberately does NOT delete the backing image file. `ChatMessage.imageFileName`'s
            // own doc comment says this file is "the *only* on-disk copy of a generated image,
            // not a chat-local copy of it" — the same file this RAG entry's `imageFileName`
            // points at can still be live in a conversation's transcript. Deleting it here used
            // to silently break that message's image (and any export of it) the moment its
            // Mindscape mirror entry was removed, with no warning either way. Leaving the file
            // behind trades a small amount of unreclaimed disk space for never breaking a
            // conversation the user didn't ask to touch.
        }
        documents.remove(atOffsets: offsets)
        saveDocuments()
    }
    
    // Keyword similarity search (TF-IDF / cosine-style simplified keyword overlap)
    func retrieveRelevantContext(query: String, maxResults: Int = 2) -> String {
        guard !documents.isEmpty else { return "" }
        
        let queryTokens = tokenize(query)
        guard !queryTokens.isEmpty else { return "" }
        
        struct ChunkScore {
            let documentName: String
            let text: String
            let score: Double
        }
        
        var scores: [ChunkScore] = []

        for doc in documents {
            // Cached at ingest/load time in `chunkTokenCache` rather than re-tokenized here —
            // this loop runs on every query against every chunk of every document, and
            // `tokenize` was being redone from scratch each time for text that never changes
            // after ingest. Falls back to tokenizing on the spot if the cache is somehow missing
            // an entry (e.g. index drift), so a stale/incomplete cache degrades to the old
            // behaviour rather than dropping a document from retrieval.
            let cachedTokens = chunkTokenCache[doc.id]
            for (index, chunk) in doc.chunks.enumerated() {
                let chunkTokens = (cachedTokens?.indices.contains(index) == true) ? cachedTokens![index] : tokenize(chunk)
                let overlapCount = queryTokens.filter { chunkTokens.contains($0) }.count
                if overlapCount > 0 {
                    let score = Double(overlapCount) / (log(Double(chunkTokens.count + 1)) + 1.0)
                    scores.append(ChunkScore(documentName: doc.name, text: chunk, score: score))
                }
            }
        }

        let topChunks = scores.sorted(by: { $0.score > $1.score }).prefix(maxResults)
        
        if topChunks.isEmpty {
            return ""
        }
        
        var context = "### Relevant Context Retrieved from Documents:\n"
        for chunk in topChunks {
            context += "[Source: \(chunk.documentName)]\n\(chunk.text)\n\n"
        }
        return context
    }
    
    private func tokenize(_ text: String) -> Set<String> {
        let lower = text.lowercased()
        let words = lower.components(separatedBy: CharacterSet.alphanumerics.inverted)
        let filtered = words.filter { $0.count > 2 && !stopWords.contains($0) }
        return Set(filtered)
    }
    
    private let stopWords: Set<String> = [
        "the", "and", "a", "of", "to", "is", "in", "it", "you", "that", "he", "was", "for", "on", "are", "as", "with",
        "his", "they", "i", "at", "be", "this", "have", "from", "or", "one", "had", "by", "word", "but", "not", "what",
        "all", "were", "we", "when", "your", "can", "said", "there", "use", "an", "each", "which", "she", "do", "how",
        "their", "if", "will", "up", "other", "about", "out", "many", "then", "them", "these", "so", "some", "her",
        "would", "make", "like", "him", "into", "time", "has", "look", "two", "more", "write", "go", "see", "number"
    ]
}
