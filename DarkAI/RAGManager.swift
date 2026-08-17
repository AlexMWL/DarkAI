import Foundation
import Combine
import SwiftUI

struct RAGDocument: Codable, Identifiable {
    var id: UUID = UUID()
    var name: String
    var content: String
    var chunks: [String]
    /// Filename (not full path) of an associated generated image, stored under
    /// Documents/GeneratedImages/. Nil for plain text documents. Optional with a
    /// default so existing persisted documents without this field decode cleanly.
    var imageFileName: String? = nil
}

class RAGManager: ObservableObject {
    @Published var documents: [RAGDocument] = []
    
    private let storageKey = "DarkAI_RAGDocuments"
    private let chunkSize = 300 // words per chunk
    private let chunkOverlap = 50

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

    init() {
        loadDocuments()
    }
    
    func loadDocuments() {
        guard let data = UserDefaults.standard.data(forKey: storageKey) else {
            // Starts empty. A previous build seeded a "Sideloading Guide" document here that
            // walked the user through installing the app outside the App Store; besides being
            // an automatic rejection, it also meant every new user's very first retrieval
            // result was app-distribution trivia rather than anything they had added.
            documents = []
            return
        }
        guard let decoded = try? JSONDecoder().decode([RAGDocument].self, from: data) else {
            // Data existed — this isn't a fresh install — but couldn't be decoded, e.g. a future
            // non-additive schema change. Starting empty is still the only real option (there is
            // no partial-recovery path for a corrupt property list), but doing that with no trace
            // at all is what turns a legitimate schema change into what reads to the single
            // maintainer as inexplicable, silent data loss. The raw bytes are left in place at
            // `storageKey` rather than cleared, in case they're worth inspecting later.
            LogManager.shared.log("RAGManager: found \(data.count) bytes of stored documents but failed to decode them — starting with an empty Mindscape rather than losing them silently")
            documents = []
            return
        }
        documents = decoded.filter { !Self.isRetiredSeedDocument($0) }
        if documents.count != decoded.count {
            saveDocuments()
        }
    }

    /// Drops the retired seed document from installs that already persisted it, so an upgrade
    /// clears it rather than leaving it retrievable forever.
    private static func isRetiredSeedDocument(_ doc: RAGDocument) -> Bool {
        doc.name == "Sideloading Guide.txt"
    }
    
    func saveDocuments() {
        if let encoded = try? JSONEncoder().encode(documents) {
            UserDefaults.standard.set(encoded, forKey: storageKey)
        }
    }
    
    /// Returns whether `content` had to be truncated to fit `maxDocumentCharacters`, so a caller
    /// that shows the user a character count can say so rather than silently storing less than
    /// what was reported as extracted.
    @discardableResult
    func ingestDocument(name: String, content: String, imageFileName: String? = nil) -> Bool {
        let wasTruncated = content.count > Self.maxDocumentCharacters
        let boundedContent = wasTruncated ? String(content.prefix(Self.maxDocumentCharacters)) : content
        let chunks = splitIntoChunks(text: boundedContent)
        let doc = RAGDocument(name: name, content: boundedContent, chunks: chunks, imageFileName: imageFileName)
        documents.append(doc)
        evictOldestIfNeeded()
        saveDocuments()
        if wasTruncated {
            LogManager.shared.log("RAGManager: '\(name)' exceeded \(Self.maxDocumentCharacters) characters — truncated on ingest.")
        }
        return wasTruncated
    }

    /// Drops the oldest documents once the corpus exceeds `maxDocuments`, cleaning up each one's
    /// backing image file the same way `deleteDocument` does below — an eviction is a deletion,
    /// not just an array trim, and leaving its file behind would orphan it in `GeneratedImages`.
    /// Safe to assume `documents` is oldest-first: every ingestion route only ever appends.
    private func evictOldestIfNeeded() {
        guard documents.count > Self.maxDocuments else { return }
        let overflow = documents.count - Self.maxDocuments
        for doc in documents.prefix(overflow) {
            if let fileName = doc.imageFileName, let dir = generatedImagesDirectory {
                try? FileManager.default.removeItem(at: dir.appendingPathComponent(fileName))
            }
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
        // Clean up any backing image file so deleted RAG entries don't leave orphaned
        // files behind in the GeneratedImages directory.
        for index in offsets {
            guard index >= 0 && index < documents.count,
                  let fileName = documents[index].imageFileName,
                  let dir = generatedImagesDirectory else { continue }
            try? FileManager.default.removeItem(at: dir.appendingPathComponent(fileName))
        }
        documents.remove(atOffsets: offsets)
        saveDocuments()
    }
    
    private func splitIntoChunks(text: String) -> [String] {
        let words = text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        var chunks: [String] = []
        
        var i = 0
        while i < words.count {
            let end = min(i + chunkSize, words.count)
            let chunkWords = words[i..<end]
            let chunkText = chunkWords.joined(separator: " ")
            chunks.append(chunkText)
            
            i += (chunkSize - chunkOverlap)
            if i >= words.count || end == words.count {
                break
            }
        }
        
        return chunks
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
            for chunk in doc.chunks {
                let chunkTokens = tokenize(chunk)
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
