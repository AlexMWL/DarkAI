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
    
    init() {
        loadDocuments()
    }
    
    func loadDocuments() {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let decoded = try? JSONDecoder().decode([RAGDocument].self, from: data) else {
            // Starts empty. A previous build seeded a "Sideloading Guide" document here that
            // walked the user through installing the app outside the App Store; besides being
            // an automatic rejection, it also meant every new user's very first retrieval
            // result was app-distribution trivia rather than anything they had added.
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
    
    func ingestDocument(name: String, content: String, imageFileName: String? = nil) {
        let chunks = splitIntoChunks(text: content)
        let doc = RAGDocument(name: name, content: content, chunks: chunks, imageFileName: imageFileName)
        documents.append(doc)
        saveDocuments()
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

    /// Creates a RAG text record for an AI-generated image so future LLM prompts
    /// can reference previously generated images by subject or date.
    func ingestGeneratedImage(prompt: String, imageData: Data) {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        let timestamp = formatter.string(from: Date())

        // Use first ~60 chars of prompt for the document name
        let shortPrompt = String(prompt.prefix(60)).trimmingCharacters(in: .whitespacesAndNewlines)
        let docName = "Generated Image – \(shortPrompt).txt"
        
        // Save the image data locally
        let imagesDir = AppFiles.generatedImages
        AppFiles.createIfNeeded(imagesDir)

        let imageFileName = "\(UUID().uuidString).jpg"
        let imageURL = imagesDir.appendingPathComponent(imageFileName)
        try? imageData.write(to: imageURL)
        AppFiles.excludeFromBackup(imageURL)

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
                    // Score is overlap normalized by query length and chunk length log
                    let score = Double(overlapCount) / (log(Double(chunkTokens.count + 1)) + 1.0)
                    scores.append(ChunkScore(documentName: doc.name, text: chunk, score: score))
                }
            }
        }
        
        // Sort descending
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
    
    // Minimal English stopwords list for filtering RAG queries
    private let stopWords: Set<String> = [
        "the", "and", "a", "of", "to", "is", "in", "it", "you", "that", "he", "was", "for", "on", "are", "as", "with",
        "his", "they", "i", "at", "be", "this", "have", "from", "or", "one", "had", "by", "word", "but", "not", "what",
        "all", "were", "we", "when", "your", "can", "said", "there", "use", "an", "each", "which", "she", "do", "how",
        "their", "if", "will", "up", "other", "about", "out", "many", "then", "them", "these", "so", "some", "her",
        "would", "make", "like", "him", "into", "time", "has", "look", "two", "more", "write", "go", "see", "number"
    ]
}
