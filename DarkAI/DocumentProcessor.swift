import Foundation
import UIKit
import Vision
import PDFKit

nonisolated class DocumentProcessor {

    enum ProcessError: Error, LocalizedError {
        case invalidData
        case textExtractionFailed
        case unsupportedType
        case fileTooLarge(Int)

        var errorDescription: String? {
            switch self {
            case .invalidData: return "Couldn't read that file."
            case .textExtractionFailed: return "Couldn't extract any text from that file."
            case .unsupportedType: return "That file type isn't supported yet."
            case .fileTooLarge(let bytes):
                let actual = ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
                let cap = ByteCountFormatter.string(fromByteCount: Int64(DocumentProcessor.maxFileBytes), countStyle: .file)
                return "That file is \(actual), which is over the \(cap) limit for attachments."
            }
        }
    }

    /// Largest source file read into memory before extracting text. Independent of
    /// `StructuredImport.maxFileBytes` (8 MB, tuned for JSON) — this path also handles photos and
    /// PDFs, which are routinely larger than that for entirely ordinary attachments, so it needs
    /// its own, more generous ceiling rather than reusing a cap sized for a different kind of file.
    static let maxFileBytes = 50 * 1024 * 1024

    /// Extracts text from common document types (TXT, RTF) or Images using OCR
    static func extractText(from url: URL) async throws -> String {
        let sizeBytes = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        guard sizeBytes <= maxFileBytes else {
            throw ProcessError.fileTooLarge(sizeBytes)
        }

        // Backgrounded regardless of which actor context called this — reading a large file,
        // parsing a PDF, and running Vision OCR are all real work, not something that should ever
        // run on whichever isolation the caller happened to have. The chat-attachment call site
        // (`ContentView.handleFileImport`) invokes this from a plain `Task {}` on a `@MainActor`
        // view, and being `async` alone doesn't move synchronous work off of that — it only means
        // the call *can* suspend, not that it does. `Task.detached` is what actually guarantees
        // this runs off the main actor. `SettingsView`'s own (separate) file-import path already
        // hit and fixed this exact class of bug — a large file read synchronously on the main
        // thread visibly freezes the app mid-tap — for its path; this one never got the same fix.
        return try await Task.detached(priority: .userInitiated) {
            try await extractTextOffMainActor(from: url)
        }.value
    }

    private static func extractTextOffMainActor(from url: URL) async throws -> String {
        let ext = url.pathExtension.lowercased()

        switch ext {
        case "txt", "md", "csv", "json":
            return try String(contentsOf: url, encoding: .utf8)

        case "rtf":
            let data = try Data(contentsOf: url)
            if let attrString = try? NSAttributedString(data: data, options: [.documentType: NSAttributedString.DocumentType.rtf], documentAttributes: nil) {
                return attrString.string
            }
            throw ProcessError.textExtractionFailed

        case "doc", "docx":
            throw ProcessError.unsupportedType

        case "pdf":
            guard let document = PDFDocument(url: url) else {
                throw ProcessError.invalidData
            }
            var fullText = ""
            for i in 0..<document.pageCount {
                if let page = document.page(at: i), let text = page.string {
                    fullText += text + "\n"
                }
            }
            if fullText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                // PDF might be image-based, could run OCR on PDF pages here, but for now fallback
                throw ProcessError.textExtractionFailed
            }
            return fullText

        case "jpg", "jpeg", "png", "gif", "heic":
            guard let image = UIImage(contentsOfFile: url.path) else {
                throw ProcessError.invalidData
            }
            return try await performOCR(on: image)

        default:
            throw ProcessError.unsupportedType
        }
    }

    private static func performOCR(on image: UIImage) async throws -> String {
        guard let cgImage = image.cgImage else {
            throw ProcessError.invalidData
        }

        return try await withCheckedThrowingContinuation { continuation in
            let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])

            let textRequest = VNRecognizeTextRequest()
            textRequest.recognitionLevel = .accurate
            textRequest.usesLanguageCorrection = true

            let classifyRequest = VNClassifyImageRequest()

            do {
                try requestHandler.perform([textRequest, classifyRequest])

                var result = ""

                if let classObservations = classifyRequest.results {
                    let topLabels = classObservations
                        .filter { $0.confidence > 0.6 }
                        .prefix(5)
                        .map { $0.identifier }

                    if !topLabels.isEmpty {
                        result += "Image Classification Tags: " + topLabels.joined(separator: ", ") + "\n\n"
                    }
                }

                if let textObservations = textRequest.results {
                    let extracted = textObservations.compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
                    if !extracted.isEmpty {
                        result += "Extracted Text (OCR):\n" + extracted
                    }
                }

                if result.isEmpty {
                    continuation.resume(returning: "No recognizable text or objects found.")
                } else {
                    continuation.resume(returning: result.trimmingCharacters(in: .whitespacesAndNewlines))
                }
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
}
