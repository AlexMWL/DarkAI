import Foundation
import UIKit
import Vision
import PDFKit

class DocumentProcessor {
    
    enum ProcessError: Error {
        case invalidData
        case textExtractionFailed
        case unsupportedType
    }
    
    /// Extracts text from common document types (TXT, RTF) or Images using OCR
    static func extractText(from url: URL) async throws -> String {
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
    
    static func extractText(from image: UIImage) async throws -> String {
        return try await performOCR(on: image)
    }
    
    private static func performOCR(on image: UIImage) async throws -> String {
        guard let cgImage = image.cgImage else {
            throw ProcessError.invalidData
        }
        
        return try await withCheckedThrowingContinuation { continuation in
            let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            
            // 1. Text Recognition
            let textRequest = VNRecognizeTextRequest()
            textRequest.recognitionLevel = .accurate
            textRequest.usesLanguageCorrection = true
            
            // 2. Image Classification
            let classifyRequest = VNClassifyImageRequest()
            
            do {
                try requestHandler.perform([textRequest, classifyRequest])
                
                var result = ""
                
                // Process Classification
                if let classObservations = classifyRequest.results as? [VNClassificationObservation] {
                    // Get top 5 high-confidence labels
                    let topLabels = classObservations
                        .filter { $0.confidence > 0.6 }
                        .prefix(5)
                        .map { $0.identifier }
                    
                    if !topLabels.isEmpty {
                        result += "Image Classification Tags: " + topLabels.joined(separator: ", ") + "\n\n"
                    }
                }
                
                // Process Text
                if let textObservations = textRequest.results as? [VNRecognizedTextObservation] {
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
