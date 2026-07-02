//
//  DocumentProcessor.swift
//  DarkAI
//
//  Created by Antigravity on 6/29/26.
//

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
            let request = VNRecognizeTextRequest { (request, error) in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }
                
                guard let observations = request.results as? [VNRecognizedTextObservation] else {
                    continuation.resume(throwing: ProcessError.textExtractionFailed)
                    return
                }
                
                let extracted = observations.compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
                continuation.resume(returning: extracted)
            }
            
            // Accurate recognition over fast
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true
            
            do {
                try requestHandler.perform([request])
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
}
