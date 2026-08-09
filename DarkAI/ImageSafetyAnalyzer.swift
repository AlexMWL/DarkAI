import Foundation
import UIKit
import CoreImage
import SensitiveContentAnalysis

/// Post-generation nudity screening for images the app produced.
///
/// Why this exists: prompt screening cannot hold the line on its own. `ContentSafety` blocks what
/// the *user asked for*, but an uncensored checkpoint will happily produce explicit output from a
/// prompt containing none of the words that screening looks for — "woman reclining, soft light,
/// intimate" is not a phrase any keyword list can reasonably reject, and the model decides the
/// rest. The only place that failure can actually be caught is after the pixels exist.
///
/// Two layers, because neither is sufficient alone:
///
/// 1. **Apple's `SensitiveContentAnalysis`** — a real on-device nudity classifier, the same one
///    behind Communication Safety. Accurate, private, no network. Its catch is that it only
///    operates when the user (or their device management) has Sensitive Content Analysis enabled;
///    `analysisPolicy` reports `.disabled` otherwise, and it needs a provisioned entitlement.
///    See `isAvailable` — when it can't run, it reports so rather than silently passing.
///
/// 2. **A skin-tone heuristic** — crude, always available, and deliberately tuned to a high
///    threshold so it flags only images that are overwhelmingly bare skin. It is not a classifier
///    and will never be one; it exists so that a device with Apple's analyzer switched off is not
///    left with *no* post-generation check at all.
///
/// A flagged image is discarded, never shown or saved.
enum ImageSafetyAnalyzer {

    enum Verdict {
        /// Nothing detected by whichever layers were able to run.
        case allowed
        /// Explicit content detected — the caller must discard the image.
        case blocked(reason: String)
    }

    // MARK: - Apple's classifier

    /// Whether Apple's analyzer can actually run right now.
    ///
    /// `.disabled` means the user hasn't turned Sensitive Content Analysis on, or the entitlement
    /// isn't provisioned. Surfaced so Settings can tell the user their strongest layer is off
    /// instead of implying protection that isn't there.
    static var isSystemAnalyzerAvailable: Bool {
        SCSensitivityAnalyzer().analysisPolicy != .disabled
    }

    // MARK: - Screening

    /// Screens a generated image. Runs off the main actor; the diffusion pipeline already has it.
    static func screen(imageData: Data) async -> Verdict {
        if let verdict = await systemAnalyzerVerdict(imageData: imageData) {
            return verdict
        }
        // Apple's analyzer couldn't run — fall back to the heuristic so something still checks.
        return heuristicVerdict(imageData: imageData)
    }

    /// Returns `nil` when the system analyzer is unavailable or errored, so the caller knows to
    /// fall back rather than treating "couldn't check" as "clean".
    private static func systemAnalyzerVerdict(imageData: Data) async -> Verdict? {
        let analyzer = SCSensitivityAnalyzer()
        guard analyzer.analysisPolicy != .disabled else { return nil }

        // Needs a file URL; the framework has no data-based entry point.
        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("scan-\(UUID().uuidString).jpg")
        defer { try? FileManager.default.removeItem(at: scratch) }

        do {
            try imageData.write(to: scratch)
            let response = try await analyzer.analyzeImage(at: scratch)
            if response.isSensitive {
                LogManager.shared.log("ImageSafety: blocked by system analyzer")
                return .blocked(reason: "explicit imagery")
            }
            return .allowed
        } catch {
            LogManager.shared.log("ImageSafety: system analyzer unavailable — \(error.localizedDescription)")
            return nil
        }
    }

    // MARK: - Fallback heuristic

    /// Fraction of the image that reads as bare skin, above which the image is rejected.
    ///
    /// Set high on purpose. This measure cannot distinguish a nude from a close-up portrait or a
    /// beach scene, so a low threshold would reject ordinary images constantly. At 0.62 it only
    /// fires on frames that are almost entirely skin, which is a weak filter — that is the honest
    /// ceiling for a heuristic of this kind, and why the system analyzer above is the real
    /// defence and worth enabling.
    private static let skinFractionThreshold = 0.62

    private static func heuristicVerdict(imageData: Data) -> Verdict {
        guard let fraction = skinFraction(imageData: imageData) else { return .allowed }
        if fraction >= skinFractionThreshold {
            LogManager.shared.log(String(format: "ImageSafety: blocked by heuristic (skin %.0f%%)", fraction * 100))
            return .blocked(reason: "likely explicit imagery")
        }
        return .allowed
    }

    /// Proportion of pixels falling in a broad skin-tone range, sampled on a downscaled copy.
    ///
    /// The RGB rule is the standard Kovac bounds, which cover a wide range of skin tones rather
    /// than being tuned to any one of them — important, because a check that fires more readily
    /// on some people than others would be both useless and offensive.
    private static func skinFraction(imageData: Data) -> Double? {
        guard let image = UIImage(data: imageData)?.cgImage else { return nil }

        // 64×64 is plenty for a proportion and keeps this off the critical path.
        let side = 64
        var pixels = [UInt8](repeating: 0, count: side * side * 4)
        guard let context = CGContext(
            data: &pixels, width: side, height: side,
            bitsPerComponent: 8, bytesPerRow: side * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        context.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))

        var skin = 0
        for i in stride(from: 0, to: pixels.count, by: 4) {
            let r = Int(pixels[i]), g = Int(pixels[i + 1]), b = Int(pixels[i + 2])
            let maxC = max(r, g, b), minC = min(r, g, b)
            if r > 95, g > 40, b > 20,
               maxC - minC > 15,
               abs(r - g) > 15,
               r > g, r > b {
                skin += 1
            }
        }
        return Double(skin) / Double(side * side)
    }
}
