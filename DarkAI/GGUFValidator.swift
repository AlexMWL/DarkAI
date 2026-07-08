import Foundation

/// Lightweight pure-Swift GGUF file validator.
///
/// The GGUF binary format (v2/v3) header layout:
///   [0..3]   magic:        "GGUF"  (4 bytes, ASCII)
///   [4..7]   version:      UInt32  (must be 2 or 3)
///   [8..15]  tensor_count: UInt64
///   [16..23] kv_count:     UInt64
///   [24..]   kv pairs:     (key_len: UInt64, key: [UInt8], value_type: UInt32, value: ...)
///
/// This validator reads only enough of the header to confirm:
///   1. The file starts with the GGUF magic bytes
///   2. The version is 2 or 3
///   3. The 'general.architecture' KV value matches the expected architecture string
///
/// It does NOT load tensors into memory, so it runs in microseconds regardless of model size.
///
enum GGUFValidator {

    // MARK: - GGUF Value Types (subset we need)
    private enum GGUFValueType: UInt32 {
        case uint8   = 0
        case int8    = 1
        case uint16  = 2
        case int16   = 3
        case uint32  = 4
        case int32   = 5
        case float32 = 6
        case bool    = 7
        case string  = 8
        case array   = 9
        case uint64  = 10
        case int64   = 11
        case float64 = 12
    }

    // MARK: - Public API

    /// Validates that `path` is a GGUF file with the expected architecture.
    /// Throws a descriptive `NSError` if validation fails.
    static func validate(path: String, expectedArchitecture: String) throws {
        let url = URL(fileURLWithPath: path)

        guard FileManager.default.fileExists(atPath: path) else {
            throw error("File not found: \(url.lastPathComponent)")
        }

        guard let handle = FileHandle(forReadingAtPath: path) else {
            throw error("Cannot open file for reading.")
        }
        defer { handle.closeFile() }

        // — Magic bytes ———————————————————————————————————————————————————
        let magicData = handle.readData(ofLength: 4)
        guard magicData.count == 4,
              String(data: magicData, encoding: .ascii) == "GGUF" else {
            throw error("\(url.lastPathComponent) is not a valid GGUF file (bad magic bytes).")
        }

        // — Version ———————————————————————————————————————————————————————
        let versionData = handle.readData(ofLength: 4)
        guard versionData.count == 4 else { throw error("Truncated GGUF header (version).") }
        let version = versionData.withUnsafeBytes { $0.load(as: UInt32.self).littleEndian }
        guard version == 2 || version == 3 else {
            throw error("Unsupported GGUF version \(version). Expected 2 or 3.")
        }

        // — tensor_count (skip) ————————————————————————————————————————————
        let _ = handle.readData(ofLength: 8)   // UInt64 tensor_count

        // — kv_count ——————————————————————————————————————————————————————
        let kvCountData = handle.readData(ofLength: 8)
        guard kvCountData.count == 8 else { throw error("Truncated GGUF header (kv_count).") }
        let kvCount = kvCountData.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }

        // — KV pairs ——————————————————————————————————————————————————————
        // Iterate just enough KV pairs to find general.architecture.
        var foundArch: String? = nil

        for _ in 0..<kvCount {
            guard let key = try? readGGUFString(from: handle) else { break }

            // Read value type
            let vtData = handle.readData(ofLength: 4)
            guard vtData.count == 4 else { break }
            let vtRaw = vtData.withUnsafeBytes { $0.load(as: UInt32.self).littleEndian }
            let vt = GGUFValueType(rawValue: vtRaw) ?? .uint8

            if key == "general.architecture" {
                if vt == .string, let archValue = try? readGGUFString(from: handle) {
                    foundArch = archValue
                    break  // We have what we need
                } else {
                    throw error("'general.architecture' has unexpected value type \(vtRaw).")
                }
            } else {
                // Skip this value so we can read the next KV pair
                try skipGGUFValue(type: vt, from: handle)
            }
        }

        guard let arch = foundArch else {
            throw error("'general.architecture' key not found in GGUF metadata.")
        }

        guard arch == expectedArchitecture else {
            throw error(
                "Architecture mismatch: file reports '\(arch)', expected '\(expectedArchitecture)'.\n" +
                "Make sure you are loading a \(expectedArchitecture) diffusion model."
            )
        }
    }

    // MARK: - Private Helpers

    private static func error(_ message: String) -> NSError {
        NSError(domain: "GGUFValidator", code: 1,
                userInfo: [NSLocalizedDescriptionKey: message])
    }

    /// Reads a GGUF length-prefixed UTF-8 string: UInt64 length + bytes.
    private static func readGGUFString(from handle: FileHandle) throws -> String {
        let lenData = handle.readData(ofLength: 8)
        guard lenData.count == 8 else {
            throw error("Unexpected EOF reading string length.")
        }
        let len = lenData.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
        guard len < 8192 else {
            throw error("GGUF string length implausibly large (\(len) bytes). File may be corrupt.")
        }
        let strData = handle.readData(ofLength: Int(len))
        guard strData.count == Int(len),
              let str = String(data: strData, encoding: .utf8) else {
            throw error("Failed to decode UTF-8 string from GGUF metadata.")
        }
        return str
    }

    /// Skips over a GGUF value in the file stream without decoding it.
    private static func skipGGUFValue(type: GGUFValueType, from handle: FileHandle) throws {
        switch type {
        case .uint8, .int8, .bool:  handle.seek(toFileOffset: handle.offsetInFile + 1)
        case .uint16, .int16:       handle.seek(toFileOffset: handle.offsetInFile + 2)
        case .uint32, .int32, .float32: handle.seek(toFileOffset: handle.offsetInFile + 4)
        case .uint64, .int64, .float64: handle.seek(toFileOffset: handle.offsetInFile + 8)
        case .string:
            _ = try readGGUFString(from: handle)
        case .array:
            // Array: elem_type (UInt32) + count (UInt64) + count × elem_values
            let elemTypeData = handle.readData(ofLength: 4)
            let countData    = handle.readData(ofLength: 8)
            guard elemTypeData.count == 4, countData.count == 8 else { return }
            let elemType = GGUFValueType(rawValue:
                elemTypeData.withUnsafeBytes { $0.load(as: UInt32.self).littleEndian }) ?? .uint8
            let count    = countData.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
            for _ in 0..<min(count, 65536) {  // cap to avoid hangs on corrupt files
                try skipGGUFValue(type: elemType, from: handle)
            }
        }
    }
}
