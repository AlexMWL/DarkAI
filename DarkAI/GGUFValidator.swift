import Foundation

/// Lightweight pure-Swift model-file validation. Header-only: nothing here reads tensor *data*,
/// so every check below costs the same on a 6 GB checkpoint as on a 200 MB one and runs in
/// milliseconds.
///
/// Two jobs, despite the name — `validate` handles GGUF chat models, and
/// `validateDiffusionCheckpoint` handles diffusion checkpoints in either GGUF or safetensors,
/// which is why the safetensors reader lives here too.
///
/// The GGUF binary format (v2/v3) header layout:
///   [0..3]   magic:        "GGUF"  (4 bytes, ASCII)
///   [4..7]   version:      UInt32  (must be 2 or 3)
///   [8..15]  tensor_count: UInt64
///   [16..23] kv_count:     UInt64
///   [24..]   kv pairs:     (key_len: UInt64, key: [UInt8], value_type: UInt32, value: ...)
///   [...]    tensor info:  (name_len: UInt64, name: [UInt8], n_dims, dims…, type, offset)
///
/// The safetensors layout is simpler: an 8-byte little-endian header length followed by that
/// many bytes of JSON, whose keys are the tensor names.
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

    // MARK: - Diffusion checkpoint compatibility

    /// ggml stores a tensor's name in a fixed `char name[GGML_MAX_NAME]` buffer.
    ///
    /// stable-diffusion.cpp's CMakeLists compiles its ggml with `GGML_MAX_NAME=160` precisely
    /// because SD checkpoints carry names around 83 bytes. Keep this in sync with that value —
    /// if the vendored library's setting ever changes, this check silently becomes wrong.
    private static let ggmlMaxNameBytes = 160

    /// Rejects diffusion checkpoints whose tensor names are too long for ggml to hold.
    ///
    /// Backstop rather than the primary defence. The crash this originally guarded against —
    /// names truncated to 63 characters, CLIP failing to resolve its weights, and
    /// `GGML_ASSERT(!chunk_hidden_states.empty())` calling `abort()` — turned out to be a
    /// *linking* fault, not a file problem: the app links two builds of ggml (this library's,
    /// with `GGML_MAX_NAME=160`, and llama.cpp's, with the default 64) and SD's calls were
    /// binding to llama's. `build_sd_ios.sh` now demotes this library's ggml symbols to private
    /// extern so each side calls the ggml it was compiled against.
    ///
    /// The check stays because the failure mode it catches is uncatchable once it happens: the
    /// `abort()` fires inside C++ on a background thread, so the process dies with no chance to
    /// show an error. A checkpoint that genuinely exceeds 160 bytes, or a future regression in
    /// the symbol privatisation, should surface as a readable message rather than a silent death.
    static func validateDiffusionCheckpoint(path: String) throws {
        let url = URL(fileURLWithPath: path)
        let fileName = url.lastPathComponent

        guard let handle = FileHandle(forReadingAtPath: path) else {
            throw error("Cannot open \(fileName) for reading.")
        }
        defer { handle.closeFile() }

        let tensorNames: [String]
        switch url.pathExtension.lowercased() {
        case "gguf":
            tensorNames = try ggufTensorNames(from: handle, fileName: fileName)
        case "safetensors":
            tensorNames = try safetensorsTensorNames(from: handle, fileName: fileName)
        default:
            // Legacy `.ckpt` is a pickle inside a zip — enumerating its tensor names means
            // implementing a pickle reader, which is both a lot of surface area and the exact
            // format whose deserialisation is a known security hazard. Left to the library,
            // as before. `.safetensors` and `.gguf` are the formats this can actually verify,
            // and they're what the catalog and effectively all current checkpoints use.
            LogManager.shared.log("Checkpoint check skipped for \(fileName) — unverifiable format")
            return
        }

        try validateComponents(tensorNames, fileName: fileName)
    }

    // MARK: Component completeness

    /// Confirms a checkpoint carries all three pieces stable-diffusion.cpp needs to render an
    /// image on its own: the diffusion UNet, a text encoder, and a VAE.
    ///
    /// Everything rejected here is a real file people genuinely try to load, because Civitai
    /// and Hugging Face list them side by side with full checkpoints and the extension is the
    /// same. A LoRA, a ControlNet, a textual-inversion embedding, or a bare VAE is *not* a
    /// model this app can run — but the loader doesn't say so. Depending on which piece is
    /// missing it either fails deep inside C++ with an assertion (uncatchable — the process
    /// dies with no error to show), or "succeeds" and emits pure noise, which reads to the user
    /// as the app being broken rather than the file being the wrong kind of thing.
    ///
    /// The VAE requirement is not a nicety: it's the component that turns the denoised latent
    /// into pixels. A checkpoint without one loads and samples happily and then produces
    /// coloured static.
    private static func validateComponents(_ names: [String], fileName: String) throws {
        guard !names.isEmpty else {
            throw error("\(fileName) contains no tensors — the file is empty or corrupt.")
        }

        func anyName(where predicate: (String) -> Bool) -> Bool { names.contains(where: predicate) }

        // — Wrong *kind* of file. Checked first: these have unambiguous signatures, and naming
        //   the actual thing the user picked is far more useful than "missing a VAE".
        if anyName(where: { $0.contains("lora_up.") || $0.contains("lora_down.")
                         || $0.contains("lora_A") || $0.contains("lora_B")
                         || $0.hasPrefix("lora_unet") || $0.hasPrefix("lora_te") }) {
            throw error("""
            \(fileName) is a LoRA, not a full model.

            A LoRA is a small add-on that adjusts an existing checkpoint — it has no image \
            model of its own, so it can't generate anything by itself. Import a full \
            checkpoint instead.
            """)
        }
        if anyName(where: { $0.hasPrefix("control_model.") || $0.contains("controlnet") }) {
            throw error("""
            \(fileName) is a ControlNet, not a full model.

            ControlNets guide an existing checkpoint using a reference image. They can't \
            generate images on their own, and this app has no ControlNet pipeline. Import a \
            full checkpoint instead.
            """)
        }
        if anyName(where: { $0.contains("string_to_param") || $0.contains("emb_params") }) {
            throw error("""
            \(fileName) is a textual-inversion embedding, not a full model.

            Embeddings add a single trained concept to an existing checkpoint. Import a full \
            checkpoint instead.
            """)
        }

        // — Component detection. Prefixes cover the single-file layouts stable-diffusion.cpp
        //   accepts: SD 1.x/2.x (`cond_stage_model`), SDXL (`conditioner.embedders`, two text
        //   encoders), and the flatter naming some GGUF conversions use.
        let hasUNet = anyName {
            $0.hasPrefix("model.diffusion_model.") || $0.hasPrefix("diffusion_model.")
                || $0.hasPrefix("unet.") || $0.hasPrefix("model.unet.")
        }
        let hasTextEncoder = anyName {
            $0.hasPrefix("cond_stage_model.") || $0.hasPrefix("conditioner.embedders.")
                || $0.hasPrefix("text_encoders.") || $0.hasPrefix("text_encoder")
                || $0.hasPrefix("te.") || $0.hasPrefix("te1.") || $0.hasPrefix("te2.")
        }
        let hasVAE = anyName {
            $0.hasPrefix("first_stage_model.") || $0.hasPrefix("vae.")
                || $0.hasPrefix("model.vae.")
        }

        // A file with only VAE-shaped tensors and nothing else is a standalone VAE — the thing
        // people download to *pair with* a checkpoint. Detected by absence of the other two
        // rather than by prefix, because a bare VAE's keys start at `decoder.`/`encoder.`
        // with no `first_stage_model.` wrapper at all.
        if !hasUNet && !hasTextEncoder {
            let looksLikeBareVAE = anyName {
                $0.hasPrefix("decoder.") || $0.hasPrefix("encoder.")
                    || $0.hasPrefix("quant_conv") || $0.hasPrefix("post_quant_conv")
            }
            if looksLikeBareVAE || hasVAE {
                throw error("""
                \(fileName) is a standalone VAE, not a full model.

                A VAE is one component of a checkpoint — the part that turns the generated \
                latent into a picture. On its own it can't generate anything. Import a full \
                checkpoint that already has a VAE baked in.
                """)
            }
        }

        // — Missing pieces of an otherwise checkpoint-shaped file.
        var missing: [String] = []
        if !hasUNet          { missing.append("the diffusion model (UNet)") }
        if !hasTextEncoder   { missing.append("a text encoder (CLIP)") }
        if !hasVAE           { missing.append("a VAE") }

        guard missing.isEmpty else {
            let list = missing.count == 1
                ? missing[0]
                : missing.dropLast().joined(separator: ", ") + " and " + missing[missing.count - 1]
            throw error("""
            \(fileName) isn't a complete checkpoint.

            It's missing \(list). \(AppInfo.displayName) needs all three parts in one file — \
            without them the model either fails to load or produces noise instead of an image.

            Look for a "full" or "all-in-one" version of this model, one that lists a baked-in \
            VAE. Pruned or "UNet only" downloads won't work.
            """)
        }

        LogManager.shared.log("Checkpoint verified: \(fileName) — \(names.count) tensors, UNet + text encoder + VAE present")
    }

    // MARK: - Resident weight size

    /// How much RAM a checkpoint's weights actually occupy once loaded, which is **not** the
    /// same as its size on disk.
    ///
    /// stable-diffusion.cpp has no 8-bit float tensor type. `safetensors_io.cpp` maps both
    /// `F8_E4M3` and `F8_E5M2` to `GGML_TYPE_F16`, and `tensor_storage.h` reads
    /// `nbytes() / 2` from disk and expands each byte into two — so an FP8 checkpoint takes
    /// **double** its file size in memory. A 4 GB FP8 SDXL file needs ~8 GB resident.
    ///
    /// Budgeting off file size therefore understated FP8 checkpoints by 2×, which is exactly
    /// how a 4.05 GB file was labelled SAFE on an 11.4 GB device and then killed the app
    /// mid-generation. `nil` means "couldn't determine" — callers fall back to file size,
    /// which is correct for GGUF (whose types are already ggml-native, so no conversion
    /// happens) and for ordinary F16 safetensors.
    static func residentWeightBytes(path: String) -> Int64? {
        let url = URL(fileURLWithPath: path)
        guard url.pathExtension.lowercased() == "safetensors",
              let handle = FileHandle(forReadingAtPath: path) else { return nil }
        defer { handle.closeFile() }
        guard let header = try? safetensorsHeader(from: handle, fileName: url.lastPathComponent) else { return nil }

        var total: Int64 = 0
        for (name, value) in header where name != "__metadata__" {
            guard let entry = value as? [String: Any],
                  let dtype = entry["dtype"] as? String,
                  let shape = entry["shape"] as? [Any] else { continue }
            let elements = shape.reduce(Int64(1)) { partial, dim in
                partial * Int64((dim as? NSNumber)?.int64Value ?? 1)
            }
            total += elements * Int64(inMemoryBytesPerElement(forDType: dtype))
        }
        return total > 0 ? total : nil
    }

    /// Bytes each element occupies **after** the loader's dtype conversion — mirrors
    /// `safetensors_dtype_to_ggml_type` in the vendored `safetensors_io.cpp`.
    private static func inMemoryBytesPerElement(forDType dtype: String) -> Int {
        switch dtype.uppercased() {
        case "F8_E4M3", "F8_E5M2": return 2   // → GGML_TYPE_F16: one disk byte becomes two
        case "F16", "BF16":        return 2
        case "F32", "F64":         return 4   // F64 → GGML_TYPE_F32
        case "I32", "I64":         return 4   // I64 → GGML_TYPE_I32
        case "I16", "U16":         return 2
        case "I8", "U8", "BOOL":   return 1
        default:                   return 2   // unknown: assume F16, the common case
        }
    }

    // MARK: Tensor-name enumeration

    /// Reads and parses the safetensors header: an 8-byte little-endian length, then that many
    /// bytes of JSON. Only the header is read, so this costs the same on a 2 GB file as on a
    /// 200 MB one.
    private static func safetensorsHeader(from handle: FileHandle, fileName: String) throws -> [String: Any] {
        guard let lengthData = try? handle.read(upToCount: 8), lengthData.count == 8 else {
            throw error("\(fileName) is too short to be a valid safetensors file.")
        }
        let headerLength = lengthData.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
        // Real headers run from a few KB to a couple of MB. A wild value here means the file
        // isn't safetensors at all (an HTML error page saved under the extension, say).
        guard headerLength > 0, headerLength < 100_000_000 else {
            throw error("\(fileName) is not a valid safetensors file (implausible header size).")
        }
        guard let headerData = try? handle.read(upToCount: Int(headerLength)),
              headerData.count == Int(headerLength) else {
            throw error("\(fileName) has a truncated safetensors header — the file may be an incomplete download.")
        }
        guard let json = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw error("\(fileName) is not a valid safetensors file (unreadable header).")
        }
        return json
    }

    private static func safetensorsTensorNames(from handle: FileHandle, fileName: String) throws -> [String] {
        // `__metadata__` is the format's own reserved key, not a tensor.
        try safetensorsHeader(from: handle, fileName: fileName).keys.filter { $0 != "__metadata__" }
    }

    /// Walks the GGUF metadata block to the tensor-info section and collects every tensor name,
    /// enforcing the `GGML_MAX_NAME` ceiling on the way past.
    private static func ggufTensorNames(from handle: FileHandle, fileName: String) throws -> [String] {
        guard let magic = try? handle.read(upToCount: 4), magic == Data("GGUF".utf8) else {
            throw error("\(fileName) is not a valid GGUF file.")
        }
        let versionData = handle.readData(ofLength: 4)
        let tensorCountData = handle.readData(ofLength: 8)
        let kvCountData = handle.readData(ofLength: 8)
        guard versionData.count == 4, tensorCountData.count == 8, kvCountData.count == 8 else {
            throw error("\(fileName) has a truncated GGUF header.")
        }
        let tensorCount = tensorCountData.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
        let kvCount = kvCountData.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }

        // Walk past the metadata block to reach the tensor-info section.
        for _ in 0..<kvCount {
            guard (try? readGGUFString(from: handle)) != nil else { return [] }
            let vtData = handle.readData(ofLength: 4)
            guard vtData.count == 4 else { return [] }
            let vtRaw = vtData.withUnsafeBytes { $0.load(as: UInt32.self).littleEndian }
            guard let vt = GGUFValueType(rawValue: vtRaw) else { return [] }
            guard (try? skipGGUFValue(type: vt, from: handle)) != nil else { return [] }
        }

        // Capped so a corrupt count can't spin. Real SD/SDXL checkpoints run 1,100–2,600
        // tensors, so this reads all of them with room to spare.
        var names: [String] = []
        var longestName = 0
        var longestExample = ""
        let scanLimit = min(tensorCount, 8192)

        for _ in 0..<scanLimit {
            guard let name = try? readGGUFString(from: handle) else { break }
            names.append(name)
            let byteCount = name.utf8.count
            if byteCount > longestName {
                longestName = byteCount
                longestExample = name
            }
            // n_dims, dims[n_dims], type, offset
            let dimsData = handle.readData(ofLength: 4)
            guard dimsData.count == 4 else { break }
            let nDims = dimsData.withUnsafeBytes { $0.load(as: UInt32.self).littleEndian }
            guard nDims <= 8 else { break }
            let skip = Int(nDims) * 8 + 4 + 8
            guard handle.readData(ofLength: skip).count == skip else { break }
        }

        guard longestName < ggmlMaxNameBytes else {
            throw error("""
            \(fileName) can't be used on this device.

            Its tensor names are longer than this build supports (\(longestName) characters, \
            limit \(ggmlMaxNameBytes - 1)). Loading it would crash the app rather than fail \
            cleanly, so it has been rejected.

            Try a different conversion of the same model.

            Longest name: \(longestExample)
            """)
        }
        return names
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
