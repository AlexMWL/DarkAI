import Foundation
import LlamaSwift

/// Structural facts about a GGUF file, read from its header without loading any weights.
///
/// The figure that matters here is how the file's bytes divide between tensors every token
/// needs and tensors only *some* tokens need. A dense model has no second category: every
/// weight participates in every forward pass, so streaming any of it from storage costs a
/// re-read per token. A mixture-of-experts model is the opposite — its routed expert stacks
/// are the bulk of the file, and a single token activates only a handful of them.
///
/// That distinction is what makes expert streaming worth doing where dense streaming isn't,
/// so the planner needs the split measured rather than guessed. These numbers come from the
/// GGUF tensor directory, which is exact, instead of from a parameter-count model that would
/// have to assume a quantisation mix per tensor class.
struct ModelProfile {

    /// Total file size.
    let totalGB: Double

    /// Hyperparameters, read from GGUF metadata.
    ///
    /// These deliberately do *not* come from a `vocab_only` model load, which is where they were
    /// read from originally and which does not work: that mode populates the vocabulary and
    /// nothing else, returning 0 for `n_layer`, `n_embd` and `n_ctx_train`, and aborting the
    /// process outright inside `llama_model_n_head_kv` — its bounds check tests the requested
    /// layer against `n_layer`, so with `n_layer == 0` even layer 0 is out of range. Reading the
    /// header directly is both correct and cheaper: it skips building a 128K-token tokenizer
    /// with 280K merges just to ask how many blocks the model has.
    let nLayer: Int
    let nHeadKV: Int
    let headDimK: Int
    let headDimV: Int
    let trainedContext: Int
    let vocabSize: Int

    /// K and V elements cached per token across every layer.
    var kvElementsPerToken: Double {
        Double(nLayer * nHeadKV) * Double(headDimK + headDimV)
    }

    /// A quantised cache needs each head dimension to divide evenly into the 32-element block.
    var supportsQuantizedKVCache: Bool {
        headDimK > 0 && headDimV > 0 && headDimK % 32 == 0 && headDimV % 32 == 0
    }

    /// Whether the geometry is complete enough to budget a KV cache from.
    var hasUsableGeometry: Bool { nLayer > 0 && nHeadKV > 0 && headDimK > 0 && headDimV > 0 }

    /// Routed-expert FFN bytes in each block, keyed by block index. Empty for a dense model.
    let expertGBByLayer: [Int: Double]

    /// Share of each block's experts a single token actually routes to — `expert_used_count`
    /// over `expert_count`. Both models this was built against use 1/8.
    ///
    /// This is the honest estimate of how much of a pinned expert stack is hot at any moment,
    /// and so of how much of it the page cache is really holding. Bounded at both ends: never
    /// below a tenth, because routing spreads across experts as a reply goes on and the cache
    /// keeps more than one token's worth, and never above a quarter, which was the flat figure
    /// used before this was read from the file.
    let routedFraction: Double

    var isMixtureOfExperts: Bool { !expertGBByLayer.isEmpty }

    /// Bytes held by routed experts across every block.
    var expertGB: Double { expertGBByLayer.values.reduce(0, +) }

    /// Everything that is not a routed expert: attention, norms, the router itself, token
    /// embeddings, the output projection, and any shared/always-on expert. Every token needs
    /// all of it, so this is the part that has to stay resident to keep the model usable.
    var denseGB: Double { max(0, totalGB - expertGB) }

    /// Every block that has routed experts, in order.
    var expertLayers: [Int] { expertGBByLayer.keys.sorted() }
}

/// Reads `ModelProfile`s, cached by file identity.
///
/// `gguf_init_from_file` with `no_alloc` parses the header, metadata and tensor directory and
/// stops before any tensor data, so a profile costs a header read rather than a model load.
/// That is still real file I/O, and the model picker asks for one per row as the list scrolls,
/// hence the cache — keyed by size and modification date so replacing a file at the same path
/// re-reads it rather than serving a stale answer.
enum ModelProfiler {

    private static let lock = NSLock()
    private static var cache: [String: ModelProfile] = [:]

    static func profile(path: String) -> ModelProfile? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attributes[.size] as? Int64, size > 0 else { return nil }
        let modified = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let key = "\(path)|\(size)|\(modified)"

        lock.lock()
        let cached = cache[key]
        lock.unlock()
        if let cached { return cached }

        guard let profile = read(path: path, totalBytes: size) else { return nil }

        lock.lock()
        cache[key] = profile
        lock.unlock()
        return profile
    }

    private static let bytesPerGB = 1024.0 * 1024.0 * 1024.0

    private static func read(path: String, totalBytes: Int64) -> ModelProfile? {
        var params = gguf_init_params()
        params.no_alloc = true
        params.ctx = nil

        guard let ctx = gguf_init_from_file(path, params) else { return nil }
        defer { gguf_free(ctx) }

        var expertBytesByLayer: [Int: Double] = [:]
        var tensorBytes = 0.0
        for index in 0..<gguf_get_n_tensors(ctx) {
            guard let cName = gguf_get_tensor_name(ctx, index) else { continue }
            let size = Double(gguf_get_tensor_size(ctx, index))
            tensorBytes += size
            guard let layer = expertLayerIndex(in: String(cString: cName)) else { continue }
            expertBytesByLayer[layer, default: 0] += size
        }

        // A GGUF header describes the whole model even when the file holding it is a partial
        // download, so the two can disagree — and when they do, every byte figure derived from
        // the header is describing a model that isn't all there. An offload plan built on that
        // would place tensors the load is going to fail on anyway, so decline to profile and let
        // the caller fall back to size-only planning.
        guard tensorBytes <= Double(totalBytes) else { return nil }

        let arch = stringValue(ctx, "general.architecture") ?? ""
        let nLayer = Int(uintValue(ctx, "\(arch).block_count") ?? 0)
        let nHead = Int(uintValue(ctx, "\(arch).attention.head_count") ?? 0)
        let nHeadKV = Int(uintValue(ctx, "\(arch).attention.head_count_kv") ?? UInt32(nHead))
        let nEmbd = Int(uintValue(ctx, "\(arch).embedding_length") ?? 0)

        // Most architectures declare head dimensions explicitly; the rest imply them from
        // embedding width over head count. Reading the explicit key first matters more than it
        // looks: GPT-OSS declares 64 while `n_embd / n_head` works out to 45, which would both
        // mis-size the cache and wrongly rule out the quantised one.
        let fallbackHeadDim = nHead > 0 ? nEmbd / nHead : 0
        let headDimK = Int(uintValue(ctx, "\(arch).attention.key_length") ?? 0).nonZero ?? fallbackHeadDim
        let headDimV = Int(uintValue(ctx, "\(arch).attention.value_length") ?? 0).nonZero ?? headDimK

        return ModelProfile(
            totalGB: Double(totalBytes) / bytesPerGB,
            nLayer: nLayer,
            nHeadKV: nHeadKV,
            headDimK: headDimK,
            headDimV: headDimV,
            trainedContext: Int(uintValue(ctx, "\(arch).context_length") ?? 0),
            vocabSize: vocabSize(ctx, arch: arch),
            expertGBByLayer: expertBytesByLayer.mapValues { $0 / bytesPerGB },
            routedFraction: routedFraction(ctx, arch: arch)
        )
    }

    /// Token count, taken from the length of the tokenizer's own token array.
    ///
    /// Preferred over `{arch}.vocab_size`, which plenty of models simply don't carry — and this
    /// figure decides the large-vocab GPU-offload cap, so silently reading zero would disable a
    /// workaround that exists to stop Gemma-class models producing zeroed Metal buffers.
    private static func vocabSize(_ ctx: OpaquePointer, arch: String) -> Int {
        let tokens = gguf_find_key(ctx, "tokenizer.ggml.tokens")
        if tokens >= 0, gguf_get_kv_type(ctx, tokens) == GGUF_TYPE_ARRAY {
            return Int(gguf_get_arr_n(ctx, tokens))
        }
        return Int(uintValue(ctx, "\(arch).vocab_size") ?? 0)
    }

    /// `expert_used_count / expert_count`, clamped. Falls back to the flat quarter if either key
    /// is missing, which is the conservative direction — it charges more memory, not less.
    private static func routedFraction(_ ctx: OpaquePointer, arch: String) -> Double {
        guard let total = uintValue(ctx, "\(arch).expert_count"), total > 0,
              let used = uintValue(ctx, "\(arch).expert_used_count") else { return 0.25 }
        return min(0.25, max(0.10, Double(used) / Double(total)))
    }

    /// Reads a scalar unsigned key, tolerating the several widths GGUF allows for one.
    ///
    /// Per-layer array forms (a few architectures vary head counts by block) resolve to the
    /// largest element, so a KV budget built on this over-estimates rather than under-estimates.
    private static func uintValue(_ ctx: OpaquePointer, _ key: String) -> UInt32? {
        let id = gguf_find_key(ctx, key)
        guard id >= 0 else { return nil }
        switch gguf_get_kv_type(ctx, id) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id)
        case GGUF_TYPE_UINT16: return UInt32(gguf_get_val_u16(ctx, id))
        case GGUF_TYPE_UINT8:  return UInt32(gguf_get_val_u8(ctx, id))
        case GGUF_TYPE_INT32:  return UInt32(max(0, gguf_get_val_i32(ctx, id)))
        case GGUF_TYPE_ARRAY:
            guard gguf_get_arr_type(ctx, id) == GGUF_TYPE_UINT32,
                  let raw = gguf_get_arr_data(ctx, id) else { return nil }
            let count = Int(gguf_get_arr_n(ctx, id))
            guard count > 0 else { return nil }
            let values = raw.assumingMemoryBound(to: UInt32.self)
            return (0..<count).map { values[$0] }.max()
        default: return nil
        }
    }

    private static func stringValue(_ ctx: OpaquePointer, _ key: String) -> String? {
        let id = gguf_find_key(ctx, key)
        guard id >= 0, gguf_get_kv_type(ctx, id) == GGUF_TYPE_STRING else { return nil }
        return String(cString: gguf_get_val_str(ctx, id))
    }

    /// Block index for a routed-expert FFN tensor, e.g. `blk.12.ffn_up_exps.weight` → 12.
    ///
    /// Matched against an explicit list rather than by looking for `exps` anywhere in the name,
    /// because several neighbouring tensors are *not* streamable and misreading one as an expert
    /// would under-report what the model needs resident on every single forward pass:
    /// `ffn_up` is a dense FFN, `ffn_up_shexp` is a shared expert every token passes through,
    /// `ffn_gate_inp` is the router that decides which experts a token wants, and
    /// `ffn_norm_exps` is a normalisation weight. `ffn_gate_up_exps` is the fused gate+up form
    /// some architectures use, and does belong here.
    ///
    /// `ffn_*_chexps` is deliberately left out. It appears in the tensor-name table but its
    /// routing semantics aren't established here, and the cost of the two mistakes is
    /// asymmetric: omitting a genuine expert only forgoes some memory that could have been
    /// streamed, while including a tensor every token needs would silently under-budget the
    /// load. If a model using that form ever needs support, confirm what it is first.
    private static func expertLayerIndex(in name: String) -> Int? {
        guard name.hasPrefix("blk.") else { return nil }
        let parts = name.split(separator: ".")
        guard parts.count >= 3, let layer = Int(parts[1]) else { return nil }
        switch parts[2] {
        case "ffn_up_exps", "ffn_down_exps", "ffn_gate_exps", "ffn_gate_up_exps": return layer
        default: return nil
        }
    }
}

private extension Int {
    /// `nil` for zero, so a missing-or-zero GGUF key can fall through to a computed default.
    var nonZero: Int? { self == 0 ? nil : self }
}
