import Foundation
import LlamaSwift

/// Tokenizer for the Core ML backends — one instance per bundled vocabulary.
///
/// Neither Core ML model this app supports ships its own tokenizer implementation: OpenELM's
/// model card specifies the stock `meta-llama/Llama-2-7b-hf` SentencePiece tokenizer, and
/// Llama-3.2-1B-Instruct's Core ML export (`smpanaro/Llama-3.2-1B-Instruct-CoreML`) uses Meta's
/// own `meta-llama/Llama-3.2-1B` BPE tokenizer. Rather than vendor a SentencePiece *and* a BPE
/// implementation purely to re-derive something the app already has a proven path to, this wraps
/// a **second, independent** `llama.cpp` model load — in `vocab_only` mode, which populates only
/// the vocabulary and skips every tensor — against one of two vocab-only GGUFs bundled with the
/// app, both llama.cpp's own `models/ggml-vocab-*.gguf` test fixtures (MIT licensed, from
/// https://github.com/ggml-org/llama.cpp):
/// - `Resources/llama-spm-vocab.gguf` (`ggml-vocab-llama-spm.gguf`) — Llama-2 SentencePiece, used
///   by `SingleWindowCoreMLEngine` for OpenELM.
/// - `Resources/llama-bpe-vocab.gguf` (`ggml-vocab-llama-bpe.gguf`) — Llama-3 BPE, used by
///   `ChunkedPipelineCoreMLEngine` for Llama 3.2. Confirmed (by inspecting the file directly) to
///   contain the exact `<|begin_of_text|>`/`<|start_header_id|>`/`<|eot_id|>` special tokens
///   Llama 3.2's chat template needs.
///
/// This actor never touches or shares state with `LlamaRunner`'s own loaded GGUF — it has its
/// own independent `model`/`vocab` pointers, and nothing here participates in the GGUF-chat
/// model's memory budgeting or context planning.
actor CoreMLTokenizer {

    enum TokenizerError: Error, LocalizedError {
        case bundledVocabMissing
        case loadFailed

        var errorDescription: String? {
            switch self {
            case .bundledVocabMissing: return "The bundled tokenizer vocabulary is missing from the app."
            case .loadFailed: return "Failed to load the bundled tokenizer vocabulary."
            }
        }
    }

    /// Which bundled vocab-only GGUF this instance loads. `.llama2SentencePiece` is the default
    /// so the existing `CoreMLTokenizer()` call site in `SingleWindowCoreMLEngine` needs no change.
    enum Vocabulary {
        case llama2SentencePiece
        case llama3BytePairEncoding

        var resourceName: String {
            switch self {
            case .llama2SentencePiece: return "llama-spm-vocab"
            case .llama3BytePairEncoding: return "llama-bpe-vocab"
            }
        }
    }

    private let vocabulary: Vocabulary
    private var model: OpaquePointer?
    private var vocab: OpaquePointer?

    init(vocabulary: Vocabulary = .llama2SentencePiece) {
        self.vocabulary = vocabulary
    }

    /// Loads the bundled vocab-only GGUF on first use. Idempotent — a second call after a
    /// successful load is a no-op, and a failed attempt is not silently retried forever (the
    /// bundled resource is static; if it's missing once, it's missing every time).
    private func ensureLoaded() throws {
        guard model == nil else { return }
        guard let url = Bundle.main.url(forResource: vocabulary.resourceName, withExtension: "gguf") else {
            throw TokenizerError.bundledVocabMissing
        }
        var params = llama_model_default_params()
        params.vocab_only = true
        guard let mdl = llama_model_load_from_file(url.path, params) else {
            throw TokenizerError.loadFailed
        }
        model = mdl
        vocab = llama_model_get_vocab(mdl)
    }

    /// Tokenizes `text`. `addBOS` should be `true` for the start of a prompt — OpenELM's model
    /// card calls this out explicitly as required, unlike most instruct models where it's
    /// implied by the chat template.
    func encode(_ text: String, addBOS: Bool) throws -> [Int32] {
        try ensureLoaded()
        let utf8 = text.utf8
        var nTokensMax = Int32(utf8.count + 8)
        var tokens = [llama_token](repeating: 0, count: Int(nTokensMax))
        var n = llama_tokenize(vocab, text, Int32(utf8.count), &tokens, nTokensMax, addBOS, true)
        if n < 0 {
            // llama.cpp's documented convention: a negative return means the supplied buffer was
            // too small, and its magnitude is the size actually needed. `utf8.count + 8` is
            // generous for ordinary BPE/SPM tokenization, but retry once at the reported size
            // rather than silently dropping the tokenization on an input that needs more.
            nTokensMax = -n
            tokens = [llama_token](repeating: 0, count: Int(nTokensMax))
            n = llama_tokenize(vocab, text, Int32(utf8.count), &tokens, nTokensMax, addBOS, true)
        }
        guard n > 0 else { return [] }
        return Array(tokens.prefix(Int(n)))
    }

    /// Detokenizes a single token to its text piece.
    func decode(_ token: Int32) throws -> String {
        try ensureLoaded()
        var bufSize: Int32 = 256
        var buf = [CChar](repeating: 0, count: Int(bufSize))
        var nChars = llama_token_to_piece(vocab, llama_token(token), &buf, bufSize, 0, false)
        if nChars < 0 {
            // Same buffer-too-small convention as `encode` above.
            bufSize = -nChars
            buf = [CChar](repeating: 0, count: Int(bufSize))
            nChars = llama_token_to_piece(vocab, llama_token(token), &buf, bufSize, 0, false)
        }
        guard nChars > 0 else { return "" }
        return String(bytes: buf.prefix(Int(nChars)).map { UInt8(bitPattern: $0) }, encoding: .utf8) ?? ""
    }

    func isEndOfGeneration(_ token: Int32) throws -> Bool {
        try ensureLoaded()
        return llama_vocab_is_eog(vocab, llama_token(token))
    }

    deinit {
        if let model { llama_model_free(model) }
    }
}
