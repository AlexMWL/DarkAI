import Foundation

/// Which engine `LLMManager` is currently dispatching chat generation to.
///
/// A plain enum rather than a shared `ModelRunner` protocol: the two backends' `load` signatures
/// are irreducibly different (GGUF needs `availableMemoryGB`/`modelSizeGB`/`contextLimit`; Core
/// ML needs none of that, plus its own tokenizer), so a shared protocol would only produce a
/// lowest-common-denominator signature both sides partly ignore. Revisit only if a third backend
/// shows up.
enum LLMBackendKind: Equatable {
    case llamaCpp
    case coreML
}
