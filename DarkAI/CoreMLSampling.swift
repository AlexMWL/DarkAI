import Foundation

/// Shared next-token sampler for both Core ML engines — the minimum viable subset of what
/// `LlamaRunner`'s sampler does (temperature + a fixed top-p of 0.9), reimplemented over a raw
/// `Float` logit buffer since neither `MLMultiArray` nor Core ML's own model outputs are
/// interchangeable with llama.cpp's logit representation.
///
/// Pulled out as a standalone utility (rather than left private to `SingleWindowCoreMLEngine`, its
/// original home) because `ChunkedPipelineCoreMLEngine` needs the exact same sampling behavior
/// over its own raw logit arrays — see the deliberate deviation from `coreml-llm-cli`'s
/// argmax-only `logit-processor.mlmodelc` documented on that type: this app already has a working
/// temperature control that model doesn't support, so it's reused here instead of loading a second
/// model that would only ever do less.
nonisolated enum CoreMLSampling {
    static func sample(row: UnsafeBufferPointer<Float>, temperature: Float) -> Int32 {
        guard temperature > 0.01 else {
            var bestIdx = 0
            var bestVal = -Float.greatestFiniteMagnitude
            for (i, v) in row.enumerated() where v.isFinite && v > bestVal {
                bestVal = v
                bestIdx = i
            }
            return Int32(bestIdx)
        }

        let invTemp = 1.0 / max(temperature, 0.01)
        var maxLogit = -Float.greatestFiniteMagnitude
        for v in row where v.isFinite { maxLogit = max(maxLogit, v * invTemp) }

        var probs = [Float](repeating: 0, count: row.count)
        var sum: Float = 0
        for (i, v) in row.enumerated() {
            let p = v.isFinite ? expf(v * invTemp - maxLogit) : 0
            probs[i] = p
            sum += p
        }
        guard sum > 0 else { return 0 }
        for i in probs.indices { probs[i] /= sum }

        let sortedIdx = probs.indices.sorted { probs[$0] > probs[$1] }
        var cumulative: Float = 0
        var cutoff = sortedIdx.count
        for (rank, idx) in sortedIdx.enumerated() {
            cumulative += probs[idx]
            if cumulative >= 0.9 { cutoff = rank + 1; break }
        }
        let candidates = sortedIdx.prefix(cutoff)
        let candidateSum = candidates.reduce(Float(0)) { $0 + probs[$1] }
        guard candidateSum > 0 else { return Int32(sortedIdx.first ?? 0) }

        let r = Float.random(in: 0..<candidateSum)
        var acc: Float = 0
        for idx in candidates {
            acc += probs[idx]
            if r <= acc { return Int32(idx) }
        }
        return Int32(candidates.last ?? 0)
    }
}
