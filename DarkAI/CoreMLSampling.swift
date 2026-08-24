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

        // Filter to plausible candidates before sorting, same technique (and same threshold) as
        // `LLMManager`'s llama.cpp sampler: sorting and computing a probability for the *whole*
        // vocabulary — up to 128K entries for the chunked pipeline's model — on every single
        // generated token is the dominant cost here, and everything past ~12 log-units below the
        // max is negligible enough that top-p (0.9) never reaches it anyway.
        var candidates: [(Int, Float)] = []
        candidates.reserveCapacity(1000)
        let logitThreshold = maxLogit - 12.0
        for (i, v) in row.enumerated() where v.isFinite {
            let scaled = v * invTemp
            if scaled > logitThreshold { candidates.append((i, scaled)) }
        }
        guard !candidates.isEmpty else { return 0 }

        candidates.sort { $0.1 > $1.1 }
        var expValues: [Float] = []
        expValues.reserveCapacity(candidates.count)
        var sum: Float = 0
        for (_, val) in candidates {
            let p = expf(val - maxLogit)
            expValues.append(p)
            sum += p
        }
        guard sum > 0 else { return 0 }

        var cumulative: Float = 0
        var cutoff = candidates.count
        for rank in 0..<candidates.count {
            cumulative += expValues[rank] / sum
            if cumulative >= 0.9 { cutoff = rank + 1; break }
        }
        let candidateSum = expValues[0..<cutoff].reduce(0, +)
        guard candidateSum > 0 else { return Int32(candidates[0].0) }

        let r = Float.random(in: 0..<candidateSum)
        var acc: Float = 0
        for rank in 0..<cutoff {
            acc += expValues[rank]
            if r <= acc { return Int32(candidates[rank].0) }
        }
        return Int32(candidates[cutoff - 1].0)
    }
}
