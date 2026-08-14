import Foundation
import Darwin
import Metal

/// How much memory this app is allowed to plan against.
///
/// The distinction that matters here: `os_proc_available_memory()` reports headroom against the
/// process's *current* dirty-memory allowance, and iOS hands that allowance out lazily — it grows
/// as the app actually grows. Reading it before allocating therefore understates what the app can
/// have, and understates it worst on exactly the devices with the most memory to give.
///
/// Budgeting straight off that reading is what left a 0.75 GB model with a 512-token context on an
/// 11 GB phone: whatever the app happened to be holding at that instant was treated as the ceiling.
/// This type plans against the allowance the app can *claim* instead, so the KV cache is sized by
/// what the device can actually provide rather than by a snapshot taken at the wrong moment.
/// `nonisolated`: every figure here is a stateless read of system/Mach/Metal APIs, all of which
/// are safe to call from any thread. It has to be — `LlamaRunner`, the actor that plans model
/// offload and sizes the KV cache against these numbers, is not the main actor, and none of this
/// needs the main actor's serialization since there is no shared mutable state to protect.
nonisolated enum MemoryBudget {

    private static let bytesPerGB = 1024.0 * 1024.0 * 1024.0

    /// Physical RAM installed in the device.
    static var physicalGB: Double {
        Double(ProcessInfo.processInfo.physicalMemory) / bytesPerGB
    }

    /// Dirty memory this process is currently holding.
    static func footprintGB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / bytesPerGB
    }

    /// Headroom remaining against the allowance iOS has granted *so far*.
    static func availableNowGB() -> Double {
        Double(os_proc_available_memory()) / bytesPerGB
    }

    /// The ceiling the app plans against.
    ///
    /// Two candidates, and the larger wins:
    ///
    /// * **Observed** — footprint plus remaining headroom, i.e. the allowance iOS has granted at
    ///   this moment. Accurate but pessimistic early on, before the app has grown into its limit.
    /// * **Entitled** — a share of physical RAM. `com.apple.developer.kernel.increased-memory-limit`
    ///   is in this app's entitlements, which lets it claim well beyond the default cap; 75% is a
    ///   deliberately conservative read of that (the entitlement permits appreciably more) and
    ///   leaves the system its own working room.
    ///
    /// Taking the maximum is what "carve out memory rather than read what's free" means in
    /// practice: the app sizes its allocations for the memory it can claim, not the memory it
    /// happens to have been handed before it asked for any.
    static func ceilingGB() -> Double {
        let observed = footprintGB() + availableNowGB()
        return max(observed, entitledGB)
    }

    /// The share of physical RAM this app can expect to claim, as a floor under `ceilingGB()`.
    ///
    /// This was a flat 75% of physical memory, which scales the wrong way. iOS keeps a roughly
    /// *fixed* amount for itself — the kernel, SpringBoard, the media stack — not a fixed
    /// fraction, so the same percentage is far more aggressive on a small device than a large
    /// one. 75% of 4 GB leaves 1 GB for the entire rest of the system, where 75% of 16 GB leaves
    /// 4 GB. The first is a device that jetsams the moment anything else wants memory; the
    /// second is merely cautious.
    ///
    /// Reserving a constant instead means the budget tightens automatically on smaller hardware
    /// and relaxes on larger, which is what makes the same model behave sensibly across the
    /// range rather than being tuned for whichever phone it was last tested on:
    ///
    ///     4 GB  → 1.9 GB   (was 3.0)
    ///     6 GB  → 3.6 GB   (was 4.5)
    ///     8 GB  → 5.3 GB   (was 6.0)
    ///     12 GB → 8.7 GB   (was 9.0)
    ///     16 GB → 12.1 GB  (was 12.0)
    ///
    /// Still a floor, not a cap: `ceilingGB()` takes the larger of this and what iOS has actually
    /// granted, so a device that hands out more than this estimate is not held back by it.
    static var entitledGB: Double {
        max(0.5, physicalGB * 0.85 - 1.5)
    }

    /// What Metal will let this process hold in GPU-resident buffers at once.
    ///
    /// A second ceiling, independent of the jetsam allowance every other figure here describes,
    /// and the one that actually decides whether a GPU-offloaded model runs. An iPhone 17 Pro
    /// reports 11.4 GB of physical memory and 8.59 GB here — so a 6.8 GB model offloaded in full,
    /// plus its KV cache, clears the process budget comfortably and still exceeds what the GPU
    /// can hold. The result is not a refused load: the weights upload, then the first command
    /// buffer of the first message fails with `kIOGPUCommandBufferCallbackErrorOutOfMemory`, the
    /// Metal backend latches into an unrecoverable error state, and because the device is shared
    /// with the rest of the app, its rendering corrupts too.
    ///
    /// Zero when Metal is unavailable, which callers should read as "no GPU constraint to apply"
    /// rather than "no memory".
    static var metalWorkingSetGB: Double {
        guard let device = MTLCreateSystemDefaultDevice() else { return 0 }
        return Double(device.recommendedMaxWorkingSetSize) / bytesPerGB
    }

    /// The budget for GPU-resident weights: whichever of the two ceilings binds first.
    ///
    /// `processHeadroomGB` is what iOS will let the app dirty; `metalWorkingSetGB` is what the
    /// GPU will hold. Planning against only the first is what let a model load and then die on
    /// its first token.
    static func gpuResidentBudgetGB(processHeadroomGB: Double) -> Double {
        let metal = metalWorkingSetGB
        guard metal > 0 else { return processHeadroomGB }
        return min(processHeadroomGB, metal)
    }

    /// Headroom held back for the OS and for transient spikes during a load — mmap page-in, KV
    /// allocation, compute buffer setup — which briefly exceed steady state.
    ///
    /// This was `max(1.0, available * 0.10)`, and the 1 GB floor is what made it wrong on small
    /// devices: on a 4 GB iPhone with roughly 1.9 GB to plan against, a flat gigabyte is half
    /// the entire budget reserved before a single weight is placed, which pushed the layer
    /// budget negative and refused models the device could actually run. Proportional in the
    /// middle, floored low enough to stay affordable on the smallest hardware, and capped so
    /// large devices reserve the same 1 GB they always did.
    static func safetyMarginGB(for availableGB: Double) -> Double {
        min(1.0, max(0.35, availableGB * 0.15))
    }

    /// Non-weight cost of having a model loaded at all: KV cache, compute buffers, and the app
    /// around them.
    ///
    /// This was a flat 2.0 GB in `checkMemorySafety`, which is larger than the entire budget on a
    /// 4 GB iPhone — so every model, of every size, failed the pre-flight there and the device
    /// could load nothing at all. Proportional below the point where 2 GB is affordable, and
    /// unchanged at 2 GB on the large devices it was tuned for.
    static func fixedOverheadGB(for availableGB: Double) -> Double {
        min(2.0, availableGB * 0.25)
    }

    /// Memory the app can still claim: the ceiling minus what it already holds.
    ///
    /// This is the number to size new allocations against. It stays stable as the app grows,
    /// where `availableNowGB()` on its own swings with whatever was resident at the moment of
    /// the reading.
    static func plannableHeadroomGB() -> Double {
        max(0, ceilingGB() - footprintGB())
    }


    /// Waits for memory headroom to actually come back after a large unload.
    ///
    /// Freeing model weights is not synchronous from the process's point of view: Metal returns
    /// buffers to the system on its own schedule, so a reading taken immediately after a free
    /// still shows the old footprint. Anything that unloads one model and loads another has to
    /// wait for that, or the incoming load sizes itself against memory that hasn't come back yet.
    ///
    /// Polls until headroom reaches `targetGB`, until it stops climbing (some of it may simply
    /// not be returning), or until `timeout`. Returns the final reading.
    @discardableResult
    static func waitForRelease(atLeastGB targetGB: Double,
                               timeout: TimeInterval = 20) async -> Double {
        let pollInterval: UInt64 = 250_000_000  // 0.25 s
        let deadline = Date().addingTimeInterval(timeout)
        var best = plannableHeadroomGB()
        var stagnantPolls = 0

        // How long a quiet stretch has to last before this concludes the memory isn't coming back.
        //
        // It used to be a flat ~1.5 s, and that was the bug behind "image generation unloaded
        // everything and then said it was out of memory." Metal does not return a multi-gigabyte
        // allocation in one smooth curve — it hands buffers back in batches, with pauses between
        // them that routinely exceed a second and a half. Giving up during one of those pauses
        // meant the caller ran its safety check against a footprint still inflated by memory that
        // was already free, and refused a model that fit perfectly well moments later.
        //
        // So patience now scales with how much is still missing: near the target, a short quiet
        // stretch really does mean it has plateaued; a long way short, it is far more likely the
        // release is simply mid-batch, and the cost of waiting is a few seconds against the cost
        // of failing a load the device could have handled.
        func stagnantLimit(reached: Double) -> Int {
            guard targetGB > 0 else { return 6 }
            return reached >= targetGB * 0.9 ? 6 : 24   // ~1.5 s vs ~6 s
        }

        while Date() < deadline {
            if best >= targetGB { break }
            try? await Task.sleep(nanoseconds: pollInterval)

            let current = plannableHeadroomGB()
            if current > best + 0.05 {
                best = current
                stagnantPolls = 0
            } else {
                best = max(best, current)
                stagnantPolls += 1
                if stagnantPolls >= stagnantLimit(reached: best) { break }
            }
        }

        LogManager.shared.log(String(
            format: "Memory after unload: %.2f GB plannable (needed %.2f GB) — %@",
            best, targetGB, describe()
        ))
        return best
    }

    static func describe() -> String {
        String(
            format: "physical %.1f GB, footprint %.2f GB, available now %.2f GB, ceiling %.2f GB, plannable %.2f GB",
            physicalGB, footprintGB(), availableNowGB(), ceilingGB(), plannableHeadroomGB()
        )
    }
}
