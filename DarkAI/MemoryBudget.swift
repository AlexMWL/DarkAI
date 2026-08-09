import Foundation
import Darwin

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
enum MemoryBudget {

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
        let entitled = physicalGB * 0.75
        return max(observed, entitled)
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
                               timeout: TimeInterval = 12) async -> Double {
        let pollInterval: UInt64 = 250_000_000  // 0.25 s
        let deadline = Date().addingTimeInterval(timeout)
        var best = plannableHeadroomGB()
        // Stop early once memory plateaus — waiting out the full timeout for memory that isn't
        // coming back just leaves the user staring at a progress label for no reason.
        var stagnantPolls = 0
        let stagnantLimit = 6   // ~1.5 s of no meaningful improvement

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
                if stagnantPolls >= stagnantLimit { break }
            }
        }

        LogManager.shared.log(String(
            format: "Memory after unload: %.2f GB plannable (needed %.2f GB)", best, targetGB
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
