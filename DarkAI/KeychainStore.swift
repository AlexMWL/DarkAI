import Foundation
import Security

/// Minimal Keychain-backed storage for secrets that don't belong in UserDefaults.
///
/// UserDefaults is an unencrypted plist on disk — fine for ordinary settings, wrong for
/// something like a user-supplied search API key. This wraps just enough of the Keychain API
/// (set/get/delete a single string by key) for that one purpose. It is not a general-purpose
/// Keychain abstraction, and it deliberately has no dependency on anything else in the app.
enum KeychainStore {

    /// Scopes every entry to this app. Matches the identifier prefix already used elsewhere
    /// (e.g. `ModelDownloadManager`'s background `URLSession` identifier).
    private static let service = "com.DDT.DarkAI.secrets"

    private static func query(forKey key: String) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key
        ]
    }

    /// Stores `value` under `key`, replacing any existing entry. Pass `nil` (or an empty string)
    /// to remove the entry instead — callers clearing a settings field don't need a separate
    /// branch for "the user cleared this."
    static func set(_ value: String?, forKey key: String) {
        guard let value, !value.isEmpty else {
            delete(forKey: key)
            return
        }
        guard let data = value.data(using: .utf8) else { return }

        // Update-in-place when an entry already exists, add only when it doesn't — rather than
        // the simpler delete-then-add, which has a real (if narrow) race: if `set` were ever
        // called concurrently for the same key, delete-then-add can interleave and leave the
        // entry deleted, or written by the "wrong" caller. `SecItemUpdate`/`SecItemAdd` each do
        // their one operation atomically at the Keychain layer, so there's no window where the
        // entry doesn't exist between a delete and the following add.
        let updateAttributes: [String: Any] = [kSecValueData as String: data]
        let updateStatus = SecItemUpdate(query(forKey: key) as CFDictionary, updateAttributes as CFDictionary)
        if updateStatus == errSecItemNotFound {
            var attributes = query(forKey: key)
            attributes[kSecValueData as String] = data
            // Available as soon as the user unlocks the device once after a restart, but not
            // before — matches how the rest of the app's data is only meaningful once the user is
            // actively using the phone, and avoids requiring "always available" access for a value
            // that's never touched from a background context.
            attributes[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
            let addStatus = SecItemAdd(attributes as CFDictionary, nil)
            if addStatus != errSecSuccess {
                LogManager.shared.log("KeychainStore: SecItemAdd failed for key '\(key)' — status \(addStatus)")
            }
        } else if updateStatus != errSecSuccess {
            LogManager.shared.log("KeychainStore: SecItemUpdate failed for key '\(key)' — status \(updateStatus)")
        }
    }

    static func get(forKey key: String) -> String? {
        var attributes = query(forKey: key)
        attributes[kSecReturnData as String] = true
        attributes[kSecMatchLimit as String] = kSecMatchLimitOne

        var result: AnyObject?
        let status = SecItemCopyMatching(attributes as CFDictionary, &result)
        guard status == errSecSuccess, let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func delete(forKey key: String) {
        let status = SecItemDelete(query(forKey: key) as CFDictionary)
        // errSecItemNotFound just means there was nothing to delete — every `set(nil, ...)` call
        // on an already-empty key hits this, so it's not a real failure worth logging.
        if status != errSecSuccess && status != errSecItemNotFound {
            LogManager.shared.log("KeychainStore: SecItemDelete failed for key '\(key)' — status \(status)")
        }
    }
}
