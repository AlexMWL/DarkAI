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

        // Delete-then-add rather than update: simpler than handling the "does an entry already
        // exist" branch, and this is called rarely (a settings field, not a hot path).
        SecItemDelete(query(forKey: key) as CFDictionary)

        var attributes = query(forKey: key)
        attributes[kSecValueData as String] = data
        // Available as soon as the user unlocks the device once after a restart, but not
        // before — matches how the rest of the app's data is only meaningful once the user is
        // actively using the phone, and avoids requiring "always available" access for a value
        // that's never touched from a background context.
        attributes[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        SecItemAdd(attributes as CFDictionary, nil)
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
        SecItemDelete(query(forKey: key) as CFDictionary)
    }
}
