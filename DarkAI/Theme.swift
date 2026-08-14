import SwiftUI
import Combine

/// Semantic palette. Every colour resolves per trait collection, so the whole app follows the
/// active appearance without a single call site changing.
///
/// These were plain constants before, which is why the app had to force `.preferredColorScheme(.dark)`
/// — the palette had no light values to offer. Each entry is now a dynamic `UIColor`, resolved by
/// UIKit at draw time against whatever `.preferredColorScheme` the root sets. That means the
/// existing several-hundred `Theme.x` references keep working untouched.
///
/// Light values were picked for contrast, not by inverting the dark ones: the dark theme's red on
/// near-black is legible, but the same red on white measures about 3.9:1 and fails AA for body
/// text, so the light theme uses a deepened red and near-black text instead.
struct Theme {

    // Backed by asset-catalog colour sets, each carrying a light and a dark variant.
    //
    // The first attempt used `Color(UIColor { traits in ... })` statics. Those looked right but
    // resolved once and stuck: with the app in light mode the status bar flipped correctly while
    // every `Theme` surface stayed dark, because SwiftUI had already resolved the wrapped
    // `UIColor` against the launch-time trait collection. Asset catalogue colours are resolved by
    // the rendering system on every draw, which is the behaviour this needs.

    /// Page background.
    static let background     = Color("ThemeBackground")
    /// Raised surfaces — cards, bubbles, input wells.
    static let cardBackground = Color("ThemeCardBackground")
    /// Primary action red. Deepened in light mode so white text on it stays legible — the dark
    /// theme's #FF0033 measures about 3.9:1 on white and fails AA for body text.
    static let accent         = Color("ThemeAccent")
    /// Secondary accent (orange-red).
    static let accentCyan     = Color("ThemeAccentCyan")
    /// Tertiary accent (crimson).
    static let accentRose     = Color("ThemeAccentRose")

    static let textPrimary    = Color("ThemeTextPrimary")
    /// ~9:1 on the dark background, ~7:1 on the light one.
    static let textSecondary  = Color("ThemeTextSecondary")
    // The dark value was #7F4F4F, which measured about 3.1:1 against the background — under the
    // 4.5:1 WCAG AA floor for body text, and that was before the animated backdrop behind it.
    // #9E7070 keeps the same muted role in the hierarchy at ~4.9:1; the light value is ~4.8:1.
    static let textMuted      = Color("ThemeTextMuted")

    static let border         = Color("ThemeBorder")
    static let glowColor      = Color("ThemeGlow")

    /// Text and icons drawn *on top of* a saturated accent fill (a red button, a purple pill).
    /// Deliberately white in both appearances: those fills are dark enough for white to read
    /// against in light mode too, so this must NOT follow the appearance the way `textPrimary`
    /// does — flipping it to near-black in light mode would make every filled button illegible.
    ///
    /// It exists to tell the two cases apart at a glance. A bare `.white` gives no signal about
    /// whether it's correct-on-a-red-button or a light-mode bug, and the app shipped both:
    /// "CHATS" in the chat drawer and the failsafe dialog's title were white on near-white
    /// surfaces and simply disappeared. Anywhere text sits on a *theme* surface
    /// (`background`, `cardBackground`, `border`) it wants `textPrimary`, not this.
    static let onAccent       = Color.white

    /// Opaque chrome fill for bars that sit above content (header, input row, status strip).
    /// These were `Color.black.opacity(0.9)`, which stayed black in light mode.
    static let chrome         = Color("ThemeBackground").opacity(0.94)

    /// How much the animated backdrop is dimmed. Light mode needs a *light* veil rather than a
    /// dark one, or the artwork reads as a dark smear on a pale page.
    static func backdropVeil(for scheme: ColorScheme) -> Color {
        scheme == .dark ? .black : .white
    }
}

// MARK: - Appearance preference

/// Light / dark / follow-the-system, persisted and applied at the app root.
enum AppearanceMode: String, CaseIterable, Identifiable {
    case system, light, dark

    var id: String { rawValue }

    var title: String {
        switch self {
        case .system: return "Automatic"
        case .light:  return "Light"
        case .dark:   return "Dark"
        }
    }

    var subtitle: String {
        switch self {
        case .system: return "Follows your iPhone's appearance setting"
        case .light:  return "Always light"
        case .dark:   return "Always dark"
        }
    }

    var icon: String {
        switch self {
        case .system: return "circle.lefthalf.filled"
        case .light:  return "sun.max.fill"
        case .dark:   return "moon.fill"
        }
    }

    /// `nil` lets SwiftUI defer to the system, which is what "Automatic" means.
    var colorScheme: ColorScheme? {
        switch self {
        case .system: return nil
        case .light:  return .light
        case .dark:   return .dark
        }
    }
}

@MainActor
final class AppearanceManager: ObservableObject {

    static let shared = AppearanceManager()

    @Published var mode: AppearanceMode {
        didSet {
            UserDefaults.standard.set(mode.rawValue, forKey: storageKey)
            AppAppearance.configure()
        }
    }

    private let storageKey = "DarkAI_AppearanceMode"

    private init() {
        let stored = UserDefaults.standard.string(forKey: storageKey) ?? AppearanceMode.dark.rawValue
        mode = AppearanceMode(rawValue: stored) ?? .dark
    }

    /// Resolved scheme, with Automatic collapsed to whatever the system is currently showing.
    /// Used for decisions that need a concrete answer (the backdrop veil, the app icon).
    var resolvedScheme: ColorScheme {
        if let explicit = mode.colorScheme { return explicit }
        return UITraitCollection.current.userInterfaceStyle == .dark ? .dark : .light
    }

    // The app ships a single icon for every appearance, on purpose. Changing it at runtime needs
    // `setAlternateIconName`, which always raises the system "You have changed the icon" alert,
    // and the only ways to suppress that rely on intercepting the alert's presentation — not
    // something to ship in an app going through review.
}

// MARK: - UIKit appearance

enum AppAppearance {
    /// Forces the UIKit chrome SwiftUI still leans on to match the theme.
    ///
    /// Navigation bar titles are drawn by UIKit, not SwiftUI, and were the one thing in the app
    /// not reading from `Theme` — which is why the Mindscape and Settings headers rendered
    /// near-black on near-black when the device was set to Light. Everything here is a dynamic
    /// colour now, so the bar resolves correctly in either appearance instead of being pinned
    /// white.
    static func configure() {
        let appearance = UINavigationBarAppearance()
        appearance.configureWithOpaqueBackground()
        appearance.backgroundColor = UIColor(Theme.background)
        appearance.shadowColor = UIColor(Theme.border)
        appearance.titleTextAttributes = [.foregroundColor: UIColor(Theme.textPrimary)]
        appearance.largeTitleTextAttributes = [.foregroundColor: UIColor(Theme.textPrimary)]

        UINavigationBar.appearance().standardAppearance = appearance
        UINavigationBar.appearance().compactAppearance = appearance
        UINavigationBar.appearance().scrollEdgeAppearance = appearance
        UINavigationBar.appearance().tintColor = UIColor(Theme.accentCyan)
    }
}

extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (255, 0, 0, 0)
        }
        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
}

struct GlassCard: ViewModifier {
    var glow: Bool = false
    var cornerRadius: CGFloat = 16
    
    func body(content: Content) -> some View {
        content
            .padding()
            .background(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .fill(Theme.cardBackground)
                    .overlay(
                        RoundedRectangle(cornerRadius: cornerRadius)
                            .stroke(
                                LinearGradient(
                                    colors: [
                                        Theme.border,
                                        glow ? Theme.accent.opacity(0.6) : Theme.border,
                                        Theme.border,
                                        glow ? Theme.accentCyan.opacity(0.6) : Theme.border
                                    ],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                lineWidth: 1.5
                            )
                    )
                    .shadow(color: glow ? Theme.glowColor : Color.clear, radius: 10, x: 0, y: 4)
            )
    }
}

extension View {
    func glassCard(glow: Bool = false, cornerRadius: CGFloat = 16) -> some View {
        self.modifier(GlassCard(glow: glow, cornerRadius: cornerRadius))
    }

    /// Backing panel for text that sits directly on the animated background.
    ///
    /// The alternative — dimming the backdrop until nothing shows through — traded one problem
    /// for another: legible text on a background that may as well not have been there. Putting
    /// the contrast behind the text instead means the artwork stays visible everywhere it isn't
    /// competing with something that has to be read.
    func readablePanel(cornerRadius: CGFloat = 20, opacity: Double = 0.72) -> some View {
        self.modifier(ReadablePanel(cornerRadius: cornerRadius, opacity: opacity))
    }
}

struct ReadablePanel: ViewModifier {
    var cornerRadius: CGFloat = 20
    var opacity: Double = 0.72

    func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    // Follows the theme: a black wash under light-mode text would be the same
                    // contrast failure this modifier exists to prevent, just inverted.
                    .fill(Theme.background.opacity(opacity))
                    .overlay(
                        // A defined edge so it reads as a deliberate panel. A blurred, edgeless
                        // version of this looked like a smudge on the artwork rather than a
                        // surface the text belongs to.
                        RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                            .stroke(Theme.border.opacity(0.8), lineWidth: 1)
                    )
                    .shadow(color: .black.opacity(0.6), radius: 18, y: 6)
            )
    }
}

struct NeonGlow: ViewModifier {
    var color: Color
    var radius: CGFloat = 8
    
    func body(content: Content) -> some View {
        content
            .shadow(color: color.opacity(0.6), radius: radius)
            .shadow(color: color.opacity(0.3), radius: radius * 2)
    }
}

extension View {
    func neonGlow(color: Color = Theme.accent, radius: CGFloat = 8) -> some View {
        self.modifier(NeonGlow(color: color, radius: radius))
    }
}
