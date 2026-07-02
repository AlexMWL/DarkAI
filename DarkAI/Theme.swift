import SwiftUI

struct Theme {
    static let background = Color(hex: "020202") // True Tech Black
    static let cardBackground = Color(hex: "120505").opacity(0.8) // Dark Red-Black
    static let accent = Color(hex: "FF0033") // Glowing Neon Red
    static let accentCyan = Color(hex: "FF5500") // Tech Orange-Red
    static let accentRose = Color(hex: "990011") // Crimson
    
    static let textPrimary = Color(hex: "FFFFFF")
    static let textSecondary = Color(hex: "CF9F9F") // Muted Red-Gray Text
    static let textMuted = Color(hex: "7F4F4F") // Dark Muted Red-Gray
    
    static let border = Color(hex: "3A1212") // Tech Dark Red Border
    static let glowColor = Color(hex: "FF0033").opacity(0.4) // Red Glow
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

// Custom Glassmorphic Card Modifier
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
}

// Subtle Neon Glow Modifier
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
