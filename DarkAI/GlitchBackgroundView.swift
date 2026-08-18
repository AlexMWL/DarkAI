import SwiftUI
import Combine

/// Animated backdrop.
///
/// Deliberately quieter than it started out. The original drew the circuit graphic at 0.7 alpha
/// with `.blendMode(.screen)` and flashed white bars at 0.3 alpha directly behind body copy —
/// which put moving light-on-dark shapes at roughly the same luminance as the text sitting on
/// top of them. It looked good on an empty screen and made the app hard to read in use, so the
/// glitch layers are dimmer and a scrim is built in rather than left for each call site to
/// remember.
struct GlitchBackgroundView: View {
    @State private var glitchOffset1: CGFloat = 0
    @State private var glitchOffset2: CGFloat = 0
    @State private var glitchOpacity: Double = 0.0
    @State private var glitchColor: Color = .red
    @Environment(\.colorScheme) private var colorScheme

    /// How much black to lay over the animation.
    ///
    /// Kept low by default so the artwork stays visible — readability is handled where the text
    /// actually is, by `readablePanel()`, rather than by washing out the whole backdrop.
    var scrimOpacity: Double = 0.2

    /// Set to `false` when this view is fully obscured (e.g. a sheet is presented over it) so the
    /// glitch animation stops doing work it can't be seen doing — the timer keeps ticking either
    /// way (see `timer` below), but the per-tick animation trigger is skipped.
    var isActive: Bool = true

    /// `@State`, not `let` — a `let` stored property is re-evaluated every time this struct is
    /// reconstructed (i.e. on every parent body recomputation, which for `ContentView` happens on
    /// every streamed token), which built a brand new `Timer.publish(...).autoconnect()` — and a
    /// fresh `onReceive` subscription to it — on each redraw instead of the one timer this view is
    /// meant to keep for its whole lifetime. `@State` gives it the same one-per-view-identity
    /// lifetime everything else here already relies on.
    @State private var timer = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()

    var body: some View {
        ZStack {
            // Base background, with a soft wash so neither appearance is a flat field. Light
            // mode especially needed this — a plain white page behind the chat read as unstyled
            // rather than as the same product with the lights on.
            Theme.background.ignoresSafeArea()

            RadialGradient(
                colors: colorScheme == .dark
                    ? [Theme.accent.opacity(0.10), Color.clear]
                    : [Theme.accent.opacity(0.09), Theme.accentCyan.opacity(0.045), Color.clear],
                center: .top,
                startRadius: 0,
                endRadius: 620
            )
            .ignoresSafeArea()

            // Abstract grid or tech lines.
            //
            // Drawn through a `GeometryReader` rather than as a fixed stack of 20 rows, and that is
            // a layout fix rather than a cosmetic one. Twenty rows at 40 pt spacing have an
            // intrinsic height of 780 pt, and because this whole view is a `ZStack` whose other
            // children are flexible, that 780 pt became the backdrop's reported size. On every
            // screen shorter than that — iPhone SE and iPhone 8, both 667 pt — any container using
            // this view as a *sibling* inherited the oversized frame and pushed its own content off
            // the bottom of the display: it is why onboarding's "Continue" button was sliced in half
            // by the screen edge on an SE while looking perfectly placed on a 15 Pro.
            //
            // A `GeometryReader` accepts whatever size it is offered instead of demanding one, so
            // the backdrop can no longer inflate its parent, and the row count now follows the
            // available height — which also means the grid reaches the bottom of a tall screen
            // rather than stopping 70 pt short of it.
            GeometryReader { geometry in
                let spacing: CGFloat = 40
                let rowCount = max(1, Int(geometry.size.height / spacing) + 1)
                VStack(spacing: spacing - 1) {
                    ForEach(0..<rowCount, id: \.self) { _ in
                        Rectangle()
                            .fill(Theme.textMuted.opacity(colorScheme == .dark ? 0.05 : 0.09))
                            .frame(height: 1)
                    }
                }
                .frame(width: geometry.size.width, alignment: .top)
            }

            // The three brain layers all dim with `.opacity` rather than a tint colour.
            //
            // Two earlier attempts got this wrong in opposite directions. Originally they used
            // `.foregroundColor(someColor.opacity(...))`, which only affects *template* images —
            // this asset is full-colour art, so every one of those tints was silently discarded
            // and the brain rendered at full strength directly behind the body copy. Forcing
            // `.renderingMode(.template)` made the tints apply but exposed the other half of the
            // problem: the PNG has no alpha channel, so the template silhouette filled its whole
            // bounding box and drew a visible rectangle on screen. Plain `.opacity` dims the real
            // artwork, works regardless of rendering mode, and leaves no box.

            // Glitching element 1 (red shift). Both glitch layers now use the alpha-keyed asset,
            // so they work in either appearance — no screen blending, which was the thing that
            // restricted them to a dark page (over a light one it lightens toward white and
            // exposes the layer's bounding box as a grey rectangle).
            Image("circuit_traces")
                .renderingMode(.template)
                .resizable()
                .scaledToFit()
                .frame(width: 240, height: 240)
                .foregroundColor(glitchColor)
                .opacity(glitchOpacity * (colorScheme == .dark ? 0.30 : 0.22))
                .offset(x: glitchOffset1, y: -glitchOffset2)

            Image("circuit_traces")
                .renderingMode(.template)
                .resizable()
                .scaledToFit()
                .frame(width: 240, height: 240)
                .foregroundColor(colorScheme == .dark ? .cyan : Theme.accentCyan)
                .opacity(glitchOpacity * (colorScheme == .dark ? 0.30 : 0.22))
                .offset(x: -glitchOffset1, y: glitchOffset2)

            // Core element, drawn in both appearances.
            //
            // `circuit_traces` is an alpha-keyed derivative of the original `circuit_brain`
            // artwork, generated by mapping the source's luminance onto alpha so the traces
            // survive and the near-black field drops out. That matters because the original is a
            // JPEG — no alpha channel at all — which is why the backdrop had to be skipped
            // entirely in light mode and left a bare white page. With real transparency the asset
            // is a proper template image, so it takes a tint and reads correctly on either
            // background instead of dragging its own black square along with it.
            Image("circuit_traces")
                .renderingMode(.template)
                .resizable()
                .scaledToFit()
                .frame(width: 240, height: 240)
                .foregroundColor(colorScheme == .dark ? .white : Theme.textPrimary)
                .opacity(colorScheme == .dark ? 0.22 : 0.16)
            
            GeometryReader { geo in
                ForEach(0..<5) { _ in
                    Rectangle()
                        .fill((colorScheme == .dark ? Color.white : Theme.border).opacity(glitchOpacity * 0.10))
                        .frame(width: geo.size.width, height: CGFloat.random(in: 2...6))
                        .position(x: geo.size.width / 2, y: CGFloat.random(in: 0...geo.size.height))
                        .offset(x: glitchOffset1 * 2)
                }
            }

            // Built-in scrim. Every screen that uses this view puts text over it, so the
            // readability floor belongs here rather than in each call site. The veil follows the
            // appearance — a black scrim over a light page would turn the backdrop into a grey
            // smear instead of softening it.
            Theme.backdropVeil(for: colorScheme)
                .opacity(scrimOpacity)
                .allowsHitTesting(false)
        }
        .onReceive(timer) { _ in
            guard isActive else { return }
            if Double.random(in: 0...1) > 0.85 {
                withAnimation(.linear(duration: 0.05)) {
                    glitchOffset1 = CGFloat.random(in: -15...15)
                    glitchOffset2 = CGFloat.random(in: -5...5)
                    glitchOpacity = Double.random(in: 0.3...0.8)
                    glitchColor = Double.random(in: 0...1) > 0.5 ? .red : .purple
                }

                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    withAnimation(.linear(duration: 0.05)) {
                        glitchOffset1 = 0
                        glitchOffset2 = 0
                        glitchOpacity = 0.0
                    }
                }
            }
        }
    }
}
