import SwiftUI

/// Layout constants shared by the chat screen.
///
/// **Nothing here may read `UIApplication`, `UIScreen`, or the key window.** An earlier version of
/// this file derived the header's top inset from `UIApplication.shared.connectedScenes` so it could
/// adapt to each device's notch. It produced the right *number* and broke the app: a view whose
/// `body` reaches into UIKit for window state during a SwiftUI update stops receiving updates
/// altogether. The symptom was subtle and total — the chat screen froze on whatever it had last
/// rendered, so loading a model left the header reading OFFLINE and the input field disabled
/// forever, while the Settings sheet (a separate view tree) showed the same model as loaded.
/// One `.padding(.top, Layout.barTopPadding())` was enough to cause it.
///
/// Everything below is therefore either a plain constant or a fraction applied through a SwiftUI
/// layout modifier. The device-dependent parts are handled the way the framework intends:
///
/// * **Safe-area insets** — by the safe area itself. The chat column is inset automatically, so the
///   header only adds a small constant on top. This is also strictly more correct than measuring:
///   the old code added a hardcoded 48 pt *on top of* the inset the column already had, which is
///   why the header sat 107 pt down a Dynamic Island phone and 68 pt down an iPhone SE.
/// * **Text that would otherwise wrap on a narrow screen** — by `lineLimit(1)`, `fixedSize()`, and
///   `minimumScaleFactor`, which let the header decide what gives way without anyone having to know
///   how wide the screen is.
enum Layout {

    // MARK: - Bars

    /// Padding above the header's content, added to whatever safe-area inset the device already
    /// contributes. Small on purpose — the inset is the device-dependent part.
    static let barTopPadding: CGFloat = 8

    static let barBottomPadding: CGFloat = 10

    /// Top padding for a floating banner that has to clear the header.
    ///
    /// The header is `barTopPadding` + a 44 pt touch target + `barBottomPadding` tall measured from
    /// the top of the safe area, and such a banner is a sibling of the chat column that starts at
    /// the same place — so this is that height plus a little breathing room.
    static let belowHeaderPadding: CGFloat = barTopPadding + 44 + barBottomPadding + 10

    // MARK: - Widths

    /// Widest the conversation column is allowed to get.
    ///
    /// The app fills the screen — header, banners, and the status strip all run edge to edge — but
    /// the *text* does not benefit from doing the same. A monospaced assistant reply set across
    /// 1180 pt of an iPad in landscape is one very long line per sentence, which is hard to read and
    /// reads as an unadapted phone app stretched to fit.
    ///
    /// 820 pt is chosen so this is a no-op on an 11-inch iPad in portrait (exactly 820 pt wide) and
    /// on every iPhone: nothing is narrowed that wasn't already comfortable, and only genuinely wide
    /// layouts — landscape, and the 13-inch iPads — get a centred column instead of a stretched one.
    static let contentMaxWidth: CGFloat = 820

    /// Conversations drawer. A constant is correct here because the app's deployment target is
    /// iOS 17, whose narrowest device is a 375 pt iPhone SE — 270 pt leaves a usable strip of the
    /// conversation visible behind it on every supported screen.
    static let drawerWidth: CGFloat = 270

    /// Largest a generated image renders in a chat bubble.
    static let chatImageMaxWidth: CGFloat = 300
}
