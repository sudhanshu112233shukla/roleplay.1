import SwiftUI

struct AppRootView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        NavigationSplitView {
            SessionsView()
        } detail: {
            if let _ = appState.activeSessionId {
                ChatView()
            } else {
                ContentUnavailableView("Select or create a session", systemImage: "bubble.left.and.bubble.right")
            }
        }
    }
}

