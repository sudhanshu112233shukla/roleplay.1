import SwiftUI

struct SessionsView: View {
    @EnvironmentObject private var appState: AppState
    @State private var sessions: [ChatSession] = []
    private let repo = SessionRepository()

    var body: some View {
        List {
            Section("Sessions") {
                Button("New Chat") {
                    let id = repo.createSession(characterId: appState.activeCharacterId)
                    appState.activeSessionId = id
                    refresh()
                }
                ForEach(sessions) { s in
                    Button(s.title) { appState.activeSessionId = s.id }
                        .contextMenu {
                            Button("Delete") {
                                repo.deleteSession(id: s.id)
                                if appState.activeSessionId == s.id { appState.activeSessionId = nil }
                                refresh()
                            }
                        }
                }
            }
            Section("Navigation") {
                NavigationLink("Characters") { CharacterView() }
                NavigationLink("Memory") { MemoryView() }
                NavigationLink("Settings") { SettingsView() }
            }
        }
        .navigationTitle("Sessions")
        .onAppear { refresh() }
    }

    private func refresh() {
        sessions = repo.listSessions()
    }
}
