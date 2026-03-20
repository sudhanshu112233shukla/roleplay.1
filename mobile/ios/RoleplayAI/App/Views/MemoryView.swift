import SwiftUI

struct MemoryView: View {
    @EnvironmentObject private var appState: AppState
    @State private var items: [MemoryItem] = []
    private let repo = MemoryRepository()

    var body: some View {
        List {
            Section("Memory") {
                if let sid = appState.activeSessionId {
                    ForEach(items) { m in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(m.text)
                            Text("score \(m.score, specifier: "%.2f")").font(.caption).foregroundStyle(.secondary)
                        }
                        .contextMenu {
                            Button("Delete") {
                                repo.delete(id: m.id)
                                refresh()
                            }
                        }
                    }
                } else {
                    Text("Select a session first.")
                }
            }
        }
        .navigationTitle("Memory")
        .onAppear { refresh() }
    }

    private func refresh() {
        guard let sid = appState.activeSessionId else { items = []; return }
        items = repo.list(sessionId: sid)
    }
}
