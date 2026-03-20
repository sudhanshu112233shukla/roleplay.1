import SwiftUI
import Foundation

struct CharacterView: View {
    @EnvironmentObject private var appState: AppState
    @State private var characters: [CharacterProfile] = []
    private let repo = CharacterRepository()

    @State private var name: String = ""
    @State private var personality: String = ""
    @State private var world: String = ""
    @State private var tone: String = ""

    var body: some View {
        Form {
            Section("Choose") {
                if characters.isEmpty {
                    Text("No characters yet. Create one below.")
                        .foregroundStyle(.secondary)
                }

                ForEach(characters) { c in
                    Button {
                        appState.activeCharacterId = c.id
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(c.name)
                                Text(c.world).font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            if c.id == appState.activeCharacterId {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundStyle(.tint)
                            }
                        }
                    }
                    .contextMenu {
                        Button("Delete", role: .destructive) {
                            repo.delete(id: c.id)
                            repo.ensureDefaults()
                            if appState.activeCharacterId == c.id {
                                appState.activeCharacterId = repo.list().first?.id ?? "wizard"
                            }
                            refresh()
                        }
                    }
                }
            }

            Section("Create Character") {
                TextField("Name", text: $name)
                TextField("Personality", text: $personality, axis: .vertical)
                TextField("World setting", text: $world)
                TextField("Voice style", text: $tone)
                Button("Save") {
                    let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmedName.isEmpty else { return }

                    let id = uniqueId(fromName: trimmedName)
                    repo.upsert(
                        CharacterProfile(
                            id: id,
                            name: trimmedName,
                            personality: personality.trimmingCharacters(in: .whitespacesAndNewlines),
                            world: world.trimmingCharacters(in: .whitespacesAndNewlines),
                            tone: tone.trimmingCharacters(in: .whitespacesAndNewlines)
                        )
                    )
                    appState.activeCharacterId = id
                    name = ""; personality = ""; world = ""; tone = ""
                    refresh()
                }
            }
        }
        .navigationTitle("Characters")
        .onAppear {
            repo.ensureDefaults()
            if repo.get(id: appState.activeCharacterId) == nil {
                appState.activeCharacterId = repo.list().first?.id ?? "wizard"
            }
            refresh()
        }
    }

    private func refresh() {
        characters = repo.list()
    }

    private func uniqueId(fromName name: String) -> String {
        let base = slugify(name)
        if repo.get(id: base) == nil { return base }
        var i = 2
        while repo.get(id: "\(base)_\(i)") != nil { i += 1 }
        return "\(base)_\(i)"
    }

    private func slugify(_ s: String) -> String {
        let lowered = s.lowercased()
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_"))
        let spaced = lowered.replacingOccurrences(of: " ", with: "_")
        let cleaned = spaced.unicodeScalars.map { allowed.contains($0) ? Character($0) : "_" }
        return String(cleaned)
            .replacingOccurrences(of: "__+", with: "_", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "_"))
    }
}
