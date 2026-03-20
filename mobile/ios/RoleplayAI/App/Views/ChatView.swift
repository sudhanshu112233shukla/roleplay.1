import SwiftUI
import UniformTypeIdentifiers

struct ChatView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var vm = ChatViewModel()
    @StateObject private var voice = VoiceService()
    @State private var isRecording: Bool = false
    @State private var showingImporter: Bool = false

    private let characterRepo = CharacterRepository()
    @State private var character: CharacterProfile =
        CharacterProfile(id: "wizard", name: "Wizard", personality: "Ancient magical teacher; wise and mysterious.", world: "Fantasy", tone: "Wise, descriptive, emotionally consistent.")

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading) {
                    Text(character.name).font(.headline)
                    Text(vm.engineName).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                NavigationLink("Characters") { CharacterView() }
                NavigationLink("Memory") { MemoryView() }
                NavigationLink("Settings") { SettingsView() }
                Button("Speak") {
                    let last = vm.messages.last(where: { $0.role == .assistant })?.content ?? ""
                    if !last.isEmpty { voice.speak(last) }
                }
            }
            .padding()

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(vm.messages) { m in
                            MessageBubble(role: m.role, text: m.content)
                        }
                        if vm.isStreaming && !vm.streamingText.isEmpty {
                            MessageBubble(role: .assistant, text: vm.streamingText + "|")
                        }
                    }
                    .padding()
                }
            }

            Divider()

            HStack(spacing: 8) {
                TextField("Message", text: $vm.input, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                Button(isRecording ? "Stop" : "Voice") {
                    Task {
                        if isRecording {
                            voice.stopTranscribing()
                            isRecording = false
                            return
                        }
                        let ok = await voice.requestSpeechAuthorization()
                        guard ok else { return }
                        do {
                            isRecording = true
                            try voice.startTranscribing { text in
                                Task { @MainActor in vm.input = text }
                            }
                        } catch {
                            isRecording = false
                        }
                    }
                }
                Button("Attach") { showingImporter = true }
                Button("Send") {
                    guard let sid = appState.activeSessionId else { return }
                    let mem = MemoryRepository().list(sessionId: sid).prefix(12).map { $0.text }
                    let historyPairs = buildPairs(messages: vm.messages.suffix(24))
                    vm.send(sessionId: sid, character: character, memory: mem, history: historyPairs)
                }
                .buttonStyle(.borderedProminent)
            }
            .padding()
        }
        .navigationTitle("Chat")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            loadCharacter()
            if let sid = appState.activeSessionId {
                vm.load(sessionId: sid)
            }
        }
        .onChange(of: appState.activeCharacterId) { _, _ in
            loadCharacter()
        }
        .fileImporter(
            isPresented: $showingImporter,
            allowedContentTypes: [.plainText, .text, .data],
            allowsMultipleSelection: false
        ) { res in
            guard let url = try? res.get().first else { return }
            let text = AttachmentService.readText(from: url)
            if !text.isEmpty {
                let block = "\n\n[Attachment: \(url.lastPathComponent)]\n\(text)"
                vm.input = vm.input.trimmingCharacters(in: .whitespacesAndNewlines) + block
            }
        }
    }

    private func loadCharacter() {
        characterRepo.ensureDefaults()
        character =
            characterRepo.get(id: appState.activeCharacterId)
            ?? characterRepo.get(id: "wizard")
            ?? character
    }
}

private struct MessageBubble: View {
    let role: Role
    let text: String

    var body: some View {
        HStack {
            if role == .assistant { bubble.foregroundStyle(.primary); Spacer(minLength: 40) }
            else { Spacer(minLength: 40); bubble.foregroundStyle(.white) }
        }
    }

    private var bubble: some View {
        Text(.init(text))
            .padding(12)
            .background(role == .assistant ? Color.secondary.opacity(0.2) : Color.accentColor)
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

private func buildPairs(messages: ArraySlice<ChatMessage>) -> [(String, String)] {
    var pairs: [(String, String)] = []
    var pendingUser: String? = nil
    for m in messages {
        if m.role == .user { pendingUser = m.content }
        if m.role == .assistant, let u = pendingUser {
            pairs.append((u, m.content))
            pendingUser = nil
        }
    }
    return pairs
}
