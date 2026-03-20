import Foundation
import SwiftUI

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var input: String = ""
    @Published var streamingText: String = ""
    @Published var isStreaming: Bool = false
    @Published var engineName: String = "Local"

    private var engine: AIEngine = MockEngine()
    private let sessions = SessionRepository()
    private let memoryRepo = MemoryRepository()

    func setEngine(_ engine: AIEngine) {
        self.engine = engine
        self.engineName = engine.name
    }

    func load(sessionId: UUID) {
        messages = sessions.listMessages(sessionId: sessionId)
    }

    func send(sessionId: UUID, character: CharacterProfile, memory: [String], history: [(String, String)]) {
        let userText = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !userText.isEmpty else { return }
        input = ""

        sessions.addMessage(sessionId: sessionId, role: .user, content: userText)
        load(sessionId: sessionId)

        let prompt = PromptBuilder.build(
            character: character,
            world: WorldState(),
            emotion: EmotionState(),
            memories: memory,
            history: history,
            userInput: userText
        )

        isStreaming = true
        streamingText = ""

        Task {
            var buf = ""
            for await token in engine.generateStream(prompt: prompt, settings: GenerationSettings()) {
                buf += token
                streamingText = buf
            }
            isStreaming = false
            streamingText = ""
            let final = buf.trimmingCharacters(in: .whitespacesAndNewlines)
            if !final.isEmpty {
                sessions.addMessage(sessionId: sessionId, role: .assistant, content: final)
                load(sessionId: sessionId)
            }

            if let cand = MemoryHeuristics.candidate(from: userText) {
                memoryRepo.add(sessionId: sessionId, text: cand.text, score: cand.score)
            }
        }
    }
}
