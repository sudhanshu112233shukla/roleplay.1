import Foundation

final class MockEngine: AIEngine {
    let name: String = "Mock (Offline)"

    func generateStream(prompt: String, settings: GenerationSettings) -> AsyncStream<String> {
        AsyncStream { continuation in
            let text = "I am running locally. Wire llama.cpp, ORT GenAI, or CoreML to replace this engine."
            Task {
                for ch in text {
                    continuation.yield(String(ch))
                    try? await Task.sleep(nanoseconds: 8_000_000)
                }
                continuation.finish()
            }
        }
    }
}

