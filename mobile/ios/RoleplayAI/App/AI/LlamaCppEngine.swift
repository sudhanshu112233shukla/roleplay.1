import Foundation

final class LlamaCppEngine: AIEngine {
    let name: String = "llama.cpp (GGUF)"

    func generateStream(prompt: String, settings: GenerationSettings) -> AsyncStream<String> {
        // Production integration: bridge to llama.cpp compiled for iOS, with token streaming.
        // This stub falls back to MockEngine behavior until the native layer is provided.
        return MockEngine().generateStream(prompt: prompt, settings: settings)
    }
}

