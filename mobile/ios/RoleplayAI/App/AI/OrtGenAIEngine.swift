import Foundation

final class OrtGenAIEngine: AIEngine {
    let name: String = "ONNX Runtime GenAI"

    func generateStream(prompt: String, settings: GenerationSettings) -> AsyncStream<String> {
        // Production integration: wire to ORT GenAI model package on iOS.
        return AsyncStream { continuation in
            continuation.yield("[ORT GenAI engine stub: provide model package + runtime integration]")
            continuation.finish()
        }
    }
}

