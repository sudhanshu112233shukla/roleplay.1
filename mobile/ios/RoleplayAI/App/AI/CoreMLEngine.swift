import Foundation

final class CoreMLEngine: AIEngine {
    let name: String = "CoreML"

    func generateStream(prompt: String, settings: GenerationSettings) -> AsyncStream<String> {
        // Production integration: load a CoreML LLM or a CoreML text generation pipeline.
        return AsyncStream { continuation in
            continuation.yield("[CoreML engine stub: integrate your CoreML model here]")
            continuation.finish()
        }
    }
}

