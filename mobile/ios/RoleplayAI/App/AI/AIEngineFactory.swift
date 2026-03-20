import Foundation

enum AIEngineFactory {
    static func make(engineId: String) -> AIEngine {
        switch engineId {
        case "llamacpp":
            return LlamaCppEngine()
        case "ortgenai":
            return OrtGenAIEngine()
        case "coreml":
            return CoreMLEngine()
        default:
            return MockEngine()
        }
    }
}

