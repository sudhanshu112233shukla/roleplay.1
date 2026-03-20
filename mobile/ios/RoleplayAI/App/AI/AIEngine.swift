import Foundation

struct GenerationSettings: Codable, Hashable {
    var maxNewTokens: Int = 180
    var temperature: Double = 0.8
    var topP: Double = 0.9
}

protocol AIEngine {
    var name: String { get }
    func generateStream(prompt: String, settings: GenerationSettings) -> AsyncStream<String>
}

