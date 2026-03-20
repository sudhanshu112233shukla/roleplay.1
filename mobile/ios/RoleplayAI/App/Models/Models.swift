import Foundation

enum Role: String, Codable {
    case user
    case assistant
    case system
}

struct CharacterProfile: Identifiable, Codable, Hashable {
    var id: String
    var name: String
    var personality: String
    var world: String
    var tone: String
}

struct ChatMessage: Identifiable, Codable, Hashable {
    var id: UUID
    var sessionId: UUID
    var role: Role
    var content: String
    var createdAt: Date
}

struct ChatSession: Identifiable, Codable, Hashable {
    var id: UUID
    var title: String
    var characterId: String
    var createdAt: Date
    var updatedAt: Date
}

struct MemoryItem: Identifiable, Codable, Hashable {
    var id: UUID
    var sessionId: UUID
    var text: String
    var score: Double
    var createdAt: Date
}

