import Foundation

struct WorldState: Codable, Hashable {
    var location: String = "unknown"
    var time: String = "unknown"
    var currentEvent: String = "none"
    var storyProgress: String = "beginning"
}

struct EmotionState: Codable, Hashable {
    var emotion: String = "neutral"
}

enum PromptBuilder {
    static func build(
        character: CharacterProfile,
        world: WorldState,
        emotion: EmotionState,
        memories: [String],
        history: [(String, String)],
        userInput: String
    ) -> String {
        let memBlock = memories.isEmpty ? "(none)" : memories.map { "- \($0)" }.joined(separator: "\n")
        let historyBlock = history.isEmpty ? "(none)" : history.map { u, a in
            "<|user|>\n\(u)\n<|assistant|>\n\(a)\n"
        }.joined(separator: "\n")

        let system = """
        You are an immersive roleplay assistant. Stay in character, emotionally consistent, descriptive, and coherent across turns.

        CHARACTER:
        Name: \(character.name)
        Personality: \(character.personality)
        World: \(character.world)
        Tone: \(character.tone)

        EMOTION STATE:
        \(emotion.emotion)
        """

        let worldBlock = """
        location: \(world.location)
        time: \(world.time)
        current_event: \(world.currentEvent)
        story_progress: \(world.storyProgress)
        """

        return """
        <|system|>
        \(system)

        WORLD STATE:
        \(worldBlock)

        MEMORY:
        \(memBlock)

        CHAT HISTORY:
        \(historyBlock)

        <|user|>
        \(userInput)
        <|assistant|>
        """
    }
}

