package com.roleplayai.app.ai

import com.roleplayai.app.data.db.CharacterEntity

data class WorldState(
    val location: String = "unknown",
    val time: String = "unknown",
    val currentEvent: String = "none",
    val storyProgress: String = "beginning"
)

data class EmotionState(
    val emotion: String = "neutral"
)

object PromptAssembler {
    fun build(
        character: CharacterEntity,
        world: WorldState,
        emotion: EmotionState,
        memories: List<String>,
        history: List<Pair<String, String>>,
        userInput: String
    ): String {
        val memBlock = if (memories.isEmpty()) "(none)" else memories.joinToString("\n") { "- $it" }
        val historyBlock =
            if (history.isEmpty()) "(none)"
            else history.joinToString("\n") { (u, a) -> "<|user|>\n$u\n<|assistant|>\n$a\n" }

        val system = buildString {
            appendLine("You are an immersive roleplay assistant. Stay in character, emotionally consistent, descriptive, and coherent across turns.")
            appendLine()
            appendLine("CHARACTER:")
            appendLine("Name: ${character.name}")
            appendLine("Personality: ${character.personality}")
            appendLine("World: ${character.world}")
            appendLine("Tone: ${character.tone}")
            appendLine()
            appendLine("EMOTION STATE:")
            appendLine(emotion.emotion)
        }

        val worldBlock = buildString {
            appendLine("location: ${world.location}")
            appendLine("time: ${world.time}")
            appendLine("current_event: ${world.currentEvent}")
            appendLine("story_progress: ${world.storyProgress}")
        }

        return buildString {
            appendLine("<|system|>")
            appendLine(system)
            appendLine()
            appendLine("WORLD STATE:")
            appendLine(worldBlock)
            appendLine()
            appendLine("MEMORY:")
            appendLine(memBlock)
            appendLine()
            appendLine("CHAT HISTORY:")
            appendLine(historyBlock)
            appendLine()
            appendLine("<|user|>")
            appendLine(userInput)
            appendLine("<|assistant|>")
        }
    }
}

