package com.roleplayai.app.ai

import kotlinx.coroutines.flow.Flow

data class GenerationSettings(
    val maxNewTokens: Int = 180,
    val temperature: Float = 0.8f,
    val topP: Float = 0.9f
)

interface AiEngine {
    val name: String
    fun generateStream(prompt: String, settings: GenerationSettings = GenerationSettings()): Flow<String>
}

