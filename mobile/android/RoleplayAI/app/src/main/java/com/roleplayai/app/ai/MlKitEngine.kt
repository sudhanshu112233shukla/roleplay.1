package com.roleplayai.app.ai

import android.content.Context
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

class MlKitEngine(private val context: Context) : AiEngine {
    override val name: String = "MLKit (Stub)"

    override fun generateStream(prompt: String, settings: GenerationSettings): Flow<String> = flow {
        emit("[MLKit engine is a placeholder. Provide an on-device model and integration here.]")
    }
}

