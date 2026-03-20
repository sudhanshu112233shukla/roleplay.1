package com.roleplayai.app.ai

import android.content.Context
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

class MelangeEngine(private val context: Context) : AiEngine {
    override val name: String = "Melange SDK (Stub)"

    override fun generateStream(prompt: String, settings: GenerationSettings): Flow<String> = flow {
        emit("[Melange engine is a placeholder. Wire your Melange SDK runtime here.]")
    }
}

