package com.roleplayai.app.ai

import android.content.Context
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

class OnnxEngine(private val context: Context) : AiEngine {
    override val name: String = "ONNX Runtime"

    override fun generateStream(prompt: String, settings: GenerationSettings): Flow<String> = flow {
        // Production integration: load an ONNX model from assets and run token generation.
        // For on-device LLM generation, prefer ORT GenAI where available.
        emit("[ONNX engine not wired yet on Android]")
    }
}

