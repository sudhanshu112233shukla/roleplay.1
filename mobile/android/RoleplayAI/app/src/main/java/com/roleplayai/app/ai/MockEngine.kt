package com.roleplayai.app.ai

import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

class MockEngine : AiEngine {
    override val name: String = "Mock (Offline)"

    override fun generateStream(prompt: String, settings: GenerationSettings): Flow<String> = flow {
        val text = "I am running locally. Wire llama.cpp or ONNX to replace this engine."
        for (ch in text) {
            emit(ch.toString())
            delay(8)
        }
    }
}

