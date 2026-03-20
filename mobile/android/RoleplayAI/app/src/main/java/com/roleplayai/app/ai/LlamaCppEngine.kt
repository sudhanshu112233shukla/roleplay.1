package com.roleplayai.app.ai

import android.content.Context
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch

class LlamaCppEngine(private val context: Context) : AiEngine {
    override val name: String = "llama.cpp (GGUF)"

    override fun generateStream(prompt: String, settings: GenerationSettings): Flow<String> = callbackFlow {
        // Production integration: replace this with JNI bridge to llama.cpp streaming decode.
        // This stub falls back to MockEngine behavior unless you provide the native layer.
        val fallback = MockEngine()
        val job = launch {
            fallback.generateStream(prompt, settings).collect { token ->
                trySend(token)
            }
            close()
        }
        awaitClose { job.cancel() }
    }
}
