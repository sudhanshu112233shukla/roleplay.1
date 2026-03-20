package com.roleplayai.app.ai

import android.content.Context

class AiEngineFactory(private val context: Context) {
    fun create(engineId: String): AiEngine {
        return when (engineId) {
            "llamacpp" -> LlamaCppEngine(context)
            "onnx" -> OnnxEngine(context)
            "mlkit" -> MlKitEngine(context)
            "melange" -> MelangeEngine(context)
            else -> MockEngine()
        }
    }
}
