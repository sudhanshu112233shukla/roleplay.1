# Mobile Model Packaging (Offline)

This repo supports a fully offline roleplay chat product. Mobile apps expect model assets to be local.

## Recommended: GGUF + llama.cpp

1. Train and export a merged HF model (server-side):
   - `python -m training.train_multistage` creates `<output_root>/final_merged`
2. Convert + quantize to GGUF:
   - Use `deployment/gguf/convert_and_quantize.py` with a local llama.cpp checkout.
3. Put the GGUF file on-device:
   - Android: import via file picker (app can store under app-private storage)
   - iOS: import via file picker (sandbox Documents)
4. Wire the native llama.cpp runtime:
   - Android: implement JNI bridge in `mobile/android/RoleplayAI/app/src/main/java/com/roleplayai/app/ai/LlamaCppEngine.kt`
   - iOS: implement bridge in `mobile/ios/RoleplayAI/App/AI/LlamaCppEngine.swift`

## ONNX Runtime

For LLMs, prefer ORT GenAI where available.

- Python-side ORT GenAI model packaging is deployment-specific and not vendored here.
- Once you have an ORT GenAI model directory, you can implement:
  - Android: `ai/OnnxEngine.kt`
  - iOS: `AI/OrtGenAIEngine.swift`

## CoreML (iOS)

If you choose CoreML, export your model to a CoreML-compatible format and implement:

- `mobile/ios/RoleplayAI/App/AI/CoreMLEngine.swift`

## Memory models (optional)

If you want vector memory on-device:

- Export an embedding model (e.g. MiniLM) to ONNX and run it locally.
- Store embeddings in Room/CoreData and do similarity search (or a FAISS-native module).

