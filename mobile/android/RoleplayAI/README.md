# Android RoleplayAI (Kotlin, Compose, On-Device)

## Build

Open `mobile/android/RoleplayAI` in Android Studio and run the `app` configuration.

## Architecture

- MVVM: `ui/` + `viewmodel/`
- Persistence: Room DB under `data/db/`
- Repositories: `repository/`
- Local AI runtime abstraction: `ai/`

## On-device AI

The app is designed to support multiple runtimes:

- `llama.cpp` (GGUF): `ai/LlamaCppEngine.kt` (stub until JNI is provided)
- ONNX Runtime: `ai/OnnxEngine.kt` (stub; use ORT GenAI where possible)
- MLKit, Melange SDK: stubs

Until you wire a runtime, the app falls back to `ai/MockEngine.kt` which streams a local placeholder response.

## Voice and Attachments

- Voice input: uses Android speech recognition intent (device-dependent offline availability).
- Voice output: uses `TextToSpeech`.
- Attachments: uses the system document picker and injects a capped preview into the prompt.

