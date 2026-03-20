# Mobile Apps (On-Device Roleplay AI)

This directory contains full Android and iOS applications that run the roleplay chat system fully offline.

- Android: Kotlin + Jetpack Compose + MVVM + Room
- iOS: Swift + SwiftUI + MVVM + CoreData (programmatic model)

Both apps share the same high-level architecture:

- UI (screens)
- ViewModels
- Repositories (sessions, memory, characters, settings)
- Local AI runtime (pluggable backends)

Supported AI engines (pluggable):

- llama.cpp (GGUF)
- ONNX Runtime
- CoreML (iOS)
- MLKit (Android) (stub)
- Melange SDK (stub)

Model files are expected to be shipped as local assets (or downloaded manually by the user and imported).

