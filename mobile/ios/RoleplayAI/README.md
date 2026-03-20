# iOS RoleplayAI (SwiftUI, On-Device)

This folder contains the full SwiftUI MVVM code for the on-device roleplay chat app.

Because `.xcodeproj` generation is toolchain-dependent, create an iOS App project in Xcode named `RoleplayAI`
and then add the `App/` folder contents to the project (preserving groups).

Key modules:

- `App/Views/`: Chat, Character, Settings, Memory, Sessions (iPhone + iPad layouts)
- `App/ViewModels/`: MVVM view models with streaming generation
- `App/AI/`: engine abstractions + backends (llama.cpp, ORT GenAI, CoreML, stubs)
- `App/Services/`: voice (STT/TTS), attachments, prompt building
- `App/Models/`: domain models
- `App/Storage/`: CoreData stack (programmatic model) + repositories

Model packaging:

- GGUF for llama.cpp (recommended) stored in app sandbox (import via Files picker)
- ORT GenAI model dir (optional) stored in app sandbox (import via Files picker)

