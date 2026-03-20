# Deployment (On-Device)

This folder contains scripts and docs for packaging the trained model for edge inference.

Supported targets:

- GGUF + `llama.cpp` (recommended for on-device CPU inference)
- ONNX Runtime GenAI (ORT GenAI) (recommended when you have an ORT GenAI workflow)

The runtime stack is backend-agnostic:

- prompt construction: `prompt_builder/`
- character profiles: `characters/`
- memory (vector store): `memory/`
- world state: `world_state/`
- emotion tracking: `emotion_engine/`
- session orchestration: `inference/session.py`

## GGUF (llama.cpp)

Prerequisites:

- a merged HF model directory produced by training, e.g. `./tinyllama-roleplay-multistage/final_merged`
- a local `llama.cpp` checkout (or an installed `llama.cpp` build that includes `quantize`)

Steps (high-level):

1. Convert HF -> GGUF using llama.cpp conversion script.
2. Quantize GGUF (q4/q5/q8) using `quantize`.
3. Run inference using `llama.cpp` CLI or `llama-cpp-python` backend.

See `deployment/gguf/CONVERT.md`.

## ONNX (ORT GenAI)

There are multiple ONNX export paths for LLMs. This repo targets ORT GenAI if you choose ONNX:

- Build/export the ORT GenAI model package (outside this repo)
- Run via `inference/backends/onnx_backend.py` (requires `onnxruntime-genai`)

See `deployment/onnx/ORT_GENAI.md`.

