# Edge Export Notes (GGUF / ONNX / Mobile)

This project keeps training and runtime modular so the same prompt/memory/world/emotion stack can run with:

- `transformers` (development)
- `llama.cpp` (GGUF; on-device)
- ONNX Runtime / TensorRT (edge acceleration)

## Adapter + Merge

Training exports:

- LoRA adapter directory (small, incremental retraining friendly)
- Merged model directory (for deployment)

## GGUF (llama.cpp)

Typical flow (outside this repo):

1. Convert the merged HF model to llama.cpp format.
2. Quantize to GGUF.
3. Run inference with llama.cpp on-device.

The prompt builder in `prompt_builder/` produces a plain string compatible with llama.cpp.

