# ONNX Runtime GenAI (ORT GenAI)

For large language models, plain ONNX export is often not enough for efficient generation on-device.
ORT GenAI provides a generation runtime for supported models.

This repo provides:

- a backend integration (`inference/backends/onnx_backend.py`) that uses ORT GenAI if installed
- guidance on the expected inputs

## Runtime dependency

Install (platform-specific):

- `onnxruntime-genai`

## Model package

ORT GenAI typically consumes a "model package" produced by an export process external to this repo.
You will point the backend at that model directory.

## Usage

Once you have an ORT GenAI model dir, use:

```python
from inference.backends.onnx_backend import OrtGenAIBackend, OrtGenAIConfig
backend = OrtGenAIBackend(OrtGenAIConfig(model_dir="path/to/ortgenai/model"))
text = backend.generate(prompt)
```

If you want, I can add an export helper once you pick the exact ORT GenAI export path/tooling you will use
for TinyLlama (varies by platform and runtime constraints).

