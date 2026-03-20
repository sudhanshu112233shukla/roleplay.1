# HF -> GGUF Conversion (llama.cpp)

This repo does not vendor llama.cpp. You provide the path to a local llama.cpp checkout.

## Inputs

- Merged HF model directory (from training):
  - multi-stage: `<output_root>/final_merged`
  - tokenizer: `<output_root>/tokenizer` (already saved by training)

## Outputs

- `*.gguf` files under your chosen output directory

## Convert + Quantize

Use the wrapper script:

```powershell
python deployment/gguf/convert_and_quantize.py `
  --llama-cpp-dir E:\\src\\llama.cpp `
  --hf-model-dir .\\tinyllama-roleplay-multistage\\final_merged `
  --out-dir .\\deployment_artifacts\\gguf `
  --quant q4_k_m
```

Notes:

- `--quant` is optional. If omitted, the script only produces an unquantized GGUF.
- The script expects llama.cpp to contain:
  - `convert_hf_to_gguf.py` (or `convert-hf-to-gguf.py`)
  - `quantize` (built binary) somewhere under the repo (commonly `build/bin/quantize`)

## Recommended quantizations

- Desktop CPU: `q4_k_m` (good balance)
- Very low RAM: `q3_k_m` / `q4_0` (quality tradeoff)
- More quality: `q5_k_m` or `q8_0` (slower / bigger)

