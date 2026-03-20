# Roleplay AI System (TinyLlama + LoRA, Edge‑Ready)

A modular roleplay/chat AI stack built around TinyLlama with LoRA fine‑tuning, dynamic personas, memory, and deployment paths for GGUF/llama.cpp and ONNX. This repo is designed to support fast iteration: train in Colab or locally, test with multi‑turn scripts, then export for lightweight inference.

## What This Gives You

- **Dynamic persona control** (runtime “be X” instructions, no static files required)
- **YAML character profiles** (optional static personas)
- **Prompt system** with memory, world state, and emotion engines
- **Training pipelines**: single‑stage, chunked SFT (local + HF), multi‑stage
- **Deployment pipeline**: merge LoRA → export → GGUF / ONNX
- **Multi‑turn test scripts** for persona switching validation

## Repository Layout

- `characters/` – character profiles, loaders, dynamic persona support
- `datasets/` – dataset loaders + local dataset cache
- `deployment/` – GGUF/llama.cpp + ONNX tooling
- `emotion_engine/` – emotion state + update logic
- `inference/` – chat session orchestration + backends
- `memory/` – FAISS memory + importance heuristics
- `models/` – loaders, LoRA helpers, export notes
- `notebooks/` – Colab training notebooks
- `prompt_builder/` – prompt assembly + truncation logic
- `training/` – training entrypoints (single, chunked, multi‑stage)
- `training_data/` – dataset schema + helpers
- `world_state/` – world state schema + update logic

## Quick Start (Local)

```powershell
pip install -r requirements.txt

# Single‑stage SFT
python -m training.train_single_stage --help

# Chunked SFT (local JSONL)
python -m training.train_chunked_sft --help

# Chunked SFT (Hugging Face streaming)
python -m training.train_chunked_sft_hf --help

# Multi‑stage pipeline
python -m training.train_multistage --help
```

## Quick Start (Colab)

Open one of these notebooks:

- `notebooks/roleplay_training_chunked_hf.ipynb` (HF datasets, streaming)
- `notebooks/roleplay_training_chunked_local.ipynb` (local JSONL/CSV)
- `notebooks/roleplay_training_refactored.ipynb`
- `notebooks/roleplay_multistage_refactored.ipynb`

If you get disconnected, use **chunked training** + auto‑resume checkpoints.

## Training (Typical Flow)

1. **Prepare dataset** (e.g., persona‑switch JSONL).
2. **Run chunked SFT** with checkpointing and optional auto‑resume.
3. **Merge LoRA** into base model (optional).
4. **Export to GGUF** and quantize for CPU/edge.
5. **Run multi‑turn persona test** to validate switching.

Example (local JSONL chunked SFT):

```powershell
python -m training.train_chunked_sft `
  --dataset-path .\artifacts\persona_switch.jsonl `
  --output-root .\artifacts\persona_switch_run `
  --max-steps 1200 `
  --save-steps 50 `
  --merge-at-end
```

## Inference

### Transformers (local dev)
```powershell
python -m inference.dynamic_persona_chat --help
```

### GGUF + llama.cpp
```powershell
python -m inference.gguf_multi_turn `
  --model .\artifacts\final_merged.q4_k_m.gguf `
  --turns-file .\artifacts\turns_persona_short.json `
  --use-remainder
```

## Dynamic Persona Control

Users can ask for any persona at runtime:

- “Be Iron Man today and talk to me.”
- “Act like a wise mentor.”
- “Talk like a funny friend.”

The prompt builder extracts persona instructions and keeps them consistent across turns.

## Deployment

- **GGUF / llama.cpp**: see `deployment/README.md`
- **ONNX**: see `deployment/onnx/ORT_GENAI.md`
- **Edge export notes**: `models/EDGE_EXPORT.md`

## Artifacts & Checkpoints (Not in Git)

Large files are excluded by `.gitignore`. Typical outputs:

- `artifacts/` (merged models, GGUF, checkpoints)
- `training_data/` (generated datasets)
- `deployment/llama.cpp*` (builds/binaries)

Keep these locally or in Drive; don’t commit them.

## Troubleshooting (Common)

- **Persona switching is weak** → dataset needs stronger persona examples.
- **Colab resets** → use chunked training + auto‑resume checkpoints.
- **GGUF conversion fails** → ensure tokenizer files exist in merged model.

## Roadmap Ideas

- Larger persona dataset generation
- Automated evaluation harness for persona adherence
- Mobile‑first inference integration (Android/iOS)

---

If you want: I can add a one‑click Colab launcher badge and a minimal “first run” script.
