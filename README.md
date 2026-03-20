# Roleplay AI System (Edge-Ready Architecture)

This repo currently contains two Colab notebooks for training a TinyLlama roleplay model.

This refactor adds a modular, production-oriented architecture while preserving all core
functionality already present in the notebooks:

- Single-stage QLoRA + LoRA SFT training on roleplay-ish conversation datasets.
- Multi-stage pipeline: Stage 1 SFT (instruction), Stage 2 SFT (roleplay), Stage 3 DPO (preference).
- Retrieval memory system (FAISS + SentenceTransformers).
- Character profile system (YAML profiles).
- Dynamic dataset expansion (log + curate).
- Prompt builder that injects: character, world state, memory, chat history, user input.
- World state and emotion tracking engines.

## Structure

- `models/`: model loading, LoRA, export (adapter + merged), edge-export notes
- `characters/`: YAML profiles + loader/repository
- `prompt_builder/`: robust prompt construction + truncation
- `memory/`: FAISS-backed long-term memory store + importance heuristics
- `world_state/`: world-state schema + update engine
- `emotion_engine/`: emotion-state schema + update engine
- `datasets/`: roleplay JSONL datasets (data only)
- `training_data/`: canonical dataset schema, HF adapters, local JSONL helpers, expansion utilities
- `training/`: single-stage + multi-stage training pipelines (SFT + DPO)
- `inference/`: chat session orchestration + backends (Transformers; stubs for edge backends)
- `notebooks/`: refactored notebooks that call the modules (original notebooks remain at repo root)

## Quick Start (Colab)

1. Clone the repo in Colab.
2. Open `notebooks/roleplay_training_refactored.ipynb` or `notebooks/roleplay_multistage_refactored.ipynb`.
3. If you keep getting disconnected / lose GPU, use chunked training notebooks:
   - `notebooks/roleplay_training_chunked_hf.ipynb` (HF datasets, streaming)
   - `notebooks/roleplay_training_chunked_local.ipynb` (local JSONL/CSV, streaming)

## Quick Start (Local)

This repo does not pin dependencies by default. Start with:

```powershell
pip install -r requirements.txt
python -m training.train_single_stage --help
python -m training.train_multistage --help
python -m training.train_chunked_sft --help
python -m training.train_chunked_sft_hf --help
```

## Notes

- Model weights are not stored in this repo.
- For edge deployment (GGUF / llama.cpp / ONNX), see `models/EDGE_EXPORT.md`.

## Characters (Static + Dynamic)

- Static characters: YAML files under `characters/profiles/` (e.g. `wizard.yaml`). Run with `--character wizard`.
- Dynamic personas: if enabled (default), users can type messages like **"Be Iron Man today and talk to me"** and the session will switch persona even if no YAML exists.
  - Optional auto-save: pass `--auto-save-dynamic-profiles-dir <dir>` to save frequently used dynamic personas under `<dir>/user/<id>.yaml` (load later via `--character user/<id>`).

## On-Device Deployment

See `deployment/README.md` for the end-to-end flow (GGUF/llama.cpp and ONNX Runtime GenAI).
