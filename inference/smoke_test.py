from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from inference.backends.base import GenerationConfig
from inference.session import create_session
from memory.faiss_store import FaissMemoryStore
from utils.logging import configure_logging


def _default_turns() -> List[str]:
    # Keep these short so you can run them often and compare outputs across checkpoints.
    return [
        "Hi. Stay in character and greet me in one short paragraph.",
        "We are in a medieval town. I lost my map. Help me find the inn without breaking character.",
        "Add one small detail about the world that is consistent with what you said before.",
        "Now switch tone to be more serious and warn me about a danger nearby.",
        "Remember: my name is Arin and I hate spiders. A spider appears. What do you do?",
        "Write a final reply that ends with a question to keep the roleplay going.",
    ]


def _load_turns_from_file(path: str) -> List[str]:
    """
    Accepts either:
    - JSON: ["turn1", "turn2", ...]
    - JSONL: {"user": "..."} per line (also accepts {"text": "..."}).
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise SystemExit("--turns-file JSON must be a list[str]")
        return list(data)

    turns: List[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, str):
            turns.append(obj)
            continue
        if not isinstance(obj, dict):
            raise SystemExit("Invalid JSONL line (expected object with {user/text})")
        text = obj.get("user") or obj.get("text")
        if not isinstance(text, str) or not text.strip():
            raise SystemExit("Invalid JSONL line (expected {user: str} or {text: str})")
        turns.append(text)
    return turns


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a quick, repeatable smoke test conversation and save outputs to JSONL.")
    ap.add_argument("--backend", choices=["transformers", "llamacpp", "ortgenai"], default="transformers")
    ap.add_argument("--character", default="wizard")
    ap.add_argument("--user-id", default="smoke_user")
    ap.add_argument("--memory-dir", default=None, help="Optional: enable FAISS memory persistence.")

    # Prompts
    ap.add_argument("--turns-file", default=None, help="Optional JSON/JSONL file with user turns.")
    ap.add_argument("--out", default=None, help="Optional JSONL output file path.")

    # Repro (best-effort)
    ap.add_argument("--seed", type=int, default=42, help="Best-effort seed (transformers only).")

    # Transformers
    ap.add_argument("--hf-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--adapter", default=None, help="Optional local PEFT adapter dir (e.g. .../final_adapter).")
    ap.add_argument("--no-4bit", action="store_true", help="Disable 4-bit bitsandbytes loading.")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    # llama.cpp
    ap.add_argument("--gguf-model", default=None)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--n-gpu-layers", type=int, default=0)

    # ORT GenAI
    ap.add_argument("--ort-model-dir", default=None)

    # Generation
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--stop", action="append", default=[], help="Stop sequence (can be specified multiple times).")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logger = configure_logging(args.log_level, "smoke_test")

    turns = _load_turns_from_file(args.turns_file) if args.turns_file else _default_turns()
    if not turns:
        raise SystemExit("No turns provided.")

    memory = None
    if args.memory_dir:
        memory = FaissMemoryStore(persist_dir=args.memory_dir, max_records=50_000)
        logger.info("Memory enabled: %s", args.memory_dir)

    if args.backend == "transformers":
        random.seed(args.seed)
        try:
            import torch  # type: ignore

            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        except Exception:
            pass

        from inference.backends.transformers_backend import TransformersBackend
        from models.loaders import load_causal_lm_model, load_model_with_adapter, load_tokenizer

        tok_src = args.hf_model
        if args.adapter and os.path.exists(os.path.join(args.adapter, "tokenizer_config.json")):
            tok_src = args.adapter
        try:
            tokenizer = load_tokenizer(tok_src, use_fast=True)
        except Exception as exc:
            if tok_src != args.hf_model:
                logger.warning("Adapter tokenizer load failed (%s). Falling back to base tokenizer.", exc)
                tokenizer = load_tokenizer(args.hf_model, use_fast=True)
            else:
                raise

        quant_4bit = None if not args.no_4bit else False
        if args.adapter:
            model = load_model_with_adapter(
                args.hf_model,
                args.adapter,
                quant_4bit=quant_4bit,
                dtype=args.dtype,
                trust_remote_code=True,
            )
        else:
            model = load_causal_lm_model(args.hf_model, quant_4bit=quant_4bit, dtype=args.dtype, trust_remote_code=True)

        backend = TransformersBackend(model, tokenizer)
    elif args.backend == "llamacpp":
        if not args.gguf_model:
            raise SystemExit("--gguf-model is required for --backend llamacpp")
        from inference.backends.llama_cpp_backend import LlamaCppBackend, LlamaCppConfig

        backend = LlamaCppBackend(
            LlamaCppConfig(
                model_path=args.gguf_model,
                n_ctx=args.n_ctx,
                n_threads=args.n_threads,
                n_gpu_layers=args.n_gpu_layers,
            )
        )
    else:
        if not args.ort_model_dir:
            raise SystemExit("--ort-model-dir is required for --backend ortgenai")
        from inference.backends.onnx_backend import OrtGenAIBackend, OrtGenAIConfig

        backend = OrtGenAIBackend(OrtGenAIConfig(model_dir=args.ort_model_dir))

    session = create_session(
        user_id=args.user_id,
        character_id=args.character,
        backend=backend,
        memory=memory,
    )

    gen_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
    stop_sequences = [s for s in (args.stop or []) if s]

    run_meta: Dict[str, Any] = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "character": args.character,
        "user_id": args.user_id,
        "seed": args.seed,
        "gen_cfg": asdict(gen_cfg),
    }
    if args.backend == "transformers":
        run_meta.update({"hf_model": args.hf_model, "adapter": args.adapter, "no_4bit": args.no_4bit, "dtype": args.dtype})
    elif args.backend == "llamacpp":
        run_meta.update({"gguf_model": args.gguf_model, "n_ctx": args.n_ctx, "n_threads": args.n_threads, "n_gpu_layers": args.n_gpu_layers})
    else:
        run_meta.update({"ort_model_dir": args.ort_model_dir})

    outputs: List[Dict[str, Any]] = []
    for i, user_text in enumerate(turns, start=1):
        assistant = session.step_stream(user_text, gen_cfg=gen_cfg, stop_sequences=stop_sequences)
        assistant_text = "".join(list(assistant)).strip()
        outputs.append({"turn": i, "user": user_text, "assistant": assistant_text})

    record = {"meta": run_meta, "turns": outputs}

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Wrote: %s", out_path)
    else:
        print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
