from __future__ import annotations

import argparse
import os
import sys

from inference.backends.base import GenerationConfig
from inference.session import create_session
from memory.faiss_store import FaissMemoryStore
from utils.logging import configure_logging


def main() -> None:
    ap = argparse.ArgumentParser(description="Roleplay chat runner (edge backends supported).")
    ap.add_argument("--backend", choices=["transformers", "llamacpp", "ortgenai"], default="llamacpp")
    ap.add_argument("--character", default="wizard", help="Character id (yaml under characters/profiles).")
    ap.add_argument("--user-id", default="user1")
    ap.add_argument("--memory-dir", default=None, help="If set, persists FAISS memory under this directory.")
    ap.add_argument(
        "--disable-dynamic-personas",
        action="store_true",
        help="Disable interpreting user messages like 'Be X' / 'Roleplay as X' as persona switches.",
    )
    ap.add_argument(
        "--auto-save-dynamic-profiles-dir",
        default=None,
        help="If set, auto-saves frequently used dynamic personas under <dir>/user/<id>.yaml.",
    )
    ap.add_argument("--auto-save-after-turns", type=int, default=50, help="Turns before auto-saving a dynamic persona.")

    # Transformers backend
    ap.add_argument("--hf-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument(
        "--adapter",
        default=None,
        help="Optional local PEFT adapter dir (e.g. .../final_adapter). If set, loads base model + adapter (no merge).",
    )
    ap.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit bitsandbytes loading for transformers backend (useful for CPU-only environments).",
    )
    ap.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="dtype for non-quantized transformers loading (ignored when 4-bit is used).",
    )

    # llama.cpp backend
    ap.add_argument("--gguf-model", default=None, help="Path to GGUF model file for llama.cpp.")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--n-gpu-layers", type=int, default=0)

    # ORT GenAI backend
    ap.add_argument("--ort-model-dir", default=None, help="Path to ORT GenAI model directory.")

    # Generation config
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--stop", action="append", default=[], help="Stop sequence (can be specified multiple times).")
    ap.add_argument("--no-stream", action="store_true", help="Disable streaming; print the full response at once.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logger = configure_logging(args.log_level, "chat_cli")

    # Memory is optional but recommended for roleplay continuity.
    memory = None
    if args.memory_dir:
        memory = FaissMemoryStore(persist_dir=args.memory_dir, max_records=50_000)
        logger.info("Memory enabled: %s", args.memory_dir)

    if args.backend == "transformers":
        from models.loaders import load_model_with_adapter, load_causal_lm_model, load_tokenizer
        from inference.backends.transformers_backend import TransformersBackend

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
            model = load_causal_lm_model(
                args.hf_model,
                quant_4bit=quant_4bit,
                dtype=args.dtype,
                trust_remote_code=True,
            )
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
        allow_dynamic_personas=not args.disable_dynamic_personas,
        auto_save_dynamic_profiles_dir=args.auto_save_dynamic_profiles_dir,
        auto_save_after_turns=args.auto_save_after_turns,
    )
    gen_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)

    logger.info("Starting chat. Ctrl+C to exit.")
    try:
        while True:
            user_text = input("\nYou> ").strip()
            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                break

            print("\nAssistant>\n", end="")
            sys.stdout.flush()

            if args.no_stream:
                assistant_text = session.step(user_text, gen_cfg=gen_cfg)
                print(assistant_text)
                continue

            for chunk in session.step_stream(user_text, gen_cfg=gen_cfg, stop_sequences=args.stop):
                print(chunk, end="")
                sys.stdout.flush()
            print()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
