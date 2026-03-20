from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _run(cmd: list[str], cwd: Optional[str] = None) -> None:
    print(">", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def _find_first(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def ensure_llama_cpp(llama_cpp_dir: Path, *, allow_clone: bool) -> None:
    if llama_cpp_dir.exists():
        return
    if not allow_clone:
        raise SystemExit(f"llama.cpp not found at {llama_cpp_dir} (pass --llama-cpp-dir or --clone)")

    # Best-effort clone (requires network).
    llama_cpp_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", str(llama_cpp_dir)])


def find_convert_script(llama_cpp_dir: Path) -> Path:
    candidates = [
        llama_cpp_dir / "convert_hf_to_gguf.py",
        llama_cpp_dir / "convert-hf-to-gguf.py",
        llama_cpp_dir / "examples" / "convert_legacy_llama.py",
    ]
    out = _find_first(candidates)
    if out is None:
        raise SystemExit(f"Could not find HF->GGUF conversion script under: {llama_cpp_dir}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a merged HF model directory to GGUF (llama.cpp).")
    ap.add_argument("--model", required=True, help="Path to merged HF model directory (e.g. ./output/final_merged).")
    ap.add_argument("--output", required=True, help="Output GGUF path (e.g. ./models/model.gguf).")
    ap.add_argument(
        "--llama-cpp-dir",
        default="./deployment/llama.cpp",
        help="Path to llama.cpp checkout (default: ./deployment/llama.cpp).",
    )
    ap.add_argument(
        "--clone",
        action="store_true",
        help="If llama.cpp is missing, attempt to clone it (requires git + internet).",
    )
    ap.add_argument("--python", default=sys.executable, help="Python interpreter for llama.cpp conversion script.")
    args = ap.parse_args()

    hf_model_dir = Path(args.model).resolve()
    out_path = Path(args.output).resolve()
    llama_cpp_dir = Path(args.llama_cpp_dir).resolve()

    if not hf_model_dir.exists():
        raise SystemExit(f"Model dir not found: {hf_model_dir}")

    ensure_llama_cpp(llama_cpp_dir, allow_clone=args.clone)
    convert_script = find_convert_script(llama_cpp_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run([args.python, str(convert_script), str(hf_model_dir), "--outfile", str(out_path)])
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()

