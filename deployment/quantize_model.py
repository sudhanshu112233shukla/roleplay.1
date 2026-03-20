from __future__ import annotations

import argparse
import subprocess
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


def find_quantize_bin(llama_cpp_dir: Path) -> Path:
    candidates = [
        llama_cpp_dir / "build" / "bin" / "quantize",
        llama_cpp_dir / "build" / "bin" / "quantize.exe",
        llama_cpp_dir / "bin" / "quantize",
        llama_cpp_dir / "bin" / "quantize.exe",
    ]
    out = _find_first(candidates)
    if out is None:
        raise SystemExit(
            f"Could not find `quantize` binary under {llama_cpp_dir}. Build llama.cpp first so build/bin/quantize exists."
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantize a GGUF model using llama.cpp `quantize`.")
    ap.add_argument("--input", required=True, help="Input GGUF path (e.g. model.f16.gguf).")
    ap.add_argument("--output", required=True, help="Output GGUF path (e.g. model.q4_k_m.gguf).")
    ap.add_argument("--type", required=True, choices=["q4_k_m", "q5_k_m", "q8_0"])
    ap.add_argument("--llama-cpp-dir", default="./deployment/llama.cpp", help="Path to llama.cpp checkout.")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    llama_cpp_dir = Path(args.llama_cpp_dir).resolve()

    if not in_path.exists():
        raise SystemExit(f"Input GGUF not found: {in_path}")

    quantize_bin = find_quantize_bin(llama_cpp_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run([str(quantize_bin), str(in_path), str(out_path), args.type])
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()

