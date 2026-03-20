from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _find_first(paths):
    for p in paths:
        if p and Path(p).exists():
            return str(Path(p))
    return None


def find_convert_script(llama_cpp_dir: Path) -> str:
    candidates = [
        llama_cpp_dir / "convert_hf_to_gguf.py",
        llama_cpp_dir / "convert-hf-to-gguf.py",
        llama_cpp_dir / "examples" / "convert_legacy_llama.py",
    ]
    out = _find_first(candidates)
    if out is None:
        raise SystemExit(f"Could not find HF->GGUF conversion script under: {llama_cpp_dir}")
    return out


def find_quantize_bin(llama_cpp_dir: Path) -> str:
    candidates = [
        llama_cpp_dir / "build" / "bin" / "quantize",
        llama_cpp_dir / "build" / "bin" / "quantize.exe",
        llama_cpp_dir / "bin" / "quantize",
        llama_cpp_dir / "bin" / "quantize.exe",
    ]
    out = _find_first(candidates)
    if out is None:
        raise SystemExit(
            "Could not find `quantize` binary. Build llama.cpp first (CMake) so `quantize` exists under build/bin/."
        )
    return out


def run(cmd, cwd: Optional[str] = None) -> None:
    print(">", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert merged HF model -> GGUF and optionally quantize (llama.cpp).")
    ap.add_argument("--llama-cpp-dir", required=True, help="Path to a local llama.cpp checkout.")
    ap.add_argument("--hf-model-dir", required=True, help="Path to merged HF model directory (e.g. final_merged).")
    ap.add_argument("--out-dir", required=True, help="Output directory for GGUF files.")
    ap.add_argument("--out-name", default=None, help="Base filename for GGUF output (default: directory name).")
    ap.add_argument(
        "--quant",
        default=None,
        help="Quantization to produce (e.g. q4_k_m). If omitted, only unquantized GGUF is produced.",
    )
    ap.add_argument("--python", default=sys.executable, help="Python to run llama.cpp conversion script.")
    args = ap.parse_args()

    llama_cpp_dir = Path(args.llama_cpp_dir).resolve()
    hf_model_dir = Path(args.hf_model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not llama_cpp_dir.exists():
        raise SystemExit(f"llama.cpp dir not found: {llama_cpp_dir}")
    if not hf_model_dir.exists():
        raise SystemExit(f"HF model dir not found: {hf_model_dir}")

    convert_script = find_convert_script(llama_cpp_dir)
    base_name = args.out_name or hf_model_dir.name
    out_gguf = out_dir / f"{base_name}.f16.gguf"

    # Convert
    run([args.python, convert_script, str(hf_model_dir), "--outfile", str(out_gguf)])

    if not args.quant:
        print(f"Done: {out_gguf}")
        return

    # Quantize
    quantize_bin = find_quantize_bin(llama_cpp_dir)
    out_quant = out_dir / f"{base_name}.{args.quant}.gguf"
    run([quantize_bin, str(out_gguf), str(out_quant), args.quant])
    print(f"Done: {out_quant}")


if __name__ == "__main__":
    main()

