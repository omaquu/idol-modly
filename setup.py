"""
setup.py - Extension setup script for IDOL Modly Extension.

Creates an isolated venv and installs all required dependencies.
Called by Modly at extension install time with:
    python setup.py <json_args>
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def pip(venv: Path, *args: str) -> None:
    is_win = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def python_exe_for(venv: Path) -> Path:
    is_win = platform.system() == "Windows"
    return venv / ("Scripts/python.exe" if is_win else "bin/python")


def setup(
    python_exe: str,
    ext_dir: Path,
    gpu_sm: int = 0,
    accelerator: str = "",
    platform_name: str = "",
) -> None:
    venv = ext_dir / "venv"

    # Resolve accelerator when not supplied
    if not accelerator:
        sys_name = platform.system()
        if sys_name == "Darwin":
            accelerator = "mps" if platform.machine() == "arm64" else "cpu"
        elif gpu_sm > 0:
            accelerator = "cuda"
        else:
            accelerator = "cpu"

    print("=" * 60)
    print("  IDOL Modly Extension - Setup")
    print("=" * 60)
    print(f"[idol-setup] accelerator={accelerator}  gpu_sm={gpu_sm}")
    print(f"[idol-setup] ext_dir={ext_dir}")
    print(f"[idol-setup] Creating isolated venv at {venv} ...")

    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # 1. Install PyTorch matching target accelerator
    if accelerator == "mps":
        print("[idol-setup] Apple Silicon (MPS) -> PyTorch from standard PyPI")
        pip(venv, "install", "torch>=2.5.0", "torchvision>=0.20.0")
    elif accelerator == "cuda":
        if gpu_sm >= 100:
            torch_index = "https://download.pytorch.org/whl/cu128"
            torch_pkgs = ["torch>=2.7.0", "torchvision>=0.22.0"]
            print(f"[idol-setup] GPU SM {gpu_sm} (Blackwell) -> PyTorch + CUDA 12.8")
        elif gpu_sm >= 70:
            torch_index = "https://download.pytorch.org/whl/cu124"
            torch_pkgs = ["torch==2.6.0", "torchvision==0.21.0"]
            print(f"[idol-setup] GPU SM {gpu_sm} -> PyTorch 2.6 + CUDA 12.4")
        else:
            torch_index = "https://download.pytorch.org/whl/cu118"
            torch_pkgs = ["torch==2.5.1", "torchvision==0.20.1"]
            print(f"[idol-setup] GPU SM {gpu_sm} (legacy) -> PyTorch 2.5 + CUDA 11.8")
        print("[idol-setup] Installing PyTorch ...")
        pip(venv, "install", *torch_pkgs, "--index-url", torch_index)
    else:
        print("[idol-setup] CPU-only -> PyTorch from standard PyPI")
        pip(venv, "install", "torch>=2.5.0", "torchvision>=0.20.0")

    # 2. Install tensordict with pinned version and --no-deps to prevent pip from upgrading PyTorch
    print("[idol-setup] Installing tensordict (pinned to avoid PyTorch version mismatch) ...")
    pip(venv, "install", "orjson", "cloudpickle", "pyvers")
    pip(venv, "install", "tensordict==0.7.0", "--no-deps")

    # 3. Install core dependencies
    print("[idol-setup] Installing core dependencies ...")
    pip(
        venv,
        "install",
        "transformers",
        "diffusers",
        "accelerate",
        "huggingface_hub",
        "safetensors",
        "numpy",
        "pillow",
        "plyfile",
        "tqdm",
        "trimesh",
        "scipy",
        "scikit-image",
        "einops",
    )

    # 4. Install the idol package itself in editable mode without reinstalling deps
    print("[idol-setup] Installing idol package in editable mode ...")
    pip(venv, "install", "-e", str(ext_dir), "--no-deps")

    # 5. Optional flash-attn on SM >= 80 CUDA
    if accelerator == "cuda" and gpu_sm >= 80:
        print("[idol-setup] Attempting optional flash-attn install ...")
        try:
            pip(venv, "install", "flash-attn", "--no-build-isolation")
            print("[idol-setup] flash-attn installed successfully.")
        except Exception:
            print("[idol-setup] flash-attn skipped (PyTorch SDPA will be used).")

    print("=" * 60)
    print(f"  IDOL Setup Complete! Venv ready at: {venv}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(
            python_exe=sys.argv[1],
            ext_dir=Path(sys.argv[2]),
            gpu_sm=int(sys.argv[3]),
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe=args.get("python_exe", sys.executable),
            ext_dir=Path(args.get("ext_dir", Path(__file__).resolve().parent)),
            gpu_sm=int(args.get("gpu_sm", 0)),
            accelerator=args.get("accelerator", ""),
            platform_name=args.get("platform", ""),
        )
    else:
        setup(
            python_exe=sys.executable,
            ext_dir=Path(__file__).resolve().parent,
            gpu_sm=89 if platform.system() != "Darwin" else 0,
        )
