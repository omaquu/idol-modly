"""
setup.py - Environment Setup for IDOL Modly Extension
Called by Modly's Electron app on extension install:
  python setup.py '{"python_exe":"...", "ext_dir":"...", "gpu_sm":86, "accelerator":"cuda", "platform":"win32"}'
Or can be run manually:
  python setup.py
"""

import os
import sys
import json
import logging
import platform
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[idol-setup] %(message)s")
logger = logging.getLogger("idol-setup")


def run_cmd(cmd, cwd=None):
    logger.info(f"Running: {' '.join(str(c) for c in cmd)}")
    subprocess.check_call([str(c) for c in cmd], cwd=cwd)


def main():
    print("=" * 60)
    print("  IDOL Modly Extension - Setup")
    print("=" * 60)

    # Parse arguments from Modly or set defaults
    payload = {}
    if len(sys.argv) > 1:
        try:
            payload = json.loads(sys.argv[1])
            logger.info(f"Received setup payload: {payload}")
        except Exception:
            logger.info("No JSON payload detected; using environment defaults.")

    python_exe = payload.get("python_exe", sys.executable)
    ext_dir = Path(payload.get("ext_dir", Path(__file__).resolve().parent))
    accelerator = payload.get("accelerator", "cuda" if platform.system() != "Darwin" else "mps")
    gpu_sm = payload.get("gpu_sm", None)

    logger.info(f"Extension directory: {ext_dir}")
    logger.info(f"Python executable:   {python_exe}")
    logger.info(f"Target accelerator:  {accelerator} (SM: {gpu_sm})")

    req_file = ext_dir / "requirements.txt"

    # Install dependencies from requirements.txt
    if req_file.exists():
        logger.info("Installing core dependencies from requirements.txt...")
        run_cmd([python_exe, "-m", "pip", "install", "-r", str(req_file), "--quiet"], cwd=ext_dir)

    # Install idol package in editable mode
    logger.info("Installing idol package in editable mode...")
    run_cmd([python_exe, "-m", "pip", "install", "-e", ".", "--no-deps", "--quiet"], cwd=ext_dir)

    # Try installing flash-attn if on compatible CUDA SM >= 80
    if accelerator == "cuda" and (gpu_sm is None or gpu_sm >= 80):
        logger.info("Attempting optional flash-attn installation for Ampere/Ada/Blackwell GPUs...")
        try:
            run_cmd([
                python_exe, "-m", "pip", "install",
                "flash-attn", "--no-build-isolation", "--quiet"
            ], cwd=ext_dir)
            logger.info("flash-attn installed successfully.")
        except Exception:
            logger.info("flash-attn installation skipped (PyTorch SDPA will be used).")

    print("\n" + "=" * 60)
    print("  IDOL Extension Setup Complete!")
    print("=" * 60)
    print("Note: Make sure your Hugging Face account has access to:")
    print("  1. facebook/dinov3-vith16plus-pretrain-lvd1689m")
    print("  2. black-forest-labs/FLUX.2-dev")
    print("Log in via terminal if not already authenticated:")
    print("  huggingface-cli login")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
