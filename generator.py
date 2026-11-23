"""
generator.py - Modly Extension Generator for IDOL
Integrates Speridlabs' IDOL (Single-Image to 3D Gaussian Splatting & Mesh) into Modly.
"""

import io
import os
import sys
import time
import uuid
import logging
import threading
from pathlib import Path
from typing import Optional, Callable, Union

from PIL import Image

# Setup logging
logger = logging.getLogger("modly.extension.idol")

# Import Modly BaseGenerator if running inside Modly, with fallback for standalone
try:
    from services.generators.base import BaseGenerator, GenerationCancelled
except ImportError:
    class GenerationCancelled(Exception):
        pass

    class BaseGenerator:
        def __init__(self, model_dir: Optional[Union[str, Path]] = None, outputs_dir: Optional[Union[str, Path]] = None):
            self.model_dir = Path(model_dir) if model_dir else Path(".")
            self.outputs_dir = Path(outputs_dir) if outputs_dir else Path("./outputs")
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            self._model = None

        def is_downloaded(self) -> bool:
            return True

        def load(self) -> None:
            pass

        def unload(self) -> None:
            pass

        def _report(self, progress_cb: Optional[Callable[[int, str], None]], percent: int, msg: str) -> None:
            if progress_cb:
                try:
                    progress_cb(percent, msg)
                except Exception:
                    pass

        def _check_cancelled(self, cancel_event: Optional[threading.Event]) -> None:
            if cancel_event and cancel_event.is_set():
                raise GenerationCancelled("Generation cancelled by user.")


def _ensure_hf_token():
    """Ensure Hugging Face token is set in environment if available."""
    if os.environ.get("HF_TOKEN"):
        return
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.is_file():
        os.environ["HF_TOKEN"] = token_file.read_text(encoding="utf-8").strip()


def _get_encoder_overrides() -> Optional[dict]:
    """Check if Meta DINOv3 is accessible; if awaiting review, use open mirror."""
    from huggingface_hub import auth_check
    overrides = {}
    try:
        auth_check("facebook/dinov3-vith16plus-pretrain-lvd1689m")
    except Exception:
        logger.info("Official Meta DINOv3 awaiting access review; using verified open mirror.")
        overrides["dinov3"] = "bschooled/dinov3-vith16plus-pretrain-lvd1689m"
    return overrides if overrides else None


def _pick_device() -> str:
    """Best available torch device: CUDA, Apple Silicon MPS, or CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class IdolGenerator(BaseGenerator):
    MODEL_ID = "idol"
    DISPLAY_NAME = "IDOL"
    VRAM_GB = 8

    def __init__(self, model_dir: Optional[Union[str, Path]] = None, outputs_dir: Optional[Union[str, Path]] = None):
        super().__init__(model_dir, outputs_dir)
        self.device = None
        self._pipeline = None

    def _weights_dir(self) -> Path:
        """Find weights directory, reusing 'generate' node directory if available."""
        if self.model_dir.name == "generate":
            return self.model_dir
        shared = self.model_dir.parent / "generate"
        if (shared / "denoiser.safetensors").exists():
            return shared
        return self.model_dir

    def is_downloaded(self) -> bool:
        """Check if model weights exist locally."""
        return (self._weights_dir() / "denoiser.safetensors").exists()

    def load(self) -> None:
        """Load the IDOL pipeline into memory."""
        if self._pipeline is not None:
            return

        import torch
        if self.device is None:
            self.device = _pick_device()

        _ensure_hf_token()
        encoder_overrides = _get_encoder_overrides()

        logger.info(f"Loading IDOL pipeline on device {self.device}...")

        if not self.is_downloaded() and hasattr(self, "_auto_download"):
            self._auto_download()

        weights_path = None
        wd = self._weights_dir()
        if (wd / "denoiser.safetensors").exists():
            weights_path = str(wd)

        # Import IdolPipeline from idol package
        from idol.pipeline import IdolPipeline

        if weights_path:
            logger.info(f"Loading IDOL from local weights: {weights_path}")
            self._pipeline = IdolPipeline.from_pretrained(
                weights_path,
                device=self.device,
                image_encoders=encoder_overrides
            )
        else:
            logger.info("Loading IDOL from HuggingFace (speridlabs/idol)...")
            self._pipeline = IdolPipeline.from_pretrained(
                "speridlabs/idol",
                device=self.device,
                image_encoders=encoder_overrides
            )

        logger.info("IDOL pipeline successfully loaded.")

    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        logger.info("Unloading IDOL model...")
        self._pipeline = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def generate(
        self,
        image_bytes: Union[bytes, str, Image.Image],
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """
        Main generation entry point invoked by Modly.

        :param image_bytes: Raw bytes of the input image, file path, or PIL Image.
        :param params: UI parameters defined in manifest.json.
        :param progress_cb: Optional callback(percent: int, message: str).
        :param cancel_event: Optional threading.Event to signal cancellation.
        :return: Path to the generated output file (.glb or .ply).
        """
        import torch
        from splat_mesh import gaussian_object_to_glb, extract_mesh_from_ply

        self._check_cancelled(cancel_event)
        self._report(progress_cb, 5, "Preparing input image...")

        # Parse image
        if isinstance(image_bytes, bytes):
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif isinstance(image_bytes, str):
            image = Image.open(image_bytes).convert("RGB")
        elif isinstance(image_bytes, Image.Image):
            image = image_bytes.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_bytes)}")

        # Ensure model is loaded
        if self._pipeline is None:
            self._report(progress_cb, 10, "Loading IDOL model weights...")
            self.load()

        self._check_cancelled(cancel_event)

        # Extract parameters
        mode = params.get("mode", "synthetic")
        steps = int(params.get("steps", 30))
        num_gaussians = int(params.get("num_gaussians", 131072))
        grid_resolution = int(params.get("grid_resolution", 160))
        cfg_strength = float(params.get("cfg_strength", 3.0))
        seed = int(params.get("seed", 42))

        if seed < 0:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        self._report(progress_cb, 20, f"Generating {num_gaussians:,} 3D Gaussians (steps={steps})...")
        self._check_cancelled(cancel_event)

        # Execute IDOL sampling
        t0 = time.time()
        output = self._pipeline(
            image=image,
            mode=mode,
            seed=seed,
            steps=steps,
            cfg_strength=cfg_strength,
            num_gaussians=num_gaussians,
            verbose=False,
        )
        elapsed = time.time() - t0
        logger.info(f"IDOL sampling completed in {elapsed:.2f}s.")

        self._check_cancelled(cancel_event)

        # Prepare output destination
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        base_id = f"idol_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        ply_path = self.outputs_dir / f"{base_id}.ply"
        glb_path = self.outputs_dir / f"{base_id}.glb"

        # Save standard 3D Gaussian Splat PLY
        self._report(progress_cb, 70, "Exporting 3D Gaussian Splat (.ply)...")
        output.gaussian.save_ply(str(ply_path))

        # Check node type: "createSplat" returns .ply, "generate" returns .glb mesh
        node_id = self.model_dir.name if self.model_dir else "generate"
        if node_id == "createSplat":
            self._report(progress_cb, 100, "Gaussian Splat complete!")
            return ply_path

        # Mesh extraction for "generate" node
        self._report(progress_cb, 80, f"Reconstructing watertight mesh (grid={grid_resolution})...")
        self._check_cancelled(cancel_event)

        try:
            gaussian_object_to_glb(
                gaussian=output.gaussian,
                output_glb_path=str(glb_path),
                grid_resolution=grid_resolution,
            )
        except Exception as e:
            logger.warning(f"Direct mesh extraction error: {e}. Falling back to PLY reconstruction...")
            extract_mesh_from_ply(
                ply_path=str(ply_path),
                output_glb_path=str(glb_path),
                grid_resolution=grid_resolution,
            )

        self._check_cancelled(cancel_event)
        self._report(progress_cb, 100, "Generation complete!")
        return glb_path
