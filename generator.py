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

import numpy as np
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

        # On Windows, torch.compile requires Triton which is not available in standard PyTorch wheels.
        # Running with optimize=False runs in native PyTorch CUDA eager mode, which is fast and avoids JIT crashes.
        optimize = False if sys.platform == "win32" else True

        if weights_path:
            logger.info(f"Loading IDOL from local weights: {weights_path} (optimize={optimize})")
            self._pipeline = IdolPipeline.from_pretrained(
                weights_path,
                device=self.device,
                optimize=optimize,
                image_encoders=encoder_overrides
            )
        else:
            logger.info(f"Loading IDOL from HuggingFace (speridlabs/idol) (optimize={optimize})...")
            self._pipeline = IdolPipeline.from_pretrained(
                "speridlabs/idol",
                device=self.device,
                optimize=optimize,
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

    def _run_projection(
        self,
        image: Image.Image,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """
        Projection Node: Project colors from a secondary photo onto an existing 3D mesh.
        """
        import torch
        import trimesh
        from colorize import colorize_from_points, _C0

        mesh_path = str(params.get("mesh_path", "")).strip()
        if not mesh_path:
            raise ValueError("The Projection node needs a mesh input — wire a 3D mesh node into it.")

        workspace_dir = self.outputs_dir.parent
        mp = Path(mesh_path)
        if mesh_path.startswith("/workspace/"):
            mesh_path = str(workspace_dir / mesh_path[len("/workspace/"):])
        elif not mp.is_absolute():
            mesh_path = str(workspace_dir / mesh_path)

        self._report(progress_cb, 5, "Loading input mesh...")
        mesh = trimesh.load(mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
            raise ValueError(f"Could not load valid triangle mesh from: {mesh_path}")

        # Parameters for secondary splat
        mode = params.get("mode", "synthetic")
        steps = int(params.get("steps", 30))
        num_gaussians = int(params.get("num_gaussians", 131072))
        cfg_strength = float(params.get("cfg_strength", 3.0))
        colorize_mode = str(params.get("colorize_mode", "Aligned (ICP)"))
        use_icp = "ICP" in colorize_mode
        color_boost = str(params.get("color_boost", "Vivid + Bright"))
        seed = int(params.get("seed", 42))

        if seed < 0:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        self._report(progress_cb, 15, "Generating secondary 3D Gaussians from image...")
        self._check_cancelled(cancel_event)

        output = self._pipeline(
            image=image,
            mode=mode,
            seed=seed,
            steps=steps,
            cfg_strength=cfg_strength,
            num_gaussians=num_gaussians,
            verbose=False,
        )

        self._check_cancelled(cancel_event)
        self._report(progress_cb, 80, f"Projecting color onto mesh ({'ICP' if use_icp else 'Direct'})...")

        xyz = output.gaussian.xyz.detach().cpu().numpy()
        opacity = output.gaussian.opacity.detach().cpu().numpy().reshape(-1)
        f_dc = output.gaussian.features_dc.detach().squeeze(1).cpu().numpy()
        rgb = np.clip(f_dc * _C0 + 0.5, 0.0, 1.0)

        mesh = colorize_from_points(
            mesh=mesh,
            src_xyz=xyz,
            src_rgb=rgb,
            opacity=opacity,
            use_icp=use_icp,
            pre_rotate=True,
            color_boost=color_boost,
        )

        self._check_cancelled(cancel_event)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        base_id = f"idol_proj_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        out_glb_path = self.outputs_dir / f"{base_id}.glb"
        mesh.export(str(out_glb_path), file_type="glb")

        self._report(progress_cb, 100, "Color projection complete!")
        return out_glb_path

    def _run_multiview(
        self,
        image: Image.Image,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """
        Multi-View Node: Automatically generates 3D model from Front view,
        then projects Back view and Side views in the correct order.
        """
        import torch
        import trimesh
        from colorize import colorize_from_points, _C0
        from splat_mesh import gaussian_object_to_glb

        mode = params.get("mode", "synthetic")
        steps = int(params.get("steps", 30))
        num_gaussians = int(params.get("num_gaussians", 131072))
        grid_resolution = int(params.get("grid_resolution", 224))
        mesh_smoothing = params.get("mesh_smoothing", "Medium")
        taubin = {"None": 0, "Light": 6, "Medium": 12, "Strong": 18}.get(mesh_smoothing, 12)
        cfg_strength = float(params.get("cfg_strength", 3.0))
        colorize_mode = str(params.get("colorize_mode", "Aligned (ICP)"))
        use_icp = "ICP" in colorize_mode
        color_boost = str(params.get("color_boost", "Vivid + Bright"))
        seed = int(params.get("seed", 42))

        if seed < 0:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        workspace_dir = getattr(self, "workspace_dir", None) or self.outputs_dir.parent

        # Extra image paths passed from Modly: Back View (slot 1), Left View (slot 2), Right View (slot 3)
        extra_raw = params.get("extra_image_paths") or []
        valid_extras = [str(p).strip() for p in extra_raw if p is not None and str(p).strip()]

        logger.info(f"[IDOL Multi-View] Starting pipeline: 1 primary view + {len(valid_extras)} secondary view(s).")

        # Step 1: Front view generation & mesh extraction
        total_steps = 1 + len(valid_extras)
        self._report(progress_cb, 5, f"Step 1/{total_steps}: Generating 3D model from Front view...")
        self._check_cancelled(cancel_event)

        front_output = self._pipeline(
            image=image,
            mode=mode,
            seed=seed,
            steps=steps,
            cfg_strength=cfg_strength,
            num_gaussians=num_gaussians,
            verbose=False,
        )

        self._check_cancelled(cancel_event)
        self._report(progress_cb, 35, f"Step 1/{total_steps}: Reconstructing watertight 3D mesh...")

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        base_id = f"idol_mv_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        temp_mesh_path = self.outputs_dir / f"{base_id}_temp.glb"

        gaussian_object_to_glb(
            front_output.gaussian,
            str(temp_mesh_path),
            grid_resolution=grid_resolution,
            taubin=taubin,
            vivid_color=True,
        )

        mesh = trimesh.load(str(temp_mesh_path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
            raise RuntimeError("Failed to extract valid triangle mesh from Front view.")

        # If no secondary views were connected, return the front mesh directly
        if not valid_extras:
            out_glb_path = self.outputs_dir / f"{base_id}.glb"
            mesh.export(str(out_glb_path), file_type="glb")
            if temp_mesh_path.exists():
                try:
                    temp_mesh_path.unlink()
                except Exception:
                    pass
            self._report(progress_cb, 100, "3D Mesh generation complete!")
            return out_glb_path

        def _load_image(raw_path: str) -> Image.Image:
            p = Path(raw_path)
            if raw_path.startswith("/workspace/") and workspace_dir:
                p = workspace_dir / raw_path[len("/workspace/"):]
            elif not p.is_absolute() and workspace_dir:
                p = workspace_dir / p
            if not p.is_file():
                raise FileNotFoundError(f"Secondary view image not found: {raw_path} -> {p}")
            return Image.open(p).convert("RGB")

        view_labels = ["Back view", "Left view", "Right view"]

        for idx, extra_p in enumerate(valid_extras):
            self._check_cancelled(cancel_event)
            step_idx = idx + 2
            label = view_labels[idx] if idx < len(view_labels) else f"View {step_idx}"

            pct_start = int(40 + (idx / len(valid_extras)) * 55)
            pct_mid = int(pct_start + (55 / (2 * len(valid_extras))))

            self._report(progress_cb, pct_start, f"Step {step_idx}/{total_steps}: Generating 3D splat from {label}...")
            sec_img = _load_image(extra_p)

            sec_output = self._pipeline(
                image=sec_img,
                mode=mode,
                seed=seed + idx + 1,
                steps=steps,
                cfg_strength=cfg_strength,
                num_gaussians=num_gaussians,
                verbose=False,
            )

            self._check_cancelled(cancel_event)
            self._report(progress_cb, pct_mid, f"Step {step_idx}/{total_steps}: Projecting {label} colors onto mesh...")

            xyz = sec_output.gaussian.xyz.detach().cpu().numpy()
            opacity = sec_output.gaussian.opacity.detach().cpu().numpy().reshape(-1)
            f_dc = sec_output.gaussian.features_dc.detach().squeeze(1).cpu().numpy()
            rgb = np.clip(f_dc * _C0 + 0.5, 0.0, 1.0)

            mesh = colorize_from_points(
                mesh=mesh,
                src_xyz=xyz,
                src_rgb=rgb,
                opacity=opacity,
                use_icp=use_icp,
                pre_rotate=True,
                color_boost=color_boost,
                blend_with_existing=True,
            )

        self._check_cancelled(cancel_event)
        out_glb_path = self.outputs_dir / f"{base_id}.glb"
        mesh.export(str(out_glb_path), file_type="glb")
        if temp_mesh_path.exists():
            try:
                temp_mesh_path.unlink()
            except Exception:
                pass

        self._report(progress_cb, 100, "Multi-view 3D model complete!")
        return out_glb_path

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
            image = Image.open(io.BytesIO(image_bytes))
        elif isinstance(image_bytes, str):
            image = Image.open(image_bytes)
        elif isinstance(image_bytes, Image.Image):
            image = image_bytes
        else:
            raise ValueError(f"Unsupported image input type: {type(image_bytes)}")

        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA" if "A" in image.mode or "transparency" in image.info else "RGB")

        # Ensure model is loaded
        if self._pipeline is None:
            self._report(progress_cb, 10, "Loading IDOL model weights...")
            self.load()

        self._check_cancelled(cancel_event)

        # Route based on node id
        node_id = os.environ.get("MODEL_NODE_ID") or (self.model_dir.name if self.model_dir else "generate")
        if node_id in ("splat_to_mesh", "splatToMesh"):
            return self._run_splat_to_mesh(image_bytes, params, progress_cb, cancel_event)
        elif node_id == "projection":
            return self._run_projection(image, params, progress_cb, cancel_event)
        elif node_id in ("multiview", "generate-mv", "generate-multiview"):
            return self._run_multiview(image, params, progress_cb, cancel_event)

        # Extract parameters for standard generation
        mode = params.get("mode", "synthetic")
        steps = int(params.get("steps", 30))
        num_gaussians = int(params.get("num_gaussians", 262144))
        mesh_detail = params.get("mesh_detail", params.get("grid_resolution", 9))
        mesh_smoothing = params.get("mesh_smoothing", "Medium")
        color_mode = params.get("color_mode", "Refined")
        texture = params.get("texture", "Auto")
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

        # Splat style: "spiky" preserves IDOL's signature needle-like anisotropic Gaussians
        # (great for stylized fur, mechs, bristles). "smooth" rounds them into softer droplets.
        splat_style = str(params.get("splat_style", "spiky")).lower()
        if splat_style == "smooth" and hasattr(output.gaussian, "scaling"):
            logger.info("Applying smooth splat regularization to Gaussians...")
            max_s = output.gaussian.scaling.max(dim=-1, keepdim=True).values
            output.gaussian.scaling = torch.clamp_min(output.gaussian.scaling, max_s * 0.28)

        # Save standard 3D Gaussian Splat PLY
        self._report(progress_cb, 70, "Exporting 3D Gaussian Splat (.ply)...")
        output.gaussian.save_ply(str(ply_path))
        try:
            output.gaussian.save_splat(str(ply_path.with_suffix(".splat")))
            logger.info(f"Splat sidecar written to {ply_path.with_suffix('.splat')}")
        except Exception as exc:
            logger.warning(f"Failed to write .splat sidecar: {exc}")

        # For "createSplat": export raw 3D Gaussian Splat (.ply and .splat) directly.
        if node_id == "createSplat":
            self._save_asset_thumbnail(ply_path, image)
            self._save_asset_thumbnail(ply_path.with_suffix(".splat"), image)
            self._report(progress_cb, 100, "Gaussian Splat complete!")
            return ply_path

        # Mesh extraction for "generate" node (IDOL Mesh) with full 2K Texture Baking
        self._report(progress_cb, 80, f"Reconstructing mesh & baking textures (detail={mesh_detail}, texture={texture})...")
        self._check_cancelled(cancel_event)

        try:
            reconstruct_gaussian_to_glb(
                gaussian=output.gaussian,
                output_glb_path=str(glb_path),
                mesh_detail=mesh_detail,
                mesh_smoothing=mesh_smoothing,
                color_mode=color_mode,
                texture=texture,
            )
        except Exception as e:
            logger.warning(f"Direct mesh extraction error: {e}. Falling back to PLY reconstruction...")
            extract_mesh_from_ply(
                ply_path=str(ply_path),
                output_glb_path=str(glb_path),
                mesh_detail=mesh_detail,
                mesh_smoothing=mesh_smoothing,
                color_mode=color_mode,
                texture=texture,
            )

        self._check_cancelled(cancel_event)
        self._save_asset_thumbnail(glb_path, image)
        self._save_asset_thumbnail(ply_path, image)
        self._save_asset_thumbnail(ply_path.with_suffix(".splat"), image)
        self._report(progress_cb, 100, "Generation complete!")
        return glb_path

    def _save_asset_thumbnail(self, asset_path: Path, image: Image.Image) -> None:
        """Cache a thumbnail in Modly's thumbnail directory so Library displays preview instantly."""
        try:
            import hashlib, json
            app_data = os.environ.get("APPDATA")
            if not app_data:
                return
            thumbs_dir = Path(app_data) / "Modly" / "thumbnails"
            thumbs_dir.mkdir(parents=True, exist_ok=True)
            workspace_dir = getattr(self, "workspace_dir", None) or self.outputs_dir.parent
            rel_path = f"Workflows/{asset_path.name}"

            key = hashlib.sha256(f"{os.path.abspath(str(workspace_dir))}\n{rel_path}".encode("utf-8")).hexdigest()[:32]
            st = asset_path.stat()
            mtime_ms = int(st.st_mtime * 1000)

            alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
            def to_b36(num):
                if num == 0: return "0"
                res = []
                num = abs(int(num))
                while num:
                    res.append(alphabet[num % 36])
                    num //= 36
                return "".join(reversed(res))

            version = f"{to_b36(mtime_ms)}-{to_b36(st.st_size)}-r1"
            thumb_path = thumbs_dir / f"{key}.{version}.webp"
            sidecar_path = thumbs_dir / f"{key}.json"

            sidecar_path.write_text(json.dumps({"workspaceDir": str(workspace_dir), "workspacePath": rel_path}), encoding="utf-8")

            thumb_img = image.copy()
            thumb_img.thumbnail((256, 256), Image.Resampling.LANCZOS)
            canvas = Image.new("RGBA", (256, 256), (24, 24, 27, 255))
            offset = ((256 - thumb_img.width) // 2, (256 - thumb_img.height) // 2)
            if thumb_img.mode == "RGBA":
                canvas.paste(thumb_img, offset, thumb_img)
            else:
                canvas.paste(thumb_img, offset)
            canvas.save(str(thumb_path), "WEBP", quality=90)
            logger.info(f"Auto-saved thumbnail to {thumb_path.name}")
        except Exception as e:
            logger.warning(f"Could not auto-save thumbnail: {e}")

    def _run_splat_to_mesh(self, input_data, params: dict, progress_cb, cancel_event) -> Path:
        """
        Convert an existing Gaussian Splat (.ply or .splat) into a high-quality watertight
        textured .glb mesh without re-running AI diffusion sampling.
        Connectable directly to downstream nodes like Prepare for Game.
        """
        from splat_mesh import reconstruct_gaussian_to_glb

        self._check_cancelled(cancel_event)
        self._report(progress_cb, 10, "Locating Gaussian splat input...")

        splat_path = None
        if isinstance(input_data, str) and (input_data.lower().endswith(".ply") or input_data.lower().endswith(".splat")):
            splat_path = Path(input_data)
        elif isinstance(input_data, Path) and (input_data.suffix.lower() in (".ply", ".splat")):
            splat_path = input_data
        elif isinstance(params.get("mesh_path"), str) and params.get("mesh_path").strip():
            splat_path = Path(params["mesh_path"].strip())
        elif isinstance(params.get("input_path"), str) and params.get("input_path").strip():
            splat_path = Path(params["input_path"].strip())
        elif isinstance(input_data, bytes):
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            # Detect header
            is_splat = (len(input_data) % 32 == 0) and not input_data[:10].startswith(b"ply")
            ext = ".splat" if is_splat else ".ply"
            temp_in = self.outputs_dir / f"input_{int(time.time())}{ext}"
            temp_in.write_bytes(input_data)
            splat_path = temp_in

        # Fallback: scan workspace for most recent .ply or .splat if path wasn't passed directly
        if not splat_path or not splat_path.exists():
            workspace_dir = getattr(self, "workspace_dir", None) or self.outputs_dir.parent
            recent_splats = sorted(list(workspace_dir.glob("**/*.ply")) + list(workspace_dir.glob("**/*.splat")),
                                   key=lambda p: p.stat().st_mtime, reverse=True)
            if recent_splats:
                splat_path = recent_splats[0]
                logger.info(f"Using most recent workspace splat: {splat_path}")

        if not splat_path or not splat_path.exists():
            raise ValueError(f"No valid .ply or .splat file found for Splat to Mesh. Provided: {input_data}")

        mesh_detail = params.get("mesh_detail", 10)
        mesh_smoothing = params.get("mesh_smoothing", "Medium")
        color_mode = params.get("color_mode", "Refined")
        texture = params.get("texture", "Baked 4K")

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        base_id = f"{splat_path.stem}_mesh"
        out_glb = self.outputs_dir / f"{base_id}.glb"

        self._report(progress_cb, 30, f"Reconstructing mesh & baking textures (detail={mesh_detail}, texture={texture})...")
        self._check_cancelled(cancel_event)

        reconstruct_gaussian_to_glb(
            gaussian=str(splat_path),
            output_glb_path=str(out_glb),
            mesh_detail=mesh_detail,
            mesh_smoothing=mesh_smoothing,
            color_mode=color_mode,
            texture=texture,
        )

        self._check_cancelled(cancel_event)
        self._report(progress_cb, 100, f"Mesh ready: {out_glb.name}")
        return out_glb
