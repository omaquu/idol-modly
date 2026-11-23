# IDOL – Modly Extension & SDK

**Single-image to 3D Gaussian Splatting & Watertight 3D Meshes**

IDOL converts a single 2D image (photo, render, or AI-generated image) into a high-density **3D Gaussian Splat (`.ply`)** and reconstructed **watertight 3D mesh (`.glb`)** running locally on your GPU.

This repository is packaged as an official **Modly extension** and can also be used as a standalone Python SDK or CLI.

<div align="center">

[![Modly Extension](https://img.shields.io/badge/Modly-Extension-6366f1?style=for-the-badge&logo=cube)](https://github.com/omaquu/idol-modly)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow?style=for-the-badge)](https://huggingface.co/speridlabs/idol)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge)](LICENSE)

</div>

---

> [!IMPORTANT]
> ### ⚠️ Required: Accept Model Terms on Hugging Face!
> IDOL uses two gated image encoder models hosted on Hugging Face. Before your first generation, you **must accept the terms** on Hugging Face with your account:
> 
> 1. 👉 **[black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev)** — Click **"Agree and access repository"**
> 2. 👉 **[facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)** — Click **"Agree and access repository"**
> 
> Then ensure your Hugging Face token is active:
> - **In Modly:** Modly manages your Hugging Face token in its settings automatically.
> - **In Terminal:** Run `huggingface-cli login` or set `export HF_TOKEN="your_token_here"`.
> 
> *(Note: If Meta's approval for DINOv3 is pending review, this extension automatically falls back to an identical open mirror so you can generate immediately without waiting!)*

---

## 🚀 Using in Modly

### 1. Install Extension in Modly
1. Open **Modly**
2. Go to **Extensions** → **Install from URL**
3. Paste:
   ```
   https://github.com/omaquu/idol-modly
   ```
4. Click **Install**. Modly will set up the isolated Python virtual environment and register the extension.

### 2. Available Modly Nodes
Once installed, IDOL adds three nodes to your workflow canvas:

* **IDOL (Mesh)**: Converts an input image into a watertight, vertex-colored `.glb` 3D mesh ready for 3D printing, game engines, or rendering.
* **IDOL (Gaussian Splat)**: Generates a high-density `.ply` 3D Gaussian Splat point cloud (up to 1,048,576 Gaussians) for real-time volumetric viewing.
* **IDOL (Projection)**: Multi-view color projection node. Wire an existing 3D mesh (e.g. from `IDOL (Mesh)`) and a secondary photo (e.g. back view or side view). It generates a secondary point cloud and transfers colors onto the mesh using 3D orientation alignment, optional ICP registration, and normal-aware color transfer.

### 3. Parameters in Modly

| Parameter | Default | Description |
|---|---|---|
| **Input Mode** | `synthetic` | `synthetic` for AI-generated / 3D renders; `real` for real-world photos. |
| **Sampling Steps** | 30 | Flow-matching steps (10–60). More steps = higher detail. |
| **Gaussians** | 131,072 | Gaussian budget (32k to 1M). Higher = denser cloud, more detail. |
| **Mesh Resolution** | 160 | Grid resolution for Marching Cubes mesh reconstruction (128–256). |
| **CFG Strength** | 3.0 | Classifier-free guidance strength. |
| **Seed** | 42 | Random seed (-1 for random). |

---

## 🔑 Prerequisites & Authentication Details

IDOL downloads weights (~3.1 GB) on first run from [`speridlabs/idol`](https://huggingface.co/speridlabs/idol).

To authenticate for the gated encoders:
1. Create a free account at [huggingface.co](https://huggingface.co).
2. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Read permissions).
3. Accept access at:
   - [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev)
   - [facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)
4. Log in locally:
   ```bash
   huggingface-cli login
   ```

---

## 💻 Manual / Standalone Installation

```bash
git clone https://github.com/omaquu/idol-modly.git
cd idol-modly
pip install -e .
```

### Standalone CLI Inference

```bash
# Generate 3D Gaussian Splat (.ply)
python -m idol.sample --image photo.jpg --mode real
python -m idol.sample --image renders/ --mode synthetic --out outputs/
```

### Standalone Python API

```python
from PIL import Image
from idol.pipeline import IdolPipeline
from splat_mesh import gaussian_object_to_glb

# Load pipeline (Windows automatically uses optimize=False for CUDA eager mode)
pipeline = IdolPipeline.from_pretrained("speridlabs/idol", device="cuda", optimize=False)

# Generate Gaussians
output = pipeline(
    image=Image.open("input.png"),
    mode="synthetic",
    steps=30,
    num_gaussians=131072
)

# Save Gaussian Splat
output.gaussian.save_ply("output.ply")

# Extract watertight GLB mesh
gaussian_object_to_glb(output.gaussian, "output.glb", grid_resolution=160)
```

---

## 🏗️ Hardware Requirements

| | Minimum | Recommended |
|---|---|---|
| **VRAM** | 8 GB | 10+ GB |
| **GPU** | NVIDIA (CUDA sm_75+) / Apple Silicon (MPS) | NVIDIA RTX 3060 / 4070+ |
| **Python** | 3.10+ | 3.10 – 3.12 |
| **PyTorch** | 2.5+ | Latest (2.6.0+cu124 recommended) |

---

## 📄 License & Credits

- **IDOL**: [Speridlabs](https://speridlabs.com) — [Hugging Face](https://huggingface.co/speridlabs/idol)
- **Modly Architecture**: Inspired by [modly-triposplat-extension](https://github.com/lightningpixel/modly-triposplat-extension) by Lorchie / lightningpixel
- **License**: Apache 2.0
