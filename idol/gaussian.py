import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


class Gaussian:
    """Built from the Gaussian decoder's raw outputs; the activated values are what renderers and PLY files use.

    xyz [N, 3] positions, features_dc [N, 1, 3] colour (degree-0 spherical harmonics), scaling [N, 3],
    rotation [N, 4] unit quaternions (w, x, y, z), opacity [N, 1]. The object is z-up, inside [-0.5, 0.5]^3."""

    def __init__(self, raw_xyz, features_dc, raw_scaling, raw_rotation, raw_opacity,
                 scaling_bias: float, opacity_bias: float, minimum_kernel_size: float):
        device = raw_xyz.device
        aabb = torch.tensor([-0.5, -0.5, -0.5, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        scaling_offset = inverse_softplus(torch.tensor(scaling_bias)).to(device)
        opacity_offset = inverse_sigmoid(torch.tensor(opacity_bias)).to(device)
        rotation_offset = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

        self.xyz = raw_xyz * aabb[None, 3:] + aabb[None, :3]
        self.features_dc = features_dc
        scaling = F.softplus(raw_scaling + scaling_offset)
        self.scaling = torch.sqrt(torch.square(scaling) + minimum_kernel_size ** 2)
        self.rotation = F.normalize(raw_rotation + rotation_offset[None, :])
        self.opacity = torch.sigmoid(raw_opacity + opacity_offset)

    def save_ply(self, path):
        """Write the standard 3D Gaussian Splatting PLY (z-up object turned to the y-up convention viewers use)."""
        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # Clamp before the inverse activations: a saturated opacity (exactly 1.0 after the sigmoid) or a zero
        # scale would otherwise be written as +/-inf, which some PLY viewers reject.
        eps = 1e-6
        opacities = inverse_sigmoid(self.opacity.clamp(eps, 1 - eps)).detach().cpu().numpy()
        scale = torch.log(self.scaling.clamp_min(eps)).detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()

        # z-up -> y-up: rotate +90 degrees about x, i.e. (x, y, z) -> (x, -z, y)
        xyz = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1)
        rotation = quaternion_multiply(np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]), rotation)

        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1).astype(np.float32)
        names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
        elements = np.empty(attributes.shape[0], dtype=[(name, 'f4') for name in names])
        for column, name in enumerate(names):
            elements[name] = attributes[:, column]
        PlyData([PlyElement.describe(elements, 'vertex')]).write(path)

    def to_splat_bytes(self) -> bytes:
        """Serialize Gaussians to standard 32-byte .splat binary format for real-time web viewers."""
        xyz = self.xyz.detach().cpu().numpy()
        scale = self.scaling.detach().cpu().numpy()
        opacity = self.opacity.detach().cpu().numpy().reshape(-1, 1)
        f_dc = self.features_dc.detach().squeeze(1).cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()

        # z-up -> y-up
        xyz = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1)
        scale = np.stack([scale[:, 0], scale[:, 2], scale[:, 1]], axis=1)
        rotation = quaternion_multiply(np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]), rotation)

        C0 = 0.28209479177387814
        rgb = np.clip((f_dc * C0 + 0.5) * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(opacity * 255, 0, 255).astype(np.uint8)
        rgba = np.concatenate([rgb, alpha], axis=1)

        rot = rotation / np.linalg.norm(rotation, axis=-1, keepdims=True)
        rot_u8 = np.clip(rot * 128 + 128, 0, 255).astype(np.uint8)

        # Sort front to back / largest opacity*extent first
        order = np.argsort(-opacity[:, 0] * np.prod(scale, axis=-1))
        xyz, scale, rgba, rot_u8 = xyz[order], scale[order], rgba[order], rot_u8[order]

        data = np.concatenate([
            xyz.astype(np.float32).view(np.uint8).reshape(-1, 12),
            scale.astype(np.float32).view(np.uint8).reshape(-1, 12),
            rgba.reshape(-1, 4),
            rot_u8.reshape(-1, 4),
        ], axis=1).reshape(-1)
        return data.tobytes()

    def save_splat(self, path):
        """Write standard 3DGS .splat file."""
        with open(path, "wb") as f:
            f.write(self.to_splat_bytes())


def quaternion_multiply(a, b):
    """Hamilton product a * b of (w, x, y, z) quaternions; a is one quaternion, b is (N, 4)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1)
