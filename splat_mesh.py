"""
splat_mesh.py - High-fidelity Gaussian Splat to Watertight Mesh Extraction.
Ported from ComfyUI / TripoSplat Surface Nets & Covariance Density Rasterizer.

Features:
  1. Covariance 3-sigma GPU density & color volume rasterization.
  2. Scale-floor to inter-Gaussian spacing to seal holes and bridge thin surfels (e.g. armors/weapons).
  3. Otsu-thresholded level with automatic fragmentation detection & lowering.
  4. Vectorized Surface Nets (dual contouring) producing watertight surfaces without Marching Cubes pitting.
  5. Volume-preserving Taubin smoothing (prevents feature shrinkage).
  6. Sub-voxel filament & whisker recovery pass.
  7. Trilinearly sampled color volume with color sharpening.
"""

import logging
import numpy as np
import torch
import trimesh
from scipy.ndimage import (map_coordinates, minimum as _ndi_min, maximum as _ndi_max,
                           distance_transform_edt, binary_fill_holes)
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

logger = logging.getLogger("idol.splat_mesh")

_C0 = 0.28209479177387814  # SH band-0 DC -> base RGB


def _inverse_covariance(scale, quat):
    """Per-splat Sigma^-1 = R diag(1/s^2) R^T. scale (N,3) linear std, quat (N,4) wxyz."""
    q = quat / quat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(-1, 3, 3)
    inv_s2 = 1.0 / scale.clamp_min(1e-8) ** 2
    return torch.einsum("nij,nj,nkj->nik", R, inv_s2, R)


def _splat_density(xyz, opacity, scale, quat, rgb, res, kernel, device,
                   color_sharpen=2.0, chunk=4096, col_dtype=torch.float16):
    """Density grid + co-splatted colour volume."""
    pad = 4.0 * scale.median()
    lo = xyz.amin(0) - pad
    hi = xyz.amax(0) + pad
    ext = (hi - lo).clamp_min(1e-8)
    ex, ey, ez = ext.tolist()
    voxel = max((ex * ey * ez) ** (1.0 / 3.0) / res, max(ex, ey, ez) / (2.0 * res))
    dx, dy, dz = (torch.ceil(ext / voxel).long() + 1).tolist()

    sinv = _inverse_covariance(scale, quat)
    kreq = torch.ceil(3.0 * scale.amax(-1) / voxel).long().clamp(1, int(kernel))
    sharp = color_sharpen != 1.0
    vol = torch.zeros(dx * dy * dz, device=device)
    colvol = torch.zeros(dx * dy * dz, 3, device=device, dtype=col_dtype)
    wcol = torch.zeros(dx * dy * dz, device=device, dtype=col_dtype) if sharp else None
    for k in range(1, int(kernel) + 1):
        sel = (kreq == k).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        rng = torch.arange(-k, k + 1, device=device, dtype=torch.float32)
        off = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
        for st in range(0, sel.numel(), chunk):
            gi = sel[st:st + chunk]
            cc = xyz[gi]
            idx = ((cc - lo) / voxel).round()[:, None, :] + off[None]
            d = (lo + idx * voxel) - cc[:, None, :]
            quad = torch.einsum("bmi,bij,bmj->bm", d, sinv[gi], d)
            wgt = opacity[gi, None] * torch.exp(-0.5 * quad)
            wgt = torch.where(quad < 9.0, wgt, torch.zeros_like(wgt))
            ii = idx.long()
            ix = ii[..., 0].clamp(0, dx - 1)
            iy = ii[..., 1].clamp(0, dy - 1)
            iz = ii[..., 2].clamp(0, dz - 1)
            flat = (ix * (dy * dz) + iy * dz + iz).reshape(-1)
            vol.index_add_(0, flat, wgt.reshape(-1))
            wp = wgt.pow(color_sharpen) if sharp else wgt
            colvol.index_add_(0, flat, (wp[..., None] * rgb[gi, None, :]).reshape(-1, 3).to(col_dtype))
            if sharp:
                wcol.index_add_(0, flat, wp.reshape(-1).to(col_dtype))
    colnorm = (wcol if sharp else vol).reshape(dx, dy, dz)
    return (vol.reshape(dx, dy, dz), colvol.reshape(dx, dy, dz, 3), colnorm,
            lo.detach().cpu().numpy(), float(voxel))


def _otsu_level(values, bins=256):
    """Otsu threshold: density value that best splits inside/outside."""
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) * 0.5
    w = np.cumsum(hist)
    mu = np.cumsum(hist * centers)
    wf = w[-1] - w
    mb = mu / np.where(w > 0, w, 1.0)
    mf = (mu[-1] - mu) / np.where(wf > 0, wf, 1.0)
    var_b = w * wf * (mb - mf) ** 2
    var_b[(w <= 0) | (wf <= 0)] = -1.0
    return float(centers[int(np.argmax(var_b))])


def _surface_nets(vol, level, voxel, origin, device):
    """Vectorized Surface Nets: one dual vertex per sign-changing cell at its
    edge-crossing mean, quads wound outward. Returns (verts world, faces)."""
    vol = vol.to(device=device, dtype=torch.float32)
    dx, dy, dz = vol.shape
    origin_t = torch.as_tensor(origin, device=device, dtype=torch.float32)
    empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64))
    if dx < 2 or dy < 2 or dz < 2:
        return empty

    inside = vol >= level
    cs8 = [inside[ox:ox + dx - 1, oy:oy + dy - 1, oz:oz + dz - 1]
           for ox, oy, oz in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                              (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))]
    any_in = cs8[0] | cs8[1] | cs8[2] | cs8[3] | cs8[4] | cs8[5] | cs8[6] | cs8[7]
    all_in = cs8[0] & cs8[1] & cs8[2] & cs8[3] & cs8[4] & cs8[5] & cs8[6] & cs8[7]
    active = any_in & ~all_in
    nv = int(active.sum())
    if nv == 0:
        return empty

    del any_in, all_in, cs8
    ac = active.nonzero(as_tuple=False)
    offs = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                         [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], device=device)
    offf = offs.to(torch.float32)
    edges = torch.tensor([[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
                          [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]], device=device)
    e0, e1 = edges[:, 0], edges[:, 1]
    oe0, oe1 = offf[e0], offf[e1]

    cstep = 1 << 18
    loc = []
    for st in range(0, nv, cstep):
        ci = ac[st:st + cstep, None, :] + offs[None]
        cval = vol[ci[..., 0], ci[..., 1], ci[..., 2]]
        csl = cval >= level
        v0, v1 = cval[:, e0], cval[:, e1]
        cross = (csl[:, e0] != csl[:, e1])[..., None].to(torch.float32)
        denom = v1 - v0
        t = torch.where(denom.abs() > 1e-12, (level - v0) / denom, torch.full_like(denom, 0.5)).clamp(0, 1)
        pts = torch.lerp(oe0, oe1, t[..., None])
        loc.append((pts * cross).sum(1) / cross.sum(1).clamp_min(1.0))
    local = torch.cat(loc, 0) if len(loc) > 1 else loc[0]
    verts = origin_t + (ac.to(torch.float32) + local) * voxel
    del loc, local

    vid = torch.full((dx - 1, dy - 1, dz - 1), -1, dtype=torch.int32, device=device)
    vid[active] = torch.arange(nv, dtype=torch.int32, device=device)
    del active

    faces = []

    def emit(cr, sol, a, b, d, c):
        valid = cr & (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
        if not bool(valid.any()):
            return
        a, b, c, d, sol = a[valid], b[valid], c[valid], d[valid], sol[valid]
        p2, p4 = torch.where(sol, b, c), torch.where(sol, c, b)
        faces.append(torch.stack([a, p2, d], 1))
        faces.append(torch.stack([a, d, p4], 1))

    a = inside[0:dx - 1, 1:dy - 1, 1:dz - 1]
    emit(a != inside[1:dx, 1:dy - 1, 1:dz - 1], a,
         vid[:, 0:dy - 2, 0:dz - 2], vid[:, 1:dy - 1, 0:dz - 2],
         vid[:, 1:dy - 1, 1:dz - 1], vid[:, 0:dy - 2, 1:dz - 1])
    a = inside[1:dx - 1, 0:dy - 1, 1:dz - 1]
    emit(a != inside[1:dx - 1, 1:dy, 1:dz - 1], a,
         vid[0:dx - 2, :, 0:dz - 2], vid[0:dx - 2, :, 1:dz - 1],
         vid[1:dx - 1, :, 1:dz - 1], vid[1:dx - 1, :, 0:dz - 2])
    a = inside[1:dx - 1, 1:dy - 1, 0:dz - 1]
    emit(a != inside[1:dx - 1, 1:dy - 1, 1:dz], a,
         vid[0:dx - 2, 0:dy - 2, :], vid[1:dx - 1, 0:dy - 2, :],
         vid[1:dx - 1, 1:dy - 1, :], vid[0:dx - 2, 1:dy - 1, :])

    if not faces:
        return empty
    return verts.cpu().numpy().astype(np.float32), torch.cat(faces, 0).cpu().numpy().astype(np.int64)


def _taubin_smooth(verts, faces, iters, lam=0.5, mu=-0.53):
    """Volume-preserving lambda|mu smoothing (no Laplacian shrinkage)."""
    if iters <= 0 or len(verts) == 0 or len(faces) == 0:
        return verts
    nv = len(verts)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], 0)
    e = np.concatenate([e, e[:, ::-1]], 0)
    adj = coo_matrix((np.ones(len(e), np.float32), (e[:, 0], e[:, 1])), shape=(nv, nv)).tocsr()
    adj.data[:] = 1.0
    deg = np.clip(np.asarray(adj.sum(1)).ravel(), 1.0, None).astype(np.float32)[:, None]
    v = verts.astype(np.float32)
    for _ in range(int(iters)):
        for fac in (lam, mu):
            v = v + np.float32(fac) * ((adj @ v) / deg - v)
    return np.ascontiguousarray(v)


def _clean_components(verts, faces, min_verts):
    """Drop floaters (< min_verts) and inner shells."""
    nv = len(verts)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], 0)
    ncomp, label = connected_components(
        coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv)), directed=False)
    flabel = label[faces[:, 0]]
    keep = np.bincount(label, minlength=ncomp) >= min_verts
    if keep.sum() > 1:
        fcount = np.bincount(flabel, minlength=ncomp)
        largest = np.where(keep, fcount, -1).argmax()
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        cvol = np.bincount(flabel, weights=np.einsum("ij,ij->i", v0, np.cross(v1, v2)), minlength=ncomp)
        cidx = np.arange(ncomp)
        cmin = np.stack([_ndi_min(verts[:, a], label, cidx) for a in range(3)], 1)
        cmax = np.stack([_ndi_max(verts[:, a], label, cidx) for a in range(3)], 1)
        tol = 1e-4 * (cmax[largest] - cmin[largest]).max()
        enclosed = (cmin >= cmin[largest] - tol).all(1) & (cmax <= cmax[largest] + tol).all(1)
        inner = enclosed & (np.sign(cvol) != np.sign(cvol[largest])) & (np.arange(ncomp) != largest)
        keep &= ~inner
    faces = faces[keep[flabel]]
    if len(faces) == 0:
        return verts[:0], faces
    used = np.unique(faces)
    remap = np.full(nv, -1, np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces]


def _gaussian_blur3d(vol: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 3D gaussian blur on a (X,Y,Z) volume."""
    import torch.nn.functional as F
    r = max(1, int(np.ceil(2.5 * sigma)))
    x = torch.arange(-r, r + 1, device=vol.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = (k / k.sum()).reshape(1, 1, -1)
    v = vol[None, None]
    v = F.conv3d(v, k.reshape(1, 1, -1, 1, 1), padding=(r, 0, 0))
    v = F.conv3d(v, k.reshape(1, 1, 1, -1, 1), padding=(0, r, 0))
    v = F.conv3d(v, k.reshape(1, 1, 1, 1, -1), padding=(0, 0, r))
    return v[0, 0]


def splat_to_mesh(xyz, opacity, scale, quat, rgb, *, resolution, device,
                  kernel=5, level_bias=0.4, min_component=500, min_opacity=0.02,
                  color_sharpen=2.0, taubin=12, scale_floor_frac=0.9, vivid_color=False,
                  vol_smooth=0.7):
    """
    Convert 3D Gaussians into a solid, watertight mesh with Surface Nets.
    Respects covariance footprint, floors scale to eliminate holes in thin armors/surfaces.
    """
    opacity = opacity.reshape(-1)
    keep = opacity >= min_opacity
    xyz, scale, quat, opacity, rgb = xyz[keep], scale[keep], quat[keep], opacity[keep], rgb[keep]
    if xyz.shape[0] == 0:
        return None

    # Floor each Gaussian's scale to inter-Gaussian spacing to prevent holes
    spacing = 0.0
    if xyz.shape[0] > 8:
        xn = xyz.detach().cpu().numpy()
        sub = xn[np.random.default_rng(0).choice(len(xn), min(20000, len(xn)), replace=False)]
        spacing = float(np.median(cKDTree(xn).query(sub, k=2)[0][:, 1]))
        if scale_floor_frac > 0:
            scale = scale.clamp_min(float(scale_floor_frac * spacing))

    vol, colvol, colnorm, origin, voxel = _splat_density(
        xyz, opacity, scale, quat, rgb, resolution, kernel, device, color_sharpen=color_sharpen,
    )

    if vol_smooth > 0:
        ratio = (spacing / voxel) if (spacing > 0 and voxel > 0) else 0.4
        mult = min(max(ratio / 0.9, 1.0), 2.5)
        s = float(vol_smooth) * mult
        vol = _gaussian_blur3d(vol, s)
        colnorm = _gaussian_blur3d(colnorm.float(), s)
        colvol = torch.stack([_gaussian_blur3d(colvol[..., ch].float(), s) for ch in range(3)], dim=-1)

    colvol_np = colvol.float().cpu().numpy()
    colnorm_np = colnorm.float().cpu().numpy()

    _valid = colnorm_np > max(1e-6, 1e-4 * float(colnorm_np.max()))
    if not _valid.all() and _valid.any():
        _ind = distance_transform_edt(~_valid, return_distances=False, return_indices=True)
        colvol_np = colvol_np[_ind[0], _ind[1], _ind[2]]
        colnorm_np = colnorm_np[_ind[0], _ind[1], _ind[2]]
        del _ind
    del colvol, colnorm

    vmin, vmax = float(vol.min()), float(vol.max())
    occ = vol[vol > vmax * 1e-3]
    if occ.numel() == 0:
        return None
    otsu = _otsu_level(occ.cpu().numpy())

    # Connectivity-driven level selection to prevent holes in weak/thin structures
    best = None
    best_ratio = -1.0
    for lb_mult in (1.0, 0.7, 0.5):
        level = min(max(otsu * level_bias * lb_mult, vmin + 1e-6 * (vmax - vmin)),
                    vmax - 1e-6 * (vmax - vmin))
        v_try, f_try = _surface_nets(vol, level, voxel, origin, device)
        if min_component > 0 and len(f_try) > 0:
            v_try, f_try = _clean_components(v_try, f_try, min_component)
        if len(v_try) == 0 or len(f_try) == 0:
            continue
        e = np.concatenate([f_try[:, [0, 1]], f_try[:, [1, 2]], f_try[:, [0, 2]]], 0)
        ncomp, label = connected_components(
            coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                       shape=(len(v_try), len(v_try))), directed=False)
        fcount = np.bincount(label[f_try[:, 0]], minlength=ncomp)
        ratio = float(fcount.max()) / max(len(f_try), 1)
        if ratio > best_ratio:
            best, best_ratio = (v_try, f_try), ratio
        if ratio >= 0.90:
            break
    del vol
    if best is None:
        return None
    verts, faces = best

    # Taubin volume-preserving smoothing (preserves sharp armor ridges)
    verts = _taubin_smooth(verts, faces, taubin)

    # Trilinear colour sampling
    coords = ((verts - origin) / voxel).T
    num = np.stack([map_coordinates(colvol_np[..., c], coords, order=1, mode="nearest") for c in range(3)], -1)
    den = map_coordinates(colnorm_np, coords, order=1, mode="nearest")
    col = np.clip(num / np.clip(den, 1e-8, None)[:, None], 0.0, 1.0).astype(np.float32)

    if vivid_color:
        luma = (col @ np.array([0.299, 0.587, 0.114], np.float32))[:, None]
        amt = np.float32(0.5) * np.clip(luma / np.float32(0.4), 0.0, 1.0)
        col = np.clip(luma + (col - luma) * (1.0 + amt), 0.0, 1.0).astype(np.float32)

    return verts.astype(np.float32), faces.astype(np.int64), col


def gaussian_object_to_glb(
    gaussian,
    output_glb_path: str,
    grid_resolution: int = 160,
    vol_smooth: float = 0.7,
    vivid_color: bool = True,
) -> str:
    """
    Extract a watertight, solid .glb mesh directly from an Idol Gaussian instance.
    """
    device = gaussian.xyz.device

    xyz = gaussian.xyz.detach().float()
    scale = gaussian.scaling.detach().float()
    opacity = gaussian.opacity.detach().reshape(-1).float()
    quat = gaussian.rotation.detach().float()
    rgb = (gaussian.features_dc[:, 0, :].detach().float() * _C0 + 0.5).clamp(0, 1)

    res = int(grid_resolution)
    out = splat_to_mesh(
        xyz=xyz,
        opacity=opacity,
        scale=scale,
        quat=quat,
        rgb=rgb,
        resolution=res,
        device=device,
        vol_smooth=vol_smooth,
        vivid_color=vivid_color,
    )

    if out is None:
        raise RuntimeError("Splat to mesh produced no surface.")

    verts, faces, colors = out

    # Convert native Z-up to standard glTF/Modly Y-up: (x, -z, y)
    verts = np.stack([verts[:, 0], -verts[:, 2], verts[:, 1]], axis=1)

    rgba = np.hstack([
        (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8),
        np.full((len(verts), 1), 255, dtype=np.uint8),
    ])

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgba)
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception as exc:
        logger.debug(f"fill_holes: {exc}")

    mesh.export(output_glb_path, file_type="glb")
    return output_glb_path


def extract_mesh_from_ply(
    ply_path: str,
    output_glb_path: str,
    grid_resolution: int = 160,
    vol_smooth: float = 0.7,
    vivid_color: bool = True,
    device=None,
) -> str:
    """
    Extract a watertight, solid .glb mesh from a 3D Gaussian Splat PLY file.
    """
    from plyfile import PlyData

    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    # Coordinates
    xyz_np = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)

    # Note: PLY written by save_ply has already been converted to Y-up: (x, -z, y).
    # To run splat_to_mesh in native Z-up: (x, z, -y)
    xyz_zup = np.stack([xyz_np[:, 0], xyz_np[:, 2], -xyz_np[:, 1]], axis=1)

    # Opacities
    if "opacity" in vertex.data.dtype.names:
        raw_op = vertex["opacity"]
        if raw_op.min() < 0 or raw_op.max() > 1:
            opacities = 1.0 / (1.0 + np.exp(-raw_op))
        else:
            opacities = raw_op
    else:
        opacities = np.ones(len(xyz_zup), dtype=np.float32)

    # Scales
    if all(k in vertex.data.dtype.names for k in ("scale_0", "scale_1", "scale_2")):
        raw_s = np.column_stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]])
        scales = np.exp(raw_s).astype(np.float32)
    else:
        scales = np.full((len(xyz_zup), 3), 0.01, dtype=np.float32)

    # Rotations
    if all(k in vertex.data.dtype.names for k in ("rot_0", "rot_1", "rot_2", "rot_3")):
        quats = np.column_stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]]).astype(np.float32)
    else:
        quats = np.tile([1.0, 0.0, 0.0, 0.0], (len(xyz_zup), 1)).astype(np.float32)

    # Colors
    if all(k in vertex.data.dtype.names for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        sh_dc = np.column_stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]])
        rgb = np.clip(sh_dc * _C0 + 0.5, 0.0, 1.0).astype(np.float32)
    elif all(k in vertex.data.dtype.names for k in ("red", "green", "blue")):
        rgb = (np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]) / 255.0).astype(np.float32)
    else:
        rgb = np.full((len(xyz_zup), 3), 0.5, dtype=np.float32)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = splat_to_mesh(
        xyz=torch.from_numpy(xyz_zup).to(device=device, dtype=torch.float32),
        opacity=torch.from_numpy(opacities).to(device=device, dtype=torch.float32),
        scale=torch.from_numpy(scales).to(device=device, dtype=torch.float32),
        quat=torch.from_numpy(quats).to(device=device, dtype=torch.float32),
        rgb=torch.from_numpy(rgb).to(device=device, dtype=torch.float32),
        resolution=int(grid_resolution),
        device=device,
        vol_smooth=vol_smooth,
        vivid_color=vivid_color,
    )

    if out is None:
        raise RuntimeError(f"Could not extract surface from PLY: {ply_path}")

    verts, faces, colors = out
    # Convert Z-up to standard Y-up
    verts = np.stack([verts[:, 0], -verts[:, 2], verts[:, 1]], axis=1)

    rgba = np.hstack([
        (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8),
        np.full((len(verts), 1), 255, dtype=np.uint8),
    ])

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgba)
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception as exc:
        logger.debug(f"fill_holes: {exc}")

    mesh.export(output_glb_path, file_type="glb")
    return output_glb_path
