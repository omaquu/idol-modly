"""
colorize.py - Color projection for IDOL's Projection node.

Transfers per-vertex color from a 3D Gaussian-splat point cloud onto an
input mesh (GLB/OBJ/PLY). The splat is generated from a secondary view
(e.g., back or side photo); the mesh comes from the primary generator.

Features:
  1. Rotates the splat from native Z-up into standard glTF Y-up.
  2. Coarse-aligns by centroid and bounding-box scale.
  3. Automatic orientation search (recovers yaw / axis swaps).
  4. Optional fine-registration with point-to-point ICP (Kabsch/SVD).
  5. Normal-aware inverse-distance-weighted k-NN color transfer.
  6. Color boost options (Natural, Vivid, Bright, Vivid + Bright).
"""

import numpy as np

_C0 = 0.28209479177387814  # SH band-0 DC -> base RGB

# Splat native frame (Z-up) -> Modly/glTF (Y-up)
_SPLAT_TO_GLTF = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def _bbox_diag(pts):
    return float(np.linalg.norm(pts.max(0) - pts.min(0)))


def _coarse_align(src_xyz, tgt_xyz):
    """Match centroid and bounding box scale."""
    src_c = src_xyz.mean(0)
    tgt_c = tgt_xyz.mean(0)
    src_d = _bbox_diag(src_xyz)
    tgt_d = _bbox_diag(tgt_xyz)
    scale = (tgt_d / src_d) if src_d > 1e-9 else 1.0
    return (src_xyz - src_c) * scale + tgt_c


def _apply_boost(col, mode):
    """Post-process transferred 0..1 colors."""
    if mode in ("Vivid", "Vivid + Bright"):
        luma = (col @ np.array([0.299, 0.587, 0.114], np.float64))[:, None]
        amt = 0.5 * np.clip(luma / 0.4, 0.0, 1.0)
        col = np.clip(luma + (col - luma) * (1.0 + amt), 0.0, 1.0)
    if mode in ("Bright", "Vivid + Bright"):
        col = np.clip(col ** 0.7, 0.0, 1.0)
    return col


def _estimate_normals(pts, k=12):
    """Estimate outward-facing normals via PCA over k nearest neighbours."""
    from scipy.spatial import cKDTree

    k = min(k, len(pts))
    _, idx = cKDTree(pts).query(pts, k=k)
    nbr = pts[idx]
    nbr = nbr - nbr.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nbr, nbr) / k
    _w, v = np.linalg.eigh(cov)
    normals = v[:, :, 0]
    flip = np.einsum("ni,ni->n", normals, pts - pts.mean(0)) < 0
    normals[flip] = -normals[flip]
    return normals


def _rot_y(theta):
    """Rotation matrix around vertical (Y) axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _axis_aligned_rotations():
    """All 24 proper rotations of a cube."""
    import itertools

    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs)):
                M[row, col] = s
            if abs(np.linalg.det(M) - 1.0) < 1e-6:
                mats.append(M)
    return mats


def _auto_orient(src_xyz, tgt_xyz, n_sample=20000, accept_margin=0.85):
    """Recover orientation by minimizing mean distance to target."""
    from scipy.spatial import cKDTree

    if len(tgt_xyz) > n_sample:
        sel = np.random.default_rng(0).choice(len(tgt_xyz), n_sample, replace=False)
        probe = tgt_xyz[sel]
    else:
        probe = tgt_xyz
    center = tgt_xyz.mean(0)
    tree = cKDTree(src_xyz)

    def cost(R):
        rp = (probe - center) @ R + center
        d, _ = tree.query(rp, k=1)
        return float(d.mean())

    c_identity = cost(np.eye(3))
    R0 = min(_axis_aligned_rotations(), key=cost)
    best_a = min(range(-45, 46, 3), key=lambda a: cost(_rot_y(np.deg2rad(a)) @ R0))
    R = _rot_y(np.deg2rad(best_a)) @ R0

    if cost(R) >= accept_margin * c_identity:
        return src_xyz
    return (src_xyz - center) @ R.T + center


def _icp_refine(src_xyz, tgt_verts, iters=40, sample=20000, tol=1e-6):
    """Rigid point-to-point ICP (Kabsch/SVD)."""
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(tgt_verts)
        cur = np.asarray(src_xyz, dtype=np.float64).copy()
        if len(cur) > sample:
            sub = np.random.default_rng(0).choice(len(cur), sample, replace=False)
        else:
            sub = np.arange(len(cur))
        diag = _bbox_diag(tgt_verts)
        for _ in range(iters):
            P = cur[sub]
            _, idx = tree.query(P, k=1)
            Q = tgt_verts[idx]
            pc, qc = P.mean(0), Q.mean(0)
            H = (P - pc).T @ (Q - qc)
            U, _s, Vt = np.linalg.svd(H)
            D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
            R = Vt.T @ D @ U.T
            t = qc - R @ pc
            cur = cur @ R.T + t
            if diag > 0 and (np.linalg.norm(t) + abs(np.arccos(
                    np.clip((np.trace(R) - 1) / 2, -1, 1))) * diag) < tol * diag:
                break
        return cur
    except Exception as exc:
        print(f"[colorize] ICP refine skipped: {exc}")
        return np.asarray(src_xyz, dtype=np.float64)


def colorize_from_points(
    mesh,
    src_xyz,
    src_rgb,
    opacity=None,
    use_icp=False,
    pre_rotate=True,
    k=12,
    min_opacity=0.1,
    color_boost="Vivid + Bright",
    blend_with_existing=False,
):
    """
    Transfer per-vertex color from point cloud onto a trimesh.Trimesh.

    :param mesh: trimesh.Trimesh instance.
    :param src_xyz: Nx3 array of points.
    :param src_rgb: Nx3 array of RGB values in [0, 1].
    :param opacity: Optional N array of opacity values.
    :param use_icp: Whether to run ICP refinement.
    :param pre_rotate: Rotate from Z-up to Y-up before alignment.
    :param k: Number of nearest neighbours to blend.
    :param min_opacity: Minimum opacity threshold to discard floaters.
    :param color_boost: Color post-processing mode.
    :param blend_with_existing: If True and mesh already has colors, blend new
                                colors onto visible/matching surfaces and keep
                                existing colors on occluded/opposite surfaces.
    :return: Modified trimesh.Trimesh with updated vertex_colors.
    """
    from scipy.spatial import cKDTree

    src = np.asarray(src_xyz, dtype=np.float64)
    rgb = np.asarray(src_rgb, dtype=np.float64)

    if opacity is not None:
        op = np.asarray(opacity, dtype=np.float64).reshape(-1)
        keep = op > min_opacity
        if keep.sum() >= max(100, int(0.05 * len(op))):
            src, rgb = src[keep], rgb[keep]

    if pre_rotate:
        src = src @ _SPLAT_TO_GLTF

    tgt = np.asarray(mesh.vertices, dtype=np.float64)
    if len(tgt) == 0:
        raise ValueError("Target mesh has no vertices")
    if len(src) == 0:
        raise ValueError("Source point cloud is empty")

    src = _coarse_align(src, tgt)
    src = _auto_orient(src, tgt)
    if use_icp:
        src = _icp_refine(src, tgt)

    k = max(1, min(int(k), len(src)))
    dist, idx = cKDTree(src).query(tgt, k=k)
    sn = None
    if k == 1:
        colors = rgb[idx]
    else:
        w = 1.0 / (dist + 1e-8)
        try:
            vn = np.asarray(mesh.vertex_normals, dtype=np.float64)
            sn = _estimate_normals(src)
            cons = np.clip(np.einsum("vi,vki->vk", vn, sn[idx]), 0.0, None)
            wn = w * cons
            bad = wn.sum(axis=1) <= 1e-12
            wn[bad] = w[bad]
            w = wn
        except Exception as exc:
            print(f"[colorize] Normal-aware weighting skipped: {exc}")

        w /= w.sum(axis=1, keepdims=True)
        colors = (rgb[idx] * w[..., None]).sum(axis=1)

    colors = _apply_boost(colors, color_boost)

    if blend_with_existing and hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors) == len(mesh.vertices):
        try:
            existing_rgb = np.asarray(mesh.visual.vertex_colors, dtype=np.float64)[:, :3] / 255.0
            vn = np.asarray(mesh.vertex_normals, dtype=np.float64)
            if sn is None:
                sn = _estimate_normals(src)
            nearest_idx = idx[:, 0] if k > 1 else idx
            sn_matched = sn[nearest_idx]
            alignment = np.einsum("vi,vi->v", vn, sn_matched)

            # Smooth weight based on normal alignment
            alpha = np.clip((alignment - 0.0) / 0.5, 0.0, 1.0)

            # Distance falloff relative to bounding box scale
            dist0 = dist[:, 0] if k > 1 else dist
            diag = _bbox_diag(tgt)
            d_scale = diag * 0.12
            dist_factor = np.clip(1.0 - (dist0 / max(d_scale, 1e-6)), 0.0, 1.0)
            alpha = alpha * dist_factor

            colors = existing_rgb * (1.0 - alpha[:, None]) + colors * alpha[:, None]
        except Exception as exc:
            print(f"[colorize] Normal-aware blending with existing colors failed: {exc}")

    rgba = np.hstack([(np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8),
                      np.full((len(mesh.vertices), 1), 255, dtype=np.uint8)])
    mesh.visual.vertex_colors = rgba
    return mesh
