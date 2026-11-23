"""
splat_mesh.py - Gaussian Splat to Watertight Mesh Extraction
Converts 3D Gaussian Splats into clean, vertex-colored .glb meshes.
"""

import logging
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from plyfile import PlyData

try:
    from skimage.measure import marching_cubes
except ImportError:
    try:
        from skimage.measure import marching_cubes_lewiner as marching_cubes
    except ImportError:
        marching_cubes = None

logger = logging.getLogger("idol.splat_mesh")


def extract_mesh_from_gaussians_data(
    xyz: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    grid_resolution: int = 160,
    iso_threshold: float = 0.1,
    smooth_iterations: int = 5,
) -> trimesh.Trimesh:
    """
    Extract a watertight 3D mesh from Gaussian point data via density voxelization
    and Marching Cubes isosurface extraction.

    :param xyz: (N, 3) point coordinates
    :param colors: (N, 3) RGB colors in [0, 255] uint8 or [0, 1] float
    :param opacities: (N,) or (N, 1) opacity values in [0, 1]
    :param grid_resolution: voxel grid resolution per axis
    :param iso_threshold: relative density threshold for isosurface (0.05 - 0.2)
    :param smooth_iterations: number of Laplacian smoothing iterations
    :return: trimesh.Trimesh with vertex colors
    """
    if marching_cubes is None:
        raise ImportError("scikit-image is required for mesh extraction (pip install scikit-image)")

    xyz = np.asarray(xyz, dtype=np.float32)
    opacities = np.asarray(opacities, dtype=np.float32).ravel()

    # Filter out near-zero opacity points
    valid_mask = opacities > 0.05
    if np.sum(valid_mask) > 1000:
        xyz = xyz[valid_mask]
        opacities = opacities[valid_mask]
        if colors is not None:
            colors = colors[valid_mask]

    # Bounds & normalization
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-5] = 1.0

    # Expand bounds slightly to prevent boundary clipping
    padding = ranges * 0.05
    mins -= padding
    maxs += padding
    ranges = maxs - mins

    # Voxelize into density grid
    grid = np.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=np.float32)
    norm_coords = (xyz - mins) / ranges * (grid_resolution - 1)
    indices = np.clip(norm_coords.astype(np.int32), 0, grid_resolution - 1)

    np.add.at(grid, (indices[:, 0], indices[:, 1], indices[:, 2]), opacities)

    # Apply Gaussian blur for smooth surface extraction
    grid = gaussian_filter(grid, sigma=1.0)

    max_val = grid.max()
    if max_val <= 0:
        logger.warning("Empty density grid; returning dummy mesh.")
        return trimesh.creation.icosphere(radius=0.5)

    level = max_val * iso_threshold
    try:
        verts, faces, normals, _ = marching_cubes(grid, level=level)
    except Exception as e:
        logger.warning(f"Marching cubes failed with level={level}: {e}. Retrying with level={max_val * 0.05}")
        verts, faces, normals, _ = marching_cubes(grid, level=max_val * 0.05)

    # Transform vertices back to world coordinates
    verts = verts / (grid_resolution - 1) * ranges + mins

    # Assign vertex colors via KDTree nearest neighbor query
    vertex_colors = None
    if colors is not None and len(colors) > 0:
        tree = cKDTree(xyz)
        _, indices = tree.query(verts, k=1)
        sampled_colors = colors[indices]
        if sampled_colors.max() <= 1.0:
            sampled_colors = (sampled_colors * 255).astype(np.uint8)
        else:
            sampled_colors = sampled_colors.astype(np.uint8)

        if sampled_colors.shape[1] == 3:
            alpha = np.full((len(verts), 1), 255, dtype=np.uint8)
            vertex_colors = np.hstack([sampled_colors, alpha])
        else:
            vertex_colors = sampled_colors

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=normals,
        vertex_colors=vertex_colors,
        process=True,
    )

    # Keep only the largest connected component (removes floating artifacts)
    try:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            largest = max(components, key=lambda c: len(c.faces))
            if len(largest.faces) > len(mesh.faces) * 0.2:
                mesh = largest
    except Exception:
        pass

    # Laplacian smoothing
    if smooth_iterations > 0:
        try:
            trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iterations)
        except Exception:
            pass

    return mesh


def extract_mesh_from_ply(
    ply_path: str,
    output_glb_path: str,
    grid_resolution: int = 160,
    iso_threshold: float = 0.1,
) -> str:
    """
    Read a 3DGS .ply file and convert it to a .glb mesh file.
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]])

    # Opacity
    if "opacity" in vertex.data.dtype.names:
        raw_op = vertex["opacity"]
        # If in logit form (inverse sigmoid), activate
        if raw_op.min() < 0 or raw_op.max() > 1:
            opacities = 1.0 / (1.0 + np.exp(-raw_op))
        else:
            opacities = raw_op
    else:
        opacities = np.ones(len(xyz), dtype=np.float32)

    # Colors
    colors = None
    if all(k in vertex.data.dtype.names for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        sh_dc = np.column_stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]])
        # Degree 0 SH to RGB
        colors = np.clip(sh_dc * 0.2820947917 + 0.5, 0.0, 1.0)
    elif all(k in vertex.data.dtype.names for k in ("red", "green", "blue")):
        colors = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]])

    mesh = extract_mesh_from_gaussians_data(
        xyz=xyz,
        colors=colors,
        opacities=opacities,
        grid_resolution=grid_resolution,
        iso_threshold=iso_threshold,
    )

    mesh.export(output_glb_path, file_type="glb")
    return output_glb_path


def gaussian_object_to_glb(
    gaussian,
    output_glb_path: str,
    grid_resolution: int = 160,
) -> str:
    """
    Convert an Idol Gaussian instance directly into a .glb mesh.
    """
    xyz = gaussian.xyz.detach().cpu().numpy()
    # Apply standard y-up conversion matching save_ply: (x, -z, y)
    xyz_yup = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1)

    f_dc = gaussian.features_dc.detach().squeeze(1).cpu().numpy()
    colors = np.clip(f_dc * 0.2820947917 + 0.5, 0.0, 1.0)

    opacities = gaussian.opacity.detach().cpu().numpy().ravel()

    mesh = extract_mesh_from_gaussians_data(
        xyz=xyz_yup,
        colors=colors,
        opacities=opacities,
        grid_resolution=grid_resolution,
    )

    mesh.export(output_glb_path, file_type="glb")
    return output_glb_path
