# src/utils/grids.py
"""
Grid helpers:
- Create uniform Cartesian grids
- (Optional) basic resampling utilities (nearest) with safe fallbacks
- Simple mask construction using SDF
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

from .geometry import signed_distance_to_polygon


def make_uniform_grid(
    xmin: float, xmax: float, ymin: float, ymax: float, nx: int, ny: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return X,Y meshgrid with shape (ny, nx) in 'xy' indexing.
    """
    xs = np.linspace(xmin, xmax, nx, dtype=np.float32)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return X, Y


def flatten_grid(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Stack X,Y into (M,2) points with M=H*W.
    """
    return np.stack([X.ravel(), Y.ravel()], axis=1)


def sdf_from_surface_on_grid(
    X: np.ndarray, Y: np.ndarray, surface_pts: np.ndarray
) -> np.ndarray:
    """
    Compute SDF on a grid given surface points of a closed polygon.
    """
    pts = flatten_grid(X, Y)
    sdf_flat = signed_distance_to_polygon(pts, surface_pts)
    return sdf_flat.reshape(Y.shape).astype(np.float32)


def mask_from_sdf(sdf: np.ndarray, inside_is_fluid: bool = True) -> np.ndarray:
    """
    Build a binary mask from SDF.
    inside_is_fluid=True -> fluid region inside the polygon; mask=1 inside
    """
    if inside_is_fluid:
        mask = (sdf <= 0).astype(np.float32)
    else:
        mask = (sdf > 0).astype(np.float32)
    return mask


def nearest_resample_to_grid(
    coords: np.ndarray, values: np.ndarray, X: np.ndarray, Y: np.ndarray
) -> np.ndarray:
    """
    Very simple nearest-neighbor resampling from scattered (coords, values) to grid (X,Y).

    Notes
    -----
    This function is O(N*M) in the worst case and meant as a safe fallback for
    moderate sizes. For large data, replace with a KD-tree based method
    (scikit-learn NearestNeighbors or SciPy cKDTree) once available.
    """
    P = coords.reshape(-1, 2).astype(np.float64)
    V = values.reshape(-1).astype(np.float64)
    G = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float64)

    # If shapes align already (structured data), just reshape fast path:
    if values.ndim == 2 and values.shape == X.shape:
        return values.astype(np.float32)

    # Fallback NN (brute force). Avoid for huge grids.
    out = np.empty(G.shape[0], dtype=np.float64)
    # Chunk to reduce peak memory
    chunk = max(1, 20000 // max(1, P.shape[0]))
    for i in range(0, G.shape[0], chunk):
        g = G[i : i + chunk]  # (Cg,2)
        # distances to all P: (Cg,N)
        d2 = (g[:, None, 0] - P[None, :, 0]) ** 2 + (g[:, None, 1] - P[None, :, 1]) ** 2
        idx = np.argmin(d2, axis=1)
        out[i : i + chunk] = V[idx]
    return out.reshape(X.shape).astype(np.float32)
