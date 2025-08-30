# src/utils/geometry.py
"""
Geometry helpers:
- Signed distance to a closed polygon (airfoil surface)
- Surface normals (approximate, outward-oriented)
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


def _point_segment_distance(px, py, ax, ay, bx, by):
    """
    Euclidean distance from point P to segment AB, plus closest point.
    """
    APx, APy = px - ax, py - ay
    ABx, ABy = bx - ax, by - ay
    denom = ABx * ABx + ABy * ABy
    if denom == 0.0:
        t = 0.0
    else:
        t = (APx * ABx + APy * ABy) / denom
        t = max(0.0, min(1.0, t))
    qx = ax + t * ABx
    qy = ay + t * ABy
    dx, dy = px - qx, py - qy
    return np.hypot(dx, dy), qx, qy


def _is_point_inside_polygon(px: float, py: float, poly: np.ndarray) -> bool:
    """
    Ray casting algorithm for point-in-polygon.
    poly: (N,2) closed or open (first==last not required).
    """
    x, y = px, py
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # Check if intersects edge
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def signed_distance_to_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Compute signed distance of arbitrary points to a closed polygon curve.
    Sign convention:
      - negative inside polygon
      - positive outside polygon
      - ~0 near boundary

    Parameters
    ----------
    points : (M,2)
    polygon: (N,2) closed or open (will be treated as closed)

    Returns
    -------
    sdf : (M,)
    """
    pts = np.asarray(points, dtype=np.float64)
    poly = np.asarray(polygon, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be (M,2)")
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("polygon must be (N,2)")

    M = pts.shape[0]
    N = poly.shape[0]
    # distances to all segments
    dists = np.empty(M, dtype=np.float64)
    for i in range(M):
        px, py = pts[i]
        mind = np.inf
        for j in range(N):
            ax, ay = poly[j]
            bx, by = poly[(j + 1) % N]
            dist, _, _ = _point_segment_distance(px, py, ax, ay, bx, by)
            if dist < mind:
                mind = dist
        dists[i] = mind

    # sign via point-in-polygon
    sign = np.ones(M, dtype=np.float64)
    for i in range(M):
        if _is_point_inside_polygon(pts[i, 0], pts[i, 1], poly):
            sign[i] = -1.0
    return sign * dists


def surface_normals(surface_pts: np.ndarray, outward: bool = True) -> np.ndarray:
    """
    Approximate unit normals along a closed polyline (airfoil surface).
    - Tangent via central finite difference on the closed loop
    - Normal = rotate tangent by +90°: ( -ty, tx )
    - If outward=True, we try to orient normals outward by checking polygon centroid.

    Returns
    -------
    normals : (N,2) unit vectors
    """
    P = np.asarray(surface_pts, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2 or len(P) < 3:
        raise ValueError("surface_pts must be (N,2) with N>=3")

    N = len(P)
    # close loop
    Pm1 = np.roll(P, 1, axis=0)
    Pp1 = np.roll(P, -1, axis=0)
    tangents = Pp1 - Pm1
    # normalize tangents
    tnorm = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12
    T = tangents / tnorm
    # rotate by +90°
    Nvec = np.stack([-T[:, 1], T[:, 0]], axis=1)

    if outward:
        # Determine polygon centroid and see if normals point outward:
        c = P.mean(axis=0)
        # Average dot between normal and (point -> centroid) should be negative if outward
        v_to_centroid = (c[None, :] - P)
        dots = (Nvec * v_to_centroid).sum(axis=1)
        if np.mean(dots) > 0:
            # normals point inward; flip
            Nvec = -Nvec

    # unit normalize
    nrm = np.linalg.norm(Nvec, axis=1, keepdims=True) + 1e-12
    return Nvec / nrm
