"""Airfoil geometry descriptors from resampled perimeter coordinates (see ``AeroDataset``)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import N_GEOM_PATCHES, build_geom_patches, normalize_coords, resample_airfoil


def _upper_lower_from_patches(coords_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split 200×2 chord-normal points into upper/lower arcs using patch surface flags."""
    coords_norm = np.asarray(coords_norm, dtype=np.float64).reshape(200, 2)
    patches = build_geom_patches(coords_norm)  # (20, 20) = 10 (x,y) per patch row
    nh = N_GEOM_PATCHES // 2
    upper_flat = patches[:nh].reshape(-1, 2)
    lower_flat = patches[nh:].reshape(-1, 2)
    return upper_flat, lower_flat


def _unique_mean_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse duplicate x bins by averaging y (helps TE/LE sampling)."""
    if x.size == 0:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    ux, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    uy = np.zeros_like(ux)
    for i in range(ux.size):
        uy[i] = np.mean(y[inv == i])
    return ux, uy


def _interp_monotone(xq: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.size < 2:
        return np.full_like(xq, np.nan, dtype=float)
    return np.interp(xq, x, y, left=np.nan, right=np.nan)


def max_thickness_ratio_and_camber(
    coords_raw: list | np.ndarray,
    *,
    n_resample: int = 200,
) -> tuple[float, float]:
    """
    Estimated maximum thickness ratio t/c and peak |camber|/c on chord-normalised coords.

    Uses upper/lower arcs implied by ``geom_patch_surface_flags`` after the same
    resampling and chord normalization as training. Chord is rotated to a horizontal
    axis between the min-x and max-x perimeter points; thickness is (y_u − y_l),
    camber is the mean line offset from the chord line in the rotated frame.
    """
    c = resample_airfoil(np.asarray(coords_raw, dtype=np.float64), n_points=int(n_resample))
    c_norm, scale, _ = normalize_coords(c)
    if float(scale) <= 0:
        return float("nan"), float("nan")

    upper, lower = _upper_lower_from_patches(c_norm)

    all_pts = np.vstack([upper, lower])
    i_le = int(np.argmin(all_pts[:, 0]))
    i_te = int(np.argmax(all_pts[:, 0]))
    le = np.array(all_pts[i_le], dtype=np.float64)
    te = np.array(all_pts[i_te], dtype=np.float64)
    vx, vy = te[0] - le[0], te[1] - le[1]
    chord_len = math.hypot(vx, vy)
    if chord_len < 1e-9:
        return float("nan"), float("nan")

    def _rotate(p: np.ndarray) -> np.ndarray:
        rel = p - le
        c0, s0 = vx / chord_len, vy / chord_len
        xn = rel[:, 0] * c0 + rel[:, 1] * s0
        yn = -rel[:, 0] * s0 + rel[:, 1] * c0
        return np.stack([xn / chord_len, yn / chord_len], axis=1)

    uu = _rotate(upper)
    ll = _rotate(lower)
    xq = np.linspace(0.03, 0.97, 80)
    xu, yu = _unique_mean_x(uu[:, 0], uu[:, 1])
    xl, yl = _unique_mean_x(ll[:, 0], ll[:, 1])
    yu_i = _interp_monotone(xq, xu, yu)
    yl_i = _interp_monotone(xq, xl, yl)
    thick = yu_i - yl_i
    mean_line = 0.5 * (yu_i + yl_i)
    camber = mean_line
    t_over_c = float(np.nanmax(np.abs(thick)))
    camber_max = float(np.nanmax(np.abs(camber)))
    if not math.isfinite(t_over_c):
        t_over_c = float("nan")
    if not math.isfinite(camber_max):
        camber_max = float("nan")
    return t_over_c, camber_max


def tertile_labels(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return categorical 0/1/2 labels (thin/med/thick) given t/c values and two cut points."""
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(v)
    q1, q2 = np.nanpercentile(v[mask], [33.333, 66.666])
    lab = np.full(v.shape[0], -1, dtype=np.int8)
    sel = mask
    lab[sel & (v <= q1)] = 0
    lab[sel & (v > q1) & (v <= q2)] = 1
    lab[sel & (v > q2)] = 2
    return lab, float(q1), float(q2)


def load_coords_column(csv_path: Path | str, max_rows: int) -> list[np.ndarray]:
    """Read raw ``coords`` JSON from the first ``max_rows`` rows (no full AeroDataset)."""
    df = pd.read_csv(csv_path, nrows=int(max_rows))
    if "coords" not in df.columns:
        raise ValueError(f"Missing coords column in {csv_path}")
    out: list[np.ndarray] = []
    for raw in df["coords"].tolist():
        out.append(json.loads(raw))
    return out

