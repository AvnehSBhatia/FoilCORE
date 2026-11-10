from __future__ import annotations

import math

import torch
import torch.nn as nn

D_POLAR = 5
D_GEOM = 20
D_MODEL = 32
N_GEOM_MEMORIES = 3


def sinusoidal_encoding_batch(positions: torch.Tensor, d_model: int, *, dtype: torch.dtype):
    """Batched sinusoidal positions; ``positions`` shaped (N,)."""
    dev = positions.device
    idx_half = torch.arange(0, d_model, 2, device=dev, dtype=dtype)
    div_term = torch.exp(idx_half * (-math.log(10000.0) / d_model))
    angles = positions.unsqueeze(-1) * div_term.unsqueeze(0)
    pe = torch.empty(positions.shape[0], d_model, device=dev, dtype=dtype)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    if d_model % 2 == 1:
        pe[:, -1] = 0
    return pe


def _masked_mean(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """feat (B, N, D), mask (B, N) bool → (B, D). Empty rows fall back to global mean of that sample."""
    mask_f = mask.unsqueeze(-1).to(dtype=feat.dtype)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    pooled = (feat * mask_f).sum(dim=1) / denom
    empty = mask.any(dim=1).logical_not()
    if empty.any():
        fallback = feat.mean(dim=1)
        pooled = torch.where(empty.unsqueeze(-1), fallback, pooled)
    return pooled


class GeometryTripleEncoder(nn.Module):
    """200×2 coords → (global, leading-edge, trailing-edge) latents, each ``d_model``.

    One shared point Linear(2, d); region-masked means. LE/TE split on chord-x.
    """

    def __init__(self, d_model: int = D_MODEL, le_xmax: float = 0.25, te_xmin: float = 0.75):
        super().__init__()
        self.d_model = int(d_model)
        self.le_xmax = float(le_xmax)
        self.te_xmin = float(te_xmin)
        self.point = nn.Linear(2, self.d_model)

    def forward(self, coords: torch.Tensor, path: nn.Module | None = None) -> torch.Tensor:
        h = self.point(coords)
        if path is not None:
            h = path(h)
        x = coords[..., 0]
        le = x <= self.le_xmax
        te = x >= self.te_xmin
        all_pts = torch.ones_like(le)
        z_g = _masked_mean(h, all_pts)
        z_le = _masked_mean(h, le)
        z_te = _masked_mean(h, te)
        return torch.stack([z_g, z_le, z_te], dim=1)
