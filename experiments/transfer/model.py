"""Generic draft → corrector model for the transfer experiments.

Same shape as FoilCORE with the airfoil-specific parts removed:
  pointwise draft   MLP(x_t) → ŷ_t                       (no neighbours)
  corrector         tokens [x_t, ỹ_t] + hidden + sinusoid(t) → 2 residual-product-attention blocks
                    (FoilCORE's own corrector, causal or bidirectional mask) → residual on the draft
where ỹ_t is the model's own draft (draft-only protocols) or, under scheduled sampling, the
ground truth with probability 1-p_ar per position (the label-leaking protocol).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.baselines import build_backbone  # noqa: E402
from src.embeddings import sinusoidal_encoding_batch  # noqa: E402
from src.model import PolarCorrectorBlock  # noqa: E402


class DraftCorrector(nn.Module):
    """corrector = 'rpa' (FoilCORE's residual-product attention, default) | 'transformer' | 'mamba' | 'tsmixer'."""

    def __init__(self, n_in: int, d: int = 32, n_blocks: int = 2, dropout: float = 0.05, corrector: str = "rpa", t_max: int = 128):
        super().__init__()
        self.d = d
        self.corrector = corrector
        self.draft_mlp = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.draft_head = nn.Linear(d, 1)
        self.token_in = nn.Linear(n_in + 1, d)
        if corrector == "rpa":
            self.blocks = nn.ModuleList([PolarCorrectorBlock(d, dropout=dropout) for _ in range(n_blocks)])
            self.backbone = None
        else:
            # width chosen so the whole model stays near the RPA version's ~4k parameters
            bd = {"transformer": 10, "mamba": 16, "tsmixer": 8}[corrector]
            self.bb_in = nn.Linear(d, bd)
            self.backbone = build_backbone(corrector, bd, n_layers=n_blocks, t_max=t_max, d_inner=12, d_state=8)
            self.bb_out = nn.Linear(bd, d)
            self.blocks = nn.ModuleList()
        self.act = nn.GELU()
        self.out = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, y_gt: torch.Tensor | None = None, *, causal: bool, p_ar: float = 1.0, draft_noise: float = 0.0):
        """x: (B,T,n_in); y_gt: (B,T) used only when p_ar < 1 (scheduled sampling). Returns (final, draft): (B,T).
        draft_noise: std (z units) of Gaussian noise on the draft the corrector reads (not on the returned draft),
        a knob on E|y - ỹ| for the leak-fraction sweep."""
        b, t, _ = x.shape
        h0 = self.draft_mlp(x)
        draft = self.draft_head(h0).squeeze(-1)
        y_in = draft.detach()
        if draft_noise > 0:
            y_in = y_in + draft_noise * torch.randn_like(y_in)
        if y_gt is not None and p_ar < 1.0 - 1e-9:
            use_draft = torch.rand_like(draft) < p_ar
            y_in = torch.where(use_draft, y_in, y_gt)
        pos = torch.arange(t, device=x.device, dtype=x.dtype) / t  # normalised position, plays the role of α
        tok = self.token_in(torch.cat([x, y_in.unsqueeze(-1)], -1)) + h0
        tok = tok + sinusoidal_encoding_batch(pos * 20.0, self.d, dtype=tok.dtype).unsqueeze(0)
        pos_b = pos.unsqueeze(0).expand(b, t)
        if self.backbone is None:
            h = tok
            for blk in self.blocks:
                h = self.act(blk(h, None, None, pos_b, causal=causal))
        else:
            mask = torch.ones(b, t, dtype=torch.bool, device=x.device)
            h = self.bb_out(self.backbone(self.bb_in(tok), mask, causal=causal))
        final = draft + self.out(h).squeeze(-1)
        return final, draft
