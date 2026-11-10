"""Standard time-series backbones as drop-in replacements for FoilCORE's polar corrector.

Each baseline keeps FoilCORE-5k's geometry front-end and pointwise query (four memories,
selective scan, draft head) and replaces the residual-product-attention corrector + requery
+ residual head by a generic sequence model over the polar:

  transformer  pre-LN encoder (MHA + FFN); PatchTST with patch length 1
  tsmixer      TSMixer / TinyTimeMixer style: time-mixing MLP over a fixed T_max grid + feature MLP
  mamba        minimal selective-SSM (S6) block; forward scan in causal mode, shared-weight
               forward+backward scans in bidirectional mode

All three are trained with the same protocol (draft-only inputs, random causal/bidirectional
batches) and read the same token: Linear(5,d)[α,Re,Ma,Ĉl,Ĉd] + context + sinusoid(α).
The backbone width ``bb_d`` is chosen so that the whole model lands near 5k parameters.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- transformer
class _MHALayer(nn.Module):
    """Pre-LN encoder layer written out by hand (no nn.TransformerEncoderLayer fast path; MPS-safe)."""

    def __init__(self, d: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh = n_heads, d // n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad: torch.Tensor, causal: bool):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)  # (3,B,h,T,dh)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dh)                                  # (B,h,T,T)
        if causal:
            scores = scores + torch.triu(torch.full((t, t), float("-inf"), device=x.device, dtype=x.dtype), 1)
        scores = scores.masked_fill(pad[:, None, None, :], float("-inf"))
        att = torch.nan_to_num(F.softmax(scores, -1), nan=0.0)
        y = (self.drop(att) @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.ff(self.ln2(x)))


class TransformerBackbone(nn.Module):
    def __init__(self, d: int, n_layers: int = 2, n_heads: int = 2, ff_mult: int = 2, dropout: float = 0.05):
        super().__init__()
        self.layers = nn.ModuleList([_MHALayer(d, n_heads, ff_mult, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *, causal: bool) -> torch.Tensor:
        pad = ~mask
        h = x
        for layer in self.layers:
            h = layer(h, pad, causal)
        return self.norm(h) * mask.unsqueeze(-1).to(h.dtype)


# ----------------------------------------------------------------------------- tsmixer
class TSMixerBackbone(nn.Module):
    """TSMixer block: time-mixing MLP over positions (Linear T_max→T_max, lower-triangular when causal)
    then feature-mixing MLP. Positions are the polar index; the α value itself is in the token."""

    def __init__(self, d: int, t_max: int = 80, n_layers: int = 2, ff_mult: int = 2, dropout: float = 0.05):
        super().__init__()
        self.t_max = int(t_max)
        self.time_norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.time_w = nn.ParameterList([nn.Parameter(torch.randn(t_max, t_max) / math.sqrt(t_max)) for _ in range(n_layers)])
        self.time_b = nn.ParameterList([nn.Parameter(torch.zeros(t_max)) for _ in range(n_layers)])
        self.feat_norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.feat_mlp = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d)) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *, causal: bool) -> torch.Tensor:
        b, t, d = x.shape
        if t > self.t_max:
            raise ValueError(f"TSMixer built for T_max={self.t_max}, got T={t}")
        m = mask.unsqueeze(-1).to(x.dtype)
        h = x * m
        if t < self.t_max:
            h = F.pad(h, (0, 0, 0, self.t_max - t))
        for i in range(len(self.time_w)):
            w = self.time_w[i]
            if causal:
                w = torch.tril(w)  # output position i only reads positions ≤ i
            y = self.time_norm[i](h)                       # (B, Tmax, d)
            y = torch.einsum("ij,bjd->bid", w, y) + self.time_b[i][None, :, None]
            h = h + self.dropout(F.gelu(y))
            y = self.feat_mlp[i](self.feat_norm[i](h))
            h = h + self.dropout(y)
        h = h[:, :t] * m
        return self.norm(h)


# ----------------------------------------------------------------------------- mamba
def _affine_scan_time(a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """h_t = a_t * h_{t-1} + u_t along dim 1 (Hillis–Steele, log2(T) steps). a,u: (B,T,...)."""
    t = a.shape[1]
    h = u
    step = 1
    while step < t:
        a_prev = a[:, :-step]
        h_prev = h[:, :-step]
        h = torch.cat([h[:, :step], h[:, step:] + a[:, step:] * h_prev], dim=1)
        a = torch.cat([a[:, :step], a[:, step:] * a_prev], dim=1)
        step *= 2
    return h


class MambaBlock(nn.Module):
    """Minimal S6 (Mamba) block: in-proj → causal depthwise conv → SiLU → selective scan → gate → out-proj."""

    def __init__(self, d: int, d_inner: int = 24, d_state: int = 8, dt_rank: int = 4, conv_k: int = 4):
        super().__init__()
        self.d_inner, self.d_state, self.conv_k = int(d_inner), int(d_state), int(conv_k)
        self.in_proj = nn.Linear(d, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, conv_k, groups=d_inner, padding=conv_k - 1)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner)
        with torch.no_grad():
            self.dt_proj.bias.copy_(torch.log(torch.expm1(torch.rand(d_inner) * 0.09 + 0.01)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def _scan_dir(self, xc: torch.Tensor) -> torch.Tensor:
        # xc: (B,T,di) already conv'd + activated, in scan order
        dbc = self.x_proj(xc)
        dt, bmat, cmat = torch.split(dbc, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))                                # (B,T,di)
        a = -torch.exp(self.A_log)                                            # (di,ds)
        da = torch.exp(delta.unsqueeze(-1) * a)                               # (B,T,di,ds)
        du = (delta * xc).unsqueeze(-1) * bmat.unsqueeze(2)                    # (B,T,di,ds)
        h = _affine_scan_time(da, du)
        y = (h * cmat.unsqueeze(2)).sum(-1) + self.D * xc                    # (B,T,di)
        return y

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *, causal: bool) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(x.dtype)
        u = self.norm(x) * m
        xz = self.in_proj(u)
        xs, z = xz.chunk(2, dim=-1)
        t = xs.shape[1]
        xc = self.conv(xs.transpose(1, 2))[..., :t].transpose(1, 2)
        xc = F.silu(xc) * m
        y = self._scan_dir(xc)
        if not causal:
            # bidirectional: shared-weight backward scan over the reversed valid sequence
            # (padding is at the end and zeroed, so flipping the whole tensor and flipping back is exact
            #  only when all rows share a length; use per-row flip via index)
            lens = mask.sum(1)
            idx = torch.arange(t, device=x.device)[None, :]
            rev = (lens[:, None] - 1 - idx).clamp(min=0)
            rev = torch.where(idx < lens[:, None], rev, idx)
            xr = torch.gather(xc, 1, rev.unsqueeze(-1).expand_as(xc))
            yr = self._scan_dir(xr)
            yb = torch.gather(yr, 1, rev.unsqueeze(-1).expand_as(yr))
            y = 0.5 * (y + yb)
        y = y * F.silu(z)
        return x + self.out_proj(y) * m


class MambaBackbone(nn.Module):
    def __init__(self, d: int, n_layers: int = 2, d_inner: int = 24, d_state: int = 8, dt_rank: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([MambaBlock(d, d_inner, d_state, dt_rank) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *, causal: bool) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, mask, causal=causal)
        return self.norm(h)


# ----------------------------------------------------------------------------- factory
def build_backbone(kind: str, d: int, *, n_layers: int, t_max: int, d_inner: int, d_state: int) -> nn.Module:
    kind = str(kind)
    if kind == "transformer":
        return TransformerBackbone(d, n_layers=n_layers)
    if kind == "tsmixer":
        return TSMixerBackbone(d, t_max=t_max, n_layers=n_layers)
    if kind == "mamba":
        return MambaBackbone(d, n_layers=n_layers, d_inner=d_inner, d_state=d_state)
    raise ValueError(f"unknown seq_backbone={kind!r}")
