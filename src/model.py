from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.embeddings import GeometryTripleEncoder, sinusoidal_encoding_batch

D_MODEL = 32
N_GEOM_MEMORIES = 3
D_COND = 3  # α, Re, Mach


@dataclass
class FoilCoreConfig:
    d_model: int = D_MODEL
    d_geom: int = D_MODEL
    model_type: str = "foilcore"  # foilcore | mlp_baseline | foilscan | foilcore2 | seqbaseline
    num_blocks: int = 2
    block_mode: str = "all_alt"  # all_alt = cheap RPA corrector; all_normal = MHA; mixed
    shared_qk_scalars: bool = True
    path_mode: str = "full"  # full | gelu | sin_linear | identity
    share_path: bool = True
    n_path_terms: int = 4
    outer_layers: int = 0  # unused; gated outer product dropped
    dropout: float = 0.05
    residual: bool = True
    use_causal_mask: bool = False  # polar corrector is bidirectional by default
    phys_residual: bool = False  # Cd = Cd0 + k·Cl² + δ with geometry-conditioned (Cd0, k)
    phys_residual_gated: bool = False  # per-point blend g·(Cd0+kCl²+δ) + (1-g)·learned Cd
    phys_base: bool = False  # Cl = 2π sin(α−α0)/√(1−Ma²) + learned residual
    invariant_memory: bool = False  # 5th memory: [r_LE, θ_TE, t_max, x_tmax, c_max, x_cmax]
    bl_features: bool = False  # scale-separation features [log10Re, 1/√Re, 1/Re^0.2, |α|/√r_LE]
    cl_mu: float = 0.0
    cl_sd: float = 1.0
    cd_mu: float = 0.0
    cd_sd: float = 1.0
    a_mu: float = 0.0
    a_sd: float = 1.0
    m_mu: float = 0.0
    m_sd: float = 1.0
    re_mu: float = 0.0
    re_sd: float = 1.0
    decoder_hidden: int = 0
    output_dims: int = 2
    mlp_hidden: int = 32
    n_heads: int = 2
    le_xmax: float = 0.25
    te_xmin: float = 0.75
    draft_loss_weight: float = 0.25
    n_geom_memories: int = N_GEOM_MEMORIES
    n_landmarks: int = 4
    contour_bidirectional: bool = True
    contour_d_cond: int = 3  # (x, y, 1-x); TE proximity is explicit
    # --- FoilSCAN v2 knobs (defaults reproduce variant 40) ---
    contour_log_a_init: str = "const"  # const (-0.5 everywhere) | spread (log-spaced horizons 3..150 pts)
    contour_mode: str = "ssm"  # ssm | pool (arch 3: FoilCORE region means, no contour scan)
    directional_landmarks: bool = False  # read fwd/bwd scan states separately at each landmark
    landmark_mean: bool = False  # prepend the mean contour state as an extra memory
    mem_log_a_init: float = -0.5  # LinearMambaMix log_A init; -1.5 flattens weights over 9 memories
    polar_mixer: str = "iir_clcd"  # iir_clcd (v40: 2-ch IIR on Cl/Cd) | iir_token (d-ch IIR + LN + PathNL)
    iir_tau_init: str = "const"  # const (softplus(0)) | spread (log-spaced 1°..11° in α)
    polar_in_clcd: bool = True  # iir_token: add Linear(2,d)(clcd) to the token
    polar_alpha_sinus: bool = True  # iir_token: add sinusoidal α encoding to the token
    # --- FoilCORE v2 (model_type="foilcore2"): 3-pool trunk, SSM polar blocks, stall router, error head ---
    err_head_hidden: int = 16
    err_head_mem_dim: int = 4  # per-memory reduction before the error MLP
    router_threshold: float = 0.5  # tokens with gate > threshold are attention keys
    use_stall_router: bool = True
    use_selective_ssm: bool = True
    use_local_kernel: bool = True
    use_err_head: bool = False  # foilcore: add ErrorHead ê to the polar token (polar_in 7→d)
    profile_stations: int = 0  # foilcore: camber/thickness sampled at N chord stations → 4th geometry memory
    moe_experts: int = 0  # foilcore: >0 → each corrector block is a router-mixed set of K RPA experts
    geom_pool: str = "region3"  # region3 (global/LE/TE hard means) | softbin (global + K learned Gaussian windows per surface)
    softbin_k: int = 4
    use_err_expert: bool = False  # foilcore: ê-gated residual expert on the corrector state → final
    err_expert_hidden: int = 16
    # --- standard sequence backbones (model_type="seqbaseline"): same front-end, generic corrector ---
    seq_backbone: str = "transformer"  # transformer | tsmixer | mamba
    bb_d: int = 16
    bb_layers: int = 2
    bb_t_max: int = 96  # tsmixer time-mixing grid (polars have T ≤ 81)
    bb_inner: int = 24  # mamba d_inner
    bb_state: int = 8   # mamba d_state


def remap_foilcore_state_dict(
    raw: dict[str, torch.Tensor],
    *,
    target_shared_qk_scalars: bool = True,
) -> dict[str, torch.Tensor]:
    """Drop dead decoder keys; align residual-product attention Q/K scalar layouts."""
    sd = {k: v for k, v in raw.items() if not k.startswith("geom_decoder.")}

    def _has_phys_shared_scalars() -> bool:
        return any(k.endswith(".phys_attn.s1") or k.endswith(".attn.s1") for k in sd)

    def _has_phys_qk_scalars() -> bool:
        return any(k.endswith(".phys_attn.s1_q") or k.endswith(".attn.s1_q") for k in sd)

    suffixes = (("phys_attn",), ("attn",))

    if target_shared_qk_scalars:
        if _has_phys_qk_scalars():
            for key in list(sd.keys()):
                for sub in ("phys_attn", "attn"):
                    tail = f".{sub}.s1_q"
                    if not key.endswith(tail):
                        continue
                    base = key[: -len("s1_q")]
                    kq = (base + "s1_q", base + "s2_q", base + "s3_q")
                    kk = (base + "s1_k", base + "s2_k", base + "s3_k")
                    if not all(k in sd for k in kq + kk):
                        continue
                    sd[base + "s1"] = 0.5 * (sd[kq[0]] + sd[kk[0]])
                    sd[base + "s2"] = 0.5 * (sd[kq[1]] + sd[kk[1]])
                    sd[base + "s3"] = 0.5 * (sd[kq[2]] + sd[kk[2]])
                    for k in kq + kk:
                        del sd[k]
        return sd

    if _has_phys_shared_scalars() and not _has_phys_qk_scalars():
        for key in list(sd.keys()):
            for sub in ("phys_attn", "attn"):
                tail = f".{sub}.s1"
                if not key.endswith(tail):
                    continue
                base = key[: -len("s1")]
                if base + "s1_q" in sd:
                    continue
                s2_key, s3_key = base + "s2", base + "s3"
                if s2_key not in sd or s3_key not in sd:
                    continue
                s1, s2, s3 = sd[key], sd[s2_key], sd[s3_key]
                sd[base + "s1_q"] = s1.clone()
                sd[base + "s2_q"] = s2.clone()
                sd[base + "s3_q"] = s3.clone()
                sd[base + "s1_k"] = s1.clone()
                sd[base + "s2_k"] = s2.clone()
                sd[base + "s3_k"] = s3.clone()
                del sd[key], sd[s2_key], sd[s3_key]
    _ = suffixes
    return sd


def config_from_checkpoint(ckpt: dict) -> FoilCoreConfig:
    cfg = FoilCoreConfig()
    model_meta = ckpt.get("variant_model")
    if isinstance(model_meta, dict):
        for k, v in model_meta.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class ResidualProductAttention(nn.Module):
    """Cheap attention: no Q/K projections, one V map. Default polar corrector."""

    def __init__(
        self,
        d_model=D_MODEL,
        *,
        dropout: float = 0.05,
        shared_qk_scalars: bool = True,
        use_causal_mask: bool = False,
    ):
        super().__init__()
        self.shared_qk_scalars = bool(shared_qk_scalars)
        self.use_causal_mask = bool(use_causal_mask)
        if self.shared_qk_scalars:
            self.s1 = nn.Parameter(torch.tensor(0.9))
            self.s2 = nn.Parameter(torch.tensor(0.1))
            self.s3 = nn.Parameter(torch.tensor(0.0))
        else:
            self.s1_q = nn.Parameter(torch.tensor(0.9))
            self.s2_q = nn.Parameter(torch.tensor(0.1))
            self.s3_q = nn.Parameter(torch.tensor(0.0))
            self.s1_k = nn.Parameter(torch.tensor(0.9))
            self.s2_k = nn.Parameter(torch.tensor(0.1))
            self.s3_k = nn.Parameter(torch.tensor(0.0))
        self.W_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)
        self.local_log_tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, z, key_padding_mask=None, alpha=None, causal=None):
        residual = torch.tanh(x - z)
        if self.shared_qk_scalars:
            Q = self.s1 * x + self.s2 * residual + self.s3
            K = Q
        else:
            Q = self.s1_q * x + self.s2_q * residual + self.s3_q
            K = self.s1_k * x + self.s2_k * residual + self.s3_k
        V = self.W_v(x)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        use_causal = self.use_causal_mask if causal is None else bool(causal)
        if use_causal:
            seq_len = scores.shape[-1]
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=scores.dtype) * float("-inf"),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0)
        if alpha is not None:
            da = alpha.unsqueeze(-1) - alpha.unsqueeze(-2)
            tau = F.softplus(self.local_log_tau) + 1e-3
            scores = scores - da.pow(2) / tau
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, :], float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        return torch.bmm(attn_weights, V)


class PathNonlinearity(nn.Module):
    """Per-channel map: linear + Fourier series of acos(b x) + Gaussian bump.

    φ(x) = a_lin x + Σ_{k=1}^{K} a_k cos(k · acos(clamp(b_k x))) + A exp(-½ ((x-μ)/σ)²) + m

    ``K = n_terms`` is a hyperparameter; a_k, b_k are learned per channel.
    """

    def __init__(self, num_features: int, n_terms: int = 4):
        super().__init__()
        n = int(num_features)
        k = int(n_terms)
        if n < 1:
            raise ValueError("num_features must be >= 1")
        if k < 0:
            raise ValueError("n_terms must be >= 0")
        self.num_features = n
        self.n_terms = k
        self.a_lin = nn.Parameter(torch.ones(n))
        self.m = nn.Parameter(torch.zeros(n))
        if k > 0:
            harmonic = torch.arange(1, k + 1, dtype=torch.float32)
            self.a = nn.Parameter((0.1 / harmonic).repeat(n, 1))
            self.b = nn.Parameter(torch.full((n, k), 0.5))
        else:
            self.register_parameter("a", None)
            self.register_parameter("b", None)
        self.gauss_amp = nn.Parameter(torch.full((n,), 0.01))
        self.gauss_mu = nn.Parameter(torch.zeros(n))
        self.gauss_log_sigma = nn.Parameter(torch.zeros(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.a_lin * x + self.m
        if self.n_terms > 0:
            u = torch.clamp(self.b * x.unsqueeze(-1), -1.0 + 1e-4, 1.0 - 1e-4)
            theta = torch.acos(u)
            k = torch.arange(1, self.n_terms + 1, device=x.device, dtype=x.dtype)
            y = y + (self.a * torch.cos(k * theta)).sum(dim=-1)
        sigma = F.softplus(self.gauss_log_sigma) + 1e-3
        y = y + self.gauss_amp * torch.exp(-0.5 * ((x - self.gauss_mu) / sigma).pow(2))
        return y


class SinLinearPath(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        n = int(num_features)
        self.c = nn.Parameter(torch.full((n,), 0.5))
        self.d = nn.Parameter(torch.full((n,), 1.0))
        self.f = nn.Parameter(torch.full((n,), 1.0))
        self.m = nn.Parameter(torch.zeros(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c * torch.sin(self.d * x).pow(3) + self.f * x + self.m


def affine_scan(a: torch.Tensor, u: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Inclusive scan ``h_t = a_t h_{t-1} + u_t`` with ``h_{-1}=0``.

    Hillis–Steele parallel prefix: ``ceil(log2 N)`` batched steps, no Python
    loop over the sequence. Multiplying ``A→0`` forgets the left segment
    (same as a sequential scan) without log-cumsum overflow.
    """
    if a.device.type == "cpu":
        from src import fused_scan as _fs

        if _fs.can_use(a, u, dim) and _fs.get_ext() is not None:
            return _fs.affine_scan_fused(a, u)
    A = a
    U = u
    n = A.size(dim)
    ndim = A.dim()
    idx = torch.arange(n, device=A.device)
    shape = [1] * ndim
    shape[dim] = n
    shift = 1
    while shift < n:
        keep = (idx >= shift).view(shape)
        A_l = torch.where(keep, torch.roll(A, shift, dims=dim), torch.ones_like(A))
        U_l = torch.where(keep, torch.roll(U, shift, dims=dim), torch.zeros_like(U))
        U = A * U_l + U
        A = A * A_l
        shift <<= 1
    return U


class LinearMambaMix(nn.Module):
    """Selective scan over geometry memories (global/LE/TE, or contour landmarks).

    Conditions (α, Re, Mach) produce per-channel Δ, B, C — no d×d QKV maps.
    Polar length T is fully parallel; the scan is only over the M memories.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_mem: int = N_GEOM_MEMORIES,
        d_cond: int = D_COND,
        log_a_init: float = -0.5,
    ):
        super().__init__()
        self.n_mem = int(n_mem)
        self.log_A = nn.Parameter(torch.full((d_model,), float(log_a_init)))
        self.W_delta = nn.Linear(d_cond, d_model)
        self.W_B = nn.Linear(d_cond, d_model)
        self.W_C = nn.Linear(d_cond, d_model)

    def forward(self, memories: torch.Tensor, cond: torch.Tensor, hidden: torch.Tensor | None = None):
        delta = F.softplus(self.W_delta(cond)) + 1e-4
        b_in = self.W_B(cond)
        c_out = self.W_C(cond)
        if hidden is not None:
            c_out = c_out + hidden
        decay = -torch.exp(self.log_A)
        dA = torch.exp(delta * decay)
        keep = 1.0 - dA
        u = keep.unsqueeze(2) * b_in.unsqueeze(2) * memories.unsqueeze(1)
        a = dA.unsqueeze(2).expand_as(u).contiguous()
        h = affine_scan(a, u, dim=2)
        return c_out * h[:, :, -1, :]


class ContourSSM(nn.Module):
    """Bidirectional selective scan along the airfoil contour.

    Δ, B, C come from local contour coords (x, y[, 1-x]), not flight condition.
    Params do not grow with the 200-point length. Fwd+rev are one batched scan.
    """

    def __init__(self, d_model: int, d_cond: int = 3, log_a_init: str = "const"):
        super().__init__()
        if log_a_init == "spread":
            # Per-step decay exp(-softplus(0)·A): horizons from ~3 to ~150 contour points
            # (one surface of the 200-point contour is ~100 points).
            init = torch.linspace(-4.5, -0.7, d_model)
        else:
            init = torch.full((d_model,), -0.5)
        self.log_A = nn.Parameter(init)
        self.W_delta = nn.Linear(d_cond, d_model)
        self.W_B = nn.Linear(d_cond, d_model)
        self.W_C = nn.Linear(d_cond, d_model)

    def _scan(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        delta = F.softplus(self.W_delta(cond)) + 1e-4
        b_in = self.W_B(cond)
        c_out = self.W_C(cond)
        decay = -torch.exp(self.log_A)
        dA = torch.exp(delta * decay)
        keep = 1.0 - dA
        h = affine_scan(dA, keep * b_in * feat, dim=1)
        return c_out * h

    def scan_both(self, feat: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward and backward states, each (B, N, D), from one batched scan."""
        n = feat.shape[0]
        h = self._scan(
            torch.cat((feat, feat.flip(1)), 0),
            torch.cat((cond, cond.flip(1)), 0),
        )
        return h[:n], h[n:].flip(1)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor, *, bidirectional: bool = True) -> torch.Tensor:
        if not bidirectional:
            return self._scan(feat, cond)
        h_f, h_b = self.scan_both(feat, cond)
        return 0.5 * (h_f + h_b)


def contour_condition(coords: torch.Tensor, d_cond: int) -> torch.Tensor:
    """Local contour features for Δ/B/C. ``d_cond=3`` adds ``1-x`` (TE side)."""
    x = coords[..., :1]
    y = coords[..., 1:2]
    if int(d_cond) <= 2:
        return torch.cat((x, y), dim=-1)
    return torch.cat((x, y, 1.0 - x), dim=-1)


def landmark_gather(h: torch.Tensor, coords: torch.Tensor, n_landmarks: int = 4) -> torch.Tensor:
    """(B, N, D) contour states → (B, L, D).

    Default L=4 is LE (min x), TE (max x), ymax, ymin. Global mean is a near-copy
    of LE/TE (cos ~0.94 in the 3.4k trunk) and is only prepended when L>=5.
    """
    x, y = coords[..., 0], coords[..., 1]
    b = torch.arange(h.shape[0], device=h.device)
    idx = torch.stack((x.argmin(1), x.argmax(1), y.argmax(1), y.argmin(1)), dim=1)
    pts = h[b.unsqueeze(1), idx]
    n_l = int(n_landmarks)
    if n_l <= 4:
        return pts[:, : max(n_l, 1)]
    mean = h.mean(dim=1, keepdim=True)
    return torch.cat((mean, pts), dim=1)


def landmark_gather_directional(
    h_f: torch.Tensor,
    h_b: torch.Tensor,
    coords: torch.Tensor,
    *,
    include_mean: bool = True,
) -> torch.Tensor:
    """Directional landmark reads → (B, 8 [+1], D).

    Contours are TE → upper → LE → lower → TE, so at the LE the forward state
    integrates the upper surface and the backward state the lower surface. The
    TE is read from the backward state at index 0 (upper surface seen from the
    TE) and the forward state at index N-1 (lower surface toward the TE); the
    other two TE states hold a single point and are skipped. ymax/ymin read both.
    Order: [mean?, ymax_f, ymax_b, ymin_f, ymin_b, TE_up, TE_lo, LE_f, LE_b].
    """
    x, y = coords[..., 0], coords[..., 1]
    b = torch.arange(h_f.shape[0], device=h_f.device)
    i_le, i_max, i_min = x.argmin(1), y.argmax(1), y.argmin(1)
    mems = [
        h_f[b, i_max],
        h_b[b, i_max],
        h_f[b, i_min],
        h_b[b, i_min],
        h_b[:, 0],
        h_f[:, -1],
        h_f[b, i_le],
        h_b[b, i_le],
    ]
    if include_mean:
        mems.insert(0, 0.5 * (h_f + h_b).mean(dim=1))
    return torch.stack(mems, dim=1)


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(y))


class LaplaceIIR(nn.Module):
    """Causal (or bidirectional) Laplace smoother on the Cl/Cd manifold.

    ``s_t = e^{-|Δα|/τ} s_{t-1} + (1 - e^{-|Δα|/τ}) x_t``, applied in sorted-α
    order. Bidirectional TF averages a forward and backward pass; AR is forward
    only. Same τ in both directions. Both directions are one batched scan.
    """

    def __init__(self, n_ch: int = 2, tau_init: str = "const"):
        super().__init__()
        if tau_init == "spread":
            # τ in normalized α (std ≈ 11°): 0.1..1.0 ≈ 1°..11°, log-spaced across channels.
            tau = torch.logspace(-1.0, 0.0, n_ch)
            self.log_tau = nn.Parameter(_inv_softplus(tau))
        else:
            self.log_tau = nn.Parameter(torch.zeros(n_ch))

    def _scan_sorted(self, x: torch.Tensor, alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tau = F.softplus(self.log_tau) + 1e-3
        da = torch.zeros_like(alpha)
        da[:, 1:] = (alpha[:, 1:] - alpha[:, :-1]).abs()
        decay = torch.exp(-da.unsqueeze(-1) / tau)
        a = decay.clone()
        u = (1.0 - decay) * x
        a[:, 0] = 1.0
        u[:, 0] = x[:, 0]
        h = affine_scan(a, u, dim=1)
        return torch.where(mask.unsqueeze(-1), h, x)

    def forward(self, x: torch.Tensor, alpha: torch.Tensor, mask: torch.Tensor, *, causal: bool) -> torch.Tensor:
        fill = torch.where(mask, alpha, torch.full_like(alpha, 1e6))
        order = fill.argsort(dim=1)
        inv = order.argsort(dim=1)
        idx3 = order.unsqueeze(-1).expand_as(x)
        xs, als, ms = x.gather(1, idx3), alpha.gather(1, order), mask.gather(1, order)
        if causal:
            mixed = self._scan_sorted(xs, als, ms)
        else:
            bsz = xs.shape[0]
            mixed = self._scan_sorted(
                torch.cat((xs, xs.flip(1)), 0),
                torch.cat((als, als.flip(1)), 0),
                torch.cat((ms, ms.flip(1)), 0),
            )
            mixed = 0.5 * (mixed[:bsz] + mixed[bsz:].flip(1))
        return mixed.gather(1, inv.unsqueeze(-1).expand_as(mixed))


class AlphaOrder:
    """Sort a padded polar by α once; blocks scan in sorted order and scatter back."""

    def __init__(self, alpha: torch.Tensor, mask: torch.Tensor):
        fill = torch.where(mask, alpha, torch.full_like(alpha, 1e6))
        self.order = fill.argsort(dim=1)
        self.inv = self.order.argsort(dim=1)
        self.alpha_s = alpha.gather(1, self.order)
        self.mask_s = mask.gather(1, self.order)
        da = torch.zeros_like(self.alpha_s)
        da[:, 1:] = (self.alpha_s[:, 1:] - self.alpha_s[:, :-1]).abs()
        self.da_s = da  # |Δα| to the previous sorted point (0 at the first)

    def sort(self, x: torch.Tensor) -> torch.Tensor:
        return x.gather(1, self.order.unsqueeze(-1).expand_as(x))

    def unsort(self, x: torch.Tensor) -> torch.Tensor:
        return x.gather(1, self.inv.unsqueeze(-1).expand_as(x))


def alpha_scan_sorted(
    xs: torch.Tensor,
    da_s: torch.Tensor,
    mask_s: torch.Tensor,
    rate: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """ZOH low-pass along sorted α: ``h_t = e^{-|Δα| r} h_{t-1} + (1 - e^{-|Δα| r}) x_t``.

    ``rate`` is (D,) or (B, T, D) — constant rate is a Laplace kernel, per-token
    rate is a selective (Mamba-style) diagonal SSM. Bidirectional = mean of the
    forward and backward passes (one batched scan). Padded rows keep ``x``.
    """
    bsz = xs.shape[0]
    rate_b = rate if rate.dim() == 3 else rate.view(1, 1, -1).expand_as(xs)

    def _run(x, da, m, r):
        decay = torch.exp(-da.unsqueeze(-1) * r)
        a = decay.clone()
        u = (1.0 - decay) * x
        a[:, 0] = 1.0
        u[:, 0] = x[:, 0]
        h = affine_scan(a, u, dim=1)
        return torch.where(m.unsqueeze(-1), h, x)

    if causal:
        return _run(xs, da_s, mask_s, rate_b)
    # Backward pass: reversing the sequence moves each |Δα| one slot; rebuild it.
    da_rev = torch.zeros_like(da_s)
    da_rev[:, 1:] = da_s.flip(1)[:, :-1]
    h = _run(
        torch.cat((xs, xs.flip(1)), 0),
        torch.cat((da_s, da_rev), 0),
        torch.cat((mask_s, mask_s.flip(1)), 0),
        torch.cat((rate_b, rate_b.flip(1)), 0),
    )
    return 0.5 * (h[:bsz] + h[bsz:].flip(1))


class SelectiveAlphaSSM(nn.Module):
    """Mamba-lite over α: per-channel decay rate and readout from the polar features."""

    def __init__(self, d_model: int, d_feat: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.linspace(-2.0, 1.0, d_model))
        self.W_rate = nn.Linear(d_feat, d_model)
        self.W_C = nn.Linear(d_feat, d_model)

    def forward(self, xs: torch.Tensor, feat_s: torch.Tensor, ao: AlphaOrder, *, causal: bool) -> torch.Tensor:
        rate = (F.softplus(self.W_rate(feat_s)) + 1e-3) * torch.exp(self.log_A)
        h = alpha_scan_sorted(xs, ao.da_s, ao.mask_s, rate, causal=causal)
        return self.W_C(feat_s) * h


class StallRouterAttention(nn.Module):
    """Sparse attention: a router gates tokens; only routed tokens serve as keys.

    Q = K = s·x with an α-locality penalty, V = x (the autopsy found ``W_v`` ≈ I).
    Keys are hard-thresholded on the gate; the output is scaled by the soft gate,
    so the router learns where mixing pays (near stall) and stays cheap elsewhere.
    """

    def __init__(self, d_model: int, threshold: float = 0.5):
        super().__init__()
        self.router = nn.Linear(d_model, 1)
        self.s1 = nn.Parameter(torch.tensor(0.9))
        self.local_log_tau = nn.Parameter(torch.tensor(0.0))
        self.scale = math.sqrt(d_model)
        self.threshold = float(threshold)

    def forward(self, x: torch.Tensor, alpha: torch.Tensor, mask: torch.Tensor, *, causal: bool):
        gate = torch.sigmoid(self.router(x)).squeeze(-1)  # (B, T)
        keys_on = mask & (gate > self.threshold)
        t = x.shape[1]
        eye = torch.eye(t, device=x.device, dtype=torch.bool)
        # Self is always a key so no softmax row is empty (empty rows gave NaN gradients).
        allowed = keys_on[:, None, :] | eye[None]
        q = self.s1 * x
        scores = torch.bmm(q, q.transpose(1, 2)) / self.scale
        da = alpha.unsqueeze(-1) - alpha.unsqueeze(-2)
        scores = scores - da.pow(2) / (F.softplus(self.local_log_tau) + 1e-3)
        if causal:
            allowed = allowed & torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))[None]
        scores = scores.masked_fill(~allowed, -1e4)
        w = F.softmax(scores, dim=-1)
        out = torch.bmm(w, x)
        return gate.unsqueeze(-1) * out, gate


class PolarSSMBlock(nn.Module):
    """Local Laplace kernel + selective α-SSM + stall-routed sparse attention → LN.

    All three paths run in O(T) state per polar except the sparse attention,
    which only touches routed tokens. Caller applies PathNL after the block.
    """

    def __init__(
        self,
        d_model: int,
        d_feat: int,
        *,
        use_local_kernel: bool = True,
        use_selective_ssm: bool = True,
        use_stall_router: bool = True,
        router_threshold: float = 0.5,
    ):
        super().__init__()
        self.local = LaplaceIIR(n_ch=d_model, tau_init="spread") if use_local_kernel else None
        self.ssm = SelectiveAlphaSSM(d_model, d_feat) if use_selective_ssm else None
        self.attn = StallRouterAttention(d_model, threshold=router_threshold) if use_stall_router else None
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, feat, alpha, mask, ao: AlphaOrder, *, causal: bool):
        y = x
        gate = None
        paths = {}
        scale = getattr(self, "path_scale", None)  # analysis knob: {"local"|"ssm"|"attn": float}
        if self.local is not None or self.ssm is not None:
            xs = ao.sort(x)
            if self.local is not None:
                rate = 1.0 / (F.softplus(self.local.log_tau) + 1e-3)
                out = ao.unsort(alpha_scan_sorted(xs, ao.da_s, ao.mask_s, rate, causal=causal))
                paths["local"] = out
            if self.ssm is not None:
                paths["ssm"] = ao.unsort(self.ssm(xs, ao.sort(feat), ao, causal=causal))
        if self.attn is not None:
            paths["attn"], gate = self.attn(x, alpha, mask, causal=causal)
        for k, v in paths.items():
            s = 1.0 if scale is None else float(scale.get(k, 1.0))
            y = y + s * v
        if getattr(self, "debug", False):
            self.last = {"in": x, "paths": paths, "gate": gate, "out": y}
        return self.norm(y), gate


class SoftBinPool(nn.Module):
    """Global mean + K learned Gaussian chordwise windows on each surface → (B, 1+2K, D).

    Replaces the three hard region means: window centres/widths are learned (2·2K params), so
    the encoder can localize (small σ) wherever the polar is shape-sensitive. Point features are
    the same PathNL(Linear(2,d)) as before; surfaces split at argmin x (contour TE→upper→LE→lower→TE).
    """

    def __init__(self, k: int = 4):
        super().__init__()
        t = (torch.arange(k, dtype=torch.float32) + 0.5) / k
        mu = 0.5 * (1.0 - torch.cos(math.pi * t))  # cosine-spaced centres, denser at LE/TE
        self.mu = nn.Parameter(torch.cat((mu, mu)))
        self.log_sigma = nn.Parameter(torch.full((2 * k,), math.log(0.6 / k)))
        self.k = int(k)

    def forward(self, h: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        x = coords[..., 0]
        idx = torch.arange(x.shape[1], device=x.device)[None, :]
        i_le = x.argmin(dim=1, keepdim=True)
        side = torch.cat((idx <= i_le, idx >= i_le), dim=0)  # (2B, N) upper rows then lower rows
        xx = torch.cat((x, x), dim=0)
        sigma = F.softplus(self.log_sigma) + 1e-3
        mu_u, mu_l = self.mu[: self.k], self.mu[self.k :]
        sig_u, sig_l = sigma[: self.k], sigma[self.k :]
        b = x.shape[0]
        w_u = torch.exp(-0.5 * ((x.unsqueeze(-1) - mu_u) / sig_u) ** 2) * side[:b].unsqueeze(-1)
        w_l = torch.exp(-0.5 * ((x.unsqueeze(-1) - mu_l) / sig_l) ** 2) * side[b:].unsqueeze(-1)
        w = torch.cat((w_u, w_l), dim=-1)  # (B, N, 2K)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        bins = torch.einsum("bnk,bnd->bkd", w, h)
        return torch.cat((h.mean(dim=1, keepdim=True), bins), dim=1)


def profile_features(coords: torch.Tensor, n_stations: int) -> torch.Tensor:
    """Camber and thickness at ``n_stations`` chord stations → (B, 2N). No parameters.

    Contours are TE → upper → LE → lower → TE (chord-normalized). Upper/lower halves are
    split at argmin x and each is linearly interpolated at cosine-spaced x stations.
    """
    x, y = coords[..., 0], coords[..., 1]
    bsz, n = x.shape
    i_le = x.argmin(dim=1)
    idx = torch.arange(n, device=x.device)[None, :]
    up = idx <= i_le[:, None]
    lo = idx >= i_le[:, None]
    t = torch.arange(1, n_stations + 1, device=x.device, dtype=x.dtype) / (n_stations + 1)
    xs = 0.5 * (1.0 - torch.cos(math.pi * t))  # cosine spacing, denser near LE/TE

    def interp(mask):
        # push the other half to +inf so it sorts last, then linear-interpolate y(x)
        xm = torch.where(mask, x, torch.full_like(x, float("inf")))
        order = xm.argsort(dim=1)
        xs_sorted = xm.gather(1, order)
        ys_sorted = y.gather(1, order)
        cnt = mask.sum(dim=1, keepdim=True)
        pos = torch.searchsorted(xs_sorted, xs.unsqueeze(0).expand(bsz, -1).contiguous())
        pos = pos.clamp(1, n - 1)
        pos = torch.minimum(pos, (cnt - 1).clamp_min(1))
        x0, x1 = xs_sorted.gather(1, pos - 1), xs_sorted.gather(1, pos)
        y0, y1 = ys_sorted.gather(1, pos - 1), ys_sorted.gather(1, pos)
        w = ((xs.unsqueeze(0) - x0) / (x1 - x0).clamp_min(1e-6)).clamp(0.0, 1.0)
        return y0 + w * (y1 - y0)

    yu, yl = interp(up), interp(lo)
    camber = 0.5 * (yu + yl)
    thick = yu - yl
    return torch.cat((camber, thick), dim=-1)


def le_radius(coords: torch.Tensor) -> torch.Tensor:
    """Leading-edge radius from a y² = 2 r x parabola fit near the nose → (B, 1)."""
    x, y = coords[..., 0], coords[..., 1]
    nose = (x > 1e-3) & (x < 0.04)
    x_n = torch.where(nose, x, torch.zeros_like(x))
    y_n = torch.where(nose, y, torch.zeros_like(x))
    den = x_n.sum(1).clamp_min(1e-4)
    return (0.5 * (y_n * y_n).sum(1) / den).clamp(1e-4, 1.0).unsqueeze(-1)


def geom_invariants(coords: torch.Tensor, n_stations: int = 40) -> torch.Tensor:
    """Zero-parameter analytic airfoil invariants → (B, 6).

    [r_LE (log-scaled), θ_TE (rad), t_max, x_tmax, c_max, x_cmax]
    r_LE = leading-edge radius from a y² = 2 r x parabola fit near the nose;
    θ_TE = trailing-edge wedge angle; the rest come from the camber/thickness profile.
    """
    x, y = coords[..., 0], coords[..., 1]
    bsz = x.shape[0]

    r_le = le_radius(coords).squeeze(-1)

    prof = profile_features(coords, n_stations)
    camber, thick = prof[..., :n_stations], prof[..., n_stations:]
    t = torch.arange(1, n_stations + 1, device=x.device, dtype=x.dtype) / (n_stations + 1)
    xs = 0.5 * (1.0 - torch.cos(math.pi * t))
    t_max, i_t = thick.max(dim=1)
    x_tmax = xs[i_t]
    c_abs = camber.abs()
    c_max, i_c = c_abs.max(dim=1)
    x_cmax = xs[i_c]

    yu = camber + 0.5 * thick
    yl = camber - 0.5 * thick
    dx_te = xs[-1] - xs[-2]
    su = (yu[:, -1] - yu[:, -2]) / dx_te
    sl = (yl[:, -1] - yl[:, -2]) / dx_te
    theta_te = (torch.atan(su) - torch.atan(sl)).abs()

    return torch.stack(
        (
            torch.log1p(100.0 * r_le) / math.log(31.0),
            theta_te / 1.0,
            t_max / 0.25,
            x_tmax,
            c_max / 0.15,
            x_cmax,
        ),
        dim=-1,
    )


def thin_airfoil_alpha0(coords: torch.Tensor, n_stations: int = 40) -> torch.Tensor:
    """Zero-lift angle α0 = −(1/π)∫ (dzc/dθ)(cosθ − 1) dθ from the camber line → (B, 1)."""
    prof = profile_features(coords, n_stations)
    camber = prof[..., :n_stations]
    t = torch.arange(1, n_stations + 1, device=coords.device, dtype=coords.dtype) / (n_stations + 1)
    xs = 0.5 * (1.0 - torch.cos(math.pi * t))
    theta = torch.acos((1.0 - 2.0 * xs).clamp(-1.0, 1.0))
    th_mid = 0.5 * (theta[1:] + theta[:-1])
    dtheta = (theta[1:] - theta[:-1]).clamp_min(1e-4)
    dz_dx = (camber[:, 1:] - camber[:, :-1]) / (xs[1:] - xs[:-1]).clamp_min(1e-4)
    dz_dtheta = dz_dx * 0.5 * torch.sin(th_mid)  # dx/dθ = sinθ/2
    integrand = dz_dtheta * (torch.cos(th_mid) - 1.0)
    alpha0 = -(1.0 / math.pi) * (integrand * dtheta).sum(dim=1)
    return alpha0.unsqueeze(-1)


class ErrorExpert(nn.Module):
    """ê-gated residual on the final prediction: fires where the error head expects a miss."""

    def __init__(self, d_model: int, out_dims: int = 2, hidden: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model + out_dims, hidden), nn.GELU(), nn.Linear(hidden, out_dims))
        self.gate = nn.Linear(out_dims, 1)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, h: torch.Tensor, err_pred: torch.Tensor):
        g = torch.sigmoid(self.gate(err_pred))
        return g * self.mlp(torch.cat((h, err_pred), dim=-1)), g.squeeze(-1)


class ErrorHead(nn.Module):
    """Predicts |draft − truth| (z-units) for Cl, Cd from the 3 geometry memories + 5-d prediction."""

    def __init__(self, d_model: int, n_mem: int = 3, mem_dim: int = 4, hidden: int = 16, out_dims: int = 2):
        super().__init__()
        self.reduce = nn.Linear(d_model, mem_dim)
        self.mlp = nn.Sequential(
            nn.Linear(n_mem * mem_dim + 3 + out_dims, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dims),
        )

    def forward(self, memories: torch.Tensor, cond: torch.Tensor, clcd: torch.Tensor) -> torch.Tensor:
        z = self.reduce(memories).flatten(1)  # (B, n_mem*mem_dim)
        z = z.unsqueeze(1).expand(-1, cond.shape[1], -1)
        return F.softplus(self.mlp(torch.cat((z, cond, clcd), dim=-1)))


class IIRPolarBlock(nn.Module):
    """α-mixer with O(1) state: per-channel Laplace IIR on the d-dim token + LayerNorm.

    Replaces ``PolarCorrectorBlock`` (RPA with a d×d ``W_v``): the autopsy showed
    the attention was a Laplace kernel on α, so the kernel is made explicit with one
    learned τ per channel. PathNL is applied by the caller (shared with geometry).
    """

    def __init__(self, d_model: int, *, tau_init: str = "spread"):
        super().__init__()
        self.iir = LaplaceIIR(n_ch=d_model, tau_init=tau_init)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, alpha: torch.Tensor, mask: torch.Tensor, *, causal: bool) -> torch.Tensor:
        return self.norm(x + self.iir(x, alpha, mask, causal=causal))


class PolarCorrectorBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        dropout: float = 0.05,
        residual: bool = True,
        shared_qk_scalars: bool = True,
        use_causal_mask: bool = False,
    ):
        super().__init__()
        self.attn = ResidualProductAttention(
            d_model,
            dropout=dropout,
            shared_qk_scalars=shared_qk_scalars,
            use_causal_mask=use_causal_mask,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.residual = bool(residual)

    def forward(self, x, causal_mask=None, key_padding_mask=None, alpha=None, causal=None):
        _ = causal_mask
        y = self.attn(x, x, key_padding_mask=key_padding_mask, alpha=alpha, causal=causal)
        out = x + self.dropout(y) if self.residual else self.dropout(y)
        return self.norm(out)


class MoEPolarCorrectorBlock(nn.Module):
    """K residual-product-attention experts mixed per token by a softmax router.

    Each expert has its own W_v / Q-K scalars / α-locality τ, so different regions of the polar
    (linear range, stall, low-Re bucket) can use different learned correctors. Router input is
    the token itself (α, Re, Mach, draft Cl/Cd and geometry context are all in it).
    """

    def __init__(self, d_model: int, n_experts: int, *, dropout: float = 0.05, residual: bool = True,
                 shared_qk_scalars: bool = True, use_causal_mask: bool = False):
        super().__init__()
        self.experts = nn.ModuleList(
            [ResidualProductAttention(d_model, dropout=dropout, shared_qk_scalars=shared_qk_scalars,
                                      use_causal_mask=use_causal_mask) for _ in range(int(n_experts))]
        )
        self.router = nn.Linear(d_model, int(n_experts))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.residual = bool(residual)
        self.last_probs = None

    def forward(self, x, causal_mask=None, key_padding_mask=None, alpha=None, causal=None):
        _ = causal_mask
        probs = torch.softmax(self.router(x), dim=-1)  # (B, T, K)
        self.last_probs = probs
        y = 0.0
        for k, ex in enumerate(self.experts):
            y = y + probs[..., k : k + 1] * ex(x, x, key_padding_mask=key_padding_mask, alpha=alpha, causal=causal)
        out = x + self.dropout(y) if self.residual else self.dropout(y)
        return self.norm(out)


class NormalAttentionBlock(nn.Module):
    def __init__(self, d_model: int, *, n_heads: int = 2, dropout: float = 0.05, residual: bool = True):
        super().__init__()
        heads = int(n_heads)
        if d_model % heads != 0:
            heads = 1
        self.attn = nn.MultiheadAttention(d_model, num_heads=heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.residual = bool(residual)

    def forward(self, x, causal_mask=None, key_padding_mask=None, alpha=None):
        _ = alpha
        y, _ = self.attn(x, x, x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        out = x + self.dropout(y) if self.residual else self.dropout(y)
        return self.norm(out)


class ResidualProductAttentionBlock(nn.Module):
    """Legacy wrapper kept so older ablation names still construct."""

    def __init__(
        self,
        d_model: int,
        *,
        dropout: float = 0.05,
        residual: bool = True,
        shared_qk_scalars: bool = True,
        use_causal_mask: bool = False,
    ):
        super().__init__()
        self.block = PolarCorrectorBlock(
            d_model,
            dropout=dropout,
            residual=residual,
            shared_qk_scalars=shared_qk_scalars,
            use_causal_mask=use_causal_mask,
        )

    def forward(self, x, causal_mask=None, key_padding_mask=None, alpha=None, causal=None):
        return self.block(x, causal_mask, key_padding_mask, alpha, causal=causal)


def _build_path_module(mode: str, d_model: int, n_terms: int = 4) -> nn.Module:
    if mode == "full":
        return PathNonlinearity(d_model, n_terms=n_terms)
    if mode == "gelu":
        return nn.GELU()
    if mode == "sin_linear":
        return SinLinearPath(d_model)
    if mode == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown path_mode={mode!r}")


def _block_kinds(num_blocks: int, block_mode: str) -> list[str]:
    bm = "all_alt" if block_mode == "all_physics" else block_mode
    if bm == "all_normal":
        return ["normal"] * num_blocks
    if bm in ("all_alt", "corrector"):
        return ["alt"] * num_blocks
    base = ["normal", "alt", "alt", "alt", "normal", "alt", "alt", "alt"]
    reps = (num_blocks + len(base) - 1) // len(base)
    return (base * reps)[:num_blocks]


class FoilCORE(nn.Module):
    """Geom memories (3×32) → independent condition queries → polar corrector → requery.

    Each (α, Re, Mach) reads the airfoil through a linear/Mamba mix over
    (global, leading-edge, trailing-edge) latents. A small transformer then
    corrects the polar using neighboring α, and a second mix requeries geometry.
    """

    def __init__(self, d_model=D_MODEL, *, config: FoilCoreConfig | None = None):
        super().__init__()
        cfg = config if config is not None else FoilCoreConfig(d_model=d_model)
        self.config = cfg
        self.d_model = int(cfg.d_model)
        d = self.d_model
        n_terms = int(cfg.n_path_terms)

        if cfg.share_path:
            self.path = _build_path_module(cfg.path_mode, d, n_terms)
            self.path_steps = nn.ModuleList()
        else:
            self.path = _build_path_module(cfg.path_mode, d, n_terms)
            self.path_steps = nn.ModuleList(
                [_build_path_module(cfg.path_mode, d, n_terms) for _ in range(int(cfg.num_blocks))]
            )

        self.geom_encoder = GeometryTripleEncoder(
            d_model=d,
            le_xmax=float(cfg.le_xmax),
            te_xmin=float(cfg.te_xmin),
        )
        if cfg.model_type == "foilscan":
            n_mem = self._scan_n_memories(cfg)
        else:
            n_mem = self._pool_n_memories(cfg) + (1 if int(cfg.profile_stations) > 0 else 0) + (1 if cfg.invariant_memory else 0)
        self.mem_mix = LinearMambaMix(d, n_mem=n_mem, d_cond=D_COND, log_a_init=float(cfg.mem_log_a_init))
        self.head = nn.Linear(d, int(cfg.output_dims))
        self.trunk = None
        self.outers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.contour_ssm = None
        self.polar_iir = None
        self.requery_lift = None
        self.inv_in = nn.Linear(6, d) if cfg.invariant_memory else None
        self.cd0_head = nn.Linear(d, 1) if cfg.phys_residual else None
        self.k_head = nn.Linear(d, 1) if cfg.phys_residual else None
        self.gate_head = nn.Linear(d, 1) if cfg.phys_residual_gated else None
        if cfg.phys_base:
            self.base_scale = nn.Parameter(torch.tensor(1.0))
            self.alpha_stall_base = nn.Parameter(torch.tensor(-1.2))  # softplus ≈ 14°
            self.w_r_le = nn.Parameter(torch.tensor(-1.0))  # softplus ≈ 0.31
        if cfg.phys_residual:
            # Physical-scale priors: softplus(-4.8)≈0.008 (profile Cd0), softplus(-3.5)≈0.03 (k).
            # Zero init would give softplus(0)=0.693 → ~70× Cd offset at epoch 0 (observed blowup).
            nn.init.constant_(self.cd0_head.bias, -4.8)
            nn.init.constant_(self.k_head.bias, -3.5)
            nn.init.zeros_(self.cd0_head.weight)
            nn.init.zeros_(self.k_head.weight)
        if cfg.phys_residual_gated:
            nn.init.constant_(self.gate_head.bias, -3.0)  # g≈0.05 at init: start ≈ learned Cd

        if cfg.model_type == "foilscan":
            self.polar_in = None
            self.agent = nn.ModuleList()
            if cfg.contour_mode == "ssm":
                self.contour_ssm = ContourSSM(
                    d, d_cond=int(cfg.contour_d_cond), log_a_init=str(cfg.contour_log_a_init)
                )
            if cfg.polar_mixer == "iir_token":
                if cfg.polar_in_clcd:
                    self.polar_in = nn.Linear(int(cfg.output_dims), d)
                self.agent = nn.ModuleList(
                    [IIRPolarBlock(d, tau_init=str(cfg.iir_tau_init)) for _ in range(max(1, int(cfg.num_blocks)))]
                )
            else:
                self.polar_iir = LaplaceIIR(n_ch=int(cfg.output_dims), tau_init=str(cfg.iir_tau_init))
                self.requery_lift = nn.Linear(int(cfg.output_dims), d)
        elif cfg.model_type == "mlp_baseline":
            self.polar_in = nn.Linear(5, d)
            self.agent = nn.ModuleList()
        elif cfg.model_type == "foilcore2":
            n_out = int(cfg.output_dims)
            d_feat = 3 + 2 * n_out  # [α, Re, Mach, Cl, Cd, ê_Cl, ê_Cd]
            self.err_head = ErrorHead(
                d, n_mem=int(cfg.n_geom_memories), mem_dim=int(cfg.err_head_mem_dim),
                hidden=int(cfg.err_head_hidden), out_dims=n_out,
            )
            self.polar_in = nn.Linear(d_feat, d)
            self.agent = nn.ModuleList(
                [
                    PolarSSMBlock(
                        d, d_feat,
                        use_local_kernel=bool(cfg.use_local_kernel),
                        use_selective_ssm=bool(cfg.use_selective_ssm),
                        use_stall_router=bool(cfg.use_stall_router),
                        router_threshold=float(cfg.router_threshold),
                    )
                    for _ in range(int(cfg.num_blocks))
                ]
            )
        elif cfg.model_type == "seqbaseline":
            from src.baselines import build_backbone
            n_out = int(cfg.output_dims)
            self.err_head = None
            self.profile_in = nn.Linear(2 * int(cfg.profile_stations), d) if int(cfg.profile_stations) > 0 else None
            self.polar_in = nn.Linear(3 + n_out, d)
            self.bb_in = nn.Linear(d, int(cfg.bb_d)) if int(cfg.bb_d) != d else nn.Identity()
            self.backbone = build_backbone(
                cfg.seq_backbone, int(cfg.bb_d), n_layers=int(cfg.bb_layers), t_max=int(cfg.bb_t_max),
                d_inner=int(cfg.bb_inner), d_state=int(cfg.bb_state),
            )
            self.bb_head = nn.Linear(int(cfg.bb_d), n_out)
            self.agent = nn.ModuleList()
        else:
            n_out = int(cfg.output_dims)
            self.err_head = None
            if cfg.use_err_head:
                self.err_head = ErrorHead(
                    d, n_mem=self._pool_n_memories(cfg) + (1 if int(cfg.profile_stations) > 0 else 0) + (1 if cfg.invariant_memory else 0),
                    mem_dim=int(cfg.err_head_mem_dim), hidden=int(cfg.err_head_hidden), out_dims=n_out,
                )
            self.polar_in = nn.Linear(3 + n_out + (n_out if cfg.use_err_head else 0) + (4 if cfg.bl_features else 0), d)
            self.softbin = SoftBinPool(int(cfg.softbin_k)) if cfg.geom_pool == "softbin" else None
            self.profile_in = nn.Linear(2 * int(cfg.profile_stations), d) if int(cfg.profile_stations) > 0 else None
            self.err_expert = (
                ErrorExpert(d, out_dims=n_out, hidden=int(cfg.err_expert_hidden))
                if (cfg.use_err_expert and cfg.use_err_head) else None
            )
            blocks: list[nn.Module] = []
            for kind in _block_kinds(cfg.num_blocks, cfg.block_mode):
                if kind == "normal":
                    blocks.append(
                        NormalAttentionBlock(
                            d,
                            n_heads=int(cfg.n_heads),
                            dropout=cfg.dropout,
                            residual=cfg.residual,
                        )
                    )
                elif int(cfg.moe_experts) > 0:
                    blocks.append(
                        MoEPolarCorrectorBlock(
                            d, int(cfg.moe_experts),
                            dropout=cfg.dropout,
                            residual=cfg.residual,
                            shared_qk_scalars=cfg.shared_qk_scalars,
                            use_causal_mask=cfg.use_causal_mask,
                        )
                    )
                else:
                    blocks.append(
                        PolarCorrectorBlock(
                            d,
                            dropout=cfg.dropout,
                            residual=cfg.residual,
                            shared_qk_scalars=cfg.shared_qk_scalars,
                            use_causal_mask=cfg.use_causal_mask,
                        )
                    )
            self.agent = nn.ModuleList(blocks)

        self.polar_decoder = self.head

    def _apply_path(self, x: torch.Tensor, step: int) -> torch.Tensor:
        if self.path_steps:
            return self.path_steps[step](x)
        return self.path(x)

    def encode_geom(self, geom_coords: torch.Tensor) -> torch.Tensor:
        if self.config.model_type == "foilscan":
            return self._encode_contour(geom_coords)
        if getattr(self, "softbin", None) is not None:
            h = self.path(self.geom_encoder.point(geom_coords))
            mem = self.softbin(h, geom_coords)
        else:
            mem = self.geom_encoder(geom_coords, path=self.path)
        if getattr(self, "profile_in", None) is not None:
            prof = profile_features(geom_coords, int(self.config.profile_stations))
            mem = torch.cat((mem, self.path(self.profile_in(prof)).unsqueeze(1)), dim=1)
        if self.inv_in is not None:
            inv = geom_invariants(geom_coords).to(dtype=geom_coords.dtype)
            mem = torch.cat((mem, self.path(self.inv_in(inv)).unsqueeze(1)), dim=1)
        return mem

    def _cl_base(self, geom_coords: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Potential-flow lift base with r_LE-conditioned stall taper → (B, T) in z-units.

        C_l,base = 2π sin(α−α0)/√(1−Ma²) · cos(α) · exp(−((α−α0)/α_stall(geom))⁴)
        α_stall = softplus(α_stall,0) + softplus(w_r)·√r_LE  (blunt LE → stalls later)
        """
        cfg = self.config
        a_deg = cond[..., 0] * cfg.a_sd + cfg.a_mu
        a_phys = a_deg * math.pi / 180.0  # dataset α is in degrees; theory needs radians
        m_phys = (cond[..., 2] * cfg.m_sd + cfg.m_mu).clamp_min(0.0)
        pg = 1.0 / torch.sqrt(1.0 - m_phys.clamp(max=0.95).square())
        alpha0 = thin_airfoil_alpha0(geom_coords)
        r_le = le_radius(geom_coords)
        alpha_stall = F.softplus(self.alpha_stall_base) + F.softplus(self.w_r_le) * torch.sqrt(r_le.clamp(min=1e-5))
        a_eff = a_phys - alpha0
        taper = torch.cos(a_phys) * torch.exp(-(a_eff / alpha_stall.clamp(min=0.05)).pow(4))
        base_phys = 2.0 * math.pi * torch.sin(a_eff) * pg * taper
        return (base_phys - cfg.cl_mu) / cfg.cl_sd

    def _cd_out(self, ctx: torch.Tensor, cl_z: torch.Tensor, cd_z: torch.Tensor) -> torch.Tensor:
        """Cd head in z-units: Cd0 + k·Cl² + δ, optionally gated with the learned Cd.

        cl_z/cd_z: normalized (z) head outputs for Cl/Cd. With the gate the output is
        g·(quadratic+δ) + (1-g)·learned, so g≈0 recovers the plain learned head and
        g≈1 the rigid drag polar. Gate reads the same ctx the (Cd0, k) heads see.
        """
        cfg = self.config
        cd0 = F.softplus(self.cd0_head(ctx))
        k = F.softplus(self.k_head(ctx))
        cl_phys = cl_z * cfg.cl_sd + cfg.cl_mu
        cd_phys = cd0 + k * cl_phys.square() + cd_z * cfg.cd_sd
        if cfg.phys_residual_gated:
            g = torch.sigmoid(self.gate_head(ctx))
            cd_learned = cd_z * cfg.cd_sd + cfg.cd_mu
            cd_phys = g * cd_phys + (1.0 - g) * cd_learned
        return (cd_phys - cfg.cd_mu) / cfg.cd_sd

    @staticmethod
    def _pool_n_memories(cfg: FoilCoreConfig) -> int:
        if cfg.model_type == "foilcore" and cfg.geom_pool == "softbin":
            return 1 + 2 * int(cfg.softbin_k)
        return int(cfg.n_geom_memories)

    @staticmethod
    def _scan_n_memories(cfg: FoilCoreConfig) -> int:
        if cfg.contour_mode == "pool":
            return int(cfg.n_geom_memories)
        if cfg.directional_landmarks:
            return 8 + (1 if cfg.landmark_mean else 0)
        return int(cfg.n_landmarks)

    def _encode_contour(self, geom_coords: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if cfg.contour_mode == "pool":
            return self.geom_encoder(geom_coords, path=self.path)
        feat = self.geom_encoder.point(geom_coords)
        feat = self.path(feat)
        cond = contour_condition(geom_coords, int(cfg.contour_d_cond))
        if cfg.directional_landmarks:
            h_f, h_b = self.contour_ssm.scan_both(feat, cond)
            return landmark_gather_directional(h_f, h_b, geom_coords, include_mean=bool(cfg.landmark_mean))
        states = self.contour_ssm(feat, cond, bidirectional=bool(cfg.contour_bidirectional))
        return landmark_gather(states, geom_coords, n_landmarks=int(cfg.n_landmarks))

    def _scheduled_clcd(
        self,
        draft: torch.Tensor,
        polar_tokens: torch.Tensor,
        polar_mask: torch.Tensor,
        p_ar: float,
    ) -> torch.Tensor:
        """Mix draft vs GT Cl/Cd for the polar mixer. ``p_ar`` is the draft fraction."""
        gt = polar_tokens[..., 3:5]
        p = float(p_ar)
        if p <= 1e-12:
            return gt
        use = (torch.rand_like(polar_mask, dtype=draft.dtype) < p) & polar_mask
        return torch.where(use.unsqueeze(-1), draft.detach(), gt)

    def _resolve_clcd(
        self,
        draft: torch.Tensor,
        polar_tokens: torch.Tensor,
        polar_mask: torch.Tensor,
        *,
        clcd_for_corrector: torch.Tensor | None,
        clcd_inject: torch.Tensor | None,
        seed_len: int,
        p_ar: float | None,
    ) -> torch.Tensor:
        if clcd_for_corrector is not None:
            return clcd_for_corrector
        if p_ar is not None and float(p_ar) < 1.0 - 1e-12:
            return self._scheduled_clcd(draft, polar_tokens, polar_mask, p_ar)
        return self._seed_clcd(draft, polar_tokens, polar_mask, seed_len, clcd_inject)

    def _seed_clcd(
        self,
        draft: torch.Tensor,
        polar_tokens: torch.Tensor,
        polar_mask: torch.Tensor,
        seed_len: int,
        clcd_inject: torch.Tensor | None,
    ) -> torch.Tensor:
        clcd = draft
        if seed_len > 0 or clcd_inject is not None:
            clcd = draft.clone()
            order = torch.cumsum(polar_mask.long(), dim=1)
            if seed_len > 0:
                seed_pos = polar_mask & (order <= int(seed_len))
                gt = polar_tokens[..., 3:5]
                clcd = torch.where(seed_pos.unsqueeze(-1), gt, clcd)
            if clcd_inject is not None:
                if seed_len > 0:
                    seed_pos = polar_mask & (order <= int(seed_len))
                    clcd = torch.where(seed_pos.unsqueeze(-1), clcd_inject, clcd)
                else:
                    clcd = clcd_inject
        return clcd

    def forward(
        self,
        geom_coords: torch.Tensor,
        polar_tokens: torch.Tensor,
        polar_mask: torch.Tensor,
        *,
        clcd_for_corrector: torch.Tensor | None = None,
        clcd_inject: torch.Tensor | None = None,
        seed_len: int = 0,
        causal: bool | None = None,
        p_ar: float | None = None,
    ):
        """
        geom_coords: (B, 200, 2) chord-normalized
        polar_tokens: (B, T, 5) normalized [α, Re, Mach, Cl, Cd]
        polar_mask: (B, T) True on real polar samples

        Returns ``(final, draft)`` each (B, T, 2).
        """
        self.aux = {}
        if self.config.model_type == "foilscan":
            return self._forward_scan(
                geom_coords,
                polar_tokens,
                polar_mask,
                clcd_for_corrector=clcd_for_corrector,
                clcd_inject=clcd_inject,
                seed_len=seed_len,
                causal=causal,
                p_ar=p_ar,
            )
        if self.config.model_type == "foilcore2":
            return self._forward_v2(
                geom_coords,
                polar_tokens,
                polar_mask,
                clcd_for_corrector=clcd_for_corrector,
                clcd_inject=clcd_inject,
                seed_len=seed_len,
                causal=causal,
                p_ar=p_ar,
            )
        if self.config.model_type == "seqbaseline":
            return self._forward_seqbaseline(geom_coords, polar_tokens, polar_mask, causal=causal)
        memories = self.encode_geom(geom_coords)
        cond = polar_tokens[..., :3]
        ctx = self.mem_mix(memories, cond)
        draft = self.head(ctx)
        if self.config.phys_base:
            base_z = self._cl_base(geom_coords, cond)
            draft = draft + self.base_scale * base_z.unsqueeze(-1) * torch.tensor([1.0, 0.0], device=draft.device, dtype=draft.dtype)
        if self.config.phys_residual:
            draft = torch.cat((draft[..., 0:1], self._cd_out(ctx, draft[..., 0:1], draft[..., 1:2])), dim=-1)

        if not self.agent:
            m = polar_mask.unsqueeze(-1).to(dtype=draft.dtype)
            return draft * m, draft * m

        clcd = self._resolve_clcd(
            draft,
            polar_tokens,
            polar_mask,
            clcd_for_corrector=clcd_for_corrector,
            clcd_inject=clcd_inject,
            seed_len=seed_len,
            p_ar=p_ar,
        )

        feat = torch.cat([cond, clcd], dim=-1)
        if getattr(self, "err_head", None) is not None:
            err_pred = self.err_head(memories, cond, clcd)
            feat = torch.cat([feat, err_pred], dim=-1)
            self.aux = {"err_pred": err_pred, "err_target": (clcd.detach() - polar_tokens[..., 3:5]).abs(), "gates": []}
        if self.config.bl_features:
            re_phys = (cond[..., 1] * self.config.re_sd + self.config.re_mu).clamp_min(5e3)
            a_deg = cond[..., 0] * self.config.a_sd + self.config.a_mu
            a_rad = a_deg * math.pi / 180.0
            r_le = le_radius(geom_coords)
            blf = torch.stack(
                (
                    torch.log10(re_phys) / 7.0,
                    200.0 / torch.sqrt(re_phys),
                    20.0 / re_phys.pow(0.2),
                    torch.tanh((a_rad.abs() / torch.sqrt(r_le.clamp(min=1e-5))).clamp(max=20.0) / 10.0),
                ),
                dim=-1,
            )
            feat = torch.cat([feat, blf], dim=-1)
        tokens = self.polar_in(feat) + ctx
        tokens = tokens + sinusoidal_encoding_batch(
            cond[..., 0].reshape(-1), self.d_model, dtype=tokens.dtype
        ).reshape(*cond.shape[:2], self.d_model)

        pad = ~polar_mask
        alpha = cond[..., 0]
        causal_mask = None
        use_causal = self.config.use_causal_mask if causal is None else bool(causal)
        if use_causal:
            t = tokens.shape[1]
            causal_mask = torch.triu(
                torch.full((t, t), float("-inf"), device=tokens.device, dtype=tokens.dtype),
                diagonal=1,
            )

        h = tokens
        for i, block in enumerate(self.agent):
            if isinstance(block, NormalAttentionBlock):
                h = block(h, causal_mask, pad, alpha)
            else:
                h = block(h, causal_mask, pad, alpha, causal=use_causal)
            h = self._apply_path(h, i)

        if getattr(self, "debug", False):
            self.aux.update(hidden=h, ctx=ctx, memories=memories)
        ctx2 = self.mem_mix(memories, cond, hidden=h)
        final = self.head(ctx2)
        if getattr(self, "err_expert", None) is not None:
            delta, gate = self.err_expert(h, self.aux["err_pred"])
            final = final + delta
            self.aux["expert_gate"] = gate
        if self.config.phys_base:
            base_z_f = self._cl_base(geom_coords, cond)
            final = final + self.base_scale * base_z_f.unsqueeze(-1) * torch.tensor([1.0, 0.0], device=final.device, dtype=final.dtype)
        if self.config.phys_residual:
            final = torch.cat((final[..., 0:1], self._cd_out(ctx2, final[..., 0:1], final[..., 1:2])), dim=-1)
        m = polar_mask.unsqueeze(-1).to(dtype=final.dtype)
        return final * m, draft * m

    def _forward_seqbaseline(self, geom_coords, polar_tokens, polar_mask, *, causal: bool | None):
        """Same front-end as FoilCORE; a generic sequence backbone replaces corrector + requery."""
        memories = self.encode_geom(geom_coords)
        cond = polar_tokens[..., :3]
        ctx = self.mem_mix(memories, cond)
        draft = self.head(ctx)
        feat = torch.cat([cond, draft.detach()], dim=-1)
        tokens = self.polar_in(feat) + ctx
        tokens = tokens + sinusoidal_encoding_batch(
            cond[..., 0].reshape(-1), self.d_model, dtype=tokens.dtype
        ).reshape(*cond.shape[:2], self.d_model)
        use_causal = self.config.use_causal_mask if causal is None else bool(causal)
        h = self.backbone(self.bb_in(tokens), polar_mask, causal=use_causal)
        final = draft + self.bb_head(h)
        m = polar_mask.unsqueeze(-1).to(dtype=final.dtype)
        return final * m, draft * m

    def _forward_v2(
        self,
        geom_coords: torch.Tensor,
        polar_tokens: torch.Tensor,
        polar_mask: torch.Tensor,
        *,
        clcd_for_corrector: torch.Tensor | None,
        clcd_inject: torch.Tensor | None,
        seed_len: int,
        causal: bool | None,
        p_ar: float | None,
    ):
        memories = self.encode_geom(geom_coords)
        cond = polar_tokens[..., :3]
        ctx = self.mem_mix(memories, cond)
        draft = self.head(ctx)
        clcd = self._resolve_clcd(
            draft, polar_tokens, polar_mask,
            clcd_for_corrector=clcd_for_corrector, clcd_inject=clcd_inject,
            seed_len=seed_len, p_ar=p_ar,
        )
        err_pred = self.err_head(memories, cond, clcd)  # (B, T, 2) predicted |error| in z-units
        feat = torch.cat((cond, clcd, err_pred), dim=-1)

        alpha = cond[..., 0]
        tokens = self.polar_in(feat) + ctx
        tokens = tokens + sinusoidal_encoding_batch(
            alpha.reshape(-1), self.d_model, dtype=tokens.dtype
        ).reshape(*cond.shape[:2], self.d_model)

        use_causal = self.config.use_causal_mask if causal is None else bool(causal)
        ao = AlphaOrder(alpha, polar_mask)
        h = tokens
        gates = []
        for i, block in enumerate(self.agent):
            h, gate = block(h, feat, alpha, polar_mask, ao, causal=use_causal)
            h = self._apply_path(h, i)
            if gate is not None:
                gates.append(gate)

        ctx2 = self.mem_mix(memories, cond, hidden=h)
        final = self.head(ctx2)
        m = polar_mask.unsqueeze(-1).to(dtype=final.dtype)
        self.aux = {
            "err_pred": err_pred,
            "err_target": (clcd.detach() - polar_tokens[..., 3:5]).abs(),
            "gates": gates,
        }
        if getattr(self, "debug", False):
            self.aux.update(memories=memories, ctx=ctx, tokens=tokens, hidden=h, ctx2=ctx2)
        return final * m, draft * m

    def _forward_scan(
        self,
        geom_coords: torch.Tensor,
        polar_tokens: torch.Tensor,
        polar_mask: torch.Tensor,
        *,
        clcd_for_corrector: torch.Tensor | None,
        clcd_inject: torch.Tensor | None,
        seed_len: int,
        causal: bool | None,
        p_ar: float | None,
    ):
        memories = self.encode_geom(geom_coords)
        cond = polar_tokens[..., :3]
        ctx = self.mem_mix(memories, cond)
        draft = self.head(ctx)
        clcd = self._resolve_clcd(
            draft,
            polar_tokens,
            polar_mask,
            clcd_for_corrector=clcd_for_corrector,
            clcd_inject=clcd_inject,
            seed_len=seed_len,
            p_ar=p_ar,
        )
        use_causal = self.config.use_causal_mask if causal is None else bool(causal)
        alpha = cond[..., 0]
        if self.config.polar_mixer == "iir_token":
            tokens = ctx
            if self.polar_in is not None:
                tokens = tokens + self.polar_in(clcd)
            if self.config.polar_alpha_sinus:
                tokens = tokens + sinusoidal_encoding_batch(
                    alpha.reshape(-1), self.d_model, dtype=tokens.dtype
                ).reshape(*cond.shape[:2], self.d_model)
            h = tokens
            for i, block in enumerate(self.agent):
                h = block(h, alpha, polar_mask, causal=use_causal)
                h = self._apply_path(h, i)
            hidden = h
        else:
            smooth = self.polar_iir(clcd, alpha, polar_mask, causal=use_causal)
            hidden = self.requery_lift(smooth - draft)
        ctx2 = self.mem_mix(memories, cond, hidden=hidden)
        final = self.head(ctx2)
        m = polar_mask.unsqueeze(-1).to(dtype=draft.dtype)
        return final * m, draft * m
