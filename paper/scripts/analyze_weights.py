#!/usr/bin/env python3
"""Dump FoilCORE checkpoint weight stats + inspection figures."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.model import FoilCORE, PathNonlinearity, ResidualProductAttention, config_from_checkpoint, remap_foilcore_state_dict

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _heatmap(mat: np.ndarray, title: str, out: Path, *, xlabel="", ylabel="", xticklabels=None, yticklabels=None, vmin=None, vmax=None, cmap="RdBu_r") -> None:
    h, w = mat.shape
    fig, ax = plt.subplots(figsize=(max(5.2, 0.22 * w + 2.4), max(3.4, 0.16 * h + 1.8)), dpi=180)
    lim = float(max(np.abs(mat).max(), 1e-8)) if vmin is None else None
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=(-lim if vmin is None else vmin), vmax=(lim if vmax is None else vmax), interpolation="nearest")
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=0)
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _bar(values: np.ndarray, title: str, out: Path, *, xlabel="", ylabel="", color="#3b6ea8") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=180)
    ax.bar(np.arange(len(values)), values, color=color, width=0.85)
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _svd_spectrum(mat: np.ndarray) -> dict:
    s = np.linalg.svd(mat, compute_uv=False)
    s = np.asarray(s, dtype=np.float64)
    energy = (s ** 2)
    tot = float(energy.sum()) + 1e-18
    cdf = np.cumsum(energy) / tot
    eff = float((s.sum() ** 2) / ((s ** 2).sum() + 1e-18))
    return {
        "singular_values": [round(float(x), 6) for x in s],
        "effective_rank": round(eff, 3),
        "rank90": int(np.searchsorted(cdf, 0.90) + 1),
        "rank99": int(np.searchsorted(cdf, 0.99) + 1),
        "spectral_norm": round(float(s[0]), 4),
        "condition": round(float(s[0] / max(s[-1], 1e-12)), 2),
    }


def _tensor_stats(t: torch.Tensor) -> dict:
    x = t.detach().float().reshape(-1)
    return {
        "shape": list(t.shape),
        "n": int(x.numel()),
        "mean": round(float(x.mean()), 5),
        "std": round(float(x.std(unbiased=False)), 5),
        "rms": round(float(x.pow(2).mean().sqrt()), 5),
        "min": round(float(x.min()), 5),
        "max": round(float(x.max()), 5),
        "abs_max": round(float(x.abs().max()), 5),
        "l2": round(float(x.norm()), 5),
    }


def _group_params(model: FoilCORE) -> dict[str, int]:
    buckets = {
        "geom_encoder": 0,
        "path_nl": 0,
        "mamba_mix": 0,
        "polar_in": 0,
        "corrector": 0,
        "head": 0,
        "other": 0,
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = int(p.numel())
        if name.startswith("geom_encoder"):
            buckets["geom_encoder"] += n
        elif name.startswith("path") or name.startswith("path_steps"):
            buckets["path_nl"] += n
        elif name.startswith("mem_mix"):
            buckets["mamba_mix"] += n
        elif name.startswith("polar_in"):
            buckets["polar_in"] += n
        elif name.startswith("agent"):
            buckets["corrector"] += n
        elif name.startswith("head") or name.startswith("polar_decoder"):
            buckets["head"] += n
        else:
            buckets["other"] += n
    return buckets


def _path_phi(path: PathNonlinearity, xs: np.ndarray) -> np.ndarray:
    x = torch.tensor(xs, dtype=torch.float32).unsqueeze(-1).expand(-1, path.num_features)
    with torch.no_grad():
        y = path(x)
    return _to_np(y)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/foilcore_best.pt")
    p.add_argument("--metrics", default="checkpoints/metrics_history.json")
    p.add_argument("--outdir", default="results/autopsy_3k/weight_analysis")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = config_from_checkpoint(ckpt)
    model = FoilCORE(config=cfg)
    m_sd = remap_foilcore_state_dict(dict(ckpt["model"]), target_shared_qk_scalars=cfg.shared_qk_scalars)
    inf = model.load_state_dict(m_sd, strict=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buckets = _group_params(model)

    path = model.path if isinstance(model.path, PathNonlinearity) else None
    mix = model.mem_mix
    polar_in = model.polar_in
    head = model.head
    geom_w = model.geom_encoder.point.weight
    geom_b = model.geom_encoder.point.bias

    log_A = _to_np(mix.log_A)
    A_mag = np.exp(log_A)
    persist = np.exp(-A_mag)  # dA at delta=1

    rpa_rows = []
    wv_svds = []
    for i, block in enumerate(model.agent):
        attn = block.attn if isinstance(block.attn, ResidualProductAttention) else None
        if attn is None:
            continue
        tau = float(F.softplus(attn.local_log_tau) + 1e-3)
        row = {
            "block": i,
            "s1": round(float(attn.s1), 4),
            "s2": round(float(attn.s2), 4),
            "s3": round(float(attn.s3), 4),
            "local_log_tau": round(float(attn.local_log_tau), 4),
            "tau": round(tau, 4),
            "alpha_half_width_rad": round(math.sqrt(tau * math.log(2.0)), 4),
        }
        rpa_rows.append(row)
        wv_svds.append({"block": i, **_svd_spectrum(_to_np(attn.W_v.weight))})

    polar_cols = ["alpha", "Re", "Mach", "Cl", "Cd"]
    cond_cols = ["alpha", "Re", "Mach"]
    polar_col_rms = np.sqrt((_to_np(polar_in.weight) ** 2).mean(axis=0))
    geom_col_rms = np.sqrt((_to_np(geom_w) ** 2).mean(axis=0))
    cond_rms = {
        "delta": np.sqrt((_to_np(mix.W_delta.weight) ** 2).mean(axis=0)).tolist(),
        "B": np.sqrt((_to_np(mix.W_B.weight) ** 2).mean(axis=0)).tolist(),
        "C": np.sqrt((_to_np(mix.W_C.weight) ** 2).mean(axis=0)).tolist(),
    }
    head_row_rms = np.sqrt((_to_np(head.weight) ** 2).mean(axis=1))

    xs = np.linspace(-2.5, 2.5, 401)
    phi = None
    path_channel_pick = []
    if path is not None:
        phi = _path_phi(path, xs)
        a_lin = _to_np(path.a_lin)
        gauss_amp = _to_np(path.gauss_amp)
        a_f = _to_np(path.a)
        fourier_energy = (a_f ** 2).sum(axis=1)
        scores = np.abs(a_lin - 1.0) + np.abs(gauss_amp) * 4 + np.sqrt(fourier_energy)
        path_channel_pick = [int(i) for i in np.argsort(-scores)[:8]]

    tensor_table = []
    for name, p in model.named_parameters():
        st = _tensor_stats(p)
        st["name"] = name
        tensor_table.append(st)
    tensor_table.sort(key=lambda r: -r["n"])

    svd_report = {
        "geom_point": _svd_spectrum(_to_np(geom_w)),
        "polar_in": _svd_spectrum(_to_np(polar_in.weight)),
        "head": _svd_spectrum(_to_np(head.weight)),
        "W_delta": _svd_spectrum(_to_np(mix.W_delta.weight)),
        "W_B": _svd_spectrum(_to_np(mix.W_B.weight)),
        "W_C": _svd_spectrum(_to_np(mix.W_C.weight)),
        "W_v": wv_svds,
    }

    metrics = None
    if Path(args.metrics).is_file():
        with open(args.metrics) as f:
            metrics = json.load(f)
        hist = metrics["history"]
        train_l = [h["train_loss"] for h in hist]
        val_l = [h["val_loss"] for h in hist]
        fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=180)
        ep = [h["epoch"] for h in hist]
        ax.plot(ep, train_l, lw=1.4, color="#3b6ea8", label="train loss")
        ax.plot(ep, val_l, lw=1.4, color="#c45c26", label="val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Polar loss (z-scored MAE mix)")
        ax.set_title("Training curves — foilcore_best.pt")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / "training_curves.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=180)
        ax.plot(ep, [h["val_cl_mae_norm"] for h in hist], lw=1.4, color="#3b6ea8", label="val Cl MAE (z)")
        ax.plot(ep, [h["val_cd_mae_norm"] for h in hist], lw=1.4, color="#c45c26", label="val Cd MAE (z)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized MAE")
        ax.set_title("Validation Cl / Cd (z-scored) vs epoch")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / "val_mae_curves.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # --- figures ---
    names = list(buckets.keys())
    counts = np.array([buckets[k] for k in names], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=180)
    ax.barh(names[::-1], counts[::-1], color="#3b6ea8")
    for y, v in enumerate(counts[::-1]):
        ax.text(v + 20, y, f"{int(v)}", va="center", fontsize=9)
    ax.set_xlabel("Trainable parameters")
    ax.set_title(f"Parameter budget — {n_params} total")
    fig.tight_layout()
    fig.savefig(outdir / "param_budget.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    _heatmap(_to_np(geom_w), "Geometry point Linear(2 → 32)", outdir / "geom_point_W.png", xlabel="input (x, y)", ylabel="hidden dim", xticklabels=["x", "y"])
    _heatmap(_to_np(polar_in.weight), "Polar-in Linear(5 → 32)", outdir / "polar_in_W.png", xlabel="token field", ylabel="hidden dim", xticklabels=polar_cols)
    _heatmap(_to_np(head.weight), "Shared head Linear(32 → 2)", outdir / "head_W.png", xlabel="hidden dim", ylabel="output", yticklabels=["Cl", "Cd"])
    _heatmap(_to_np(mix.W_delta.weight), "Mamba W_Δ  Linear(3 → 32)", outdir / "mamba_W_delta.png", xlabel="condition", ylabel="channel", xticklabels=cond_cols)
    _heatmap(_to_np(mix.W_B.weight), "Mamba W_B  Linear(3 → 32)", outdir / "mamba_W_B.png", xlabel="condition", ylabel="channel", xticklabels=cond_cols)
    _heatmap(_to_np(mix.W_C.weight), "Mamba W_C  Linear(3 → 32)", outdir / "mamba_W_C.png", xlabel="condition", ylabel="channel", xticklabels=cond_cols)

    _bar(log_A, "Mamba log A (per channel)", outdir / "mamba_logA.png", xlabel="channel k", ylabel="log A")
    _bar(persist, "Mamba persistence exp(−e^{log A}) at Δ=1", outdir / "mamba_persistence.png", xlabel="channel k", ylabel="dA | Δ=1")

    for i, block in enumerate(model.agent):
        attn = block.attn
        _heatmap(_to_np(attn.W_v.weight), f"Corrector block {i}  W_v (32×32)", outdir / f"corrector_Wv_block{i}.png", xlabel="in", ylabel="out")
        ln = block.norm
        fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=180)
        ax.plot(_to_np(ln.weight), lw=1.4, label="γ")
        ax.plot(_to_np(ln.bias), lw=1.4, label="β")
        ax.axhline(1.0, color="#aaa", lw=0.6, ls="--")
        ax.set_title(f"Corrector block {i} LayerNorm")
        ax.set_xlabel("channel")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / f"corrector_ln_block{i}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    if path is not None:
        names_p = ["a_lin", "m", "A_g", "μ", "log σ"]
        stack = np.stack(
            [
                _to_np(path.a_lin),
                _to_np(path.m),
                _to_np(path.gauss_amp),
                _to_np(path.gauss_mu),
                _to_np(path.gauss_log_sigma),
            ]
        )
        _heatmap(stack, "PathNL per-channel scalars", outdir / "pathnl_scalars.png", xlabel="channel", yticklabels=names_p, cmap="viridis", vmin=float(stack.min()), vmax=float(stack.max()))
        _heatmap(_to_np(path.a), "PathNL Fourier amplitudes a_k", outdir / "pathnl_a.png", xlabel="harmonic k", ylabel="channel")
        _heatmap(_to_np(path.b), "PathNL Fourier scales b_k", outdir / "pathnl_b.png", xlabel="harmonic k", ylabel="channel")

        fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=180)
        for ch in path_channel_pick:
            ax.plot(xs, phi[:, ch], lw=1.3, label=f"ch {ch}")
        ax.axhline(0, color="#888", lw=0.5)
        ax.axvline(0, color="#888", lw=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("φ(x)")
        ax.set_title("PathNL φ(x) — 8 most non-linear channels")
        ax.legend(frameon=False, ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "pathnl_phi_curves.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=180)
        mean = phi.mean(axis=1)
        p10, p90 = np.percentile(phi, 10, axis=1), np.percentile(phi, 90, axis=1)
        ax.fill_between(xs, p10, p90, color="#3b6ea8", alpha=0.18, label="10–90% channels")
        ax.plot(xs, mean, color="#3b6ea8", lw=1.6, label="mean φ")
        ax.plot(xs, xs, color="#888", lw=0.9, ls="--", label="identity")
        ax.set_xlabel("x")
        ax.set_ylabel("φ(x)")
        ax.set_title("PathNL envelope across 32 channels")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / "pathnl_phi_envelope.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # SVD overlay
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=180)
    for label, spec in [
        ("polar_in", svd_report["polar_in"]),
        ("W_Δ", svd_report["W_delta"]),
        ("W_B", svd_report["W_B"]),
        ("W_C", svd_report["W_C"]),
        ("W_v0", svd_report["W_v"][0]),
        ("W_v1", svd_report["W_v"][1]) if len(svd_report["W_v"]) > 1 else None,
    ]:
        if spec is None:
            continue
        s = spec["singular_values"]
        ax.plot(np.arange(1, len(s) + 1), s, lw=1.3, marker="o", ms=3, label=label)
    ax.set_xlabel("singular index")
    ax.set_ylabel("σ")
    ax.set_title("Singular spectra of learned maps")
    ax.set_yscale("log")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "svd_spectra.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # polar-in / mamba column energy
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=180)
    ax.bar(polar_cols, polar_col_rms, color="#3b6ea8")
    ax.set_ylabel("RMS of W columns")
    ax.set_title("Polar-in: which token fields the corrector reads")
    fig.tight_layout()
    fig.savefig(outdir / "polar_in_col_rms.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=180)
    x = np.arange(3)
    w = 0.25
    ax.bar(x - w, cond_rms["delta"], w, label="Δ", color="#3b6ea8")
    ax.bar(x, cond_rms["B"], w, label="B", color="#c45c26")
    ax.bar(x + w, cond_rms["C"], w, label="C", color="#5a9a4a")
    ax.set_xticks(x, cond_cols)
    ax.set_ylabel("RMS of W columns")
    ax.set_title("Mamba: which conditions drive Δ / B / C")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "mamba_cond_rms.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=180)
    ax.bar(["x", "y"], geom_col_rms, color="#3b6ea8")
    ax.set_ylabel("RMS of W columns")
    ax.set_title("Geometry encoder: x vs y energy")
    fig.tight_layout()
    fig.savefig(outdir / "geom_col_rms.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    report = {
        "ckpt": args.ckpt,
        "n_params": n_params,
        "missing_keys": list(inf.missing_keys),
        "unexpected_keys": list(inf.unexpected_keys),
        "config": {k: (v if not isinstance(v, float) else v) for k, v in vars(cfg).items()},
        "param_budget": buckets,
        "rpa": rpa_rows,
        "mamba": {
            "log_A_mean": round(float(log_A.mean()), 4),
            "log_A_min": round(float(log_A.min()), 4),
            "log_A_max": round(float(log_A.max()), 4),
            "persist_mean": round(float(persist.mean()), 4),
            "persist_min": round(float(persist.min()), 4),
            "persist_max": round(float(persist.max()), 4),
            "log_A": [round(float(x), 4) for x in log_A],
            "persist": [round(float(x), 4) for x in persist],
            "cond_rms": {k: [round(float(x), 4) for x in v] for k, v in cond_rms.items()},
            "W_delta_bias_rms": round(float(mix.W_delta.bias.detach().pow(2).mean().sqrt()), 4),
            "W_B_bias_rms": round(float(mix.W_B.bias.detach().pow(2).mean().sqrt()), 4),
            "W_C_bias_rms": round(float(mix.W_C.bias.detach().pow(2).mean().sqrt()), 4),
        },
        "polar_in_col_rms": {k: round(float(v), 4) for k, v in zip(polar_cols, polar_col_rms)},
        "geom_col_rms": {"x": round(float(geom_col_rms[0]), 4), "y": round(float(geom_col_rms[1]), 4)},
        "geom_bias_rms": round(float(geom_b.detach().pow(2).mean().sqrt()), 4),
        "head_row_rms": {"Cl": round(float(head_row_rms[0]), 4), "Cd": round(float(head_row_rms[1]), 4)},
        "head_bias": [round(float(x), 4) for x in _to_np(head.bias)],
        "svd": svd_report,
        "tensors": tensor_table,
        "pathnl": None,
        "training": None,
        "note_s2_dead": "PolarCorrectorBlock calls attn(x, x), so residual tanh(x-z)=0 and s2 never contributes at inference/train.",
    }
    if path is not None:
        report["pathnl"] = {
            "n_terms": path.n_terms,
            "a_lin_mean": round(float(path.a_lin.mean()), 4),
            "a_lin_std": round(float(path.a_lin.std(unbiased=False)), 4),
            "m_rms": round(float(path.m.pow(2).mean().sqrt()), 4),
            "gauss_amp_mean": round(float(path.gauss_amp.mean()), 4),
            "gauss_amp_absmax": round(float(path.gauss_amp.abs().max()), 4),
            "fourier_a_rms": round(float(path.a.pow(2).mean().sqrt()), 4),
            "fourier_b_mean": round(float(path.b.mean()), 4),
            "fourier_b_frac_near_one": round(float((path.b.abs() > 0.9).float().mean()), 4),
            "picked_channels": path_channel_pick,
            "a_lin": [round(float(x), 4) for x in _to_np(path.a_lin)],
            "gauss_amp": [round(float(x), 4) for x in _to_np(path.gauss_amp)],
        }
    if metrics is not None:
        hist = metrics["history"]
        best = min(hist, key=lambda h: h["val_loss"])
        report["training"] = {
            "n_epochs": len(hist),
            "best_epoch": best["epoch"],
            "best_val_loss": round(best["val_loss"], 4),
            "final_val_loss": round(hist[-1]["val_loss"], 4),
            "final_val_cl": round(hist[-1]["val_cl_mae_norm"], 4),
            "final_val_cd": round(hist[-1]["val_cd_mae_norm"], 4),
            "epochs": [h["epoch"] for h in hist],
            "train_loss": [round(h["train_loss"], 4) for h in hist],
            "val_loss": [round(h["val_loss"], 4) for h in hist],
            "val_cl": [round(h["val_cl_mae_norm"], 4) for h in hist],
            "val_cd": [round(h["val_cd_mae_norm"], 4) for h in hist],
        }

    with open(outdir / "stats.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"n_params={n_params}")
    print("budget", buckets)
    print("rpa", rpa_rows)
    print("missing", inf.missing_keys)
    print(f"wrote {outdir.resolve()}")


if __name__ == "__main__":
    main()
