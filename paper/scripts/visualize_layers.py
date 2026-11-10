#!/usr/bin/env python3
"""One figure per FoilCORE layer: weights + activations on a val polar."""
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
import pandas as pd
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data import build_polar_tokens, normalize_coords, resample_airfoil
from src.embeddings import sinusoidal_encoding_batch
from src.model import FoilCORE, PathNonlinearity, config_from_checkpoint, remap_foilcore_state_dict
from src.train_model import get_device

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
CM_W = "RdBu_r"
CM_A = "magma"


def _np(t) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _heatmap(ax, mat, title, *, cmap=CM_W, center=True, xlabel="", ylabel="", xtl=None, ytl=None, cbar=True):
    mat = np.asarray(mat, dtype=float)
    if center:
        lim = float(max(np.abs(mat).max(), 1e-8))
        vmin, vmax = -lim, lim
    else:
        vmin, vmax = None, None
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xtl is not None:
        ax.set_xticks(range(len(xtl)))
        ax.set_xticklabels(xtl)
    if ytl is not None:
        ax.set_yticks(range(len(ytl)))
        ax.set_yticklabels(ytl)
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return im


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_sample(csv: Path, row_index: int, stats, device):
    df = pd.read_csv(csv)
    row = df.iloc[int(row_index)]
    coords = np.asarray(json.loads(row["coords"]), dtype=np.float64)
    coords_n, _, _ = normalize_coords(resample_airfoil(coords, 200))
    alpha = np.asarray(json.loads(row["alpha"]), dtype=np.float64)
    cl = np.asarray(json.loads(row["Cl"]), dtype=np.float64)
    cd = np.asarray(json.loads(row["Cd"]), dtype=np.float64)
    re = float(row["Re"])
    mach = float(row["mach"])
    tok = build_polar_tokens(alpha, cl, cd, re, mach)
    mean = np.asarray(stats["polar_mean"], dtype=np.float64)
    std = np.asarray(stats["polar_std"], dtype=np.float64)
    tok_z = ((tok - mean) / std).astype(np.float32)
    geom = torch.tensor(coords_n, dtype=torch.float32, device=device).unsqueeze(0)
    polar = torch.tensor(tok_z, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.ones(1, polar.shape[1], dtype=torch.bool, device=device)
    meta = {
        "airfoil": str(row["airfoil"]),
        "Re": re,
        "mach": mach,
        "alpha": alpha.astype(np.float32),
        "cl": cl.astype(np.float32),
        "cd": cd.astype(np.float32),
        "coords": coords_n.astype(np.float32),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }
    return geom, polar, mask, meta


@torch.no_grad()
def _attn_map(attn, x, alpha, pad, causal: bool) -> np.ndarray:
    residual = torch.tanh(x - x)
    Q = attn.s1 * x + attn.s2 * residual + attn.s3
    scores = torch.bmm(Q, Q.transpose(1, 2)) / attn.scale
    if causal:
        t = scores.shape[-1]
        scores = scores + torch.triu(torch.full((t, t), float("-inf"), device=x.device, dtype=scores.dtype), diagonal=1)
    if alpha is not None:
        da = alpha.unsqueeze(-1) - alpha.unsqueeze(-2)
        tau = F.softplus(attn.local_log_tau) + 1e-3
        scores = scores - da.pow(2) / tau
    scores = scores.masked_fill(pad[:, None, :], float("-inf"))
    w = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
    return _np(w[0])


@torch.no_grad()
def trace(model, geom, polar, mask, *, causal=False):
    path: PathNonlinearity = model.path
    enc = model.geom_encoder
    mix = model.mem_mix
    h_lin = enc.point(geom)
    h_path = path(h_lin)
    x = geom[..., 0]
    le = x <= enc.le_xmax
    te = x >= enc.te_xmin
    memories = enc(geom, path=path)
    cond = polar[..., :3]
    delta = F.softplus(mix.W_delta(cond)) + 1e-4
    b_in = mix.W_B(cond)
    c_out = mix.W_C(cond)
    decay = -torch.exp(mix.log_A)
    dA = torch.exp(delta * decay)
    h_scan = torch.zeros_like(c_out)
    keep = 1.0 - dA
    scan_states = []
    for i in range(mix.n_mem):
        x_i = memories[:, i, :].unsqueeze(1)
        h_scan = dA * h_scan + keep * (b_in * x_i)
        scan_states.append(_np(h_scan[0]))
    ctx = c_out * h_scan
    draft = model.head(ctx)
    clcd = draft
    packed = torch.cat([cond, clcd], dim=-1)
    tokens = model.polar_in(packed) + ctx
    pe = sinusoidal_encoding_batch(cond[..., 0].reshape(-1), model.d_model, dtype=tokens.dtype).reshape(*cond.shape[:2], model.d_model)
    tokens = tokens + pe
    pad = ~mask
    alpha = cond[..., 0]
    h = tokens
    blocks = []
    for i, block in enumerate(model.agent):
        attn_w = _attn_map(block.attn, h, alpha, pad, causal)
        pre = h
        h = block(h, None, pad, alpha, causal=causal)
        post_ln = h
        h = model._apply_path(h, i)
        blocks.append(
            {
                "attn": attn_w,
                "pre": _np(pre[0]),
                "ln": _np(post_ln[0]),
                "out": _np(h[0]),
                "Wv": _np(block.attn.W_v.weight),
                "ln_g": _np(block.norm.weight),
                "ln_b": _np(block.norm.bias),
                "s1": float(block.attn.s1),
                "s2": float(block.attn.s2),
                "s3": float(block.attn.s3),
                "tau": float(F.softplus(block.attn.local_log_tau) + 1e-3),
            }
        )
    c2 = mix.W_C(cond) + h
    ctx2 = c2 * h_scan
    final = model.head(ctx2)
    mean = None
    return {
        "coords": _np(geom[0]),
        "h_lin": _np(h_lin[0]),
        "h_path": _np(h_path[0]),
        "le": _np(le[0]),
        "te": _np(te[0]),
        "memories": _np(memories[0]),
        "delta": _np(delta[0]),
        "B": _np(b_in[0]),
        "C": _np(c_out[0]),
        "dA": _np(dA[0]),
        "scan": scan_states,
        "ctx": _np(ctx[0]),
        "draft": _np(draft[0]),
        "polar_in_act": _np(model.polar_in(packed)[0]),
        "pe": _np(pe[0]),
        "tokens": _np(tokens[0]),
        "blocks": blocks,
        "C2": _np(c2[0]),
        "ctx2": _np(ctx2[0]),
        "final": _np(final[0]),
        "W_point": _np(enc.point.weight),
        "b_point": _np(enc.point.bias),
        "W_delta": _np(mix.W_delta.weight),
        "W_B": _np(mix.W_B.weight),
        "W_C": _np(mix.W_C.weight),
        "log_A": _np(mix.log_A),
        "W_polar": _np(model.polar_in.weight),
        "b_polar": _np(model.polar_in.bias),
        "W_head": _np(model.head.weight),
        "b_head": _np(model.head.bias),
        "path": path,
    }


def fig_pipeline(model, out: Path, meta: dict) -> None:
    names = [
        "geom Linear(2→32)",
        "PathNL",
        "3 memories",
        "Mamba query",
        "head draft",
        "polar-in + PE",
        "corrector 0",
        "corrector 1",
        "Mamba requery",
        "head final",
    ]
    n = [96, 416, 0, 416, 66, 192, 1124, 1124, 0, 0]
    labels = ["96", "416", "pool", "416", "66", "192", "1,124", "1,124", "shared", "shared"]
    fig, ax = plt.subplots(figsize=(12.4, 2.6), dpi=160)
    x = np.arange(len(names))
    heights = [max(v, 40) if v == 0 else v for v in n]
    colors = ["#3b6ea8" if v else "#c5d4e8" for v in n]
    ax.bar(x, heights, color=colors, width=0.72)
    for i, lab in enumerate(labels):
        ax.text(i, heights[i] + 30, lab, ha="center", fontsize=8)
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel("params")
    ax.set_title(f"FoilCORE layers — {meta['airfoil']}  Re={meta['Re']:.0f}  M={meta['mach']:g}")
    _save(fig, out)


def fig_geom_point(tr, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), dpi=160, gridspec_kw={"width_ratios": [0.7, 1.4, 1.4]})
    _heatmap(axes[0], tr["W_point"], "W  Linear(2→32)", xlabel="in", xtl=["x", "y"], ylabel="channel")
    _heatmap(axes[1], tr["h_lin"].T, "activation  point Linear  (ch × surface)", cmap=CM_A, center=True, xlabel="surface index 0→200", ylabel="channel")
    axes[2].plot(tr["coords"][:, 0], tr["coords"][:, 1], color="#1f3a5f", lw=1.5)
    axes[2].set_aspect("equal")
    axes[2].set_title("input geometry")
    axes[2].set_xlabel("x/c")
    axes[2].set_ylabel("y/c")
    _save(fig, out)


def fig_pathnl(tr, out: Path, *, title: str) -> None:
    path: PathNonlinearity = tr["path"]
    xs = np.linspace(-2.5, 2.5, 401)
    x = torch.tensor(xs, dtype=torch.float32).unsqueeze(-1).expand(-1, path.num_features)
    phi = _np(path(x))
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.0), dpi=160)
    names = ["a_lin", "m", "A", "μ", "log σ"]
    stack = np.stack([_np(path.a_lin), _np(path.m), _np(path.gauss_amp), _np(path.gauss_mu), _np(path.gauss_log_sigma)])
    _heatmap(axes[0, 0], stack, "per-channel scalars", cmap="viridis", center=False, ytl=names, xlabel="channel")
    _heatmap(axes[0, 1], _np(path.a), "Fourier a_k", xlabel="k", ylabel="channel")
    axes[1, 0].plot(xs, phi.mean(1), color="#3b6ea8", lw=1.6, label="mean φ")
    p10, p90 = np.percentile(phi, 10, axis=1), np.percentile(phi, 90, axis=1)
    axes[1, 0].fill_between(xs, p10, p90, color="#3b6ea8", alpha=0.18)
    axes[1, 0].plot(xs, xs, color="#888", lw=0.8, ls="--", label="identity")
    axes[1, 0].set_title("φ(x) envelope")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].legend(frameon=False, fontsize=8)
    _heatmap(axes[1, 1], tr["h_path"].T, "activation after PathNL (geom)", cmap=CM_A, xlabel="surface index", ylabel="channel")
    fig.suptitle(title, y=1.01)
    _save(fig, out)


def fig_memories(tr, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), dpi=160)
    axes[0].plot(tr["coords"][:, 0], tr["coords"][:, 1], color="#bbb", lw=1.2)
    le, te = tr["le"].astype(bool), tr["te"].astype(bool)
    axes[0].scatter(tr["coords"][le, 0], tr["coords"][le, 1], s=8, c="#c45c26", label="LE x≤0.25")
    axes[0].scatter(tr["coords"][te, 0], tr["coords"][te, 1], s=8, c="#3b6ea8", label="TE x≥0.75")
    axes[0].set_aspect("equal")
    axes[0].set_title("region masks")
    axes[0].legend(frameon=False, fontsize=8)
    _heatmap(axes[1], tr["memories"], "3×32 memories", xlabel="channel", ytl=["global", "LE", "TE"])
    rms = np.sqrt((tr["memories"] ** 2).mean(1))
    axes[2].bar(["global", "LE", "TE"], rms, color=["#555", "#c45c26", "#3b6ea8"])
    axes[2].set_title("memory RMS")
    axes[2].set_ylabel("RMS")
    _save(fig, out)


def fig_mamba(tr, out: Path, *, stage: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.6), dpi=160)
    _heatmap(axes[0, 0], tr["W_delta"], r"$W_\Delta$ (3→32)", xlabel="cond", ylabel="ch", xtl=["α", "Re", "M"])
    _heatmap(axes[0, 1], tr["W_B"], r"$W_B$ (3→32)", xlabel="cond", xtl=["α", "Re", "M"])
    _heatmap(axes[0, 2], tr["W_C"], r"$W_C$ (3→32)", xlabel="cond", xtl=["α", "Re", "M"])
    persist = np.exp(-np.exp(tr["log_A"]))
    axes[1, 0].bar(np.arange(32), persist, color="#3b6ea8", width=0.85)
    axes[1, 0].set_title(r"persistence  $e^{-e^{\log A}}$ at Δ=1")
    axes[1, 0].set_xlabel("channel")
    if stage == "query":
        _heatmap(axes[1, 1], tr["C"].T, "C(α)  query", cmap=CM_A, xlabel="α index", ylabel="channel")
        _heatmap(axes[1, 2], tr["ctx"].T, "ctx = C·h  (query)", cmap=CM_A, xlabel="α index", ylabel="channel")
        fig.suptitle("Layer  Mamba query", y=1.01)
    else:
        _heatmap(axes[1, 1], tr["C2"].T, "C+hidden  requery", cmap=CM_A, xlabel="α index", ylabel="channel")
        _heatmap(axes[1, 2], tr["ctx2"].T, "ctx2  requery", cmap=CM_A, xlabel="α index", ylabel="channel")
        fig.suptitle("Layer  Mamba requery", y=1.01)
    _save(fig, out)


def fig_head(tr, out: Path, *, which: str, meta: dict) -> None:
    pred = tr["draft"] if which == "draft" else tr["final"]
    title = "Layer  shared head → draft" if which == "draft" else "Layer  shared head → final"
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), dpi=160)
    _heatmap(axes[0], tr["W_head"], "W  Linear(32→2)", ytl=["Cl", "Cd"], xlabel="hidden")
    a = meta["alpha"]
    mean, std = meta["mean"], meta["std"]
    cl = pred[:, 0] * std[3] + mean[3]
    cd = pred[:, 1] * std[4] + mean[4]
    axes[1].plot(a, meta["cl"], color="#222", lw=1.5, label="XFOIL")
    axes[1].plot(a, cl, color="#2b6cb0", lw=1.5, label=which)
    axes[1].set_title("Cl")
    axes[1].set_xlabel("α (deg)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(a, meta["cd"], color="#222", lw=1.5, label="XFOIL")
    axes[2].plot(a, cd, color="#2b6cb0", lw=1.5, label=which)
    axes[2].set_title("Cd")
    axes[2].set_xlabel("α (deg)")
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle(title, y=1.02)
    _save(fig, out)


def fig_polar_in(tr, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), dpi=160)
    _heatmap(axes[0], tr["W_polar"], "W  Linear(5→32)", xlabel="field", ylabel="channel", xtl=["α", "Re", "M", "Cl", "Cd"])
    _heatmap(axes[1], tr["polar_in_act"].T, "polar-in(cond, draft)", cmap=CM_A, xlabel="α index", ylabel="channel")
    _heatmap(axes[2], tr["tokens"].T, "tokens = polar-in + ctx + PE", cmap=CM_A, xlabel="α index", ylabel="channel")
    fig.suptitle("Layer  polar-in + residual query + α PE", y=1.02)
    _save(fig, out)


def fig_corrector(tr, out: Path, idx: int) -> None:
    b = tr["blocks"][idx]
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.8), dpi=160)
    _heatmap(axes[0, 0], b["Wv"], f"block {idx}  W_v  32×32", xlabel="in", ylabel="out")
    _heatmap(axes[0, 1], b["attn"], f"attention  α→α  (s1={b['s1']:.2f}, τ={b['tau']:.3f})", cmap="Blues", center=False, xlabel="key α", ylabel="query α")
    axes[0, 2].plot(b["ln_g"], label="γ")
    axes[0, 2].plot(b["ln_b"], label="β")
    axes[0, 2].axhline(1.0, color="#aaa", lw=0.6, ls="--")
    axes[0, 2].set_title("LayerNorm")
    axes[0, 2].set_xlabel("channel")
    axes[0, 2].legend(frameon=False, fontsize=8)
    _heatmap(axes[1, 0], b["pre"].T, "in (pre-attn)", cmap=CM_A, xlabel="α index", ylabel="channel")
    _heatmap(axes[1, 1], b["ln"].T, "after attn+LN", cmap=CM_A, xlabel="α index", ylabel="channel")
    _heatmap(axes[1, 2], b["out"].T, "after PathNL", cmap=CM_A, xlabel="α index", ylabel="channel")
    fig.suptitle(f"Layer  polar corrector block {idx}", y=1.01)
    _save(fig, out)


def fig_scan(tr, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), dpi=160)
    labels = ["after global", "after LE", "after TE"]
    for ax, st, lab in zip(axes, tr["scan"], labels):
        _heatmap(ax, st.T, lab, cmap=CM_A, xlabel="α index", ylabel="channel")
    fig.suptitle("Mamba hidden state through the 3-memory scan", y=1.02)
    _save(fig, out)


def fig_delta(tr, meta, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), dpi=160)
    a = meta["alpha"]
    axes[0].plot(a, tr["delta"].mean(1), color="#3b6ea8", lw=1.5)
    axes[0].set_title("mean Δ(α)")
    axes[0].set_xlabel("α (deg)")
    _heatmap(axes[1], tr["delta"].T, "Δ per channel", cmap=CM_A, center=False, xlabel="α index", ylabel="channel")
    _heatmap(axes[2], tr["dA"].T, "dA = exp(Δ·decay)", cmap=CM_A, center=False, xlabel="α index", ylabel="channel")
    fig.suptitle("Mamba step size Δ and decay dA vs α", y=1.02)
    _save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/foilcore_best.pt")
    ap.add_argument("--csv", default="data/val.csv")
    ap.add_argument("--row", type=int, default=6427)
    ap.add_argument("--outdir", default="results/autopsy_3k/layer_atlas")
    ap.add_argument("--causal", action="store_true")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = config_from_checkpoint(ckpt)
    model = FoilCORE(config=cfg)
    sd = remap_foilcore_state_dict(dict(ckpt["model"]), target_shared_qk_scalars=cfg.shared_qk_scalars)
    model.load_state_dict(
        {k: v for k, v in sd.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape},
        strict=False,
    )
    model.eval()
    geom, polar, mask, meta = _load_sample(Path(args.csv), args.row, ckpt["stats"], device)
    tr = trace(model, geom, polar, mask, causal=bool(args.causal))

    fig_pipeline(model, out / "00_pipeline.png", meta)
    fig_geom_point(tr, out / "01_geom_point.png")
    fig_pathnl(tr, out / "02_pathnl.png", title="Layer  shared PathNonlinearity")
    fig_memories(tr, out / "03_geom_memories.png")
    fig_mamba(tr, out / "04_mamba_query.png", stage="query")
    fig_scan(tr, out / "04b_mamba_scan.png")
    fig_delta(tr, meta, out / "04c_mamba_delta.png")
    fig_head(tr, out / "05_head_draft.png", which="draft", meta=meta)
    fig_polar_in(tr, out / "06_polar_in.png")
    fig_corrector(tr, out / "07_corrector_block0.png", 0)
    fig_corrector(tr, out / "08_corrector_block1.png", 1)
    fig_mamba(tr, out / "09_mamba_requery.png", stage="requery")
    fig_head(tr, out / "10_head_final.png", which="final", meta=meta)

    names_png = [
        "00_pipeline.png",
        "01_geom_point.png",
        "02_pathnl.png",
        "03_geom_memories.png",
        "04_mamba_query.png",
        "04b_mamba_scan.png",
        "04c_mamba_delta.png",
        "05_head_draft.png",
        "06_polar_in.png",
        "07_corrector_block0.png",
        "08_corrector_block1.png",
        "09_mamba_requery.png",
        "10_head_final.png",
    ]
    files = [out / n for n in names_png]
    imgs = [plt.imread(p) for p in files]
    ncol, nrow = 3, int(math.ceil(len(imgs) / 3))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14.5, 4.1 * nrow), dpi=110)
    axes = np.atleast_2d(axes)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < len(imgs):
            ax.imshow(imgs[i])
            ax.set_title(files[i].stem, fontsize=8, pad=4)
    fig.suptitle(f"Layer atlas — {meta['airfoil']}", y=0.995)
    _save(fig, out / "00_contact_sheet.png")

    index = {
        "airfoil": meta["airfoil"],
        "row": args.row,
        "causal": bool(args.causal),
        "files": sorted(p.name for p in out.glob("*.png")),
    }
    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(json.dumps(index, indent=2))
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
