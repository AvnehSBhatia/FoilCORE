"""
Near-stall → +ΔAoA polar prediction benchmark.

For each CSV row, defines pre-stall from **max-Cl** (classic 2-D stall proxy), takes the last polar
sample with AoA in ``[stall_aoa - window, stall_aoa]`` as **context end** ``i``, and targets the
sample ``j`` whose AoA is closest to ``alpha[i] + delta_aoa`` (default +3°).

**Teacher forcing:** full gold embeddings → read ``polar_pred[b, j]`` vs gold polar ``j``.

**AR (same semantics as ``src.benchmark``):** first forward on gold; build mixed inputs with
``ar_teacher_prefix = i + 1`` (gold through polar index ``i`` after the fixed BOS+geom prefix);
second forward. Polar tokens in the AR region still use **ground-truth AoA, Re, Mach** with
**predicted Cl/Cd** from the prior step (see ``mix_autoregressive_inputs``).

Geometry is always the dataset airfoil (full chord, gold embeddings only — not predicted).

Example:
  python -m src.benchmark_stall --ckpt checkpoints/fc_best.pt --csv data/test.csv
  python -m src.benchmark_stall --random-one --seed 42 --ckpt checkpoints/fc_best.pt --csv data/test.csv
  python -m src.benchmark_stall --random-one --plot-dir plots/stall_run --ckpt checkpoints/fc_best.pt --csv data/test.csv
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.benchmark import load_models
from src.data import AeroDataset, collate_variable_sequences
from src.train_model import get_device, polar_forward


def _as_float_array(col) -> np.ndarray:
    """CSV cell may be a JSON string or already parsed (e.g. after ``AeroDataset`` loads the frame)."""
    if isinstance(col, str):
        col = json.loads(col)
    return np.asarray(col, dtype=np.float64)


def _denormalize_polar(norm: torch.Tensor, stats: dict) -> torch.Tensor:
    offset = 3 if norm.shape[-1] == 2 else 0
    mean = torch.as_tensor(stats["polar_mean"], device=norm.device, dtype=norm.dtype)
    std = torch.as_tensor(stats["polar_std"], device=norm.device, dtype=norm.dtype)
    mean = mean[offset : offset + norm.shape[-1]]
    std = std[offset : offset + norm.shape[-1]]
    return norm * std + mean


def _reconstruct_full_polar(pred_norm: torch.Tensor, target_norm: torch.Tensor, stats: dict) -> torch.Tensor:
    if pred_norm.shape[-1] == 5:
        return _denormalize_polar(pred_norm, stats)
    full = _denormalize_polar(target_norm, stats).clone()
    full[..., 3:5] = _denormalize_polar(pred_norm, stats)
    return full


def _plt_agg():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_stall_polar_curves(
    orig: np.ndarray,
    recon_tf: np.ndarray,
    recon_ar: np.ndarray,
    out_path: str,
    context_i: int,
    target_j: int,
    suptitle: str,
) -> None:
    """orig / recon_*: (N_polar, 5) physical; x-axis is polar index."""
    plt = _plt_agg()
    names = ["AoA", "Re", "Mach", "Cl", "Cd"]
    n = orig.shape[0]
    steps = np.arange(n)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    axes_flat = axes.flatten()
    for fi in range(5):
        ax = axes_flat[fi]
        ax.plot(steps, orig[:, fi], "b.-", ms=4, lw=1.2, label="gold")
        ax.plot(steps, recon_tf[:, fi], "r.--", ms=3, lw=1.0, label="TF pred")
        ax.plot(steps, recon_ar[:, fi], "g.:", ms=3, lw=1.0, label="AR pred")
        if 0 <= context_i < n:
            ax.axvline(context_i, color="k", ls=":", alpha=0.45)
        if 0 <= target_j < n:
            ax.axvline(target_j, color="m", ls=":", alpha=0.45)
        ax.set_title(names[fi])
        ax.grid(True, alpha=0.3)
        if fi == 0:
            ax.legend(fontsize=7, loc="best")
    axes_flat[5].axis("off")
    axes_flat[5].text(
        0.1,
        0.55,
        "Vertical lines: black=context i, magenta=target j",
        fontsize=10,
        transform=axes_flat[5].transAxes,
    )
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_stall_target_bars(
    phys_tgt: np.ndarray,
    phys_tf: np.ndarray,
    phys_ar: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    plt = _plt_agg()
    names = ["AoA", "Re", "Mach", "Cl", "Cd"]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
    x = np.arange(3)
    for k, ax in enumerate(axes):
        ax.bar(x, [phys_tgt[k], phys_tf[k], phys_ar[k]], color=["#4c72b0", "#dd8452", "#55a868"])
        ax.set_xticks(x, ["gold", "TF", "AR"])
        ax.tick_params(axis="x", rotation=15)
        ax.set_title(names[k])
        ax.grid(True, axis="y", alpha=0.35)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def find_stall_plus_delta_indices(
    alpha: np.ndarray,
    cl: np.ndarray,
    *,
    near_stall_deg: float,
    delta_aoa: float,
    max_alpha_err: float,
) -> Optional[tuple[int, int, int]]:
    """
    Returns (context_i, target_j, stall_idx) or None if invalid.
    ``context_i`` is the last index with AoA in [stall_aoa - near_stall_deg, stall_aoa].
    ``target_j`` is argmin |alpha - (alpha[i] + delta_aoa)|.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    cl = np.asarray(cl, dtype=np.float64)
    if alpha.ndim != 1 or cl.ndim != 1 or len(alpha) != len(cl) or len(alpha) < 3:
        return None
    stall_idx = int(np.argmax(cl))
    stall_aoa = float(alpha[stall_idx])
    mask_near = (alpha <= stall_aoa) & (alpha >= stall_aoa - float(near_stall_deg))
    cand = np.where(mask_near)[0]
    if cand.size == 0:
        return None
    i = int(cand[-1])
    target_aoa = float(alpha[i]) + float(delta_aoa)
    j = int(np.argmin(np.abs(alpha - target_aoa)))
    if j <= i:
        return None
    if float(np.abs(alpha[j] - target_aoa)) > max_alpha_err:
        return None
    return i, j, stall_idx


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark next polar point after near-stall (+ΔAoA).")
    p.add_argument("--ckpt", type=str, default="checkpoints/fc_best.pt")
    p.add_argument("--csv", type=str, default="data/test.csv")
    p.add_argument("--near-stall-deg", type=float, default=5.0, help="AoA window below stall peak Cl.")
    p.add_argument("--delta-aoa", type=float, default=3.0, help="Target offset in degrees from context AoA.")
    p.add_argument(
        "--max-alpha-err",
        type=float,
        default=1.25,
        help="Skip row if closest AoA to target is farther than this (deg).",
    )
    p.add_argument("--max-samples", type=int, default=None, help="Cap evaluated rows (after filtering). Ignored with --random-one.")
    p.add_argument("--limit-rows", type=int, default=None, help="Only consider row indices < N (scan or random draw).")
    p.add_argument(
        "--random-one",
        action="store_true",
        help="Sample random rows until one has a valid stall task; evaluate that row only (fast).",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed for --random-one.")
    p.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="If set, save polar curves, target-step bars, and airfoil PNGs (first --plot-max tasks).",
    )
    p.add_argument("--plot-max", type=int, default=32, help="Max number of tasks to emit plots for.")
    args = p.parse_args()

    device_s = get_device()
    device = torch.device(device_s)

    if not os.path.isfile(args.ckpt):
        sys.exit(f"Missing checkpoint: {args.ckpt}")
    if not os.path.isfile(args.csv):
        sys.exit(f"Missing CSV: {args.csv}")

    model, stats, model_cfg, _runtime = load_models(args.ckpt, device)
    _ = model_cfg
    full_ds = AeroDataset(args.csv, stats=stats, training=False)

    n_rows = len(full_ds) if args.limit_rows is None else min(len(full_ds), args.limit_rows)
    tasks: list[tuple[int, int, int, int]] = []

    if args.random_one:
        rng = random.Random(args.seed)
        max_tries = min(20_000, max(500, n_rows * 4))
        for _ in range(max_tries):
            idx = rng.randrange(0, n_rows)
            row = full_ds.df.iloc[idx]
            alpha = _as_float_array(row["alpha"])
            cl = _as_float_array(row["Cl"])
            ij = find_stall_plus_delta_indices(
                alpha,
                cl,
                near_stall_deg=args.near_stall_deg,
                delta_aoa=args.delta_aoa,
                max_alpha_err=args.max_alpha_err,
            )
            if ij is None:
                continue
            i, j, s = ij
            tasks = [(idx, i, j, s)]
            break
        if not tasks:
            sys.exit(
                f"No valid (near-stall, +ΔAoA) task after {max_tries} random tries in [0, {n_rows}); "
                "loosen --max-alpha-err / --near-stall-deg or use full scan without --random-one."
            )
    else:
        for idx in range(n_rows):
            row = full_ds.df.iloc[idx]
            alpha = _as_float_array(row["alpha"])
            cl = _as_float_array(row["Cl"])
            ij = find_stall_plus_delta_indices(
                alpha,
                cl,
                near_stall_deg=args.near_stall_deg,
                delta_aoa=args.delta_aoa,
                max_alpha_err=args.max_alpha_err,
            )
            if ij is None:
                continue
            i, j, s = ij
            tasks.append((idx, i, j, s))
            if args.max_samples is not None and len(tasks) >= args.max_samples:
                break

        if not tasks:
            sys.exit(
                "No valid (near-stall, +ΔAoA) tasks; try larger --max-alpha-err or --near-stall-deg, "
                "or a different CSV."
            )

    print(f"Device: {device_s}")
    print(f"Checkpoint: {args.ckpt}")
    if args.random_one:
        idx0, i0, j0, s0 = tasks[0]
        print(f"CSV: {args.csv}  (--random-one, row index {idx0}, context_i={i0} target_j={j0} stall_idx={s0}, seed={args.seed})")
    else:
        print(f"CSV: {args.csv}  (scanned {n_rows} rows)")
    print(
        f"Task: context = last AoA in [stall−{args.near_stall_deg:.1f}°, stall], "
        f"target ≈ context AoA + {args.delta_aoa:.1f}° (|error| ≤ {args.max_alpha_err:.2f}°)."
    )
    print(f"Evaluating {len(tasks)} samples (batch size 1).")
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        print(f"Plots: {args.plot_dir}/ (up to {args.plot_max} tasks)")

    dl = DataLoader(
        Subset(full_ds, [t[0] for t in tasks]),
        batch_size=1,
        shuffle=False,
        num_workers=0 if device_s == "mps" else 2,
        pin_memory=(device_s != "mps"),
        collate_fn=collate_variable_sequences,
    )

    sum_abs_tf = np.zeros(5, dtype=np.float64)
    sum_abs_ar = np.zeros(5, dtype=np.float64)
    sum_sq_tf_cd = 0.0
    sum_sq_ar_cd = 0.0
    n = 0
    saved_plots: list[str] = []

    for task_num, (batch, (idx, i, j, stall_idx)) in enumerate(zip(dl, tasks)):
        polar_tf, _, targets_polar, mask = polar_forward(model, batch, device, causal=False)
        polar_ar, _, _, _ = polar_forward(
            model, batch, device, seed_len=i + 1, causal=True
        )
        pred_tf_j = polar_tf[0, j]
        pred_ar_j = polar_ar[0, j]
        tgt_j = targets_polar[0, j]
        phys_tgt = _denormalize_polar(tgt_j, stats).cpu().numpy()
        phys_tf = _reconstruct_full_polar(pred_tf_j, tgt_j, stats).cpu().numpy()
        phys_ar = _reconstruct_full_polar(pred_ar_j, tgt_j, stats).cpu().numpy()
        err_tf = np.abs(phys_tf - phys_tgt)
        err_ar = np.abs(phys_ar - phys_tgt)
        sum_abs_tf += err_tf
        sum_abs_ar += err_ar
        sum_sq_tf_cd += float((phys_tf[3] - phys_tgt[3]) ** 2 + (phys_tf[4] - phys_tgt[4]) ** 2)
        sum_sq_ar_cd += float((phys_ar[3] - phys_tgt[3]) ** 2 + (phys_ar[4] - phys_tgt[4]) ** 2)
        n += 1

        if args.plot_dir and task_num < args.plot_max:
            P = int(batch["polar_lengths"][0].item())
            orig = np.stack(
                [_denormalize_polar(targets_polar[0, k], stats).cpu().numpy() for k in range(P)]
            )
            recon_tf = np.stack(
                [
                    _reconstruct_full_polar(
                        polar_tf[0, k], targets_polar[0, k], stats
                    ).cpu().numpy()
                    for k in range(P)
                ]
            )
            recon_ar = np.stack(
                [
                    _reconstruct_full_polar(
                        polar_ar[0, k], targets_polar[0, k], stats
                    ).cpu().numpy()
                    for k in range(P)
                ]
            )
            stem = os.path.join(args.plot_dir, f"stall_r{idx:05d}")
            sub = f"row {idx}  context_i={i}  target_j={j}  stall_idx={stall_idx}"
            p_curves = f"{stem}_polar_curves.png"
            p_bars = f"{stem}_target_bars.png"
            _save_stall_polar_curves(
                orig,
                recon_tf,
                recon_ar,
                p_curves,
                i,
                j,
                f"Stall benchmark — full polar (physical)\n{sub}",
            )
            _save_stall_target_bars(
                phys_tgt,
                phys_tf,
                phys_ar,
                p_bars,
                f"Target polar (step j={j}) — {sub}",
            )
            saved_plots.extend([p_curves, p_bars])

    names = ["AoA", "Re", "Mach", "Cl", "Cd"]
    mae_tf = sum_abs_tf / n
    mae_ar = sum_abs_ar / n
    rmse_cd_pair_tf = (sum_sq_tf_cd / n) ** 0.5
    rmse_cd_pair_ar = (sum_sq_ar_cd / n) ** 0.5

    print()
    print("=== Mean absolute error at target polar (physical units) ===")
    print(f"{'field':<6} {'TF MAE':>12} {'AR MAE':>12} {'Δ(AR-TF)':>12}")
    for k, nm in enumerate(names):
        d = mae_ar[k] - mae_tf[k]
        print(f"{nm:<6} {mae_tf[k]:12.6g} {mae_ar[k]:12.6g} {d:12.6g}")
    print()
    print(f"RMSE on (Cl, Cd) joint at target:  TF={rmse_cd_pair_tf:.6g}  AR={rmse_cd_pair_ar:.6g}")
    print(f"(n={n} tasks; AR uses GT AoA/Re/Mach in mixed polar embeds, benchmark two-forward recipe.)")

    if saved_plots:
        print()
        print(f"Saved {len(saved_plots)} plot files under {args.plot_dir}")


if __name__ == "__main__":
    main()
