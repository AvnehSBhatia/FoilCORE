"""
Evaluate a FoilCORE checkpoint on a CSV (same schema as train/val).

Runs three passes: (1) **teacher forcing** — full gold embeddings (matches validation).
(2) **seeded AR** — gold embeddings for the first K **polar** steps after the fixed BOS+geom prefix,
then greedy AR (same polar AoA/Re/Mach rule as training).
(3) **full greedy AR** — like (2) but K=0: every content step uses predicted prefixes.

In AR regions, polar inputs use **ground-truth AoA, Re, and Mach** (sequence-scheduled AoA plus
true Re/Mach); only Cl/Cd are taken from greedy predictions.

Each pass uses next-token ``batched_sequence_loss`` on ``pred[:, :-1]`` vs ``target[:, 1:]``.

Example:
  python -m src.benchmark --ckpt checkpoints/foilcore_best.pt --csv data/test.csv
  python -m src.benchmark --random-one --seed 42 --ckpt checkpoints/foilcore_epoch_004.pt

Reports TF, seeded AR, and full AR pooled metrics (polar MAE/MAPE; geometry heads are unused).

PNG plots (optional): with ``--plot-dir``, saves per-pass subdirs ``tf/``, ``seeded/``, ``ar/``
(``--plot-max`` applies separately to each pass).

  python -m src.benchmark --plot-dir plots/bench --plot-max 8 --ckpt checkpoints/foilcore_best.pt --csv data/val.csv
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data import AeroDataset, collate_variable_sequences
from src.model import FoilCORE, config_from_checkpoint, remap_foilcore_state_dict
from src.train_model import get_device, polar_forward
from src.utils import (
    compute_domain_metrics,
    polar_clcd_components,
)


def _plot_polar_suptitle(plot_mode: str) -> str:
    if plot_mode == "tf":
        return "(teacher-forced)"
    if plot_mode == "seeded":
        return "(seeded AR)"
    return "(greedy AR)"


def load_models(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "stats" not in ckpt:
        sys.exit("Checkpoint must contain 'stats' (use foilcore_epoch_*.pt or re-save best with stats).")
    stats = ckpt["stats"]

    model_cfg = config_from_checkpoint(ckpt)
    model = FoilCORE(config=model_cfg).to(device)
    m_sd = remap_foilcore_state_dict(
        dict(ckpt["model"]), target_shared_qk_scalars=model_cfg.shared_qk_scalars
    )
    model_sd = model.state_dict()
    compatible = {
        k: v for k, v in m_sd.items()
        if k in model_sd and getattr(v, "shape", None) == model_sd[k].shape
    }
    inf = model.load_state_dict(compatible, strict=False)
    if inf.missing_keys or inf.unexpected_keys or len(compatible) != len(m_sd):
        print(
            "Note: model checkpoint loaded strict=False "
            f"(loaded={len(compatible)} skipped={len(m_sd) - len(compatible)} "
            f"missing={len(inf.missing_keys)} unexpected={len(inf.unexpected_keys)} keys)."
        )

    model.eval()
    runtime = ckpt.get("variant_training", {}) if isinstance(ckpt, dict) else {}
    return model, stats, model_cfg, runtime


def _denormalize_polar(norm: torch.Tensor, stats: dict) -> torch.Tensor:
    """Normalized polar outputs → physical units. Supports full 5D or [Cl, Cd] only."""
    offset = 3 if norm.shape[-1] == 2 else 0
    mean = torch.as_tensor(stats["polar_mean"], device=norm.device, dtype=norm.dtype)
    std = torch.as_tensor(stats["polar_std"], device=norm.device, dtype=norm.dtype)
    mean = mean[offset : offset + norm.shape[-1]]
    std = std[offset : offset + norm.shape[-1]]
    return norm * std + mean


def _reconstruct_full_polar(pred_norm: torch.Tensor, target_norm: torch.Tensor, stats: dict) -> torch.Tensor:
    """Return physical 5D polar, forcing AoA/Re/Mach from target when model predicts [Cl, Cd]."""
    if pred_norm.shape[-1] == 5:
        return _denormalize_polar(pred_norm, stats)
    full = _denormalize_polar(target_norm, stats).clone()
    full[..., 3:5] = _denormalize_polar(pred_norm, stats)
    return full


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _dataset_source_row(ds: object, linear_idx: int) -> int:
    """Map a position in the current Dataset (e.g. Subset) to the underlying CSV row index."""
    if isinstance(ds, Subset):
        return int(ds.indices[linear_idx])
    return int(linear_idx)


def _save_polar_png(
    orig: np.ndarray,
    recon: np.ndarray,
    out_path: str,
    suptitle_suffix: str = "(teacher-forced)",
) -> None:
    """orig, recon: (N, 5) physical polars vs sequence step."""
    plt = _plt()
    names = ["AoA", "Re", "Mach", "Cl", "Cd"]
    n = orig.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    axes_flat = axes.flatten()
    steps = np.arange(n)
    for j in range(5):
        ax = axes_flat[j]
        ax.plot(steps, orig[:, j], "b.-", ms=5, lw=1.2, label="original")
        ax.plot(steps, recon[:, j], "r.--", ms=4, lw=1.0, label="recon")
        ax.set_title(names[j])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes_flat[5].axis("off")
    fig.suptitle(f"Polar coefficients vs sequence step {suptitle_suffix}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_pooled_polar_mae_png(polar_metrics: dict, out_path: str) -> None:
    plt = _plt()
    names = ["AoA", "Re", "Mach", "Cl", "Cd"]
    vals = [float(polar_metrics.get(f"{n}_mae", float("nan"))) for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(names, vals, color="steelblue")
    ax.set_ylabel("MAE (physical units)")
    ax.set_title("Pooled polar MAE — original vs reconstruction")
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _print_polar_mae_mape(polar_metrics: dict) -> None:
    print(
        "  MAE / MAPE in physical units (MAPE only where |original| >= per-field floor; "
        "see src.utils._POLAR_MAPE_MIN_ABS):"
    )
    for name in ["AoA", "Re", "Mach", "Cl", "Cd"]:
        km, kp = f"{name}_mae", f"{name}_mape"
        if km in polar_metrics:
            m = polar_metrics[km]
            p = polar_metrics.get(kp, float("nan"))
            p_s = "n/a (no samples above floor)" if p != p else f"{p:.6f}"
            print(f"  {name:4s}  MAE={m:.6f}  MAPE={p_s}")


def _print_random_one_polar_table(
    pp: torch.Tensor,
    tp: torch.Tensor,
    mp: torch.Tensor,
    stats: dict,
) -> None:
    """Teacher-aligned preds/targets/masks (shifted tensors); print original vs reconstructed rows."""
    if not mp.any():
        return
    tgt_norm = tp[mp]
    pred_norm = pp[mp]
    orig = _denormalize_polar(tgt_norm, stats).cpu()
    recon = _reconstruct_full_polar(pred_norm, tgt_norm, stats).cpu()
    names = ("AoA", "Re", "Mach", "Cl", "Cd")
    print()
    print("--- Original vs reconstructed polar (each visible next-token step) ---")
    hdr = f"{'#':>3}  " + "".join(f"{n:>14}" for n in names)
    print(hdr)
    print("     " + "".join(f"{'orig→recon':>14}" for _ in names))
    for i in range(orig.shape[0]):
        parts = [f"{i:>3}  "]
        for j, n in enumerate(names):
            a, r = orig[i, j].item(), recon[i, j].item()
            parts.append(f"{a:.6g}→{r:.6g}".rjust(14))
        print("".join(parts))


@torch.no_grad()
def benchmark(
    ckpt_path: str,
    csv_path: str,
    batch_size: int = 256,
    num_workers: Optional[int] = None,
    max_samples: Optional[int] = None,
    random_one: bool = False,
    seed: Optional[int] = None,
    plot_dir: Optional[str] = None,
    plot_max: int = 16,
    ar_prefix_len: int = 1,
):
    device_s = get_device()
    device = torch.device(device_s)
    prefix_k = int(max(0, ar_prefix_len))
    print(f"Device: {device_s}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Data: {csv_path}")
    if prefix_k > 0:
        print(
            "Runs three passes: teacher forcing; seeded AR (gold prefix of K steps after BOS, "
            f"K={prefix_k}); full greedy AR. AR polar uses GT AoA/Re/Mach; Cl/Cd from preds."
        )
    else:
        print(
            "Runs two passes: teacher forcing and full greedy AR "
            "(--ar-prefix-len 0: seeded pass would match full AR, so it is skipped)."
        )
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        print(
            f"Plot output: {plot_dir}/{{tf,seeded,ar}}/ (up to {plot_max} samples per pass; "
            "pooled_polar_mae.png in each subdir)"
        )

    if not os.path.isfile(ckpt_path):
        sys.exit(f"Missing checkpoint: {ckpt_path}")
    if not os.path.isfile(csv_path):
        sys.exit(f"Missing CSV: {csv_path}")

    model, stats, model_cfg, runtime = load_models(ckpt_path, device)
    _ = runtime

    full_ds = AeroDataset(csv_path, stats=stats, training=False)
    ds = full_ds
    if random_one:
        rng = random.Random(seed)
        idx = rng.randint(0, len(full_ds) - 1)
        ds = Subset(full_ds, [idx])
        batch_size = 1
        print(f"Single random row: dataset index {idx} (seed={seed})")
    elif max_samples is not None:
        n = min(max_samples, len(ds))
        ds = Subset(ds, range(n))
        print(f"Using first {n} rows (--max-samples)")
    if num_workers is None:
        num_workers = 0 if device_s == "mps" else 4
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device_s != "mps"),
        collate_fn=collate_variable_sequences,
    )

    def run_pass(
        ar_feedback: bool,
        tqdm_desc: str,
        plot_mode: Optional[str],
        ar_teacher_prefix: int = 0,
    ) -> dict:
        total_loss_loc = 0.0
        n_batches_loc = 0
        sum_polar_mse = 0.0
        sum_geom_mse = 0.0
        sum_geom_mae_norm = 0.0
        sum_geom_loss = 0.0
        sum_geom_smooth = 0.0
        sum_special_ce = 0.0
        sum_modality_loss = 0.0
        spec_correct = 0
        spec_n = 0
        mod_correct = 0
        mod_n = 0
        polar_preds_all: list[torch.Tensor] = []
        polar_targets_all: list[torch.Tensor] = []
        geom_preds_all: list[torch.Tensor] = []
        geom_targets_all: list[torch.Tensor] = []
        scales_all: list[torch.Tensor] = []
        xmins_all: list[torch.Tensor] = []
        sum_airfoil_mae = 0.0
        sum_airfoil_max = 0.0
        n_airfoil_loc = 0
        data_row_offset = 0
        n_plot_samples = 0
        allow_plots = plot_dir is not None and plot_mode is not None
        plot_subdir: Optional[str] = None
        polar_suptitle = "(teacher-forced)"
        if allow_plots:
            plot_subdir = os.path.join(plot_dir, plot_mode)
            os.makedirs(plot_subdir, exist_ok=True)
            polar_suptitle = _plot_polar_suptitle(plot_mode)

        t_loc = time.perf_counter()
        _iter = dl if random_one else tqdm(dl, desc=tqdm_desc, leave=True)
        for batch in _iter:
            causal = bool(ar_feedback)
            seed_len = int(ar_teacher_prefix) if ar_feedback else 0
            final, _draft, polar, mask = polar_forward(
                model, batch, device, seed_len=seed_len, causal=causal
            )
            pp = final
            tp = polar
            mp = mask

            comp = polar_clcd_components(pp, tp, mp, stats)
            loss = comp["total"]
            total_loss_loc += loss.item()
            n_batches_loc += 1
            sum_polar_mse += comp["polar_mse"].item()

            if mp.any():
                polar_preds_all.append(pp[mp])
                polar_targets_all.append(tp[mp])

            B = polar.shape[0]
            for b in range(B):
                if allow_plots and plot_subdir is not None and n_plot_samples < plot_max:
                    wrote = False
                    src_row = _dataset_source_row(ds, data_row_offset + b)
                    stem = os.path.join(plot_subdir, f"recon_r{src_row:05d}")

                    if mp[b].any():
                        tgt_norm = tp[b, mp[b]]
                        pred_norm = pp[b, mp[b]]
                        o = _denormalize_polar(tgt_norm, stats).cpu().numpy()
                        r = _reconstruct_full_polar(pred_norm, tgt_norm, stats).cpu().numpy()
                        _save_polar_png(o, r, stem + "_polar.png", suptitle_suffix=polar_suptitle)
                        wrote = True

                    if wrote:
                        n_plot_samples += 1

            data_row_offset += B

            if random_one:
                if not ar_feedback:
                    lbl = "(teacher-forced / bidirectional corrector, this batch)"
                elif ar_teacher_prefix > 0:
                    lbl = (
                        f"(seeded causal corrector, {ar_teacher_prefix} GT Cl/Cd seeds, this batch)"
                    )
                else:
                    lbl = "(causal corrector, no Cl/Cd seeds, this batch)"
                print()
                print(f"--- Loss breakdown {lbl} ---")
                print(f"  polar_mse:      {comp['polar_mse'].item():.6f}")
                print(f"  polar_cl_mae:   {comp['polar_cl_mae'].item():.6f}")
                print(f"  polar_cd_mae:   {comp['polar_cd_mae'].item():.6f}")
                print(f"  total:          {comp['total'].item():.6f}")
                _print_random_one_polar_table(pp, tp, mp, stats)

        elapsed = time.perf_counter() - t_loc
        nb = max(1, n_batches_loc)
        mean_loss = total_loss_loc / nb
        polar_metrics: dict = {}
        if polar_preds_all:
            pp_cat = torch.cat(polar_preds_all, dim=0).cpu()
            pt_cat = torch.cat(polar_targets_all, dim=0).cpu()
            polar_metrics = compute_domain_metrics(pp_cat, pt_cat, stats)

        geom_mae_v = None
        if geom_preds_all:
            gp_cat = torch.cat(geom_preds_all, dim=0).cpu()
            gt_cat = torch.cat(geom_targets_all, dim=0).cpu()
            sc = torch.stack(scales_all).float().cpu()
            xm = torch.stack(xmins_all).float().cpu()
            geom_mae_v = compute_geom_mae(gp_cat, gt_cat, sc, xm)

        chord_mae = chord_max = None
        if n_airfoil_loc > 0:
            chord_mae = sum_airfoil_mae / n_airfoil_loc
            chord_max = sum_airfoil_max / n_airfoil_loc

        return {
            "elapsed": elapsed,
            "mean_batch_loss": mean_loss,
            "n_batches": n_batches_loc,
            "sum_polar_mse": sum_polar_mse,
            "sum_geom_mse": sum_geom_mse,
            "sum_geom_mae_norm": sum_geom_mae_norm,
            "sum_geom_loss": sum_geom_loss,
            "sum_geom_smooth": sum_geom_smooth,
            "sum_special_ce": sum_special_ce,
            "sum_modality_loss": sum_modality_loss,
            "spec_correct": spec_correct,
            "spec_n": spec_n,
            "mod_correct": mod_correct,
            "mod_n": mod_n,
            "polar_metrics": polar_metrics,
            "geom_mae": geom_mae_v,
            "airfoil_chord_mae": chord_mae,
            "airfoil_chord_max_mean": chord_max,
            "n_airfoil": n_airfoil_loc,
            "n_plot_samples": n_plot_samples,
        }

    t0_all = time.perf_counter()
    plot_m = "tf" if plot_dir else None
    plot_s = "seeded" if plot_dir and prefix_k > 0 else None
    plot_a = "ar" if plot_dir else None

    tf_results = run_pass(False, "Benchmark (TF)", plot_m)
    ar_results = run_pass(True, "Benchmark (AR greedy)", plot_a, ar_teacher_prefix=0)
    if prefix_k > 0:
        seeded_desc = f"Benchmark (AR seeded, K={prefix_k})"
        seeded_results = run_pass(True, seeded_desc, plot_s, ar_teacher_prefix=prefix_k)
    else:
        seeded_results = ar_results

    def _print_block(title: str, r: dict, mode: str) -> None:
        nb = max(1, r["n_batches"])
        samples_per_s = len(ds) / max(r["elapsed"], 1e-9)
        print()
        print(f"=== {title} ===")
        print(f"Samples: {len(ds)}  Batches: {r['n_batches']}  Batch size: {batch_size}")
        print(f"Pass wall time: {r['elapsed']:.1f}s  ({samples_per_s:.1f} samples/s)")
        print(f"Mean batch loss: {r['mean_batch_loss']:.6f}")
        if r["n_batches"] > 0:
            print("Mean batch loss breakdown (avg over batches):")
            print(f"  polar_mse:      {r['sum_polar_mse'] / nb:.6f}")
            print(f"  geom_mae(norm): {r['sum_geom_mae_norm'] / nb:.6f}  (not supervised)")
            print(f"  geom_loss:      {r['sum_geom_loss'] / nb:.6f}")
            print(f"  geom_smooth:    {r['sum_geom_smooth'] / nb:.6f}")
            print(f"  geom_mse:       {r['sum_geom_mse'] / nb:.6f}   (auxiliary)")
            print(f"  special_ce:     {r['sum_special_ce'] / nb:.6f}")
            print(f"  modality_loss:  {r['sum_modality_loss'] / nb:.6f}  (0.1 × modality_ce)")
        if r["spec_n"] > 0 or r["mod_n"] > 0:
            print("Global top-1 accuracy (micro-averaged over all batches):")
            if r["spec_n"] > 0:
                print(f"  special_acc:    {r['spec_correct'] / r['spec_n']:.6f}  (n={r['spec_n']})")
            if r["mod_n"] > 0:
                print(f"  modality_acc:   {r['mod_correct'] / r['mod_n']:.6f}  (n={r['mod_n']})")

        pm = r["polar_metrics"]
        if pm:
            print()
            if mode == "tf":
                print("=== Polar (pooled) — teacher-forced prefixes ===")
            elif mode == "seeded":
                print(
                    "=== Polar (pooled) — gold prefix then greedy AR; mixed polar uses GT AoA, Re, Mach ==="
                )
            else:
                print(
                    "=== Polar (pooled) — greedy AR prefixes; polar inputs retain GT AoA, Re, Mach ==="
                )
            _print_polar_mae_mape(pm)

        if r["geom_mae"] is not None:
            print()
            if mode == "tf":
                print(
                    "=== Geometry (patch-wise next-token) — teacher-forced prefixes, physical MAE ==="
                )
            elif mode == "seeded":
                print(
                    "=== Geometry (patch-wise next-token) — gold prefix then greedy AR, physical MAE ==="
                )
            else:
                print(
                    "=== Geometry (patch-wise next-token) — greedy AR prefixes, physical MAE ==="
                )
            print(
                "  Mean coordinate MAE in physical units (per-patch aggregation): "
                f"{r['geom_mae']:.6f}"
            )

        if r["airfoil_chord_mae"] is not None:
            print()
            if mode == "tf":
                subtitle = "teacher-forced prefixes"
            elif mode == "seeded":
                subtitle = "gold prefix then greedy predicted prefixes"
            else:
                subtitle = "greedy predicted prefixes"
            print(
                f"=== Full 200-point chord ({subtitle}, mean over {r['n_airfoil']} airfoils) ==="
            )
            print(
                "  Mean coordinate MAE over 400 x/y values per airfoil: "
                f"{r['airfoil_chord_mae']:.6f}"
            )
            print(f"  Mean of per-airfoil max coordinate error: {r['airfoil_chord_max_mean']:.6f}")

    _print_block("Teacher forcing (gold token embeddings)", tf_results, mode="tf")
    if prefix_k > 0:
        _print_block(
            f"Seeded autoregressive (gold prefix K={prefix_k} after BOS, then greedy; polar uses GT AoA/Re/Mach in AR region)",
            seeded_results,
            mode="seeded",
        )
    else:
        print()
        print(
            "=== Seeded autoregressive ==="
            "\n  Skipped: --ar-prefix-len 0 matches full greedy AR (see block below)."
        )
    _print_block(
        "Full greedy autoregressive (predicted embeddings; polar uses GT AoA/Re/Mach, pred Cl/Cd)",
        ar_results,
        mode="ar",
    )

    if plot_dir:
        pooled_written: list[str] = []
        plot_passes: list[tuple[str, dict]] = [("tf", tf_results), ("ar", ar_results)]
        if prefix_k > 0:
            plot_passes.insert(1, ("seeded", seeded_results))
        for sub, r in plot_passes:
            pm = r.get("polar_metrics") or {}
            if pm:
                bar_path = os.path.join(plot_dir, sub, "pooled_polar_mae.png")
                os.makedirs(os.path.dirname(bar_path), exist_ok=True)
                _save_pooled_polar_mae_png(pm, bar_path)
                pooled_written.append(bar_path)
        if pooled_written:
            print()
            for p in pooled_written:
                print(f"  Saved pooled polar MAE: {p}")

        n_tf = int(tf_results["n_plot_samples"])
        n_sd = int(seeded_results["n_plot_samples"]) if prefix_k > 0 else 0
        n_ar = int(ar_results["n_plot_samples"])
        if n_tf > 0 or n_sd > 0 or n_ar > 0:
            print()
            print(
                f"=== Reconstruction PNGs (per-pass samples: tf={n_tf}, seeded={n_sd}, ar={n_ar}; max {plot_max} each) ==="
            )
            print(f"  Root: {plot_dir}")
            print("  Subdirs: tf/, seeded/ (if K>0), ar/ — recon_rXXXXX_airfoil.png, _polar.png, pooled_polar_mae.png")
        else:
            print()
            print("=== Plot dir set but no per-sample PNGs (no polar/geom in sampled rows?) ===")

    elapsed_all = time.perf_counter() - t0_all
    print()
    if prefix_k > 0:
        print(f"Total benchmark wall time (TF + seeded AR + full AR): {elapsed_all:.1f}s")
    else:
        print(f"Total benchmark wall time (TF + full AR): {elapsed_all:.1f}s")

    return {
        "teacher_forced": {
            "mean_batch_loss": tf_results["mean_batch_loss"],
            "polar_metrics": tf_results["polar_metrics"],
            "geom_mae": tf_results["geom_mae"],
            "airfoil_chord_mae": tf_results["airfoil_chord_mae"],
            "airfoil_chord_max_mean": tf_results["airfoil_chord_max_mean"],
        },
        "seeded_autoregressive": {
            "ar_prefix_len": prefix_k,
            "mean_batch_loss": seeded_results["mean_batch_loss"],
            "polar_metrics": seeded_results["polar_metrics"],
            "geom_mae": seeded_results["geom_mae"],
            "airfoil_chord_mae": seeded_results["airfoil_chord_mae"],
            "airfoil_chord_max_mean": seeded_results["airfoil_chord_max_mean"],
        },
        "autoregressive": {
            "mean_batch_loss": ar_results["mean_batch_loss"],
            "polar_metrics": ar_results["polar_metrics"],
            "geom_mae": ar_results["geom_mae"],
            "airfoil_chord_mae": ar_results["airfoil_chord_mae"],
            "airfoil_chord_max_mean": ar_results["airfoil_chord_max_mean"],
        },
    }


def main():
    p = argparse.ArgumentParser(
        description="Teacher-forced, seeded-AR, and full-AR benchmark for FoilCORE on a CSV."
    )
    p.add_argument("--ckpt", default="checkpoints/foilcore_best.pt", help="Model checkpoint path")
    p.add_argument("--csv", default="data/test.csv", help="Evaluation CSV path")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Only evaluate the first N rows. Ignored if --random-one.",
    )
    p.add_argument(
        "--random-one",
        action="store_true",
        help="Evaluate one random row (TF, seeded AR, full AR detailed loss tables; forces batch size 1).",
    )
    p.add_argument(
        "--ar-prefix-len",
        type=int,
        default=1,
        help="For seeded AR: number of sequence positions after BOS kept teacher-forced (default: 1).",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed for --random-one.")
    p.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="PNG overlays for TF, seeded AR, and full AR (subdirs tf/, seeded/, ar/; up to --plot-max samples per pass).",
    )
    p.add_argument(
        "--plot-max",
        type=int,
        default=16,
        help="Max dataset rows to emit per-sample PNGs for (each row may get _airfoil and _polar).",
    )
    args = p.parse_args()
    benchmark(
        args.ckpt,
        args.csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        random_one=args.random_one,
        seed=args.seed,
        plot_dir=args.plot_dir,
        plot_max=args.plot_max,
        ar_prefix_len=args.ar_prefix_len,
    )


if __name__ == "__main__":
    main()
