#!/usr/bin/env python3
"""
Ablation evaluation: FoilCORE vs NeuralFoil — produces paper/data/ablation_val.json.

FoilCORE conditions evaluated on the full val set:
  foilcore_tf           — teacher-forced, single forward pass
  foilcore_ar_gt_{N}    — AR with N GT Cl/Cd seed points  (N ∈ {0,1,2,3,5})
  foilcore_ar_rnd_{N}   — AR with N random-Cl/Cd seeds, GT AoA/Re/Mach  (N ∈ {1,2,3,5})
  neuralfoil_{size}      — NeuralFoil xxsmall / xxxlarge

A row is included only when FoilCORE TF **and** all NeuralFoil sizes succeed.
All conditions are then evaluated on the identical airfoil set at the same truncated length.

Metrics saved per condition (global + stall region):
  Cl MAE / Cd MAE  →  n, mean, std, median, p25, p75, p95
  Pairwise win rates vs each NeuralFoil model

Stall region: ±{STALL_WIN}° around peak-Cl; only rows where the stall peak is not the
  last polar point, with ≥ 2 points in the window.

AR seeding notes:
  GT seeds   — first N polar tokens given exact GT Cl/Cd embeddings.
  Rand seeds — first N polar tokens given GT AoA/Re/Mach but Cl/Cd ~ N(0,1) (normalized space).
  All AR conditions use GT AoA/Re/Mach at every predicted step (same as training AR mixing).

Paper / reviewer disclosure (also written under ``meta.notes`` in the JSON):
  * Default metrics use ``data/val.csv`` (model selection). Paper numbers should use
    ``data/test.csv``. Splits are geometry-held-out; see ``python -m src.dataset_split_report``.
  * NeuralFoil is a pointwise MLP: each (α, Re) uses the same airfoil coordinates and receives
    scalar Re matching that sample — comparable information to FoilCORE teacher-forced polar
    inputs. FoilCORE autoregressive modes withhold Cl/Cd history and are strictly harder for
    FoilCORE only.
  * ``--ood-geometry-only`` restricts evaluation to validation curves whose geometry
    fingerprint never appears in ``--train-csv`` (geometry out-of-distribution).
  * ``--latency-samples K`` adds wall-clock ms/polar-curve for FoilCORE TF vs NeuralFoil
    (see ``meta.latency_benchmark`` for devices).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date
import multiprocessing as mp
from pathlib import Path

import pandas as pd

# Keep each NeuralFoil worker single-threaded; parallelism comes from processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import neuralfoil as nf
from tqdm import tqdm

from src.data import AeroDataset, collate_variable_sequences, geom_fingerprint
from src.model import FoilCORE, config_from_checkpoint, remap_foilcore_state_dict
from src.train_model import (
    get_device,
    polar_forward,
)

# ── Configuration ────────────────────────────────────────────────────────────
CKPT        = "checkpoints/foilcore_best.pt"
CSV         = "data/val.csv"
NF_SIZES    = ["xxsmall", "xxxlarge"]
SEED_COUNTS = [0, 1, 2, 3, 5]   # GT-seeded AR prefix lengths
RAND_SEEDS  = [1, 2, 3, 5]      # random-seeded AR prefix lengths
STALL_WIN   = 3.0               # ±deg around peak-Cl for stall-region MAE
OUT_JSON    = "paper/data/ablation_val.json"
# ─────────────────────────────────────────────────────────────────────────────

_device = torch.device(get_device())


# ── Helpers ──────────────────────────────────────────────────────────────────

def _stats(vals: list) -> dict | None:
    a = np.asarray(vals, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return None
    return {
        "n":      int(len(a)),
        "mean":   float(np.mean(a)),
        "std":    float(np.std(a)),
        "median": float(np.median(a)),
        "p25":    float(np.percentile(a, 25)),
        "p75":    float(np.percentile(a, 75)),
        "p95":    float(np.percentile(a, 95)),
    }


def _default_batch() -> int:
    return 2048


def _default_nf_workers() -> int:
    return max(1, os.cpu_count() or 1)


def _label(cond: str) -> str:
    if cond == "tf":
        return "foilcore_tf"
    if cond.startswith("ar_gt_"):
        return "foilcore_" + cond
    if cond.startswith("ar_rnd_"):
        return "foilcore_" + cond
    if cond.startswith("nf_"):
        return "neuralfoil_" + cond[3:]
    return cond


def load_foilcore(ckpt_path: str):
    ck    = torch.load(ckpt_path, map_location=_device, weights_only=False)
    stats = ck["stats"]
    raw   = dict(ck["model"])
    model_cfg = config_from_checkpoint(ck)
    model = FoilCORE(config=model_cfg).to(_device)
    state_dict = remap_foilcore_state_dict(raw, target_shared_qk_scalars=model_cfg.shared_qk_scalars)
    model_sd = model.state_dict()
    compatible = {
        k: v for k, v in state_dict.items()
        if k in model_sd and getattr(v, "shape", None) == model_sd[k].shape
    }
    inf = model.load_state_dict(compatible, strict=False)
    if inf.missing_keys or inf.unexpected_keys or len(compatible) != len(state_dict):
        print(
            "Warning: checkpoint only partially matches current architecture "
            f"(loaded={len(compatible)} skipped={len(state_dict) - len(compatible)} "
            f"missing={len(inf.missing_keys)} unexpected={len(inf.unexpected_keys)})."
        )
    model.eval()
    runtime = ck.get("variant_training", {}) if isinstance(ck, dict) else {}
    return model, stats, model_cfg, runtime


def _decode(polar_out: torch.Tensor, mask_pol: torch.Tensor, mean, std) -> list:
    """Same-index decode → list of (cl_phys, cd_phys) arrays or None per row."""
    out = []
    if polar_out.shape[-1] == 2:
        mean = mean[3:5]
        std = std[3:5]
    for b in range(polar_out.shape[0]):
        mp = mask_pol[b]
        if not mp.any():
            out.append(None)
            continue
        phys = (polar_out[b][mp] * std + mean).cpu().numpy()
        if polar_out.shape[-1] == 2:
            out.append((phys[:, 0], phys[:, 1]))
        else:
            out.append((phys[:, 3], phys[:, 4]))
    return out


def _rand_clcd_inject(polar: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    inj = torch.randn(polar.shape[0], polar.shape[1], 2, device=polar.device, dtype=polar.dtype)
    return inj


def _fc_batch(model, items: list, stats: dict, model_cfg, runtime: dict) -> dict[str, list]:
    """Run TF + seeded/causal FoilCORE conditions on one collated batch."""
    _ = (model_cfg, runtime)
    batch = collate_variable_sequences(items)
    res = {}

    with torch.no_grad():
        mean = torch.as_tensor(stats["polar_mean"], device=_device)
        std = torch.as_tensor(stats["polar_std"], device=_device)

        final, _, polar, mask = polar_forward(model, batch, _device, causal=False)
        res["tf"] = _decode(final, mask, mean, std)

        for N in SEED_COUNTS:
            fin, _, _, m = polar_forward(model, batch, _device, seed_len=N, causal=True)
            res[f"ar_gt_{N}"] = _decode(fin, m, mean, std)

        inj = _rand_clcd_inject(polar, mask)
        for N in RAND_SEEDS:
            fin, _, _, m = polar_forward(
                model, batch, _device, seed_len=N, causal=True, clcd_inject=inj
            )
            res[f"ar_rnd_{N}"] = _decode(fin, m, mean, std)

    return res


def _fc_safe(model, items: list, stats: dict, model_cfg, runtime: dict) -> dict[str, list]:
    """_fc_batch with per-item fallback on batch-level exception."""
    try:
        return _fc_batch(model, items, stats, model_cfg, runtime)
    except Exception:
        pass
    B    = len(items)
    conds = (["tf"]
             + [f"ar_gt_{N}"  for N in SEED_COUNTS]
             + [f"ar_rnd_{N}" for N in RAND_SEEDS])
    out  = {c: [None] * B for c in conds}
    for i, it in enumerate(items):
        try:
            sub = _fc_batch(model, [it], stats, model_cfg, runtime)
            for c, lst in sub.items():
                out[c][i] = lst[0]
        except Exception:
            pass
    return out


def _nf_preds(rec, nf_sizes: list) -> dict:
    """NeuralFoil predictions per size; None on failure."""
    try:
        coords = np.asarray(rec["coords"], dtype=float)
        alpha  = np.asarray(rec["alpha"],  dtype=float)
        Re     = np.full_like(alpha, float(rec["Re"]), dtype=float)
    except Exception:
        return {sz: None for sz in nf_sizes}
    out = {}
    for sz in nf_sizes:
        try:
            p = nf.get_aero_from_coordinates(
                coordinates=coords, alpha=alpha, Re=Re, model_size=sz
            )
            out[sz] = (np.asarray(p["CL"], dtype=float),
                       np.asarray(p["CD"], dtype=float))
        except Exception:
            out[sz] = None
    return out


def _nf_preds_worker(args: tuple[int, dict, tuple[str, ...]]) -> tuple[int, dict]:
    """ProcessPool wrapper. Keep this top-level so it is spawn-picklable."""
    idx, rec, nf_sizes = args
    return idx, _nf_preds(rec, list(nf_sizes))


def _stall_region(alpha: np.ndarray, cl: np.ndarray) -> np.ndarray | None:
    """
    Boolean mask of points within ±STALL_WIN deg of peak Cl.
    Returns None if the stall peak is the last point (no post-stall visible)
    or if fewer than 2 points fall inside the window.
    """
    si = int(np.argmax(cl))
    if si >= len(alpha) - 1:
        return None
    m = np.abs(alpha - alpha[si]) <= STALL_WIN
    return m if m.sum() >= 2 else None


def _sync_torch_device(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def _train_geom_hash_cache_path(train_csv: str) -> Path:
    p = Path(train_csv).resolve()
    return p.parent / ".train_geom_fingerprints_cache.pkl"


def load_train_geom_hashes(train_csv: str) -> set[str]:
    """All training-set geometry fingerprints (cached next to the CSV by mtime)."""
    p = Path(train_csv)
    if not p.is_file():
        raise FileNotFoundError(train_csv)
    cache = _train_geom_hash_cache_path(train_csv)
    mtime = p.stat().st_mtime
    if cache.is_file():
        try:
            with open(cache, "rb") as f:
                blob = pickle.load(f)
            if blob.get("mtime") == mtime and isinstance(blob.get("hashes"), set):
                print(f"Loaded {len(blob['hashes'])} train geometry hashes from cache {cache.name}")
                return blob["hashes"]
        except (OSError, pickle.UnpicklingError, EOFError, TypeError):
            pass
    df = pd.read_csv(p, usecols=["coords"])
    tqdm.pandas(desc="train coords json")
    df["coords"] = df["coords"].progress_apply(json.loads)
    h: set[str] = set()
    for c in tqdm(df["coords"], desc="train geom fingerprints", total=len(df)):
        h.add(geom_fingerprint(c))
    try:
        with open(cache, "wb") as f:
            pickle.dump({"mtime": mtime, "hashes": h}, f, protocol=4)
        print(f"Wrote geometry hash cache ({len(h)} unique) → {cache}")
    except OSError as exc:
        print(f"Could not write geometry hash cache ({exc}); continuing.")
    return h


def _fc_tf_forward_once(model, item, stats: dict, model_cfg, runtime: dict, device: torch.device):
    """Single teacher-forced forward (matches ``_fc_batch`` TF path) for timing."""
    _ = (stats, model_cfg, runtime)
    batch = collate_variable_sequences([item])
    with torch.inference_mode():
        _ = polar_forward(model, batch, device, causal=False)


def run_latency_benchmark(
    model,
    ds: AeroDataset,
    eval_indices: list[int],
    stats: dict,
    model_cfg,
    runtime: dict,
    n_samples: int,
    nf_size: str,
    warmup: int = 5,
) -> dict:
    """Wall-clock ms per full polar curve: FoilCORE TF (neural device) vs NeuralFoil (CPU)."""
    rng = np.random.default_rng(0)
    pool = list(eval_indices)
    if len(pool) > n_samples:
        pool = list(rng.choice(np.asarray(pool), size=n_samples, replace=False))
    elif len(pool) < n_samples:
        n_samples = len(pool)

    dev = _device
    for _ in range(warmup):
        _fc_tf_forward_once(model, ds[pool[0]], stats, model_cfg, runtime, dev)
    _sync_torch_device(dev)

    fc_times: list[float] = []
    for idx in pool:
        _sync_torch_device(dev)
        t0 = time.perf_counter()
        _fc_tf_forward_once(model, ds[idx], stats, model_cfg, runtime, dev)
        _sync_torch_device(dev)
        fc_times.append((time.perf_counter() - t0) * 1000.0)

    nf_times: list[float] = []
    for idx in pool:
        rec = ds.df.iloc[idx].to_dict()
        coords = np.asarray(rec["coords"], dtype=float)
        alpha = np.asarray(rec["alpha"], dtype=float)
        Re = np.full_like(alpha, float(rec["Re"]), dtype=float)
        t0 = time.perf_counter()
        _ = nf.get_aero_from_coordinates(
            coordinates=coords, alpha=alpha, Re=Re, model_size=nf_size
        )
        nf_times.append((time.perf_counter() - t0) * 1000.0)

    def _summ(v: list[float]) -> dict:
        a = np.asarray(v, dtype=float)
        return {"mean_ms": float(np.mean(a)), "std_ms": float(np.std(a)), "n": int(len(a))}

    return {
        "curves_timed": n_samples,
        "indices_seed": 0,
        "foilcore": {
            "mode": "teacher_forced_one_forward",
            "batch_size": 1,
            "device": str(dev),
            **_summ(fc_times),
        },
        "neuralfoil": {
            "model_size": nf_size,
            "device": "cpu",
            "note": "NeuralFoil runs in this repo on CPU inside the latency loop; compare to FoilCORE on GPU with care.",
            **_summ(nf_times),
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="FoilCORE ablation eval (TF / AR seeding / NeuralFoil)."
    )
    ap.add_argument("--ckpt",       default=CKPT,  help="FoilCORE checkpoint")
    ap.add_argument("--csv",        default=CSV,   help="Validation CSV")
    ap.add_argument("--out",        default=OUT_JSON)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="FoilCORE batch size (default: 2048)")
    ap.add_argument("--nf-workers", type=int, default=None,
                    help="Parallel NeuralFoil worker processes (default: CPU count)")
    ap.add_argument("--nf-chunksize", type=int, default=64,
                    help="Rows per NeuralFoil worker chunk (default: 64)")
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Evaluate only the first N CSV rows (default: entire dataset).",
    )
    ap.add_argument(
        "--train-csv",
        type=str,
        default=None,
        help="Training CSV (required with --ood-geometry-only). Used for geometry-held-out filtering.",
    )
    ap.add_argument(
        "--ood-geometry-only",
        action="store_true",
        help="Keep only validation rows whose geometry fingerprint never appears in --train-csv.",
    )
    ap.add_argument(
        "--latency-samples",
        type=int,
        default=0,
        help="If >0, measure wall-clock ms per polar for FoilCORE TF vs NeuralFoil xxsmall on this many curves.",
    )
    args       = ap.parse_args()
    batch_size = args.batch_size or _default_batch()
    nf_workers = args.nf_workers if args.nf_workers is not None else _default_nf_workers()
    nf_workers = max(1, int(nf_workers))
    nf_chunksize = max(1, int(args.nf_chunksize))

    if args.ood_geometry_only and not args.train_csv:
        raise SystemExit("--ood-geometry-only requires --train-csv")

    print("=" * 65)
    print("FoilCORE Ablation Evaluation")
    print("=" * 65)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    stats = ck["stats"]
    ds = AeroDataset(args.csv, stats=stats, training=False)
    n_full = len(ds)
    n_cap = min(n_full, int(args.max_rows)) if args.max_rows is not None else n_full
    eval_indices = list(range(n_cap))
    train_hashes: set[str] | None = None
    if args.ood_geometry_only:
        train_hashes = load_train_geom_hashes(args.train_csv)
        eval_indices = [
            i for i in eval_indices
            if geom_fingerprint(ds.df.iloc[i]["coords"]) not in train_hashes
        ]
        print(
            f"OOD geometry filter: {len(eval_indices)} / {n_cap} rows "
            f"(fingerprints absent from {args.train_csv})"
        )
        if len(eval_indices) == 0:
            raise SystemExit("No rows left after --ood-geometry-only; check train/val paths.")
        if len(eval_indices) < 50:
            print(
                f"Warning: only {len(eval_indices)} geometry-OOD rows — see "
                "paper/data/dataset_split_report.json (often near-zero for this dataset)."
            )
    n = len(eval_indices)
    if n_cap < n_full:
        print(f"Dataset : using first {n_cap} of {n_full} rows (--max-rows), evaluating {n} after filters")
    else:
        print(f"Dataset : {n_full} CSV rows, evaluating {n} after filters")
    print(f"Device  : {_device}   batch-size: {batch_size}")
    print(f"NeuralFoil workers: {nf_workers} processes")
    print("Decode  : full-sequence forwards (no KV cache)")
    print(f"Output  : {args.out}")

    print("\nPhase 1/2: NeuralFoil precompute")
    nf_cache = [None] * n
    nf_payloads = [
        (j, ds.df.iloc[eval_indices[j]].to_dict(), tuple(NF_SIZES))
        for j in range(n)
    ]
    # NF first: fork workers before ROCm model load. This avoids forking a live GPU context.
    nf_start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    nf_ctx = mp.get_context(nf_start_method)
    with ProcessPoolExecutor(max_workers=nf_workers, mp_context=nf_ctx) as nf_pool:
        nf_iter = nf_pool.map(_nf_preds_worker, nf_payloads, chunksize=nf_chunksize)
        for i, pred in tqdm(nf_iter, total=n, desc="NeuralFoil"):
            nf_cache[i] = pred

    print("\nPhase 2/2: FoilCORE ablations")
    model, _, model_cfg, runtime = load_foilcore(args.ckpt)

    latency_meta: dict | None = None
    if int(args.latency_samples) > 0:
        latency_meta = run_latency_benchmark(
            model,
            ds,
            eval_indices,
            stats,
            model_cfg,
            runtime,
            n_samples=int(args.latency_samples),
            nf_size="xxsmall",
        )
        print("\nLatency (ms / full polar curve, batch size 1):")
        print(f"  FoilCORE TF      mean={latency_meta['foilcore']['mean_ms']:.3f} ms")
        print(f"  NeuralFoil xxs   mean={latency_meta['neuralfoil']['mean_ms']:.3f} ms  ({latency_meta['neuralfoil']['note']})")

    fc_conds  = (["tf"]
                 + [f"ar_gt_{N}"  for N in SEED_COUNTS]
                 + [f"ar_rnd_{N}" for N in RAND_SEEDS])
    nf_conds  = [f"nf_{sz}" for sz in NF_SIZES]
    all_conds = fc_conds + nf_conds

    # Per-airfoil MAE lists (nan = condition failed for that row)
    g_cl: dict[str, list] = defaultdict(list)   # global Cl MAE
    g_cd: dict[str, list] = defaultdict(list)   # global Cd MAE
    s_cl: dict[str, list] = defaultdict(list)   # stall-region Cl MAE
    s_cd: dict[str, list] = defaultdict(list)   # stall-region Cd MAE
    n_valid = 0
    n_stall = 0

    for start in tqdm(range(0, n, batch_size), desc="FoilCORE"):
        batch_positions = list(range(start, min(start + batch_size, n)))
        indices = [eval_indices[j] for j in batch_positions]
        items   = [ds[i] for i in indices]
        recs    = [ds.df.iloc[i] for i in indices]
        fc = _fc_safe(model, items, stats, model_cfg, runtime)

        for k, rec in enumerate(recs):
                # ── Ground truth ──────────────────────────────────────────────
                try:
                    cl_gt = np.asarray(rec["Cl"],    dtype=float)
                    cd_gt = np.asarray(rec["Cd"],    dtype=float)
                    alpha = np.asarray(rec["alpha"], dtype=float)
                except Exception:
                    continue
                if cl_gt.size < 2 or cl_gt.shape != cd_gt.shape:
                    continue

                # ── Require TF + all NF sizes to succeed ───────────────────
                tf_pair = (fc.get("tf") or [None] * len(items))[k]
                if tf_pair is None:
                    continue
                nf_res = nf_cache[batch_positions[k]]
                if nf_res is None:
                    continue
                if any(nf_res[sz] is None for sz in NF_SIZES):
                    continue

                # ── Common evaluation length ──────────────────────────────
                min_len = min(
                    len(cl_gt),
                    len(tf_pair[0]),
                    *[len(nf_res[sz][0]) for sz in NF_SIZES],
                )
                if min_len < 2:
                    continue

                cl_g = cl_gt[:min_len]
                cd_g = cd_gt[:min_len]
                a    = alpha[:min_len]

                sm        = _stall_region(a, cl_g)
                has_stall = sm is not None

                n_valid += 1
                if has_stall:
                    n_stall += 1

                # ── Collect all condition predictions ─────────────────────
                preds: dict[str, tuple | None] = {}
                preds["tf"] = tf_pair
                for cond in fc_conds:
                    if cond == "tf":
                        continue
                    lst       = fc.get(cond) or [None] * len(items)
                    preds[cond] = lst[k]
                for sz in NF_SIZES:
                    preds[f"nf_{sz}"] = nf_res[sz]

                # ── Accumulate MAEs ───────────────────────────────────────
                for cond in all_conds:
                    pair = preds.get(cond)
                    if pair is None:
                        g_cl[cond].append(float("nan"))
                        g_cd[cond].append(float("nan"))
                        if has_stall:
                            s_cl[cond].append(float("nan"))
                            s_cd[cond].append(float("nan"))
                        continue
                    cl_p = pair[0][:min_len]
                    cd_p = pair[1][:min_len]
                    g_cl[cond].append(float(np.mean(np.abs(cl_p - cl_g))))
                    g_cd[cond].append(float(np.mean(np.abs(cd_p - cd_g))))
                    if has_stall:
                        s_cl[cond].append(float(np.mean(np.abs(cl_p[sm] - cl_g[sm]))))
                        s_cd[cond].append(float(np.mean(np.abs(cd_p[sm] - cd_g[sm]))))

    # ── Build output ─────────────────────────────────────────────────────────
    conditions_out = {}
    for cond in all_conds:
        conditions_out[_label(cond)] = {
            "global": {
                "cl_mae": _stats(g_cl[cond]),
                "cd_mae": _stats(g_cd[cond]),
            },
            "stall": {
                "cl_mae": _stats(s_cl[cond]),
                "cd_mae": _stats(s_cd[cond]),
            },
        }

    # Win rates: each FC condition vs each NF model (pairwise, same airfoils)
    win_rates = {}
    for fc_c in fc_conds:
        cl_fc = np.asarray(g_cl[fc_c])
        cd_fc = np.asarray(g_cd[fc_c])
        for sz in NF_SIZES:
            nf_c  = f"nf_{sz}"
            cl_nf = np.asarray(g_cl[nf_c])
            cd_nf = np.asarray(g_cd[nf_c])
            both  = ~np.isnan(cl_fc) & ~np.isnan(cl_nf)
            nb    = int(both.sum())
            key   = f"{_label(fc_c)}_vs_neuralfoil_{sz}"
            win_rates[key] = {
                "n":          nb,
                "cl_win_rate": float(np.mean(cl_fc[both] < cl_nf[both])) if nb else None,
                "cd_win_rate": float(np.mean(cd_fc[both] < cd_nf[both])) if nb else None,
            }

    summary = {
        "meta": {
            "checkpoint":       args.ckpt,
            "dataset":          args.csv,
            "dataset_rows_full": n_full,
            "rows_after_max_rows_cap": n_cap,
            "rows_evaluated":   n,
            "max_rows_arg":     args.max_rows,
            "ood_geometry_only": bool(args.ood_geometry_only),
            "train_csv_for_ood": args.train_csv,
            "device":           str(_device),
            "n_valid_airfoils": n_valid,
            "n_stall_airfoils": n_stall,
            "stall_window_deg": STALL_WIN,
            "seed_counts_gt":   SEED_COUNTS,
            "seed_counts_rand": RAND_SEEDS,
            "neuralfoil_sizes": NF_SIZES,
            "neuralfoil_workers": nf_workers,
            "neuralfoil_chunksize": nf_chunksize,
            "neuralfoil_start_method": nf_start_method,
            "date":             str(date.today()),
            "latency_benchmark": latency_meta,
            "notes": {
                "train_val_test_split": "Geometry-held-out splits from tools/build_bigfoil_dataset.py: "
                "train.csv / val.csv / test.csv. Default evals.py path is val.csv; report test.csv for paper tables. "
                "Counts and overlap checks: python -m src.dataset_split_report",
                "inclusion":    "Row included only if FoilCORE TF and all NeuralFoil sizes succeed.",
                "eval_length":  "All conditions truncated to min(GT, TF, NF_xxsmall, NF_xxxlarge) length.",
                "evaluation_symmetry": "NeuralFoil is evaluated pointwise with ground-truth α and Re per sample "
                "(and fixed Mach) — the fair comparison to FoilCORE is foilcore_tf, which also receives "
                "ground-truth AoA/Re/Mach at every polar step. Autoregressive FoilCORE conditions intentionally "
                "omit Cl/Cd teacher forcing and are a harder, FoilCORE-only stress test.",
                "in_distribution_scope": "Val and test airfoil shapes are held out from train (near-duplicate "
                "clusters cannot cross splits). --ood-geometry-only is a sanity check and should be a no-op "
                "on the rebuilt CSVs.",
                "ar_mixing":    "AR uses GT AoA/Re/Mach at every step; Cl/Cd propagated from prior prediction.",
                "rand_seeds":   "Random seeds: Cl/Cd ~ N(0,1) in normalised space; AoA/Re/Mach from GT.",
                "stall_region": f"±{STALL_WIN}° around peak-Cl; excluded if peak is last point or <2 pts in window.",
            },
        },
        "conditions": conditions_out,
        "win_rates":  win_rates,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\nValid airfoils : {n_valid}  |  with stall region : {n_stall}")
    w = 26
    print(f"\n{'Condition':<{w}} {'Cl MAE':>10} {'Cd MAE':>10} {'Stall Cl':>10} {'Stall Cd':>10}")
    print("─" * (w + 42))
    for cond in all_conds:
        lbl  = _label(cond)
        d    = conditions_out[lbl]
        g_cm = d["global"]["cl_mae"]
        g_dm = d["global"]["cd_mae"]
        s_cm = d["stall"]["cl_mae"]
        s_dm = d["stall"]["cd_mae"]
        row  = (
            f"{lbl:<{w}}"
            f" {g_cm['mean']:>10.5f}" if g_cm else f"{'—':>11}"
        )
        def _fmt(s): return f"{s['mean']:>10.5f}" if s else f"{'—':>10}"
        print(f"{lbl:<{w}} {_fmt(g_cm)} {_fmt(g_dm)} {_fmt(s_cm)} {_fmt(s_dm)}")

    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
