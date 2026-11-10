#!/usr/bin/env python3
"""Regime table for the physics champion vs NeuralFoil on the SAME strided rows.

Recomputes Table 4 (ts_regime.tex) on data/whisp_dataset/val_disjoint.csv:
  1,024 strided polars; FoilCORE-5k+phys (causal AR) per bin, NeuralFoil
  xxsmall / xxxlarge per bin. Bins: Mach, Re, Mach&Re (NeuralFoil's regime),
  stall/pre-stall, all.

Usage:
  FOILCORE_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 python tools/regime_table.py \
      --ckpt checkpoints/whisp2_combo_seed1/foilcore_best.pt \
      --csv data/whisp_dataset/val_disjoint.csv --n-nf 1024 \
      --out paper/tables/ts_regime.tex
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.benchmark import load_models  # noqa: E402
from src.data import AeroDataset, collate_variable_sequences  # noqa: E402
from src.train_model import prepare_polar_batch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

REGIMES = [
    ("all", None),
    ("Ma $<0.05$", lambda a, r, m: m < 0.05),
    ("Ma $[0.05,0.15)$", lambda a, r, m: (m >= 0.05) & (m < 0.15)),
    ("Ma $[0.15,0.30)$", lambda a, r, m: (m >= 0.15) & (m < 0.30)),
    ("Ma $[0.30,0.50]$", lambda a, r, m: m >= 0.30),
    ("Re $<10^5$", lambda a, r, m: r < 1e5),
    ("Re $[10^5,3{\\times}10^5)$", lambda a, r, m: (r >= 1e5) & (r < 3e5)),
    ("Re $[3{\\times}10^5,10^6)$", lambda a, r, m: (r >= 3e5) & (r < 1e6)),
    ("Re $[10^6,3{\\times}10^6)$", lambda a, r, m: (r >= 1e6) & (r < 3e6)),
    ("Re $\\ge 3{\\times}10^6$", lambda a, r, m: r >= 3e6),
    ("Ma $<0.05$ \\& Re $[10^5,3{\\times}10^6)$ (NeuralFoil's regime)", lambda a, r, m: (m < 0.05) & (r >= 1e5) & (r < 3e6)),
    ("stall $|\\alpha|\\ge 12^\\circ$", lambda a, r, m: a >= 12.0),
    ("pre-stall $|\\alpha|<12^\\circ$", lambda a, r, m: a < 12.0),
]


@torch.no_grad()
def eval_bins(model, stats, ds, rows, device, bs=256):
    dl = DataLoader(Subset(ds, rows), batch_size=bs, shuffle=False, collate_fn=collate_variable_sequences)
    std = torch.as_tensor(stats["polar_std"][3:5], dtype=torch.float32, device=device)
    a_sd, a_mu = stats["polar_std"][0], stats["polar_mean"][0]
    re_sd, re_mu = stats["polar_std"][1], stats["polar_mean"][1]
    m_sd, m_mu = stats["polar_std"][2], stats["polar_mean"][2]
    tot = {name: torch.zeros(2, device=device) for name, _ in REGIMES}
    cnt = {name: 0 for name, _ in REGIMES}
    for b in dl:
        g, p, m = prepare_polar_batch(b, device)
        f, _ = model(g, p, m, causal=True)
        err = (f - p[..., 3:5]).abs() * std
        A = (p[..., 0] * a_sd + a_mu).abs()
        R = p[..., 1] * re_sd + re_mu
        M = p[..., 2] * m_sd + m_mu
        for name, fn in REGIMES:
            s = m if fn is None else (m & fn(A, R, M))
            if s.any():
                tot[name] += err[s].sum(0)
                cnt[name] += int(s.sum())
    return {name: (cnt[name], float(tot[name][0] / max(cnt[name], 1)), float(tot[name][1] / max(cnt[name], 1))) for name, _ in REGIMES}


def neuralfoil_bins(ds, rows, sizes=("xxsmall", "xxxlarge"), workers=6):
    import neuralfoil as nf
    from multiprocessing import Pool
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    recs = [(ds.df.iloc[i][["coords", "alpha", "Cl", "Cd", "Re", "mach"]].to_dict()) for i in rows]
    with Pool(workers) as pool:
        res = pool.starmap(_nf_worker, [(r, sizes) for r in recs], chunksize=4)
    out = {}
    for sz in sizes:
        tot = {name: [0.0, 0.0] for name, _ in REGIMES}
        cnt = {name: 0 for name, _ in REGIMES}
        for alpha, cl, cd, o in res:
            p = o.get(sz)
            if p is None:
                continue
            pcl, pcd = p
            e1, e2 = np.abs(pcl - cl), np.abs(pcd - cd)
            A = np.abs(alpha)
            R = np.full_like(alpha, float(o["_Re"]))
            M = np.full_like(alpha, float(o["_Ma"]))
            for name, fn in REGIMES:
                s = np.ones_like(alpha, bool) if fn is None else fn(A, R, M)
                if s.any():
                    tot[name][0] += e1[s].sum()
                    tot[name][1] += e2[s].sum()
                    cnt[name] += int(s.sum())
        out[sz] = {name: (cnt[name], tot[name][0] / max(cnt[name], 1), tot[name][1] / max(cnt[name], 1)) for name, _ in REGIMES}
    return out


def _nf_worker(rec, sizes):
    import neuralfoil as nf
    coords = np.asarray(rec["coords"], float)
    alpha = np.asarray(rec["alpha"], float)
    out = {"_Re": float(rec["Re"]), "_Ma": float(rec["mach"])}
    for sz in sizes:
        try:
            p = nf.get_aero_from_coordinates(coordinates=coords, alpha=alpha, Re=np.full_like(alpha, float(rec["Re"])), model_size=sz)
            out[sz] = (np.asarray(p["CL"], float), np.asarray(p["CD"], float))
        except Exception:
            out[sz] = None
    return np.asarray(alpha), np.asarray(rec["Cl"], float), np.asarray(rec["Cd"], float), out


def fmt(v):
    if v < 0.01:
        return f"{v:.4f}"
    return f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", default="data/whisp_dataset/val_disjoint.csv")
    ap.add_argument("--n-nf", type=int, default=1024)
    ap.add_argument("--out", default="paper/tables/ts_regime.tex")
    ap.add_argument("--skip-neuralfoil", action="store_true")
    args = ap.parse_args()
    device = torch.device(os.environ.get("FOILCORE_DEVICE", "cpu"))
    model, stats, cfg, _ = load_models(args.ckpt, device)
    model.eval()
    ds = AeroDataset(args.csv, stats=stats, training=False)
    step = max(1, len(ds) // args.n_nf)
    rows = list(range(0, step * args.n_nf, step))
    print(f"eval {args.n_nf} strided rows of {len(ds)}")
    ours = eval_bins(model, stats, ds, rows, device)
    nfres = {} if args.skip_neuralfoil else neuralfoil_bins(ds, rows)
    n_pts = ours["all"][0]
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        f"Regime & pts & FoilCORE-5k+phys & NF xxsmall & NF xxxlarge \\\\",
        "\\midrule",
    ]
    for name, _ in REGIMES:
        if name == "all":
            continue
        n = ours[name][0]
        cells = f"{name} & {n/1000:.0f}k & {fmt(ours[name][1])} / {fmt(ours[name][2])}"
        if nfres:
            for sz in ("xxsmall", "xxxlarge"):
                if nfres[sz][name][0] > 0:
                    cells += f" & {fmt(nfres[sz][name][1])} / {fmt(nfres[sz][name][2])}"
                else:
                    cells += " & --"
        lines.append(cells + " \\\\")
    cells = f"all & {n_pts/1000:.0f}k & {fmt(ours['all'][1])} / {fmt(ours['all'][2])}"
    if nfres:
        for sz in ("xxsmall", "xxxlarge"):
            cells += f" & {fmt(nfres[sz]['all'][1])} / {fmt(nfres[sz]['all'][2])}"
    lines.append(cells + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    text = "\n".join(lines)
    Path(args.out).write_text(text)
    print(f"wrote {args.out}")
    print(text)


if __name__ == "__main__":
    main()
