"""NeuralFoil pooled MAE (physical units) on the same strided val/test sample used for FoilCORE.

python tools/neuralfoil_baseline.py --csv data/val.csv --n 2048 --sizes xxsmall,xxxlarge --workers 40
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.data import AeroDataset  # noqa: E402

_SIZES: list[str] = []


def _worker(rec):
    import neuralfoil as nf

    coords = np.asarray(rec["coords"], dtype=float)
    alpha = np.asarray(rec["alpha"], dtype=float)
    out = {}
    kw = {}
    if "mach" in inspect.signature(nf.get_aero_from_coordinates).parameters:
        kw["mach"] = float(rec["mach"])
    for sz in _SIZES:
        try:
            p = nf.get_aero_from_coordinates(
                coordinates=coords, alpha=alpha, Re=np.full_like(alpha, float(rec["Re"])), model_size=sz, **kw
            )
            out[sz] = (np.asarray(p["CL"], float), np.asarray(p["CD"], float))
        except Exception as e:  # noqa: BLE001
            out[sz] = None
    return alpha, np.asarray(rec["Cl"], float), np.asarray(rec["Cd"], float), out


def main():
    global _SIZES
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/val.csv")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--sizes", default="xxsmall,xxxlarge")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stats-ckpt", default="checkpoints/foilcore_best.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    _SIZES = args.sizes.split(",")
    stats = torch.load(args.stats_ckpt, map_location="cpu", weights_only=False)["stats"]
    ds = AeroDataset(args.csv, stats=stats, training=False)
    step = max(1, len(ds) // args.n)
    idx = list(range(0, step * min(args.n, len(ds)), step))
    recs = [ds.df.iloc[i][["coords", "alpha", "Cl", "Cd", "Re", "mach"]].to_dict() for i in idx]
    print(f"NeuralFoil {_SIZES} on {len(recs)} rows of {args.csv} (every {step}th)")
    with Pool(args.workers) as pool:
        res = pool.map(_worker, recs, chunksize=8)
    summ = {}
    for sz in _SIZES:
        e_cl = []; e_cd = []; s_cl = []; s_cd = []; fails = 0
        for alpha, cl, cd, out in res:
            if out.get(sz) is None:
                fails += 1
                continue
            pcl, pcd = out[sz]
            ecl, ecd = np.abs(pcl - cl), np.abs(pcd - cd)
            e_cl.append(ecl); e_cd.append(ecd)
            st = np.abs(alpha) >= 12.0
            s_cl.append(ecl[st]); s_cd.append(ecd[st])
        e_cl = np.concatenate(e_cl); e_cd = np.concatenate(e_cd); s_cl = np.concatenate(s_cl); s_cd = np.concatenate(s_cd)
        summ[sz] = {"cl": float(e_cl.mean()), "cd": float(e_cd.mean()), "stall_cl": float(s_cl.mean()),
                    "stall_cd": float(s_cd.mean()), "points": int(e_cl.size), "failed_rows": fails}
        print(f"  {sz:9s} Cl MAE {summ[sz]['cl']:.4f}  Cd MAE {summ[sz]['cd']:.5f}  | stall(|α|>=12) Cl {summ[sz]['stall_cl']:.4f} Cd {summ[sz]['stall_cd']:.5f}  ({e_cl.size} pts, {fails} failed rows)")
    if args.out:
        json.dump(summ, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
