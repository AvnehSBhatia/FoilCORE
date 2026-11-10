#!/usr/bin/env python3
"""Pick a random val airfoil and compare FoilCORE causal AR vs NeuralFoil vs XFOIL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neuralfoil as nf
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data import normalize_coords, resample_airfoil
from src.inference import FoilCOREGenerator
from src.train_model import get_device


def _nf(coords, alpha, re, size: str):
    p = nf.get_aero_from_coordinates(
        coordinates=np.asarray(coords, dtype=float),
        alpha=np.asarray(alpha, dtype=float),
        Re=np.full_like(np.asarray(alpha, dtype=float), float(re)),
        model_size=size,
    )
    cl = np.asarray(p["CL"], dtype=float)
    cd = np.asarray(p["CD"], dtype=float)
    if not (np.all(np.isfinite(cl)) and np.all(np.isfinite(cd))):
        raise RuntimeError(f"NeuralFoil {size} returned non-finite values")
    return cl, cd


def _mae(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/foilcore_best.pt")
    ap.add_argument("--csv", default="data/val.csv")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--min-polar", type=int, default=16)
    ap.add_argument("--outdir", default="results/autopsy_3k/val_ar_example")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.csv)
    df["alpha"] = df["alpha"].map(json.loads)
    npts = df["alpha"].map(len)
    pool = df.index[npts >= args.min_polar].to_numpy()
    if len(pool) == 0:
        raise SystemExit("no val rows with enough polar points")
    idx = int(rng.choice(pool))
    row = df.iloc[idx]
    alpha = np.asarray(row["alpha"], dtype=float)
    cl_gt = np.asarray(json.loads(row["Cl"]) if isinstance(row["Cl"], str) else row["Cl"], dtype=float)
    cd_gt = np.asarray(json.loads(row["Cd"]) if isinstance(row["Cd"], str) else row["Cd"], dtype=float)
    coords = np.asarray(json.loads(row["coords"]) if isinstance(row["coords"], str) else row["coords"], dtype=float)
    re = float(row["Re"])
    mach = float(row["mach"])
    name = str(row["airfoil"])
    src = str(row["datasource"])

    coords_n, _, _ = normalize_coords(resample_airfoil(coords, n_points=200))

    device = torch.device(get_device())
    gen = FoilCOREGenerator(args.ckpt, device=device)
    ar = gen.predict(coords, alpha, re, mach, causal=True, seed_len=0)
    tf = gen.predict(coords, alpha, re, mach, causal=False, seed_len=0)
    cl_ar = ar["Cl"].numpy()
    cd_ar = ar["Cd"].numpy()
    cl_tf = tf["Cl"].numpy()
    cd_tf = tf["Cd"].numpy()
    cl_draft = ar["draft_Cl"].numpy()
    cd_draft = ar["draft_Cd"].numpy()

    cl_xs, cd_xs = _nf(coords, alpha, re, "xxsmall")
    cl_xl, cd_xl = _nf(coords, alpha, re, "xxxlarge")

    payload = {
        "csv": args.csv,
        "row_index": idx,
        "airfoil": name,
        "datasource": src,
        "Re": re,
        "mach": mach,
        "n_polar": int(len(alpha)),
        "alpha_range": [float(alpha.min()), float(alpha.max())],
        "seed": args.seed,
        "mode": "FoilCORE causal AR, seed_len=0 (draft Cl/Cd, causal corrector; GT α/Re/Mach)",
        "coords": coords_n.tolist(),
        "alpha": [round(float(x), 4) for x in alpha],
        "gt": {"Cl": [round(float(x), 5) for x in cl_gt], "Cd": [round(float(x), 6) for x in cd_gt]},
        "foilcore_ar": {"Cl": [round(float(x), 5) for x in cl_ar], "Cd": [round(float(x), 6) for x in cd_ar]},
        "foilcore_tf": {"Cl": [round(float(x), 5) for x in cl_tf], "Cd": [round(float(x), 6) for x in cd_tf]},
        "foilcore_draft": {"Cl": [round(float(x), 5) for x in cl_draft], "Cd": [round(float(x), 6) for x in cd_draft]},
        "neuralfoil_xxsmall": {"Cl": [round(float(x), 5) for x in cl_xs], "Cd": [round(float(x), 6) for x in cd_xs]},
        "neuralfoil_xxxlarge": {"Cl": [round(float(x), 5) for x in cl_xl], "Cd": [round(float(x), 6) for x in cd_xl]},
        "mae_vs_xfoil": {
            "foilcore_ar": {"Cl": _mae(cl_ar, cl_gt), "Cd": _mae(cd_ar, cd_gt)},
            "foilcore_tf": {"Cl": _mae(cl_tf, cl_gt), "Cd": _mae(cd_tf, cd_gt)},
            "neuralfoil_xxsmall": {"Cl": _mae(cl_xs, cl_gt), "Cd": _mae(cd_xs, cd_gt)},
            "neuralfoil_xxxlarge": {"Cl": _mae(cl_xl, cl_gt), "Cd": _mae(cd_xl, cd_gt)},
        },
        "mae_ar_vs_nf_xxxlarge": {"Cl": _mae(cl_ar, cl_xl), "Cd": _mae(cd_ar, cd_xl)},
    }
    with open(outdir / "sample.json", "w") as f:
        json.dump(payload, f, indent=2)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), dpi=160, gridspec_kw={"width_ratios": [1.05, 1.35, 1.35]})

    ax = axes[0]
    ax.plot(coords_n[:, 0], coords_n[:, 1], color="#1f3a5f", lw=1.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title(f"{name}\nRe={re:.0f}, M={mach:g}")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(alpha, cl_gt, color="#222", lw=1.8, label="XFOIL ncrit=9")
    ax.plot(alpha, cl_ar, color="#2b6cb0", lw=1.7, label="FoilCORE AR")
    ax.plot(alpha, cl_tf, color="#2b6cb0", lw=1.1, ls="--", alpha=0.7, label="FoilCORE TF")
    ax.plot(alpha, cl_xl, color="#c0392b", lw=1.4, ls=":", label="NeuralFoil xxxlarge")
    ax.plot(alpha, cl_xs, color="#e67e22", lw=1.2, ls=":", label="NeuralFoil xxsmall")
    ax.set_xlabel("α (deg)")
    ax.set_ylabel("Cl")
    ax.set_title("Lift")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(alpha, cd_gt, color="#222", lw=1.8, label="XFOIL ncrit=9")
    ax.plot(alpha, cd_ar, color="#2b6cb0", lw=1.7, label="FoilCORE AR")
    ax.plot(alpha, cd_tf, color="#2b6cb0", lw=1.1, ls="--", alpha=0.7, label="FoilCORE TF")
    ax.plot(alpha, cd_xl, color="#c0392b", lw=1.4, ls=":", label="NeuralFoil xxxlarge")
    ax.plot(alpha, cd_xs, color="#e67e22", lw=1.2, ls=":", label="NeuralFoil xxsmall")
    ax.set_xlabel("α (deg)")
    ax.set_ylabel("Cd")
    ax.set_title("Drag")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "polar_compare.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({k: payload[k] for k in ("airfoil", "row_index", "Re", "mach", "n_polar", "alpha_range", "mae_vs_xfoil", "mae_ar_vs_nf_xxxlarge")}, indent=2))
    print(f"wrote {outdir.resolve()}")


if __name__ == "__main__":
    main()
