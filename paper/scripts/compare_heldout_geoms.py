"""FoilCORE (causal AR) vs NeuralFoil on N held-out test geometries, one polar each, with figures.

python paper/scripts/compare_heldout_geoms.py --ckpt checkpoints/v6/foilcore_best.pt --csv data/test.csv \
    --n-geoms 50 --out paper/figures/heldout50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.benchmark import load_models  # noqa: E402
from src.data import AeroDataset, collate_variable_sequences  # noqa: E402
from src.train_model import prepare_polar_batch  # noqa: E402

# categorical slots (dataviz palette): 1 blue, 2 orange, 3 aqua; truth in ink gray
C_FOIL, C_NFXXXL, C_NFXXS, C_TRUTH = "#2a78d6", "#eb6834", "#1baf7a", "#3a3a3a"


def pick_rows(df, n_geoms: int, re_target: float, seed: int):
    keys = df["coords"].map(lambda c: hashlib.md5(np.asarray(c, dtype=np.float64).tobytes()).hexdigest())
    df = df.assign(_gkey=keys.values)
    groups = list(df.groupby("_gkey").indices.items())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))[:n_geoms]
    rows = []
    for gi in order:
        _, idxs = groups[gi]
        sub = df.iloc[idxs]
        j = int(np.argmin(np.abs(np.log(sub["Re"].values.astype(float)) - np.log(re_target))))
        rows.append(int(idxs[j]))
    return rows


@torch.no_grad()
def foilcore_predict(model, stats, ds, rows, device):
    from torch.utils.data import DataLoader, Subset

    dl = DataLoader(Subset(ds, rows), batch_size=len(rows), shuffle=False, collate_fn=collate_variable_sequences)
    g, p, m = prepare_polar_batch(next(iter(dl)), device)
    t = time.perf_counter()
    f, _ = model(g, p, m, causal=True)
    dt = (time.perf_counter() - t) / len(rows) * 1e3
    mean = torch.as_tensor(stats["polar_mean"][3:5], dtype=torch.float32)
    std = torch.as_tensor(stats["polar_std"][3:5], dtype=torch.float32)
    phys = f * std + mean
    out = []
    for i in range(len(rows)):
        n = int(m[i].sum())
        out.append((phys[i, :n, 0].numpy(), phys[i, :n, 1].numpy()))
    return out, dt


def neuralfoil_predict(recs, size):
    import neuralfoil as nf

    out = []
    t = time.perf_counter()
    for r in recs:
        alpha = np.asarray(r["alpha"], float)
        p = nf.get_aero_from_coordinates(
            coordinates=np.asarray(r["coords"], float), alpha=alpha,
            Re=np.full_like(alpha, float(r["Re"])), model_size=size,
        )
        out.append((np.asarray(p["CL"], float), np.asarray(p["CD"], float)))
    return out, (time.perf_counter() - t) / len(recs) * 1e3


def metrics(preds, recs):
    per = []
    ecl = []; ecd = []; scl = []; scd = []
    for (pcl, pcd), r in zip(preds, recs):
        cl, cd, al = np.asarray(r["Cl"], float), np.asarray(r["Cd"], float), np.asarray(r["alpha"], float)
        e1, e2 = np.abs(pcl - cl), np.abs(pcd - cd)
        st = np.abs(al) >= 12.0
        per.append({"cl": float(e1.mean()), "cd": float(e2.mean()),
                    "stall_cl": float(e1[st].mean()) if st.any() else float("nan"),
                    "stall_cd": float(e2[st].mean()) if st.any() else float("nan")})
        ecl.append(e1); ecd.append(e2); scl.append(e1[st]); scd.append(e2[st])
    ecl, ecd, scl, scd = map(np.concatenate, (ecl, ecd, scl, scd))
    pooled = {"cl": float(ecl.mean()), "cd": float(ecd.mean()), "stall_cl": float(scl.mean()), "stall_cd": float(scd.mean()),
              "median_geom_cl": float(np.median([p["cl"] for p in per])), "median_geom_cd": float(np.median([p["cd"] for p in per]))}
    return pooled, per


def style(ax):
    ax.grid(True, color="#d8d8d8", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9a9a")
    ax.tick_params(colors="#555555", labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", default="data/test.csv")
    ap.add_argument("--n-geoms", type=int, default=50)
    ap.add_argument("--re-target", type=float, default=5e5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="paper/figures/heldout50")
    ap.add_argument("--n-plot", type=int, default=12)
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cpu")
    torch.set_num_threads(1)
    model, stats, cfg, _ = load_models(args.ckpt, device)
    ds = AeroDataset(args.csv, stats=stats, training=False)
    rows = pick_rows(ds.df, args.n_geoms, args.re_target, args.seed)
    recs = [ds.df.iloc[i][["airfoil", "coords", "alpha", "Cl", "Cd", "Re", "mach"]].to_dict() for i in rows]
    print(f"{len(rows)} held-out geometries from {args.csv}; Re nearest {args.re_target:g}; "
          f"Re range {min(r['Re'] for r in recs):.3g}..{max(r['Re'] for r in recs):.3g}, Mach {min(r['mach'] for r in recs):.2f}..{max(r['mach'] for r in recs):.2f}")

    preds = {}
    lat = {}
    preds["FoilCORE (AR)"], lat["FoilCORE (AR)"] = foilcore_predict(model, stats, ds, rows, device)
    for sz, name in (("xxxlarge", "NeuralFoil xxxlarge"), ("xxsmall", "NeuralFoil xxsmall")):
        preds[name], lat[name] = neuralfoil_predict(recs, sz)
    results = {}
    for name, pr in preds.items():
        pooled, per = metrics(pr, recs)
        results[name] = {"pooled": pooled, "per_geom": per, "ms_per_polar_batchcpu": lat[name]}
        print(f"  {name:22s} Cl MAE {pooled['cl']:.4f}  Cd MAE {pooled['cd']:.5f} | stall Cl {pooled['stall_cl']:.4f} Cd {pooled['stall_cd']:.5f} | "
              f"median-geom Cl {pooled['median_geom_cl']:.4f} Cd {pooled['median_geom_cd']:.5f}")
    n_params = sum(p.numel() for p in model.parameters())
    meta = {"ckpt": args.ckpt, "n_params": n_params, "csv": args.csv, "n_geoms": len(rows), "rows": rows,
            "airfoils": [r["airfoil"] for r in recs], "Re": [float(r["Re"]) for r in recs], "mach": [float(r["mach"]) for r in recs],
            "note": "NeuralFoil has no Mach input; polars here span the dataset's Mach range. FoilCORE run in causal AR mode.",
            "results": results}
    json.dump(meta, open(os.path.join(args.out, "results.json"), "w"), indent=1)

    colors = {"FoilCORE (AR)": C_FOIL, "NeuralFoil xxxlarge": C_NFXXXL, "NeuralFoil xxsmall": C_NFXXS}
    names = list(colors)

    # ---- Fig 1: small multiples of polars (Cl and Cd vs α) ----
    k = min(args.n_plot, len(recs))
    ncol = 4
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow * 2, ncol, figsize=(3.2 * ncol, 2.2 * nrow * 2), squeeze=False)
    for j in range(k):
        r = recs[j]
        al = np.asarray(r["alpha"], float)
        rr, cc = (j // ncol) * 2, j % ncol
        ax_cl, ax_cd = axes[rr, cc], axes[rr + 1, cc]
        ax_cl.plot(al, r["Cl"], ".", color=C_TRUTH, ms=3, label="XFOIL")
        ax_cd.plot(al, r["Cd"], ".", color=C_TRUTH, ms=3, label="XFOIL")
        for name in names:
            pcl, pcd = preds[name][j]
            ax_cl.plot(al, pcl, "-", color=colors[name], lw=1.4, label=name)
            ax_cd.plot(al, pcd, "-", color=colors[name], lw=1.4, label=name)
        ax_cl.set_title(f"{r['airfoil']}  Re={r['Re']:.2g} M={r['mach']:.2f}", fontsize=8, color="#333")
        ax_cd.set_yscale("log")
        for ax in (ax_cl, ax_cd):
            style(ax)
        ax_cl.set_ylabel("Cl", fontsize=8, color="#555")
        ax_cd.set_ylabel("Cd", fontsize=8, color="#555")
        ax_cd.set_xlabel("α [deg]", fontsize=8, color="#555")
    for j in range(k, nrow * ncol):
        axes[(j // ncol) * 2, j % ncol].axis("off"); axes[(j // ncol) * 2 + 1, j % ncol].axis("off")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Held-out test geometries — FoilCORE {n_params} params (causal AR) vs NeuralFoil", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "polars_small_multiples.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: per-geometry MAE, FoilCORE vs NeuralFoil xxxlarge (Cl, Cd) ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, key, lab in ((axes[0], "cl", "Cl MAE per airfoil"), (axes[1], "cd", "Cd MAE per airfoil")):
        x = np.array([p[key] for p in results["NeuralFoil xxxlarge"]["per_geom"]])
        y = np.array([p[key] for p in results["FoilCORE (AR)"]["per_geom"]])
        lim = [min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.2]
        ax.plot(lim, lim, "-", color="#bbbbbb", lw=1)
        ax.scatter(x, y, s=22, color=C_FOIL, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("NeuralFoil xxxlarge (1.5M params)", fontsize=9, color="#555")
        ax.set_ylabel(f"FoilCORE ({n_params} params, AR)", fontsize=9, color="#555")
        wins = int((y < x).sum())
        ax.set_title(f"{lab} — FoilCORE better on {wins}/{len(x)}", fontsize=10, color="#333")
        style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "per_geom_mae_scatter.png"), dpi=170)
    plt.close(fig)

    # ---- Fig 3: pooled MAE bars (overall and stall) ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, lab in ((axes[0], "cl", "Cl MAE"), (axes[1], "cd", "Cd MAE")):
        vals = [results[n]["pooled"][key] for n in names]
        svals = [results[n]["pooled"]["stall_" + key] for n in names]
        xs = np.arange(len(names))
        ax.bar(xs - 0.2, vals, 0.36, color=[colors[n] for n in names], label="all α")
        ax.bar(xs + 0.2, svals, 0.36, color=[colors[n] for n in names], alpha=0.45, hatch="///", edgecolor="white", label="|α| ≥ 12°")
        for i, (v, s) in enumerate(zip(vals, svals)):
            ax.text(i - 0.2, v, f"{v:.4f}" if key == "cl" else f"{v:.5f}", ha="center", va="bottom", fontsize=7.5, color="#333")
            ax.text(i + 0.2, s, f"{s:.4f}" if key == "cl" else f"{s:.5f}", ha="center", va="bottom", fontsize=7.5, color="#333")
        ax.set_xticks(xs); ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
        ax.set_title(f"{lab} — {len(recs)} held-out geometries", fontsize=10, color="#333")
        ax.legend(fontsize=8, frameon=False)
        style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "pooled_mae_bars.png"), dpi=170)
    plt.close(fig)
    print(f"wrote {args.out}/{{polars_small_multiples,per_geom_mae_scatter,pooled_mae_bars}}.png and results.json")


if __name__ == "__main__":
    main()
