"""Paper figures from the results folder: training curves, accuracy-vs-params/latency Pareto, knockout bars.

python paper/scripts/make_paper_figures.py --results results/paper_final --runs results/pod_runs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}


def style(ax):
    ax.grid(True, color="#dddddd", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors="#444")


def nf_latency(T=68):
    try:
        import neuralfoil as nf
        x = np.concatenate([np.linspace(1, 0, 100), np.linspace(0, 1, 100)])
        y = np.concatenate([0.06 * np.sin(np.pi * np.linspace(0, 1, 100)), -0.04 * np.sin(np.pi * np.linspace(0, 1, 100))])
        coords = np.stack([x, y], 1); alpha = np.linspace(-15, 15, T); Re = np.full(T, 5e5)
        out = {}
        for sz in ("xxsmall", "xxxlarge"):
            for _ in range(5): nf.get_aero_from_coordinates(coordinates=coords, alpha=alpha, Re=Re, model_size=sz)
            ts = []
            for _ in range(30):
                t = time.perf_counter(); nf.get_aero_from_coordinates(coordinates=coords, alpha=alpha, Re=Re, model_size=sz); ts.append(time.perf_counter() - t)
            out[sz] = float(np.median(ts) * 1e3)
        return out
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--runs", default="results/pod_runs")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(args.results) / "figures"; out.mkdir(parents=True, exist_ok=True)

    # ---------- training curves ----------
    want = {0: ("label-leaking protocol, 1:31", C["red"]), 3: ("draft-only dual-mask, 1:31", C["violet"]), 6: ("draft-only dual-mask, 1:1 (FoilCORE-3k)", C["blue"]),
            54: ("+ error head", C["aqua"]), 60: ("+ profile memory", C["yellow"]), 61: ("+ ê-gated residual head (FoilCORE-5k)", C["orange"]), 74: ("d=16 stack (FoilCORE-2k)", C["green"]), 70: ("d=8 (FoilCORE-0.5k)", C["magenta"])}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for d in sorted(Path(args.runs).glob("*")):
        f = d / "metrics_history.json"
        if not f.is_file() or d.name.startswith("aborted"):
            continue
        h = json.load(open(f)); vid = h.get("variant", {}).get("id")
        if vid not in want:
            continue
        env = d / "env.txt"
        if env.is_file() and ("train-frac" in env.read_text() or "--seed" in env.read_text()):
            continue
        hist = h["history"]; ep = [r["epoch"] for r in hist]
        lab, col = want[vid]
        axes[0].plot(ep, [r["val_ar_cl"] for r in hist], color=col, lw=1.4, label=lab)
        axes[1].plot(ep, [r["val_ar_cd"] for r in hist], color=col, lw=1.4, label=lab)
    for ax, t in zip(axes, ("val causal-AR Cl MAE", "val causal-AR Cd MAE")):
        ax.set_yscale("log"); ax.set_xlabel("epoch", fontsize=9); ax.set_title(t, fontsize=10); style(ax)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=170); plt.close(fig)

    # ---------- Pareto: accuracy vs params and vs latency ----------
    nj = Path(args.results) / "main" / "numbers.json"
    if nj.is_file():
        N = json.load(open(nj))
        pts = []
        for name, e in N["models"].items():
            a = e["accuracy_nf_rows"]["ar"] if "accuracy_nf_rows" in e else e["accuracy"]["ar"]
            fz = e["latency"].get("fused_cpp_b1_us", {})
            pts.append((name, e["n_params"], a["cl"], a["cd"], e["latency"]["cpu_b1_1thread_ms"]["median"], fz.get("full_median")))
        nfl = nf_latency()
        nfp = {"xxsmall": 13302, "xxxlarge": 1428166}
        nfacc = N.get("neuralfoil_test", {})
        PN = {"v61": "FoilCORE-5k", "v74": "FoilCORE-2k", "v6": "FoilCORE-3k", "v70": "FoilCORE-0.5k", "old_protocol": "3k, label-leaking"}
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, key, lab in ((axes[0], 2, "Cl MAE"), (axes[1], 3, "Cd MAE")):
            fam = sorted([p_ for p_ in pts if p_[0] != "old_protocol"], key=lambda r: r[1])
            ax.plot([r[1] for r in fam], [r[key] for r in fam], "-", color=C["blue"], lw=1, alpha=0.6, zorder=2)
            for name, npar, cl, cd, lat_ms, fz in pts:
                v = cl if key == 2 else cd
                ax.scatter(npar, v, s=40, color=C["blue"] if name != "old_protocol" else C["red"], zorder=3)
                ax.annotate(PN.get(name, name), (npar, v), fontsize=7.5, xytext=(5, 4), textcoords="offset points")
            for sz, acc in nfacc.items():
                if not isinstance(acc, dict): continue
                v = acc["cl"] if key == 2 else acc["cd"]
                ax.scatter(nfp[sz], v, s=40, color=C["orange"], marker="s", zorder=3)
                ax.annotate(f"NeuralFoil {sz}", (nfp[sz], v), fontsize=7.5, xytext=(4, -10), textcoords="offset points")
            ax.set_xscale("log"); ax.set_xlabel("parameters", fontsize=9); ax.set_ylabel(lab + " (held-out test, causal AR)", fontsize=9); style(ax)
        fig.suptitle("Accuracy vs. size — FoilCORE family (circles) vs NeuralFoil (squares)", fontsize=10)
        fig.tight_layout(); fig.savefig(out / "pareto_params.png", dpi=170); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.2, 4))
        for name, npar, cl, cd, lat_ms, fz in pts:
            ax.scatter(lat_ms, cl, s=40, color=C["blue"], zorder=3); ax.annotate(f"{PN.get(name,name)} (eager)", (lat_ms, cl), fontsize=7, xytext=(4, 4), textcoords="offset points")
            if fz:
                ax.scatter(fz / 1e3, cl, s=40, color=C["green"], zorder=3); ax.annotate(f"{PN.get(name,name)} (fused C++)", (fz / 1e3, cl), fontsize=7, xytext=(4, -9), textcoords="offset points")
        for sz, acc in nfacc.items():
            if isinstance(acc, dict) and sz in nfl:
                ax.scatter(nfl[sz], acc["cl"], s=40, color=C["orange"], marker="s", zorder=3); ax.annotate(f"NeuralFoil {sz}", (nfl[sz], acc["cl"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel("B=1 CPU latency per polar [ms], single thread", fontsize=9); ax.set_ylabel("Cl MAE (test, causal AR)", fontsize=9); style(ax)
        fig.tight_layout(); fig.savefig(out / "pareto_latency.png", dpi=170); plt.close(fig)
        json.dump({"neuralfoil_b1_latency_ms": nfl}, open(out / "nf_latency.json", "w"))

    # ---------- knockout bars ----------
    for name in ("v61", "v6"):
        kj = Path(args.results) / "main" / f"knockouts_{name}" / "analysis.json"
        if not kj.is_file():
            continue
        ko = json.load(open(kj))["knockouts"]; b = ko["baseline_causal"]
        items = [(k, v["cl"] - b["cl"], v["cd"] - b["cd"]) for k, v in ko.items() if not k.startswith("baseline") and not k.endswith("_knockouts")]
        items.sort(key=lambda r: r[1])
        fig, axes = plt.subplots(1, 2, figsize=(10, 0.28 * len(items) + 1.5))
        ys = np.arange(len(items))
        axes[0].barh(ys, [r[1] for r in items], color=C["blue"]); axes[0].set_yticks(ys); axes[0].set_yticklabels([r[0] for r in items], fontsize=7)
        axes[0].set_xscale("symlog", linthresh=0.01); axes[0].set_title(f"{name}: ΔCl MAE when removed (baseline {b['cl']:.4f})", fontsize=9)
        axes[1].barh(ys, [r[2] for r in items], color=C["orange"]); axes[1].set_yticks(ys); axes[1].set_yticklabels([]); axes[1].set_xscale("symlog", linthresh=0.001)
        axes[1].set_title(f"ΔCd MAE (baseline {b['cd']:.5f})", fontsize=9)
        for ax in axes: style(ax)
        fig.tight_layout(); fig.savefig(out / f"knockouts_{name}.png", dpi=170); plt.close(fig)
    # ---------- error-head reliability + OOD ----------
    xp = Path(args.results) / "v61_extras" / "v61_extras.json"
    if xp.is_file():
        X = json.load(open(xp)); cal = X["error_head_calibration"]; ood = X["ood_error_head"]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        for ax, key, lab, col in ((axes[0], "cl", "$C_l$", C["blue"]), (axes[1], "cd", "$C_d$", C["orange"])):
            d = cal[key]["deciles_physical"]
            x = [r["pred_mean"] for r in d]; y = [r["true_mean"] for r in d]
            lim = [0, max(max(x), max(y)) * 1.1]
            ax.plot(lim, lim, color="#bbbbbb", lw=1)
            ax.plot(x, y, "o-", color=col, ms=5, lw=1.4)
            ax.set_xlabel(f"predicted |error| {lab} (decile mean)", fontsize=8.5); ax.set_ylabel(f"true draft |error| {lab}", fontsize=8.5)
            ax.set_title(f"{lab}: Spearman {cal[key]['spearman']:.2f}, P@10% {cal[key]['precision_at_10pct_worst']:.2f}", fontsize=9)
            style(ax)
        labs = ["clean", "noise σ=0.01", "noise σ=0.03", "random cloud"]; keys = ["clean", "noise_0.01", "noise_0.03", "random_cloud"]
        vals = [ood[k]["ehat_cd_mean_phys"] for k in keys]
        axes[2].bar(range(4), vals, color=[C["blue"], C["blue"], C["blue"], C["red"]])
        axes[2].set_xticks(range(4)); axes[2].set_xticklabels(labs, fontsize=8, rotation=15)
        axes[2].set_title("error head $\\hat e_{C_d}$ (mean) — OOD response", fontsize=9); style(axes[2])
        for i, v in enumerate(vals): axes[2].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout(); fig.savefig(out / "error_head.png", dpi=170); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
