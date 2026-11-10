"""Tabulate all runs under runs/ (latest epoch + best causal-AR epoch per run).

python tools/runs_status.py [--root runs]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    args = ap.parse_args()
    rows = []
    for d in sorted(Path(args.root).glob("*")):
        mh = d / "metrics_history.json"
        if not mh.is_file():
            continue
        try:
            h = json.load(open(mh))
        except json.JSONDecodeError:
            continue
        hist = h.get("history", [])
        if not hist:
            continue
        last = hist[-1]
        best = min(hist, key=lambda r: r["val_ar_loss"])
        rows.append((d.name, h.get("variant", {}).get("id"), h.get("n_params"), last, best))
    print(f"{'run':28s} {'var':>3s} {'params':>6s} {'ep':>3s} {'AR Cl':>7s} {'AR Cd':>8s} {'stallCd':>8s} {'TF Cl':>7s} {'TF Cd':>8s} {'best ep':>7s} {'best AR Cl':>10s} {'best AR Cd':>10s} {'s/ep':>5s}")
    for name, vid, npar, last, best in rows:
        print(
            f"{name:28s} {vid!s:>3s} {npar!s:>6s} {last['epoch']:3d} {last['val_ar_cl']:7.4f} {last['val_ar_cd']:8.5f} "
            f"{last['val_ar_stall_cd']:8.5f} {last['val_tf_cl']:7.4f} {last['val_tf_cd']:8.5f} {best['epoch']:7d} "
            f"{best['val_ar_cl']:10.4f} {best['val_ar_cd']:10.5f} {last['train_time_s']+last['val_time_s']:5.0f}"
        )


if __name__ == "__main__":
    main()
