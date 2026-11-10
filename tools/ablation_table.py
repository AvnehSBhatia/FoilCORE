"""Ablation/variant table from runs/*/metrics_history.json (best causal-AR val epoch per run).

python tools/ablation_table.py --root runs --out results/ablations.md
Reads locally or from a directory synced from the pod. Reports epochs reached so unequal budgets are explicit.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DESCR = {
    0: "FoilCORE 3.4k, original protocol (GT Cl/Cd mixed in, 1:31 loss)",
    3: "FoilCORE 3.4k, AR protocol (draft-only, causal/bidir mix), 1:31 loss",
    6: "FoilCORE 3.4k, AR protocol, 1:1 loss",
    50: "FoilCORE-v2 (SSM blocks + router + err head), 1:31",
    51: "v2 without router, 1:31",
    52: "v2 depth 3, 1:31",
    53: "v2, 1:1",
    54: "FoilCORE + error head, 1:1",
    55: "v2 no router, 1:1",
    60: "FoilCORE + err head + profile memory, 1:1",
    61: "FoilCORE-5k: + err head + profile + ê-gated residual head, 1:1",
    62: "soft-bin pooling + profile + err head, 1:1",
    63: "soft-bin + profile + err head + expert, 1:1",
    70: "FoilCORE d=8, 1:1",
    71: "FoilCORE d=8, MoE-4 corrector, 1:1",
    72: "FoilCORE d=16, MoE-4, 1:1",
    73: "d=16 soft-bin + profile + err head + MoE-4, 1:1",
    74: "FoilCORE-2k: full stack at d=16",
    42: "FoilSCAN v2 (contour SSM)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for d in sorted(Path(args.root).glob("*")):
        f = d / "metrics_history.json"
        if not f.is_file():
            continue
        try:
            h = json.load(open(f))
        except json.JSONDecodeError:
            continue
        hist = h.get("history", [])
        if not hist:
            continue
        vid = h.get("variant", {}).get("id")
        if len(hist) < 20 or d.name.startswith("deferred"):
            continue
        best = min(hist, key=lambda r: r["val_ar_loss"])
        last = hist[-1]
        # matched-budget snapshot at epoch 50 and 100 if available
        at = {e: next((r for r in hist if r["epoch"] == e), None) for e in (50, 100)}
        tag = ""
        env = d / "env.txt"
        if env.is_file():
            e = env.read_text()
            m1 = re.search(r"train-frac ([0-9.]+)", e); m2 = re.search(r"--seed (\d+)", e)
            if m1: tag = f" ({int(float(m1.group(1))*100)}\\% of geometries)"
            elif m2: tag = f" (seed {m2.group(1)})"
        rows.append((vid, d.name, h.get("n_params"), last["epoch"], best, at, tag))
    rows.sort(key=lambda r: (r[0] if r[0] is not None else 999, r[1]))
    md = ["| variant | description | params | epochs | best AR Cl | best AR Cd | best stall Cd | best TF Cl | AR Cl @50 | AR Cl @100 | run |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for vid, name, npar, ep, b, at, tag in rows:
        c50 = f"{at[50]['val_ar_cl']:.4f}" if at[50] else "–"
        c100 = f"{at[100]['val_ar_cl']:.4f}" if at[100] else "–"
        md.append(f"| {vid} | {DESCR.get(vid, '')}{tag} | {npar} | {ep} | {b['val_ar_cl']:.4f} | {b['val_ar_cd']:.5f} | {b['val_ar_stall_cd']:.5f} | {b['val_tf_cl']:.4f} | {c50} | {c100} | {name} |")
    text = "\n".join(md)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
