"""Aggregate runs/transfer/*.json → markdown table + summary.json (mean ± std over seeds).

python experiments/transfer/summarize.py runs/transfer
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/transfer")
rows = [r for r in (json.load(open(p)) for p in sorted(root.glob("*.json")) if p.name != "summary.json") if "dataset" in r]
agg = defaultdict(list)
for r in rows:
    agg[(r["dataset"], r["protocol"], r.get("corrector", "rpa"))].append(r)
out = {}
print("| dataset | protocol | corrector | n | params | test MAE causal | test MAE bidir | gap % | draft MAE |")
print("|---|---|---|---|---|---|---|---|---|")
for (ds, proto, corr), rs in sorted(agg.items()):
    c = np.array([r["test_causal_mae"] for r in rs]); b = np.array([r["test_bidir_mae"] for r in rs])
    d = np.array([r["test_draft_mae"] for r in rs]); g = np.array([r["gap_pct"] for r in rs])
    key = f"{ds}/{proto}" if corr == "rpa" else f"{ds}/{proto}/{corr}"
    out[key] = {"n": len(rs), "corrector": corr, "params": rs[0]["n_params"], "unit": rs[0]["unit"], "name": rs[0]["dataset_name"],
                            "causal_mean": float(c.mean()), "causal_std": float(c.std()), "bidir_mean": float(b.mean()),
                            "bidir_std": float(b.std()), "gap_mean": float(g.mean()), "draft_mean": float(d.mean())}
    print(f"| {rs[0]['dataset_name']} ({rs[0]['unit']}) | {proto} | {corr} | {len(rs)} | {rs[0]['n_params']} | "
          f"{c.mean():.4f} ± {c.std():.4f} | {b.mean():.4f} ± {b.std():.4f} | {g.mean():+.1f} | {d.mean():.4f} |")
json.dump(out, open(root / "summary.json", "w"), indent=1)
