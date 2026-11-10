"""Covariate-matched classical reference points for the ETTh1 lag-1 task: ridge regression and gradient boosting
on exactly the per-hour features the 4k model sees (loads, calendar, season, OT_{t-1}), same held-out block split.

python experiments/transfer/ridge_gbm_ett.py --out results/transfer/ridge_gbm_ett.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
from experiments.transfer.datasets import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/transfer/ridge_gbm_ett.json"); args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    out = {}
    for key in ("ett", "ett_lag1"):
        b = load(key)
        Xtr = b["x_train"].reshape(-1, b["x_train"].shape[-1]); ytr = b["y_train"].reshape(-1)
        Xte = b["x_test"].reshape(-1, b["x_test"].shape[-1]); yte = b["y_test"].reshape(-1)
        sd = float(b["y_std"])
        res = {}
        for name, mdl in (("ridge", Ridge(alpha=1.0)), ("hist_gbm", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0))):
            mdl.fit(Xtr, ytr)
            res[name] = float(np.mean(np.abs(mdl.predict(Xte) - yte)) * sd)
        out[key] = {"name": str(b["name"]), "n_train_points": int(len(ytr)), "n_test_points": int(len(yte)), "test_mae_degC": res}
        print(key, res)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
