"""Extra analyses for the error-head / profile / expert model (v61-style FoilCORE).

* error head calibration on held-out test (causal AR): correlation, per-decile calibration, hit-rate on large errors
* OOD: ê on real airfoils vs geometry-noise levels vs a pure random cloud
* expert gate: mean gate and |Δ| by α / Re regime
* profile memory and other memories: knockouts in causal mode
python paper/scripts/analyze_v61_extras.py --ckpt checkpoints/v61/foilcore_best.pt --out results/paper_final/v61_extras
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.benchmark import load_models  # noqa: E402
from src.data import AeroDataset, collate_variable_sequences  # noqa: E402
from src.train_model import prepare_polar_batch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", default="data/test.csv")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cpu")
    model, stats, cfg, _ = load_models(args.ckpt, dev)
    model.eval()
    ds = AeroDataset(args.csv, stats=stats, training=False)
    step = len(ds) // args.n
    rows = list(range(0, step * args.n, step))
    dl = DataLoader(Subset(ds, rows), batch_size=512, shuffle=False, collate_fn=collate_variable_sequences)
    std = torch.as_tensor(stats["polar_std"][3:5], dtype=torch.float32)
    a_sd, a_mu = stats["polar_std"][0], stats["polar_mean"][0]
    re_sd, re_mu = stats["polar_std"][1], stats["polar_mean"][1]
    R = {"ckpt": args.ckpt, "n_rows": len(rows)}

    E, EH, G, DL, AL, RE, DR = [], [], [], [], [], [], []
    for b in dl:
        g, p, m = prepare_polar_batch(b, dev)
        f, d = model(g, p, m, causal=True)
        aux = model.aux
        E.append(((f - p[..., 3:5]).abs())[m])              # final error, z-units
        DR.append(((d - p[..., 3:5]).abs())[m])             # draft error, z-units (what ê is trained on)
        EH.append(aux["err_pred"][m])
        if "expert_gate" in aux:
            G.append(aux["expert_gate"][m])
            # expert delta magnitude: recompute
            delta, _ = model.err_expert(_hidden(model, g, p, m), aux["err_pred"])
            DL.append(delta[m])
        AL.append((p[..., 0] * a_sd + a_mu)[m]); RE.append((p[..., 1] * re_sd + re_mu)[m])
    E, EH, DR, AL, RE = map(torch.cat, (E, EH, DR, AL, RE))
    # ---- error head calibration ----
    cal = {}
    for j, nm in enumerate(("cl", "cd")):
        x, y = EH[:, j], DR[:, j]
        corr = float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std() + 1e-9))
        # spearman via ranks
        rx, ry = x.argsort().argsort().float(), y.argsort().argsort().float()
        spear = float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (rx.std() * ry.std() + 1e-9))
        q = torch.quantile(x, torch.linspace(0, 1, 11))
        dec = []
        for i in range(10):
            s = (x >= q[i]) & (x <= q[i + 1])
            dec.append({"pred_mean": float(x[s].mean() * std[j]), "true_mean": float(y[s].mean() * std[j]), "n": int(s.sum())})
        # does ê rank the worst 10% points? precision@10%
        k = int(0.1 * len(x))
        top_pred = set(x.topk(k).indices.tolist()); top_true = set(y.topk(k).indices.tolist())
        cal[nm] = {"pearson_vs_draft_err": corr, "spearman": spear, "deciles_physical": dec,
                   "precision_at_10pct_worst": len(top_pred & top_true) / k,
                   "corr_vs_final_err": float((((x - x.mean()) * (E[:, j] - E[:, j].mean())).mean() / (x.std() * E[:, j].std() + 1e-9)))}
    R["error_head_calibration"] = cal
    # ---- expert gate by regime ----
    if G:
        G = torch.cat(G); DL = torch.cat(DL)
        stall = AL.abs() >= 12
        R["expert"] = {
            "gate_mean": float(G.mean()), "gate_frac_gt_0.5": float((G > 0.5).float().mean()),
            "gate_mean_stall": float(G[stall].mean()), "gate_mean_linear": float(G[~stall].mean()),
            "delta_abs_mean_phys": (DL.abs().mean(0) * std).tolist(),
            "delta_abs_mean_phys_stall": (DL[stall].abs().mean(0) * std).tolist(),
            "gate_by_re": [{"lo": lo, "hi": hi, "gate": float(G[(RE >= lo) & (RE < hi)].mean())} for lo, hi in ((4e4, 1e5), (1e5, 3e5), (3e5, 1e6), (1e6, 3e6), (3e6, 2.1e7))],
        }
    # ---- OOD: ê on real vs noisy vs random cloud ----
    b = next(iter(DataLoader(Subset(ds, rows[:256]), batch_size=256, shuffle=False, collate_fn=collate_variable_sequences)))
    g, p, m = prepare_polar_batch(b, dev)
    ood = {}
    torch.manual_seed(0)
    for lab, gg in (("clean", g), ("noise_0.01", g + torch.cat((torch.zeros_like(g[..., :1]), 0.01 * torch.randn_like(g[..., 1:])), -1)),
                    ("noise_0.03", g + torch.cat((torch.zeros_like(g[..., :1]), 0.03 * torch.randn_like(g[..., 1:])), -1)),
                    ("random_cloud", torch.stack((torch.rand_like(g[..., 0]), (torch.rand_like(g[..., 1]) - 0.5) * 0.3), -1))):
        model(gg, p, m, causal=True)
        eh = model.aux["err_pred"][m]
        ood[lab] = {"ehat_cl_mean_phys": float(eh[:, 0].mean() * std[0]), "ehat_cd_mean_phys": float(eh[:, 1].mean() * std[1]),
                    "ehat_cl_p90_phys": float(torch.quantile(eh[:, 0], 0.9) * std[0])}
    R["ood_error_head"] = ood
    # ---- memory knockouts incl. profile ----
    def evaluate(mm):
        tot = torch.zeros(2); n = 0
        for bb in dl:
            gg, pp, mk = prepare_polar_batch(bb, dev)
            f, _ = mm(gg, pp, mk, causal=True)
            tot += ((f - pp[..., 3:5]).abs() * std)[mk].sum(0); n += int(mk.sum())
        return (tot / n).tolist()
    base = evaluate(model)
    kos = {"baseline": base}
    names = ["global", "le", "te"] + (["profile"] if int(cfg.profile_stations) > 0 else [])
    for j, nm in enumerate(names):
        m2 = copy.deepcopy(model)
        orig = m2.encode_geom
        def f(coords, orig=orig, j=j):
            mem = orig(coords).clone(); mem[:, j] = 0.0; return mem
        m2.encode_geom = f
        kos[f"drop_memory_{nm}"] = evaluate(m2)
    if getattr(model, "err_expert", None) is not None:
        m2 = copy.deepcopy(model); m2.err_expert = None; kos["no_expert"] = evaluate(m2)
    if getattr(model, "err_head", None) is not None:
        m2 = copy.deepcopy(model)
        inner = m2.err_head
        class Z(torch.nn.Module):
            def forward(self, a, b_, c): return torch.zeros_like(inner(a, b_, c))
        m2.err_head = Z(); kos["err_head_zero"] = evaluate(m2)
    R["knockouts_test_causal"] = kos
    json.dump(R, open(os.path.join(args.out, "v61_extras.json"), "w"), indent=1)
    print(json.dumps(R, indent=1)[:6000])


def _hidden(model, g, p, m):
    model.debug = True
    model(g, p, m, causal=True)
    h = model.aux.get("hidden")
    model.debug = False
    if h is None:
        # foilcore path doesn't store hidden; recompute via hook-free approach: run corrector manually
        raise RuntimeError("hidden not available")
    return h


if __name__ == "__main__":
    main()
