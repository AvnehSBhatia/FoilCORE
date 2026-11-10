"""Fused-kernel latency vs T for several checkpoints → JSON (used by the streaming figure + macros).
python tools/latency_sweep.py --ckpts v61=... v6=... v70=... --out results/paper_final/latency_sweep.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
_root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(_root))
from src.benchmark import load_models
from src.fused_forward import FusedFoilCORE
ap = argparse.ArgumentParser(); ap.add_argument("--ckpts", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--Ts", default="8,34,68")
a = ap.parse_args(); torch.set_num_threads(1)
g = torch.rand(200, 2); g[:, 0] = torch.cat([torch.linspace(1, 0, 100), torch.linspace(0, 1, 100)]); g[:100, 1] = 0.05; g[100:, 1] = -0.05
out = {}
for pair in a.ckpts:
    name, path = pair.split("=", 1)
    m, stats, cfg, _ = load_models(path, torch.device("cpu")); m.eval()
    ff = FusedFoilCORE(m); mem = ff.encode(m, g)
    res = {"n_params": sum(p.numel() for p in m.parameters()), "T": [], "full_us": [], "cached_us": []}
    for T in [int(t) for t in a.Ts.split(",")]:
        p = torch.randn(T, 5); p[:, 0] = torch.linspace(-1.8, 1.8, T)
        for _ in range(200): ff(g, p, True)
        tf, tc = [], []
        for _ in range(3000):
            t = time.perf_counter(); ff(g, p, True); tf.append(time.perf_counter() - t)
            t = time.perf_counter(); ff(g, p, True, mem); tc.append(time.perf_counter() - t)
        res["T"].append(T); res["full_us"].append(float(np.median(tf) * 1e6)); res["cached_us"].append(float(np.median(tc) * 1e6))
    out[name] = res; print(name, {k: (v if k == "n_params" else [round(x, 1) for x in v]) for k, v in res.items()})
json.dump(out, open(a.out, "w"), indent=1)
