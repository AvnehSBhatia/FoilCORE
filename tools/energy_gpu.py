"""GPU energy per polar: board power (nvidia-smi, 100 ms samples) minus idle, batched causal inference.

python tools/energy_gpu.py --ckpts v61=checkpoints/v61/foilcore_best.pt v70=checkpoints/v70/foilcore_best.pt --out runs/energy_gpu.json
Needs an otherwise idle GPU; numbers are an upper bound (a 200 W board running a 5k-parameter model).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
from src.benchmark import load_models  # noqa: E402


def _sampler():
    return subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-lms", "100"],
                            stdout=subprocess.PIPE, text=True)


def _read(p) -> float:
    p.terminate()
    vals = [float(x) for x in p.stdout.read().split() if x.replace(".", "", 1).isdigit()]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="name=path ...")
    ap.add_argument("--out", default="runs/energy_gpu.json")
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()
    p = _sampler(); time.sleep(5); idle = _read(p); print(f"idle {idle:.1f} W")
    dev = torch.device("cuda"); T = 68; out = {}
    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        m, stats, cfg, _ = load_models(path, dev); m.eval()
        bs = args.bs
        g = torch.rand(bs, 200, 2, device=dev)
        g[..., 0] = torch.cat([torch.linspace(1, 0, 100), torch.linspace(0, 1, 100)]).to(dev); g[:, :100, 1] = 0.05; g[:, 100:, 1] = -0.05
        pol = torch.randn(bs, T, 5, device=dev); pol[..., 0] = torch.linspace(-1.8, 1.8, T, device=dev)
        mk = torch.ones(bs, T, dtype=torch.bool, device=dev)
        with torch.inference_mode():
            for _ in range(20):
                m(g, pol, mk, causal=True)
            torch.cuda.synchronize()
            n = 0; stop = time.time() + args.seconds; ps = _sampler(); t0 = time.time()
            while time.time() < stop:
                m(g, pol, mk, causal=True); n += 1
            torch.cuda.synchronize(); dt = time.time() - t0; P = _read(ps)
        polars = n * bs
        out[name] = {"bs": bs, "polars_per_s": polars / dt, "board_W": P, "idle_W": idle,
                     "uJ_per_polar_total": P * dt / polars * 1e6, "uJ_per_polar_dynamic": (P - idle) * dt / polars * 1e6}
        print(f"{name}: {polars/dt:,.0f} polars/s, {P:.1f} W (idle {idle:.1f}) -> {out[name]['uJ_per_polar_total']:.1f} uJ/polar total, {out[name]['uJ_per_polar_dynamic']:.1f} above idle")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
