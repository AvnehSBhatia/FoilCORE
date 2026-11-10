"""Run the single-kernel FoilCORE forward in a tight loop for --seconds and print polars/s.
Used by tools/energy_cpu_mac.sh (powermetrics needs sudo, so the loop and the meter are separate processes).

python tools/energy_cpu_loop.py --ckpt checkpoints/v61/foilcore_best.pt --seconds 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch  # noqa: E402

torch.set_num_threads(1)
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
from src.benchmark import load_models  # noqa: E402
from src.fused_forward import FusedFoilCORE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v61/foilcore_best.pt")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--T", type=int, default=68)
    ap.add_argument("--cached", action="store_true", help="encode geometry once (streaming case)")
    ap.add_argument("--neuralfoil", default=None, help="run NeuralFoil <size> (xxsmall|xxxlarge) instead of FoilCORE, same polar")
    args = ap.parse_args()
    if args.neuralfoil:
        import numpy as np
        import neuralfoil as nf
        x = np.concatenate([np.linspace(1, 0, 100), np.linspace(0, 1, 100)])
        y = np.concatenate([0.06 * np.sqrt(np.clip(x[:100], 0, 1)) * (1 - x[:100]), -0.06 * np.sqrt(np.clip(x[100:], 0, 1)) * (1 - x[100:])])
        coords = np.stack([x, y], 1); alpha = np.linspace(-15, 15, args.T); Re = np.full_like(alpha, 5e5)
        for _ in range(20):
            nf.get_aero_from_coordinates(coordinates=coords, alpha=alpha, Re=Re, model_size=args.neuralfoil)
        n, t0 = 0, time.time()
        while time.time() - t0 < args.seconds:
            for _ in range(20):
                nf.get_aero_from_coordinates(coordinates=coords, alpha=alpha, Re=Re, model_size=args.neuralfoil)
            n += 20
        dt = time.time() - t0
        print(f"POLARS_PER_S {n/dt:.1f}  us_per_polar {dt/n*1e6:.1f}  n {n}  seconds {dt:.1f}  neuralfoil {args.neuralfoil}")
        return
    m, stats, cfg, _ = load_models(args.ckpt, torch.device("cpu")); m.eval()
    ff = FusedFoilCORE(m)
    coords = torch.zeros(200, 2)
    coords[:, 0] = torch.cat([torch.linspace(1, 0, 100), torch.linspace(0, 1, 100)]); coords[:100, 1] = 0.05; coords[100:, 1] = -0.05
    polar = torch.randn(args.T, 5); polar[:, 0] = torch.linspace(-1.8, 1.8, args.T)
    mem = ff.encode(m, coords) if args.cached else None
    for _ in range(200):
        ff(coords, polar, True, mem)
    n, t0 = 0, time.time()
    while time.time() - t0 < args.seconds:
        for _ in range(500):
            ff(coords, polar, True, mem)
        n += 500
    dt = time.time() - t0
    print(f"POLARS_PER_S {n/dt:.1f}  us_per_polar {dt/n*1e6:.1f}  n {n}  seconds {dt:.1f}")


if __name__ == "__main__":
    main()
