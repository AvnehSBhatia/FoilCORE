"""B=1 CPU latency per polar (ms), single BLAS thread, causal AR mode.

python tools/latency_b1.py --ckpt checkpoints/foilcore_best.pt
python tools/latency_b1.py --variant 50            # random-init model of a variant
FOILCORE_NO_FUSED=1 python tools/latency_b1.py ... # disable the fused C++ scan
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.model import FoilCORE, FoilCoreConfig, count_parameters  # noqa: E402


def build_from_variant(vid: int):
    from src.ablations import get_variant_spec

    v = get_variant_spec(vid)
    kw = {k: x for k, x in v.model.items() if hasattr(FoilCoreConfig(), k)}
    return FoilCORE(config=FoilCoreConfig(**kw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--variant", type=int, default=None)
    ap.add_argument("--T", type=int, default=68, help="polar length")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--causal", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    if args.ckpt:
        from src.benchmark import load_models

        model, stats, cfg, _ = load_models(args.ckpt, torch.device("cpu"))
        label = args.ckpt
    else:
        model = build_from_variant(args.variant)
        label = f"variant {args.variant} (random init)"
    model.eval()
    g = torch.rand(1, 200, 2)
    g[..., 0] = torch.linspace(1, 0, 100).repeat(2)[:200]
    p = torch.randn(1, args.T, 5)
    p[..., 0] = torch.linspace(-1.8, 1.8, args.T)
    m = torch.ones(1, args.T, dtype=torch.bool)
    from src import fused_scan

    fused = fused_scan.get_ext() is not None
    with torch.inference_mode():
        for _ in range(50):
            model(g, p, m, causal=bool(args.causal))
        ts = []
        for _ in range(args.iters):
            t = time.perf_counter()
            model(g, p, m, causal=bool(args.causal))
            ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts)
    print(
        f"{label}: {count_parameters(model)} params  T={args.T}  causal={bool(args.causal)}  fused_scan={fused}  "
        f"median {np.median(ts):.3f} ms  p90 {np.percentile(ts, 90):.3f} ms  min {ts.min():.3f} ms"
    )


if __name__ == "__main__":
    main()
