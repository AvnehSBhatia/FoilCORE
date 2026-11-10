"""Throughput of a checkpoint on N held-out geometries (one polar each) across batch sizes.

python tools/throughput_test.py --ckpt checkpoints/v6/foilcore_best.pt --csv data/test.csv --n 1000
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.benchmark import load_models  # noqa: E402
from src.data import AeroDataset, collate_variable_sequences  # noqa: E402
from src.train_model import prepare_polar_batch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", default="data/test.csv")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch-sizes", default="1,8,32,128,256,500,1000")
    ap.add_argument("--devices", default="cpu,mps")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    model_cpu, stats, cfg, _ = load_models(args.ckpt, torch.device("cpu"))
    ds = AeroDataset(args.csv, stats=stats, training=False)
    keys = ds.df["coords"].map(lambda c: hashlib.md5(np.asarray(c, dtype=np.float64).tobytes()).hexdigest())
    first_rows = ds.df.assign(_k=keys.values).groupby("_k", sort=False).head(1).index.tolist()
    rows = first_rows[: args.n]
    if len(rows) < args.n:  # fewer unique geometries than requested: top up with more polars
        rows += [i for i in range(len(ds)) if i not in set(rows)][: args.n - len(rows)]
    dl = DataLoader(Subset(ds, rows), batch_size=len(rows), shuffle=False, collate_fn=collate_variable_sequences)
    g, p, m = prepare_polar_batch(next(iter(dl)), torch.device("cpu"))
    n_pts = int(m.sum())
    print(f"{len(rows)} polars ({len(first_rows)} unique geometries available), {n_pts} points, T_max={p.shape[1]}, "
          f"torch threads={torch.get_num_threads()}, params={sum(q.numel() for q in model_cpu.parameters())}")
    for dev in args.devices.split(","):
        if dev == "mps" and not torch.backends.mps.is_available():
            continue
        device = torch.device(dev)
        model = load_models(args.ckpt, device)[0].eval()
        G, P, M = g.to(device), p.to(device), m.to(device)
        best = None
        for bs in [int(b) for b in args.batch_sizes.split(",")]:
            with torch.inference_mode():
                for _ in range(2):  # warmup
                    for i in range(0, len(rows), bs):
                        model(G[i:i + bs], P[i:i + bs], M[i:i + bs], causal=True)
                if dev == "mps":
                    torch.mps.synchronize()
                ts = []
                for _ in range(args.repeats):
                    t = time.perf_counter()
                    for i in range(0, len(rows), bs):
                        model(G[i:i + bs], P[i:i + bs], M[i:i + bs], causal=True)
                    if dev == "mps":
                        torch.mps.synchronize()
                    ts.append(time.perf_counter() - t)
            tot = min(ts)
            line = f"  {dev:4s} bs={bs:5d}: {tot*1e3:8.1f} ms total  {tot/len(rows)*1e3:7.3f} ms/polar  {len(rows)/tot:9.0f} polars/s  {n_pts/tot:10.0f} points/s"
            print(line)
            if best is None or tot < best[1]:
                best = (bs, tot)
        print(f"  {dev} best: bs={best[0]} → {best[1]*1e3:.1f} ms for {len(rows)} polars ({best[1]/len(rows)*1e6:.0f} µs/polar)")


if __name__ == "__main__":
    main()
