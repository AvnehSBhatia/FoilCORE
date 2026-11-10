"""Quick physical-unit val MAE for a checkpoint: draft / final(TF) / final(causal).

python tools/quick_val.py --ckpt checkpoints/foilscan/foilcore_best.pt --csv data/val.csv --n 1024
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, Subset

from src.benchmark import load_models
from src.data import AeroDataset, collate_variable_sequences
from src.train_model import prepare_polar_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--csv", default="data/val.csv")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default=os.environ.get("FOILCORE_DEVICE", "cpu"))
    args = p.parse_args()

    device = torch.device(args.device)
    model, stats, cfg, _ = load_models(args.ckpt, device)
    ds = AeroDataset(args.csv, stats=stats, training=False)
    if args.n:
        step = max(1, len(ds) // args.n)  # strided: prefix rows are low-Re-biased
        ds = Subset(ds, range(0, step * min(args.n, len(ds)), step))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_variable_sequences)
    std = torch.as_tensor(stats["polar_std"][3:5], device=device)

    sums = {}
    cnt = 0
    stall_sums = {}
    stall_cnt = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in dl:
            geom, polar, mask = prepare_polar_batch(batch, device)
            outs = {}
            final_tf, draft = model(geom, polar, mask, causal=False)
            outs["draft"] = draft
            outs["final_tf"] = final_tf
            outs["final_causal"], _ = model(geom, polar, mask, causal=True)
            outs["final_seed1"], _ = model(geom, polar, mask, causal=True, seed_len=1)
            tgt = polar[..., 3:5]
            alpha_phys = polar[..., 0] * stats["polar_std"][0] + stats["polar_mean"][0]
            stall = mask & (alpha_phys.abs() >= 12.0)
            for k, v in outs.items():
                err = (v - tgt).abs() * std
                sums[k] = sums.get(k, torch.zeros(2, device=device)) + err[mask].sum(0)
                stall_sums[k] = stall_sums.get(k, torch.zeros(2, device=device)) + err[stall].sum(0)
            cnt += int(mask.sum())
            stall_cnt += int(stall.sum())
    print(f"ckpt={args.ckpt} n_rows={len(ds)} points={cnt} stall_pts={stall_cnt} device={device} ({time.time()-t0:.1f}s)")
    for k in sums:
        cl, cd = (sums[k] / cnt).tolist()
        scl, scd = (stall_sums[k] / max(stall_cnt, 1)).tolist()
        print(f"  {k:13s} Cl {cl:.4f}  Cd {cd:.5f}   | stall(|a|>=12) Cl {scl:.4f}  Cd {scd:.5f}")


if __name__ == "__main__":
    main()
