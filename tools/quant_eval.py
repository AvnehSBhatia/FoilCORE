"""Post-training weight quantization: fp32 vs fp16 vs int8 (per-tensor symmetric) accuracy on held-out test.

python tools/quant_eval.py --ckpts v61=checkpoints/v61/foilcore_best.pt v70=... --n 2048
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.benchmark import load_models  # noqa: E402
from src.data import AeroDataset, collate_variable_sequences  # noqa: E402
from src.train_model import prepare_polar_batch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402


def quantize_int(model, bits=8, per_channel=True, matrices_only=False):
    m = copy.deepcopy(model)
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for name, p in m.named_parameters():
            w = p.data
            if matrices_only and w.dim() < 2:
                p.data = w.half().float()  # 1-D params (biases, norms, per-channel scalars) kept in fp16
                continue
            if per_channel and w.dim() == 2:
                s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
            else:
                s = w.abs().max().clamp_min(1e-12) / qmax
            p.data = torch.round(w / s).clamp(-qmax, qmax) * s
    return m


def to_half_and_back(model):
    m = copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            p.data = p.data.half().float()
    return m


@torch.no_grad()
def evaluate(model, dl, std, a_sd, a_mu, device=torch.device("cpu")):
    tot = torch.zeros(2, device=device); st = torch.zeros(2, device=device); n = 0; ns = 0
    for b in dl:
        g, p, m = prepare_polar_batch(b, device)
        f, _ = model(g, p, m, causal=True)
        e = ((f - p[..., 3:5]).abs() * std)
        stall = m & ((p[..., 0] * a_sd + a_mu).abs() >= 12)
        tot += e[m].sum(0); st += e[stall].sum(0); n += int(m.sum()); ns += int(stall.sum())
    return {"cl": float(tot[0] / n), "cd": float(tot[1] / n), "stall_cd": float(st[1] / max(ns, 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--csv", default="data/test.csv")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = {}
    ds = None
    device = torch.device(os.environ.get("FOILCORE_DEVICE", "cpu"))
    for pair in args.ckpts:
        name, path = pair.split("=", 1)
        model, stats, cfg, _ = load_models(path, device)
        model.eval()
        if ds is None:
            ds = AeroDataset(args.csv, stats=stats, training=False)
            step = len(ds) // args.n
            dl = DataLoader(Subset(ds, range(0, step * args.n, step)), batch_size=512, shuffle=False, collate_fn=collate_variable_sequences)
            std = torch.as_tensor(stats["polar_std"][3:5], dtype=torch.float32, device=device)
            a_sd, a_mu = stats["polar_std"][0], stats["polar_mean"][0]
        n_params = sum(p.numel() for p in model.parameters())
        r = {"n_params": n_params, "bytes_fp32": n_params * 4, "bytes_fp16": n_params * 2, "bytes_int8": n_params}
        r["fp32"] = evaluate(model, dl, std, a_sd, a_mu, device)
        r["fp16"] = evaluate(to_half_and_back(model), dl, std, a_sd, a_mu, device)
        r["int8_per_channel"] = evaluate(quantize_int(model, 8, True), dl, std, a_sd, a_mu, device)
        r["int8_matrices_fp16_rest"] = evaluate(quantize_int(model, 8, True, matrices_only=True), dl, std, a_sd, a_mu, device)
        n2 = sum(p.numel() for p in model.parameters() if p.dim() == 2); n1 = n_params - n2
        r["bytes_int8_matrices_fp16_rest"] = n2 + 2 * n1
        r["int8_per_tensor"] = evaluate(quantize_int(model, 8, False), dl, std, a_sd, a_mu, device)
        r["int6_per_channel"] = evaluate(quantize_int(model, 6, True), dl, std, a_sd, a_mu, device)
        r["int4_per_channel"] = evaluate(quantize_int(model, 4, True), dl, std, a_sd, a_mu, device)
        res[name] = r
        print(f"{name} ({n_params} params, {n_params/1024:.1f} KB int8): " + " | ".join(f"{k}: Cl {v['cl']:.4f} Cd {v['cd']:.5f}" for k, v in r.items() if isinstance(v, dict)))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
