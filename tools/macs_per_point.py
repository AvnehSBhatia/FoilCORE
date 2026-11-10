"""Matrix-multiply MACs per polar point for a checkpoint (torch FlopCounterMode: mm/bmm/addmm/linear; FLOPs/2/T).

python tools/macs_per_point.py checkpoints/v61/foilcore_best.pt results/pod_runs/20260819_001018_v80/foilcore_best.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmark import load_models  # noqa: E402


def macs_per_point(path: str, T: int = 68) -> float:
    m, stats, cfg, _ = load_models(path, torch.device("cpu")); m.eval()
    g = torch.zeros(1, 200, 2); g[..., 0] = torch.cat([torch.linspace(1, 0, 100), torch.linspace(0, 1, 100)]); g[:, :100, 1] = 0.05; g[:, 100:, 1] = -0.05
    p = torch.randn(1, T, 5); p[..., 0] = torch.linspace(-1.8, 1.8, T); mk = torch.ones(1, T, dtype=torch.bool)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(g, p, mk, causal=True)
    return fc.get_total_flops() / 2 / T


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(f"{path}: {macs_per_point(path):,.0f} MACs/point (T=68, incl. geometry encode)")
