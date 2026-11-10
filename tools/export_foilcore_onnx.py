#!/usr/bin/env python3
"""
Best-effort ONNX export for the FoilCORE polar path (fixed polar length).

Usage:
  python tools/export_foilcore_onnx.py --ckpt checkpoints/foilcore_best.pt --out foilcore_dummy.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.benchmark import load_models  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/foilcore_best.pt")
    p.add_argument("--out", default="foilcore_trace.onnx")
    p.add_argument("--polar-len", type=int, default=64)
    p.add_argument("--batch", type=int, default=1)
    args = p.parse_args()

    ck = str(_ROOT / args.ckpt)
    device = torch.device("cpu")
    model, _, cfg, _ = load_models(ck, device)

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, geom, polar, mask):
            final, _draft = self.m(geom, polar, mask)
            return final

    w = Wrapper(model).eval()
    bs, t = args.batch, args.polar_len
    dummy_g = torch.randn(bs, 200, 2)
    dummy_p = torch.randn(bs, t, 5)
    dummy_m = torch.ones(bs, t, dtype=torch.bool)
    try:
        torch.onnx.export(
            w,
            (dummy_g, dummy_p, dummy_m),
            args.out,
            input_names=["geom_coords", "polar_tokens", "polar_mask"],
            output_names=["polar_pred"],
            opset_version=17,
            dynamic_axes=None,
        )
        print(f"Wrote {args.out} (fixed T={t}, d={cfg.d_model}).")
    except Exception as exc:
        print(f"ONNX export failed (expected for some builds): {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
