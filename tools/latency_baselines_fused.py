"""Fused C++ kernels for seqbaseline correctors: numerical check vs PyTorch and T=68 B=1 latency.

python tools/latency_baselines_fused.py \
  --ckpts v80=results/pod_runs/20260819_001018_v80/foilcore_best.pt \
          v81=results/pod_runs/20260819_001018_v81/foilcore_best.pt \
          v82=results/pod_runs/20260819_001018_v82/foilcore_best.pt \
  --out results/paper_final/baselines/fused_latency.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
from src.benchmark import load_models  # noqa: E402
from src.fused_baselines import FusedSeqBaseline, check  # noqa: E402


def _geom_polar(T: int):
    g = torch.empty(200, 2)
    g[:, 0] = torch.cat([torch.linspace(1, 0, 100), torch.linspace(0, 1, 100)])
    g[:100, 1] = 0.05
    g[100:, 1] = -0.05
    p = torch.randn(T, 5)
    p[:, 0] = torch.linspace(-1.8, 1.8, T)
    return g, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", default="results/paper_final/baselines/fused_latency.json")
    ap.add_argument("--T", type=int, default=68)
    ap.add_argument("--iters", type=int, default=2000)
    args = ap.parse_args()
    torch.set_num_threads(1)
    g, p = _geom_polar(args.T)
    mk = torch.ones(1, args.T, dtype=torch.bool)
    out = {}
    for pair in args.ckpts:
        name, path = pair.split("=", 1)
        m, _, cfg, _ = load_models(path, torch.device("cpu"))
        m.eval()
        print(f"[{name}] {cfg.seq_backbone}  params={sum(x.numel() for x in m.parameters())}  compiling…", flush=True)
        err_c = check(m, g, p, True)
        err_b = check(m, g, p, False)
        ff = FusedSeqBaseline(m)
        mem = ff.encode(m, g)
        with torch.inference_mode():
            for _ in range(50):
                m(g[None], p[None], mk, causal=True)
            te = []
            for _ in range(200):
                t0 = time.perf_counter()
                m(g[None], p[None], mk, causal=True)
                te.append(time.perf_counter() - t0)
        for _ in range(100):
            ff(g, p, True)
        tf, tc = [], []
        for _ in range(args.iters):
            t0 = time.perf_counter(); ff(g, p, True); tf.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); ff(g, p, True, mem); tc.append(time.perf_counter() - t0)
        rec = {
            "seq_backbone": cfg.seq_backbone,
            "n_params": int(sum(x.numel() for x in m.parameters())),
            "T": args.T,
            "max_abs_diff_causal": err_c,
            "max_abs_diff_bidir": err_b,
            "eager_ms": float(np.median(te) * 1e3),
            "fused_full_us": float(np.median(tf) * 1e6),
            "fused_cached_us": float(np.median(tc) * 1e6),
        }
        out[name] = rec
        print(
            f"  check causal {err_c:.2e}  bidir {err_b:.2e}  "
            f"eager {rec['eager_ms']:.2f} ms  fused {rec['fused_full_us']:.0f} µs "
            f"(cached {rec['fused_cached_us']:.0f} µs)",
            flush=True,
        )
        bp = _root / "results/paper_final/baselines/numbers.json"
        if bp.is_file() and name in json.loads(bp.read_text()).get("models", {}):
            N = json.loads(bp.read_text())
            N["models"][name]["latency"]["fused_cpp_b1_us"] = {
                "full_median": rec["fused_full_us"],
                "geom_cached_median": rec["fused_cached_us"],
                "max_abs_diff_vs_torch": rec["max_abs_diff_causal"],
            }
            bp.write_text(json.dumps(N, indent=1) + "\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
