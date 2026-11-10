"""Zero-shot time-series foundation models on the ETTh1 test blocks, for context in the transfer section.

They solve a different task from ours (forecast OT from its own past; no covariates, no smoothing mode), so
this is not a head-to-head. It answers: what does a 10M-500M-parameter pretrained forecaster achieve on the
same target hours, at what latency, and can it do what the 4k model does (virtual sensing from covariates,
stream and smooth with one set of weights)? Answers: MAE at horizons 1 and 24 h, params, ms per forecast.

python experiments/transfer/foundation_ett.py --out runs/transfer/foundation_ett.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_root = Path(__file__).resolve().parents[2]
sys.path = [q for q in sys.path if not q.rstrip("/").endswith("experiments/transfer")]  # our datasets.py must not shadow HF `datasets`
sys.path.insert(0, str(_root))
from experiments.transfer.datasets import DATA_DIR, _fetch  # noqa: E402

CONTEXT = 512
BLOCK = 4 * 7 * 24
HORIZONS = (1, 24)


def test_positions(n: int, stride: int = 4):
    """Hours in test blocks (block index % 6 == 0) with a full context behind them; same blocks as load_ett."""
    pos = []
    for b in range(n // BLOCK):
        if b % 6 != 0:
            continue
        for t in range(b * BLOCK, (b + 1) * BLOCK, stride):
            if t >= CONTEXT and t + max(HORIZONS) <= n:
                pos.append(t)
    return np.array(pos)


def mae_from_forecasts(y, pos, fc):
    """fc: (N, H) forecasts for hours pos+1..pos+H (1-indexed horizons)."""
    out = {}
    for h in HORIZONS:
        out[f"mae_h{h}"] = float(np.mean(np.abs(fc[:, h - 1] - y[pos + h])))
    return out


def run_chronos(name, ctx, y, pos, dev):
    from chronos import BaseChronosPipeline
    t0 = time.time()
    pipe = BaseChronosPipeline.from_pretrained(name, device_map=dev, torch_dtype=torch.float32)
    load_s = time.time() - t0
    n_params = sum(p.numel() for p in pipe.model.parameters())
    fcs = []
    torch.cuda.synchronize() if dev == "cuda" else None
    t0 = time.time()
    for i in range(0, len(ctx), 64):
        batch = torch.tensor(ctx[i : i + 64], dtype=torch.float32)
        q, mean = pipe.predict_quantiles(batch, prediction_length=max(HORIZONS), quantile_levels=[0.5])
        fcs.append(mean.cpu().numpy())
    torch.cuda.synchronize() if dev == "cuda" else None
    gpu_ms = (time.time() - t0) / len(ctx) * 1e3
    fc = np.concatenate(fcs)
    # single-context CPU latency (what an edge deployment would pay)
    pipe_cpu = BaseChronosPipeline.from_pretrained(name, device_map="cpu", torch_dtype=torch.float32)
    torch.set_num_threads(1)
    one = torch.tensor(ctx[:1], dtype=torch.float32)
    for _ in range(3):
        pipe_cpu.predict_quantiles(one, prediction_length=max(HORIZONS), quantile_levels=[0.5])
    t0 = time.time()
    for _ in range(10):
        pipe_cpu.predict_quantiles(one, prediction_length=max(HORIZONS), quantile_levels=[0.5])
    cpu_ms = (time.time() - t0) / 10 * 1e3
    return {"model": name, "n_params": n_params, **mae_from_forecasts(y, pos, fc), "gpu_batched_ms_per_forecast": gpu_ms,
            "cpu_1thread_ms_per_forecast": cpu_ms, "load_s": load_s}


def run_timesfm(ctx, y, pos, dev):
    import timesfm
    t0 = time.time()
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(max_context=CONTEXT, max_horizon=max(HORIZONS), normalize_inputs=True,
                                         use_continuous_quantile_head=True, force_flip_invariance=True,
                                         infer_is_positive=False, fix_quantile_crossing=True))
    load_s = time.time() - t0
    n_params = 200_000_000
    try:
        n_params = sum(p.numel() for p in model.model.parameters())
    except Exception:  # noqa: BLE001
        pass
    t0 = time.time()
    fc, _ = model.forecast(horizon=max(HORIZONS), inputs=[c for c in ctx])
    batched_ms = (time.time() - t0) / len(ctx) * 1e3
    fc = np.asarray(fc)[:, : max(HORIZONS)]
    t0 = time.time()
    for _ in range(3):
        model.forecast(horizon=max(HORIZONS), inputs=[ctx[0]])
    per = (time.time() - t0) / 3 * 1e3
    return {"model": "google/timesfm-2.5-200m-pytorch", "n_params": int(n_params), **mae_from_forecasts(y, pos, fc),
            "gpu_batched_ms_per_forecast": batched_ms, "single_forecast_ms_same_backend": per, "load_s": load_s}


def run_ttm(ctx, y, pos, dev):
    from tsfm_public.toolkit.get_model import get_model
    t0 = time.time()
    model = get_model("ibm-granite/granite-timeseries-ttm-r2", context_length=CONTEXT, prediction_length=96).to(dev).eval()
    load_s = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    fcs = []
    with torch.no_grad():
        t0 = time.time()
        for i in range(0, len(ctx), 128):
            x = torch.tensor(ctx[i : i + 128], dtype=torch.float32, device=dev).unsqueeze(-1)  # (B, L, 1)
            mu, sd = x.mean(1, keepdim=True), x.std(1, keepdim=True) + 1e-6
            out = model(past_values=(x - mu) / sd).prediction_outputs
            fcs.append((out * sd + mu)[:, : max(HORIZONS), 0].cpu().numpy())
        torch.cuda.synchronize() if dev == "cuda" else None
        gpu_ms = (time.time() - t0) / len(ctx) * 1e3
        fc = np.concatenate(fcs)
        model_cpu = model.to("cpu"); torch.set_num_threads(1)
        one = torch.tensor(ctx[:1], dtype=torch.float32).unsqueeze(-1)
        for _ in range(3):
            model_cpu(past_values=one)
        t0 = time.time()
        for _ in range(20):
            model_cpu(past_values=one)
        cpu_ms = (time.time() - t0) / 20 * 1e3
    return {"model": "ibm-granite/granite-timeseries-ttm-r2 (512-96)", "n_params": n_params, **mae_from_forecasts(y, pos, fc),
            "gpu_batched_ms_per_forecast": gpu_ms, "cpu_1thread_ms_per_forecast": cpu_ms, "load_s": load_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/transfer/foundation_ett.json")
    ap.add_argument("--models", default="naive,chronos-bolt-tiny,chronos-bolt-small,ttm,timesfm")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = _fetch("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv", DATA_DIR / "ETTh1.csv")
    y = pd.read_csv(p)["OT"].values.astype(np.float64)
    pos = test_positions(len(y))
    ctx = np.stack([y[t - CONTEXT : t] for t in pos])
    print(f"[foundation] {len(pos)} test hours (stride 4, {len(pos)*4} covered), context {CONTEXT}, device {dev}", flush=True)
    res = {"n_test_points": int(len(pos)), "context": CONTEXT, "horizons": list(HORIZONS), "models": {}}
    if "naive" in args.models:
        fc = np.repeat(ctx[:, -1:], max(HORIZONS), axis=1)
        res["models"]["naive-last-value"] = {"model": "last value", "n_params": 0, **mae_from_forecasts(y, pos, fc)}
        # seasonal naive: value 24 h earlier
        fc24 = np.stack([y[t + np.arange(1, max(HORIZONS) + 1) - 24] for t in pos])
        res["models"]["seasonal-naive-24h"] = {"model": "same hour yesterday", "n_params": 0, **mae_from_forecasts(y, pos, fc24)}
        print("naive", res["models"]["naive-last-value"], res["models"]["seasonal-naive-24h"], flush=True)
    for key, fn in (("chronos-bolt-tiny", lambda: run_chronos("amazon/chronos-bolt-tiny", ctx, y, pos, dev)),
                    ("chronos-bolt-small", lambda: run_chronos("amazon/chronos-bolt-small", ctx, y, pos, dev)),
                    ("ttm", lambda: run_ttm(ctx, y, pos, dev)),
                    ("timesfm", lambda: run_timesfm(ctx, y, pos, dev))):
        if key not in args.models:
            continue
        try:
            res["models"][key] = fn()
            print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["models"][key].items()}, flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            res["models"][key] = {"error": str(e)[:300]}
            print(key, "FAILED", str(e)[:200], flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
