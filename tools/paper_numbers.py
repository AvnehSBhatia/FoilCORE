"""Every number for the paper, for a set of checkpoints, in one run.

For each checkpoint:
  * held-out TEST set: causal-AR, TF and seeded-AR(K=1) pooled physical MAE, stall slice, α / Re / Mach bins
  * NeuralFoil xxsmall / xxxlarge on the same test rows (strided sample; NeuralFoil ignores Mach)
  * causal-mode knockouts (via paper/scripts/analyze_foilcore2.py)
  * latency: B=1 CPU single-thread, batched CPU, batched MPS
  * params, per-module param breakdown
Writes results/<stamp>/{numbers.json, tables.md, per-model json}.

python tools/paper_numbers.py --ckpts v6=checkpoints/v6/foilcore_best.pt v61=... --csv data/test.csv --n 4096
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.benchmark import load_models  # noqa: E402
from src.data import AeroDataset, collate_variable_sequences  # noqa: E402
from src.model import count_parameters  # noqa: E402
from src.train_model import prepare_polar_batch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402


def strided_rows(n_total, n):
    step = max(1, n_total // n)
    return list(range(0, step * min(n, n_total), step))


@torch.no_grad()
def accuracy(model, stats, ds, rows, device, bs=512):
    dl = DataLoader(Subset(ds, rows), batch_size=bs, shuffle=False, collate_fn=collate_variable_sequences)
    std = torch.as_tensor(stats["polar_std"][3:5], dtype=torch.float32, device=device)
    a_sd, a_mu = stats["polar_std"][0], stats["polar_mean"][0]
    re_sd, re_mu = stats["polar_std"][1], stats["polar_mean"][1]
    m_sd, m_mu = stats["polar_std"][2], stats["polar_mean"][2]
    modes = {"ar": dict(causal=True), "tf": dict(causal=False), "seeded1": dict(causal=True, seed_len=1)}
    E = {k: [] for k in modes}
    D = []
    AL, RE, MA = [], [], []
    for b in dl:
        g, p, m = prepare_polar_batch(b, device)
        for k, kw in modes.items():
            f, d = model(g, p, m, **kw)
            E[k].append(((f - p[..., 3:5]).abs() * std)[m])
            if k == "ar":
                D.append(((d - p[..., 3:5]).abs() * std)[m])
        AL.append((p[..., 0] * a_sd + a_mu)[m]); RE.append((p[..., 1] * re_sd + re_mu)[m]); MA.append((p[..., 2] * m_sd + m_mu)[m])
    AL, RE, MA = torch.cat(AL), torch.cat(RE), torch.cat(MA)
    D = torch.cat(D)
    out = {"n_rows": len(rows), "n_points": int(AL.numel())}
    stall = AL.abs() >= 12.0
    for k in modes:
        e = torch.cat(E[k])
        out[k] = {"cl": float(e[:, 0].mean()), "cd": float(e[:, 1].mean()),
                  "stall_cl": float(e[stall, 0].mean()), "stall_cd": float(e[stall, 1].mean()),
                  "linear_cl": float(e[~stall, 0].mean()), "linear_cd": float(e[~stall, 1].mean())}
    out["draft"] = {"cl": float(D[:, 0].mean()), "cd": float(D[:, 1].mean())}
    e = torch.cat(E["ar"])
    def bins(v, edges):
        res = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (v >= lo) & (v < hi)
            if s.any():
                res.append({"lo": float(lo), "hi": float(hi), "n": int(s.sum()), "cl": float(e[s, 0].mean()), "cd": float(e[s, 1].mean())})
        return res
    out["ar_by_alpha"] = bins(AL, [-20, -12, -6, 0, 6, 12, 20.1])
    out["ar_by_re"] = bins(RE, [4e4, 1e5, 3e5, 1e6, 3e6, 2.1e7])
    out["ar_by_mach"] = bins(MA, [-0.01, 0.05, 0.15, 0.3, 0.51])
    return out


def neuralfoil(ds, rows, sizes=("xxsmall", "xxxlarge"), workers=6):
    import inspect
    from multiprocessing import Pool
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    recs = [(ds.df.iloc[i][["coords", "alpha", "Cl", "Cd", "Re", "mach"]].to_dict(), list(sizes)) for i in rows]
    with Pool(workers) as pool:
        res = pool.starmap(_nf_worker, recs, chunksize=4)
    out = {}
    for sz in sizes:
        ecl, ecd, scl, scd = [], [], [], []
        for al, cl, cd, o in res:
            if o.get(sz) is None:
                continue
            pcl, pcd = o[sz]
            e1, e2 = np.abs(pcl - cl), np.abs(pcd - cd)
            st = np.abs(al) >= 12
            ecl.append(e1); ecd.append(e2); scl.append(e1[st]); scd.append(e2[st])
        ecl, ecd, scl, scd = map(np.concatenate, (ecl, ecd, scl, scd))
        out[sz] = {"cl": float(ecl.mean()), "cd": float(ecd.mean()), "stall_cl": float(scl.mean()), "stall_cd": float(scd.mean()),
                   }
    return out


_NF_SIZES = []


def _nf_worker(rec, sizes):
    import neuralfoil as nf
    coords = np.asarray(rec["coords"], float); alpha = np.asarray(rec["alpha"], float)
    out = {}
    for sz in sizes:
        try:
            p = nf.get_aero_from_coordinates(coordinates=coords, alpha=alpha, Re=np.full_like(alpha, float(rec["Re"])), model_size=sz)
            out[sz] = (np.asarray(p["CL"], float), np.asarray(p["CD"], float))
        except Exception:
            out[sz] = None
    return alpha, np.asarray(rec["Cl"], float), np.asarray(rec["Cd"], float), out


def latency(model, T=68, iters=200):
    import copy
    res = {}
    torch.set_num_threads(1)
    m = copy.deepcopy(model).cpu().eval()
    g = torch.rand(1, 200, 2); g[..., 0] = torch.cat([torch.linspace(1, 0, 100), torch.linspace(0, 1, 100)[1:]]); g[:, :100, 1] = 0.05; g[:, 100:, 1] = -0.05
    p = torch.randn(1, T, 5); p[..., 0] = torch.linspace(-1.8, 1.8, T); p[..., 1] = 0.0; p[..., 2] = 0.0; mk = torch.ones(1, T, dtype=torch.bool)
    with torch.inference_mode():
        for _ in range(50): m(g, p, mk, causal=True)
        ts = []
        for _ in range(iters):
            t = time.perf_counter(); m(g, p, mk, causal=True); ts.append(time.perf_counter() - t)
    res["cpu_b1_1thread_ms"] = {"median": float(np.median(ts) * 1e3), "p90": float(np.percentile(ts, 90) * 1e3)}
    try:
        c0, p0 = g[0], p[0].contiguous()
        if getattr(m.config, "model_type", "") == "seqbaseline":
            from src.fused_baselines import FusedSeqBaseline, check as _chk
            ff = FusedSeqBaseline(m)
        else:
            from src.fused_forward import FusedFoilCORE
            from src.fused_forward import check as _chk
            ff = FusedFoilCORE(m)
        err = _chk(m, c0, p0, True)
        mem = ff.encode(m, c0)
        for _ in range(100): ff(c0, p0, True)
        tf, tc = [], []
        for _ in range(2000):
            t = time.perf_counter(); ff(c0, p0, True); tf.append(time.perf_counter() - t)
            t = time.perf_counter(); ff(c0, p0, True, mem); tc.append(time.perf_counter() - t)
        res["fused_cpp_b1_us"] = {"full_median": float(np.median(tf) * 1e6), "geom_cached_median": float(np.median(tc) * 1e6),
                                  "max_abs_diff_vs_torch": float(err)}
    except Exception as exc:  # noqa: BLE001
        res["fused_cpp_b1_us"] = {"error": str(exc)[:200]}
    torch.set_num_threads(8)
    for dev in ("cpu", "mps"):
        if dev == "mps" and not torch.backends.mps.is_available():
            continue
        md = copy.deepcopy(model).to(dev).eval()
        best = None
        for bs in (128, 256, 512, 1024):
            G = g.repeat(bs, 1, 1).to(dev); P = p.repeat(bs, 1, 1).to(dev); M = mk.repeat(bs, 1).to(dev)
            with torch.inference_mode():
                for _ in range(5): md(G, P, M, causal=True)
                if dev == "mps": torch.mps.synchronize()
                n = 20
                t = time.perf_counter()
                for _ in range(n): md(G, P, M, causal=True)
                if dev == "mps": torch.mps.synchronize()
                us = (time.perf_counter() - t) / n / bs * 1e6
            if best is None or us < best[1]:
                best = (bs, us)
        res[f"{dev}_batched_us_per_polar"] = {"bs": best[0], "us": float(best[1])}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--csv", default="data/test.csv")
    ap.add_argument("--n", type=int, default=4096, help="strided test rows for model eval")
    ap.add_argument("--n-nf", type=int, default=1024, help="strided test rows for NeuralFoil")
    ap.add_argument("--nf-workers", type=int, default=6)
    ap.add_argument("--skip-knockouts", action="store_true")
    ap.add_argument("--skip-neuralfoil", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(args.out or f"results/{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(os.environ.get("FOILCORE_DEVICE", "cpu"))
    results = {"csv": args.csv, "n_rows": args.n, "models": {}}
    ds = None
    for pair in args.ckpts:
        name, path = pair.split("=", 1)
        print(f"\n===== {name}: {path}")
        model, stats, cfg, _ = load_models(path, device)
        model.eval()
        if ds is None:
            ds = AeroDataset(args.csv, stats=stats, training=False)
            rows = strided_rows(len(ds), args.n)
            rows_nf = strided_rows(len(ds), args.n_nf)
        n_params = count_parameters(model)
        breakdown = {}
        for n_, p_ in model.named_parameters():
            breakdown[n_.split(".")[0]] = breakdown.get(n_.split(".")[0], 0) + p_.numel()
        t = time.time()
        acc = accuracy(model, stats, ds, rows, device)
        print(f"  accuracy ({time.time()-t:.0f}s): AR Cl {acc['ar']['cl']:.4f} Cd {acc['ar']['cd']:.5f} | TF {acc['tf']['cl']:.4f}/{acc['tf']['cd']:.5f} | seeded1 {acc['seeded1']['cl']:.4f}/{acc['seeded1']['cd']:.5f} | stall AR Cd {acc['ar']['stall_cd']:.5f}")
        lat = latency(model)
        fz = lat.get("fused_cpp_b1_us", {})
        print(f"  latency: B=1 CPU eager {lat['cpu_b1_1thread_ms']['median']:.3f} ms | fused C++ {fz.get('full_median', float('nan')):.0f} µs (geom cached {fz.get('geom_cached_median', float('nan')):.0f} µs) | " + " | ".join(f"{k}: {v['us']:.1f} µs @bs{v['bs']}" for k, v in lat.items() if k.endswith("polar")))
        entry = {"ckpt": path, "n_params": n_params, "param_breakdown": breakdown, "config": cfg.__dict__, "accuracy": acc, "latency": lat}
        ck = torch.load(path, map_location="cpu", weights_only=False)
        entry["train"] = {"epoch": ck.get("epoch"), "val_causal": ck.get("val_causal"), "variant": ck.get("variant")}
        if not args.skip_knockouts:
            ko_out = out_dir / f"knockouts_{name}"
            cmd = [sys.executable, str(_root / "paper/scripts/analyze_foilcore2.py"), "--ckpt", path, "--csv", args.csv, "--n", "2048", "--out", str(ko_out)]
            t = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_root), env={**os.environ, "PYTHONPATH": str(_root)})
            (ko_out / "stdout.txt").write_text(r.stdout + "\n" + r.stderr)
            kj = ko_out / "analysis.json"
            if kj.is_file():
                entry["knockouts_val2048"] = json.load(open(kj)).get("knockouts")
                print(f"  knockouts done ({time.time()-t:.0f}s)")
            else:
                print(f"  knockouts FAILED: {r.stderr[-400:]}")
        results["models"][name] = entry
        json.dump(entry, open(out_dir / f"{name}.json", "w"), indent=1)
    if not args.skip_neuralfoil:
        t = time.time()
        results["neuralfoil_test"] = {"n_rows": len(rows_nf), **neuralfoil(ds, rows_nf, workers=args.nf_workers)}
        print(f"\nNeuralFoil ({time.time()-t:.0f}s): " + " | ".join(f"{k}: Cl {v['cl']:.4f} Cd {v['cd']:.5f} stallCd {v['stall_cd']:.5f}" for k, v in results['neuralfoil_test'].items() if isinstance(v, dict)))
        # models on the same NF rows for a strict apples-to-apples
        for name, entry in results["models"].items():
            model, stats, cfg, _ = load_models(entry["ckpt"], device)
            entry["accuracy_nf_rows"] = accuracy(model, stats, ds, rows_nf, device)
    json.dump(results, open(out_dir / "numbers.json", "w"), indent=1)
    # ---- markdown tables ----
    md = [f"# FoilCORE paper numbers ({stamp})\n", f"Test CSV: {args.csv}, {args.n} strided rows (models), {args.n_nf} rows (NeuralFoil head-to-head)\n"]
    md.append("\n## Main table (held-out test, causal AR unless noted; physical units)\n")
    md.append("| model | params | AR Cl | AR Cd | AR stall Cd | TF Cl | TF Cd | seeded-1 Cl | seeded-1 Cd | draft Cl | B=1 eager ms | B=1 fused µs (geom cached) | CPU µs/polar | MPS µs/polar |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, e in results["models"].items():
        a, l = e["accuracy"], e["latency"]
        fz = l.get("fused_cpp_b1_us", {})
        md.append(f"| {name} | {e['n_params']} | {a['ar']['cl']:.4f} | {a['ar']['cd']:.5f} | {a['ar']['stall_cd']:.5f} | {a['tf']['cl']:.4f} | {a['tf']['cd']:.5f} | {a['seeded1']['cl']:.4f} | {a['seeded1']['cd']:.5f} | {a['draft']['cl']:.4f} | {l['cpu_b1_1thread_ms']['median']:.3f} | {fz.get('full_median', float('nan')):.0f} ({fz.get('geom_cached_median', float('nan')):.0f}) | {l['cpu_batched_us_per_polar']['us']:.0f} | {l.get('mps_batched_us_per_polar', {}).get('us', float('nan')):.0f} |")
    if "neuralfoil_test" in results:
        md.append("\n## NeuralFoil head-to-head (same test rows; NeuralFoil has no Mach input)\n")
        md.append("| model | params | Cl | Cd | stall Cd |"); md.append("|---|---|---|---|---|")
        for name, e in results["models"].items():
            a = e["accuracy_nf_rows"]["ar"]
            md.append(f"| {name} (AR) | {e['n_params']} | {a['cl']:.4f} | {a['cd']:.5f} | {a['stall_cd']:.5f} |")
        for sz, v in results["neuralfoil_test"].items():
            if isinstance(v, dict):
                np_ = {"xxsmall": 13302, "xxxlarge": 1428166}.get(sz, "?")
                md.append(f"| NeuralFoil {sz} | {np_} | {v['cl']:.4f} | {v['cd']:.5f} | {v['stall_cd']:.5f} |")
    for name, e in results["models"].items():
        a = e["accuracy"]
        md.append(f"\n## {name}: AR error by regime\n")
        md.append("| bin | n | Cl | Cd |"); md.append("|---|---|---|---|")
        for key, lab in (("ar_by_alpha", "α"), ("ar_by_re", "Re"), ("ar_by_mach", "Mach")):
            for r in a[key]:
                md.append(f"| {lab} [{r['lo']:g}, {r['hi']:g}) | {r['n']} | {r['cl']:.4f} | {r['cd']:.5f} |")
        if e.get("knockouts_val2048"):
            ko = e["knockouts_val2048"]; b = ko["baseline_causal"]
            md.append(f"\n## {name}: causal knockouts (val, 2048 rows; baseline Cl {b['cl']:.4f} Cd {b['cd']:.5f})\n")
            md.append("| knockout | Cl | ΔCl | Cd | ΔCd |"); md.append("|---|---|---|---|---|")
            for k, v in ko.items():
                if k.startswith("baseline") or k.endswith("_knockouts"):
                    continue
                md.append(f"| {k} | {v['cl']:.4f} | {v['cl']-b['cl']:+.4f} | {v['cd']:.5f} | {v['cd']-b['cd']:+.5f} |")
    (out_dir / "tables.md").write_text("\n".join(md))
    print(f"\nwrote {out_dir}/numbers.json and tables.md")


if __name__ == "__main__":
    main()
