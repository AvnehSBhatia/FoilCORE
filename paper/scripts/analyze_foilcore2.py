"""Weight / activation / knockout analysis for FoilCORE v2 (model_type=foilcore2).

Everything is measured in **causal AR** mode (draft in, forward filter) unless noted.

python paper/scripts/analyze_foilcore2.py --ckpt checkpoints/foilcore2/foilcore_best.pt \
    --csv data/val.csv --n 1024 --out paper/figures/foilcore2_analysis
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.benchmark import load_models  # noqa: E402
from src.data import AeroDataset, collate_variable_sequences  # noqa: E402
from src.model import PathNonlinearity  # noqa: E402
from src.train_model import prepare_polar_batch  # noqa: E402


def load_batches(csv, stats, n, bs, device):
    ds = AeroDataset(csv, stats=stats, training=False)
    # Strided sample: prefix rows are biased to low Re (hard); every k-th row is representative.
    step = max(1, len(ds) // n)
    ds = Subset(ds, range(0, step * min(n, len(ds)), step))
    dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_variable_sequences)
    return [prepare_polar_batch(b, device) for b in dl]


@torch.no_grad()
def evaluate(model, batches, stats, *, causal=True, seed_len=0):
    """Pooled physical MAE (Cl, Cd) overall and on |α|>=12."""
    std = torch.as_tensor(stats["polar_std"][3:5], dtype=torch.float32, device=next(model.parameters()).device)
    a_mu, a_sd = float(stats["polar_mean"][0]), float(stats["polar_std"][0])
    dev = next(model.parameters()).device
    tot = torch.zeros(2, device=dev)
    stall = torch.zeros(2, device=dev)
    n = 0
    ns = 0
    for g, p, m in batches:
        f, _ = model(g, p, m, causal=causal, seed_len=seed_len)
        err = (f - p[..., 3:5]).abs() * std
        st = m & ((p[..., 0] * a_sd + a_mu).abs() >= 12.0)
        tot += err[m].sum(0)
        stall += err[st].sum(0)
        n += int(m.sum())
        ns += int(st.sum())
    return {
        "cl": float(tot[0] / n), "cd": float(tot[1] / n),
        "stall_cl": float(stall[0] / max(ns, 1)), "stall_cd": float(stall[1] / max(ns, 1)),
    }


def weight_report(model):
    rows = []
    for name, p in model.named_parameters():
        w = p.detach()
        rows.append({
            "name": name, "numel": w.numel(), "mean_abs": float(w.abs().mean()),
            "max_abs": float(w.abs().max()),
            "frac_lt_1e-2": float((w.abs() < 1e-2).float().mean()),
            "frac_lt_1e-3": float((w.abs() < 1e-3).float().mean()),
        })
    per_channel = {}
    d = model.d_model
    # Which of the d channels does each linear read/write? Row norms of (d, ·) and col norms of (·, d).
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            W = mod.weight.detach()
            if W.shape[0] == d:
                per_channel[f"{name}.out_rows"] = W.norm(dim=1).tolist()
            if W.shape[1] == d:
                per_channel[f"{name}.in_cols"] = W.norm(dim=0).tolist()
        elif isinstance(mod, nn.LayerNorm):
            per_channel[f"{name}.gamma"] = mod.weight.detach().abs().tolist()
        elif isinstance(mod, PathNonlinearity):
            per_channel[f"{name}.a_lin"] = mod.a_lin.detach().tolist()
            per_channel[f"{name}.a_harm_norm"] = mod.a.detach().norm(dim=1).tolist()
    for name, mod in model.named_modules():
        if hasattr(mod, "log_tau"):
            per_channel[f"{name}.tau"] = (F.softplus(mod.log_tau.detach()) + 1e-3).tolist()
        if hasattr(mod, "log_A") and mod.log_A.dim() == 1:
            per_channel[f"{name}.A"] = mod.log_A.detach().exp().tolist()
    return rows, per_channel


@torch.no_grad()
def activation_report(model, batches):
    model.debug = True
    for blk in model.agent:
        blk.debug = True
    if model.config.model_type != "foilcore2":
        model.debug = False
        return {}, {"err_pred_vs_target_corr": [], "err_pred_mean": [], "err_target_mean": []}, None
    stats = {}
    acc = {}
    gate_hist = {"all": [], "stall": []}
    err_pairs = []
    a_std = None
    for g, p, m in batches:
        model(g, p, m, causal=True)
        aux = model.aux
        sel = m
        def add(key, t):
            v = t[sel]
            acc.setdefault(key, []).append(v)
        add("ctx", aux["ctx"]); add("tokens", aux["tokens"]); add("hidden", aux["hidden"]); add("ctx2", aux["ctx2"])
        for i, blk in enumerate(model.agent):
            for k, v in blk.last["paths"].items():
                add(f"block{i}.{k}", v)
            add(f"block{i}.in", blk.last["in"])
        err_pairs.append(torch.cat((aux["err_pred"][m], aux["err_target"][m]), -1))
        for gi, gt in enumerate(aux["gates"]):
            gate_hist["all"].append(gt[m])
            # stall slice needs α in physical units; caller passes normalized, use |z|>1 as proxy stored later
        acc.setdefault("alpha_z", []).append(p[..., 0][m])
    for k, vs in acc.items():
        v = torch.cat(vs)
        if v.dim() == 1:
            continue
        stats[k] = {
            "rms": float(v.pow(2).mean().sqrt()),
            "per_channel_std": v.std(0).tolist(),
            "per_channel_mean_abs": v.abs().mean(0).tolist(),
        }
    ep = torch.cat(err_pairs)
    n_out = ep.shape[1] // 2
    corr = []
    for j in range(n_out):
        x, y = ep[:, j], ep[:, n_out + j]
        corr.append(float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std() + 1e-9)))
    gates = torch.cat(gate_hist["all"]) if gate_hist["all"] else None
    alpha_z = torch.cat(acc["alpha_z"])
    gate_info = None
    if gates is not None:
        n_blk = len(model.agent)
        gates = gates.view(n_blk, -1) if gates.numel() == n_blk * alpha_z.numel() else gates.view(1, -1)
        gate_info = []
        for gb in gates:
            gate_info.append({
                "mean": float(gb.mean()), "frac_on": float((gb > 0.5).float().mean()),
                "frac_on_|alpha_z|>1": float((gb[alpha_z.abs() > 1.0] > 0.5).float().mean()),
                "frac_on_|alpha_z|<=1": float((gb[alpha_z.abs() <= 1.0] > 0.5).float().mean()),
            })
    model.debug = False
    for blk in model.agent:
        blk.debug = False
    return stats, {"err_pred_vs_target_corr": corr, "err_pred_mean": ep[:, :n_out].mean(0).tolist(),
                   "err_target_mean": ep[:, n_out:].mean(0).tolist()}, gate_info


@torch.no_grad()
def knockouts(model, batches, stats):
    base = evaluate(model, batches, stats)
    out = {"baseline_causal": base, "baseline_tf": evaluate(model, batches, stats, causal=False)}
    d = model.d_model

    def run(label, fn):
        m2 = copy.deepcopy(model)
        fn(m2)
        out[label] = evaluate(m2, batches, stats)

    is_v2 = model.config.model_type == "foilcore2"
    for i, blk in enumerate(model.agent):
        if is_v2:
            for path in ("local", "ssm", "attn"):
                if getattr(blk, path) is None:
                    continue
                run(f"block{i}.no_{path}", lambda m2, i=i, path=path: setattr(m2.agent[i], "path_scale", {path: 0.0}))
            run(f"block{i}.identity", lambda m2, i=i: setattr(m2.agent[i], "path_scale", {"local": 0.0, "ssm": 0.0, "attn": 0.0}))
        else:
            run(f"block{i}.no_attn", lambda m2, i=i: setattr(m2.agent[i], "attn", _ZeroAttn()))
            run(f"block{i}.Wv_identity", lambda m2, i=i: setattr(m2.agent[i].attn, "W_v", nn.Identity()))
            run(f"block{i}.tau_off", lambda m2, i=i: setattr(m2.agent[i].attn, "local_log_tau", nn.Parameter(torch.tensor(20.0))))
        run(f"block{i}.no_layernorm", lambda m2, i=i: setattr(m2.agent[i], "norm", nn.Identity()))
    if is_v2:
        run("all_blocks_identity", lambda m2: [setattr(b, "path_scale", {"local": 0.0, "ssm": 0.0, "attn": 0.0}) for b in m2.agent])
        run("err_head_zero", lambda m2: setattr(m2, "err_head", _ZeroErr(m2.err_head)))
        run("no_alpha_sinus", lambda m2: setattr(m2, "_forward_v2", _no_sinus_forward(m2)))
    else:
        run("all_blocks_no_attn", lambda m2: [setattr(b, "attn", _ZeroAttn()) for b in m2.agent])
    run("pathnl_identity", lambda m2: setattr(m2, "path", nn.Identity()))
    for j, nm in enumerate(("global", "le", "te")):
        run(f"drop_memory_{nm}", lambda m2, j=j: setattr(m2, "encode_geom", _drop_mem(m2, j)))
    # per-channel: zero channel c of the corrector hidden state fed to requery
    ch = []
    for c in range(d):
        m2 = copy.deepcopy(model)
        m2.mem_mix = _MaskHidden(m2.mem_mix, c)
        r = evaluate(m2, batches, stats)
        ch.append({"channel": c, "cl": r["cl"], "cd": r["cd"]})
    out["hidden_channel_knockouts"] = ch
    # per-channel: zero channel c of ctx (memory-mix output) → both draft and everything after
    ch2 = []
    for c in range(d):
        m2 = copy.deepcopy(model)
        m2.mem_mix = _MaskOut(m2.mem_mix, c)
        r = evaluate(m2, batches, stats)
        ch2.append({"channel": c, "cl": r["cl"], "cd": r["cd"]})
    out["ctx_channel_knockouts"] = ch2
    return out


class _ZeroAttn(nn.Module):
    def forward(self, x, z, key_padding_mask=None, alpha=None, causal=None):
        return torch.zeros_like(x)


class _ZeroErr(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, memories, cond, clcd):
        return torch.zeros_like(self.inner(memories, cond, clcd))


class _MaskHidden(nn.Module):
    """Wrap LinearMambaMix: zero channel c of ``hidden`` (requery path only)."""

    def __init__(self, inner, c):
        super().__init__()
        self.inner = inner
        self.c = c
        self.n_mem = inner.n_mem

    def forward(self, memories, cond, hidden=None):
        if hidden is not None:
            hidden = hidden.clone()
            hidden[..., self.c] = 0.0
        return self.inner(memories, cond, hidden)


class _MaskOut(nn.Module):
    def __init__(self, inner, c):
        super().__init__()
        self.inner = inner
        self.c = c
        self.n_mem = inner.n_mem

    def forward(self, memories, cond, hidden=None):
        out = self.inner(memories, cond, hidden).clone()
        out[..., self.c] = 0.0
        return out


def _drop_mem(m2, j):
    orig = m2.encode_geom

    def f(coords):
        mem = orig(coords).clone()
        mem[:, j] = 0.0
        return mem
    return f


def _no_sinus_forward(m2):
    import types
    from src import model as M
    orig = M.sinusoidal_encoding_batch

    def f(*a, **k):
        M.sinusoidal_encoding_batch = lambda pos, d, dtype: torch.zeros(pos.shape[0], d, dtype=dtype, device=pos.device)
        try:
            return type(m2)._forward_v2(m2, *a, **k)
        finally:
            M.sinusoidal_encoding_batch = orig
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", default="data/val.csv")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="paper/figures/foilcore2_analysis")
    ap.add_argument("--skip-knockouts", action="store_true")
    args = ap.parse_args()
    device = torch.device(os.environ.get("FOILCORE_DEVICE", "cpu"))
    model, stats, cfg, _ = load_models(args.ckpt, device)
    if cfg.model_type not in ("foilcore2", "foilcore"):
        sys.exit(f"Expected model_type foilcore2|foilcore, got {cfg.model_type}")
    batches = load_batches(args.csv, stats, args.n, args.batch_size, device)
    os.makedirs(args.out, exist_ok=True)

    wrows, per_channel = weight_report(model)
    act, err_info, gate_info = activation_report(model, batches)
    report = {
        "ckpt": args.ckpt, "n_params": sum(p.numel() for p in model.parameters()),
        "weights": wrows, "per_channel": per_channel, "activations": act,
        "error_head": err_info, "router": gate_info,
    }
    if not args.skip_knockouts:
        report["knockouts"] = knockouts(model, batches, stats)
    with open(os.path.join(args.out, "analysis.json"), "w") as f:
        json.dump(report, f, indent=1)

    # ---- console summary ----
    print(f"\n=== {args.ckpt}  ({report['n_params']} params, {args.n} val rows, causal AR) ===")
    print("\nWeights (mean|w|, frac<1e-2):")
    for r in wrows:
        print(f"  {r['name']:40s} n={r['numel']:5d}  mean|w|={r['mean_abs']:.4f}  <1e-2:{r['frac_lt_1e-2']:.2f}")
    print("\nActivation RMS:")
    for k, v in act.items():
        pcs = torch.tensor(v["per_channel_std"])
        print(f"  {k:16s} rms={v['rms']:.3f}  weakest ch std={pcs.min():.3f} (ch {int(pcs.argmin())})  n(ch std<0.05·max)={int((pcs < 0.05 * pcs.max()).sum())}")
    print(f"\nError head: corr(pred,target)={['%.3f' % c for c in err_info['err_pred_vs_target_corr']]}  "
          f"pred mean={['%.3f' % c for c in err_info['err_pred_mean']]} target mean={['%.3f' % c for c in err_info['err_target_mean']]}")
    if gate_info:
        for i, gi in enumerate(gate_info):
            print(f"Router block{i}: mean gate {gi['mean']:.3f}  on {gi['frac_on']:.2f}  on(|α_z|>1) {gi['frac_on_|alpha_z|>1']:.2f}  on(|α_z|<=1) {gi['frac_on_|alpha_z|<=1']:.2f}")
    if "knockouts" in report:
        ko = report["knockouts"]
        b = ko["baseline_causal"]
        print(f"\nKnockouts (causal AR).  baseline Cl {b['cl']:.4f} Cd {b['cd']:.5f} | stall Cl {b['stall_cl']:.4f} Cd {b['stall_cd']:.5f}")
        t = ko["baseline_tf"]
        print(f"  (TF for reference: Cl {t['cl']:.4f} Cd {t['cd']:.5f})")
        for k, v in ko.items():
            if k.startswith("baseline") or k.endswith("_knockouts"):
                continue
            print(f"  {k:24s} Cl {v['cl']:.4f} ({v['cl']-b['cl']:+.4f})  Cd {v['cd']:.5f} ({v['cd']-b['cd']:+.5f})  stall Cd {v['stall_cd']:.5f}")
        for key in ("hidden_channel_knockouts", "ctx_channel_knockouts"):
            rows = sorted(ko[key], key=lambda r: r["cl"] + 31 * r["cd"] / (stats["polar_std"][4] / stats["polar_std"][3]))
            print(f"\n  {key}: least load-bearing channels (Cl,Cd after zeroing):")
            for r in rows[:8]:
                print(f"    ch {r['channel']:2d}  Cl {r['cl']:.4f} ({r['cl']-b['cl']:+.4f})  Cd {r['cd']:.5f} ({r['cd']-b['cd']:+.5f})")
            print(f"  most load-bearing: " + ", ".join(f"ch{r['channel']}({r['cl']-b['cl']:+.3f})" for r in rows[-5:]))
    print(f"\nWrote {os.path.join(args.out, 'analysis.json')}")


if __name__ == "__main__":
    main()
