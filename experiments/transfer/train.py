"""Train the generic draft→corrector model on a public dataset under one of four protocols and
report causal (streaming) and bidirectional (smoothing) test MAE.

  gtmix        legacy: scheduled sampling of the ground truth into the corrector, bidirectional batches
  draft_causal draft-only, every batch causal              (streaming specialist)
  draft_bidir  draft-only, every batch bidirectional       (smoothing specialist)
  draft_dual   draft-only, causal with p=0.5               (FoilCORE's recipe)
  gtmix_consist gtmix + both masks every batch + L2 consistency between the two outputs (unified-ASR-style)

--q Q fixes the leak fraction (ground truth replaces the corrector's input w.p. Q at every position) instead of
the gtmix anneal; the mask still follows --protocol (draft_dual + --q = dual mask with a fixed leak).
--draft-noise S adds N(0,S^2) (z units) to the draft the corrector reads, at train and test time.

python experiments/transfer/train.py --dataset ett --protocol draft_dual --seed 0 --out runs/transfer/ett_draft_dual_s0.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from experiments.transfer.datasets import load  # noqa: E402
from experiments.transfer.model import DraftCorrector  # noqa: E402
from src.model import count_parameters  # noqa: E402
from src.train_model import ar_mix_probability, get_device  # noqa: E402

PROTOCOLS = ("gtmix", "draft_causal", "draft_bidir", "draft_dual", "gtmix_consist")


@torch.no_grad()
def evaluate(model, x, y, *, causal: bool, bs: int, y_std: float, q: float = 0.0, draft_noise: float = 0.0) -> dict:
    """q > 0 evaluates the training-time (leaked) risk: ground truth replaces the corrector input w.p. q."""
    model.eval()
    torch.manual_seed(1234)
    err_f, err_d, n = 0.0, 0.0, 0
    for i in range(0, len(x), bs):
        xb, yb = x[i : i + bs], y[i : i + bs]
        f, d = model(xb, yb if q > 0 else None, causal=causal, p_ar=1.0 - q, draft_noise=draft_noise)
        err_f += float((f - yb).abs().sum())
        err_d += float((d - yb).abs().sum())
        n += yb.numel()
    model.train()
    return {"mae": err_f / n * y_std, "draft_mae": err_d / n * y_std, "mae_z": err_f / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ett", "air", "ecg", "ett_lag1"])
    ap.add_argument("--protocol", required=True, choices=PROTOCOLS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None, help="default 150 (ecg: 60)")
    ap.add_argument("--batch-size", type=int, default=None, help="default 64 (ecg: 256)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--corrector", default="rpa", choices=["rpa", "transformer", "mamba", "tsmixer"])
    ap.add_argument("--patience", type=int, default=None, help="default 25 (ecg: 12)")
    ap.add_argument("--q", type=float, default=None, help="fixed leak fraction (replaces the gtmix anneal)")
    ap.add_argument("--draft-noise", type=float, default=0.0)
    ap.add_argument("--consist-weight", type=float, default=1.0, help="gtmix_consist only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    big = args.dataset == "ecg"  # 38k training windows of 128 samples: fewer, larger epochs
    args.epochs = args.epochs or (60 if big else 150)
    args.batch_size = args.batch_size or (256 if big else 64)
    args.patience = args.patience or (12 if big else 25)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(get_device())
    blob = load(args.dataset)
    to = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=dev)
    xtr, ytr, xva, yva, xte, yte = map(to, (blob["x_train"], blob["y_train"], blob["x_val"], blob["y_val"], blob["x_test"], blob["y_test"]))
    y_std = float(blob["y_std"])
    model = DraftCorrector(xtr.shape[-1], d=args.d, corrector=args.corrector, t_max=int(xtr.shape[1])).to(dev)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=6, min_lr=args.lr * 1e-3)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    print(f"[transfer] {args.dataset} {blob['name']} | {args.protocol} corrector={args.corrector} seed {args.seed} | {n_params} params | "
          f"train {tuple(xtr.shape)} val {tuple(xva.shape)} test {tuple(xte.shape)} | {dev}", flush=True)

    best, best_state, bad, hist = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        leaky = args.protocol in ("gtmix", "gtmix_consist") or args.q is not None
        if args.q is not None:
            p_ar = 1.0 - args.q
        else:
            p_ar = ar_mix_probability(ep) if leaky else 1.0
        perm = torch.randperm(len(xtr), generator=gen).to(dev)
        tot, nb = 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb, yb = xtr[idx], ytr[idx]
            dn = args.draft_noise
            if args.protocol == "gtmix_consist":
                fc, dc = model(xb, yb, causal=True, p_ar=p_ar, draft_noise=dn)
                fb, _ = model(xb, yb, causal=False, p_ar=p_ar, draft_noise=dn)
                loss = 0.5 * ((fc - yb).abs().mean() + (fb - yb).abs().mean()) + 0.25 * (dc - yb).abs().mean() \
                    + args.consist_weight * (fc - fb).pow(2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss)
                nb += 1
                continue
            if args.protocol == "gtmix":
                causal = False
            elif args.protocol == "draft_causal":
                causal = True
            elif args.protocol == "draft_bidir":
                causal = False
            else:
                causal = bool(torch.rand((), generator=gen).item() < 0.5)
            f, d = model(xb, yb if leaky else None, causal=causal, p_ar=p_ar, draft_noise=dn)
            loss = (f - yb).abs().mean() + 0.25 * (d - yb).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        vc = evaluate(model, xva, yva, causal=True, bs=512, y_std=y_std, draft_noise=args.draft_noise)
        vb = evaluate(model, xva, yva, causal=False, bs=512, y_std=y_std, draft_noise=args.draft_noise)
        # model selection in the protocol's native mode (val always sees the model's own draft)
        if args.protocol == "draft_causal":
            sel = vc["mae_z"]
        elif args.protocol in ("draft_bidir", "gtmix"):
            sel = vb["mae_z"]
        else:  # draft_dual, gtmix_consist
            sel = 0.5 * (vc["mae_z"] + vb["mae_z"])
        sched.step(sel)
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_causal": vc["mae"], "val_bidir": vb["mae"], "p_ar": p_ar})
        if sel < best - 1e-6:
            best, bad = sel, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:3d} loss {tot/max(nb,1):.4f} | val causal {vc['mae']:.4f} bidir {vb['mae']:.4f} | p_ar {p_ar:.2f}", flush=True)
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    dn = args.draft_noise
    tc = evaluate(model, xte, yte, causal=True, bs=512, y_std=y_std, draft_noise=dn)
    tb = evaluate(model, xte, yte, causal=False, bs=512, y_std=y_std, draft_noise=dn)
    # Proposition 1 check: training-time (leaked) risk vs deployment risk, and E|y - ỹ_draft| per position
    q_eval = args.q if args.q is not None else 0.0
    lc = evaluate(model, xte, yte, causal=True, bs=512, y_std=y_std, q=q_eval, draft_noise=dn) if q_eval > 0 else tc
    lb = evaluate(model, xte, yte, causal=False, bs=512, y_std=y_std, q=q_eval, draft_noise=dn) if q_eval > 0 else tb
    with torch.no_grad():
        torch.manual_seed(1234)
        dz = model.draft_head(model.draft_mlp(xte)).squeeze(-1)
        dz_in = dz + dn * torch.randn_like(dz) if dn > 0 else dz
        draft_in_err_z = float((dz_in - yte).abs().mean())
    res = {
        "dataset": args.dataset, "dataset_name": str(blob["name"]), "unit": str(blob["unit"]), "protocol": args.protocol, "corrector": args.corrector,
        "seed": args.seed, "q": args.q, "draft_noise": args.draft_noise, "n_params": n_params, "epochs_run": len(hist), "train_time_s": time.time() - t0,
        "test_causal_mae": tc["mae"], "test_bidir_mae": tb["mae"], "test_draft_mae": tc["draft_mae"],
        "test_causal_mae_z": tc["mae_z"], "test_bidir_mae_z": tb["mae_z"],
        "gap_pct": 100.0 * (tc["mae"] - tb["mae"]) / max(tb["mae"], 1e-9),
        "test_leaked_causal_mae_z": lc["mae_z"], "test_leaked_bidir_mae_z": lb["mae_z"],
        "test_corrector_input_err_z": draft_in_err_z,
        "history": hist,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[transfer] TEST {args.dataset} {args.protocol} [{args.corrector}] s{args.seed}: causal {tc['mae']:.4f} bidir {tb['mae']:.4f} "
          f"draft {tc['draft_mae']:.4f} ({res['unit']}) gap {res['gap_pct']:+.1f}%", flush=True)


if __name__ == "__main__":
    main()
