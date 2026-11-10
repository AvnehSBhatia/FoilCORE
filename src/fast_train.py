"""GPU-resident fast trainer for FoilCORE variants.

Differences from ``src/train_model.py`` (same model, same losses, same checkpoint keys):
  * dataset is tensorized once (unique geometries + padded polars) and cached, then lives on the GPU;
    batches are gathered by index — no DataLoader, no per-item Python.
  * length-bucketed batches: rows sorted by polar length, batch order shuffled → ~no padding.
  * validation runs in **causal AR** mode (what we care about) and TF; best = causal val loss.
  * optional torch.compile (CUDA), optional bf16 autocast.
  * multiple runs fit on one GPU: just launch several processes.

python -m src.fast_train --variant-id 50 --save-dir checkpoints/foilcore2 --epochs 100 --batch-size 1024
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.ablations import get_variant_spec  # noqa: E402
from src.data import AeroDataset, build_polar_tokens, default_aero_cache_dir, normalize_coords, resample_airfoil  # noqa: E402
from src.model import FoilCORE, FoilCoreConfig, count_parameters  # noqa: E402
from src.train_model import get_device, set_training_seed  # noqa: E402
from src.utils import POLAR_LOSS_WEIGHTS_5, polar_clcd_components, set_polar_loss_weights  # noqa: E402

TENSOR_CACHE_VERSION = 1


# ----------------------------------------------------------------------------- data
def tensorize(csv_path: str, stats: dict | None, training: bool, cache_dir: Path | None = None) -> dict:
    """Parse (or load cached) CSV → dict of tensors on CPU. Stats come from the train split."""
    csv_path = Path(csv_path)
    cdir = Path(cache_dir) if cache_dir else default_aero_cache_dir(csv_path)
    cdir.mkdir(parents=True, exist_ok=True)
    sig = f"{csv_path.stat().st_size}-{int(csv_path.stat().st_mtime)}-v{TENSOR_CACHE_VERSION}"
    tpath = cdir / f"{csv_path.stem}_tensors_{hashlib.md5(sig.encode()).hexdigest()[:10]}.pt"
    if tpath.is_file():
        blob = torch.load(tpath, map_location="cpu", weights_only=False)
        if stats is not None:
            blob["stats"] = stats
            blob["polar"] = _renormalize(blob, stats)
        print(f"[fast_train] loaded tensor cache {tpath.name}: {blob['polar'].shape[0]} rows, T_max={blob['polar'].shape[1]}")
        return blob
    ds = AeroDataset(str(csv_path), stats=stats, training=training)
    stats = ds.stats
    df = ds.df
    n = len(df)
    print(f"[fast_train] tensorizing {n} rows from {csv_path.name} ...")
    # unique geometries
    geom_key_to_idx: dict[str, int] = {}
    geom_list: list[np.ndarray] = []
    row_geom = np.zeros(n, dtype=np.int64)
    coords_col = df["coords"].values
    for i in range(n):
        c = coords_col[i]
        key = hashlib.md5(np.asarray(c, dtype=np.float64).tobytes()).hexdigest()
        j = geom_key_to_idx.get(key)
        if j is None:
            j = len(geom_list)
            geom_key_to_idx[key] = j
            cr = resample_airfoil(c, n_points=200)
            cn, _, _ = normalize_coords(cr)
            geom_list.append(cn.astype(np.float32))
        row_geom[i] = j
    geom = torch.from_numpy(np.stack(geom_list))
    # polars (store raw physical, normalize on load so a stats swap is cheap)
    alphas, cls, cds = df["alpha"].values, df["Cl"].values, df["Cd"].values
    res, machs = df["Re"].values, df["mach"].values
    lengths = np.array([len(a) for a in alphas], dtype=np.int64)
    t_max = int(lengths.max())
    raw = np.zeros((n, t_max, 5), dtype=np.float32)
    for i in range(n):
        tok = build_polar_tokens(np.array(alphas[i]), np.array(cls[i]), np.array(cds[i]), res[i], machs[i])
        raw[i, : tok.shape[0]] = tok
    blob = {
        "cache_version": TENSOR_CACHE_VERSION,
        "geom": geom, "row_geom": torch.from_numpy(row_geom),
        "polar_raw": torch.from_numpy(raw), "lengths": torch.from_numpy(lengths),
        "stats": stats,
    }
    blob["polar"] = _renormalize(blob, stats)
    torch.save(blob, tpath)
    print(f"[fast_train] wrote {tpath.name}: {n} rows, {geom.shape[0]} unique geometries, T_max={t_max}")
    return blob


def _renormalize(blob: dict, stats: dict) -> torch.Tensor:
    mean = torch.as_tensor(np.asarray(stats["polar_mean"], dtype=np.float32))
    std = torch.as_tensor(np.asarray(stats["polar_std"], dtype=np.float32))
    pol = (blob["polar_raw"] - mean) / std
    # zero the padding again (raw padding is 0 → normalized would be -mean/std)
    t_max = pol.shape[1]
    m = torch.arange(t_max)[None, :] < blob["lengths"][:, None]
    return pol * m.unsqueeze(-1)


class GpuPolarData:
    def __init__(self, blob: dict, device: torch.device, pad_multiple: int = 8):
        self.device = device
        self.geom = blob["geom"].to(device)
        self.row_geom = blob["row_geom"].to(device)
        self.polar = blob["polar"].to(device)
        self.lengths = blob["lengths"].to(device)
        self.stats = blob["stats"]
        self.n = self.polar.shape[0]
        self.pad_multiple = int(pad_multiple)
        self._len_cpu = blob["lengths"].numpy()

    def batches(self, batch_size: int, shuffle: bool, gen: torch.Generator | None = None):
        """Length-bucketed index batches. Sort by length (with noise when shuffling), chunk, shuffle chunks."""
        if shuffle:
            noise = np.random.default_rng(int(torch.randint(0, 2**31, (1,), generator=gen).item())).random(self.n)
            order = np.lexsort((noise, self._len_cpu + np.random.default_rng(int(noise[0] * 1e9)).integers(0, 4, self.n)))
        else:
            order = np.argsort(self._len_cpu, kind="stable")
        chunks = [order[i : i + batch_size] for i in range(0, self.n, batch_size)]
        if shuffle:
            perm = np.random.default_rng(int(noise[1] * 1e9)).permutation(len(chunks))
            chunks = [chunks[i] for i in perm]
        for ch in chunks:
            yield torch.as_tensor(ch, device=self.device)

    def gather(self, idx: torch.Tensor):
        lens = self.lengths[idx]
        t = int(lens.max().item())
        t = int(math.ceil(t / self.pad_multiple) * self.pad_multiple)
        t = min(t, self.polar.shape[1])
        polar = self.polar[idx, :t]
        mask = torch.arange(t, device=self.device)[None, :] < lens[:, None]
        geom = self.geom[self.row_geom[idx]]
        return geom, polar, mask


# ----------------------------------------------------------------------------- model
def build_model(variant, d_model_override=None) -> tuple[FoilCORE, FoilCoreConfig]:
    kw = {k: v for k, v in variant.model.items() if hasattr(FoilCoreConfig(), k)}
    if d_model_override is not None:
        kw["d_model"] = int(d_model_override)
    kw.setdefault("d_geom", kw.get("d_model", 32))
    kw.setdefault("mlp_hidden", kw.get("d_model", 32))
    kw["use_causal_mask"] = bool(variant.training.get("use_causal_mask", variant.model.get("use_causal_mask", False)))
    kw["draft_loss_weight"] = float(variant.model.get("draft_loss_weight", variant.training.get("draft_loss_weight", 0.25)))
    cfg = FoilCoreConfig(**kw)
    return FoilCORE(config=cfg), cfg


def load_embeddings(model: FoilCORE, path: str) -> None:
    if not os.path.exists(path):
        print(f"[fast_train] no embedding ckpt at {path}; random geom encoder")
        return
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw = ckpt.get("geom_encoder") or ckpt.get("embed_layer")
    if isinstance(raw, dict):
        cur = model.geom_encoder.state_dict()
        compat = {k.split("geom_encoder.", 1)[-1]: v for k, v in raw.items()}
        compat = {k: v for k, v in compat.items() if k in cur and v.shape == cur[k].shape}
        model.geom_encoder.load_state_dict(compat, strict=False)
        print(f"[fast_train] loaded {len(compat)} geom encoder tensors")
    if isinstance(ckpt.get("path"), dict) and hasattr(model, "path") and hasattr(model.path, "state_dict"):
        cur = model.path.state_dict()
        pc = {k: v for k, v in ckpt["path"].items() if k in cur and v.shape == cur[k].shape}
        if pc:
            model.path.load_state_dict(pc, strict=False)
            print(f"[fast_train] loaded {len(pc)} PathNonlinearity tensors")


# ----------------------------------------------------------------------------- loss
def compute_loss(model, final, draft, polar, mask, *, draft_w, err_w, router_w, polar_w,
                 stall_w=0.0, a_mu=0.0, a_sd=1.0, cd_mode="mae", cd_mu=0.0, cd_sd=1.0):
    comp = polar_clcd_components(final, polar, mask)
    loss = comp["total"]
    parts = {"cl": comp["polar_cl_mae"].detach(), "cd": comp["polar_cd_mae"].detach()}
    if draft_w > 0:
        aux = polar_clcd_components(draft, polar, mask)
        loss = loss + draft_w * aux["total"]
    if cd_mode != "mae":
        p_phys = final[..., 1] * cd_sd + cd_mu
        t_phys = polar[..., 4] * cd_sd + cd_mu
        if cd_mode == "rel":
            cd_term = ((p_phys - t_phys).abs() / (t_phys.abs() + 1e-4))[mask].mean()
        else:
            cd_term = (torch.log1p(p_phys.clamp_min(-0.009) * 100.0) - torch.log1p(t_phys.clamp_min(0.0) * 100.0)).abs()[mask].mean()
        w = POLAR_LOSS_WEIGHTS_5.to(device=final.device, dtype=final.dtype)
        loss = w[3] * comp["polar_cl_mae"] + w[4] * cd_term
        parts[cd_mode] = cd_term.detach()
    if stall_w > 0:
        alpha_phys = polar[..., 0] * a_sd + a_mu
        stall = mask & (alpha_phys.abs() >= 12.0)
        if stall.any():
            s_pred = final[stall]
            s_tgt = polar[stall][..., 3:5]
            w = POLAR_LOSS_WEIGHTS_5.to(device=s_pred.device, dtype=s_pred.dtype)
            s_cl = F.l1_loss(s_pred[..., 0], s_tgt[..., 0])
            s_cd = F.l1_loss(s_pred[..., 1], s_tgt[..., 1])
            stall_loss = w[3] * s_cl + w[4] * s_cd
            loss = loss + stall_w * stall_loss
            parts["stall"] = stall_loss.detach()
    a = getattr(model, "aux", None) or {}
    if err_w > 0 and "err_pred" in a:
        ep, et = a["err_pred"][mask], a["err_target"][mask]
        e = ((ep - et).abs() * polar_w).sum(-1).mean()
        loss = loss + err_w * e
        parts["err"] = e.detach()
    if router_w > 0 and a.get("gates"):
        gsum = torch.stack([g[mask].mean() for g in a["gates"]]).mean()
        loss = loss + router_w * gsum
        parts["gate"] = gsum.detach()
    return loss, parts


@torch.no_grad()
def evaluate(model, data: GpuPolarData, batch_size, *, causal: bool, draft_w, polar_w, err_w=0.0):
    model.eval()
    std = torch.as_tensor(np.asarray(data.stats["polar_std"][3:5], dtype=np.float32), device=data.device)
    a_mu, a_sd = float(data.stats["polar_mean"][0]), float(data.stats["polar_std"][0])
    tot_loss = 0.0
    nb = 0
    err_sum = torch.zeros(2, device=data.device)
    stall_sum = torch.zeros(2, device=data.device)
    npts = 0
    nstall = 0
    for idx in data.batches(batch_size, shuffle=False):
        geom, polar, mask = data.gather(idx)
        final, draft = model(geom, polar, mask, causal=causal)
        loss, _ = compute_loss(model, final, draft, polar, mask, draft_w=draft_w, err_w=err_w, router_w=0.0, polar_w=polar_w)
        tot_loss += float(loss)
        nb += 1
        e = (final - polar[..., 3:5]).abs() * std
        st = mask & ((polar[..., 0] * a_sd + a_mu).abs() >= 12.0)
        err_sum += e[mask].sum(0)
        stall_sum += e[st].sum(0)
        npts += int(mask.sum())
        nstall += int(st.sum())
    model.train()
    return {
        "loss": tot_loss / max(nb, 1),
        "cl": float(err_sum[0] / npts), "cd": float(err_sum[1] / npts),
        "stall_cl": float(stall_sum[0] / max(nstall, 1)), "stall_cd": float(stall_sum[1] / max(nstall, 1)),
    }


# ----------------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser(description="Fast GPU-resident FoilCORE trainer")
    ap.add_argument("--variant-id", type=int, default=0)
    ap.add_argument("--save-dir", default="checkpoints/fast")
    ap.add_argument("--train-csv", default="data/train.csv")
    ap.add_argument("--val-csv", default="data/val.csv")
    ap.add_argument("--embed-ckpt", default="checkpoints/embeddings_best.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--val-batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--plateau-patience", type=int, default=4)
    ap.add_argument("--plateau-factor", type=float, default=0.5)
    ap.add_argument("--min-lr-ratio", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--resume", default=None, help="checkpoint to resume weights/epoch from")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--tag", default="", help="free-form label stored in the checkpoint")
    ap.add_argument("--early-stop-patience", type=int, default=0, help="stop after N epochs without val-AR improvement (0=off)")
    ap.add_argument("--train-frac", type=float, default=1.0, help="fraction of unique training geometries to keep (label-efficiency runs)")
    ap.add_argument("--resume-history", default=None, help="metrics_history.json of the run being continued (history is appended)")
    ap.add_argument("--stall-loss-weight", type=float, default=0.0, help="extra loss weight on |alpha|>=12deg band (training only)")
    ap.add_argument("--cd-loss-mode", default="mae", choices=["mae", "rel", "log"], help="Cd loss: mae (norm) | rel (physical rel err) | log (physical log-space)")
    ap.add_argument("--causal-mask", action="store_true", help="force causal attention mask in the polar corrector (train=inference protocol)")
    ap.add_argument("--phys-base", action="store_true", help="Cl = 2pi sin(a-a0)/sqrt(1-Ma^2) analytic base + learned residual")
    ap.add_argument("--invariant-memory", action="store_true", help="5th geom memory: [r_LE, theta_TE, t_max, x_tmax, c_max, x_cmax]")
    ap.add_argument("--bl-features", action="store_true", help="scale-separation features [log10Re, 1/sqrtRe, 1/Re^0.2, |a|/sqrt(r_LE)]")
    ap.add_argument("--phys-residual", action="store_true", help="Cd = Cd0 + k*Cl^2 + delta with geometry-conditioned (Cd0, k)")
    ap.add_argument("--phys-residual-gated", action="store_true", help="per-point blend g*(Cd0+kCl^2+d) + (1-g)*learned Cd (implies --phys-residual)")
    ap.add_argument("--causal-polish-epochs", type=int, default=0, help="extra epochs at the end with forced causal drafts + low LR")
    ap.add_argument("--polish-lr-factor", type=float, default=0.2, help="LR multiplier during causal polish")
    args = ap.parse_args()

    set_training_seed(args.seed)
    device = torch.device(get_device())
    print(f"[fast_train] device {device}  variant {args.variant_id}")
    variant = get_variant_spec(args.variant_id)
    print(f"[fast_train] {variant.name} [{variant.slug}]")
    if args.causal_mask:
        tr = dict(variant.training)
        tr["use_causal_mask"] = True
        variant = dataclasses.replace(variant, training=tr)
        print("[fast_train] overriding use_causal_mask=True (causal corrector)")
    else:
        tr = variant.training
    phys_overrides = {}
    if args.phys_base:
        phys_overrides["phys_base"] = True
    if args.invariant_memory:
        phys_overrides["invariant_memory"] = True
    if args.bl_features:
        phys_overrides["bl_features"] = True
    if args.phys_residual:
        phys_overrides["phys_residual"] = True
    if args.phys_residual_gated:
        phys_overrides["phys_residual"] = True
        phys_overrides["phys_residual_gated"] = True
    if phys_overrides:
        variant = dataclasses.replace(variant, model={**variant.model, **phys_overrides})
        print(f"[fast_train] physics overrides: {sorted(phys_overrides)}")
    polar_loss_weights = tr.get("polar_loss_weights", [0.0, 0.0, 0.0, 1.0, 31.0])
    set_polar_loss_weights(polar_loss_weights)
    polar_w = torch.tensor(polar_loss_weights[3:5], dtype=torch.float32, device=device)
    scheduled_sampling = bool(tr.get("scheduled_sampling", True))
    draft_w = float(variant.model.get("draft_loss_weight", tr.get("draft_loss_weight", 0.25)))
    err_w = float(tr.get("err_loss_weight", 0.0))
    router_w = float(tr.get("router_sparsity_weight", 0.0))
    causal_p = float(tr.get("causal_train_prob", 0.0))
    stall_w = float(tr.get("stall_loss_weight", args.stall_loss_weight))
    from src.train_model import ar_mix_probability

    train_blob = tensorize(args.train_csv, None, True)
    stats = train_blob["stats"]
    a_mu, a_sd = float(stats["polar_mean"][0]), float(stats["polar_std"][0])
    cd_mu, cd_sd = float(stats["polar_mean"][4]), float(stats["polar_std"][4])
    if args.train_frac < 1.0:
        gsel = torch.rand(int(train_blob["geom"].shape[0]), generator=torch.Generator().manual_seed(args.seed)) < args.train_frac
        keep = gsel[train_blob["row_geom"]]
        for k in ("polar", "polar_raw", "lengths", "row_geom"):
            train_blob[k] = train_blob[k][keep]
        print(f"[fast_train] train-frac {args.train_frac}: {int(gsel.sum())}/{gsel.numel()} geometries, {int(keep.sum())} rows")
    val_blob = tensorize(args.val_csv, stats, False)
    train_data = GpuPolarData(train_blob, device)
    val_data = GpuPolarData(val_blob, device)
    del train_blob, val_blob

    model, cfg = build_model(variant, args.d_model)
    if args.phys_base or args.phys_residual or args.bl_features:
        cfg.a_mu, cfg.a_sd = float(stats["polar_mean"][0]), float(stats["polar_std"][0])
        cfg.m_mu, cfg.m_sd = float(stats["polar_mean"][2]), float(stats["polar_std"][2])
        cfg.cl_mu, cfg.cl_sd = float(stats["polar_mean"][3]), float(stats["polar_std"][3])
        cfg.cd_mu, cfg.cd_sd = float(stats["polar_mean"][4]), float(stats["polar_std"][4])
        cfg.re_mu, cfg.re_sd = float(stats["polar_mean"][1]), float(stats["polar_std"][1])
        print(f"[fast_train] physics stats: a {cfg.a_mu:.3f}+-{cfg.a_sd:.3f}  m {cfg.m_mu:.3f}+-{cfg.m_sd:.3f}  "
              f"cl {cfg.cl_mu:.3f}+-{cfg.cl_sd:.3f}  cd {cfg.cd_mu:.4f}+-{cfg.cd_sd:.4f}  re {cfg.re_mu:.3e}+-{cfg.re_sd:.3e}")
    load_embeddings(model, args.embed_ckpt)
    model = model.to(device)
    n_params = count_parameters(model)
    print(f"[fast_train] {n_params} params  bs={args.batch_size}  lr={args.lr}  compile={args.compile}  bf16={args.bf16}")

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = model.state_dict()
        compat = {k: v for k, v in ck["model"].items() if k in sd and v.shape == sd[k].shape}
        model.load_state_dict(compat, strict=False)
        start_epoch = int(ck.get("epoch", 0))
        print(f"[fast_train] resumed {len(compat)} tensors from {args.resume} (epoch {start_epoch})")

    fwd = torch.compile(model, dynamic=True) if args.compile else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=args.plateau_factor, patience=args.plateau_patience,
        threshold=1e-4, min_lr=args.lr * args.min_lr_ratio, cooldown=1,
    )
    os.makedirs(args.save_dir, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed + start_epoch)
    history = []
    best = float("inf")
    if args.resume:
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
            print("[fast_train] restored optimizer state")
        if "plateau" in ck:
            plateau.load_state_dict(ck["plateau"])
            print("[fast_train] restored plateau scheduler state")
        for g in opt.param_groups:
            g["lr"] = args.lr  # explicit --lr wins (use the LR the run ended at)
        best = float(ck.get("best_val_loss", float("inf")))
        if args.resume_history and os.path.isfile(args.resume_history):
            history = json.load(open(args.resume_history)).get("history", [])
            print(f"[fast_train] appended to {len(history)} epochs of history")
    bad_epochs = 0
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if (args.bf16 and device.type == "cuda") else (lambda: torch.autocast("cpu", enabled=False))

    def save(path, epoch, val):
        torch.save({
            "epoch": epoch, "rng_seed": args.seed, "model": model.state_dict(),
            "geom_encoder": model.geom_encoder.state_dict(), "n_params": n_params,
            "variant": {"id": variant.idx, "slug": variant.slug, "name": variant.name},
            "variant_model": cfg.__dict__, "variant_training": tr,
            "stats": stats, "val_loss": val["loss"], "best_val_loss": best, "val_causal": val,
            "trainer": "fast_train", "tag": args.tag,
            "optimizer": opt.state_dict(), "plateau": plateau.state_dict(),
        }, path)

    total_epochs = args.epochs + args.causal_polish_epochs
    for epoch in range(start_epoch + 1, total_epochs + 1):
        polishing = epoch > args.epochs
        if polishing:
            for g in opt.param_groups:
                g["lr"] = args.lr * args.polish_lr_factor
            p_ar = 1.0
        else:
            if epoch <= args.warmup_epochs:
                for g in opt.param_groups:
                    g["lr"] = args.lr * epoch / args.warmup_epochs
            p_ar = ar_mix_probability(epoch) if scheduled_sampling else 1.0
        model.train()
        t0 = time.time()
        sums = {}
        nb = 0
        for idx in train_data.batches(args.batch_size, shuffle=True, gen=gen):
            geom, polar, mask = train_data.gather(idx)
            causal = True if polishing else (bool(torch.rand((), generator=gen).item() < causal_p) if causal_p > 0 else None)
            with amp_ctx():
                final, draft = fwd(geom, polar, mask, p_ar=p_ar, causal=causal)
                loss, parts = compute_loss(model, final.float(), draft.float(), polar, mask,
                                           draft_w=draft_w, err_w=err_w, router_w=router_w, polar_w=polar_w,
                                           stall_w=stall_w, a_mu=a_mu, a_sd=a_sd,
                                           cd_mode=args.cd_loss_mode, cd_mu=cd_mu, cd_sd=cd_sd)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sums["loss"] = sums.get("loss", 0.0) + float(loss.detach())
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            nb += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_train = time.time() - t0
        t0 = time.time()
        vc = evaluate(model, val_data, args.val_batch_size, causal=True, draft_w=draft_w, polar_w=polar_w, err_w=err_w)
        vt = evaluate(model, val_data, args.val_batch_size, causal=False, draft_w=draft_w, polar_w=polar_w, err_w=err_w)
        t_val = time.time() - t0
        if not polishing and epoch > args.warmup_epochs:
            plateau.step(vc["loss"])
        lr_now = opt.param_groups[0]["lr"]
        rec = {"epoch": epoch, "lr": lr_now, "p_ar": p_ar, "train_time_s": t_train, "val_time_s": t_val,
               **{f"train_{k}": v / max(nb, 1) for k, v in sums.items()},
               **{f"val_ar_{k}": v for k, v in vc.items()}, **{f"val_tf_{k}": v for k, v in vt.items()}}
        history.append(rec)
        extra = "".join(f" {k}={sums[k]/nb:.4f}" for k in ("err", "gate", "stall", "rel", "log") if k in sums)
        print(
            f"Epoch {epoch:3d}/{total_epochs}{'|polish' if polishing else '        '} | {t_train:.0f}s+{t_val:.0f}s | train L {sums['loss']/nb:.4f}"
            f" (Cl {sums['cl']/nb:.4f} Cd {sums['cd']/nb:.4f}){extra} | "
            f"val AR L {vc['loss']:.4f} Cl {vc['cl']:.4f} Cd {vc['cd']:.5f} stallCd {vc['stall_cd']:.5f}"
            f" | TF Cl {vt['cl']:.4f} Cd {vt['cd']:.5f} | lr {lr_now:.2e}",
            flush=True,
        )
        if epoch % args.save_every == 0 or epoch == 1:
            save(os.path.join(args.save_dir, f"foilcore_epoch_{epoch:03d}.pt"), epoch, vc)
        save(os.path.join(args.save_dir, "foilcore_last.pt"), epoch, vc)
        if vc["loss"] < best - 1e-6:
            best = vc["loss"]
            bad_epochs = 0
            save(os.path.join(args.save_dir, "foilcore_best.pt"), epoch, vc)
        else:
            bad_epochs += 1
        with open(os.path.join(args.save_dir, "metrics_history.json"), "w") as f:
            json.dump({"variant": {"id": variant.idx, "slug": variant.slug}, "n_params": n_params,
                       "args": vars(args), "history": history}, f, indent=1)
        if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
            print(f"[fast_train] early stop: no val-AR improvement for {bad_epochs} epochs (best {best:.5f})", flush=True)
            break
    print(f"[fast_train] done. best val AR loss {best:.5f}")


if __name__ == "__main__":
    main()
