# train.py - CRITICAL FIX: Remove torch.compile, add faster batch building

import sys
from pathlib import Path

# Allow ``python src/train_model.py`` (repo root must be on path for ``import src.*``).
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LambdaLR,
    ReduceLROnPlateau,
)
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext
from src.data import (
    AeroDataset,
    collate_variable_sequences,
    patches_to_coords,
    polar_length_mask,
)
from src.model import FoilCORE, FoilCoreConfig, count_parameters, remap_foilcore_state_dict
from src.utils import polar_clcd_components, set_polar_loss_weights
from src.ablations import get_variant_spec
# Training/val polar term: 1*MAE(Cl_z) + 31*MAE(Cd_z); logs raw per-channel MAEs separately.
# Geometry and AoA/Re/Mach are input-only / teacher-forced; specials are not predicted.
import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm
import warnings
import time
import json

# Suppress warnings
warnings.filterwarnings('ignore')

# ------------------ device helper ------------------
def get_device():
    """
    Select compute device. Override with env ``FOILCORE_DEVICE`` = ``cuda`` | ``mps`` | ``cpu``
    (useful to force MPS when multiple backends are visible).
    """
    import os

    force = os.environ.get("FOILCORE_DEVICE", "").strip().lower()
    if force == "cpu":
        return "cpu"
    if force == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("Note: FOILCORE_DEVICE=cuda but CUDA is unavailable; falling back to auto.")
    if force == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        print("Note: FOILCORE_DEVICE=mps but MPS is unavailable; falling back to auto.")
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_training_seed(seed: int) -> None:
    """Deterministic defaults for a single run (Python, NumPy, Torch)."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def warmup_lambda(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return float(epoch) / float(max(1, warmup_epochs))
    return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))))


def ar_mix_probability(epoch: int) -> float:
    """
    Fraction of polar timesteps using predicted Cl/Cd embeddings (scheduled sampling).

    Ep 1-3: 0%; then +3% each block of 3 epochs (4-6: 3%, 7-9: 6%, …), capped at 100%.
    """
    if epoch <= 3:
        return 0.0
    p = 0.03 * (((epoch - 4) // 3) + 1)
    return float(min(1.0, p))


# Polar layout [AoA, Re, Mach, Cl, Cd]. Queries use α/Re/Mach only; Cl/Cd are outputs.
POLAR_IX_TEACHER_FORCED = (0, 1, 2)


def prepare_polar_batch(batch, device):
    """Unpack a collated AeroDataset batch onto ``device``."""
    if "geom_coords" in batch:
        geom = batch["geom_coords"].to(device, non_blocking=True)
    else:
        geom = patches_to_coords(batch["geom_patches"]).to(device, non_blocking=True)
    polar = batch["polar_tokens"].to(device, non_blocking=True)
    lengths = batch["polar_lengths"].to(device, non_blocking=True)
    mask = polar_length_mask(lengths, polar.shape[1])
    return geom, polar, mask


def polar_forward(
    model,
    batch,
    device,
    *,
    seed_len: int = 0,
    causal: bool | None = None,
    clcd_for_corrector=None,
    clcd_inject=None,
    p_ar=None,
):
    geom, polar, mask = prepare_polar_batch(batch, device)
    final, draft = model(
        geom,
        polar,
        mask,
        clcd_for_corrector=clcd_for_corrector,
        clcd_inject=clcd_inject,
        seed_len=seed_len,
        causal=causal,
        p_ar=p_ar,
    )
    return final, draft, polar, mask


def mix_corrector_clcd(
    draft: torch.Tensor,
    polar: torch.Tensor,
    mask: torch.Tensor,
    *,
    p_ar: float,
    seed_len: int = 0,
) -> torch.Tensor:
    """Scheduled sampling for the polar corrector: mix draft vs GT Cl/Cd.

    ``p_ar`` is the fraction of (non-seed) positions that keep the queried draft.
    Seed prefix ``seed_len`` is always GT.
    """
    gt = polar[..., 3:5]
    clcd = draft.detach()
    if p_ar < 1.0 - 1e-12:
        use_draft = torch.rand_like(mask, dtype=clcd.dtype) < float(p_ar)
        use_draft = use_draft & mask
        clcd = torch.where(use_draft.unsqueeze(-1), clcd, gt)
    if seed_len > 0:
        order = torch.cumsum(mask.long(), dim=1)
        seed_pos = mask & (order <= int(seed_len))
        clcd = torch.where(seed_pos.unsqueeze(-1), gt, clcd)
    return clcd


def mix_autoregressive_inputs(*_args, **_kwargs):
    raise RuntimeError(
        "mix_autoregressive_inputs was removed; use polar_forward(..., seed_len=K) "
        "or mix_corrector_clcd for the query/correct/requery model."
    )


def _apply_linear_warmup(optimizer, base_lr: float, epoch: int, warmup_epochs: int) -> None:
    """Set LR to ``base_lr * epoch / warmup_epochs`` for epoch in 1..warmup_epochs (inclusive cap at base_lr)."""
    if warmup_epochs <= 0:
        w = 1.0
    else:
        w = min(1.0, float(epoch) / float(warmup_epochs))
    for g in optimizer.param_groups:
        g["lr"] = base_lr * w


# cache for causal masks — avoids rebuilding every batch
_causal_mask_cache: dict = {}


def get_causal_mask(seq_len: int, device) -> torch.Tensor:
    key = (seq_len, str(device))
    if key not in _causal_mask_cache:
        _causal_mask_cache[key] = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )
    return _causal_mask_cache[key]


def build_sequence_batch(*_args, **_kwargs):
    raise RuntimeError(
        "build_sequence_batch was removed; use prepare_polar_batch / polar_forward."
    )


def train(train_path='data/train.csv',
          val_path='data/val.csv',
          embed_ckpt='checkpoints/embeddings_best.pt',
          epochs=100,
          lr=3e-4,
          batch_size=900,
          warmup_epochs=10,
          save_dir='checkpoints',
          resume_from_best=False,
          resume_ckpt=None,
          lr_schedule='plateau',
          plateau_patience=4,
          plateau_factor=0.5,
          plateau_threshold=1e-4,
          plateau_cooldown=1,
          min_lr_ratio=1e-5,
          warm_restart_T0=10,
          warm_restart_T_mult=2,
          early_stop_patience=0,
          num_workers: Optional[int] = None,
          variant_id: int = 0,
          dataset_cache_dir: Optional[str] = None,
          use_dataset_cache: bool = True,
          rng_seed: int = 42,
          d_model_override: Optional[int] = None):

    set_training_seed(rng_seed)
    device = get_device()
    print(f"Using device: {device}")
    print(f"RNG seed: {rng_seed}")
    variant = get_variant_spec(int(variant_id))
    print(f"Variant {variant.idx:02d}: {variant.name} [{variant.slug}]")
    d_model = int(d_model_override) if d_model_override is not None else int(variant.model.get("d_model", 32))
    if d_model <= 0:
        raise ValueError(f"d_model must be a positive integer, got {d_model}")
    if d_model_override is not None:
        print(f"  d_model override: {d_model} (variant default would be {int(variant.model.get('d_model', 32))})")
    scheduled_sampling = bool(variant.training.get("scheduled_sampling", True))
    use_causal_mask = bool(variant.training.get("use_causal_mask", variant.model.get("use_causal_mask", False)))
    polar_loss_weights = variant.training.get("polar_loss_weights", [0.0, 0.0, 0.0, 1.0, 31.0])
    set_polar_loss_weights(polar_loss_weights)
    draft_loss_weight = float(variant.model.get("draft_loss_weight", variant.training.get("draft_loss_weight", 0.25)))
    
    # CUDA: fast kernels, TF32 on matmul/cudnn, high-precision matmul for inductor paths where supported
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
        di = torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(di)
        print(
            f"CUDA: {torch.cuda.get_device_name(di)}  "
            f"(capability {cap[0]}.{cap[1]}, {torch.cuda.device_count()} device(s))"
        )

    resume_path = resume_ckpt if resume_ckpt else os.path.join(save_dir, 'foilcore_best.pt')
    resume_state = None
    if resume_from_best:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"--resume-from-best requires checkpoint at {resume_path}")
        print(f"Loading resume checkpoint: {resume_path}")
        resume_state = torch.load(resume_path, map_location='cpu', weights_only=False)

    print("Loading training dataset...")
    _cdc = Path(dataset_cache_dir) if dataset_cache_dir else None
    train_ds = AeroDataset(
        train_path,
        training=True,
        cache_dir=_cdc,
        use_cache=use_dataset_cache,
    )
    print("Loading validation dataset...")
    val_ds = AeroDataset(
        val_path,
        stats=train_ds.stats,
        training=False,
        cache_dir=_cdc,
        use_cache=use_dataset_cache,
    )

    if resume_state is not None and 'stats' in resume_state:
        train_ds.stats = resume_state['stats']
        val_ds.stats = resume_state['stats']
        print("  Using stats dict from resume checkpoint (matches saved weights).")

    # Fewer workers on MPS; persistent_workers prefetch helps pipelining on CUDA/CPU workers
    if num_workers is None:
        num_workers_eff = 0 if device == 'mps' else 4
    else:
        num_workers_eff = max(0, int(num_workers))
    dl_kwargs: dict = dict(
        pin_memory=(device != 'mps'),
        collate_fn=collate_variable_sequences,
    )
    if num_workers_eff > 0:
        dl_kwargs['persistent_workers'] = True
        dl_kwargs['prefetch_factor'] = 4

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers_eff,
        **dl_kwargs,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers_eff,
        **dl_kwargs,
    )
    print(f"DataLoader batch_size={batch_size}  num_workers={num_workers_eff}  pin_memory={dl_kwargs['pin_memory']}")

    print("Initializing model...")
    model_cfg = FoilCoreConfig(
        d_model=d_model,
        d_geom=int(variant.model.get("d_geom", d_model)),
        model_type=variant.model.get("model_type", "foilcore"),
        num_blocks=int(variant.model.get("num_blocks", 2)),
        block_mode=variant.model.get("block_mode", "all_alt"),
        shared_qk_scalars=bool(variant.model.get("shared_qk_scalars", True)),
        path_mode=variant.model.get("path_mode", "full"),
        share_path=bool(variant.model.get("share_path", True)),
        n_path_terms=int(variant.model.get("n_path_terms", 4)),
        outer_layers=0,
        dropout=float(variant.model.get("dropout", 0.05)),
        residual=bool(variant.model.get("residual", True)),
        use_causal_mask=use_causal_mask,
        decoder_hidden=int(variant.model.get("decoder_hidden", 0)),
        output_dims=int(variant.model.get("output_dims", 2)),
        mlp_hidden=int(variant.model.get("mlp_hidden", d_model)),
        n_heads=int(variant.model.get("n_heads", 2)),
        le_xmax=float(variant.model.get("le_xmax", 0.25)),
        te_xmin=float(variant.model.get("te_xmin", 0.75)),
        draft_loss_weight=draft_loss_weight,
        n_geom_memories=int(variant.model.get("n_geom_memories", 3)),
        n_landmarks=int(variant.model.get("n_landmarks", 4)),
        contour_bidirectional=bool(variant.model.get("contour_bidirectional", True)),
        contour_d_cond=int(variant.model.get("contour_d_cond", 3)),
        contour_log_a_init=str(variant.model.get("contour_log_a_init", "const")),
        contour_mode=str(variant.model.get("contour_mode", "ssm")),
        directional_landmarks=bool(variant.model.get("directional_landmarks", False)),
        landmark_mean=bool(variant.model.get("landmark_mean", False)),
        mem_log_a_init=float(variant.model.get("mem_log_a_init", -0.5)),
        polar_mixer=str(variant.model.get("polar_mixer", "iir_clcd")),
        iir_tau_init=str(variant.model.get("iir_tau_init", "const")),
        polar_in_clcd=bool(variant.model.get("polar_in_clcd", True)),
        polar_alpha_sinus=bool(variant.model.get("polar_alpha_sinus", True)),
        err_head_hidden=int(variant.model.get("err_head_hidden", 16)),
        err_head_mem_dim=int(variant.model.get("err_head_mem_dim", 4)),
        router_threshold=float(variant.model.get("router_threshold", 0.5)),
        use_stall_router=bool(variant.model.get("use_stall_router", True)),
        use_selective_ssm=bool(variant.model.get("use_selective_ssm", True)),
        use_local_kernel=bool(variant.model.get("use_local_kernel", True)),
        use_err_head=bool(variant.model.get("use_err_head", False)),
        profile_stations=int(variant.model.get("profile_stations", 0)),
        geom_pool=str(variant.model.get("geom_pool", "region3")),
        moe_experts=int(variant.model.get("moe_experts", 0)),
        softbin_k=int(variant.model.get("softbin_k", 4)),
        use_err_expert=bool(variant.model.get("use_err_expert", False)),
        err_expert_hidden=int(variant.model.get("err_expert_hidden", 16)),
    )
    err_loss_weight = float(variant.training.get("err_loss_weight", 0.0))
    router_sparsity_weight = float(variant.training.get("router_sparsity_weight", 0.0))
    causal_train_prob = float(variant.training.get("causal_train_prob", 0.0))
    if err_loss_weight or router_sparsity_weight or causal_train_prob:
        print(
            f"Aux: err_loss_weight={err_loss_weight} router_sparsity_weight={router_sparsity_weight} "
            f"causal_train_prob={causal_train_prob}"
        )
    _polar_w = torch.tensor(polar_loss_weights[3:5], dtype=torch.float32)

    def aux_losses(mdl, mask):
        """Error-head L1 (polar-weighted) + router sparsity, from ``mdl.aux`` set in forward."""
        aux = getattr(mdl, "aux", None) or {}
        out = {}
        if err_loss_weight > 0 and "err_pred" in aux:
            ep, et = aux["err_pred"][mask], aux["err_target"][mask]
            w = _polar_w.to(ep.device, ep.dtype)
            out["err"] = err_loss_weight * ((ep - et).abs() * w).sum(-1).mean()
        if router_sparsity_weight > 0 and aux.get("gates"):
            out["router"] = router_sparsity_weight * torch.stack(
                [g[mask].mean() for g in aux["gates"]]
            ).mean()
        return out
    model = FoilCORE(config=model_cfg).to(device)
    n_params = count_parameters(model)
    print(f"Model initialized: {n_params} trainable parameters")

    embed_ckpt_path = embed_ckpt
    if os.path.exists(embed_ckpt_path):
        print(f"Loading pretrained geom encoder from {embed_ckpt_path}")
        ckpt = torch.load(embed_ckpt_path, map_location=device, weights_only=False)
        raw_enc = None
        if "geom_encoder" in ckpt and isinstance(ckpt["geom_encoder"], dict):
            raw_enc = dict(ckpt["geom_encoder"])
        elif "embed_layer" in ckpt and isinstance(ckpt["embed_layer"], dict):
            raw_enc = dict(ckpt["embed_layer"])
        if raw_enc is not None:
            cur = model.geom_encoder.state_dict()
            prefixed = {k.split("geom_encoder.", 1)[-1]: v for k, v in raw_enc.items()}
            compat = {
                k: v for k, v in prefixed.items()
                if k in cur and getattr(v, "shape", None) == cur[k].shape
            }
            inf = model.geom_encoder.load_state_dict(compat, strict=False)
            print(
                f"  Loaded {len(compat)} geom encoder tensors "
                f"(missing={len(inf.missing_keys)} unexpected={len(inf.unexpected_keys)})."
            )
        if "path" in ckpt and isinstance(ckpt["path"], dict) and hasattr(model, "path"):
            cur_p = model.path.state_dict()
            p_compat = {
                k: v for k, v in ckpt["path"].items()
                if k in cur_p and getattr(v, "shape", None) == cur_p[k].shape
            }
            if p_compat:
                model.path.load_state_dict(p_compat, strict=False)
                print(f"  Loaded {len(p_compat)} shared PathNonlinearity tensors.")
        else:
            print("  Warning: unrecognized embedding checkpoint keys; skipping geom encoder load.")
    else:
        print(f"Warning: embedding checkpoint not found at {embed_ckpt_path}. Using random geom encoder.")

    # DO NOT use torch.compile - it's too slow on this setup
    print("torch.compile disabled for compatibility")

    # Default: standard AdamW. Fused can fail at step() on some CUDA/driver combos; opt-in only.
    params = list(model.parameters())
    use_fused = (
        device == "cuda"
        and os.environ.get("FOILCORE_FUSED_ADAMW", "").strip().lower() in ("1", "true", "yes")
        and hasattr(torch.optim.AdamW, "fused")
    )
    if use_fused:
        try:
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4, fused=True)
            print("Using fused AdamW (FOILCORE_FUSED_ADAMW=1).")
        except (TypeError, RuntimeError) as e:
            print(f"Fused AdamW unavailable ({e}); using standard AdamW.")
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    lr_schedule = lr_schedule.lower().strip()
    if lr_schedule not in ('plateau', 'cosine', 'warm_restarts'):
        raise ValueError(f"lr_schedule must be plateau|cosine|warm_restarts, got {lr_schedule!r}")

    lambda_scheduler = None
    plateau_scheduler = None
    wr_scheduler = None
    min_lr = max(lr * min_lr_ratio, 1e-12)

    if lr_schedule == 'cosine':
        lambda_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda e: warmup_lambda(e, warmup_epochs, epochs),
        )
        print(f"LR: cosine decay after {warmup_epochs} warmup epochs (LambdaLR).")
    elif lr_schedule == 'warm_restarts':
        wr_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=warm_restart_T0, T_mult=warm_restart_T_mult, eta_min=min_lr
        )
        print(
            f"LR: CosineAnnealingWarmRestarts(T_0={warm_restart_T0}, T_mult={warm_restart_T_mult}, "
            f"eta_min={min_lr:.2e}) + linear warmup first {warmup_epochs} epochs."
        )
    else:
        plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            threshold_mode='rel',
            min_lr=min_lr,
            cooldown=plateau_cooldown,
        )
        print(
            f"LR: linear warmup {warmup_epochs} ep, then ReduceLROnPlateau "
            f"(patience={plateau_patience}, factor={plateau_factor}, threshold={plateau_threshold}, "
            f"cooldown={plateau_cooldown}, min_lr={min_lr:.2e})."
        )

    start_epoch = 0
    best_val_loss = float('inf')
    if resume_state is not None:
        m_sd = remap_foilcore_state_dict(
            dict(resume_state["model"]),
            target_shared_qk_scalars=model_cfg.shared_qk_scalars,
        )
        model_sd = model.state_dict()
        compatible = {
            k: v for k, v in m_sd.items()
            if k in model_sd and getattr(v, "shape", None) == model_sd[k].shape
        }
        inf = model.load_state_dict(compatible, strict=False)
        if inf.missing_keys or inf.unexpected_keys or len(compatible) != len(m_sd):
            print(
                "  Resume: model strict=False "
                f"(loaded={len(compatible)} skipped={len(m_sd) - len(compatible)} "
                f"missing={len(inf.missing_keys)} unexpected={len(inf.unexpected_keys)} keys)."
            )
        start_epoch = int(resume_state.get('epoch', 0))
        best_val_loss = float(resume_state.get('val_loss', resume_state.get('best_val_loss', float('inf'))))
        print(f"  Loaded weights. Resuming after epoch {start_epoch} (best val loss so far: {best_val_loss:.5f}).")

    if lambda_scheduler is not None and start_epoch > 0:
        for _ in range(start_epoch):
            lambda_scheduler.step()
        print(f"  Fast-forwarded LambdaLR by {start_epoch} steps (cosine schedule).")
    if wr_scheduler is not None and start_epoch > warmup_epochs:
        n_wr = start_epoch - warmup_epochs
        for _ in range(n_wr):
            wr_scheduler.step()
        print(f"  Fast-forwarded CosineAnnealingWarmRestarts by {n_wr} steps.")

    os.makedirs(save_dir, exist_ok=True)

    # Initialize AMP scaler for mixed precision (CUDA only)
    use_amp = device == 'cuda'
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using Automatic Mixed Precision (AMP)")

    bad_epochs = 0
    history: list[dict] = []
    print(f"\nStarting training from epoch {start_epoch + 1} through {epochs} (total target epochs={epochs}).")

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        sum_train_cl = 0.0
        sum_train_cd = 0.0
        sum_train_err = 0.0
        n_train_polar_batches = 0

        if lr_schedule == 'plateau':
            if epoch <= warmup_epochs:
                _apply_linear_warmup(optimizer, lr, epoch, warmup_epochs)
        elif lr_schedule == 'warm_restarts':
            if epoch <= warmup_epochs:
                _apply_linear_warmup(optimizer, lr, epoch, warmup_epochs)
            elif epoch == warmup_epochs + 1 and start_epoch <= warmup_epochs:
                for g in optimizer.param_groups:
                    g['lr'] = lr
        else:
            pass

        p_ar_ep = ar_mix_probability(epoch) if scheduled_sampling else 1.0
        epoch_start = time.time()

        pbar = tqdm(train_dl, desc=f"Epoch {epoch:3d}/{epochs} [Train p_ar={p_ar_ep:.0%}]", leave=False)
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            geom, polar, mask = prepare_polar_batch(batch, device)

            causal_kw = {}
            if causal_train_prob > 0:
                causal_kw["causal"] = bool(torch.rand(()).item() < causal_train_prob)
            with autocast() if use_amp else nullcontext():
                final, draft = model(geom, polar, mask, p_ar=p_ar_ep, **causal_kw)
                comp = polar_clcd_components(final, polar, mask, train_ds.stats)
                loss = comp["total"]
                if draft_loss_weight > 0:
                    aux = polar_clcd_components(draft, polar, mask, train_ds.stats)
                    loss = loss + draft_loss_weight * aux["total"]
                extra = aux_losses(model, mask)
                for v in extra.values():
                    loss = loss + v
                if "err" in extra:
                    sum_train_err += extra["err"].detach().item()

            clip_params = list(model.parameters())

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                optimizer.step()

            total_train_loss += loss.item()
            if mask.any():
                sum_train_cl += comp["polar_cl_mae"].detach().item()
                sum_train_cd += comp["polar_cd_mae"].detach().item()
                n_train_polar_batches += 1

            postfix = {'L': f"{loss.item():.4f}"}
            if mask.any():
                postfix['Cl'] = f"{comp['polar_cl_mae'].detach().item():.4f}"
                postfix['Cd'] = f"{comp['polar_cd_mae'].detach().item():.4f}"
            pbar.set_postfix(postfix)

        if lambda_scheduler is not None:
            lambda_scheduler.step()
        elif wr_scheduler is not None and epoch > warmup_epochs:
            wr_scheduler.step()

        avg_train = total_train_loss / len(train_dl)
        avg_train_cl = sum_train_cl / n_train_polar_batches if n_train_polar_batches else float("nan")
        avg_train_cd = sum_train_cd / n_train_polar_batches if n_train_polar_batches else float("nan")

        epoch_time = time.time() - epoch_start

        # ---------- validation ----------
        model.eval()
        total_val_loss = 0.0
        sum_val_cl = 0.0
        sum_val_cd = 0.0
        n_val_polar_batches = 0

        val_start = time.time()

        with torch.no_grad():
            v_pbar = tqdm(val_dl, desc=f"Epoch {epoch:3d}/{epochs} [Val  ]", leave=False)
            for batch in v_pbar:
                geom, polar, mask = prepare_polar_batch(batch, device)
                with autocast() if use_amp else nullcontext():
                    final, draft = model(geom, polar, mask)
                    comp = polar_clcd_components(final, polar, mask, train_ds.stats)
                    loss = comp["total"]
                    if draft_loss_weight > 0:
                        aux = polar_clcd_components(draft, polar, mask, train_ds.stats)
                        loss = loss + draft_loss_weight * aux["total"]

                total_val_loss += loss.item()
                if mask.any():
                    sum_val_cl += comp["polar_cl_mae"].detach().item()
                    sum_val_cd += comp["polar_cd_mae"].detach().item()
                    n_val_polar_batches += 1

        avg_val = total_val_loss / len(val_dl) if len(val_dl) > 0 else 0.0
        avg_val_cl = sum_val_cl / n_val_polar_batches if n_val_polar_batches else float("nan")
        avg_val_cd = sum_val_cd / n_val_polar_batches if n_val_polar_batches else float("nan")
        val_time = time.time() - val_start

        cur_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:3d}/{epochs} | Time: {epoch_time:.1f}s (val: {val_time:.1f}s) | "
            f"Train L: {avg_train:.5f} (raw MAE Cl {avg_train_cl:.5f}, Cd {avg_train_cd:.5f}) | "
            f"Val L: {avg_val:.5f} (raw MAE Cl {avg_val_cl:.5f}, Cd {avg_val_cd:.5f}) | "
            f"LR: {cur_lr:.2e} | p_ar={p_ar_ep:.0%}"
            + (f" | err_loss={sum_train_err / max(1, len(train_dl)):.4f}" if err_loss_weight > 0 else "")
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(avg_train),
                "val_loss": float(avg_val),
                "train_cl_mae_norm": float(avg_train_cl),
                "train_cd_mae_norm": float(avg_train_cd),
                "val_cl_mae_norm": float(avg_val_cl),
                "val_cd_mae_norm": float(avg_val_cd),
                "train_err_loss": float(sum_train_err / max(1, len(train_dl))),
                "lr": float(cur_lr),
                "p_ar": float(p_ar_ep),
                "epoch_time_s": float(epoch_time),
                "val_time_s": float(val_time),
            }
        )

        if plateau_scheduler is not None and epoch > warmup_epochs:
            plateau_scheduler.step(avg_val)

        if early_stop_patience > 0:
            if avg_val < best_val_loss - 1e-12:
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= early_stop_patience:
                    print(f"Early stopping: val loss did not improve for {early_stop_patience} epochs.")
                    break

        # Save checkpoint
        if epoch % 5 == 0 or epoch == 1:  # Save less frequently to save time
            checkpoint_path = os.path.join(save_dir, f"foilcore_epoch_{epoch:03d}.pt")
            sched_state = {}
            if lambda_scheduler is not None:
                sched_state['lambda'] = lambda_scheduler.state_dict()
            if plateau_scheduler is not None:
                sched_state['plateau'] = plateau_scheduler.state_dict()
            if wr_scheduler is not None:
                sched_state['warm_restarts'] = wr_scheduler.state_dict()
            torch.save({
                'epoch': epoch,
                'rng_seed': rng_seed,
                'model': model.state_dict(),
                'geom_encoder': model.geom_encoder.state_dict(),
                'n_params': n_params,
                'variant': {'id': variant.idx, 'slug': variant.slug, 'name': variant.name},
                'variant_model': model_cfg.__dict__,
                'variant_training': variant.training,
                'optimizer': optimizer.state_dict(),
                'schedulers': sched_state,
                'lr_schedule': lr_schedule,
                'scaler': scaler.state_dict() if scaler else None,
                'stats': train_ds.stats,
                'val_loss': avg_val,
            }, checkpoint_path)
            print(f"  ✓ saved {checkpoint_path}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_path = os.path.join(save_dir, "foilcore_best.pt")
            torch.save({
                'epoch': epoch,
                'rng_seed': rng_seed,
                'model': model.state_dict(),
                'geom_encoder': model.geom_encoder.state_dict(),
                'n_params': n_params,
                'variant': {'id': variant.idx, 'slug': variant.slug, 'name': variant.name},
                'variant_model': model_cfg.__dict__,
                'variant_training': variant.training,
                'stats': train_ds.stats,
                'val_loss': best_val_loss,
                'best_val_loss': best_val_loss,
            }, best_path)
            print(f"  ✓ new best model (val loss: {best_val_loss:.5f})")

    print("Training finished.")
    metrics_path = os.path.join(save_dir, "metrics_history.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "variant": {"id": variant.idx, "slug": variant.slug, "name": variant.name},
                "rng_seed": int(rng_seed),
                "config": {
                    "d_model": d_model,
                    "n_params": int(n_params),
                    "scheduled_sampling": scheduled_sampling,
                    "use_causal_mask": use_causal_mask,
                    "draft_loss_weight": float(draft_loss_weight),
                    "polar_loss_weights": list(map(float, polar_loss_weights)),
                },
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"Saved metrics history: {metrics_path}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if history:
            epochs_x = [h["epoch"] for h in history]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(epochs_x, [h["train_loss"] for h in history], label="train")
            axes[0].plot(epochs_x, [h["val_loss"] for h in history], label="val")
            axes[0].set_title("Loss")
            axes[0].grid(alpha=0.3)
            axes[0].legend()
            axes[1].plot(epochs_x, [h["val_cl_mae_norm"] for h in history], label="val Cl")
            axes[1].plot(epochs_x, [h["val_cd_mae_norm"] for h in history], label="val Cd")
            axes[1].set_title("Val normalized MAE")
            axes[1].grid(alpha=0.3)
            axes[1].legend()
            fig.tight_layout()
            curves_path = os.path.join(save_dir, "training_curves.png")
            fig.savefig(curves_path, dpi=160)
            plt.close(fig)
            print(f"Saved curves: {curves_path}")
    except Exception as exc:
        print(f"Skipping curve plot due to error: {exc}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Train FoilCORE.')
    p.add_argument('--resume-from-best', action='store_true',
                   help='Load weights (and stats) from resume checkpoint before training.')
    p.add_argument('--resume-ckpt', type=str, default=None,
                   help='Path to resume (default: <save_dir>/foilcore_best.pt).')
    p.add_argument('--save-dir', type=str, default='checkpoints')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--batch-size', type=int, default=900, help='Training/validation DataLoader batch size.')
    p.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='DataLoader worker processes (default: 0 on MPS, 4 on CUDA/CPU).',
    )
    p.add_argument('--warmup-epochs', type=int, default=10)
    p.add_argument('--lr-schedule', type=str, default='plateau',
                   choices=['plateau', 'cosine', 'warm_restarts'],
                   help='plateau: ReduceLROnPlateau on val after warmup. cosine: original LambdaLR. '
                        'warm_restarts: CosineAnnealingWarmRestarts after warmup.')
    p.add_argument('--plateau-patience', type=int, default=4)
    p.add_argument('--plateau-factor', type=float, default=0.5)
    p.add_argument('--plateau-threshold', type=float, default=1e-4)
    p.add_argument('--plateau-cooldown', type=int, default=1)
    p.add_argument('--min-lr-ratio', type=float, default=1e-5,
                   help='Floor LR = lr * this ratio for plateau / warm_restarts eta_min.')
    p.add_argument('--warm-restart-T0', type=int, default=10)
    p.add_argument('--warm-restart-T-mult', type=int, default=2)
    p.add_argument('--early-stop-patience', type=int, default=0,
                   help='Stop if val loss does not improve for this many epochs (0=disabled).')
    p.add_argument('--variant-id', type=int, default=0, help='Ablation variant id from src.ablations.')
    p.add_argument(
        '--d-model',
        type=int,
        default=None,
        help='Override trunk width d (default 32).',
    )
    p.add_argument(
        '--seed',
        type=int,
        default=42,
        help='RNG seed (logged in checkpoints and metrics_history.json).',
    )
    p.add_argument('--train-csv', type=str, default='data/train.csv')
    p.add_argument('--val-csv', type=str, default='data/val.csv')
    p.add_argument(
        '--dataset-cache-dir',
        type=str,
        default=None,
        help='Pickled parsed DataFrames live here (default: <train_csv_dir>/.aero_cache unless AERO_CACHE_DIR is set).',
    )
    p.add_argument(
        '--no-dataset-cache',
        action='store_true',
        help='Always read and parse CSV (no pickle load/save for this run).',
    )
    args = p.parse_args()
    import sys
    import traceback

    try:
        train(
            train_path=args.train_csv,
            val_path=args.val_csv,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            warmup_epochs=args.warmup_epochs,
            save_dir=args.save_dir,
            resume_from_best=args.resume_from_best,
            resume_ckpt=args.resume_ckpt,
            lr_schedule=args.lr_schedule,
            plateau_patience=args.plateau_patience,
            plateau_factor=args.plateau_factor,
            plateau_threshold=args.plateau_threshold,
            plateau_cooldown=args.plateau_cooldown,
            min_lr_ratio=args.min_lr_ratio,
            warm_restart_T0=args.warm_restart_T0,
            warm_restart_T_mult=args.warm_restart_T_mult,
            early_stop_patience=args.early_stop_patience,
            num_workers=args.num_workers,
            variant_id=args.variant_id,
            dataset_cache_dir=args.dataset_cache_dir,
            use_dataset_cache=not args.no_dataset_cache,
            rng_seed=args.seed,
            d_model_override=args.d_model,
        )
    except Exception:
        traceback.print_exc()
        sys.exit(1)