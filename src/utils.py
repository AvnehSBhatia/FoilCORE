import torch
import torch.nn.functional as F

from src.data import MODALITY

# Polar targets are z-scored in ``AeroDataset``. Polar term: 1*MAE(Cl_z) + 31*MAE(Cd_z) (weights on dim_mae).
POLAR_LOSS_WEIGHTS_5 = torch.tensor([0.0, 0.0, 0.0, 1.0, 31.0], dtype=torch.float32)


def set_polar_loss_weights(weights_5: list[float]) -> None:
    global POLAR_LOSS_WEIGHTS_5
    if len(weights_5) != 5:
        raise ValueError(f"Expected 5 polar loss weights [AoA,Re,Mach,Cl,Cd], got {weights_5}")
    POLAR_LOSS_WEIGHTS_5 = torch.tensor(list(map(float, weights_5)), dtype=torch.float32)


def polar_clcd_components(pred, polar_tokens, polar_mask, stats=None):
    """MAE on (Cl, Cd) at polar positions. ``pred`` is (B, T, 2)."""
    _ = stats
    device = polar_mask.device
    zero = torch.tensor(0.0, device=device)
    if not polar_mask.any():
        return {
            "polar_mse": zero,
            "polar_cl_mae": zero,
            "polar_cd_mae": zero,
            "total": zero,
        }
    pred_v = pred[polar_mask]
    tgt = polar_tokens[polar_mask][..., 3:5]
    cl_mae = F.l1_loss(pred_v[..., 0], tgt[..., 0])
    cd_mae = F.l1_loss(pred_v[..., 1], tgt[..., 1])
    w = POLAR_LOSS_WEIGHTS_5.to(device=pred.device, dtype=pred.dtype)
    total = w[3] * cl_mae + w[4] * cd_mae
    return {
        "polar_mse": total,
        "polar_cl_mae": cl_mae,
        "polar_cd_mae": cd_mae,
        "total": total,
    }


def batched_sequence_loss_components(
    polar_preds,
    geom_preds,
    special_logits,
    mod_logits,
    targets_polar,
    mask_polar,
    targets_geom,
    mask_geom,
    targets_special,
    mask_special,
    stats,
    modality_weight=0.1,
):
    """
    Polar term MAE(z): 1*Cl + 31*Cd (other dims weighted 0 unless variant overrides). Signature kept for scripts.
    """
    device = mask_polar.device
    zero = torch.tensor(0.0, device=device)
    _ = (
        stats,
        modality_weight,
        geom_preds,
        special_logits,
        mod_logits,
        targets_geom,
        mask_geom,
        targets_special,
        mask_special,
    )

    polar_mse = zero
    polar_aoa_mae = zero
    polar_re_mae = zero
    polar_mach_mae = zero
    polar_cl_mae = zero
    polar_cd_mae = zero
    if mask_polar.any():
        polar_targets = targets_polar[mask_polar]
        pred = polar_preds[mask_polar]
        if pred.shape[-1] == 2:
            tgt_2 = polar_targets[..., 3:5]
            cl_mae = F.l1_loss(pred[..., 0], tgt_2[..., 0])
            cd_mae = F.l1_loss(pred[..., 1], tgt_2[..., 1])
            w = POLAR_LOSS_WEIGHTS_5.to(device=pred.device, dtype=pred.dtype)
            polar_cl_mae = cl_mae
            polar_cd_mae = cd_mae
            polar_mse = w[3] * cl_mae + w[4] * cd_mae
        elif pred.shape[-1] == 5:
            w = POLAR_LOSS_WEIGHTS_5.to(device=pred.device, dtype=pred.dtype)
            dim_mae = torch.mean(torch.abs(pred - polar_targets), dim=0)
            polar_aoa_mae = dim_mae[0]
            polar_re_mae = dim_mae[1]
            polar_mach_mae = dim_mae[2]
            polar_cl_mae = dim_mae[3]
            polar_cd_mae = dim_mae[4]
            polar_mse = (w * dim_mae).sum()
        else:
            raise ValueError(f"Unsupported polar prediction width: {pred.shape[-1]}")

    geom_mae = geom_mse = geom_loss = geom_smooth = special_ce = modality_ce = modality_loss = zero
    total = polar_mse
    return {
        "polar_mse": polar_mse,
        "polar_aoa_mae": polar_aoa_mae,
        "polar_re_mae": polar_re_mae,
        "polar_mach_mae": polar_mach_mae,
        "polar_cl_mae": polar_cl_mae,
        "polar_cd_mae": polar_cd_mae,
        "geom_mae": geom_mae,
        "geom_mse": geom_mse,
        "geom_loss": geom_loss,
        "geom_smooth": geom_smooth,
        "special_ce": special_ce,
        "modality_ce": modality_ce,
        "modality_loss": modality_loss,
        "total": total,
    }


def batched_sequence_loss(
    polar_preds,
    geom_preds,
    special_logits,
    mod_logits,
    targets_polar,
    mask_polar,
    targets_geom,
    mask_geom,
    targets_special,
    mask_special,
    stats,
    modality_weight=0.1,
):
    c = batched_sequence_loss_components(
        polar_preds,
        geom_preds,
        special_logits,
        mod_logits,
        targets_polar,
        mask_polar,
        targets_geom,
        mask_geom,
        targets_special,
        mask_special,
        stats,
        modality_weight=modality_weight,
    )
    return c["total"]


def _modality_targets(mask_polar, mask_geom, mask_special):
    mod_targets = torch.zeros_like(mask_polar, dtype=torch.long, device=mask_polar.device)
    mod_targets[mask_polar] = MODALITY["polar"]
    mod_targets[mask_geom] = MODALITY["geom"]
    mod_targets[mask_special] = MODALITY["special"]
    return mod_targets


def special_and_modality_accuracy(
    mod_logits,
    special_logits,
    targets_special,
    mask_special,
    mask_polar,
    mask_geom,
):
    if mask_special.any():
        pred_s = special_logits[mask_special]
        tgt_s = targets_special[mask_special].long()
        special_acc = (pred_s.argmax(dim=-1) == tgt_s).float().mean().item()
    else:
        special_acc = float("nan")

    valid = mask_polar | mask_geom | mask_special
    if valid.any() and mod_logits is not None:
        mod_targets = _modality_targets(mask_polar, mask_geom, mask_special)
        pred_m = mod_logits[valid]
        tgt_m = mod_targets[valid]
        modality_acc = (pred_m.argmax(dim=-1) == tgt_m).float().mean().item()
    else:
        modality_acc = float("nan")

    return special_acc, modality_acc


_POLAR_MAPE_MIN_ABS = {
    "AoA": 0.5,
    "Re": 1.0e4,
    "Mach": 0.01,
    "Cl": 0.05,
    "Cd": 1.0e-3,
}


def compute_domain_metrics(predictions, targets, stats):
    """Per-dimension MAE and MAPE on denormalized polars."""
    metrics = {}

    names = ["AoA", "Re", "Mach", "Cl", "Cd"]
    offset = 0
    if predictions.shape[-1] == 2:
        targets = targets[..., 3:5]
        names = ["Cl", "Cd"]
        offset = 3

    polar_mean = torch.as_tensor(stats["polar_mean"], device=predictions.device, dtype=predictions.dtype)[
        offset : offset + predictions.shape[-1]
    ]
    polar_std = torch.as_tensor(stats["polar_std"], device=predictions.device, dtype=predictions.dtype)[
        offset : offset + predictions.shape[-1]
    ]
    pred_unnorm = predictions * polar_std + polar_mean
    tgt_unnorm = targets * polar_std + polar_mean

    for i, name in enumerate(names):
        if i < pred_unnorm.shape[1]:
            mae = F.l1_loss(pred_unnorm[:, i], tgt_unnorm[:, i])
            abs_tgt = torch.abs(tgt_unnorm[:, i])
            floor = torch.as_tensor(
                _POLAR_MAPE_MIN_ABS[name], device=predictions.device, dtype=predictions.dtype
            )
            ok = abs_tgt >= floor
            if ok.any():
                mape = (torch.abs(pred_unnorm[:, i] - tgt_unnorm[:, i])[ok] / abs_tgt[ok]).mean()
                metrics[f"{name}_mape"] = mape.item()
            else:
                metrics[f"{name}_mape"] = float("nan")
            metrics[f"{name}_mae"] = mae.item()

    return metrics


def compute_geom_mae(geom_preds, geom_targets, scales, x_mins):
    """Physical coordinate MAE for geometry patches (normalized 20-dim patches)."""
    batch = geom_preds.shape[0]
    total_err = 0.0
    total_pts = 0

    for b in range(batch):
        s = scales[b].item()
        x0 = x_mins[b].item()
        pred_pts = geom_preds[b].view(10, 2)
        targ_pts = geom_targets[b].view(10, 2)
        pred_denorm = pred_pts.clone()
        pred_denorm[:, 0] = pred_pts[:, 0] * s + x0
        pred_denorm[:, 1] = pred_pts[:, 1] * s
        targ_denorm = targ_pts.clone()
        targ_denorm[:, 0] = targ_pts[:, 0] * s + x0
        targ_denorm[:, 1] = targ_pts[:, 1] * s

        total_err += torch.abs(pred_denorm - targ_denorm).sum().item()
        total_pts += 10 * 2

    return total_err / total_pts if total_pts > 0 else 0.0
