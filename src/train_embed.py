import sys
from pathlib import Path

# Allow ``python src/train_embed.py`` (repo root must be on path for ``import src.*``).
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import math
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import AeroDataset, collate_variable_sequences, patches_to_coords
from src.embeddings import D_MODEL, GeometryTripleEncoder
from src.model import PathNonlinearity


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def warmup_lambda(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return float(epoch) / float(max(1, warmup_epochs))
    return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))


def pretrain_embeddings(
    train_path="data/train.csv",
    val_path="data/val.csv",
    epochs=50,
    lr=1e-3,
    batch_size=10240,
    warmup_epochs=5,
    save_dir="checkpoints",
    num_workers=None,
    dataset_cache_dir=None,
    use_dataset_cache=True,
    d_model=D_MODEL,
    n_path_terms=4,
):
    """Autoencode 200-point sections through 3×d_model (global, LE, TE) latents."""
    device = get_device()
    print(f"Using device: {device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    nb = device == "cuda"
    cdir = Path(dataset_cache_dir) if dataset_cache_dir else None
    train_ds = AeroDataset(train_path, training=True, cache_dir=cdir, use_cache=use_dataset_cache)
    val_ds = AeroDataset(
        val_path, stats=train_ds.stats, training=False, cache_dir=cdir, use_cache=use_dataset_cache
    )

    if num_workers is None:
        num_workers_eff = 0 if device == "mps" else 4
    else:
        num_workers_eff = max(0, int(num_workers))
    dl_kw: dict = dict(
        num_workers=num_workers_eff,
        pin_memory=(device != "mps"),
        collate_fn=collate_variable_sequences,
    )
    if num_workers_eff > 0:
        dl_kw["persistent_workers"] = True
        dl_kw["prefetch_factor"] = 4
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kw)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kw)
    print(f"DataLoader batch_size={batch_size}  num_workers={num_workers_eff}")

    path = PathNonlinearity(d_model, n_terms=n_path_terms).to(device)
    encoder = GeometryTripleEncoder(d_model=d_model).to(device)
    decoder = nn.Sequential(
        nn.Linear(3 * d_model, 128),
        nn.GELU(),
        nn.Linear(128, 400),
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(path.parameters()) + list(decoder.parameters()),
        lr=lr,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda e: warmup_lambda(e, warmup_epochs, epochs))
    criterion = nn.L1Loss()

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_path = os.path.join(save_dir, "embeddings_best.pt")

    def _coords(batch):
        if "geom_coords" in batch:
            return batch["geom_coords"].to(device, non_blocking=nb)
        return patches_to_coords(batch["geom_patches"]).to(device, non_blocking=nb)

    def _step(batch):
        coords = _coords(batch)
        z = encoder(coords, path=path)
        recon = decoder(z.reshape(z.shape[0], -1)).reshape(-1, 200, 2)
        return criterion(recon, coords)

    for epoch in range(1, epochs + 1):
        encoder.train()
        path.train()
        decoder.train()
        total_train = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch:3d}/{epochs} [Train]", leave=False)
        for batch in pbar:
            optimizer.zero_grad()
            loss = _step(batch)
            loss.backward()
            optimizer.step()
            total_train += loss.item()
            pbar.set_postfix({"L": f"{loss.item():.4f}"})
        scheduler.step()

        encoder.eval()
        path.eval()
        decoder.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in val_dl:
                total_val += _step(batch).item()
        avg_train = total_train / max(1, len(train_dl))
        avg_val = total_val / max(1, len(val_dl))
        print(
            f"Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train:.6f} | "
            f"Val Loss: {avg_val:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {
                    "epoch": epoch,
                    "geom_encoder": encoder.state_dict(),
                    "path": path.state_dict(),
                    "geom_decoder": decoder.state_dict(),
                    "stats": train_ds.stats,
                    "val_loss": best_val_loss,
                    "best_val_loss": best_val_loss,
                    "d_model": d_model,
                    "n_path_terms": n_path_terms,
                },
                best_path,
            )
            print(f"  ✓ new best embeddings saved to {best_path} (val loss: {best_val_loss:.6f})")

    print(f"Best embeddings saved to {best_path}")
    return train_ds.stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Pretrain 3×32 geometry autoencoder.")
    ap.add_argument("--train-path", type=str, default="data/train.csv")
    ap.add_argument("--val-path", type=str, default="data/val.csv")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=10240)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--save-dir", type=str, default="checkpoints")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--dataset-cache-dir", type=str, default=None)
    ap.add_argument("--no-dataset-cache", action="store_true")
    args = ap.parse_args()
    pretrain_embeddings(
        train_path=args.train_path,
        val_path=args.val_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        save_dir=args.save_dir,
        num_workers=args.num_workers,
        dataset_cache_dir=args.dataset_cache_dir,
        use_dataset_cache=not args.no_dataset_cache,
    )
