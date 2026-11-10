import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from tqdm import tqdm

_CACHE_VERSION = 1

# --- Special token indices ---
SPECIAL_TOKENS = {
    'PAD':       0,
    'BOS':       1,
    'EOS':       2,
    'END_POLAR': 3,
    'END_GEOM':  4,
    'SWITCH':    5,
    'MASK':      6,
}
N_SPECIAL = len(SPECIAL_TOKENS)

# --- Modality indices ---
MODALITY = {
    'special':  0,
    'polar':    1,
    'geom':     2,
}

D_MODEL    = 32  # must match ``src.model.D_MODEL`` (geom memory / polar corrector width)
D_POLAR    = 5     # [AoA, Re, Mach, Cl, Cd]
D_GEOM     = 20    # 10 (x,y) pairs flattened per patch (raw coords, unchanged)
# Binary surface concatenated before geom projection: 0 upper / 1 lower (half-perimeter heuristic).
D_GEOM_IN = D_GEOM + 1
N_GEOM_PATCHES = 20
POINTS_PER_PATCH = 10
N_GEOM_POINTS = N_GEOM_PATCHES * POINTS_PER_PATCH


def patches_to_coords(patches: torch.Tensor) -> torch.Tensor:
    """(B, 20, 20) or (20, 20) flattened patches → (B, 200, 2) or (200, 2)."""
    if patches.dim() == 2:
        return patches.reshape(N_GEOM_POINTS, 2)
    return patches.reshape(patches.shape[0], N_GEOM_POINTS, 2)


def polar_length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """(B,) lengths → (B, max_len) bool mask of real (non-pad) polar steps."""
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def polar_sequence_starts(geom_visible: torch.Tensor) -> torch.Tensor:
    """Per-row index of the first polar token (after optional geom block plus splitter). Shape (B,)."""
    return 1 + geom_visible.long() * N_GEOM_PATCHES


def resample_airfoil(coords, n_points=200):
    """Resample to exactly n_points via arc-length parameterization."""
    coords = np.array(coords)
    diffs = np.diff(coords, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    arc = np.concatenate([[0], np.cumsum(seg_lengths)])
    arc /= arc[-1]
    fx = interp1d(arc, coords[:, 0], kind='linear')
    fy = interp1d(arc, coords[:, 1], kind='linear')
    t_new = np.linspace(0, 1, n_points)
    return np.stack([fx(t_new), fy(t_new)], axis=1)


def normalize_coords(coords):
    """
    Normalize so x ∈ [0,1], y scaled by the same factor.
    Returns (coords_normalized, scale, x_min)
    """
    coords = np.array(coords)
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    scale = x_max - x_min
    coords_norm = coords.copy()
    coords_norm[:, 0] = (coords[:, 0] - x_min) / scale
    coords_norm[:, 1] = coords[:, 1] / scale
    return coords_norm, scale, x_min


def translate_normalized_geometry(coords_norm, dx: float, dy: float) -> np.ndarray:
    r"""
    Rigid translation in chord-normalized space before ``build_geom_patches`` / geom embeddings.

    Use when polar tokens (e.g. AoA) should stay fixed while the section is shifted in the
    (x, y) plane seen by the geometry encoder. ``dx, dy`` are in the same units as
    ``coords_norm`` (typically x chord fraction and y scaled by chord).
    """
    if dx == 0.0 and dy == 0.0:
        return np.asarray(coords_norm, dtype=np.float64)
    out = np.asarray(coords_norm, dtype=np.float64).copy()
    out[:, 0] += float(dx)
    out[:, 1] += float(dy)
    return out


def geom_fingerprint(coords_raw) -> str:
    """SHA256 over resampled, chord-normalised coordinates (same preprocessing as ``AeroDataset``)."""
    c = resample_airfoil(np.asarray(coords_raw, dtype=np.float64), n_points=200)
    c_norm, _, _ = normalize_coords(c)
    return hashlib.sha256(np.ascontiguousarray(c_norm).tobytes()).hexdigest()


def denormalize_coords(coords_norm, scale, x_min):
    """Reverse the normalization."""
    coords = coords_norm.copy()
    coords[:, 0] = coords[:, 0] * scale + x_min
    coords[:, 1] = coords[:, 1] * scale
    return coords


def build_geom_patches(coords):
    """coords: (200, 2) -> (20, 20) flat patches."""
    flat = coords.flatten()
    return flat.reshape(N_GEOM_PATCHES, D_GEOM)


def geom_patch_surface_flags():
    """
    Surface id per patch row (broadcast to every sample).

    Rows follow arc-length sampling in ``build_geom_patches``: first half of the 20 patches
    along the perimeter use 0.0 (upper), second half use 1.0 (lower). If your CAD ordering
    is reversed, flip the labels or remap here.
    """
    flags = np.zeros(N_GEOM_PATCHES, dtype=np.float32)
    flags[N_GEOM_PATCHES // 2 :] = 1.0
    return flags


_GEOM_SURFACE_TORCH = torch.tensor(geom_patch_surface_flags(), dtype=torch.float32)


def build_polar_tokens(alpha, cl, cd, re, mach):
    """Returns (N,5) array [AoA, Re, Mach, Cl, Cd]."""
    n = len(alpha)
    re_arr   = np.full(n, re)
    mach_arr = np.full(n, mach)
    return np.stack([alpha, re_arr, mach_arr, cl, cd], axis=1)


def default_aero_cache_dir(csv_path: Path | str) -> Path:
    """Directory for parsed DataFrame pickles (default: ``<csv_dir>/.aero_cache``; override with ``AERO_CACHE_DIR``)."""
    env = os.environ.get("AERO_CACHE_DIR")
    if env:
        return Path(env)
    return Path(csv_path).parent / ".aero_cache"


def _csv_signature(path: Path) -> tuple[int, int]:
    st = path.stat()
    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
    return (int(mtime_ns), int(st.st_size))


def _cache_stem(csv_path: Path) -> str:
    h = hashlib.sha256(str(csv_path.resolve()).encode()).hexdigest()[:12]
    return f"{csv_path.stem}_{h}"


def _cache_paths(cache_dir: Path, stem: str) -> tuple[Path, Path]:
    return (cache_dir / f"{stem}.pkl", cache_dir / f"{stem}.meta.json")


def _meta_matches(meta_path: Path, csv_path: Path) -> bool:
    if not meta_path.is_file():
        return False
    try:
        with open(meta_path, encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    mt_ns, sz = _csv_signature(csv_path)
    return (
        m.get("cache_version") == _CACHE_VERSION
        and m.get("path") == str(csv_path.resolve())
        and int(m.get("mtime_ns", -1)) == mt_ns
        and int(m.get("size", -1)) == sz
    )


def _write_meta(meta_path: Path, csv_path: Path) -> None:
    mt_ns, sz = _csv_signature(csv_path)
    meta = {
        "cache_version": _CACHE_VERSION,
        "path": str(csv_path.resolve()),
        "mtime_ns": mt_ns,
        "size": sz,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def warmup_aero_dataset_caches(
    train_path: str | Path,
    val_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> None:
    """Parse ``train``/``val`` once and write pickle caches; safe to call before parallel training jobs."""
    train_path = Path(train_path)
    val_path = Path(val_path)
    cd = Path(cache_dir) if cache_dir is not None else default_aero_cache_dir(train_path)
    train_ds = AeroDataset(train_path, training=True, cache_dir=cd, use_cache=True)
    AeroDataset(val_path, stats=train_ds.stats, training=False, cache_dir=cd, use_cache=True)


class AeroDataset(Dataset):
    def __init__(
        self,
        csv_path,
        stats=None,
        training=True,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
    ):
        self.training = bool(training)
        self.csv_path = Path(csv_path)
        self._cache_dir = Path(cache_dir) if cache_dir is not None else default_aero_cache_dir(self.csv_path)
        self._use_cache = bool(use_cache)

        loaded = False
        if self._use_cache:
            loaded = self._try_load_from_cache(stats)

        if not loaded:
            self._parse_csv(stats)
            if self._use_cache:
                self._write_cache()

    def _try_load_from_cache(self, stats) -> bool:
        stem = _cache_stem(self.csv_path)
        pkl_path, meta_path = _cache_paths(self._cache_dir, stem)
        if not _meta_matches(meta_path, self.csv_path) or not pkl_path.is_file():
            return False
        try:
            with open(pkl_path, "rb") as f:
                blob = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError):
            return False
        if blob.get("cache_version") != _CACHE_VERSION or "df" not in blob:
            return False
        self.df = blob["df"]
        if self.training:
            self.stats = stats if stats is not None else blob.get("stats")
            if self.stats is None:
                return False
        else:
            if stats is None:
                raise ValueError("validation AeroDataset requires stats= (e.g. train_ds.stats)")
            self.stats = stats
        print(f"AeroDataset: loaded {len(self.df)} rows from cache {pkl_path.name}")
        return True

    def _parse_csv(self, stats):
        self.df = pd.read_csv(self.csv_path)

        tqdm.pandas(desc="Parsing coords")
        self.df["coords"] = self.df["coords"].progress_apply(json.loads)

        tqdm.pandas(desc="Parsing alpha")
        self.df["alpha"] = self.df["alpha"].progress_apply(json.loads)

        tqdm.pandas(desc="Parsing Cl")
        self.df["Cl"] = self.df["Cl"].progress_apply(json.loads)

        tqdm.pandas(desc="Parsing Cd")
        self.df["Cd"] = self.df["Cd"].progress_apply(json.loads)

        if self.training:
            if stats is None:
                self.stats = self._compute_stats()
            else:
                self.stats = stats
        else:
            if stats is None:
                raise ValueError("validation AeroDataset requires stats= when building from CSV")
            self.stats = stats

    def _write_cache(self):
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        stem = _cache_stem(self.csv_path)
        pkl_path, meta_path = _cache_paths(self._cache_dir, stem)
        if self.training:
            blob = {"cache_version": _CACHE_VERSION, "df": self.df, "stats": self.stats}
        else:
            blob = {"cache_version": _CACHE_VERSION, "df": self.df}
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(blob, f, protocol=4)
            _write_meta(meta_path, self.csv_path)
            print(f"AeroDataset: wrote cache {pkl_path.name}")
        except OSError as exc:
            print(f"AeroDataset: could not write cache ({exc}); continuing without.")

    def _compute_stats(self):
        all_polar = []
        for _, row in self.df.iterrows():
            tokens = build_polar_tokens(
                np.array(row['alpha']), np.array(row['Cl']),
                np.array(row['Cd']),   row['Re'], row['mach']
            )
            all_polar.append(tokens)
        all_polar = np.concatenate(all_polar, axis=0)
        return {
            'polar_mean': all_polar.mean(axis=0),
            'polar_std':  all_polar.std(axis=0) + 1e-8,
        }

    def normalize_polar(self, tokens):
        return (tokens - self.stats['polar_mean']) / self.stats['polar_std']

    def denormalize_polar(self, tokens):
        return tokens * self.stats['polar_std'] + self.stats['polar_mean']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- geometry ---
        coords_raw = resample_airfoil(row['coords'], n_points=200)
        coords, scale, x_min = normalize_coords(coords_raw)
        patches = build_geom_patches(coords)          # (20, 20)

        # --- polar ---
        tokens = build_polar_tokens(
            np.array(row['alpha']), np.array(row['Cl']),
            np.array(row['Cd']),    row['Re'], row['mach']
        )
        tokens = self.normalize_polar(tokens)         # (N, 5)
        n_polar = len(tokens)
        # Polar is a curve conditioned on geometry: keep every sample. Geometry is input-only.
        polar_mask = np.ones(n_polar, dtype=bool)

        return {
            'polar_tokens':  torch.tensor(tokens,   dtype=torch.float32),
            'geom_patches':  torch.tensor(patches,  dtype=torch.float32),
            'geom_coords':   torch.tensor(coords,   dtype=torch.float32),
            'geom_patch_surface': _GEOM_SURFACE_TORCH.clone(),
            'polar_mask':    torch.tensor(polar_mask),
            'geom_visible':  True,
            'scale':         scale,
            'x_min':         x_min,
        }
        
def collate_variable_sequences(batch):
    """
    Custom collate for DataLoader to handle variable-length polar arrays.
    Pads polar_tokens and polar_mask to max length in the batch.
    """
    polar_tokens_list = [item['polar_tokens'] for item in batch]
    polar_mask_list   = [item['polar_mask'] for item in batch]
    geom_patches_list = [item['geom_patches'] for item in batch]
    geom_visible_list = [item['geom_visible'] for item in batch]
    scale_list        = [item['scale'] for item in batch]
    x_min_list        = [item['x_min'] for item in batch]
    geom_surface_list = [item['geom_patch_surface'] for item in batch]

    # Stack fixed-size tensors
    geom_patches = torch.stack(geom_patches_list)           # (B, 20, 20)
    geom_patch_surface = torch.stack(geom_surface_list)     # (B, 20)
    geom_visible = torch.tensor(geom_visible_list)          # (B,)
    scale        = torch.tensor(scale_list, dtype=torch.float32)
    x_min        = torch.tensor(x_min_list, dtype=torch.float32)
    if "geom_coords" in batch[0]:
        geom_coords = torch.stack([item["geom_coords"] for item in batch])
    else:
        geom_coords = patches_to_coords(geom_patches)

    # Pad polar tokens and mask
    max_len = max(t.shape[0] for t in polar_tokens_list)
    B = len(polar_tokens_list)

    polar_tokens_padded = torch.zeros(B, max_len, 5)
    polar_mask_padded   = torch.zeros(B, max_len, dtype=torch.bool)
    polar_lengths = torch.zeros(B, dtype=torch.long)

    for i, (tok, msk) in enumerate(zip(polar_tokens_list, polar_mask_list)):
        length = tok.shape[0]
        polar_tokens_padded[i, :length] = tok
        polar_mask_padded[i, :length]   = msk
        polar_lengths[i] = length

    return {
        'polar_tokens': polar_tokens_padded,
        'polar_mask':   polar_mask_padded,
        'polar_lengths':polar_lengths,
        'geom_patches': geom_patches,
        'geom_coords': geom_coords,
        'geom_patch_surface': geom_patch_surface,
        'geom_visible': geom_visible,
        'scale':        scale,
        'x_min':        x_min,
    }