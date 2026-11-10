"""Public streaming-regression datasets used to test whether FoilCORE's training recipe transfers.

Every dataset is turned into the same shape as an airfoil polar: fixed-length windows
``x: (N, T, n_in)`` of pointwise conditions and ``y: (N, T)`` of the target, z-scored with
training statistics. The task is always "predict y_t from x_t, refined by the neighbours",
which is what the polar task is with (α, Re, Ma) → (Cl, Cd).

  ett   ETTh1 (electricity transformer, hourly): 6 load channels + time-of-day/week/year → oil temperature.
        Windows T=96 (4 days). Split by held-out 4-week blocks (virtual-sensor task, not forecasting).
  air   Beijing PM2.5 (UCI, hourly): dew point, temperature, pressure, wind → log1p(PM2.5).
        Windows T=72 (3 days). Split by year: 2010-12 / 2013 / 2014.
  ecg   MIT-BIH arrhythmia (PhysioNet): the second lead (mostly V1) → lead MLII, 90 Hz,
        windows T=128 (1.4 s). Split by record (32 / 6 / 10). Needs ``pip install wfdb``.

Downloads go to ``experiments/transfer/data/`` and are cached as .npz.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"


def _fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return dest
    print(f"[transfer] downloading {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def _windows(x: np.ndarray, y: np.ndarray, t: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for s in range(0, len(x) - t + 1, stride):
        xw, yw = x[s : s + t], y[s : s + t]
        if np.isnan(xw).any() or np.isnan(yw).any():
            continue
        xs.append(xw)
        ys.append(yw)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def _zscore(train_x, train_y, *others):
    mx, sx = train_x.reshape(-1, train_x.shape[-1]).mean(0), train_x.reshape(-1, train_x.shape[-1]).std(0) + 1e-6
    my, sy = float(train_y.mean()), float(train_y.std() + 1e-6)
    out = [((train_x - mx) / sx, (train_y - my) / sy)]
    for x, y in others:
        out.append(((x - mx) / sx, (y - my) / sy))
    return out, {"y_mean": my, "y_std": sy}


def _time_feats(hours: np.ndarray, dows: np.ndarray) -> np.ndarray:
    return np.stack([
        np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24),
        np.sin(2 * np.pi * dows / 7), np.cos(2 * np.pi * dows / 7),
    ], 1)


def _block_split(x, y, t, *, block: int, stride_train: int, stride_eval: int):
    """Assign contiguous blocks to train/val/test (test = every 6th block, val = every 6th offset by 3)."""
    n_blocks = len(x) // block
    parts = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for b in range(n_blocks):
        role = "test" if b % 6 == 0 else ("val" if b % 6 == 3 else "train")
        xb, yb = x[b * block : (b + 1) * block], y[b * block : (b + 1) * block]
        xw, yw = _windows(xb, yb, t, stride_train if role == "train" else stride_eval)
        if len(xw):
            parts[role][0].append(xw); parts[role][1].append(yw)
    cat = lambda p: (np.concatenate(p[0]), np.concatenate(p[1]))
    return cat(parts["train"]), cat(parts["val"]), cat(parts["test"])


# ----------------------------------------------------------------------------- ETTh1
def load_ett(t: int = 96, lag: int = 0) -> dict:
    """lag>0 adds OT_{t-lag} as an input feature (forecaster-style information set; the default is pure virtual sensing)."""
    import pandas as pd

    cache = DATA_DIR / (f"ett_T{t}.npz" if lag == 0 else f"ett_T{t}_lag{lag}.npz")
    if cache.is_file():
        return dict(np.load(cache, allow_pickle=True))
    p = _fetch("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv", DATA_DIR / "ETTh1.csv")
    df = pd.read_csv(p, parse_dates=["date"])
    feats = df[["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]].values.astype(np.float64)
    tf = _time_feats(df["date"].dt.hour.values, df["date"].dt.dayofweek.values)
    doy = df["date"].dt.dayofyear.values / 365.25
    season = np.stack([np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy)], 1)  # ambient temperature proxy
    x = np.concatenate([feats, tf, season], 1)
    y = df["OT"].values.astype(np.float64)
    if lag > 0:
        ylag = np.concatenate([np.full(lag, np.nan), y[:-lag]])[:, None]
        x = np.concatenate([x, ylag], 1)  # windows containing the leading NaNs are dropped by _windows
    # virtual-sensor task, so the split is by held-out 4-week blocks (every 6th block test, every 6th val,
    # windows never straddle a block); the standard forecasting split puts all of winter 2017/18 in test
    # and OT from loads alone does not transfer across that shift.
    tr, va, te = _block_split(x, y, t, block=4 * 7 * 24, stride_train=8, stride_eval=24)
    (tr, va, te), st = _zscore(*tr, va, te)
    blob = {"x_train": tr[0], "y_train": tr[1], "x_val": va[0], "y_val": va[1], "x_test": te[0], "y_test": te[1],
            "y_mean": st["y_mean"], "y_std": st["y_std"], "name": "ETTh1 oil temperature" + (f" (+OT lag {lag})" if lag else ""), "unit": "degC"}
    np.savez(cache, **blob)
    return blob


# ----------------------------------------------------------------------------- Beijing PM2.5
def load_air(t: int = 72) -> dict:
    import pandas as pd

    cache = DATA_DIR / f"air_T{t}.npz"
    if cache.is_file():
        return dict(np.load(cache, allow_pickle=True))
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
    p = _fetch(url, DATA_DIR / "PRSA_data_2010.1.1-2014.12.31.csv")
    df = pd.read_csv(p)
    wind = pd.get_dummies(df["cbwd"]).values.astype(np.float64)
    feats = df[["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]].values.astype(np.float64)
    tf = _time_feats(df["hour"].values, pd.to_datetime(df[["year", "month", "day"]]).dt.dayofweek.values)
    x = np.concatenate([feats, wind, tf], 1)
    y = np.log1p(df["pm2.5"].values.astype(np.float64))
    yr = df["year"].values
    tr = _windows(x[yr <= 2012], y[yr <= 2012], t, stride=8)
    va = _windows(x[yr == 2013], y[yr == 2013], t, stride=24)
    te = _windows(x[yr == 2014], y[yr == 2014], t, stride=24)
    (tr, va, te), st = _zscore(*tr, va, te)
    blob = {"x_train": tr[0], "y_train": tr[1], "x_val": va[0], "y_val": va[1], "x_test": te[0], "y_test": te[1],
            "y_mean": st["y_mean"], "y_std": st["y_std"], "name": "Beijing PM2.5", "unit": "log1p(ug/m3)"}
    np.savez(cache, **blob)
    return blob


# ----------------------------------------------------------------------------- MIT-BIH ECG
def load_ecg(t: int = 128, down: int = 4, seed: int = 0) -> dict:
    cache = DATA_DIR / f"ecg_T{t}_d{down}.npz"
    if cache.is_file():
        return dict(np.load(cache, allow_pickle=True))
    import zipfile

    import wfdb  # pip install wfdb

    zpath = _fetch("https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip",
                   DATA_DIR / "mitdb.zip")
    mit_dir = DATA_DIR / "mit-bih-arrhythmia-database-1.0.0"
    if not mit_dir.is_dir():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(DATA_DIR)
    recs = sorted({p.stem for p in mit_dir.glob("*.dat")})
    xs_by_rec, ys_by_rec = {}, {}
    for r in recs:
        try:
            rec = wfdb.rdrecord(str(mit_dir / r))
        except Exception as e:  # noqa: BLE001
            print(f"[transfer] ecg record {r} failed: {e}", flush=True)
            continue
        names = list(rec.sig_name)
        if "MLII" not in names or len(names) < 2:
            continue
        i_y = names.index("MLII")
        i_x = 1 - i_y
        sig = rec.p_signal.astype(np.float64)
        n = (len(sig) // down) * down
        sig = sig[:n].reshape(-1, down, 2).mean(1)  # 360 Hz → 90 Hz
        x = sig[:, i_x : i_x + 1]
        y = sig[:, i_y]
        xw, yw = _windows(x, y, t, stride=t)
        xs_by_rec[r], ys_by_rec[r] = xw, yw
    keys = sorted(xs_by_rec)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_te, n_va = 10, 6
    te_k, va_k, tr_k = keys[:n_te], keys[n_te : n_te + n_va], keys[n_te + n_va :]
    cat = lambda ks: (np.concatenate([xs_by_rec[k] for k in ks]), np.concatenate([ys_by_rec[k] for k in ks]))
    tr, va, te = cat(tr_k), cat(va_k), cat(te_k)
    (tr, va, te), st = _zscore(*tr, va, te)
    blob = {"x_train": tr[0], "y_train": tr[1], "x_val": va[0], "y_val": va[1], "x_test": te[0], "y_test": te[1],
            "y_mean": st["y_mean"], "y_std": st["y_std"], "name": "MIT-BIH lead MLII from lead 2", "unit": "mV",
            "records_test": np.array(te_k), "records_val": np.array(va_k), "records_train": np.array(tr_k)}
    np.savez(cache, **blob)
    return blob


LOADERS = {"ett": load_ett, "air": load_air, "ecg": load_ecg, "ett_lag1": lambda: load_ett(lag=1)}


def load(name: str) -> dict:
    return LOADERS[name]()


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(LOADERS)):
        try:
            b = load(n)
            print(n, b["name"], "train", b["x_train"].shape, "val", b["x_val"].shape, "test", b["x_test"].shape)
        except Exception as e:  # noqa: BLE001
            print(n, "FAILED:", e)
