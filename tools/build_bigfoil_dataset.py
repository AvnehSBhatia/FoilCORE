#!/usr/bin/env python3
"""
Build FoilCORE train/val/test CSVs from a BigFoil dump (paired .dat + .pkl).

Design (NeurIPS-style evaluation, not a random row shuffle):

* Labels are XFOIL ncrit=9 only. NeuralFoil-generated ("NF") and JavaFoil polars
  are dropped so a later NeuralFoil comparison is not circular and the solver
  is not mixed inside the target.
* Geometry is canonicalised (trailing-edge start, upper surface first) and
  checked (chord, thickness, trailing-edge gap).
* Polars must be a usable sweep: finite Cl/Cd, Cd>=0, dense alpha, and coverage
  of both negative and positive alpha so autoregressive stall tests mean something.
* Split unit is a near-duplicate geometry cluster, not a polar row. Exact
  duplicate files are collapsed; near-duplicates stay in the same split.
* Clusters are stratified by thickness x camber so val/test are not accidentally
  all thin or all highly cambered.
* Val/test rows are written round-robin across airfoils so a ``--max-samples``
  prefix still covers many shapes.

Usage (from repo root):

  python tools/build_bigfoil_dataset.py \\
      --src /path/to/BIGFOIL20250823 \\
      --out-dir data --seed 42
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pandas.compat import pickle_compat
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data import geom_fingerprint, normalize_coords, resample_airfoil  # noqa: E402
from src.geom_metrics import max_thickness_ratio_and_camber  # noqa: E402

CSV_COLUMNS = [
    "coords",
    "alpha",
    "Cl",
    "Cd",
    "Re",
    "mach",
    "airfoil",
    "datasource",
    "roughness",
]

DEFAULT_SOURCES = ("XFOIL ncrit=9",)


def _dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))


def load_selig_dat(path: Path) -> np.ndarray:
    pts: list[list[float]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        pts.append([x, y])
    if len(pts) < 20:
        raise ValueError(f"too few coordinate pairs ({len(pts)})")
    return np.asarray(pts, dtype=np.float64)


def canonicalize_loop(coords: np.ndarray) -> np.ndarray:
    """Selig-style closed loop: start at TE (max x), upper surface first."""
    coords = np.asarray(coords, dtype=np.float64)
    chord = float(coords[:, 0].max() - coords[:, 0].min())
    if chord <= 1e-8:
        raise ValueError("zero chord")
    if np.linalg.norm(coords[0] - coords[-1]) < 1e-5 * chord:
        coords = coords[:-1]
        if len(coords) < 20:
            raise ValueError("too few points after dropping closing TE")
    i_te = int(np.argmax(coords[:, 0]))
    coords = np.roll(coords, -i_te, axis=0)
    n = len(coords)
    if float(np.mean(coords[: n // 2, 1])) < float(np.mean(coords[n // 2 :, 1])):
        coords = np.vstack([coords[0:1], coords[:0:-1]])
    return coords


def coarse_geom_key(coords: np.ndarray, quant: float = 0.004) -> str:
    resampled = resample_airfoil(coords, n_points=200)
    normed, scale, _ = normalize_coords(resampled)
    if float(scale) <= 1e-8:
        raise ValueError("zero chord")
    q = np.round(normed / quant).astype(np.int16)
    return hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest()


def geometry_qc(
    coords: np.ndarray,
    *,
    min_tc: float,
    max_tc: float,
    max_te_gap: float,
    max_abs_y: float,
) -> tuple[float, float] | str:
    """Return (t/c, |camber|) or a reason string."""
    chord = float(coords[:, 0].max() - coords[:, 0].min())
    if chord <= 1e-8:
        return "zero_chord"
    te_gap = float(np.linalg.norm(coords[0] - coords[-1]) / chord)
    if te_gap > max_te_gap:
        return "open_te"
    try:
        resampled = resample_airfoil(coords, n_points=200)
        normed, scale, _ = normalize_coords(resampled)
    except (ValueError, ZeroDivisionError):
        return "resample_fail"
    if float(scale) <= 1e-8:
        return "zero_chord"
    if float(np.max(np.abs(normed[:, 1]))) > max_abs_y:
        return "huge_y"
    tc, camber = max_thickness_ratio_and_camber(coords)
    if not (math.isfinite(tc) and math.isfinite(camber)):
        return "tc_nan"
    if tc < min_tc:
        return "too_thin"
    if tc > max_tc:
        return "too_thick"
    return float(tc), float(camber)


def _as_float_list(values) -> list[float] | None:
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        return None
    out: list[float] = []
    for v in values:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x):
            return None
        out.append(x)
    return out


def _clean_polar(
    alpha: list[float],
    cl: list[float],
    cd: list[float],
    *,
    min_len: int,
    max_abs_cl: float,
    max_abs_alpha: float,
    cd_neg_tol: float,
    max_cd: float,
    min_alpha: float,
    max_alpha: float,
    max_gap: float,
) -> tuple[list[float], list[float], list[float]] | str:
    if not (len(alpha) == len(cl) == len(cd)):
        return "length_mismatch"
    order = np.argsort(alpha, kind="mergesort")
    a = [alpha[i] for i in order]
    cl_s = [cl[i] for i in order]
    cd_s = [cd[i] for i in order]

    uniq_a: list[float] = []
    uniq_cl: list[float] = []
    uniq_cd: list[float] = []
    prev = None
    for x, y, z in zip(a, cl_s, cd_s):
        if prev is not None and x == prev:
            continue
        if abs(x) > max_abs_alpha or abs(y) > max_abs_cl:
            continue
        if z < -cd_neg_tol or z > max_cd:
            continue
        uniq_a.append(x)
        uniq_cl.append(y)
        uniq_cd.append(0.0 if z < 0.0 else z)
        prev = x

    if len(uniq_a) < min_len:
        return "short"
    if uniq_a[0] > min_alpha or uniq_a[-1] < max_alpha:
        return "narrow_alpha"
    gaps = np.diff(uniq_a)
    if float(np.max(gaps)) > max_gap:
        return "alpha_gap"
    return uniq_a, uniq_cl, uniq_cd


def _is_nf_source(name: str) -> bool:
    s = name.strip()
    return s == "NF" or s.startswith("NF ")


def parse_airfoil(
    pkl_path: Path,
    *,
    allowed_sources: set[str] | None,
    exclude_nf: bool,
    min_polar_len: int,
    max_abs_cl: float,
    max_abs_alpha: float,
    cd_neg_tol: float,
    max_cd: float,
    min_alpha: float,
    max_alpha: float,
    max_gap: float,
    min_re: float,
    max_mach: float,
    min_tc: float,
    max_tc: float,
    max_te_gap: float,
    max_abs_y: float,
    min_polars: int,
) -> dict:
    stem = pkl_path.stem
    dat_path = pkl_path.with_suffix(".dat")
    rec = {
        "stem": stem,
        "ok": False,
        "reason": None,
        "geom_hash": None,
        "cluster_key": None,
        "n_coords": 0,
        "coords": None,
        "tc": None,
        "camber": None,
        "rows": [],
        "n_source_rows": 0,
        "drop_reasons": Counter(),
    }
    if not dat_path.is_file():
        rec["reason"] = "missing_dat"
        return rec
    try:
        coords = canonicalize_loop(load_selig_dat(dat_path))
        qc = geometry_qc(
            coords,
            min_tc=min_tc,
            max_tc=max_tc,
            max_te_gap=max_te_gap,
            max_abs_y=max_abs_y,
        )
        if isinstance(qc, str):
            rec["reason"] = qc
            rec["drop_reasons"][qc] += 1
            return rec
        rec["tc"], rec["camber"] = qc
        rec["n_coords"] = int(coords.shape[0])
        rec["geom_hash"] = geom_fingerprint(coords)
        rec["cluster_key"] = coarse_geom_key(coords)
        rec["coords"] = coords.tolist()
    except Exception as exc:
        rec["reason"] = f"bad_dat:{type(exc).__name__}"
        rec["drop_reasons"][rec["reason"]] += 1
        return rec

    try:
        with open(pkl_path, "rb") as fh:
            df = pickle_compat.load(fh)
    except Exception as exc:
        rec["reason"] = f"bad_pkl:{type(exc).__name__}"
        rec["drop_reasons"][rec["reason"]] += 1
        return rec

    rec["n_source_rows"] = int(len(df))
    rows = []
    drops = rec["drop_reasons"]
    seen_re_mach: set[tuple[float, float]] = set()
    for _, row in df.iterrows():
        src = str(row.get("datasource", "") or "")
        if exclude_nf and _is_nf_source(src):
            drops["nf_source"] += 1
            continue
        if allowed_sources is not None and src not in allowed_sources:
            drops["datasource"] += 1
            continue
        roughness = row.get("roughness")
        roughness_s = "" if roughness is None else str(roughness).strip().lower()
        if roughness_s and roughness_s not in {"smooth", "nan", "none"}:
            drops["rough"] += 1
            continue
        try:
            re_val = float(row["Re"])
            mach = float(row["Mach"])
        except (TypeError, ValueError, KeyError):
            drops["re_mach"] += 1
            continue
        if not math.isfinite(re_val) or re_val < min_re:
            drops["re_mach"] += 1
            continue
        if not math.isfinite(mach) or mach < 0.0 or mach > max_mach:
            drops["re_mach"] += 1
            continue
        key = (round(re_val, 3), round(mach, 4))
        if key in seen_re_mach:
            drops["dup_re_mach"] += 1
            continue

        alpha = _as_float_list(row.get("alpha"))
        cl = _as_float_list(row.get("Cl"))
        cd = _as_float_list(row.get("Cd"))
        if alpha is None or cl is None or cd is None:
            drops["nonfinite_or_type"] += 1
            continue
        cleaned = _clean_polar(
            alpha,
            cl,
            cd,
            min_len=min_polar_len,
            max_abs_cl=max_abs_cl,
            max_abs_alpha=max_abs_alpha,
            cd_neg_tol=cd_neg_tol,
            max_cd=max_cd,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            max_gap=max_gap,
        )
        if isinstance(cleaned, str):
            drops[cleaned] += 1
            continue
        a, cl_c, cd_c = cleaned
        seen_re_mach.add(key)
        rows.append(
            {
                "alpha": a,
                "Cl": cl_c,
                "Cd": cd_c,
                "Re": re_val,
                "mach": mach,
                "datasource": src,
                "roughness": "smooth",
            }
        )

    if len(rows) < min_polars:
        rec["reason"] = "too_few_polars"
        rec["drop_reasons"]["too_few_polars"] += 1
        return rec
    rec["ok"] = True
    rec["rows"] = rows
    rec["reason"] = None
    return rec


def _tertile_ids(values: list[float]) -> list[int]:
    v = np.asarray(values, dtype=float)
    q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
    out = []
    for x in v:
        if x <= q1:
            out.append(0)
        elif x <= q2:
            out.append(1)
        else:
            out.append(2)
    return out


def _assign_stratified(
    cluster_ids: list[str],
    strata: list[str],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> dict[str, str]:
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for cid, st in zip(cluster_ids, strata):
        by_stratum[st].append(cid)
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for st, members in sorted(by_stratum.items()):
        uniq = list(dict.fromkeys(members))
        rng.shuffle(uniq)
        n = len(uniq)
        if n == 1:
            assignment[uniq[0]] = "train"
            continue
        if n == 2:
            assignment[uniq[0]] = "train"
            assignment[uniq[1]] = "test"
            continue
        n_train = max(1, int(round(train_frac * n)))
        n_val = max(1, int(round(val_frac * n)))
        if n_train + n_val >= n:
            n_val = max(1, min(n_val, n - 2))
            n_train = n - n_val - 1
        for i, cid in enumerate(uniq):
            if i < n_train:
                assignment[cid] = "train"
            elif i < n_train + n_val:
                assignment[cid] = "val"
            else:
                assignment[cid] = "test"
    return assignment


def _round_robin_rows(groups: list[list[dict]], rng: np.random.Generator) -> list[dict]:
    """Emit one polar at a time from each airfoil so a file prefix is geometry-diverse."""
    order = np.arange(len(groups))
    rng.shuffle(order)
    buckets = [list(groups[i]) for i in order]
    for b in buckets:
        b.sort(key=lambda r: (r["Re"], r["mach"]))
    out: list[dict] = []
    max_len = max((len(b) for b in buckets), default=0)
    for k in range(max_len):
        for b in buckets:
            if k < len(b):
                out.append(b[k])
    return out


def _write_split_csv(path: Path, records: list[dict], coords_by_stem: dict[str, list]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "coords": _dumps(coords_by_stem[rec["airfoil"]]),
                    "alpha": _dumps(rec["alpha"]),
                    "Cl": _dumps(rec["Cl"]),
                    "Cd": _dumps(rec["Cd"]),
                    "Re": f"{rec['Re']:.6g}",
                    "mach": f"{rec['mach']:.6g}",
                    "airfoil": rec["airfoil"],
                    "datasource": rec["datasource"],
                    "roughness": rec["roughness"],
                }
            )
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Build geometry-held-out FoilCORE CSVs from BigFoil.")
    ap.add_argument("--src", type=Path, required=True, help="Directory of paired .dat / .pkl files.")
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.80)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument(
        "--datasource",
        action="append",
        default=None,
        help="Keep this datasource (repeatable). Default: XFOIL ncrit=9. Use 'all' for every non-NF source.",
    )
    ap.add_argument(
        "--allow-neuralfoil-labels",
        action="store_true",
        help="Do not drop datasource=NF. Invalidates a NeuralFoil baseline comparison.",
    )
    ap.add_argument("--min-polar-len", type=int, default=25)
    ap.add_argument("--min-polars-per-airfoil", type=int, default=8)
    ap.add_argument("--max-abs-cl", type=float, default=4.0)
    ap.add_argument("--max-abs-alpha", type=float, default=25.0)
    ap.add_argument("--min-alpha", type=float, default=-8.0, help="Polar must start at or below this α.")
    ap.add_argument("--max-alpha-end", type=float, default=8.0, help="Polar must end at or above this α.")
    ap.add_argument("--max-alpha-gap", type=float, default=1.51)
    ap.add_argument("--cd-neg-tol", type=float, default=1e-4)
    ap.add_argument("--max-cd", type=float, default=0.6)
    ap.add_argument("--min-re", type=float, default=5e4)
    ap.add_argument("--max-mach", type=float, default=0.5)
    ap.add_argument("--min-tc", type=float, default=0.03)
    ap.add_argument("--max-tc", type=float, default=0.35)
    ap.add_argument("--max-te-gap", type=float, default=0.02)
    ap.add_argument("--max-abs-y", type=float, default=0.6)
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"Source directory not found: {src}")

    if args.datasource is None:
        allowed: set[str] | None = set(DEFAULT_SOURCES)
    elif any(s.strip().lower() == "all" for s in args.datasource):
        allowed = None
    else:
        allowed = {s.strip() for s in args.datasource}

    exclude_nf = not args.allow_neuralfoil_labels
    pkls = sorted(src.glob("*.pkl"))
    if not pkls:
        raise SystemExit(f"No .pkl files in {src}")

    airfoils: list[dict] = []
    fail_reasons = Counter()
    drop_reasons = Counter()
    source_kept = Counter()
    for pkl_path in tqdm(pkls, desc="Parse BigFoil"):
        rec = parse_airfoil(
            pkl_path,
            allowed_sources=allowed,
            exclude_nf=exclude_nf,
            min_polar_len=args.min_polar_len,
            max_abs_cl=args.max_abs_cl,
            max_abs_alpha=args.max_abs_alpha,
            cd_neg_tol=args.cd_neg_tol,
            max_cd=args.max_cd,
            min_alpha=args.min_alpha,
            max_alpha=args.max_alpha_end,
            max_gap=args.max_alpha_gap,
            min_re=args.min_re,
            max_mach=args.max_mach,
            min_tc=args.min_tc,
            max_tc=args.max_tc,
            max_te_gap=args.max_te_gap,
            max_abs_y=args.max_abs_y,
            min_polars=args.min_polars_per_airfoil,
        )
        drop_reasons.update(rec["drop_reasons"])
        if not rec["ok"]:
            fail_reasons[rec["reason"] or "unknown"] += 1
            continue
        airfoils.append(rec)
        for row in rec["rows"]:
            source_kept[row["datasource"]] += 1

    if not airfoils:
        raise SystemExit("No airfoils survived filters.")

    # Collapse exact-duplicate geometries: keep the file with the most valid polars.
    by_exact: dict[str, list[dict]] = defaultdict(list)
    for a in airfoils:
        by_exact[a["geom_hash"]].append(a)
    collapsed: list[dict] = []
    n_exact_dropped = 0
    for group in by_exact.values():
        group.sort(key=lambda r: (-len(r["rows"]), r["stem"]))
        collapsed.append(group[0])
        n_exact_dropped += len(group) - 1
    airfoils = collapsed

    # Near-duplicate clusters (quantised coordinates) stay in one split.
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for a in airfoils:
        by_cluster[a["cluster_key"]].append(a)
    cluster_ids = sorted(by_cluster)
    cluster_tc = []
    cluster_cam = []
    for cid in cluster_ids:
        members = by_cluster[cid]
        cluster_tc.append(float(np.median([m["tc"] for m in members])))
        cluster_cam.append(float(np.median([m["camber"] for m in members])))
    t_ids = _tertile_ids(cluster_tc)
    c_ids = _tertile_ids(cluster_cam)
    strata = [f"t{t}c{c}" for t, c in zip(t_ids, c_ids)]
    stratum_of = dict(zip(cluster_ids, strata))
    assignment = _assign_stratified(
        cluster_ids,
        strata,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    coords_by_stem = {a["stem"]: a["coords"] for a in airfoils}
    split_groups: dict[str, list[list[dict]]] = {"train": [], "val": [], "test": []}
    split_airfoils: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    cluster_split_count = Counter()
    for cid, members in by_cluster.items():
        split = assignment[cid]
        cluster_split_count[split] += 1
        for a in members:
            rows = [{"airfoil": a["stem"], **row} for row in a["rows"]]
            split_groups[split].append(rows)
            split_airfoils[split].append(
                {
                    "airfoil": a["stem"],
                    "geom_hash": a["geom_hash"],
                    "cluster_key": a["cluster_key"],
                    "n_rows": len(a["rows"]),
                    "n_coords": a["n_coords"],
                    "tc": a["tc"],
                    "camber": a["camber"],
                    "stratum": stratum_of[cid],
                }
            )

    rng = np.random.default_rng(args.seed)
    split_rows: dict[str, list[dict]] = {}
    # Train stays grouped by airfoil (stable, cache-friendly). Val/test are
    # round-robin so --max-samples prefixes remain geometry-diverse.
    train_flat: list[dict] = []
    for group in split_groups["train"]:
        group.sort(key=lambda r: (r["airfoil"], r["Re"], r["mach"]))
        train_flat.extend(group)
    train_flat.sort(key=lambda r: (r["airfoil"], r["Re"], r["mach"]))
    split_rows["train"] = train_flat
    split_rows["val"] = _round_robin_rows(split_groups["val"], rng)
    split_rows["test"] = _round_robin_rows(split_groups["test"], rng)

    out_dir = args.out_dir
    counts = {}
    for split in ("train", "val", "test"):
        split_airfoils[split].sort(key=lambda r: r["airfoil"])
        n = _write_split_csv(out_dir / f"{split}.csv", split_rows[split], coords_by_stem)
        counts[split] = n
        print(
            f"Wrote {out_dir / f'{split}.csv'}  "
            f"({n} rows, {len(split_airfoils[split])} airfoils, "
            f"{cluster_split_count[split]} clusters)"
        )

    n_unique_geom = len({a["geom_hash"] for a in airfoils})
    n_clusters = len(by_cluster)
    n_near = sum(1 for g in by_cluster.values() if len(g) > 1)
    manifest = {
        "schema": "foilcore_bigfoil_split_v2",
        "source_dir": str(src),
        "seed": int(args.seed),
        "split_unit": "near_duplicate_geometry_cluster",
        "eval_row_order": {
            "train": "grouped by airfoil, then Re, Mach",
            "val": "round-robin across shuffled airfoils",
            "test": "round-robin across shuffled airfoils",
        },
        "split_fractions": {
            "train": args.train_frac,
            "val": args.val_frac,
            "test": round(1.0 - args.train_frac - args.val_frac, 6),
        },
        "filters": {
            "datasources": sorted(allowed) if allowed is not None else "all_non_nf",
            "exclude_neuralfoil_labels": exclude_nf,
            "min_polar_len": args.min_polar_len,
            "min_polars_per_airfoil": args.min_polars_per_airfoil,
            "max_abs_cl": args.max_abs_cl,
            "max_abs_alpha": args.max_abs_alpha,
            "min_alpha": args.min_alpha,
            "max_alpha_end": args.max_alpha_end,
            "max_alpha_gap": args.max_alpha_gap,
            "cd_neg_tol": args.cd_neg_tol,
            "max_cd": args.max_cd,
            "min_re": args.min_re,
            "max_mach": args.max_mach,
            "min_tc": args.min_tc,
            "max_tc": args.max_tc,
            "max_te_gap": args.max_te_gap,
        },
        "counts": {
            "pkl_files": len(pkls),
            "airfoils_kept": len(airfoils),
            "airfoils_dropped": sum(fail_reasons.values()),
            "exact_duplicate_files_dropped": n_exact_dropped,
            "unique_geometries": n_unique_geom,
            "near_dup_clusters": n_clusters,
            "clusters_with_multiple_files": n_near,
            "rows": counts,
            "rows_total": sum(counts.values()),
            "airfoils_per_split": {k: len(v) for k, v in split_airfoils.items()},
            "clusters_per_split": dict(cluster_split_count),
        },
        "datasource_rows_kept": dict(source_kept),
        "airfoils_dropped_reasons": dict(fail_reasons),
        "polar_drop_reasons": dict(drop_reasons),
        "splits": split_airfoils,
    }
    man_path = out_dir / "split_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {man_path}")
    print(json.dumps(manifest["counts"], indent=2))

    sets = {k: {a["cluster_key"] for a in v} for k, v in split_airfoils.items()}
    geom_sets = {k: {a["geom_hash"] for a in v} for k, v in split_airfoils.items()}
    overlap = {
        "cluster_train∩val": len(sets["train"] & sets["val"]),
        "cluster_train∩test": len(sets["train"] & sets["test"]),
        "cluster_val∩test": len(sets["val"] & sets["test"]),
        "geom_train∩val": len(geom_sets["train"] & geom_sets["val"]),
        "geom_train∩test": len(geom_sets["train"] & geom_sets["test"]),
        "geom_val∩test": len(geom_sets["val"] & geom_sets["test"]),
    }
    print("overlap", overlap)
    if any(overlap.values()):
        raise SystemExit("Geometry leakage across splits — aborting.")

    # Stratum balance (airfoils).
    for split, items in split_airfoils.items():
        st = Counter(a["stratum"] for a in items)
        print(f"stratum {split}:", dict(sorted(st.items())))


if __name__ == "__main__":
    main()
