#!/usr/bin/env python3
"""
Emit train/val/test split statistics and geometry-overlap checks for paper disclosure.

Geometry identity is a hash of resampled/normalised coordinates (same rule as training:
arc-length resample to 200 points, then ``src.data.normalize_coords``). With the BigFoil
builder, splits are assigned by that hash, so train/val/test geometry sets should be
pairwise disjoint.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data import geom_fingerprint


def _load_split(path: Path) -> pd.DataFrame:
    cols = lambda c: c in ("coords", "Re", "mach", "airfoil")
    df = pd.read_csv(path, usecols=cols)
    tqdm.pandas(desc=f"parse coords {path.name}")
    df["coords"] = df["coords"].progress_apply(json.loads)
    return df


def _split_stats(df: pd.DataFrame, label: str) -> dict:
    tqdm.pandas(desc=f"{label} fingerprints")
    df = df.copy()
    df["_g"] = df["coords"].progress_apply(geom_fingerprint)
    geoms = set(df["_g"].unique())
    out = {
        "rows": int(len(df)),
        "unique_geometries": len(geoms),
        "unique_re": int(df["Re"].nunique()) if "Re" in df.columns else None,
        "unique_mach": int(df["mach"].nunique()) if "mach" in df.columns else None,
        "unique_airfoil_names": int(df["airfoil"].nunique()) if "airfoil" in df.columns else None,
        "_geoms": geoms,
        "_df": df,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/val/test split and geometry overlap report.")
    ap.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    ap.add_argument("--val-csv", type=Path, default=Path("data/val.csv"))
    ap.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/paper_final/dataset_split_report.json"))
    args = ap.parse_args()

    splits: dict[str, dict] = {}
    for name, path in (("train", args.train_csv), ("val", args.val_csv), ("test", args.test_csv)):
        if not path.is_file():
            raise SystemExit(f"Missing {name} CSV: {path}")
        splits[name] = _split_stats(_load_split(path), name)

    geoms = {k: v["_geoms"] for k, v in splits.items()}
    overlap = {
        "train_val": len(geoms["train"] & geoms["val"]),
        "train_test": len(geoms["train"] & geoms["test"]),
        "val_test": len(geoms["val"] & geoms["test"]),
    }

    def _re_overlap(a: str, b: str) -> dict:
        df_a, df_b = splits[a]["_df"], splits[b]["_df"]
        shared = geoms[a] & geoms[b]
        by_a: dict[str, set] = defaultdict(set)
        by_b: dict[str, set] = defaultdict(set)
        for g, r in zip(df_a["_g"], df_a["Re"]):
            by_a[g].add(float(r))
        for g, r in zip(df_b["_g"], df_b["Re"]):
            by_b[g].add(float(r))
        n_shared_re = sum(1 for g in shared if by_a[g] & by_b[g])
        return {
            "shared_geometries": len(shared),
            "shared_geometries_with_common_Re": n_shared_re,
        }

    out = {
        "schema": "dataset_split_report_v2",
        "paths": {
            "train_csv": str(args.train_csv),
            "val_csv": str(args.val_csv),
            "test_csv": str(args.test_csv),
        },
        "row_counts": {
            "train_rows": splits["train"]["rows"],
            "val_rows": splits["val"]["rows"],
            "test_rows": splits["test"]["rows"],
            "total_rows": splits["train"]["rows"] + splits["val"]["rows"] + splits["test"]["rows"],
            "note": "Splits are assigned by geometry fingerprint (tools/build_bigfoil_dataset.py). "
            "An airfoil shape cannot appear in more than one split; polar rows for that shape "
            "(all Re/Mach) stay together.",
        },
        "unique_geometry_hashes": {
            "train": splits["train"]["unique_geometries"],
            "val": splits["val"]["unique_geometries"],
            "test": splits["test"]["unique_geometries"],
            "train_val_intersection": overlap["train_val"],
            "train_test_intersection": overlap["train_test"],
            "val_test_intersection": overlap["val_test"],
        },
        "unique_names_and_conditions": {
            k: {
                "airfoil_names": splits[k]["unique_airfoil_names"],
                "Re": splits[k]["unique_re"],
                "mach": splits[k]["unique_mach"],
            }
            for k in ("train", "val", "test")
        },
        "geometry_held_out": {
            "train_val": _re_overlap("train", "val"),
            "train_test": _re_overlap("train", "test"),
            "val_test": _re_overlap("val", "test"),
            "ok": all(v == 0 for v in overlap.values()),
            "disclosure": "Main tables should use data/test.csv (geometry never seen in train). "
            "data/val.csv is for model selection only. Family-level holdout (e.g. all NACAs) "
            "is a separate experiment and is not this split.",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["row_counts"], indent=2))
    print(json.dumps(out["unique_geometry_hashes"], indent=2))
    print(f"Wrote {args.out}")
    if not out["geometry_held_out"]["ok"]:
        raise SystemExit("Geometry overlap across splits — not a held-out split.")


if __name__ == "__main__":
    main()
