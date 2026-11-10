import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

from src.fast_train import build_model, tensorize, GpuPolarData
from src.train_model import get_device


class _V:
    pass


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    v = _V()
    v.model = ck["variant_model"]
    v.training = ck["variant_training"]
    v.idx = ck["variant"]["id"]
    v.slug = ck["variant"]["slug"]
    v.name = ck["variant"]["name"]
    model, cfg = build_model(v, None)
    model.load_state_dict(ck["model"], strict=False)
    return model.to(device).eval(), ck


def audit(ckpt_path, csv_path, max_rows, device, tag):
    t0 = time.time()
    model, ck = load_model(ckpt_path, device)
    stats = ck["stats"]
    blob = tensorize(csv_path, stats, False)
    if max_rows > 0:
        for k in ("polar", "polar_raw", "lengths", "row_geom"):
            blob[k] = blob[k][:max_rows]
    data = GpuPolarData(blob, device)
    cl_mu, cl_sd = float(stats["polar_mean"][3]), float(stats["polar_std"][3])
    cd_mu, cd_sd = float(stats["polar_mean"][4]), float(stats["polar_std"][4])
    a_mu, a_sd = float(stats["polar_mean"][0]), float(stats["polar_std"][0])

    n_pts = 0
    n_neg_cd = 0
    n_pairs_lin = 0
    n_viol_lin = 0
    n_polars = 0
    n_polars_lin_viol = 0
    n_polars_conv_viol = 0
    n_polars_with_attached = 0

    with torch.no_grad():
        for idx in data.batches(512, shuffle=False):
            geom, polar, mask = data.gather(idx)
            final, _ = model(geom, polar, mask, p_ar=1.0, causal=True)
            a = (polar[..., 0] * a_sd + a_mu).cpu().numpy()
            cl = (final[..., 0] * cl_sd + cl_mu).cpu().numpy()
            cd = (final[..., 1] * cd_sd + cd_mu).cpu().numpy()
            m = mask.cpu().numpy()
            for i in range(a.shape[0]):
                row = m[i]
                if row.sum() < 3:
                    continue
                n_polars += 1
                aa, cc, dd = a[i][row], cl[i][row], cd[i][row]
                order = np.argsort(aa)
                aa, cc, dd = aa[order], cc[order], dd[order]
                n_pts += len(aa)
                n_neg_cd += int((dd <= 0.0).sum())

                lin = (aa >= -4.0) & (aa <= 6.0)
                idx_lin = np.where(lin)[0]
                viol = 0
                for j in idx_lin[:-1]:
                    n_pairs_lin += 1
                    if cc[j + 1] - cc[j] < -1e-4:
                        viol += 1
                n_viol_lin += viol
                n_polars_lin_viol += int(viol > 0)

                att = (aa >= -8.0) & (aa <= 8.0)
                ia = np.where(att)[0]
                if len(ia) >= 4:
                    n_polars_with_attached += 1
                    x = cc[ia] ** 2
                    y = dd[ia]
                    slopes = []
                    for j in range(len(ia) - 1):
                        dx = x[j + 1] - x[j]
                        if dx > 1e-6:
                            slopes.append((y[j + 1] - y[j]) / dx)
                    viol = 0
                    for j in range(len(slopes) - 1):
                        if slopes[j + 1] - slopes[j] < -1e-3:
                            viol += 1
                            break
                    n_polars_conv_viol += int(viol > 0)

    print(f"[audit] {tag}  ckpt={ckpt_path}")
    print(f"  rows={n_polars}  points={n_pts}  ({time.time() - t0:.0f}s)")
    print(f"  P(Cd <= 0)                    : {n_neg_cd / max(n_pts, 1) * 100:.4f}%  ({n_neg_cd} pts)")
    print(f"  P(dCl/da <= 0 | |a|<=6deg)     : {n_viol_lin / max(n_pairs_lin, 1) * 100:.4f}% of pairs; "
          f"{n_polars_lin_viol / max(n_polars, 1) * 100:.2f}% of polars")
    print(f"  P(non-convex drag polar, |a|<=8): {n_polars_conv_viol / max(n_polars_with_attached, 1) * 100:.2f}% of polars")


def main():
    ap = argparse.ArgumentParser(description="Physics-violation audit on held-out polars")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="model")
    args = ap.parse_args()
    device = torch.device(args.device or get_device())
    audit(args.ckpt, args.csv, args.max_rows, device, args.tag)


if __name__ == "__main__":
    main()
