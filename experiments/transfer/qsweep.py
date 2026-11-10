"""Summarize the leak-fraction sweep (Proposition 1 test) and the consistency-loss baseline.

Runs come from train.py with --protocol draft_dual --q Q --draft-noise S (dual mask, fixed leak) and
--protocol gtmix_consist. Writes summary.json / summary.md, a LaTeX table and a two-panel figure.

python experiments/transfer/qsweep.py runs/transfer_qsweep --tex paper/tables/ts_qsweep.tex --fig paper/figures/ts/qsweep.pdf
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--tex", default=None)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()
    rd = Path(args.run_dir)
    groups = defaultdict(list)
    for p in sorted(rd.glob("*.json")):
        if p.name.startswith("summary"):
            continue
        r = json.loads(p.read_text())
        key = (r["dataset"], "consist" if r["protocol"] == "gtmix_consist" else r["q"], r.get("draft_noise", 0.0))
        groups[key].append(r)

    rows = []
    for (ds, q, dn), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
        a = lambda k: np.array([r[k] for r in rs], dtype=float)
        stream, smooth = a("test_causal_mae"), a("test_bidir_mae")
        # Proposition 1 quantities in z units, averaged over the two masks
        dep = 0.5 * (a("test_causal_mae_z") + a("test_bidir_mae_z"))
        lk = 0.5 * (a("test_leaked_causal_mae_z") + a("test_leaked_bidir_mae_z"))
        e_in = a("test_corrector_input_err_z")
        qq = 0.0 if q == "consist" else float(q)
        rows.append({
            "dataset": ds, "q": q, "draft_noise": dn, "n": len(rs), "unit": rs[0]["unit"],
            "stream": float(stream.mean()), "stream_sd": float(stream.std()),
            "smooth": float(smooth.mean()), "smooth_sd": float(smooth.std()),
            "gap_pct": float((100 * (stream - smooth) / smooth).mean()),
            "deploy_z": float(dep.mean()), "leaked_z": float(lk.mean()),
            "discrepancy_z": float((dep - lk).mean()), "q_times_err_z": float(qq * e_in.mean()), "input_err_z": float(e_in.mean()),
        })
    (rd / "summary.json").write_text(json.dumps(rows, indent=1))
    hdr = "| dataset | q | draft noise | n | stream | smooth | gap % | deploy − leaked (z) | q·E|y−ỹ| (z) |\n|---|---|---|---|---|---|---|---|---|\n"
    md = hdr + "".join(
        f"| {r['dataset']} | {r['q']} | {r['draft_noise']} | {r['n']} | {r['stream']:.3f} ± {r['stream_sd']:.3f} | "
        f"{r['smooth']:.3f} ± {r['smooth_sd']:.3f} | {r['gap_pct']:+.1f} | {r['discrepancy_z']:.3f} | {r['q_times_err_z']:.3f} |\n"
        for r in rows)
    (rd / "summary.md").write_text(md)
    print(md)

    if args.tex:
        sel = [r for r in rows if r["q"] != "consist" and r["draft_noise"] == 0.0]
        qs = sorted({float(r["q"]) for r in sel})
        get = lambda ds, q: next((r for r in sel if r["dataset"] == ds and float(r["q"]) == q), None)
        lines = [r"\begin{tabular}{l" + "r" * len(qs) + "}", r"\toprule",
                 "leak fraction $q$ & " + " & ".join(f"{q:g}" for q in qs) + r" \\", r"\midrule"]
        for ds, name, fmt in (("ett", r"ETTh1 stream ($^\circ$C)", "{:.2f}"), ("air", "PM2.5 stream (log)", "{:.3f}")):
            for field, lab in (("stream", name), ("gap_pct", "\\quad gap stream$-$smooth (\\%)")):
                vals = []
                for q in qs:
                    r = get(ds, q)
                    vals.append("--" if r is None else (fmt.format(r[field]) if field == "stream" else f"{r[field]:+.1f}"))
                lines.append(lab + " & " + " & ".join(vals) + r" \\")
        cons = {r["dataset"]: r for r in rows if r["q"] == "consist"}
        lines += [r"\midrule", r"\multicolumn{" + str(len(qs) + 1) + r"}{l}{\emph{Annealed leak + L2 mode-consistency loss:} "
                  + ", ".join(f"{ {'ett': 'ETTh1', 'air': 'PM2.5'}[d]} {c['stream']:.3g} / {c['smooth']:.3g} ({c['gap_pct']:+.1f}\\%)"
                              for d, c in sorted(cons.items(), reverse=True)) + r"} \\",
                  r"\bottomrule", r"\end{tabular}"]
        Path(args.tex).write_text("\n".join(lines) + "\n")
        # consistency-baseline numbers quoted in the text
        mac = [f"\\newcommand{{\\Cons{ {'ett': 'Ett', 'air': 'Air'}[d]}}}{{{c['stream']:.{2 if d == 'ett' else 3}f}/{c['smooth']:.{2 if d == 'ett' else 3}f}}}"
               for d, c in sorted(cons.items())]
        Path(args.tex).with_name(Path(args.tex).stem + "_macros.tex").write_text("\n".join(mac) + "\n")

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(5.5, 1.5))
        cols = {0.0: "#86b6ef", 0.3: "#2a78d6", 0.6: "#104281"}  # one blue ramp: draft noise is a magnitude
        ett = [r for r in rows if r["dataset"] == "ett" and r["q"] != "consist"]
        for dn in sorted({r["draft_noise"] for r in ett}):
            rr = sorted((r for r in ett if r["draft_noise"] == dn), key=lambda r: float(r["q"]))
            base = rr[0]["stream"]
            ax[0].errorbar([float(r["q"]) for r in rr], [r["stream"] - base for r in rr], yerr=[r["stream_sd"] for r in rr],
                           marker="o", ms=3, lw=1.2, capsize=2, color=cols.get(dn, "k"), label=f"noise {dn:g}")
            ax[1].scatter([r["q_times_err_z"] for r in rr], [r["discrepancy_z"] for r in rr], s=14, color=cols.get(dn, "k"),
                          label=f"ETTh1, noise {dn:g}")
        air = sorted((r for r in rows if r["dataset"] == "air" and r["q"] != "consist"), key=lambda r: float(r["q"]))
        ax[1].scatter([r["q_times_err_z"] for r in air], [r["discrepancy_z"] for r in air], s=14, marker="^", color="#eb6834", label="PM2.5")
        ax[0].axhline(0, color="#bbb", lw=0.8)
        ax[0].set_xlabel("leak fraction $q$", fontsize=7.5)
        ax[0].set_ylabel("ETTh1 stream MAE\nminus $q{=}0$ (°C)", fontsize=7)
        ax[0].legend(fontsize=6.5, frameon=False, title="draft noise", title_fontsize=6.5, alignment="left")
        m = max(r["q_times_err_z"] for r in rows if r["q"] != "consist") * 1.05
        ax[1].plot([0, m], [0, m], color="#bbb", lw=0.8, ls="--")
        ax[1].set_xlabel(r"$q\,\mathbb{E}|y-\tilde y|$ ($z$ units)", fontsize=7.5)
        ax[1].set_ylabel("deployment minus\nleaked risk ($z$)", fontsize=7)
        ax[1].legend(fontsize=6.5, frameon=False, loc="upper left", handletextpad=0.2)
        ax[1].text(m * 1.0, m * 0.78, "$y=x$", fontsize=6.5, color="#9a9994", ha="right", va="top")
        for a in ax:
            a.tick_params(labelsize=6.5)
            a.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=0.3)
        fig.savefig(args.fig, bbox_inches="tight")
        fig.savefig(str(Path(args.fig).with_suffix(".png")), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
