"""Figures and the appendix protocol table for the FMTS camera-ready.

  modes.pdf       Figure 1: what each mode may read, and what the corrector reads in training
  protocols.pdf   Figure 2: stream vs smooth MAE per training protocol (results/transfer/summary.json)
  ts_transfer_sd.tex  appendix table: the same cells with seed standard deviations

python paper/scripts/fmts_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 7
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures" / "ts"
TAB = ROOT / "paper" / "tables"
# blue = filter / causal mask, orange = smoother / full mask (validated pair); aqua = draft
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
INK, INK2, MUTE, LIGHT = "#0b0b0b", "#52514e", "#9a9994", "#e9e8e4"


def _box(ax, x, y, w, h, text, fc, tc="white", fs=7, ec="none", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=fc, ec=ec, lw=0.8, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3, linespacing=1.25)


def _arrow(ax, p, q, color=INK2, ls="-", lw=1.0, ms=7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 shrinkA=1, shrinkB=1, zorder=1))


DARK = "#3b3a37"  # true-target cells


def _title(fig, x, y, letter, text):
    fig.text(x, y, f"$\\bf{{({letter})}}$  {text}", ha="left", va="center", fontsize=7.5, color=INK)


def modes():
    """Figure 1. (a) a real ETTh1 load window: what the filter and the smoother may read when estimating hour t.
    (b) what the corrector reads, cell by cell: scheduled sampling swaps in true targets, ours never does.
    (c) the two attention masks, one drawn per batch."""
    import numpy as np
    import pandas as pd

    df = pd.read_csv(ROOT / "experiments" / "transfer" / "data" / "ETTh1.csv")
    start, T, t = 24 * 7 * 60 + 6, 72, 46  # a 72-hour window with clear daily load cycles
    load = df["HUFL"].to_numpy()[start:start + T]
    hrs = np.arange(T)

    fig = plt.figure(figsize=(5.5, 1.45))
    ty = 0.93
    # (a): two stacked strips of the load input
    _title(fig, 0.005, ty, "a", "Input load on one ETTh1 window")
    lo, hi = load.min() - 0.5, load.max() + 0.5
    top = hi + 0.62 * (hi - lo)
    strips = ((0.50, BLUE, t, "filter reads hours up to $t$"), (0.15, ORANGE, T - 1, "smoother reads the whole window"))
    axs = []
    for y0, col, upto, lab in strips:
        ax = fig.add_axes([0.012, y0, 0.375, 0.30])
        ax.axvspan(-0.5, upto + 0.5, ymax=(hi - lo) / (top - lo), color=col, alpha=0.10, lw=0, zorder=0)
        ax.plot(hrs[: upto + 1], load[: upto + 1], color=col, lw=1.2, zorder=2, solid_capstyle="round")
        if upto < T - 1:
            ax.plot(hrs[upto:], load[upto:], color=MUTE, lw=0.9, ls=(0, (2, 1.6)), zorder=1)
            ax.text(T - 1, top, "not yet arrived", ha="right", va="top", fontsize=6.2, color=INK2)
        ax.plot([t, t], [lo, hi], color=INK, lw=0.8, zorder=3)
        ax.text(0, top, lab, ha="left", va="top", fontsize=6.6, color=INK)
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(lo, top)
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(MUTE)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.tick_params(axis="x", length=2, width=0.5, labelsize=6.3, colors=INK2, pad=1.5)
        axs.append(ax)
    axs[0].set_xticks([])
    axs[1].set_xticks([0, 24, t, T - 1])
    axs[1].set_xticklabels(["0 h", "24 h", "$t$", "72 h"])

    # (b): what the corrector reads, cell by cell
    _title(fig, 0.425, ty, "b", "What the corrector reads")
    axb = fig.add_axes([0.425, 0.02, 0.31, 0.82])
    axb.set_xlim(0, 10)
    axb.set_ylim(0, 6)
    axb.set_axis_off()
    n, cw, x0, ch = 8, 0.85, 3.2, 0.95
    leak = {1, 4, 5}
    rows = ((4.35, "scheduled\nsampling", leak), (2.75, "ours", set()))
    for yy, lab, lk in rows:
        axb.text(x0 - 0.3, yy + ch / 2, lab, ha="right", va="center", fontsize=6.6, color=INK, linespacing=1.0)
        for k in range(n):
            is_y = k in lk
            axb.add_patch(Rectangle((x0 + k * cw, yy), cw * 0.86, ch, fc=DARK if is_y else AQUA, ec="none"))
            axb.text(x0 + k * cw + cw * 0.43, yy + ch / 2, "$y$" if is_y else r"$\hat y$", ha="center", va="center",
                     fontsize=6.3, color="white")
    for yy, fc, txt in ((1.55, AQUA, "the model's own draft"), (0.6, DARK, "true target, swapped in with prob. $q$")):
        axb.add_patch(Rectangle((0.15, yy - 0.3), cw * 0.86, 0.6, fc=fc, ec="none"))
        axb.text(0.15 + cw * 0.43, yy, "$y$" if fc == DARK else r"$\hat y$", ha="center", va="center", fontsize=5.8, color="white")
        axb.text(0.15 + cw + 0.15, yy, txt, ha="left", va="center", fontsize=6.3, color=INK2)

    # (c): the two attention masks
    _title(fig, 0.765, ty, "c", "Mask per batch")
    axc = fig.add_axes([0.765, 0.02, 0.23, 0.82])
    axc.set_xlim(0, 10)
    axc.set_ylim(0, 6)
    axc.set_axis_off()
    g, gs, gy0 = 6, 0.6, 2.25
    for gx0, causal, col, lab in ((0.3, True, BLUE, "causal\n(filter)"), (6.0, False, ORANGE, "full\n(smoother)")):
        for r in range(g):
            for c in range(g):
                on = c <= r if causal else True
                axc.add_patch(Rectangle((gx0 + c * gs, gy0 + (g - 1 - r) * gs), gs * 0.84, gs * 0.84,
                                        fc=col if on else "#ecebe7", ec="none"))
        axc.text(gx0 + g * gs / 2, gy0 - 0.3, lab, ha="center", va="top", fontsize=6.6, color=INK, linespacing=1.0)
    axc.text(5.15, gy0 + g * gs / 2, "or", ha="center", va="center", fontsize=6.6, color=INK2)
    axc.text(5.15, 0.35, "drawn at random, ½ each", ha="center", va="center", fontsize=6.3, color=INK2)

    fig.savefig(FIG / "modes.pdf")
    fig.savefig(FIG / "modes.png", dpi=300, facecolor="white")
    plt.close(fig)


PROTOS = (("gtmix", "label-leaking, full mask"), ("draft_causal", "draft-only, causal mask"),
          ("draft_bidir", "draft-only, full mask"), ("draft_dual", "draft-only, both (ours)"))
TASKS = (("ett", "ETTh1 MAE (°C)"), ("air", "PM2.5 MAE (log units)"), ("ecg", "MIT-BIH MAE (mV)"))
TICKS = {"ett": [4, 5, 6], "air": [0.5, 0.6, 0.7], "ecg": [0.22, 0.26, 0.30]}
CORRS = (("rpa", "our corrector"), ("transformer", "transformer corrector"))


def _cell(s, ds, proto, corr):
    return s[f"{ds}/{proto}" if corr == "rpa" else f"{ds}/{proto}/{corr}"]


def protocols():
    s = json.loads((ROOT / "results" / "transfer" / "summary.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.62), sharey=True, gridspec_kw={"wspace": 0.08})
    ys, labels, y = [], [], 0.0
    layout = []
    for corr, cname in CORRS:
        layout.append(("group", cname, y))
        y -= 0.85
        for proto, pname in PROTOS:
            layout.append(("row", (corr, proto), y))
            ys.append(y)
            labels.append(pname)
            y -= 1.0
        y -= 0.25
    for ax, (ds, tname) in zip(axes, TASKS):
        for kind, what, yy in layout:
            if kind == "group":
                continue
            corr, proto = what
            c = _cell(s, ds, proto, corr)
            if proto == "draft_dual":
                ax.axhspan(yy - 0.45, yy + 0.45, color="#f1f0ec", zorder=0, lw=0)
            ax.plot([c["causal_mean"], c["bidir_mean"]], [yy, yy], color=MUTE, lw=1.6, zorder=1, solid_capstyle="round")
            ax.errorbar(c["causal_mean"], yy + 0.13, xerr=c["causal_std"], fmt="none", ecolor=BLUE, elinewidth=0.6, alpha=0.55, zorder=1)
            ax.errorbar(c["bidir_mean"], yy - 0.13, xerr=c["bidir_std"], fmt="none", ecolor=ORANGE, elinewidth=0.6, alpha=0.55, zorder=1)
            ax.scatter(c["bidir_mean"], yy, s=22, marker="D", color=ORANGE, edgecolor="white", linewidth=0.6, zorder=3)
            ax.scatter(c["causal_mean"], yy, s=26, marker="o", color=BLUE, edgecolor="white", linewidth=0.6, zorder=4)
        ax.set_title(tname, fontsize=7, color=INK, pad=3)
        ax.tick_params(axis="x", labelsize=6.5, colors=INK2, length=2, width=0.5)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#ecebe7", lw=0.5, zorder=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(MUTE)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.set_xticks(TICKS[ds])
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels, fontsize=6.5, color=INK)
    for kind, what, yy in layout:
        if kind == "group":
            axes[0].text(-0.02, yy, what, transform=axes[0].get_yaxis_transform(), ha="right", va="center",
                         fontsize=6.5, style="italic", color=INK2)
    axes[0].set_ylim(y + 0.6, 0.4)
    h1 = axes[0].scatter([], [], s=26, marker="o", color=BLUE, label="filter (stream)")
    h2 = axes[0].scatter([], [], s=22, marker="D", color=ORANGE, label="smoother (smooth)")
    fig.legend(handles=[h1, h2], loc="lower center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.62, -0.02),
               handletextpad=0.2, columnspacing=1.2)
    fig.subplots_adjust(left=0.235, right=0.99, top=0.9, bottom=0.2)
    fig.savefig(FIG / "protocols.pdf")
    fig.savefig(FIG / "protocols.png", dpi=300, facecolor="white")
    plt.close(fig)


def table():
    s = json.loads((ROOT / "results" / "transfer" / "summary.json").read_text())
    fmt = {"ett": "{:.2f}", "air": "{:.3f}", "ecg": "{:.3f}"}
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Training protocol & \multicolumn{2}{c}{ETTh1 oil temp.\ ($^\circ$C)} & \multicolumn{2}{c}{Beijing PM2.5 (log)} & \multicolumn{2}{c}{MIT-BIH lead II (mV)} \\",
             r" & stream & smooth & stream & smooth & stream & smooth \\", r"\midrule"]
    names = {"gtmix": "label-leaking (sched.\\ sampling)", "draft_causal": "draft-only, causal",
             "draft_bidir": "draft-only, bidirectional", "draft_dual": "draft-only, dual mask (ours)"}
    for corr, cname in CORRS:
        if corr != "rpa":
            lines += [r"\midrule", r"\multicolumn{7}{l}{\emph{Same, with a transformer corrector in place of ours:}} \\"]
        for proto, _ in PROTOS:
            cells = []
            for ds, _ in TASKS:
                c = _cell(s, ds, proto, corr)
                f = fmt[ds]
                cells += [f"{f.format(c['causal_mean'])}\\,{{\\scriptsize$\\pm${f.format(c['causal_std'])}}}",
                          f"{f.format(c['bidir_mean'])}\\,{{\\scriptsize$\\pm${f.format(c['bidir_std'])}}}"]
            lab = names[proto] if corr == "rpa" else "\\quad " + names[proto].replace(" (ours)", "").replace(" (sched.\\ sampling)", "")
            lines.append(lab + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "ts_transfer_sd.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    modes()
    protocols()
    table()
    print("wrote", FIG / "modes.pdf", FIG / "protocols.pdf", TAB / "ts_transfer_sd.tex")
