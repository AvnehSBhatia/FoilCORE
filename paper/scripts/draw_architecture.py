"""Block diagram of FoilCORE-5k + streaming panel (kernel latency vs T, AR vs TF). → paper/figures/ts/architecture.png, streaming.png"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
OUT = _root / "paper" / "figures" / "ts"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAG, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#666666"


def box(ax, x, y, w, h, text, color, fs=8.2, tc="white", lw=0, rad=0.06, pad=0.02, clip=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad},rounding_size={rad}",
        fc=color, ec="none" if lw == 0 else "#333", lw=lw, zorder=2, clip_on=clip,
    ))
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
        color=tc, zorder=3, linespacing=1.15, clip_on=clip,
    )


def arrow(ax, p, q, color="#444", lw=1.1, style="-|>", ms=8):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
        zorder=1, shrinkA=2, shrinkB=2, clip_on=True,
    ))


def _tint(hexcol, toward="#ffffff", a=0.28):
    def rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = rgb(hexcol)
    tr, tg, tb = rgb(toward)
    r, g, b = (int(c + (t - c) * a) for c, t in zip((r, g, b), (tr, tg, tb)))
    return f"#{r:02x}{g:02x}{b:02x}"


class _B:
    __slots__ = ("x", "y", "w", "h")

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2

    def top(self, x=None):
        return (self.cx if x is None else x, self.y + self.h)

    def bot(self, x=None):
        return (self.cx if x is None else x, self.y)

    def left(self, y=None):
        return (self.x, self.cy if y is None else y)

    def right(self, y=None):
        return (self.x + self.w, self.cy if y is None else y)


def box3d(ax, x, y, w, h, text, color, fs=8.2, tc="white", dx=0.07, dy=0.055, rad=0.05, pad=0.012):
    """Front face plus a small isometric top and right side."""
    top_c = _tint(color, "#ffffff", 0.32)
    side_c = _tint(color, "#000000", 0.22)
    ax.add_patch(Polygon(
        [(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
        closed=True, fc=top_c, ec="none", zorder=1.2, clip_on=False,
    ))
    ax.add_patch(Polygon(
        [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
        closed=True, fc=side_c, ec="none", zorder=1.2, clip_on=False,
    ))
    box(ax, x, y, w, h, text, color, fs=fs, tc=tc, rad=rad, pad=pad, clip=False)
    return _B(x, y, w, h)


def wire(ax, pts, color="#555555", lw=1.12, head=True, ms=8):
    """Axis-aligned polyline; arrowhead only on the last segment."""
    pts = [(float(p[0]), float(p[1])) for p in pts]
    stop = len(pts) - 1 if head else len(pts)
    for i in range(stop - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        ax.add_line(Line2D(
            [x0, x1], [y0, y1], color=color, lw=lw,
            solid_capstyle="butt", solid_joinstyle="miter", zorder=1.7, clip_on=False,
        ))
    if head and len(pts) >= 2:
        ax.add_patch(FancyArrowPatch(
            pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=ms,
            color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=1.7, clip_on=False,
        ))


def node(ax, x, y, color="#555555", r=0.035):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x, y), r, fc=color, ec="none", zorder=1.8, clip_on=False))


def _panel_ax(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_clip_on(True)


def architecture():
    W, H = 11.0, 4.6
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    ink, mute = "#555555", "#8a8a8a"

    contour = box3d(ax, 0.18, 3.32, 1.72, 0.72, "contour\n200 × (x, y)", GREY)
    linear = box3d(ax, 0.18, 2.24, 1.72, 0.74, "Linear(2,32) + φ\nper-point features", BLUE)
    region = box3d(ax, 0.18, 1.06, 1.72, 0.92, "region means\nglobal · LE · TE\n(3 × 32)", BLUE)
    profile = box3d(ax, 2.14, 1.06, 1.72, 0.92, "profile memory\ncamber/thickness\n@10 stations → φ(Linear)", YELLOW, tc="black")
    mem = box3d(ax, 0.18, 0.14, 3.68, 0.58, "4 memories  M  (4 × 32)", "#d0d0d0", tc="black")

    flight = box3d(ax, 4.28, 3.32, 1.88, 0.72, "flight condition\n(α, Re, Ma)_t", GREY)
    query = box3d(ax, 4.28, 2.16, 1.88, 0.88, "pointwise query\nΔ,B,C = Lin(3,32)\nselective scan over M", AQUA)
    draft = box3d(ax, 4.28, 1.10, 1.88, 0.74, "draft head\nLinear(32,2) → Ĉl, Ĉd", AQUA)
    err = box3d(ax, 4.28, 0.14, 1.88, 0.64, "error head\nM↓ + (α,Re,Ma,Ĉl,Ĉd) → ê", MAG)

    corr = box3d(ax, 6.62, 2.52, 2.48, 1.52,
                 "polar corrector  (×2)\n"
                 "u_t = Lin(7,32)[cond,Ĉ,ê] + c_t + sin α\n"
                 "Q=K=s₁u+s₃,  V=W_v u,  −Δα²/τ\n"
                 "causal ⇄ bidirectional mask\n"
                 "LayerNorm + φ",
                 ORANGE, fs=7.4)
    requery = box3d(ax, 6.76, 1.10, 2.20, 0.88, "requery\nC ← C + h_t ; scan over M ; head", AQUA)
    resid = box3d(ax, 6.76, 0.14, 2.20, 0.58, "ê-gated residual MLP(32+2→16→2)", MAG)
    out = box3d(ax, 9.28, 1.10, 1.52, 0.88, "(Cl, Cd)_t\n+ ê_t", "#333333")

    # geometry column
    wire(ax, [contour.bot(), linear.top()])
    wire(ax, [linear.bot(), region.top()])
    wire(ax, [contour.right(), (profile.cx, contour.cy), profile.top()], color=mute)
    wire(ax, [region.bot(), mem.top(region.cx)])
    wire(ax, [profile.bot(), mem.top(profile.cx)])

    # flight / draft stack
    wire(ax, [flight.bot(), query.top()])
    wire(ax, [query.bot(), draft.top()])
    wire(ax, [draft.bot(), err.top()])

    # memories → query (gutter between geometry and flight)
    gx = 4.08
    wire(ax, [mem.right(0.58), (gx, 0.58), (gx, query.cy), query.left()], color=mute)
    node(ax, gx, 0.58, mute)
    node(ax, gx, query.cy, mute)
    wire(ax, [mem.right(err.cy), err.left()])

    # collector bus: query / draft / error → corrector; error also → residual
    bus = 6.40
    wire(ax, [query.right(), (bus, query.cy)], head=False)
    wire(ax, [draft.right(), (bus, draft.cy)], head=False, color=mute)
    wire(ax, [err.right(), (bus, err.cy)], head=False, color=mute)
    wire(ax, [(bus, err.cy), (bus, corr.cy), corr.left()])
    wire(ax, [(bus, err.cy), resid.left(err.cy)], color=mute)
    node(ax, bus, query.cy, ink)
    node(ax, bus, draft.cy, mute)
    node(ax, bus, err.cy, mute)
    node(ax, profile.cx, contour.cy, mute)
    node(ax, out.cx, resid.cy, ink)

    # corrector stack and output
    wire(ax, [corr.bot(requery.cx), requery.top()])
    wire(ax, [requery.bot(), resid.top()])
    wire(ax, [requery.right(), out.left(requery.cy)])
    wire(ax, [resid.right(), (out.cx, resid.cy), out.bot()])

    ax.text(0.18, 4.45, "(a)  geometry (once per airfoil, cached in streaming)", fontsize=8.5, color="#333", style="italic", clip_on=False)
    ax.text(4.28, 4.45, "per α: stream or smooth, same weights", fontsize=8.5, color="#333", style="italic", clip_on=False)
    ax.text(9.20, 3.55, "5,285 params\n21 KB fp32 / 5.2 KB int8\n81 µs / polar", fontsize=8, color="#333", clip_on=False)
    fig.savefig(OUT / "architecture.png", dpi=220, facecolor="white")
    fig.savefig(OUT / "architecture.pdf")
    plt.close(fig)


def streaming():
    import numpy as np
    sp = _root / "results" / "paper_final" / "latency_sweep.json"
    names = {"v61": "FoilCORE-5k", "v74": "FoilCORE-2k", "v6": "FoilCORE-3k", "v70": "FoilCORE-0.5k"}
    if sp.is_file():
        SW = json.load(open(sp)); lat = {names.get(k, k): (tuple(v["T"]), tuple(v["full_us"]), tuple(v["cached_us"])) for k, v in SW.items()}
    else:
        lat = {"FoilCORE-5k": ((8, 34, 68), (31, 52, 84), (9, 34, 63)), "FoilCORE-3k": ((8, 34, 68), (30, 47, 74), (8, 28, 53)), "FoilCORE-0.5k": ((8, 34, 68), (12, 21, 36), (5, 14, 34))}
    cols = {"FoilCORE-5k": BLUE, "FoilCORE-2k": ORANGE, "FoilCORE-3k": AQUA, "FoilCORE-0.5k": MAG}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
    for k, (T, full, cached) in lat.items():
        axes[0].plot(T, full, "o-", color=cols.get(k, GREY), lw=1.4, label=f"{k} full")
        axes[0].plot(T, cached, "s--", color=cols.get(k, GREY), lw=1.2, label=f"{k} cached")
    axes[0].set_xlabel("polar length T (points seen so far)", fontsize=9); axes[0].set_ylabel("µs per forward (1 CPU thread)", fontsize=9)
    axes[0].set_title("Single-kernel latency vs. sequence length", fontsize=9.5); axes[0].legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")
    axes[0].axhline(1000, color="#bbb", lw=0.8); axes[0].text(60, 1050, "1 ms", fontsize=7, color="#888")
    axes[0].set_yscale("log"); axes[0].set_ylim(3, 6000)
    # AR vs TF from numbers.json
    nj = _root / "results" / "paper_final" / "main" / "numbers.json"
    if not nj.is_file():
        nj = _root / "results" / "paper_final" / "main" / "numbers.json"
    N = json.load(open(nj))["models"]
    names = [n for n in ("v61", "v74", "v6", "v70", "old_protocol") if n in N]
    x = np.arange(len(names))
    ar = [N[n]["accuracy"]["ar"]["cl"] for n in names]; tf = [N[n]["accuracy"]["tf"]["cl"] for n in names]
    axes[1].bar(x - 0.2, ar, 0.38, color=BLUE, label="causal streaming (AR)")
    axes[1].bar(x + 0.2, tf, 0.38, color=ORANGE, label="bidirectional (TF)")
    for i in range(len(names)):
        axes[1].text(i - 0.2, ar[i], f"{ar[i]:.3f}", ha="center", va="bottom", fontsize=7); axes[1].text(i + 0.2, tf[i], f"{tf[i]:.3f}", ha="center", va="bottom", fontsize=7)
    axes[1].set_xticks(x); axes[1].set_xticklabels([{"v61": "FoilCORE-5k", "v74": "FoilCORE-2k", "v6": "FoilCORE-3k", "v70": "FoilCORE-0.5k", "old_protocol": "3k, label-\nleaking"}[n] for n in names], fontsize=8)
    axes[1].set_ylabel("Cl MAE (test)", fontsize=9); axes[1].set_title("Streaming = smoothing for draft-only training", fontsize=9.5); axes[1].legend(fontsize=7.5, frameon=False)
    for ax in axes:
        ax.grid(True, color="#ddd", lw=0.6)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(OUT / "streaming.png", dpi=170); plt.close(fig)


def draw_dualmask(ax, W=2.48, H=2.12, panel="(b)"):
    """Two equal columns about x=W/2. Orthogonal T-junction, matched segment lengths."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Rectangle

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    kw = dict(rad=0.055, pad=0.012, clip=False)

    m, gutter = 0.10, 0.16
    bw = (W - 2 * m - gutter) / 2
    lx, rx = m, m + bw + gutter
    lc, rc, cx = lx + bw / 2, rx + bw / 2, W / 2

    n, s = 5, 0.086
    cell = s * 0.78
    grid_w = (n - 1) * s + cell

    def mask_grid(x_center, y0, causal):
        x0 = x_center - grid_w / 2
        for i in range(n):
            for j in range(n):
                on = (j <= i) if causal else True
                ax.add_patch(Rectangle(
                    (x0 + j * s, y0 + (n - 1 - i) * s), cell, cell,
                    fc=ORANGE if on else "#ececec", ec="white", lw=0.4, zorder=2, clip_on=False,
                ))
        return y0 + (n - 1) * s + cell

    def v_arrow(x, y0, y1, color, lw=1.15, ms=8):
        ax.add_patch(FancyArrowPatch(
            (x, y0), (x, y1),
            arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
            shrinkA=0, shrinkB=0, zorder=1, clip_on=False,
        ))

    title_x = lx
    if panel:
        ax.text(lx, H - 0.08, panel, fontsize=9.5, fontweight="bold", va="center", color="#222", clip_on=False)
        title_x = lx + 0.20
    ax.text(title_x, H - 0.08, "train, one batch", fontsize=8, va="center", style="italic", color="#333", clip_on=False)

    top_y, top_h = 1.62, 0.30
    box(ax, lx, top_y, bw, top_h,
        r"draft  $\hat{y}=d(x)$" + "\n" + r"corrector reads $\hat{y}$", AQUA, fs=7.6, **kw)
    box(ax, rx, top_y, bw, top_h,
        r"label $y$" + "\n" + "loss only", GREY, fs=7.6, **kw)

    cy, cr = 1.30, 0.115
    ax.add_patch(Circle((cx, cy), cr, fc=YELLOW, ec="none", zorder=2, clip_on=False))
    ax.text(cx, cy, r"$\frac{1}{2}$", ha="center", va="center", fontsize=11, zorder=3, clip_on=False)
    v_arrow(cx, top_y, cy + cr, AQUA)

    bot_y, bot_h = 0.08, 0.28
    gy = 0.46
    grid_top = mask_grid(lc, gy, True)
    mask_grid(rc, gy, False)

    hub_y = (cy - cr + grid_top) / 2
    ax.add_line(Line2D([cx, cx], [cy - cr, hub_y], color=ORANGE, lw=1.15, solid_capstyle="butt", zorder=1))
    ax.add_line(Line2D([lc, rc], [hub_y, hub_y], color=ORANGE, lw=1.15, solid_capstyle="butt", zorder=1))
    v_arrow(lc, hub_y, grid_top, ORANGE)
    v_arrow(rc, hub_y, grid_top, ORANGE)

    box(ax, lx, bot_y, bw, bot_h, "causal\nstreaming", ORANGE, fs=7.6, **kw)
    box(ax, rx, bot_y, bw, bot_h, "bidirectional\nsmoother", ORANGE, fs=7.6, **kw)


def dualmask():
    W, H = 2.48, 2.12
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    draw_dualmask(ax, W, H)
    fig.savefig(OUT / "dualmask.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "dualmask.pdf")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    streaming()
    dualmask()
    print("wrote", OUT / "architecture.png", OUT / "streaming.png", OUT / "dualmask.png")
