"""results/<stamp>/ → paper/tables/ts_*.tex (booktabs). Re-run after every number refresh.

python tools/export_latex_tables.py --results results/paper_final --out paper/tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NAMES = {"v61": r"FoilCORE-5k (ours)", "v74": r"FoilCORE-2k ($d{=}16$)", "v6": r"FoilCORE-3k (plain)", "v70": r"FoilCORE-0.5k ($d{=}8$)",
         "old_protocol": r"3k, label-leaking"}
NF_PARAMS = {"xxsmall": 13302, "xxxlarge": 1428166}


def fmt(x, nd):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="paper/tables")
    args = ap.parse_args()
    R = Path(args.results); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    N = json.load(open(R / "main" / "numbers.json"))
    models = N["models"]
    sweep_p = R / "latency_sweep.json"
    if sweep_p.is_file():  # canonical kernel numbers: the T-sweep (same run for figure, table and text)
        SW = json.load(open(sweep_p))
        for k, v in SW.items():
            if k in models:
                i = v["T"].index(68) if 68 in v["T"] else -1
                models[k]["latency"]["fused_cpp_b1_us"] = {**models[k]["latency"].get("fused_cpp_b1_us", {}), "full_median": v["full_us"][i], "geom_cached_median": v["cached_us"][i],
                                                            "t8_cached": v["cached_us"][0], "t8_full": v["full_us"][0]}
    order = [k for k in ("v61", "v74", "v6", "v70", "old_protocol") if k in models]

    # ---- main table ----
    L = [r"\begin{tabular}{lrrrrrrl}", r"\toprule",
         r"Model & Params & $C_l$ & $C_d$ (stall) & smooth.\ $C_l$/$C_d$ & MACs/pt & latency & framework \\", r"\midrule"]
    for k in order:
        e = models[k]; a = e["accuracy"]; l = e["latency"]; fz = l.get("fused_cpp_b1_us", {})
        lat = f"{fz['full_median']:.0f}\\,$\\mu$s" if "full_median" in fz else f"{l['cpu_b1_1thread_ms']['median']:.2f}\\,ms"
        macs = {"v61": "12.8k", "v6": "11.8k", "v70": "2.6k", "old_protocol": "11.8k", "v74": "6.1k"}.get(k, "--")
        L.append(f"{NAMES.get(k,k)} & {e['n_params']:,} & {fmt(a['ar']['cl'],3)} & {fmt(a['ar']['cd'],4)} ({fmt(a['ar']['stall_cd'],4)}) & {fmt(a['tf']['cl'],3)} / {fmt(a['tf']['cd'],4)} & {macs} & {lat} & C++ kernel \\\\")
    sp = R / "seed_stats.json"
    if sp.is_file():
        SS = json.load(open(sp))
        seed_row = f"\\quad 3 seeds & & {fmt(SS['cl']['mean'],3)} $\\pm$ {fmt(SS['cl']['std'],3)} & {fmt(SS['cd']['mean'],4)} $\\pm$ {fmt(SS['cd']['std'],4)} & & & & \\\\"
        pass  # seed spread is quoted in the text; row: seed_row
    # ---- standard sequence backbones on the same front-end (results/<stamp>/baselines/numbers.json) ----
    bp = R / "baselines" / "numbers.json"
    BASE_NAMES = {"v80": r"\quad Transformer", "v81": r"\quad TSMixer",
                  "v82": r"\quad Mamba (S6)"}
    if bp.is_file():
        BM = json.load(open(bp))["models"]
        if any(k in BM for k in BASE_NAMES):
            L.append(r"\midrule")
            L.append(r"\multicolumn{8}{l}{\emph{Same front-end and protocol, standard backbone as corrector:}} \\")
        for k, nm in BASE_NAMES.items():
            if k not in BM:
                continue
            e = BM[k]; a = e["accuracy"]; l = e["latency"]
            fz = l.get("fused_cpp_b1_us", {})
            bmacs = {"v80": "8.1k", "v81": "2.4k$^{\\ast}$", "v82": "3.7k$^{\\ast}$"}.get(k, "--")
            if "full_median" in fz:
                lat, fw = f"{fz['full_median']:.0f}\\,$\\mu$s", "C++ kernel"
            else:
                lat, fw = f"{l['cpu_b1_1thread_ms']['median']:.2f}\\,ms", "eager PyTorch"
            L.append(f"{nm} & {e['n_params']:,} & {fmt(a['ar']['cl'],3)} & {fmt(a['ar']['cd'],4)} ({fmt(a['ar']['stall_cd'],4)}) & {fmt(a['tf']['cl'],3)} / {fmt(a['tf']['cd'],4)} & {bmacs} & {lat} & {fw} \\\\")
    L.append(r"\midrule")
    nfl = {}
    p = R / "figures" / "nf_latency.json"
    if p.is_file():
        nfl = json.load(open(p)).get("neuralfoil_b1_latency_ms", {})
    for sz in ("xxsmall", "xxxlarge"):
        v = N.get("neuralfoil_test", {}).get(sz)
        if v:
            lat = f"{nfl[sz]:.2f}\\,ms" if sz in nfl else "--"
            macs = {"xxsmall": "26k", "xxxlarge": "2.9M"}[sz]
            L.append(f"NeuralFoil {sz} & {NF_PARAMS[sz]:,} & {fmt(v['cl'],3)} & {fmt(v['cd'],4)} ({fmt(v['stall_cd'],4)}) & -- & {macs} & {lat} & NumPy \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "ts_main.tex").write_text("\n".join(L) + "\n")

    # ---- NeuralFoil by regime ----
    rp = R / "main" / "neuralfoil_by_regime.json"
    if rp.is_file():
        reg = json.load(open(rp))
        L = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Regime & pts & FoilCORE-5k & NF xxsmall & NF xxxlarge \\", r"\midrule"]
        pretty = {"all": "all", "Mach [-0.01,0.05)": r"Ma $<0.05$", "Mach [0.05,0.15)": r"Ma $[0.05,0.15)$", "Mach [0.15,0.30)": r"Ma $[0.15,0.30)$", "Mach [0.30,0.51)": r"Ma $[0.30,0.50]$",
                  "Re [4e+04,1e+05)": r"Re $<10^5$", "Re [1e+05,3e+05)": r"Re $[10^5,3{\times}10^5)$", "Re [3e+05,1e+06)": r"Re $[3{\times}10^5,10^6)$", "Re [1e+06,3e+06)": r"Re $[10^6,3{\times}10^6)$", "Re [3e+06,2e+07)": r"Re $\ge 3{\times}10^6$",
                  "Mach<0.05 & Re in [1e5,3e6)": r"Ma $<0.05$, Re $[10^5,3{\times}10^6)$ (NF regime)", "|alpha|>=12": r"stall $|\alpha|\ge 12^\circ$", "|alpha|<12": r"pre-stall $|\alpha|<12^\circ$"}
        for k, v in reg.items():
            L.append(f"{pretty.get(k, k)} & {v['n']//1000}k & {fmt(v['v61'][0],3)} / {fmt(v['v61'][1],4)} & {fmt(v['xxsmall'][0],3)} / {fmt(v['xxsmall'][1],4)} & {fmt(v['xxxlarge'][0],3)} / {fmt(v['xxxlarge'][1],4)} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (out / "ts_regime.tex").write_text("\n".join(L) + "\n")

    # ---- knockouts (v61) ----
    KO_PRETTY = {"block0.tau_off": r"$\alpha$-locality $\tau$ off (block 0)", "block1.tau_off": r"$\alpha$-locality $\tau$ off (block 1)",
                 "block0.no_attn": "no attention (block 0)", "block1.no_attn": "no attention (block 1)", "all_blocks_no_attn": "no attention (both blocks)",
                 "block0.Wv_identity": r"$W_v{=}I$ (block 0)", "block1.Wv_identity": r"$W_v{=}I$ (block 1)",
                 "block0.no_layernorm": "no LayerNorm (block 0)", "block1.no_layernorm": "no LayerNorm (block 1)",
                 "drop_memory_global": "drop global memory", "drop_memory_le": "drop LE memory", "drop_memory_te": "drop TE memory",
                 "drop_memory_profile": "drop profile memory", "pathnl_identity": r"$\phi$ = identity", "err_head_zero": "error head zeroed",
                 "no_expert": "no residual head"}
    pk = lambda kk: KO_PRETTY.get(kk, kk.replace("_", r"\_"))
    for k in ("v61", "v6"):
        ko = models.get(k, {}).get("knockouts_val2048")
        if not ko:
            continue
        b = ko["baseline_causal"]
        rows = [(kk, v) for kk, v in ko.items() if not kk.startswith("baseline") and not kk.endswith("_knockouts")]
        rows.sort(key=lambda r: r[1]["cl"])
        L = [r"\begin{tabular}{lrr}", r"\toprule", r"Component removed & $\Delta C_l$ MAE & $\Delta C_d$ MAE \\", r"\midrule"]
        for kk, v in rows:
            L.append(f"{pk(kk)} & {v['cl']-b['cl']:+.3f} & {v['cd']-b['cd']:+.4f} \\\\")
        xp = R / "v61_extras" / "v61_extras.json"
        if k == "v61" and xp.is_file():
            KX = json.load(open(xp))["knockouts_test_causal"]; bx = KX["baseline"]
            L.append(r"\midrule")
            for kk in ("drop_memory_profile", "err_head_zero", "no_expert"):
                if kk in KX:
                    L.append(f"{pk(kk)}$^\\dagger$ & {KX[kk][0]-bx[0]:+.3f} & {KX[kk][1]-bx[1]:+.4f} \\\\")
            (out / "ts_ko_macros.tex").write_text(f"\\newcommand{{\\KoBaseCl}}{{{fmt(b['cl'],3)}}}\n\\newcommand{{\\KoBaseCd}}{{{fmt(b['cd'],4)}}}\n\\newcommand{{\\KoTestBaseCl}}{{{fmt(bx[0],3)}}}\n\\newcommand{{\\KoTestBaseCd}}{{{fmt(bx[1],4)}}}\n")
        L += [r"\bottomrule", r"\end{tabular}"]
        (out / f"ts_knockouts_{k}.tex").write_text("\n".join(L) + "\n")

    # ---- quantization ----
    qp = R / "quantization.json"
    if qp.is_file():
        Q = json.load(open(qp))
        L = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Model & fp32 (21\,KB for 5k) & fp16 & int8 matrices + fp16 rest (bytes) & int8 all/channel & int6 \\", r"\midrule"]
        for k, v in Q.items():
            m8 = v.get("int8_matrices_fp16_rest", v["int8_per_channel"]); b8 = v.get("bytes_int8_matrices_fp16_rest", v["bytes_int8"])
            L.append(f"{NAMES.get(k,k)} & {fmt(v['fp32']['cl'],3)}/{fmt(v['fp32']['cd'],5)} & {fmt(v['fp16']['cl'],3)}/{fmt(v['fp16']['cd'],5)} & {fmt(m8['cl'],3)}/{fmt(m8['cd'],5)} ({b8:,}) & {fmt(v['int8_per_channel']['cl'],3)}/{fmt(v['int8_per_channel']['cd'],5)} & {fmt(v['int6_per_channel']['cl'],3)}/{fmt(v['int6_per_channel']['cd'],5)} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (out / "ts_quant.tex").write_text("\n".join(L) + "\n")

    # ---- ablations from ablations.md (variant, description, params, epochs, best AR Cl, Cd) ----
    ap_ = R / "ablations.md"
    if ap_.is_file():
        L = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Variant & Params & Epochs & $C_l$ MAE & $C_d$ MAE \\", r"\midrule"]
        for line in ap_.read_text().splitlines()[2:]:
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) < 8 or "aborted" in c[-1]:
                continue
            L.append(f"{c[1].replace('&', r'\&').replace('_', r'\_')} & {c[2]} & {c[3]} & {c[4]} & {c[5]} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (out / "ts_ablations.tex").write_text("\n".join(L) + "\n")

    # ---- regime breakdown for v61 ----
    a = models[order[0]]["accuracy"]
    L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Slice & pts & $C_l$ MAE & $C_d$ MAE \\", r"\midrule"]
    for key, lab in (("ar_by_alpha", r"$\alpha$"), ("ar_by_re", "Re"), ("ar_by_mach", "Mach")):
        for r in a[key]:
            lo = f"{r['lo']:.0e}" if key == "ar_by_re" else f"{r['lo']:g}"; hi = f"{r['hi']:.0e}" if key == "ar_by_re" else f"{r['hi']:g}"
            L.append(f"{lab} $[{lo}, {hi})$ & {r['n']//1000}k & {fmt(r['cl'],3)} & {fmt(r['cd'],4)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "ts_regimes.tex").write_text("\n".join(L) + "\n")
    # ---- label efficiency ----
    lp = R / "label_efficiency" / "numbers.json"
    if lp.is_file():
        LE = json.load(open(lp))["models"]
        L = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Train.\ geometries & Polars & $C_l$ MAE & $C_d$ MAE & stall $C_d$ \\", r"\midrule"]
        rowsLE = [("v61_frac10", "10\% (431)", "27k"), ("v61_frac30", "30\% (1.3k)", "82k")]
        for k, lab, npol in rowsLE:
            if k in LE:
                a = LE[k]["accuracy"]["ar"]
                L.append(f"{lab} & {npol} & {fmt(a['cl'],3)} & {fmt(a['cd'],4)} & {fmt(a['stall_cd'],4)} \\\\")
        a = models[order[0]]["accuracy"]["ar"]
        L.append(f"100\% (4,313) & 273k & {fmt(a['cl'],3)} & {fmt(a['cd'],4)} & {fmt(a['stall_cd'],4)} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (out / "ts_labeleff.tex").write_text("\n".join(L) + "\n")
    # ---- transfer of the training recipe (results/transfer/summary.json) ----
    tp = R.parent / "transfer" / "summary.json"
    if not tp.is_file():
        tp = Path("results/transfer/summary.json")
    if tp.is_file():
        TS = json.load(open(tp))
        dsets = [("ett", r"ETTh1 oil temp.\ ($^\circ$C)"), ("air", "Beijing PM2.5 (log)"), ("ecg", "MIT-BIH lead II (mV)")]
        protos = [("gtmix", "label-leaking (sched.\\ sampling)"), ("draft_causal", "draft-only, causal"),
                  ("draft_bidir", "draft-only, bidirectional"), ("draft_dual", "draft-only, dual mask (ours)")]
        have = [(k, n) for k, n in dsets if any(f"{k}/{p}" in TS for p, _ in protos)]
        if have:
            L = [r"\begin{tabular}{l" + "rr" * len(have) + "}", r"\toprule",
                 "Training protocol & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n in have) + r" \\"]
            L.append(" & " + " & ".join("stream & smooth" for _ in have) + r" \\")
            L.append(r"\midrule")
            for p, pn in protos:
                cells = []
                for k, _ in have:
                    v = TS.get(f"{k}/{p}")
                    if v is None:
                        cells += ["--", "--"]; continue
                    nd = 3 if v["causal_mean"] < 1 else 2
                    cells += [f"{v['causal_mean']:.{nd}f}", f"{v['bidir_mean']:.{nd}f}"]
                L.append(f"{pn} & " + " & ".join(cells) + r" \\")
            # same protocols with a different corrector architecture (recipe is architecture-agnostic?)
            for corr, cn in (("transformer", "transformer corrector"),):  # Mamba-corrector runs: results/transfer (ECG incomplete)
                rows_c = []
                for p, pn in (("gtmix", "label-leaking"), ("draft_bidir", "draft-only, bidirectional"), ("draft_dual", "draft-only, dual mask")):
                    cells = []
                    ok = False
                    for k, _ in have:
                        v = TS.get(f"{k}/{p}/{corr}")
                        if v is None:
                            cells += ["--", "--"]; continue
                        ok = True
                        nd = 3 if v["causal_mean"] < 1 else 2
                        cells += [f"{v['causal_mean']:.{nd}f}", f"{v['bidir_mean']:.{nd}f}"]
                    if ok:
                        rows_c.append(f"\\quad {pn} & " + " & ".join(cells) + r" \\")
                if rows_c:
                    L.append(r"\midrule")
                    L.append(f"\\multicolumn{{{1 + 2 * len(have)}}}{{l}}{{\\emph{{Same, with a {cn} in place of ours:}}}} \\\\")
                    L += rows_c
            L += [r"\bottomrule", r"\end{tabular}"]
            (out / "ts_transfer.tex").write_text("\n".join(L) + "\n")

    # ---- key-number macros ----
    def m(name, val): return f"\\newcommand{{\\{name}}}{{{val}}}"
    e = models[order[0]]; a = e["accuracy"]; fz = e["latency"].get("fused_cpp_b1_us", {})
    macros = [m("NParamsMain", f"{e['n_params']:,}"), m("MainClAR", fmt(a['ar']['cl'],3)), m("MainCdAR", fmt(a['ar']['cd'],4)),
              m("MainStallCd", fmt(a['ar']['stall_cd'],4)), m("MainClTF", fmt(a['tf']['cl'],3)), m("MainCdTF", fmt(a['tf']['cd'],4)),
              m("MainFusedUs", f"{fz.get('full_median', 0):.0f}"), m("MainFusedCachedUs", f"{fz.get('geom_cached_median', 0):.0f}"),
              m("MainEagerMs", fmt(e['latency']['cpu_b1_1thread_ms']['median'],2)),
              m("MainMpsUs", f"{e['latency'].get('mps_batched_us_per_polar', {}).get('us', 0):.0f}"),
              m("MainDraftCl", fmt(a['draft']['cl'],3)), m("MainStreamTeightUs", f"{fz.get('t8_cached', 0):.0f}")]
    if "v70" in models:
        e7 = models["v70"]; a7 = e7["accuracy"]; f7 = e7["latency"].get("fused_cpp_b1_us", {})
        macros += [m("TinyParams", f"{e7['n_params']:,}"), m("TinyClAR", fmt(a7['ar']['cl'],3)), m("TinyCdAR", fmt(a7['ar']['cd'],4)), m("TinyFusedUs", f"{f7.get('full_median',0):.0f}"), m("TinyFusedCachedUs", f"{f7.get('geom_cached_median',0):.0f}")]
    if "v6" in models:
        e6 = models["v6"]; a6 = e6["accuracy"]
        macros += [m("PlainClAR", fmt(a6['ar']['cl'],3)), m("PlainCdAR", fmt(a6['ar']['cd'],4))]
    if "old_protocol" in models:
        eo = models["old_protocol"]; ao = eo["accuracy"]
        macros += [m("OldClAR", fmt(ao['ar']['cl'],3)), m("OldCdAR", fmt(ao['ar']['cd'],4)), m("OldClTF", fmt(ao['tf']['cl'],3)), m("OldCdTF", fmt(ao['tf']['cd'],4))]
    if "v74" in models:
        e4 = models["v74"]; a4 = e4["accuracy"]; f4 = e4["latency"].get("fused_cpp_b1_us", {})
        macros += [m("MidParams", f"{e4['n_params']:,}"), m("MidClAR", fmt(a4['ar']['cl'],3)), m("MidCdAR", fmt(a4['ar']['cd'],4)), m("MidFusedUs", f"{f4.get('full_median',0):.0f}")]
    nf = N.get("neuralfoil_test", {})
    for sz, tag in (("xxsmall", "NFxxs"), ("xxxlarge", "NFxxxl")):
        if sz in nf:
            macros += [m(tag + "Cl", fmt(nf[sz]['cl'],3)), m(tag + "Cd", fmt(nf[sz]['cd'],4)), m(tag + "StallCd", fmt(nf[sz]['stall_cd'],4))]
            if sz in nfl: macros.append(m(tag + "LatMs", fmt(nfl[sz],2)))
    sp = R / "seed_stats.json"
    if sp.is_file():
        SS = json.load(open(sp))
        macros += [m("SeedClMean", fmt(SS['cl']['mean'],3)), m("SeedClStd", fmt(SS['cl']['std'],3)), m("SeedCdMean", fmt(SS['cd']['mean'],4)), m("SeedCdStd", fmt(SS['cd']['std'],4)),
                   m("SeedStallCdMean", fmt(SS['stall_cd']['mean'],4)), m("SeedStallCdStd", fmt(SS['stall_cd']['std'],4)), m("SeedN", str(len(SS['cl']['values'])))]
    ep_ = R / "energy_gpu.json"
    if ep_.is_file():
        EG = json.load(open(ep_))
        for k, tag in (("v61", "Main"), ("v74", "Mid"), ("v70", "Tiny"), ("v6", "Plain")):
            if k in EG:
                macros += [m(tag + "GpuMJ", fmt(EG[k]['uJ_per_polar_dynamic'] / 1000, 1)), m(tag + "GpuPolarsPerS", f"{EG[k]['polars_per_s']/1000:.0f}k")]
    (out / "ts_numbers.tex").write_text("\n".join(macros) + "\n")
    print("wrote", sorted(p.name for p in out.glob("ts_*.tex")))


if __name__ == "__main__":
    main()
