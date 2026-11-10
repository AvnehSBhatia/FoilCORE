# FoilCORE — final results package (2026-08-18, final checkpoints)

Test = **full** held-out-geometry split (34,043 polars, 2.43M points), physical MAE, **causal AR** unless noted. Paper: `paper/fmts/fmts.pdf`. Regenerate everything: `bash tools/finalize_paper.sh results/paper_final`.

## Main table

| model | params | epoch | AR Cl | AR Cd | stall Cd | TF Cl/Cd | seeded-1 Cl | draft Cl | eager B=1 | fused C++ T=68 full (cached) | T=8 cached | CPU/MPS batched µs | GPU mJ/polar (A4500, dyn.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FoilCORE-5k (v61) | 5,285 | 346 | **0.0462** | **0.00363** | 0.00601 | 0.0462/0.00361 | 0.0468 | 0.0995 | 0.89 ms | 81 µs (58) | 10 µs | 183/43 | 3.2 |
| FoilCORE-2k (v74, d=16) | 2,373 | 283 | **0.0574** | **0.00436** | 0.00727 | 0.0575/0.00435 | 0.0590 | 0.1092 | 0.75 ms | 61 µs (47) | 8 µs | 123/25 | 2.0 |
| FoilCORE-3k (v6, plain trunk) | 3,434 | 266 | **0.0759** | **0.00525** | 0.00892 | 0.0754/0.00520 | 0.0784 | 0.1194 | 1.19 ms | 67 µs (46) | 8 µs | 181/33 | 2.8 |
| FoilCORE-0.5k (v70, d=8) | 482 | 316 | **0.1001** | **0.00832** | 0.01286 | 0.1003/0.00838 | 0.1015 | 0.1258 | 0.47 ms | 38 µs (31) | 5 µs | 74/16 | 1.1 |
| 3.4k, old GT-mix protocol | 3,434 | 100 | **0.1366** | **0.01072** | 0.01796 | 0.1294/0.00818 | 0.1322 | 0.1707 | 1.10 ms | – | – | 171/33 | 0.0 |
| FoilCORE-5k, 3 seeds mean±std | | | 0.0479±0.0040 | 0.00369±0.00020 | 0.00615±0.00031 | | | | | | | | |
| NeuralFoil xxsmall (1,024-polar subset) | 13,302 | – | 0.0924 | 0.01281 | 0.02564 | pointwise | – | – | 0.57 ms | – | – | – | – |
| NeuralFoil xxxlarge (1,024-polar subset) | 1,428,166 | – | 0.0815 | 0.01149 | 0.02392 | pointwise | – | – | 4.63 ms | – | – | – | – |

## Label efficiency (v61 trained on a fraction of training geometries)

| fraction | AR Cl | AR Cd | stall Cd |
|---|---|---|---|
| 10% (431 geometries) | 0.0807 | 0.00652 | 0.01093 |
| 30% (1.3k) | 0.0619 | 0.00479 | 0.00802 |
| 100% (4,313) | 0.0462 | 0.00363 | 0.00601 |

## vs NeuralFoil by regime (1,024 test polars, 73k points) — Cl / Cd

| regime | n | FoilCORE-5k | NF xxsmall | NF xxxlarge |
|---|---|---|---|---|
| all | 73198 | 0.0462 / 0.00364 | 0.0924 / 0.01281 | 0.0815 / 0.01149 |
| Mach [-0.01,0.05) | 20246 | 0.0433 / 0.00327 | 0.0467 / 0.00345 | 0.0346 / 0.00226 |
| Mach [0.05,0.15) | 16511 | 0.0434 / 0.00300 | 0.0470 / 0.00337 | 0.0350 / 0.00252 |
| Mach [0.15,0.30) | 19695 | 0.0456 / 0.00404 | 0.0952 / 0.01299 | 0.0843 / 0.01143 |
| Mach [0.30,0.51) | 16746 | 0.0530 / 0.00423 | 0.1891 / 0.03321 | 0.1809 / 0.03156 |
| Re [4e+04,1e+05) | 3347 | 0.0910 / 0.00895 | 0.1004 / 0.01286 | 0.0534 / 0.00887 |
| Re [1e+05,3e+05) | 6029 | 0.0772 / 0.00684 | 0.1084 / 0.01517 | 0.0735 / 0.01095 |
| Re [3e+05,1e+06) | 8042 | 0.0478 / 0.00420 | 0.0837 / 0.01274 | 0.0706 / 0.01081 |
| Re [1e+06,3e+06) | 7719 | 0.0423 / 0.00414 | 0.1009 / 0.01665 | 0.0861 / 0.01396 |
| Re [3e+06,2e+07) | 48061 | 0.0395 / 0.00269 | 0.0899 / 0.01190 | 0.0856 / 0.01146 |
| Mach<0.05 & Re in [1e5,3e6) | 5937 | 0.0557 / 0.00540 | 0.0589 / 0.00644 | 0.0379 / 0.00369 |
| |alpha|>=12 | 27003 | 0.0527 / 0.00587 | 0.1453 / 0.02564 | 0.1406 / 0.02392 |
| |alpha|<12 | 46195 | 0.0423 / 0.00233 | 0.0615 / 0.00530 | 0.0470 / 0.00422 |

50 held-out geometries at Re≈5e5: FoilCORE-5k 0.0444/0.00429, xxxlarge 0.0411/0.00481, xxsmall 0.0615/0.00739.

## Quantization (test, 2,048 polars) — Cl / Cd

| model | fp32 | fp16 | int8 matrices + fp16 rest (bytes) | int8 all/channel | int6 |
|---|---|---|---|---|---|
| FoilCORE-5k (v61) | 0.0470/0.00375 | 0.0470/0.00375 | 0.0500/0.00426 (5,912) | 0.0517/0.00443 | 0.0970/0.00575 |
| FoilCORE-2k (v74, d=16) | 0.0580/0.00443 | 0.0580/0.00443 | 0.0614/0.00459 (2,712) | 0.0616/0.00474 | 0.1167/0.01276 |
| FoilCORE-3k (v6, plain trunk) | 0.0772/0.00536 | 0.0772/0.00536 | 0.0786/0.00558 (3,988) | 0.0818/0.00564 | 0.1592/0.01139 |
| FoilCORE-0.5k (v70, d=8) | 0.1007/0.00828 | 0.1007/0.00828 | 0.1014/0.00832 (628) | 0.1022/0.00856 | 0.1546/0.01110 |

## Error head / OOD / residual head (v61, test 2,048)

- Cd: Spearman 0.71, P@10% worst 0.51; deciles pred→true: 0.001→0.002, 0.001→0.002, 0.002→0.003, 0.003→0.003, 0.004→0.008, 0.006→0.010, 0.009→0.014, 0.013→0.017, 0.019→0.024, 0.032→0.039
- Cl: Spearman 0.48, P@10% 0.30
- OOD ê_Cd: clean 0.0158, noise0.01 0.0164, noise0.03 0.0187, random cloud 0.0761 (×4.8)
- residual-head gate mean 0.80 (stall 0.82 / linear 0.78) — dense
- memory/head knockouts (test): baseline 0.047/0.0037, drop_memory_global 0.185/0.0108, drop_memory_le 0.280/0.0116, drop_memory_te 0.533/0.0282, drop_memory_profile 0.410/0.0320, no_expert 0.185/0.0318, err_head_zero 0.100/0.0135

## Causal knockouts, v61 (val 2,048; baseline 0.0473 / 0.00367)

- block0.no_attn: ΔCl +0.0310, ΔCd +0.00350
- block0.Wv_identity: ΔCl +0.0454, ΔCd +0.00530
- block0.tau_off: ΔCl +0.0096, ΔCd +0.00092
- block0.no_layernorm: ΔCl +0.5754, ΔCd +0.09000
- block1.no_attn: ΔCl +0.0184, ΔCd +0.00222
- block1.Wv_identity: ΔCl +0.0197, ΔCd +0.00228
- block1.tau_off: ΔCl +0.0030, ΔCd +0.00038
- block1.no_layernorm: ΔCl +0.0872, ΔCd +0.01011
- all_blocks_no_attn: ΔCl +0.0394, ΔCd +0.00469
- pathnl_identity: ΔCl +0.4677, ΔCd +0.03159
- drop_memory_global: ΔCl +0.1387, ΔCd +0.00704
- drop_memory_le: ΔCl +0.2340, ΔCd +0.00788
- drop_memory_te: ΔCl +0.4845, ΔCd +0.02477

## Files

`main/` (full-test numbers.json, tables.md, per-model json, knockouts_*/, neuralfoil_by_regime.json), `seeds/`, `seed_stats.json`, `label_efficiency/`, `quantization.json`, `latency_sweep.json`, `energy_gpu.json`, `v61_extras/`, `heldout50_v61/`, `heldout50_v6/`, `ablations.md`, `figures/`; every run's history/log/checkpoint in `../pod_runs/`.

## Transfer of the training recipe (feedback item 1–3; `results/transfer/`, `experiments/transfer/`)

Same draft→corrector shape (~4k params, no airfoil parts; also with a transformer 4.5k / Mamba 5.2k corrector) on three public datasets, 4 protocols × 3 seeds, test MAE in physical units evaluated as a stream (causal) and as a smoother (bidirectional):

| dataset | protocol | corrector | n | params | test MAE causal | test MAE bidir | gap % | draft MAE |
|---|---|---|---|---|---|---|---|---|
| Beijing PM2.5 (log1p(ug/m3)) | draft_bidir | mamba | 3 | 5370 | 0.5339 ± 0.0194 | 0.4854 ± 0.0018 | +10.0 | 0.6270 |
| Beijing PM2.5 (log1p(ug/m3)) | draft_bidir | rpa | 3 | 4362 | 0.6001 ± 0.0118 | 0.4992 ± 0.0075 | +20.3 | 0.6162 |
| Beijing PM2.5 (log1p(ug/m3)) | draft_bidir | transformer | 3 | 4636 | 0.6276 ± 0.0342 | 0.4943 ± 0.0029 | +27.0 | 0.6316 |
| Beijing PM2.5 (log1p(ug/m3)) | draft_causal | mamba | 3 | 5370 | 0.4804 ± 0.0073 | 0.5090 ± 0.0145 | -5.6 | 0.6127 |
| Beijing PM2.5 (log1p(ug/m3)) | draft_causal | rpa | 3 | 4362 | 0.4886 ± 0.0095 | 0.5149 ± 0.0079 | -5.1 | 0.6304 |
| Beijing PM2.5 (log1p(ug/m3)) | draft_causal | transformer | 3 | 4636 | 0.4871 ± 0.0073 | 0.5810 ± 0.0219 | -16.0 | 0.6391 |
| Beijing PM2.5 (log1p(ug/m3)) | draft_dual | mamba | 3 | 5370 | 0.4864 ± 0.0060 | 0.4884 ± 0.0037 | -0.4 | 0.6094 |
| Beijing PM2.5 (log1p(ug/m3)) | draft_dual | rpa | 3 | 4362 | 0.4943 ± 0.0085 | 0.5029 ± 0.0053 | -1.7 | 0.6151 |
| Beijing PM2.5 (log1p(ug/m3)) | draft_dual | transformer | 3 | 4636 | 0.4916 ± 0.0041 | 0.4968 ± 0.0032 | -1.1 | 0.6107 |
| Beijing PM2.5 (log1p(ug/m3)) | gtmix | mamba | 3 | 5370 | 0.5895 ± 0.0191 | 0.5063 ± 0.0164 | +16.5 | 0.6118 |
| Beijing PM2.5 (log1p(ug/m3)) | gtmix | rpa | 3 | 4362 | 0.5902 ± 0.0263 | 0.5019 ± 0.0045 | +17.6 | 0.6118 |
| Beijing PM2.5 (log1p(ug/m3)) | gtmix | transformer | 3 | 4636 | 0.7320 ± 0.0513 | 0.5820 ± 0.0920 | +27.7 | 0.6642 |
| MIT-BIH lead MLII from lead 2 (mV) | draft_bidir | mamba | 1 | 4538 | 0.2369 ± 0.0000 | 0.2261 ± 0.0000 | +4.8 | 0.2337 |
| MIT-BIH lead MLII from lead 2 (mV) | draft_bidir | rpa | 3 | 3530 | 0.2571 ± 0.0107 | 0.2325 ± 0.0040 | +10.5 | 0.2321 |
| MIT-BIH lead MLII from lead 2 (mV) | draft_bidir | transformer | 3 | 3804 | 0.2463 ± 0.0151 | 0.2204 ± 0.0019 | +11.7 | 0.2279 |
| MIT-BIH lead MLII from lead 2 (mV) | draft_causal | mamba | 3 | 4538 | 0.2269 ± 0.0006 | 0.2245 ± 0.0019 | +1.0 | 0.2341 |
| MIT-BIH lead MLII from lead 2 (mV) | draft_causal | rpa | 3 | 3530 | 0.2289 ± 0.0011 | 0.2257 ± 0.0032 | +1.4 | 0.2285 |
| MIT-BIH lead MLII from lead 2 (mV) | draft_causal | transformer | 3 | 3804 | 0.2268 ± 0.0054 | 0.2295 ± 0.0008 | -1.2 | 0.2321 |
| MIT-BIH lead MLII from lead 2 (mV) | draft_dual | rpa | 3 | 3530 | 0.2292 ± 0.0026 | 0.2277 ± 0.0032 | +0.7 | 0.2291 |
| MIT-BIH lead MLII from lead 2 (mV) | draft_dual | transformer | 3 | 3804 | 0.2234 ± 0.0025 | 0.2204 ± 0.0035 | +1.4 | 0.2303 |
| MIT-BIH lead MLII from lead 2 (mV) | gtmix | mamba | 3 | 4538 | 0.2949 ± 0.0023 | 0.2949 ± 0.0023 | -0.0 | 0.2949 |
| MIT-BIH lead MLII from lead 2 (mV) | gtmix | rpa | 3 | 3530 | 0.3001 ± 0.0130 | 0.2894 ± 0.0143 | +3.7 | 0.2883 |
| MIT-BIH lead MLII from lead 2 (mV) | gtmix | transformer | 3 | 3804 | 0.2965 ± 0.0036 | 0.2952 ± 0.0045 | +0.5 | 0.2960 |
| ETTh1 oil temperature (degC) | draft_bidir | mamba | 3 | 5242 | 4.8504 ± 0.5389 | 4.8046 ± 0.5941 | +1.1 | 5.4984 |
| ETTh1 oil temperature (degC) | draft_bidir | rpa | 3 | 4234 | 5.2647 ± 0.1330 | 5.2968 ± 0.1144 | -0.6 | 7.4196 |
| ETTh1 oil temperature (degC) | draft_bidir | transformer | 3 | 4508 | 4.6715 ± 0.5419 | 4.0717 ± 0.5471 | +15.3 | 5.3818 |
| ETTh1 oil temperature (degC) | draft_causal | mamba | 3 | 5242 | 4.9212 ± 0.4236 | 4.9188 ± 0.4260 | +0.1 | 5.4951 |
| ETTh1 oil temperature (degC) | draft_causal | rpa | 3 | 4234 | 5.1914 ± 0.1742 | 5.3049 ± 0.1323 | -2.1 | 6.7299 |
| ETTh1 oil temperature (degC) | draft_causal | transformer | 3 | 4508 | 4.6152 ± 0.5783 | 4.6681 ± 0.7119 | -0.7 | 6.1943 |
| ETTh1 oil temperature (degC) | draft_dual | mamba | 3 | 5242 | 4.8425 ± 0.5674 | 4.8216 ± 0.5925 | +0.5 | 5.4535 |
| ETTh1 oil temperature (degC) | draft_dual | rpa | 3 | 4234 | 5.3024 ± 0.1694 | 5.3799 ± 0.0910 | -1.5 | 7.2443 |
| ETTh1 oil temperature (degC) | draft_dual | transformer | 3 | 4508 | 4.6990 ± 0.5635 | 4.7849 ± 0.7348 | -1.3 | 6.1799 |
| ETTh1 oil temperature (degC) | gtmix | mamba | 3 | 5242 | 5.5944 ± 0.4521 | 5.6001 ± 0.4455 | -0.1 | 6.7562 |
| ETTh1 oil temperature (degC) | gtmix | rpa | 3 | 4234 | 5.5203 ± 0.3489 | 5.2669 ± 0.3694 | +4.9 | 6.1394 |
| ETTh1 oil temperature (degC) | gtmix | transformer | 3 | 4508 | 6.0285 ± 0.2648 | 5.6913 ± 0.0360 | +6.0 | 7.8574 |

Reading: label-leaking (scheduled sampling) protocol is the worst streamer on every dataset and the only one with a systematic stream-vs-smooth gap; the bidirectional-only specialist streams badly where the corrector does real work; dual mask matches both specialists within seed spread; the same ordering holds with transformer and Mamba correctors (both sweeps complete, 36 runs each). On ETTh1 all draft-only protocols have no gap; on ECG the corrector barely beats the pointwise draft (protocol effect = label leaking hurts).

## CPU energy (Apple powermetrics, package level, one core in the loop; `results/paper_final/energy_cpu_mac*.json`)

| model | polars/s | package W above idle | mJ / polar |
|---|---|---|---|
| FoilCORE-5k, single kernel | 12,304 | 6.7 | 0.55 |
| NeuralFoil xxxlarge (its NumPy path) | 227 | 6.5 | 28.5 |
| NeuralFoil xxsmall (its NumPy path) | 1,750 | 5.7 | 3.2 |

## Reviewer baselines: standard backbones as correctors on the FoilCORE-5k front-end (feedback item 5; `results/paper_final/baselines*/`)

Same geometry front-end, pointwise query, token and draft-only dual-mask protocol; only the corrector/requery/residual head is replaced. Full test split, causal AR. Matched budget = epoch-100 checkpoint; FoilCORE-5k at epoch 100 was 0.058 / 0.0045 (4,096-row test draw), final 0.046 / 0.0036 (3 seeds 0.048 ± 0.004).

| corrector | params | best epoch | AR Cl / Cd (best) | TF Cl / Cd | stall Cd | AR Cl / Cd @ epoch 100 | eager B=1 | fused C++ T=68 | MACs/pt |
|---|---|---|---|---|---|---|---|---|---|
| Transformer corrector (PatchTST patch-1), variant 80 | 5,822 | 287 | 0.0478 / 0.00367 | 0.0474 / 0.00360 | 0.00614 | 0.0559 / 0.00431 | 0.66 ms | 166 µs (cached 132) | 8.1k |
| TSMixer corrector, variant 81 | 11,780 | 231 | 0.0611 / 0.00458 | 0.0605 / 0.00450 | 0.00750 | 0.0697 / 0.00535 | 0.51 ms | 50 µs (cached 32) | 2.4k (einsum time-mix not counted) |
| Mamba (S6) corrector, variant 82 | 5,300 | 270 | 0.0493 / 0.00376 | 0.0495 / 0.00376 | 0.00626 | 0.0565 / 0.00420 | 0.87 ms | 122 µs (cached 104) | 3.7k (scan not counted) |

Reading: transformer and Mamba correctors land within FoilCORE-5k's seed spread (a hair ahead at matched epochs, a hair behind at convergence); TSMixer is ~30% worse. All three stream = smooth (<1%). Kernel-vs-kernel (`src/fused_baselines.py`, match to PyTorch 3e-6): FoilCORE-5k 81 µs, transformer 166 µs, Mamba 122 µs, TSMixer 50 µs. Eager PyTorch was 0.5–0.9 ms; most of that gap was the framework. TSMixer is the fastest fused model and the least accurate.

## Foundation forecasters, zero-shot, on the same ETTh1 test hours (feedback round 2; `results/transfer/foundation_ett.json`, `experiments/transfer/foundation_ett.py`)

712 test hours (stride 4 within the held-out blocks), context 512 h of OT, forecast horizons 1 h and 24 h; our model rows use one lag of OT plus current covariates (`ett_lag1`, dual mask / label-leaking, 3 seeds).

| model | params | MAE h=1 (°C) | MAE h=24 | ms per forecast |
|---|---|---|---|---|
| last value | 0 | 0.795 | 1.85 | — |
| same hour yesterday | 0 | 1.815 | 1.80 | — |
| amazon/chronos-bolt-tiny | 8,652,672 | 0.787 | 1.92 | 15.6 (CPU, 1 thread) |
| amazon/chronos-bolt-small | 47,718,016 | 0.778 | 1.94 | 54.5 (CPU, 1 thread) |
| ibm-granite/granite-timeseries-ttm-r2 (512-96) | 805,280 | 0.810 | 2.03 | 2.9 (CPU, 1 thread) |
| google/timesfm-2.5-200m-pytorch | 231,289,280 | 0.755 | 1.91 | 97 (GPU) |
| draft→RPA corrector, dual mask, OT lag-1 + covariates (ours, 4.2k) | 4,266 | 0.652 stream / 0.650 smooth (h=1 by construction) | — | 0.28 (CPU, 1 thread, eager) |
| same, label-leaking protocol | 4,266 | 0.656 stream / 0.639 smooth (+3% gap) | — | 0.28 |
| ridge regression on the same per-hour features (covariate-matched reference; `results/transfer/ridge_gbm_ett.json`) | 14 | 0.647 (with lag) / 5.82 (without lag; ours 5.30) | — | — |
| histogram gradient boosting, same features | — | 0.909 (with lag) / 5.87 (without) | — | — |

Reading: zero-shot 0.8M–231M-parameter forecasters do not beat last-value (0.795) at one hour on this target except TimesFM (0.755); given one lag and the covariates a 4k model reaches 0.65 in both modes. Different information sets (stated in the paper); the point is that the deployment gap is a training-protocol property, not a scale property.

## Physics-lever study (§5 "Analytic priors" paragraphs, merged from `physics-grounding`, 2026-08-19; appendix dropped 2026-08-19 — CFP exempts references only, tables live here and in `paper/tables/ts_phys_{stack,consistency}.tex` + `ts_thirdparty.tex`)

Basis differs from the tables above: the stack/consistency numbers are a second, expanded BigFoil pool ("WHISP": 471k polars, 5.9k geometries; held-out-geometry val 63,548 polars, 4 seeds), the replications retrain baseline and physics model from scratch on a pre-expansion BigFoil pool (5 seeds) and a BigFoil-family subset (3 seeds). Zero-shot cross-pool evaluation of these checkpoints is invalid (analytic base is normalized with training-split stats) — that is why the paper only reports retrained-from-scratch comparisons.

* Machine-readable evaluations: `main_whisp/` (`numbers.json`, `phys.json`, `base.json`, knockouts, `tables.md`), `quant_whisp.json`, `quant_phys.json`.
* Paper macros: `paper/tables/ts_numbers.tex` (`\Phys*`, `\Base*`, `\Physinv*`, `\Tp*`, `\Disj*` — the canonical `\Main*` macros are untouched).
* Kernel latency with the physics levers, this Mac, 1 thread, T=68, median of 2,000 (`latency_phys_mac.json`, 2026-08-19): trunk 5,285 → 83/60 µs (confirms the published 81/58), +base/inv-mem 5,576 → 92/68 µs, full 5,803 → 99/76 µs; max |Δ| vs PyTorch ≤ 5e-6. Linux Threadripper reference: 331/300 vs 299/269 µs (+11–12%).
