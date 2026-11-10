# FoilCORE

A **5k-parameter streaming airfoil-polar predictor** and the training recipe behind it. Lift and drag over angle of attack for an unseen airfoil at a given (Re, Mach), predicted as a causal stream, in **81 µs per polar on one CPU thread**, from **5,285 parameters** (fp16 weights: 10.6 KB). Paper: *Streaming for free: one weight vector for the filter and the smoother*, accepted (poster) at the FMTS workshop @ NeurIPS 2026.

* Paper: [`paper/fmts/fmts.pdf`](paper/fmts/fmts.pdf) (camera-ready, source `paper/fmts/fmts.tex`); as reviewed: `paper/fmts/fmts_submitted.{tex,pdf}`
* Paper sources, build and figure provenance: [`paper/fmts/README.md`](paper/fmts/README.md)
* Every number in the paper: [`results/paper_final/README.md`](results/paper_final/README.md), [`results/transfer/`](results/transfer), [`results/transfer_qsweep/`](results/transfer_qsweep)
* Exhaustive file listing: [`docs/INDEX.md`](docs/INDEX.md) (regenerate with `python tools/make_index.py`)

## Headline numbers

Held-out-**geometry** test split (34,043 XFOIL polars, 2.4M points, none of the 539 airfoils seen in training), causal streaming mode, physical MAE:

| model | params | Cl MAE | Cd MAE | stall Cd (\|α\|≥12°) | B=1 latency (C++ kernel, 1 thread) |
|---|---|---|---|---|---|
| **FoilCORE-5k** (`checkpoints/v61`) | 5,285 | **0.046** (3 seeds 0.048 ± 0.004) | **0.0036** (0.0037 ± 0.0002) | 0.0060 | **81 µs** (58 µs with cached geometry, 10 µs per step at T=8) |
| FoilCORE-2k (`checkpoints/v74`, d=16) | 2,373 | 0.057 | 0.0044 | 0.0073 | 61 µs |
| FoilCORE-3k (`checkpoints/v6`, plain trunk) | 3,434 | 0.076 | 0.0052 | 0.0089 | 67 µs |
| FoilCORE-0.5k (`checkpoints/v70`, d=8) | 482 | 0.100 | 0.0083 | 0.0129 | 38 µs |
| NeuralFoil xxsmall (same rows) | 13,302 | 0.092 | 0.0128 | 0.0256 | 0.57 ms |
| NeuralFoil xxxlarge (same rows) | 1,428,166 | 0.082 | 0.0115 | 0.0239 | 4.6 ms |

* Streaming (causal) and full-polar (bidirectional) predictions agree within 1% for FoilCORE, and within 2% for every model trained with the recipe: the same weights are an online filter and an offline smoother. This comes from the training protocol (draft-only inputs, random causal/bidirectional mask per batch), and the paper tests the same protocol on three unrelated public datasets (`experiments/transfer/`).
* NeuralFoil xxxlarge is still 1.2×/1.4× better in incompressible mid-Re flow (it has no Mach input; our data spans Mach 0–0.5); FoilCORE wins overall, in stall (2.7–4×), and beats the 13k xxsmall in every regime.
* Trained on 10% of the training geometries FoilCORE-5k still reaches 0.081 / 0.0065; on 30%, 0.062 / 0.0048.
* The error head is calibrated for drag (Spearman 0.71) and rises 4.8× on a non-airfoil point cloud (OOD flag).
* fp16 weights are lossless; INT8 matrices + fp16 rest (5.9 KB) cost +6% Cl / +14% Cd.
* Standard backbones as correctors on the same front-end and protocol (`src/baselines.py`, variants 80–82): transformer (5,822 p) 0.049 / 0.0037, Mamba (5,300 p) 0.050 / 0.0038, TSMixer (11,780 p) 0.061 / 0.0046, all within 2% between modes. Fused C++ kernels (`src/fused_baselines.py`) for all three: 166 / 122 / 50 µs vs FoilCORE-5k 81 µs (eager was 0.7–0.9 ms). The recipe and the geometry code carry the accuracy; RPA is the smallest corrector. Numbers: `results/paper_final/baselines/`, matched-budget (epoch 100) in `results/paper_final/baselines_ep100/`.
* Why it works: §2 of the paper gives two propositions that explain the recipe (they do not guarantee that the modes agree): letting the corrector read true labels shifts its inputs, with a training/deployment difference ≤ L·q·draft error; the random mask optimizes the average of the two modes' risks. Foundation forecasters zero-shot on ETTh1 (Chronos-Bolt, TTM, TimesFM-2.5): 0.75–0.81 °C at 1 h vs 0.80 last-value; our 4k model with one lag + covariates 0.65 in both modes (`results/transfer/foundation_ett.json`).
* The training recipe transfers: on ETTh1 / Beijing PM2.5 / MIT-BIH ECG (`experiments/transfer/`, `results/transfer/summary.md`) models trained under one mask are up to 27% worse in the other mode, while the dual-mask model keeps both within 2% and streams within seed spread of a causal-only model. In one setting (transformer corrector, ETTh1) a full-mask specialist smooths 15% better. With the leak fraction q fixed (`results/transfer_qsweep/`), leaking labels raises the error of both modes, by up to 25%.

## What the model is

`pointwise query → polar-wise refinement → error estimate`, width 32 throughout (`src/model.py`, `model_type="foilcore"`, variant 61 in `src/ablations.py`):

1. **Geometry → 4 memories (1.2k params).** 200 contour points → shared `Linear(2,32)` + per-channel *path nonlinearity* φ (Chebyshev series + Gaussian bump; pretrained as an autoencoder, `checkpoints/embeddings_best.pt`) → masked means over whole contour / leading edge / trailing edge, plus a **profile memory**: camber and thickness sampled analytically at 10 cosine chord stations → `φ(Linear(20,32))`.
2. **Pointwise query (0.5k).** Each (α, Re, Ma) makes per-channel Δ, B, C; a selective scan over the 4 memories gives a context; `Linear(32,2)` → draft (Cl, Cd). Every point independent.
3. **Error head (0.5k).** Memories (reduced) + predicted 5-vector → 16-unit MLP → ê = predicted |draft error|, own L1 loss.
4. **Polar corrector (2.5k).** Tokens `Linear(7,32)[α,Re,Ma,Ĉl,Ĉd,ê] + context + sinusoid(α)` → two residual-product-attention blocks (Q=K=s₁u+s₃, V=W_v u, −Δα²/τ, causal or full mask) + LayerNorm + φ.
5. **Requery + residual head (0.6k).** Corrector state added to the output gate, memories re-queried with the same head; an ê-gated 16-unit MLP adds a residual.

**Training protocol** (the biggest single lever, `src/fast_train.py`): the corrector only ever sees the model's own draft (no scheduled sampling of ground truth), each batch is causal or bidirectional with p=½, equal-weight MAE on z-scored Cl/Cd + 0.25× draft MAE + 0.1× error-head L1, AdamW 3e-4, batch 1024, plateau LR, early stopping on causal validation loss.

## Repository index

Everything needed to reproduce the paper is here except the raw data CSVs (2.8 GB, not redistributed, rebuild below) and the C++ build cache.

```
README.md              this file (overview + index)
LICENSE, requirements.txt, configs/default.yaml

docs/
  INDEX.md             generated listing of every file with sizes (tools/make_index.py)

paper/
  fmts/                FMTS @ NeurIPS 2026: fmts.tex (camera-ready), fmts_submitted.tex/.pdf (as reviewed), checklist, build.sh
  references.bib, neurips_2026.sty (2025 style + 2026 workshop notice; swap for the official file)
  tables/ts_*.tex      GENERATED (do not hand-edit): ts_main/ts_numbers/ts_ko_macros by tools/export_latex_tables.py, ts_qsweep by
                       experiments/transfer/qsweep.py, ts_transfer_sd by paper/scripts/fmts_figures.py (ts_transfer: as reviewed)
  figures/ts/          modes, protocols (fmts_figures.py), qsweep (qsweep.py), architecture, dualmask (draw_architecture.py)
  scripts/
    fmts_figures.py          Figure 1 (modes), Figure 2 (protocols) and the appendix protocol table
    make_paper_figures.py    training curves, Pareto plots, knockout bars, error-head figure
    draw_architecture.py     block diagram + streaming/latency panel
    compare_heldout_geoms.py 50 held-out airfoils vs NeuralFoil (small multiples, scatter, noise robustness)
    analyze_v61_extras.py    error-head calibration, OOD response, expert gate, test-set knockouts
    analyze_foilcore2.py     causal-mode knockouts for any checkpoint (used by tools/paper_numbers.py)
    analyze_weights.py, visualize_layers.py, compare_val_airfoil_ar.py   weight autopsy / layer atlas tools (outputs → results/autopsy_3k, regenerable)

src/
  model.py             FoilCORE (+ FoilSCAN, foilcore2, seqbaseline model types), PathNonlinearity, RPA attention, selective scans, profile features
  baselines.py         transformer / TSMixer / Mamba backbones used as drop-in correctors (reviewer baselines)
  ablations.py         variant registry: 0/3/6 trunk protocols, 40–42 FoilSCAN, 50–55 v2, 60–63 profile/soft-bin, 70–74 width, 80–82 baselines
  fast_train.py        GPU-resident trainer (tensorised data, length buckets, causal + TF val each epoch, --compile on CUDA)
  train_model.py       original DataLoader trainer (kept for compatibility; defines ar_mix_probability, prepare_polar_batch)
  train_embed.py       stage-1 geometry autoencoder → checkpoints/embeddings_best.pt
  fused_forward.py     single-kernel C++ inference for FoilCORE (JIT-built; verified vs PyTorch to 5e-6)
  fused_baselines.py   same for transformer / TSMixer / Mamba correctors (variants 80–82)
  fused_scan.py        C++ affine scan for the FoilSCAN variants on CPU
  benchmark.py         load_models + TF / seeded / full-AR benchmark;  benchmark_stall.py  stall-window metrics
  data.py, embeddings.py, utils.py, geom_metrics.py, inference.py, dataset_split_report.py

experiments/transfer/  cross-domain test of the training recipe
  datasets.py          ETTh1, Beijing PM2.5, MIT-BIH loaders → windows (downloads to experiments/transfer/data/, gitignored)
  model.py             generic draft → RPA-corrector model (FoilCORE's corrector, no airfoil parts)
  train.py             one run: --dataset --protocol {gtmix, draft_causal, draft_bidir, draft_dual, gtmix_consist} [--q --draft-noise] --seed → JSON
  qsweep.py            leak-fraction sweep + consistency baseline → summary, ts_qsweep.tex, qsweep.pdf
  summarize.py         runs/transfer/*.json → summary.md + summary.json (mean ± std over seeds)
  foundation_ett.py    zero-shot Chronos-Bolt / TinyTimeMixer / TimesFM-2.5 on the ETTh1 test hours (params, MAE, latency)
  ridge_gbm_ett.py     covariate-matched ridge / gradient-boosting reference points on the same ETTh1 features

tools/
  finalize_paper.sh    one shot: pull pod runs → all numbers → tables → figures → PDF
  paper_numbers.py     test-set suite for a set of checkpoints (AR/TF/seeded, regimes, NeuralFoil head-to-head, knockouts, latency)
  export_latex_tables.py  results/<pkg> → paper/tables/ts_*.tex (+ baseline rows, transfer table)
  latency_sweep.py, latency_b1.py, throughput_test.py, quant_eval.py, ablation_table.py, seed_stats.py, nf_regime.py, quick_val.py
  energy_gpu.py        GPU J/polar (nvidia-smi board power minus idle);  energy_cpu_mac.sh + energy_cpu_loop.py  CPU J/polar via powermetrics (sudo)
  macs_per_point.py    matmul MACs per polar point for any checkpoint (FlopCounterMode)
  launch_runs.sh, continue_run.sh, runs_status.py   multi-run training on one GPU with per-run logs
  pod_setup.sh / pod_feedback_runs.sh / pod_pull.sh  new RunPod: set up, run every feedback experiment, pull results
  pod_transfer_arch.sh                                transfer study with a transformer / Mamba corrector (recipe is architecture-agnostic)
  local_feedback_runs.sh / local_transfer_runs.sh    same on the laptop (MPS / CPU) when no pod is available
  evals_neuralfoil.py, neuralfoil_baseline.py        NeuralFoil comparisons;  export_foilcore_onnx.py  best-effort ONNX
  build_bigfoil_dataset.py  geometry-held-out train/val/test split from a BigFoil dump;  run_ablation_suite.py  legacy sweep driver

results/
  paper_final/         the results package behind the paper (README.md indexes it): main/ (numbers.json, per-model json, knockouts,
                       neuralfoil_by_regime.json), main_fulltest/, seeds/, label_efficiency/, quantization.json, latency_sweep.json,
                       energy_gpu.json, v61_extras/, heldout50_v61/, heldout50_v6/, figures/, ablations.md
                       baselines/ + baselines_ep100/ (variants 80–82 test numbers, best epoch and matched budget); energy_cpu_mac.json once produced
  transfer/            cross-domain protocol study outputs (json per run incl. transformer/Mamba-corrector runs, summary.md/json)
  transfer_qsweep/     leak-fraction sweep (fixed q, three draft-noise levels) and consistency baseline
  pod_runs/            airfoil training runs behind the checkpoints and ablations: metrics_history.json, train.log, env.txt, best checkpoint
  analysis_v6_geometry_code/  PCA of the plain model's memories (the "geometry code lives in 3–7 dims" finding)

checkpoints/
  v61 (FoilCORE-5k), v74 (2k), v6 (3k plain), v70 (0.5k), v61_seed1/2, v61_frac10/30, each with metrics_history.json
  foilcore_best.pt     the old 3.4k label-leaking-protocol model (the "before" baseline in Table 1)
  embeddings_best.pt   d=32 path nonlinearity + point encoder pretrained as an autoencoder

data/                  train/val/test CSVs + split_manifest.json + .aero_cache/ (gitignored, see below)
runs/                  local training outputs (gitignored; keepers are copied into results/)
build/                 JIT-built C++ extensions (gitignored)
```

## Reproduce

Python 3 with PyTorch ≥ 2.6 (`pip install -r requirements.txt`). Data CSVs are not redistributed (`python tools/build_bigfoil_dataset.py --src <BigFoil dump> --out-dir data --seed 42` rebuilds the geometry-held-out split); everything else is in the repo.

```bash
# train FoilCORE-5k (variant 61) on a GPU (~40 min / 100 epochs on an RTX A4500)
python -m src.fast_train --variant-id 61 --save-dir runs/v61 --compile --batch-size 1024 --epochs 300 --early-stop-patience 12
# several variants at once with full logging (80 81 82 = the reviewer baselines)
bash tools/launch_runs.sh "80 81 82" --compile --batch-size 1024 --val-batch-size 1024 --epochs 300 --early-stop-patience 12
python tools/runs_status.py

# every paper number for a set of checkpoints (test set, NeuralFoil head-to-head, knockouts, latency)
python tools/paper_numbers.py --ckpts v61=checkpoints/v61/foilcore_best.pt --n 40000 --n-nf 1024
# or the whole package + tables + figures + PDF
bash tools/finalize_paper.sh results/paper_final

# cross-domain protocol study (small; CPU is fine)
python experiments/transfer/train.py --dataset air --protocol draft_dual --seed 0 --out runs/transfer/air_draft_dual_s0.json
python experiments/transfer/summarize.py runs/transfer

# single-kernel inference
python - <<'EOF'
import torch; from src.benchmark import load_models; from src.fused_forward import FusedFoilCORE
m, stats, cfg, _ = load_models("checkpoints/v61/foilcore_best.pt", torch.device("cpu"))
ff = FusedFoilCORE(m)              # coords: (200,2) chord-normalised; polar: (T,5) z-scored [α,Re,Ma,·,·]
# out = ff(coords, polar, causal=True)   → (T,2) normalised Cl,Cd; ff.encode(m, coords) caches the geometry
EOF

# paper
bash paper/fmts/build.sh                  # → paper/fmts/fmts.pdf
python experiments/transfer/qsweep.py results/transfer_qsweep --tex paper/tables/ts_qsweep.tex --fig paper/figures/ts/qsweep.pdf
python paper/scripts/fmts_figures.py      # Figures 1-2 and the appendix protocol table
```

## Notes and caveats
* Labels are XFOIL (ncrit = 9) only. NeuralFoil was trained on XFOIL sweeps of UIUC-derived airfoils, so its test overlap cannot be ruled out; it takes no Mach input.
* All latencies are single-thread on an Apple M-series laptop core; NeuralFoil measured on the same core with its own NumPy path. GPU energy (3.2 mJ/polar for FoilCORE-5k) measured on an RTX A4500 board minus idle; CPU energy 0.55 mJ/polar (NeuralFoil xxsmall 3.2, xxxlarge 28.5 mJ) measured with `tools/energy_cpu_mac.sh` (powermetrics, package level, upper bound; `results/paper_final/energy_cpu_mac.json`).
* Ablation rows are single training runs; the main model is reported over three seeds.
