# results/ — index

| folder | what | produced by |
|---|---|---|
| `paper_final/` | The airfoil results package behind `paper/fmts/fmts.pdf`. `README.md` inside lists every number. `main/numbers.json` (test-set suite for v61/v74/v6/v70/old_protocol), `main/neuralfoil_by_regime.json`, `main_fulltest/`, `seeds/` + `seed_stats.json`, `label_efficiency/`, `quantization.json`, `latency_sweep.json`, `energy_gpu.json`, `v61_extras/` (error head, OOD, expert gate, test knockouts), `heldout50_v61/`, `heldout50_v6/`, `figures/`, `ablations.md`, `baselines/` and `baselines_ep100/` (variants 80–82), `energy_cpu_mac.json`. | `tools/finalize_paper.sh` |
| `transfer/` | Cross-domain protocol study: one JSON per (dataset, protocol, corrector, seed) with test causal/bidir/draft MAE and the training history; `summary.md` / `summary.json` = mean ± std over seeds; `per_seed.md` = every seed of every cell; `foundation_ett.json`, `ridge_gbm_ett.json` = the ETTh1 reference points. | `experiments/transfer/train.py`, `summarize.py` |
| `pod_runs/` | Every training run of the campaign (RunPod A4500): `metrics_history.json` (per-epoch causal-AR + TF + stall metrics), `train.log`, `env.txt`, `foilcore_best.pt`. Directory names are `<UTC stamp>_v<variant>`; the mapping to model names is in `pod_runs/README.md`. | `tools/launch_runs.sh` on the pod, pulled by `tools/pod_pull.sh` |
| `analysis_v6_geometry_code/` | PCA / cosine analysis of the plain 3.4k model's three region memories over 2,048 held-out airfoils (the "geometry code lives in 3–7 dimensions" finding, paper §7). | `paper/scripts/analyze_foilcore2.py` (geometry section) |
| (removed 2026-08-19) `autopsy_3k/` weight analysis / layer atlas of the old 3.4k label-leaking model: in git history at `c86d7c2`. |  |
| (removed 2026-08-19) `archive/paper_20260817/` epoch-100 package and `ablations_draft`: in git history at `c86d7c2`. |  |
