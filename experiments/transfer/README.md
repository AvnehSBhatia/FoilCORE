# experiments/transfer — does the training recipe transfer?

Tests FoilCORE's two training choices (corrector sees only the model's own draft; each batch randomly causal or bidirectional) on three public datasets that have nothing to do with airfoils, using the same *shape* of model with the airfoil parts removed (`model.py`: pointwise MLP draft → FoilCORE's two-block residual-product-attention corrector → residual head, ~4k parameters).

| key | dataset | task | windows | split |
|---|---|---|---|---|
| `ett` | ETTh1 (Zhou et al. 2021) | 6 load channels + hour/day/season features → oil temperature | T=96 h | held-out 4-week blocks (virtual-sensor task; the standard forecasting split puts all of winter 2017/18 in test and loads→OT does not transfer across it) |
| `air` | Beijing PM2.5 (Liang et al. 2015, UCI) | dew point, temperature, pressure, wind → log1p PM2.5 | T=72 h | 2010–12 train / 2013 val / 2014 test |
| `ecg` | MIT-BIH arrhythmia (PhysioNet) | second lead → lead MLII, 90 Hz | T=128 samples | by record (32 / 6 / 10) |

Protocols (`train.py --protocol`): `gtmix` = scheduled sampling of the true target into the corrector, bidirectional batches (what FoilCORE used before); `draft_causal`, `draft_bidir` = draft-only specialists; `draft_dual` = draft-only, causal with p=0.5 (the recipe). Model selection on validation in the protocol's native mode; report test MAE evaluated both as a stream (causal mask) and as a smoother (full mask).

```bash
python experiments/transfer/datasets.py            # download + cache all three (data/ is gitignored)
python experiments/transfer/train.py --dataset air --protocol draft_dual --seed 0 --out runs/transfer/air_draft_dual_s0.json
python experiments/transfer/summarize.py runs/transfer   # → summary.md / summary.json
```

Full sweep: `tools/local_transfer_runs.sh` (laptop) or the transfer block of `tools/pod_feedback_runs.sh` (pod). Results are copied to `results/transfer/`; `tools/export_latex_tables.py` turns `summary.json` into `paper/tables/ts_transfer.tex`.
