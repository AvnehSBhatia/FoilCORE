# checkpoints/

| path | model | params | test causal AR Cl / Cd |
|---|---|---|---|
| `v61/foilcore_best.pt` | **FoilCORE-5k** (variant 61: trunk + error head + profile memory + ê-gated residual head) | 5,285 | 0.046 / 0.0036 |
| `v61_seed1/`, `v61_seed2/` | same, other seeds | 5,285 | 0.045 / 0.0035, 0.053 / 0.0039 |
| `v61_frac10/`, `v61_frac30/` | same, trained on 10% / 30% of the training geometries | 5,285 | 0.081 / 0.0065, 0.062 / 0.0048 |
| `v74/foilcore_best.pt` | FoilCORE-2k (d=16) | 2,373 | 0.057 / 0.0044 |
| `v6/foilcore_best.pt` | FoilCORE-3k plain trunk, draft-only dual-mask protocol | 3,434 | 0.076 / 0.0052 |
| `v70/foilcore_best.pt` | FoilCORE-0.5k (d=8) | 482 | 0.100 / 0.0083 |
| `foilcore_best.pt` | old 3.4k model, label-leaking (scheduled sampling) protocol | 3,434 | 0.137 / 0.0107 |
| `embeddings_best.pt` | stage-1 geometry autoencoder (d=32 point encoder + path nonlinearity) | — | — |

Each `vXX/` folder also holds `metrics_history.json` (per-epoch train/val curves). Checkpoints store `model`, `geom_encoder`, `stats` (z-score statistics; always evaluate with these), `variant_model` (config), `variant_training`. Load with `src.benchmark.load_models(path, device)`.
