# results/pod_runs — run directory → model name

| run dir | variant | model / purpose | used as |
|---|---|---|---|
| 20260818_043056_v61 | 61 | FoilCORE-5k (final) | `checkpoints/v61` |
| 20260818_065901_v61, 20260818_065906_v61 | 61 | seeds 1, 2 | `checkpoints/v61_seed1/2` |
| 20260818_062041_v61, 20260818_062046_v61 | 61 | 10% / 30% of training geometries | `checkpoints/v61_frac10/30` |
| 20260818_060304_v74 | 74 | FoilCORE-2k (d=16) | `checkpoints/v74` |
| 20260818_023002_v6 | 6 | FoilCORE-3k plain trunk, draft-only dual-mask, 1:1 loss | `checkpoints/v6` |
| 20260818_045339_v70 | 70 | FoilCORE-0.5k (d=8) | `checkpoints/v70` |
| 20260818_043056_v60 | 60 | + error head + profile memory (no residual head) | ablation ladder |
| 20260818_044612_v62/63 | 62/63 | soft-bin pooling variants | negative result |
| 20260818_045339_v71/72/73 | 71–73 | MoE corrector variants at d=8/16 | negative result |
| 20260818_014821_v50, 015133_v51, 023002_v53, 031003_v54 | 50–54 | FoilCORE-v2 (SSM blocks + router + err head) | "what did not work" |
| stopped_ep47_v0_oldprotocol | 0 | 3.4k, label-leaking (GT-mix) protocol, 1:31 | protocol ablation |
| stopped_ep15x_v3_1to31 | 3 | 3.4k, draft-only dual-mask, 1:31 loss | protocol ablation |
| 20260819_001018_v80 / _v81 / _v82 | 80–82 | transformer / TSMixer / Mamba correctors on the FoilCORE-5k front-end (reviewer baselines), seed 0; `foilcore_epoch_100.pt` kept for the matched-budget comparison | Table 1 lower block; `results/paper_final/baselines*/` |

`python tools/ablation_table.py --root results/pod_runs` rebuilds `results/paper_final/ablations.md` from these histories.
