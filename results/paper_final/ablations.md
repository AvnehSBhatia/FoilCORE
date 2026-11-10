| variant | description | params | epochs | best AR Cl | best AR Cd | best stall Cd | best TF Cl | AR Cl @50 | AR Cl @100 | run |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | FoilCORE 3.4k, original protocol (GT Cl/Cd mixed in, 1:31 loss) | 3434 | 51 | 0.1930 | 0.01615 | 0.02874 | 0.1908 | 0.1943 | – | stopped_ep47_v0_oldprotocol |
| 3 | FoilCORE 3.4k, AR protocol (draft-only, causal/bidir mix), 1:31 loss | 3434 | 154 | 0.1150 | 0.00572 | 0.00981 | 0.1128 | 0.1854 | 0.1253 | stopped_ep15x_v3_1to31 |
| 6 | FoilCORE 3.4k, AR protocol, 1:1 loss | 3434 | 278 | 0.0744 | 0.00529 | 0.00897 | 0.0740 | 0.1151 | 0.0882 | 20260818_023002_v6 |
| 50 | FoilCORE-v2 (SSM blocks + router + err head), 1:31 | 3054 | 100 | 0.1487 | 0.00682 | 0.01163 | 0.1494 | 0.1946 | 0.1487 | 20260818_014821_v50 |
| 51 | v2 without router, 1:31 | 2984 | 100 | 0.1659 | 0.00699 | 0.01207 | 0.1622 | 0.1846 | 0.1659 | 20260818_015133_v51 |
| 52 | v2 depth 3, 1:31 | 3729 | 54 | 0.1913 | 0.00816 | 0.01368 | 0.1931 | 0.1959 | – | stopped_v52_depth3 |
| 53 | v2, 1:1 | 3054 | 115 | 0.0863 | 0.00635 | 0.01052 | 0.0875 | 0.1147 | 0.0912 | 20260818_023002_v53 |
| 54 | FoilCORE + error head, 1:1 | 3952 | 115 | 0.0831 | 0.00575 | 0.00986 | 0.0825 | 0.1092 | 0.0904 | 20260818_031003_v54 |
| 55 | v2 no router, 1:1 | 2984 | 77 | 0.1064 | 0.00760 | 0.01231 | 0.1072 | 0.1170 | – | stopped_v55_v2norouter |
| 60 | FoilCORE + err head + profile memory, 1:1 | 4688 | 41 | 0.0772 | 0.00612 | 0.00972 | 0.0768 | – | – | 20260818_043056_v60 |
| 61 | FoilCORE-5k: + err head + profile + ê-gated residual head, 1:1 | 5285 | 358 | 0.0468 | 0.00371 | 0.00614 | 0.0467 | 0.0695 | 0.0585 | 20260818_043056_v61 |
| 61 | FoilCORE-5k: + err head + profile + ê-gated residual head, 1:1 (10\% of geometries) | 5285 | 300 | 0.0796 | 0.00642 | 0.01054 | 0.0781 | 0.1732 | 0.1160 | 20260818_062041_v61 |
| 61 | FoilCORE-5k: + err head + profile + ê-gated residual head, 1:1 (30\% of geometries) | 5285 | 300 | 0.0620 | 0.00469 | 0.00775 | 0.0621 | 0.0940 | 0.0800 | 20260818_062046_v61 |
| 61 | FoilCORE-5k: + err head + profile + ê-gated residual head, 1:1 (seed 1) | 5285 | 300 | 0.0454 | 0.00362 | 0.00608 | 0.0453 | 0.0689 | 0.0574 | 20260818_065901_v61 |
| 61 | FoilCORE-5k: + err head + profile + ê-gated residual head, 1:1 (seed 2) | 5285 | 254 | 0.0525 | 0.00397 | 0.00660 | 0.0525 | 0.0701 | 0.0596 | 20260818_065906_v61 |
| 70 | FoilCORE d=8, 1:1 | 482 | 328 | 0.0990 | 0.00840 | 0.01280 | 0.0994 | 0.1272 | 0.1218 | 20260818_045339_v70 |
| 74 | FoilCORE-2k: full stack at d=16 | 2373 | 295 | 0.0582 | 0.00443 | 0.00737 | 0.0582 | 0.0779 | 0.0679 | 20260818_060304_v74 |