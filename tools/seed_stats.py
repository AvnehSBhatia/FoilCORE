"""Mean/std over the main run + seed runs → <out>/seed_stats.json"""
import json, sys, numpy as np
out = sys.argv[1]
M = json.load(open(f"{out}/main/numbers.json"))["models"]["v61"]["accuracy"]["ar"]
S = json.load(open(f"{out}/seeds/numbers.json"))["models"]
vals = {k: [M[k]] + [S[s]["accuracy"]["ar"][k] for s in ("v61_seed1", "v61_seed2")] for k in ("cl", "cd", "stall_cd")}
json.dump({k: {"values": v, "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))} for k, v in vals.items()}, open(f"{out}/seed_stats.json", "w"), indent=1)
print({k: (round(np.mean(v), 5), round(np.std(v, ddof=1), 5)) for k, v in vals.items()})
