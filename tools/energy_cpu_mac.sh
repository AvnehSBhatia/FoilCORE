#!/usr/bin/env bash
# CPU energy per polar on the laptop: Apple powermetrics (needs sudo) around the single-kernel loop.
#   sudo bash tools/energy_cpu_mac.sh [ckpt] [seconds]                 → results/paper_final/energy_cpu_mac.json
#   sudo bash tools/energy_cpu_mac.sh neuralfoil:xxxlarge [seconds]     → results/paper_final/energy_cpu_mac_neuralfoil_xxxlarge.json
# Reports package CPU power idle vs. under the loop; energy/polar = (P_loop - P_idle) / polars_per_s.
set -euo pipefail
CKPT="${1:-checkpoints/v61/foilcore_best.pt}"; SEC="${2:-30}"
PY="${PY:-python}"
cd "$(dirname "$0")/.."
OUT=results/paper_final/energy_cpu_mac.json
LOOPARGS="--ckpt $CKPT"
case "$CKPT" in neuralfoil:*) SZ="${CKPT#neuralfoil:}"; LOOPARGS="--neuralfoil $SZ"; OUT="results/paper_final/energy_cpu_mac_neuralfoil_${SZ}.json";; esac
TMP=$(mktemp -d)
echo "== idle ${SEC}s"
powermetrics --samplers cpu_power -i 1000 -n "$SEC" > "$TMP/idle.txt" 2>/dev/null
echo "== loop ${SEC}s"
sudo -u "${SUDO_USER:-$USER}" env OMP_NUM_THREADS=1 PYTHONPATH="$PWD" $PY tools/energy_cpu_loop.py $LOOPARGS --seconds "$SEC" > "$TMP/loop.txt" &
LOOP=$!
sleep 3
powermetrics --samplers cpu_power -i 1000 -n "$((SEC-6))" > "$TMP/busy.txt" 2>/dev/null
wait $LOOP
$PY - "$TMP" "$OUT" "$CKPT" <<'PYEOF'
import re, sys, json, statistics, pathlib
tmp, out, ckpt = sys.argv[1:4]
def mw(f):
    v=[float(x) for x in re.findall(r"CPU Power:\s*([\d.]+)\s*mW", open(f).read())]
    return statistics.mean(v) if v else float("nan"), len(v)
idle,ni=mw(f"{tmp}/idle.txt"); busy,nb=mw(f"{tmp}/busy.txt")
pps=float(re.search(r"POLARS_PER_S ([\d.]+)", open(f"{tmp}/loop.txt").read()).group(1))
res={"ckpt":ckpt,"idle_cpu_mW":idle,"busy_cpu_mW":busy,"samples":[ni,nb],"polars_per_s":pps,
     "uJ_per_polar_dynamic":(busy-idle)/pps*1e3,"uJ_per_polar_total":busy/pps*1e3}
pathlib.Path(out).parent.mkdir(parents=True,exist_ok=True); json.dump(res,open(out,"w"),indent=1)
print(f"idle {idle:.0f} mW, loop {busy:.0f} mW, {pps:,.0f} polars/s -> {res['uJ_per_polar_dynamic']:.1f} uJ/polar above idle ({res['uJ_per_polar_total']:.1f} total)")
PYEOF
rm -rf "$TMP"
