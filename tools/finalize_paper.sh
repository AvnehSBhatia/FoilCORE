#!/usr/bin/env bash
# One-shot: pull latest checkpoints/histories from the pod, run every number, rebuild tables/figures/PDF.
#   bash tools/finalize_paper.sh results/paper_final
set -euo pipefail
OUT="${1:-results/paper_final}"
POD_HOST="${POD_HOST:-213.173.107.104}"; POD_PORT="${POD_PORT:-12227}"   # override: POD_HOST=... POD_PORT=... bash tools/finalize_paper.sh
POD="root@$POD_HOST"; SSH="ssh -o ConnectTimeout=10 -p $POD_PORT -i $HOME/.ssh/id_ed25519"; PY="${PY:-python}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT" OMP_NUM_THREADS=1
cd "$ROOT"
mkdir -p "$OUT" results/pod_runs
echo "== pull runs (skipped if the pod is unreachable)"
rsync -rlt --include='*/' --include='foilcore_best.pt' --include='metrics_history*.json' --include='env.txt' --include='train*.log' --exclude='*' -e "$SSH" $POD:/workspace/FoilCORE/runs/ results/pod_runs/ || echo "   pod unreachable; using local results/pod_runs"
# map run dirs -> names
for pair in v61:20260818_043056_v61 v74:20260818_060304_v74 v70:20260818_045339_v70 v6:20260818_023002_v6 v61_frac10:20260818_062041_v61 v61_frac30:20260818_062046_v61 v61_seed1:20260818_065901_v61 v61_seed2:20260818_065906_v61; do
  k="${pair%%:*}"; d="${pair#*:}"
  mkdir -p "checkpoints/$k"; cp "results/pod_runs/$d/foilcore_best.pt" "checkpoints/$k/" 2>/dev/null || echo "no ckpt for $k"; cp "results/pod_runs/$d/metrics_history.json" "checkpoints/$k/" 2>/dev/null || true
done
$PY - <<EOF
import torch
for k in "v61 v74 v70 v6 v61_frac10 v61_frac30".split():
    try:
        ck=torch.load(f"checkpoints/{k}/foilcore_best.pt",map_location="cpu",weights_only=False); print(k,"epoch",ck["epoch"],{a:round(b,5) for a,b in ck["val_causal"].items()})
    except Exception as e: print(k,"missing",e)
EOF
echo "== main suite"
$PY tools/paper_numbers.py --ckpts v61=checkpoints/v61/foilcore_best.pt v74=checkpoints/v74/foilcore_best.pt v6=checkpoints/v6/foilcore_best.pt v70=checkpoints/v70/foilcore_best.pt old_protocol=checkpoints/foilcore_best.pt --n 40000 --n-nf 1024 --nf-workers 6 --out "$OUT/main" 2>&1 | grep -v Warning | grep -v AeroDataset
echo "== label efficiency"
$PY tools/paper_numbers.py --ckpts v61_frac10=checkpoints/v61_frac10/foilcore_best.pt v61_frac30=checkpoints/v61_frac30/foilcore_best.pt --n 4096 --skip-knockouts --skip-neuralfoil --out "$OUT/label_efficiency" 2>&1 | grep -v Warning | grep -v AeroDataset
echo "== seeds"
$PY tools/paper_numbers.py --ckpts v61_seed1=checkpoints/v61_seed1/foilcore_best.pt v61_seed2=checkpoints/v61_seed2/foilcore_best.pt --n 40000 --skip-knockouts --skip-neuralfoil --out "$OUT/seeds" 2>&1 | grep -E "accuracy \\(|====="
$PY tools/seed_stats.py "$OUT"
rsync -rlt -e "$SSH" $POD:/workspace/FoilCORE/runs/energy_gpu.json "$OUT/energy_gpu.json" 2>/dev/null || true
echo "== regime vs neuralfoil"
sed "s#results/paper_final/main/neuralfoil_by_regime.json#$OUT/main/neuralfoil_by_regime.json#" tools/nf_regime.py > /tmp/nf_regime_final.py
$PY /tmp/nf_regime_final.py 2>&1 | grep -v Warning | grep -v AeroDataset
echo "== quantization"
$PY tools/quant_eval.py --ckpts v61=checkpoints/v61/foilcore_best.pt v74=checkpoints/v74/foilcore_best.pt v6=checkpoints/v6/foilcore_best.pt v70=checkpoints/v70/foilcore_best.pt --n 2048 --out "$OUT/quantization.json" 2>&1 | grep -v Warning | grep -v AeroDataset
echo "== extras + heldout50"
$PY paper/scripts/analyze_v61_extras.py --ckpt checkpoints/v61/foilcore_best.pt --out "$OUT/v61_extras" > "$OUT/v61_extras.log" 2>&1 || echo "extras failed"
$PY paper/scripts/compare_heldout_geoms.py --ckpt checkpoints/v61/foilcore_best.pt --csv data/test.csv --n-geoms 50 --out "$OUT/heldout50_v61" 2>&1 | grep -v Warning | grep -v AeroDataset
true
echo "== latency sweep"
$PY tools/latency_sweep.py --ckpts v61=checkpoints/v61/foilcore_best.pt v74=checkpoints/v74/foilcore_best.pt v6=checkpoints/v6/foilcore_best.pt v70=checkpoints/v70/foilcore_best.pt --out "$OUT/latency_sweep.json" 2>&1 | grep -v Warning
echo "== reviewer baselines (variants 80-82) if their checkpoints exist"
BASE=""; for v in 80 81 82; do for d in results/pod_runs/*_v$v runs/local_v$v; do [ -f "$d/foilcore_best.pt" ] && BASE="$BASE v$v=$d/foilcore_best.pt" && break; done; done
if [ -n "$BASE" ]; then $PY tools/paper_numbers.py --ckpts $BASE --n 40000 --skip-knockouts --skip-neuralfoil --out "$OUT/baselines" 2>&1 | grep -E "accuracy \\(|====="; fi
echo "== transfer study summary if runs exist"
[ -d runs/transfer ] && $PY experiments/transfer/summarize.py runs/transfer > runs/transfer/summary.md 2>/dev/null && mkdir -p results/transfer && cp runs/transfer/*.json runs/transfer/summary.md results/transfer/ 2>/dev/null || true
echo "== ablations + figures + tables + pdf"
$PY tools/ablation_table.py --root results/pod_runs --out "$OUT/ablations.md" > /dev/null
$PY paper/scripts/make_paper_figures.py --results "$OUT" --runs results/pod_runs 2>&1 | grep -v Warning | tail -1
$PY tools/export_latex_tables.py --results "$OUT" --out paper/tables
$PY paper/scripts/draw_architecture.py > /dev/null
mkdir -p paper/figures/ts   # only architecture.png is used by the tex (draw_architecture.py writes it); all other figures stay in $OUT/figures
bash paper/fmts/build.sh fmts
cp paper/fmts/fmts.pdf "$OUT/fmts.pdf"
echo "== done: $OUT"
