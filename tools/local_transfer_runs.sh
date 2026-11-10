#!/usr/bin/env bash
# Transfer suite only (CPU is enough): 3 datasets × 4 protocols × 3 seeds.
#   nohup bash tools/local_transfer_runs.sh > runs/transfer_local.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" FOILCORE_DEVICE="${FOILCORE_DEVICE:-cpu}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
PY="${PY:-python}"
mkdir -p runs/transfer
for ds in ${DATASETS:-ett air ecg}; do for proto in gtmix draft_causal draft_bidir draft_dual; do for seed in 0 1 2; do
  out="runs/transfer/${ds}_${proto}_s${seed}.json"; [ -f "$out" ] && continue
  $PY experiments/transfer/train.py --dataset "$ds" --protocol "$proto" --seed "$seed" --out "$out" 2>&1 | grep -E "TEST|FAILED|Error|Traceback"
done; done; done
$PY experiments/transfer/summarize.py runs/transfer > runs/transfer/summary.md; mkdir -p results/transfer && cp runs/transfer/*.json runs/transfer/summary.md results/transfer/
echo "[transfer] all done $(date)"
