#!/usr/bin/env bash
# Laptop (MPS) fallback for tools/pod_feedback_runs.sh: baselines sequentially, then transfer suite.
#   nohup bash tools/local_feedback_runs.sh > runs/feedback_local.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" FOILCORE_DEVICE="${FOILCORE_DEVICE:-mps}"
PY="${PY:-python}"
mkdir -p runs/transfer
for V in 80 82 81; do
  DIR="runs/local_v${V}"; [ -f "$DIR/foilcore_best.pt" ] && [ -f "$DIR/DONE" ] && continue
  mkdir -p "$DIR"
  echo "[local] variant $V -> $DIR $(date)"
  $PY -u -m src.fast_train --variant-id "$V" --save-dir "$DIR" --batch-size 1024 --val-batch-size 2048 --epochs 150 --early-stop-patience 10 > "$DIR/train.log" 2>&1 && touch "$DIR/DONE"
done
for ds in ett air ecg; do for proto in gtmix draft_causal draft_bidir draft_dual; do for seed in 0 1 2; do
  out="runs/transfer/${ds}_${proto}_s${seed}.json"; [ -f "$out" ] && continue
  $PY experiments/transfer/train.py --dataset "$ds" --protocol "$proto" --seed "$seed" --out "$out" 2>&1 | grep -E "TEST|FAILED|Error"
done; done; done
$PY experiments/transfer/summarize.py runs/transfer > runs/transfer/summary.md; mkdir -p results/transfer && cp runs/transfer/*.json runs/transfer/summary.md results/transfer/
echo "[local] all done $(date)"
