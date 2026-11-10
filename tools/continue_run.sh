#!/usr/bin/env bash
# Continue a finished fast_train run until early stopping (val causal-AR loss).
#   bash tools/continue_run.sh runs/<dir> [max_epochs=400] [patience=12]
# Waits for the run's "[fast_train] done" line, then resumes from its last checkpoint at the LR it
# ended with, appending to metrics_history.json and logging to train_continue.log.
set -euo pipefail
DIR="$1"; MAX="${2:-400}"; PAT="${3:-12}"
until grep -q "\[fast_train\] done" "$DIR/train.log" 2>/dev/null; do sleep 60; done
VID=$(python -c "import json;print(json.load(open('$DIR/metrics_history.json'))['variant']['id'])")
LR=$(python -c "import json;print(json.load(open('$DIR/metrics_history.json'))['history'][-1]['lr'])")
LAST=$(python -c "import json;print(json.load(open('$DIR/metrics_history.json'))['history'][-1]['epoch'])")
CK="$DIR/foilcore_last.pt"; [ -f "$CK" ] || CK=$(ls "$DIR"/foilcore_epoch_*.pt | sort | tail -1)
cp "$DIR/metrics_history.json" "$DIR/metrics_history_first_$LAST.json"
echo "continuing variant $VID from $CK (epoch $LAST, lr $LR) to max $MAX epochs, patience $PAT"
exec python -u -m src.fast_train --variant-id "$VID" --save-dir "$DIR" --resume "$CK" \
  --resume-history "$DIR/metrics_history_first_$LAST.json" --lr "$LR" --warmup-epochs 0 \
  --epochs "$MAX" --early-stop-patience "$PAT" --compile --batch-size 1024 --val-batch-size 1024 \
  --tag "continue_from_$LAST" > "$DIR/train_continue.log" 2>&1
