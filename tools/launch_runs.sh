#!/usr/bin/env bash
# Launch several FoilCORE variants concurrently on one GPU with full per-run logging.
#
#   bash tools/launch_runs.sh "50 51 52" [extra fast_train args...]
#
# Each run gets runs/<UTC-stamp>_v<ID>/ with: train.log, metrics_history.json (per-epoch causal-AR +
# TF + stall metrics), checkpoints, code_snapshot.tar.gz (src/ tools/), env.txt (pip freeze, GPU, args).
set -euo pipefail
VARIANTS="$1"; shift || true
ROOT="${RUN_ROOT:-runs}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$ROOT"
for V in $VARIANTS; do
  DIR="$ROOT/${STAMP}_v${V}"
  mkdir -p "$DIR"
  tar czf "$DIR/code_snapshot.tar.gz" src tools 2>/dev/null || true
  {
    echo "variant=$V"; echo "args=$*"; echo "started_utc=$STAMP"; echo "host=$(hostname)";
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
    python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"
    pip freeze 2>/dev/null | grep -Ei "^(torch|numpy|pandas|scipy)=" || true
  } > "$DIR/env.txt"
  nohup python -u -m src.fast_train --variant-id "$V" --save-dir "$DIR" --tag "$STAMP" "$@" > "$DIR/train.log" 2>&1 &
  echo "launched variant $V -> $DIR (pid $!)"
done
