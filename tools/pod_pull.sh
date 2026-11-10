#!/usr/bin/env bash
# Pull the feedback runs back from the pod. Runs on the LAPTOP.
#   bash tools/pod_pull.sh <host> <port>
set -euo pipefail
HOST="${1:?host}"; PORT="${2:?port}"
SSH="ssh -p $PORT -i $HOME/.ssh/id_ed25519"; POD="root@$HOST"
cd "$(dirname "$0")/.."
mkdir -p results/pod_runs results/transfer
rsync -rlt --include='*/' --include='foilcore_best.pt' --include='metrics_history*.json' --include='env.txt' --include='train*.log' --exclude='*' -e "$SSH" $POD:/workspace/FoilCORE/runs/ results/pod_runs/
rsync -rlt -e "$SSH" $POD:/workspace/FoilCORE/runs/transfer/ results/transfer/
rsync -rlt -e "$SSH" $POD:/workspace/FoilCORE/runs/energy_gpu_feedback.json results/ 2>/dev/null || true
rsync -rlt -e "$SSH" $POD:/workspace/FoilCORE/runs/feedback.log results/feedback.log 2>/dev/null || true
echo "== baseline runs:"; ls -d results/pod_runs/*_v8[0-2] 2>/dev/null || echo "(none yet)"
echo "== transfer:"; ls results/transfer | head; cat results/transfer/summary.md 2>/dev/null || true
