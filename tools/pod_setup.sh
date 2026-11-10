#!/usr/bin/env bash
# Set up a fresh RunPod for the feedback runs. Runs on the LAPTOP.
#
#   bash tools/pod_setup.sh <host> <port>            e.g. bash tools/pod_setup.sh 213.173.107.104 12227
#
# Before this, in the RunPod WEB CONSOLE (so the laptop key can ssh/rsync in):
#   mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGpsg/IEYlRZQbbj4tY8CYXBihuVg6Gzqf7pVhGirCxx avnehb@gmail.com' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
# The pod must expose a direct TCP SSH port (RunPod "SSH over exposed TCP"); the ssh.runpod.io proxy has no rsync.
set -euo pipefail
HOST="${1:?host}"; PORT="${2:?port}"
KEY="$HOME/.ssh/id_ed25519"
SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new"
POD="root@$HOST"
cd "$(dirname "$0")/.."

echo "== ssh check"; $SSH $POD 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; python -c "import torch;print(torch.__version__, torch.cuda.is_available())"'
echo "== code"
$SSH $POD 'mkdir -p /workspace/FoilCORE/data /workspace/FoilCORE/runs'
rsync -rlt -e "$SSH" --exclude __pycache__ src tools experiments checkpoints requirements.txt $POD:/workspace/FoilCORE/
echo "== data (CSV ~1.3 GB; the pod tensorises them on first use)"
rsync -rlt --progress -e "$SSH" data/train.csv data/val.csv data/test.csv $POD:/workspace/FoilCORE/data/
echo "== python deps"
$SSH $POD 'cd /workspace/FoilCORE && pip install -q --break-system-packages -r requirements.txt wfdb 2>&1 | tail -2; python -c "import neuralfoil, wfdb, pandas; print(\"deps ok\")"'
echo "== done. Launch the runs with:"
echo "   $SSH $POD 'cd /workspace/FoilCORE && nohup bash tools/pod_feedback_runs.sh > runs/feedback.log 2>&1 &'"
echo "   pull later:  bash tools/pod_pull.sh $HOST $PORT"
