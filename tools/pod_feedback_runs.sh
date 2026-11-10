#!/usr/bin/env bash
# Everything the reviewer feedback asked for that needs a GPU, in one detached script. Runs on the POD.
#
#   cd /workspace/FoilCORE && nohup bash tools/pod_feedback_runs.sh > runs/feedback.log 2>&1 &
#
# 1. Standard time-series backbones on the airfoil task, same front-end and protocol as FoilCORE-5k:
#    variant 80 transformer (PatchTST, patch 1), 81 TSMixer/TinyTimeMixer, 82 Mamba — 2 seeds each.
# 2. Transfer of the training recipe to three public domains (ETTh1, Beijing PM2.5, MIT-BIH ECG):
#    4 protocols × 3 seeds each; small models, minutes per run.
# 3. GPU energy per polar for FoilCORE-5k (repeat of the paper number, needs an idle GPU → runs last).
# Outputs: runs/<stamp>_v8x/ (fast_train layout), runs/transfer/*.json, runs/feedback_summary.json
set -uo pipefail
cd /workspace/FoilCORE
export PYTHONPATH=/workspace/FoilCORE
STAMP="$(date -u +%Y%m%d_%H%M%S)"
mkdir -p runs/transfer
echo "[feedback] start $STAMP"; nvidia-smi --query-gpu=name --format=csv,noheader

# ---- 1. baselines (concurrent on one GPU; each ~10 s/epoch, early stopping)
bash tools/launch_runs.sh "80 81 82" --compile --batch-size 1024 --val-batch-size 1024 --epochs 300 --early-stop-patience 12
sleep 5
RUN_ROOT=runs bash tools/launch_runs.sh "80 82" --compile --batch-size 1024 --val-batch-size 1024 --epochs 300 --early-stop-patience 12 --seed 1
echo "[feedback] baselines launched"; python tools/runs_status.py || true

# ---- 2. transfer (sequential, small; shares the GPU with the baselines)
python experiments/transfer/datasets.py || true
for ds in ett air ecg; do
  for proto in gtmix draft_causal draft_bidir draft_dual; do
    for seed in 0 1 2; do
      out="runs/transfer/${ds}_${proto}_s${seed}.json"
      [ -f "$out" ] && continue
      python experiments/transfer/train.py --dataset "$ds" --protocol "$proto" --seed "$seed" --out "$out" 2>&1 | grep -E "TEST|params|FAILED|Error" || echo "[feedback] $ds $proto s$seed failed"
    done
  done
done
python experiments/transfer/summarize.py runs/transfer > runs/transfer/summary.md 2>&1 || true
echo "[feedback] transfer done"

# ---- wait for the baseline runs to finish
while pgrep -f "src.fast_train" > /dev/null; do sleep 60; done
python tools/runs_status.py || true

# ---- 3. GPU energy (idle GPU now)
python tools/energy_gpu.py --ckpts v61=checkpoints/v61/foilcore_best.pt v70=checkpoints/v70/foilcore_best.pt --out runs/energy_gpu_feedback.json || true
echo "[feedback] all done $(date -u +%Y%m%d_%H%M%S)"
