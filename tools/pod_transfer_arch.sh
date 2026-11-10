#!/usr/bin/env bash
# Transfer study with a different corrector architecture (recipe is architecture-agnostic?). Runs on the POD.
#   cd /workspace/FoilCORE && nohup bash tools/pod_transfer_arch.sh transformer > runs/transfer_transformer.log 2>&1 &
set -uo pipefail
cd /workspace/FoilCORE; export PYTHONPATH=/workspace/FoilCORE
CORR="${1:-transformer}"
for ds in ett air ecg; do for proto in gtmix draft_causal draft_bidir draft_dual; do for seed in 0 1 2; do
  out="runs/transfer/${ds}_${proto}_${CORR}_s${seed}.json"; [ -f "$out" ] && continue
  python experiments/transfer/train.py --dataset "$ds" --protocol "$proto" --corrector "$CORR" --seed "$seed" --out "$out" 2>&1 | grep -E "TEST|FAILED|Error|Traceback"
done; done; done
echo "[transfer-$CORR] all done $(date -u)"
