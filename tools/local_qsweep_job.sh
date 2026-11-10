#!/usr/bin/env bash
# Full sweep: xargs -P 4 -L 1 tools/local_qsweep_job.sh < results/transfer_qsweep/jobs.txt; then copy runs/transfer_qsweep/*.json to results/transfer_qsweep/
# one job: dataset q noise seed  (q=C -> gtmix_consist)
cd "$(dirname "$0")/.."
PY="${PY:-python}"
ds=$1; q=$2; n=$3; s=$4
if [[ $q == C ]]; then
  out=runs/transfer_qsweep/${ds}_consist_s${s}.json
  [[ -f $out ]] || "$PY" experiments/transfer/train.py --dataset $ds --protocol gtmix_consist --seed $s --out $out > ${out%.json}.log 2>&1
else
  out=runs/transfer_qsweep/${ds}_q${q}_n${n}_s${s}.json
  [[ -f $out ]] || "$PY" experiments/transfer/train.py --dataset $ds --protocol draft_dual --q $q --draft-noise $n --seed $s --out $out > ${out%.json}.log 2>&1
fi
