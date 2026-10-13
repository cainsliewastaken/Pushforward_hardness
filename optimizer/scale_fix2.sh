#!/bin/bash
# Scaling criterion retest with the step budget (REPORT 61 item 4): the Adam-memory variant and the two other failing
# reductions at width 4096 on both targets; jobs 0-7 -> GPU i % 4. usage (inside srun, one task on the node): scale_fix2.sh
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_fix2; mkdir -p $F
BASE="--opt kronq --lr 1e-3 --width 4096 --depth 2 --steps 5000 --batch 256 --refresh 20 --eval_every 250 --diag 1"
JOBS=(
  "--p 2 --block 256 --compress 1 --mom_bf16 1 --rem 0.1"
  "--p 2 --block 256 --compress 1 --mom_bf16 1 --gnls 1"
  "--p 2 --block 64 --rem 0.1"
  "--p 2 --max_dim 1024 --rem 0.1"
  "--p 0 --block 256 --compress 1 --mom_bf16 1 --rem 0.1"
  "--p 0 --block 256 --compress 1 --mom_bf16 1 --gnls 1"
  "--p 0 --block 64 --rem 0.1"
  "--p 0 --max_dim 1024 --rem 0.1"
)
for i in "${!JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=2 python -u "$HERE/scale_study.py" $BASE ${JOBS[$i]} --out "$F/res_j$i.jsonl" 2>&1 | grep -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
done
wait; echo "ALLDONE $(date +%T)"
