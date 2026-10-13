#!/bin/bash
# Hypergradient vs derived allocation under the two-sided controller, Adam-memory KronQ (b256 + compressed state) at
# width 4096, both targets; jobs 0-11 -> GPU i % 4 (3 per GPU). usage (inside srun, one task on the node): scale_alloc.sh
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_alloc; mkdir -p $F
BASE="--opt kronq --lr 1e-3 --width 4096 --depth 2 --steps 5000 --batch 256 --refresh 20 --eval_every 250 --diag 1 --block 256 --compress 1 --mom_bf16 1"
JOBS=(
  "--p 2 --rem 0.1"
  "--p 2 --rem_ctrl 0.1"
  "--p 2 --rem_ctrl 0.03"
  "--p 2 --alloc 1 --rem_ctrl 0.1"
  "--p 2 --alloc 1 --rem_ctrl 0.03"
  "--p 2 --alloc 1 --rem 0.1"
  "--p 0 --rem 0.1"
  "--p 0 --rem_ctrl 0.1"
  "--p 0 --rem_ctrl 0.03"
  "--p 0 --alloc 1 --rem_ctrl 0.1"
  "--p 0 --alloc 1 --rem_ctrl 0.03"
  "--p 0 --alloc 1 --rem 0.1"
)
for i in "${!JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=2 python -u "$HERE/scale_study.py" $BASE ${JOBS[$i]} --out "$F/res_j$i.jsonl" 2>&1 | grep -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
done
wait; echo "ALLDONE $(date +%T)"
