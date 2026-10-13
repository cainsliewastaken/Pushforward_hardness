#!/bin/bash
# Collapse diagnosis at width 4096 on the rough target (REPORT 61 item 4): six KronQ runs with the diagnostics on,
# spread over the 4 GPUs of one node (jobs 0-5 -> GPU i % 4). usage (inside srun, one task on the node): scale_diag.sh
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_diag; mkdir -p $F
BASE="--opt kronq --lr 1e-3 --width 4096 --depth 2 --p 2 --steps 5000 --batch 256 --refresh 20 --eval_every 250 --diag 1"
JOBS=(
  "--block 256 --hyper 0"
  "--block 256 --u_clamp 1"
  "--block 256 --beta_u 0.001"
  "--block 256"
  ""
  "--max_dim 1024 --hyper 0"
)
for i in "${!JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=2 python -u "$HERE/scale_study.py" $BASE ${JOBS[$i]} --out "$F/res_j$i.jsonl" 2>&1 | grep -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
done
wait; echo "ALLDONE $(date +%T)"
