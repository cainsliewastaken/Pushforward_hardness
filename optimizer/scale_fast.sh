#!/bin/bash
# Fast KronQ (reparametrised, all options) at width 4096: phase 1 = isolated timing (one run per GPU, 600 steps);
# phase 2 = accuracy, both targets, Adam-memory blocks of 256 (3 per GPU). usage (inside srun): scale_fast.sh
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_fast; mkdir -p $F
COMMON="--width 4096 --depth 2 --batch 256 --block 256 --compress 1 --mom_bf16 1 --lr 1e-3"
T1=(
  "--opt kronq  --p 2 --steps 600 --refresh 20 --eval_every 300 $COMMON"
  "--opt kronqf --p 2 --steps 600 --refresh 20 --eval_every 300 $COMMON"
  "--opt kronqf --p 2 --steps 600 --refresh 20 --eval_every 300 --tf32 1 $COMMON"
  "--opt kronqf --p 2 --steps 600 --refresh 20 --eval_every 300 --tf32 1 --eig_iters 2 --adapt_refresh 0.1 $COMMON"
)
i=0
for job in "${T1[@]}"; do
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=2 python -u "$HERE/scale_study.py" $job --out "$F/time_j$i.jsonl" 2>&1 | grep -v -i 'pynvml\|FutureWarning' > "$F/time_log_j$i.txt" &
  i=$((i+1))
done
wait; echo "TIMING DONE $(date +%T)"
ACC="--opt kronqf --steps 5000 --refresh 20 --eval_every 250 --diag 1 --tf32 1 $COMMON"
A=(
  "--rem 0.2"
  "--lr_kappa 0.1"
  "--rule adam"
  "--rule adam --ekfac 1"
  "--ekfac 1 --gnls 1"
  "--stats_every 1 --eig_iters 2 --adapt_refresh 0.1 --rem 0.2"
)
i=0
for p in 2 0; do
  for extra in "${A[@]}"; do
    CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=2 python -u "$HERE/scale_study.py" $ACC --p $p $extra --out "$F/res_j$i.jsonl" 2>&1 | grep -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
    i=$((i+1))
  done
done
wait; echo "ALLDONE $(date +%T)"
