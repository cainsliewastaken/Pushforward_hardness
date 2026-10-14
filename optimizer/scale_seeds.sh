#!/bin/bash
# The paper's table: 3 seeds x 2 targets x {KronQ fast (eager), KronQ fast (fused), AdamW, SOAP b256}, width 4096, batch
# 256, 5000 steps, one run per GPU (isolated wall), 4 at a time. ~30 min.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_seeds; mkdir -p $F
C="--width 4096 --depth 2 --batch 256 --steps 5000 --eval_every 250 --tf32 1"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --lr 1e-3 --rem 0.2 --cap 1 --hyper 1"
JOBS=()
for seed in 1 2 3; do for p in 2 0; do
  JOBS+=("$C $K --p $p --seed $seed")
  JOBS+=("$C $K --p $p --seed $seed --fused 1")
  JOBS+=("$C --opt adamw --lr 3e-4 --p $p --seed $seed")
  JOBS+=("$C --opt soap --block 256 --lr 3e-3 --refresh 20 --p $p --seed $seed")
done; done
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1))
  if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait; echo "ALLDONE $(date +%T)"
