#!/bin/bash
# Refresh-cost tests: Cholesky-QR2 subspace iteration (default now), refresh floor 5 vs 20, S side every 4th refresh +
# fixed input-layer A basis (T3). One run per GPU (isolated wall). Reference (batched QR, floor 5): 0.0612 / 1.41e-4 at
# 70 ms per step; AdamW 3.3-6.6 ms.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_fastref; mkdir -p $F
K="--opt kronqf --width 4096 --depth 2 --batch 256 --steps 5000 --eval_every 250 --diag 1 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --reltol 1e-12 --lr 1e-3 --rem 0.2 --cap 1 --hyper 1"
JOBS=("$K --p 2 --refresh_min 5" "$K --p 2 --refresh_min 20" "$K --p 2 --refresh_min 20 --S_every 4 --A0_fixed 1" "$K --p 0 --refresh_min 20")
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1))
done
wait; echo "ALLDONE $(date +%T)"
