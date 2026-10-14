#!/bin/bash
# Validating job for the pre-flight: (1) pre-flight choices (block / S_mode) + the budget tracking the held-out break-even;
# (2) pre-flight choices + the plain budget 0.2; (3) hand-set choices + the tracking budget. Both targets, one run per GPU.
# Reference (hand-set, budget 0.2): rough 0.062, smooth 1.38e-4 (item 36).
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_preflight; mkdir -p $F
K="--opt kronqf --width 4096 --depth 2 --batch 256 --steps 5000 --eval_every 250 --diag 1 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --lr 1e-3 --cap 1 --hyper 1 --fused 1"
JOBS=("$K --p 2 --preflight 1 --rem 0.2 --rem_val_track 0.8" "$K --p 2 --preflight 1 --rem 0.2" "$K --p 2 --rem 0.2 --rem_val_track 0.8" "$K --p 0 --preflight 1 --rem 0.2 --rem_val_track 0.8" "$K --p 0 --preflight 1 --rem 0.2" "$K --p 0 --rem 0.2 --rem_val_track 0.8")
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1)); if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait; echo "ALLDONE $(date +%T)"
