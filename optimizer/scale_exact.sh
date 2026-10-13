#!/bin/bash
# Approximation controls (REPORT 61 item 25 plan): width 4096, both targets: (1) reference fast configuration + tracking
# diagnostics; (2) every approximation inside the Kronecker model removed: full factors, fp32 state, float64 eigh every
# >= 20 steps (adaptive), stats every step, + tracking; (3) as (2) with the statistics horizon halved (beta_stats 0.9).
# 6 runs, 2 per GPU (GPUs 0-2). usage (inside srun): scale_exact.sh
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_exact; mkdir -p $F
C="--opt kronqf --width 4096 --depth 2 --steps 5000 --batch 256 --eval_every 250 --diag 1 --tf32 1 --stats_every 1 --adapt_refresh 0.1 --reltol 1e-12 --lr 1e-3 --rem 0.2 --cap 1 --hyper 1 --track 1 --track_every 500"
REF="--block 256 --compress 1 --mom_bf16 1 --eig_iters 2 --refresh 20 --refresh_min 5"
EXACT="--block 0 --compress 0 --mom_bf16 0 --eig_iters 0 --eig_dtype float64 --refresh 20 --refresh_min 20"
JOBS=()
for p in 2 0; do
  JOBS+=("$C $REF --p $p")
  JOBS+=("$C $EXACT --p $p")
  JOBS+=("$C $EXACT --p $p --beta_stats 0.9")
done
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 3)) OMP_NUM_THREADS=2 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1))
done
wait; echo "ALLDONE $(date +%T)"
