#!/bin/bash
# Rules compared at equal measured Taylor remainder (the "sphere" test): hypergradient sign push vs the derived allocation,
# each with a unit-rms direction and the per-layer length set by the KronQ2 remainder controller at target kappa, at width
# 4096, both targets, Adam-memory blocks of 256 (+ compressed state), fast path. Jobs -> GPU i % 4 (3 per GPU).
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_sphere2; mkdir -p $F
B="--opt kronqf --width 4096 --depth 2 --steps 5000 --batch 256 --eval_every 250 --diag 1 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --eig_dtype float64 --reltol 1e-12 --lr 3e-3 --refresh 10 --adapt_refresh 0.1 --check_every 10"
JOBS=()
for p in 2 0; do
  for k in 0.03 0.1 0.3; do JOBS+=("$B --p $p --hyper 1 --sphere $k"); done
  for k in 0.03 0.1 0.3; do JOBS+=("$B --p $p --alloc 1 --hyper 0 --sphere $k"); done
done
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=2 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1))
done
wait; echo "ALLDONE $(date +%T)"
