#!/bin/bash
# Mechanism measurement (mechanism.py): four rules x two targets at width 4096, 5000 steps, diagnostics every 250 steps.
# Fast path in its fast configuration (subspace iteration, refresh floor 5), the rotated Adam rule, the frozen GN ridge,
# AdamW. 8 runs, 2 per GPU. usage (inside srun): scale_mech.sh
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_mech; mkdir -p $F
C="--width 4096 --depth 2 --steps 5000 --batch 256 --eval_every 250 --diag 1 --tf32 1 --mech 1 --mech_every 250"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --lr 1e-3 --rem 0.2"
JOBS=()
for p in 2 0; do
  JOBS+=("$C $K --p $p --hyper 1 --cap 1")
  JOBS+=("$C $K --p $p --rule adam")
  JOBS+=("$C $K --p $p --alloc2 1 --cfac 0 --box 1")
  JOBS+=("$C --opt adamw --lr 3e-4 --p $p")
done
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=2 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1))
done
wait; echo "ALLDONE $(date +%T)"
