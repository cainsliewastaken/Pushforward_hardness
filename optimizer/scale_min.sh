#!/bin/bash
# Minimal-method test (OPTIMIZER_THEORY.md section 0, 2026-10-13): the three ingredients (basis, sign of the momentum, learned
# per-mode magnitude) plus the output trust region as safety only. A = reference (cap on, budget 0.2); B = cap off, budget 0.2;
# C = cap off, budget 0.5 (safety only). Both field targets, seeds 1-3. Prediction: B = A within the seed spread (1.09x); C = A
# or mildly worse early. Null (the cap covers for something): B worse than A by > 1.2x on rough.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_min; mkdir -p $F
C="--width 4096 --depth 2 --batch 256 --eval_every 250 --tf32 1 --steps 5000"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --lr 1e-3"
JOBS=()
for seed in 1 2 3; do for p in 2 0; do
  JOBS+=("$C $K --p $p --seed $seed --cap 1 --rem 0.2")
  JOBS+=("$C $K --p $p --seed $seed --cap 0 --rem 0.2")
  JOBS+=("$C $K --p $p --seed $seed --cap 0 --rem 0.5")
done; done
echo "${#JOBS[@]} jobs" > $F/njobs
NODE=${1:-0}; NN=${2:-1}; i=0; k=0
for job in "${JOBS[@]}"; do
  if [ $((i % NN)) -eq $NODE ]; then
    CUDA_VISIBLE_DEVICES=$((k % 4)) OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
    k=$((k+1)); if [ $((k % 4)) -eq 0 ]; then wait; fi
  fi
  i=$((i+1))
done
wait; echo "NODE $NODE DONE $(date +%T)"
