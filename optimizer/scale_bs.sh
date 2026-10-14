#!/bin/bash
# Batch-size scaling of the optimal length (REPORT 61 item 44 plan; OPTIMIZER_THEORY 0 / SIGN_PROCESS 2). Minimal method (cap off,
# budget, linear decay) at batch 512 and 1024, lr {1e-3, 2e-3, 4e-3} x budget {0.15, 0.3, 0.6}, both field targets, seeds 1-2,
# 5000 steps (equal steps: the question is where the optimum sits, not the equal-wall comparison). Batch 256 reference grid:
# item 40 (optimum lr 1e-3; budget 0.15 rough / 0.3 smooth; 1.5e-3 diverges at budget >= 0.2).
# Prediction (sign process: drift per step ~ mean sign ~ sqrt(n_B), cost independent of n_B): optimal lr and budget ~ sqrt(n_B),
# i.e. x1.4 at 512 and x2 at 1024. Null: the optimum does not move. Linear (x4 at 1024) would refute the sign-process scaling.
# usage: scale_bs.sh NODE NNODES
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_bs; mkdir -p $F
C="--width 4096 --depth 2 --eval_every 250 --tf32 1 --steps 5000"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0"
JOBS=()
for seed in 1 2; do for p in 2 0; do for bs in 512 1024; do
  for lr in 1e-3 2e-3 4e-3; do for rem in 0.15 0.3 0.6; do JOBS+=("$C $K --batch $bs --p $p --seed $seed --lr $lr --rem $rem"); done; done
done; done; done
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
