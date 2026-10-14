#!/bin/bash
# Extensive tests: (1) KronQ fast (fused) lr {5e-4, 1e-3, 1.5e-3} x budget {0.15, 0.2, 0.3}; (2) AdamW lr {1e-4, 3e-4, 1e-3, 3e-3} at
# 5000 and 29000 steps (equal wall); (3) SOAP b256 lr {1e-3, 2e-3, 3e-3}; both targets, seeds 1-3, one run per GPU.
# usage: scale_grid.sh NODE NNODES  (jobs with index % NNODES == NODE; 4 concurrent per node)
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_grid; mkdir -p $F
C="--width 4096 --depth 2 --batch 256 --eval_every 250 --tf32 1"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --cap 1 --hyper 1 --fused 1 --steps 5000"
JOBS=()
for seed in 1 2 3; do for p in 2 0; do
  for lr in 5e-4 1e-3 1.5e-3; do for rem in 0.15 0.2 0.3; do JOBS+=("$C $K --p $p --seed $seed --lr $lr --rem $rem"); done; done
  for lr in 1e-4 3e-4 1e-3 3e-3; do for st in 5000 29000; do JOBS+=("$C --opt adamw --p $p --seed $seed --lr $lr --steps $st"); done; done
  for lr in 1e-3 2e-3 3e-3; do JOBS+=("$C --opt soap --block 256 --refresh 20 --p $p --seed $seed --lr $lr --steps 5000"); done
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
