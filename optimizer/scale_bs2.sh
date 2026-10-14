#!/bin/bash
# Batch-size scaling, second pass (REPORT 61 item 44 b, d): (1) resolve the exponent of the noise-set constant with a finer grid:
# rough: lr 1e-3, budget {0.2, 0.4, 0.8} at batch 512 and 2048; smooth: budget 0.3, lr {1.4e-3, 2.8e-3, 5.6e-3} at 512 and 2048;
# (2) AdamW at batch 1024 (lr {1e-4, 3e-4, 1e-3}, 5000 and 29000 steps) for the equal-wall comparison with KronQ at batch 1024
# (item 44: rough 0.0098, smooth 5.85e-5). Seeds 1-2. 36 runs.  usage: scale_bs2.sh NODE NNODES
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_bs2; mkdir -p $F
C="--width 4096 --depth 2 --eval_every 250 --tf32 1 --steps 5000"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0"
JOBS=()
for seed in 1 2; do
  for bs in 512 2048; do
    for rem in 0.2 0.4 0.8; do JOBS+=("$C $K --batch $bs --p 2 --seed $seed --lr 1e-3 --rem $rem"); done
    for lr in 1.4e-3 2.8e-3 5.6e-3; do JOBS+=("$C $K --batch $bs --p 0 --seed $seed --lr $lr --rem 0.3"); done
  done
  for p in 2 0; do for lr in 1e-4 3e-4 1e-3; do for st in 5000 29000; do JOBS+=("$C --opt adamw --batch 1024 --p $p --seed $seed --lr $lr --steps $st"); done; done; done
done
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
