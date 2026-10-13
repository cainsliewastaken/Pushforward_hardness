#!/bin/bash
# Momentum window test: reference (beta 0.9) vs longer global windows (0.97, 0.99) vs the per-mode window from the smoothed
# online SNR (--mom_adapt, T_max 200), with and without the learned per-mode weight; width 4096, both targets, Adam
# memory, fast configuration. 10 runs, 3 per GPU.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_momadapt; mkdir -p $F
K="--opt kronqf --width 4096 --depth 2 --steps 5000 --batch 256 --eval_every 250 --diag 1 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --lr 1e-3 --rem 0.2 --cap 1"
JOBS=()
for p in 2 0; do
  JOBS+=("$K --p $p --hyper 1")
  JOBS+=("$K --p $p --hyper 1 --b1 0.97")
  JOBS+=("$K --p $p --hyper 1 --b1 0.99")
  JOBS+=("$K --p $p --hyper 1 --mom_adapt 1 --mom_T_max 200")
  JOBS+=("$K --p $p --hyper 0 --mom_adapt 1 --mom_T_max 200")
done
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=2 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1))
done
wait; echo "ALLDONE $(date +%T)"
