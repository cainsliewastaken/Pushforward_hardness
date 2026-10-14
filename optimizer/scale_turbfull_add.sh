#!/bin/bash
# Addendum to scale_turbfull.sh: KronQ at batch 1024 with budgets 0.1 / 0.15 (the batch-law values 0.3 / 0.6 diverged on the increment:
# the budget ceiling on this problem is 0.15-0.3 at batch 256, item 45). Both targets, seeds 1-2, 7810 steps (20 epochs). 8 runs.
# usage: scale_turbfull_add.sh NODE NNODES  (indices offset by 100 so they do not collide with the main set)
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_turbfull; mkdir -p $F
C="--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_400k.npz --turb_cpu 1 --turb_onepass 1 --width 4096 --depth 2 --eval_every 250 --tf32 1 --nval 4096"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --lr 1e-3"
JOBS=()
for seed in 1 2; do for tg in delta direct; do for rem in 0.1 0.15; do JOBS+=("$C $K --batch 1024 --steps 7810 --seed $seed --turb_target $tg --rem $rem"); done; done; done
NODE=${1:-0}; NN=${2:-1}; i=0; k=0
for job in "${JOBS[@]}"; do
  if [ $((i % NN)) -eq $NODE ]; then
    CUDA_VISIBLE_DEVICES=$((k % 4)) OMP_NUM_THREADS=8 python -u "$HERE/scale_study.py" $job --out "$F/res_j$((100 + i)).jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$((100 + i)).txt" &
    k=$((k+1)); if [ $((k % 4)) -eq 0 ]; then wait; fi
  fi
  i=$((i+1))
done
wait; echo "NODE $NODE ADD DONE $(date +%T)"
