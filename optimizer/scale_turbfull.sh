#!/bin/bash
# Full training run on the 2D turbulence one-step problem (user's request, 2026-10-13): all 400000 pairs at 64 x 64 (cache
# turb64_lead10_400k.npz, CPU-resident, sampled without replacement epoch by epoch), 20 epochs, both targets (increment = rough,
# direct = smooth control), batch 256 (31240 steps) and 1024 (7810 steps), seeds 1-2. Minimal KronQ (cap off) lr 1e-3 at two budgets
# (256: 0.15 / 0.3; 1024: 0.3 / 0.6, the batch law of item 46); AdamW lr {1e-4, 3e-4, 1e-3} and AdamW lr 3e-4 with weight decay 0.1.
# Validation every 250 steps on 4096 held-out pairs from the test region; val_best beside val_final; passes logged.
# usage: scale_turbfull.sh NODE NNODES
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_turbfull; mkdir -p $F
C="--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_400k.npz --turb_cpu 1 --turb_onepass 1 --width 4096 --depth 2 --eval_every 250 --tf32 1 --nval 4096"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --lr 1e-3"
JOBS=()
for seed in 1 2; do for tg in delta direct; do
  for rem in 0.15 0.3; do JOBS+=("$C $K --batch 256 --steps 31240 --seed $seed --turb_target $tg --rem $rem"); done
  for rem in 0.3 0.6; do JOBS+=("$C $K --batch 1024 --steps 7810 --seed $seed --turb_target $tg --rem $rem"); done
  for lr in 1e-4 3e-4 1e-3; do
    JOBS+=("$C --opt adamw --batch 256 --steps 31240 --seed $seed --turb_target $tg --lr $lr")
    JOBS+=("$C --opt adamw --batch 1024 --steps 7810 --seed $seed --turb_target $tg --lr $lr")
  done
  JOBS+=("$C --opt adamw --batch 256 --steps 31240 --seed $seed --turb_target $tg --lr 3e-4 --wd 0.1")
  JOBS+=("$C --opt adamw --batch 1024 --steps 7810 --seed $seed --turb_target $tg --lr 3e-4 --wd 0.1")
done; done
echo "${#JOBS[@]} jobs" > $F/njobs
NODE=${1:-0}; NN=${2:-1}; i=0; k=0
for job in "${JOBS[@]}"; do
  if [ $((i % NN)) -eq $NODE ]; then
    CUDA_VISIBLE_DEVICES=$((k % 4)) OMP_NUM_THREADS=8 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
    k=$((k+1)); if [ $((k % 4)) -eq 0 ]; then wait; fi
  fi
  i=$((i+1))
done
wait; echo "NODE $NODE DONE $(date +%T)"
