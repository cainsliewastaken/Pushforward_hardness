#!/bin/bash
# Step 3 of the divergence fix (2026-10-13): the held-out-capped budget, budget_t = min(rem, kappa_max x 2 c_t) with c_t the windowed
# held-out alignment (slow_adapt --sa_cap 1), budget relative to min(current, initial residual) (--rem_anchor min).
# Field problem: both targets, 3 seeds, 5000 steps, reference config (cap off, budget 0.2), kappa_max 0.75 / 1.0; must stay within ~10 % of
# item 43's A (0.0634 rough, 1.385e-4 smooth). Turbulence: all 400000 pairs, 20 epochs, batch 256, lr 1e-3, budget 0.15, both targets, kappa_max
# 0.75 / 1.0, plus kappa 1.0 with weight decay 0.01; must not degrade (AdamW: 0.156 increment, 0.066 direct). 18 runs.  usage: NODE NNODES
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_capfix; mkdir -p $F
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --lr 1e-3 --slow_adapt 1 --sa_cap 1 --rem_anchor min"
T="--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_400k.npz --turb_cpu 1 --turb_onepass 1 --width 4096 --depth 2 --eval_every 250 --tf32 1 --nval 4096 --seed 1 --batch 256 --steps 31240 --rem 0.15"
Fd="--width 4096 --depth 2 --batch 256 --eval_every 250 --tf32 1 --steps 5000 --rem 0.2"
JOBS=()
for tg in delta direct; do
  for km in 0.75 1.0; do JOBS+=("$T $K --turb_target $tg --sa_kappa_max $km"); done
  JOBS+=("$T $K --turb_target $tg --sa_kappa_max 1.0 --wd 0.01")
done
for seed in 1 2 3; do for p in 2 0; do for km in 0.75 1.0; do JOBS+=("$Fd $K --p $p --seed $seed --sa_kappa_max $km"); done; done; done
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
