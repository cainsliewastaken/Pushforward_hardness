#!/bin/bash
# Item 55 (2026-10-13): the budget with strong decay present. Every turbulence result since item 50 ran at the held-out cap's FLOOR 0.03
# (a constant chosen, not derived); the decay per step lr wd ran at lr 1e-3 while the budget cut the step 20x. Here: fixed budget, no
# slow adaptation, 400k increment, wd 1.0, seed 1: budget 0.03 (predicted = the capped 0.111), 0.015 / 0.06 (an optimum in range),
# 0.15 (predicted: no divergence now that the decay bounds the weight diffusion; worse than 0.03; null = item 48's divergence returns),
# 0.03 with --diag 1 (weight quantiles, norms), and direct at 0.03 wd 0.3 (predicted = the capped 0.063). 6 runs, 7.5 min each.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_budget; mkdir -p $F
T="--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_400k.npz --turb_cpu 1 --turb_onepass 1 --width 4096 --depth 2 --eval_every 250 --tf32 1 --nval 4096 --batch 256 --steps 31240 --seed 1"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --lr 1e-3 --rem_anchor min"
JOBS=()
for rem in 0.03 0.015 0.06 0.15; do JOBS+=("$T $K --turb_target delta --wd 1.0 --rem $rem"); done
JOBS+=("$T $K --turb_target delta --wd 1.0 --rem 0.03 --diag 1")
JOBS+=("$T $K --turb_target direct --wd 0.3 --rem 0.03")
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
