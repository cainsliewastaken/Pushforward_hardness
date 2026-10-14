#!/bin/bash
# Item 57 (2026-10-14): the method without the output budget (and without the cap): basis + sign of momentum + learned weights + lr with
# linear decay (+ weight decay on replayed data). No field run has ever used the preset with rem 0. The budget held the step at 0.15-0.2 of
# the residual on the field where the raw step at lr 1e-3 is 0.67 and the collapse edge is near 0.3, so lr ~ 2e-4 is the expected working
# point at batch 256. Field both targets x lr {1e-4, 2e-4, 4e-4}, 5000 steps, no decay (fresh data); turbulence 400k increment, 20 epochs,
# lr {1e-4, 2e-4, 4e-4}, decay 1e-3 per step as its own constant (--wd_abs). Predictions: field best lr within 20 % of 0.062 / 1.4e-4 with
# the top lr collapsing to the mean; turbulence best lr within 15 % of 0.111, the lowest lr under-fitting (too small a push against the
# decay, item 55's collapse edge). A loss > 20 % at every field lr = the budget's shrink with the residual does real work.  9 runs, 1 node.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_nobudget; mkdir -p $F
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --rem 0"
T="--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_400k.npz --turb_cpu 1 --turb_onepass 1 --width 4096 --depth 2 --eval_every 250 --tf32 1 --nval 4096 --batch 256 --steps 31240 --seed 1 --turb_target delta --wd_abs 1e-3"
Fd="--width 4096 --depth 2 --batch 256 --eval_every 250 --tf32 1 --steps 5000 --seed 1"
JOBS=()
for lr in 1e-4 2e-4 4e-4; do JOBS+=("$T $K --lr $lr"); done
for p in 2 0; do for lr in 1e-4 2e-4 4e-4; do JOBS+=("$Fd $K --p $p --lr $lr"); done; done
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
