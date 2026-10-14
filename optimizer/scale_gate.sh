#!/bin/bash
# Item 57 (2026-10-14): the decay/weight/budget fight. Item 55: under uniform decay the learned weights inflate 2-8x to hold the shared
# fit, the proposed step is 3.5-4x the residual and the budget undoes 99 % of it. Fixes: (1) --wd_gate: decay per mode in the eigenbasis
# at rate gamma (1 - s_k), s_k the mode's momentum SNR over q0 (a supported mode is not decayed); (2) --lr_track: lr follows the budget.
# 400k increment, 20 epochs, budget 0.03, decay 1e-3 per step (= wd 1.0 at lr 1e-3; reference item 55: 0.1196, u medians 0.8-2.0, raw 3.5-4):
#   a. gated q0 0.5;  b. gated q0 0.25;  c. uniform + lr_track;  d. gated + lr_track;  e. = a with --diag 1;  f. direct, gated, gamma 3e-4
#   (uniform decay collapsed direct at budget 0.03, item 55).
# Predictions: a within 10 % of 0.12 with u medians back near 0-0.7 and raw change near 0.7 (the fight gone); null: the gate cannot separate
# (momentum SNR of signal 0.3 vs noise 0.23 on the field) and a behaves like no decay (0.195). c: same loss, raw near the budget, lr_scale ~ 0.01,
# u pinned at the clamp (the inflation moves, it does not go). f: no collapse.  6 runs, 7.5 min each, one node.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_gate; mkdir -p $F
T="--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_400k.npz --turb_cpu 1 --turb_onepass 1 --width 4096 --depth 2 --eval_every 250 --tf32 1 --nval 4096 --batch 256 --steps 31240 --seed 1"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --lr 1e-3 --rem_anchor min --rem 0.03"
JOBS=()
JOBS+=("$T $K --turb_target delta --wd_abs 1e-3 --wd_gate 1 --wd_gate_q0 0.5")
JOBS+=("$T $K --turb_target delta --wd_abs 1e-3 --wd_gate 1 --wd_gate_q0 0.25")
JOBS+=("$T $K --turb_target delta --wd_abs 1e-3 --lr_track 1")
JOBS+=("$T $K --turb_target delta --wd_abs 1e-3 --wd_gate 1 --wd_gate_q0 0.5 --lr_track 1")
JOBS+=("$T $K --turb_target delta --wd_abs 1e-3 --wd_gate 1 --wd_gate_q0 0.5 --diag 1")
JOBS+=("$T $K --turb_target direct --wd_abs 3e-4 --wd_gate 1 --wd_gate_q0 0.5")
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
