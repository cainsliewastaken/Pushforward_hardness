#!/bin/bash
# Item 58 (2026-10-14): the late spikes on the field (critical-batch scan, xfold session: AdamW and KronQ both spike once the loss is low,
# more with steps, lr and batch; sharpening at a fixed step). None of those runs had decay. Field rough, batch 1024, budget 0.3, lr 1e-3,
# 10000 steps (the scan's run with 5 spikes), seed 1:  a. reference (wd 0);  b. uniform decay 3e-4;  c. uniform 1e-3;  d. gated 1e-3;
#   e. cap on, wd 0;  f. cap on + uniform 1e-3.   Predictions: decay reduces the spike count if the sharpening is weight-norm driven; the cap
# removes them if they are per-mode overshoot (the theory's bound); both nulls are informative. 6 runs, ~6.5 min each, one node.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_fieldspike; mkdir -p $F
C="--width 4096 --depth 2 --batch 1024 --eval_every 125 --tf32 1 --steps 10000 --seed 1 --p 2 --nval 4096"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --lr 1e-3 --rem 0.3"
JOBS=()
JOBS+=("$C $K --cap 0")
JOBS+=("$C $K --cap 0 --wd_abs 3e-4")
JOBS+=("$C $K --cap 0 --wd_abs 1e-3")
JOBS+=("$C $K --cap 0 --wd_abs 1e-3 --wd_gate 1 --wd_gate_q0 0.5")
JOBS+=("$C $K --cap 1")
JOBS+=("$C $K --cap 1 --wd_abs 1e-3")
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
