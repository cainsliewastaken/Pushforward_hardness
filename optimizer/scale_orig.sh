#!/bin/bash
# The refactor (kronq_orig.py, --orig 1): weights in ordinary coordinates. CPU check (200 steps): trajectories within 2.4e-4 of KronQFast,
# refresh function change exactly 0. GPU test, all TF32 (mixed precision), guard on:
#  (a) presets, 5000 steps, batch 256, seeds 1-2: rough budget 0.2 (KronQFast 0.0622 / 0.0626), smooth budget 0.3 (1.42e-4 / 1.38e-4);
#  (b) spiking configs: smooth B1024 T10000 lr 2e-3 seeds 1-4 (KronQFast TF32: 4 of 4 spiked; with the guard 3 of 4 small spikes, basis
#      error ~5e-3), rough B1024 T10000 seeds 1-2 (KronQFast TF32 2 of 2 spiked, best 0.0073; fp32 0.0033, no spikes);
#  (c) runtime: rough 5000 steps at batch 1024 and 4096 (KronQFast 59 s / 127 s; AdamW 15 s / 30 s), one run per GPU at a time.
# Predictions: (a) within seed noise of KronQFast; (b) no spikes, refresh change 0, basis error ~1e-7, rough near the fp32 value 0.0033;
# (c) faster than KronQFast at both batches. Null: (b) spikes persist with exact refreshes (then the basis was not the only cause).
F=$SCRATCH/l63_held_rough/runs/scale_orig
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --orig 1 --qr_guard 1 --tf32 1"
PR="--width 4096 --depth 2 --batch 256 --eval_every 250 --steps 5000 --nval 4096 $K --lr 1e-3"
C1="--width 4096 --depth 2 --batch 1024 --eval_every 250 --steps 10000 --p 0 --nval 4096 $K --lr 2e-3 --rem 0.3 --spike_diag 2000"
C3="--width 4096 --depth 2 --batch 1024 --eval_every 125 --steps 10000 --p 2 --nval 4096 $K --lr 1e-3 --rem 0.3 --spike_diag 2000"
SP="--width 4096 --depth 2 --eval_every 250 --steps 5000 --nval 4096 --seed 1 --p 2 $K --lr 1e-3"
SLOT0=("0 $C3 --seed 1" "1 $C1 --seed 3" "2 $PR --p 2 --rem 0.2 --seed 1")
SLOT1=("3 $C3 --seed 2" "4 $C1 --seed 4" "5 $PR --p 2 --rem 0.2 --seed 2")
SLOT2=("6 $C1 --seed 1" "7 $SP --batch 4096 --rem 0.8" "8 $PR --p 0 --rem 0.3 --seed 1")
SLOT3=("9 $C1 --seed 2" "10 $SP --batch 1024 --rem 0.3" "11 $PR --p 0 --rem 0.3 --seed 2")
echo "12 jobs" > $F/njobs
run_slot() { local g=$1; shift; for job in "$@"; do idx=${job%% *}; args=${job#* }
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 python -u "$F/code/optimizer/scale_study.py" $args --out "$F/res_j$idx.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$idx.txt"; done; }
run_slot 0 "${SLOT0[@]}" & run_slot 1 "${SLOT1[@]}" & run_slot 2 "${SLOT2[@]}" & run_slot 3 "${SLOT3[@]}" &
wait; echo "DONE $(date +%T)"
