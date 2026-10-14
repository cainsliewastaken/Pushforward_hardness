#!/bin/bash
# Spike fix test (2026-10-14): --qr_guard 1 (Cholesky-QR rotation checked for orthonormality in float64; failing blocks redone by
# Householder QR in float64). Reference (scale_tf32, same configs and seeds, no guard): smooth B1024 T10000 lr 2e-3 spiked in 4 of 4 TF32
# seeds and 4 of 4 fp32 seeds, each time 250-1500 steps after the basis lost orthonormality at a Cholesky-QR refresh; rough B1024 T10000
# TF32 spiked in 2 of 2. Prediction: with the guard the basis error stays < 1e-3, qr_fallbacks > 0 near steps 4500-7500, and no spikes
# (0 of 4 TF32, 0 of 2 fp32 smooth seeds; rough TF32 0 of 2). Null: spikes persist with an orthonormal basis (then the basis failure was a
# symptom, not the cause).  8 runs, 1 node.  Frozen code in $F/code.
F=$SCRATCH/l63_held_rough/runs/scale_qrguard
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --qr_guard 1"
C1="--width 4096 --depth 2 --batch 1024 --eval_every 250 --steps 10000 --p 0 --nval 4096 $K --lr 2e-3 --rem 0.3 --spike_diag 2000"
C3="--width 4096 --depth 2 --batch 1024 --eval_every 125 --steps 10000 --p 2 --nval 4096 $K --lr 1e-3 --rem 0.3 --spike_diag 2000"
SLOT0=("0 $C1 --tf32 0 --seed 1" "1 $C1 --tf32 1 --seed 3")
SLOT1=("2 $C1 --tf32 0 --seed 2" "3 $C1 --tf32 1 --seed 4")
SLOT2=("4 $C3 --tf32 1 --seed 1" "5 $C1 --tf32 1 --seed 1")
SLOT3=("6 $C3 --tf32 1 --seed 2" "7 $C1 --tf32 1 --seed 2")
echo "8 jobs" > $F/njobs
run_slot() { local g=$1; shift; for job in "$@"; do idx=${job%% *}; args=${job#* }
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 python -u "$F/code/optimizer/scale_study.py" $args --out "$F/res_j$idx.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$idx.txt"; done; }
run_slot 0 "${SLOT0[@]}" & run_slot 1 "${SLOT1[@]}" & run_slot 2 "${SLOT2[@]}" & run_slot 3 "${SLOT3[@]}" &
wait; echo "DONE $(date +%T)"
