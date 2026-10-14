#!/bin/bash
# The hybrid (2026-10-14): weights kept in the basis (the slow modes keep their relative precision), refresh made precise and non-compounding:
# --refresh_hp 1 (refresh products and transports in full fp32), --reorth_transport 1 (weights transformed at re-orthonormalisation),
# --qr_guard 1. Network TF32. Precision pairs (scale_precpair): in fp32 the refactor equals KronQFast on the rough preset (0.054 vs 0.049),
# so the refactor's TF32 deficit was precision; AdamW is identical in fp32 and TF32 (0.390); KronQFast gains in fp32 (0.049 vs ~0.062-0.071).
# Runs: rough preset seeds 1-2: hybrid / hybrid + --rot_hp 1 (activation rotations fp32 too); rough B1024 T10000 seeds 1-2: hybrid
# (KronQFast TF32 0.0073 / 0.0078 with spikes, fp32 0.0033 / 0.0034); smooth B1024 T10000 seeds 1-2: hybrid (KronQFast TF32 11 / 11 spikes).
# Predictions: no catastrophic spikes; rough between TF32 and fp32, closer to fp32 with --rot_hp.  8 runs, 1 node.  Frozen code in $F/code.
F=$SCRATCH/l63_held_rough/runs/scale_hybrid
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --qr_guard 1 --refresh_hp 1 --reorth_transport 1 --tf32 1"
PR="--width 4096 --depth 2 --batch 256 --eval_every 250 --steps 5000 --nval 4096 --p 2 --lr 1e-3 --rem 0.2 $K"
C3="--width 4096 --depth 2 --batch 1024 --eval_every 125 --steps 10000 --p 2 --nval 4096 --lr 1e-3 --rem 0.3 --spike_diag 2000 $K"
C1="--width 4096 --depth 2 --batch 1024 --eval_every 250 --steps 10000 --p 0 --nval 4096 --lr 2e-3 --rem 0.3 --spike_diag 2000 $K"
SLOT0=("0 $C3 --seed 1" "1 $PR --seed 1")
SLOT1=("2 $C3 --seed 2" "3 $PR --seed 2")
SLOT2=("4 $C1 --seed 1" "5 $PR --seed 1 --rot_hp 1")
SLOT3=("6 $C1 --seed 2" "7 $PR --seed 2 --rot_hp 1")
echo "8 jobs" > $F/njobs
run_slot() { local g=$1; shift; for job in "$@"; do idx=${job%% *}; args=${job#* }
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 python -u "$F/code/optimizer/scale_study.py" $args --out "$F/res_j$idx.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$idx.txt"; done; }
run_slot 0 "${SLOT0[@]}" & run_slot 1 "${SLOT1[@]}" & run_slot 2 "${SLOT2[@]}" & run_slot 3 "${SLOT3[@]}" &
wait; echo "DONE $(date +%T)"
