#!/bin/bash
# Spike defect, second pass (2026-10-14). First pass (scale_spikefield): every spike onset had the largest learned weight at its clamp
# (u 4.59 of 4.6, w = 100); the batch output change after the budget was 1.0-1.3x the budget (the budget held on the batch); exact QR
# (--cholqr 0) and the repeated rescale (--rem_iter 2) still spiked; --u_clamp 2.3 (w <= 10) did not spike and ended at the best value.
# Runs crashed at the first realised change > 3x budget (logger bug, fixed in aadc39b). This pass, full trajectories:
#   C1 smooth B1024 T10000, C2 smooth B512 T20000, C3 rough B1024 T10000: reference / --u_clamp 2.3 / --cap 1 (the per-mode completion bound
#   of ingredient 3, removed in item 43 on 5000-step evidence); C1 seed 2 reference and clamp; C1 --tf32 0 (arithmetic precision: refresh_df
#   was 3-6 % of the residual per refresh); cost at the 5000-step batch-256 presets (rough, smooth) with --u_clamp 2.3 and with --cap 1.
# Predictions: clamp 2.3 and cap 1 remove the spikes on all three configs and cost < 5 % at the presets; fp32 lowers refresh_df ~100x and
# does not by itself remove the spike. Null: spikes persist with the clamp on C2 / C3 / seed 2 (then the clamp result was luck).
# Frozen code copy in $F/code.   usage: scale_spikefield2.sh
F=$SCRATCH/l63_held_rough/runs/scale_spikefield2
C="--width 4096 --depth 2 --seed 1 --nval 4096 --eval_every 250"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --rem_check 1"
C1="$C $K --p 0 --batch 1024 --steps 10000 --lr 2e-3 --rem 0.3 --spike_diag 4000"
C2="$C $K --p 0 --batch 512 --steps 20000 --lr 2e-3 --rem 0.3 --spike_diag 4000"
C3="$C $K --p 2 --batch 1024 --steps 10000 --lr 1e-3 --rem 0.3 --spike_diag 3000"
PR="--width 4096 --depth 2 --seed 1 --nval 4096 --eval_every 250 --tf32 1 --batch 256 --steps 5000 --rem 0.2 --lr 1e-3 $K"
SLOT0=("0 $C2 --tf32 1 --cap 0" "1 $C1 --tf32 1 --cap 0 --seed 2" "2 $PR --p 2 --cap 0 --u_clamp 2.3" "3 $PR --p 0 --cap 0 --u_clamp 2.3")
SLOT1=("4 $C2 --tf32 1 --cap 0 --u_clamp 2.3" "5 $C1 --tf32 1 --cap 0 --u_clamp 2.3 --seed 2" "6 $PR --p 2 --cap 1" "7 $PR --p 0 --cap 1")
SLOT2=("8 $C2 --tf32 1 --cap 1" "9 $C1 --tf32 1 --cap 0" "10 $C1 --tf32 1 --cap 1")
SLOT3=("11 $C3 --tf32 1 --cap 0" "12 $C3 --tf32 1 --cap 0 --u_clamp 2.3" "13 $C3 --tf32 1 --cap 1" "14 $C1 --tf32 0 --cap 0" "15 $C1 --tf32 1 --cap 0 --u_clamp 2.3")
echo "16 jobs" > $F/njobs
run_slot() { local g=$1; shift; for job in "$@"; do idx=${job%% *}; args=${job#* }
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 python -u "$F/code/optimizer/scale_study.py" $args --out "$F/res_j$idx.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$idx.txt"; done; }
run_slot 0 "${SLOT0[@]}" & run_slot 1 "${SLOT1[@]}" & run_slot 2 "${SLOT2[@]}" & run_slot 3 "${SLOT3[@]}" &
wait; echo "DONE $(date +%T)"
