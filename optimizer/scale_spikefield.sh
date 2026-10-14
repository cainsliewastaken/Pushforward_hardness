#!/bin/bash
# Spike defect on the field (2026-10-14). scale_crit: 11 of 73 KronQ runs spike (> 3x the running best), first spike at step 4500-17750,
# none in runs of <= 2500 steps, up to x9e5 on the smooth target. Three candidates, each with an instrument (--spike_diag, per step) and a
# remedy: (A) budget rescale: the sign step's proposed output change does not shrink with the residual, so late it is >> the budget and the
# one-forward LINEAR undo is wrong -> rho_raw >> budget and rho_real >> budget at the spike; --rem_iter 2 removes it (the other session's
# item 56 hypothesis). (B) basis numerics: Cholesky-QR rotations drift and the every-10th re-orthonormalisation changes the function
# without transporting the weights -> refresh_df >> 0 at the spike; --cholqr 0 removes it. (C) learned weights at the clamp -> umax 4.6;
# --u_clamp 2.3 removes it. Null: no variant removes the spike and no instrument moves at it.
# Configs that spiked: C1 smooth B1024 T10000 lr 2e-3 (step 7500, x9e5); C2 smooth B512 T20000 lr 2e-3 (7500); C3 rough B1024 T10000 (5000).
# Runs from a frozen copy of the code ($F/code), so edits in the worktree cannot reach running jobs.   usage: scale_spikefield.sh NODE
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_spikefield
C="--width 4096 --depth 2 --tf32 1 --seed 1 --nval 4096 --eval_every 250"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --rem_check 1"
C1="$C $K --p 0 --batch 1024 --steps 10000 --lr 2e-3 --rem 0.3 --spike_diag 4000"
C2="$C $K --p 0 --batch 512 --steps 20000 --lr 2e-3 --rem 0.3 --spike_diag 4000"
C3="$C $K --p 2 --batch 1024 --steps 10000 --lr 1e-3 --rem 0.3 --spike_diag 3000"
SLOT0=("0 $C2" "1 $C1 --cholqr 0")
SLOT1=("2 $C2 --rem_iter 2" "3 $C1 --u_clamp 2.3")
SLOT2=("4 $C1" "5 $C3")
SLOT3=("6 $C1 --rem_iter 2" "7 $C3 --rem_iter 2")
echo "8 jobs" > $F/njobs
run_slot() { local g=$1; shift; for job in "$@"; do idx=${job%% *}; args=${job#* }
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 python -u "$F/code/optimizer/scale_study.py" $args --out "$F/res_j$idx.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$idx.txt"; done; }
run_slot 0 "${SLOT0[@]}" & run_slot 1 "${SLOT1[@]}" & run_slot 2 "${SLOT2[@]}" & run_slot 3 "${SLOT3[@]}" &
wait; echo "DONE $(date +%T)"
