#!/bin/bash
# Item 58 (2026-10-14): small network, small and large batch, no cap, no budget, with the spike fix (--qr_guard 1: Cholesky-QR rotations
# checked for orthonormality, failing blocks redone by Householder QR in float64; SPIKE_RESULT.md) and full fp32 (--tf32 0: the guarded fp32
# configuration is the one verified spike-free; TF32 sets a floor on the rough target). Field, width 1024, depth 2, 5000 steps, seed 1,
# both targets, batch 256 and 2048. KronQ: lr {1e-4, 2e-4, 4e-4} at 256 and {2e-4, 4e-4, 8e-4} at 2048 (the batch law shifts the window);
# AdamW: lr {3e-4, 1e-3} at 256, {1e-3, 3e-3} at 2048. 20 runs, one node.
# Predictions: at 2048 KronQ needs no budget (slack there at 4096 width) and beats AdamW on rough by > 3x; at 256 without the budget the rough
# gap to AdamW narrows (item 57's noise floor, 1.8x at width 4096) but KronQ still leads; smooth within 20 % of AdamW at both; no spikes.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_smallnet; mkdir -p $F
C="--width 1024 --depth 2 --eval_every 250 --tf32 0 --steps 5000 --seed 1 --nval 4096"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --rem 0 --qr_guard 1"
JOBS=()
for p in 2 0; do
  for lr in 2e-4 4e-4 8e-4; do JOBS+=("$C $K --p $p --batch 2048 --lr $lr"); done
  for lr in 1e-3 3e-3; do JOBS+=("$C --opt adamw --p $p --batch 2048 --lr $lr"); done
  for lr in 1e-4 2e-4 4e-4; do JOBS+=("$C $K --p $p --batch 256 --lr $lr"); done
  for lr in 3e-4 1e-3; do JOBS+=("$C --opt adamw --p $p --batch 256 --lr $lr"); done
done
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
