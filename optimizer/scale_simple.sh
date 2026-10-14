#!/bin/bash
# Item 52 (2026-10-13): simplification tests on the field problem (5000 steps, batch 256, seed 1, width 4096; references: KronQ minimal
# preset 0.0622 / 1.42e-4, AdamW lr 3e-4 0.392 / 8.5e-4).
#  1. Is the rough result mostly input whitening?  AdamW on PCA-rotated and on fully whitened inputs, lr 1e-4 / 3e-4 / 1e-3, both targets (12).
#     Prediction: rotation alone changes little (AdamW is nearly rotation-insensitive on a near-isotropic start); full whitening helps the rough
#     target but stays well above KronQ, since T1 (input basis only) itself was 0.091 and the hidden bases carry the rest.
#  2. Bases frozen after step 20 / 200 on every layer (4). Prediction: a loss on rough (the kernel moves 2-3 % per step, item 21e) of 10-50 %.
#  3. Probe statistics every 2 / 5 steps (4). Prediction: within noise at 2, small loss at 5.                 usage: scale_simple.sh NODE NNODES
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_simple; mkdir -p $F
C="--width 4096 --depth 2 --batch 256 --eval_every 250 --tf32 1 --steps 5000 --seed 1"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --lr 1e-3 --rem 0.2"
JOBS=()
for p in 2 0; do for wh in pca full; do for lr in 1e-4 3e-4 1e-3; do JOBS+=("$C --opt adamw --p $p --whiten $wh --lr $lr"); done; done; done
for p in 2 0; do for fa in 20 200; do JOBS+=("$C $K --p $p --freeze_after $fa"); done; done
for p in 2 0; do for se in 2 5; do JOBS+=("$C $K --p $p --stats_every $se"); done; done
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
