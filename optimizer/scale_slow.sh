#!/bin/bash
# Slow adaptation test (slow_adapt.py) on both problems, one node, one run per GPU, 4 concurrent.
# KronQ: reference fixed constants (budget 0.2, MC probe, linear decay) | slow adaptation + linear decay | slow adaptation, no decay.
# Field problem: rough (p 2) and smooth (p 0); AdamW baselines are in runs/scale_grid (item 40).
# Coordinate fitting (camera 512x512): raw coordinates (rough), Fourier features ff 8 (the literature's INR setting),
# lowpass 8 raw (smooth control); AdamW lr 1e-3 at 5000 steps and lr 3e-4 / 1e-3 at 29000 steps (equal wall) as baselines.
# usage: scale_slow.sh [NODE NNODES]
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_slow; mkdir -p $F
IMG=$SCRATCH/l63_held_rough/data/camera.png
C="--width 4096 --depth 2 --batch 256 --eval_every 250 --tf32 1 --steps 5000 --seed 1"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --cap 1 --hyper 1 --fused 1 --lr 1e-3 --rem 0.2"
JOBS=()
for tgt in "--p 2" "--p 0" "--problem coord --image $IMG" "--problem coord --image $IMG --ff 8" "--problem coord --image $IMG --lowpass 8"; do
  JOBS+=("$C $K $tgt")
  JOBS+=("$C $K $tgt --slow_adapt 1")
  JOBS+=("$C $K $tgt --slow_adapt 1 --decay none")
  case "$tgt" in *coord*)
    JOBS+=("$C --opt adamw --lr 1e-3 $tgt")
    JOBS+=("$C --opt adamw --lr 1e-3 $tgt --steps 29000")
    JOBS+=("$C --opt adamw --lr 3e-4 $tgt --steps 29000");;
  esac
done
echo "${#JOBS[@]} jobs" > $F/njobs
NODE=${1:-0}; NN=${2:-1}; i=0; k=0
for job in "${JOBS[@]}"; do
  if [ $((i % NN)) -eq $NODE ]; then
    CUDA_VISIBLE_DEVICES=$((k % 4)) OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
    k=$((k+1)); if [ $((k % 4)) -eq 0 ]; then wait; fi
  fi
  i=$((i+1))
done
wait; echo "NODE $NODE DONE $(date +%T)"
