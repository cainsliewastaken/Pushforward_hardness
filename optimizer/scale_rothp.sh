#!/bin/bash
# Item 60 (2026-10-14): the hybrid with fp32 activation rotations (--rot_hp 1) on the LONG runs (SPIKE_RESULT "Hybrid": the hybrid alone is
# 0.0079 / 0.0076 on rough B1024 T10000 against fp32's 0.0033 / 0.0034; --rot_hp was run only on the 5000-step preset, where it reached
# 0.052-0.054 against fp32's 0.049). Same configurations and seeds as scale_hybrid.sh: rough B1024 T10000 seeds 1-2 and smooth B1024 T10000
# seeds 1-2 (hybrid alone: 1 / 2 small spikes), all with --rot_hp 1. Prediction: rough within 20 % of fp32 (the 2.4x gap is the rotated
# forward's TF32 rounding); null: the gap stays, and the forward itself needs fp32. 4 runs, one wave, one node.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_rothp; mkdir -p $F
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --qr_guard 1 --refresh_hp 1 --reorth_transport 1 --tf32 1 --rot_hp 1"
C3="--width 4096 --depth 2 --batch 1024 --eval_every 125 --steps 10000 --p 2 --nval 4096 --lr 1e-3 --rem 0.3 --spike_diag 2000 $K"
C1="--width 4096 --depth 2 --batch 1024 --eval_every 250 --steps 10000 --p 0 --nval 4096 --lr 2e-3 --rem 0.3 --spike_diag 2000 $K"
JOBS=("$C3 --seed 1" "$C3 --seed 2" "$C1 --seed 1" "$C1 --seed 2")
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
