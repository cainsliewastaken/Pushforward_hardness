#!/bin/bash
# Horizon measurement along the FIXED reference run (SIGN_PROCESS.md section 5): --sa_measure 1 (windows of 40 steps every 250:
# kappa_W for W = 1, 2, 5, 10, 20, 40 from the net first-order output displacement on a fixed fresh batch, per-step held-out
# alignment c, lag-1 reversal r, batch / held-out output-change ratio) with --diag 1 (per-decile output-energy share of the
# step, cap-active fraction, batch rho). Field rough, field smooth, coordinate fitting raw. 3 runs, one per GPU.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_measure; mkdir -p $F
IMG=$SCRATCH/l63_held_rough/data/camera.png
K="--opt kronqf --width 4096 --depth 2 --batch 256 --eval_every 250 --tf32 1 --steps 5000 --seed 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --cap 1 --hyper 1 --fused 1 --lr 1e-3 --rem 0.2 --slow_adapt 1 --sa_measure 1 --diag 1"
JOBS=("$K --p 2" "$K --p 0" "$K --problem coord --image $IMG")
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1))
done
wait; echo "ALLDONE $(date +%T)"
