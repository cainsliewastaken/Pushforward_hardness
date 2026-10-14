#!/bin/bash
# kappa test (side-chat plan): rho_t = kappa x 2 c_t, c_t the held-out alignment measured every 10 steps on a fresh batch, no
# lr schedule (lr 3e-3 nominal, decay none), both targets. Reference 0.062 / 1.38e-4. 7 runs, one per GPU, two batches.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_kappa; mkdir -p $F
K="--opt kronqf --width 4096 --depth 2 --batch 256 --steps 5000 --eval_every 250 --diag 1 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --lr 3e-3 --decay none --rem 0.2 --rem_val_every 10 --cap 1 --hyper 1 --fused 1"
JOBS=("$K --p 2 --rem_val_track 0.5" "$K --p 2 --rem_val_track 0.75" "$K --p 2 --rem_val_track 1.0" "$K --p 0 --rem_val_track 0.5" "$K --p 0 --rem_val_track 0.75" "$K --p 0 --rem_val_track 1.0" "$K --p 0 --rem_val_track 0.3")
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1)); if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait; echo "ALLDONE $(date +%T)"
