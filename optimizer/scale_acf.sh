#!/bin/bash
# Gradient autocorrelation per lambda decile (basis held for 64 steps) at four points along the reference run, both targets.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_acf; mkdir -p $F
K="--opt kronqf --width 4096 --depth 2 --steps 5000 --batch 256 --eval_every 250 --diag 1 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --lr 1e-3 --rem 0.2 --cap 1 --hyper 1 --acf 250,1000,2500,4000 --acf_len 64"
i=0
for p in 2 0; do
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $K --p $p --out "$F/res_p$p.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_p$p.txt" &
  i=$((i+1))
done
wait; echo "ALLDONE $(date +%T)"
