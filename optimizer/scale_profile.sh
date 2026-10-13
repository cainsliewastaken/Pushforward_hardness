#!/bin/bash
# Per-decile step profile (weight, |D|, |D|/w, output-energy share) along the run for the sign push and the fair Adam rule.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_profile; mkdir -p $F
K="--opt kronqf --width 4096 --depth 2 --batch 256 --steps 5000 --eval_every 250 --diag 1 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --lr 1e-3 --rem 0.2 --cap 1 --hyper 1"
JOBS=()
for p in 2 0; do
  JOBS+=("$K --p $p")
  JOBS+=("$K --p $p --rule adam --adam_fair 1")
done
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1))
done
wait; echo "ALLDONE $(date +%T)"
