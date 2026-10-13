#!/bin/bash
# Held-out line search (--ls_val, REPORT 61 item 31): the step length alpha = <E_v, -df_v> / ||df_v||^2 on a fresh batch, no
# batch budget, no constant. Rough lr 1e-3 / 2e-3 / 3e-3, smooth lr 1e-3. 4 runs, 1 per GPU. Reference 0.0612 / 1.41e-4.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_lsval; mkdir -p $F
K="--opt kronqf --width 4096 --depth 2 --batch 256 --steps 5000 --eval_every 250 --diag 1 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --rem 0 --ls_val 1 --cap 1 --hyper 1"
JOBS=("$K --p 2 --lr 1e-3" "$K --p 2 --lr 2e-3" "$K --p 2 --lr 3e-3" "$K --p 0 --lr 1e-3")
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1))
done
wait; echo "ALLDONE $(date +%T)"
