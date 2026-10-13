#!/bin/bash
# Under-stepping test (REPORT 61 item 28 prediction): on the rough target the reference's natural step is below the budget
# for 95 % of the run. Reference lr 1e-3 vs lr 2e-3 / 3e-3 (budget 0.2 binds longer) vs lr 1e-3 with the decay removed
# after half the run (--sphere-free: no; use --rem_ctrl 0 and warmup 0; here: lr 1e-3 constant = --rem_ctrl-style no decay is not
# available, so the third variant is lr 2e-3 with budget 0.3). Rough only, 4 runs, 1 per GPU.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_understep; mkdir -p $F
K="--opt kronqf --width 4096 --depth 2 --batch 256 --steps 5000 --eval_every 250 --diag 1 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --cap 1 --hyper 1 --p 2"
JOBS=("$K --lr 1e-3 --rem 0.2" "$K --lr 2e-3 --rem 0.2" "$K --lr 3e-3 --rem 0.2" "$K --lr 2e-3 --rem 0.3")
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1))
done
wait; echo "ALLDONE $(date +%T)"
