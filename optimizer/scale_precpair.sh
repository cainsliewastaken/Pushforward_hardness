#!/bin/bash
# Precision pairs on the rough preset (batch 256, 5000 steps, seed 1): (a) refactor fp32 vs KronQFast fp32 (equal -> the refactor's TF32
# deficit, 0.099 vs 0.062, is precision); (b) AdamW (lr 3e-4) fp32 vs TF32 (is the rough target precision-limited in ordinary coordinates
# for any optimizer?). 4 runs, one per GPU.  Frozen code in $F/code.
F=$SCRATCH/l63_held_rough/runs/scale_precpair
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --qr_guard 1 --lr 1e-3 --rem 0.2"
B="--width 4096 --depth 2 --batch 256 --eval_every 250 --steps 5000 --nval 4096 --seed 1 --p 2"
JOBS=("$B $K --orig 1 --tf32 0" "$B $K --orig 0 --tf32 0" "$B --opt adamw --lr 3e-4 --tf32 0" "$B --opt adamw --lr 3e-4 --tf32 1")
echo "4 jobs" > $F/njobs
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 python -u "$F/code/optimizer/scale_study.py" ${JOBS[$g]} --out "$F/res_j$g.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$g.txt" &
done
wait; echo "DONE $(date +%T)"
