#!/bin/bash
# Fused update (torch.compile) and the T3 fix (input-layer A basis frozen after 5 refreshes), Cholesky-QR, floor 5; one
# run per GPU. Reference (unfused): rough 0.0601 / smooth 1.37-1.41e-4 at 15.8 ms per step.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_fused; mkdir -p $F
K="--opt kronqf --width 4096 --depth 2 --batch 256 --steps 5000 --eval_every 250 --diag 1 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --lr 1e-3 --rem 0.2 --cap 1 --hyper 1 --fused 1"
JOBS=("$K --p 2" "$K --p 0" "$K --p 2 --A0_fixed 5" "$K --p 0 --A0_fixed 5")
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1))
done
wait; echo "ALLDONE $(date +%T)"
