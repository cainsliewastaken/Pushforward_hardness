#!/bin/bash
# Input-side box test (TRUST_REGION.md): weight every hidden layer's modes by (alpha_j / alpha_max)^(qa/2), floored, with cap
# and budget as in the reference; plus the radius scan binned by alpha decile (and lambda decile with alpha, sigma logged).
# Width 4096, both targets, Adam memory. 10 runs, 3 per GPU. usage (inside srun): scale_alpha.sh
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_alpha; mkdir -p $F
B="--opt kronqf --width 4096 --depth 2 --steps 5000 --batch 256 --eval_every 250 --diag 1 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --refresh 20 --adapt_refresh 0.1 --reltol 1e-12 --lr 1e-3 --hyper 1 --cap 1 --rem 0.2"
R="--opt kronqf --width 4096 --depth 2 --steps 2000 --batch 256 --eval_every 500 --diag 1 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --refresh 20 --adapt_refresh 0.1 --reltol 1e-12 --lr 1e-3 --hyper 1 --cap 1 --rem 0.2 --radius_scan 100,500,1500"
JOBS=()
for p in 2 0; do
  JOBS+=("$B --p $p")
  JOBS+=("$B --p $p --qa -1 --q_floor 1e-4")
  JOBS+=("$B --p $p --qa -1 --q_floor 1e-6")
  JOBS+=("$B --p $p --qa -0.5 --q_floor 1e-4")
  JOBS+=("$R --p $p")
done
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=2 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1))
done
wait; echo "ALLDONE $(date +%T)"
