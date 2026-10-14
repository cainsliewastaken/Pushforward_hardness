#!/bin/bash
# Component ablation of the standing configuration (Cholesky-QR, floor 5, fused), one change at a time, both targets, one
# seed, one run per GPU. usage: scale_ablate2.sh NODE NNODES
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_ablate2; mkdir -p $F
K="--opt kronqf --width 4096 --depth 2 --batch 256 --steps 5000 --eval_every 250 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --lr 1e-3 --rem 0.2 --cap 1 --hyper 1 --fused 1"
V=("" "--b1 0.0" "--b1 0.8" "--b1 0.95" "--beta_stats 0.9" "--beta_stats 0.99" "--beta_u 0.003" "--beta_u 0.03" "--cap_L 0" "--decay none" "--decay cosine" "--block 128" "--block 512" "--S_mode emp")
JOBS=()
for p in 2 0; do for v in "${V[@]}"; do JOBS+=("$K --p $p $v"); done; done
NODE=${1:-0}; NN=${2:-1}; i=0; k=0
for job in "${JOBS[@]}"; do
  if [ $((i % NN)) -eq $NODE ]; then
    CUDA_VISIBLE_DEVICES=$((k % 4)) OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
    k=$((k+1)); if [ $((k % 4)) -eq 0 ]; then wait; fi
  fi
  i=$((i+1))
done
wait; echo "NODE $NODE DONE $(date +%T)"
