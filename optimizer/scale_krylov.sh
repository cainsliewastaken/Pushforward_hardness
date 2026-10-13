#!/bin/bash
# Krylov (matrix-free ridge) step on the batch against the reference (COLLECTIVE_RULE.md section 4): width 4096, both
# targets, 5000 steps, output-space radius rem 0.2 with decay; M = 1 / 3 / 8 CG terms, damping 0.1 / 1 x Rayleigh,
# momentum 0 / 0.9; plus the Taylor-radius scan with the alignment column on the reference. 12 runs, 3 per GPU.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_krylov; mkdir -p $F
B="--opt krylov --width 4096 --depth 2 --steps 5000 --batch 256 --eval_every 250 --diag 1 --tf32 1 --lr 1e-3 --rem 0.2"
R="--opt kronqf --width 4096 --depth 2 --steps 3000 --batch 256 --eval_every 500 --diag 1 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --refresh 20 --adapt_refresh 0.1 --reltol 1e-12 --lr 1e-3 --hyper 1 --cap 1 --rem 0.2 --radius_scan 100,500,1500"
JOBS=()
for p in 2 0; do
  JOBS+=("$B --p $p --kr_M 1 --kr_gamma 1")
  JOBS+=("$B --p $p --kr_M 3 --kr_gamma 0.1")
  JOBS+=("$B --p $p --kr_M 3 --kr_gamma 1")
  JOBS+=("$B --p $p --kr_M 8 --kr_gamma 0.1")
  JOBS+=("$B --p $p --kr_M 3 --kr_gamma 0.1 --kr_mom 0.9")
  JOBS+=("$R --p $p")
done
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=2 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1))
done
wait; echo "ALLDONE $(date +%T)"
