#!/bin/bash
# LOOKAHEAD_THEORY section 4 experiment at width 4096, both targets, Adam memory (blocks of 256 + compressed state),
# 5000 steps, stats every step, adaptive refresh: (a1) the two-regime rule (alloc2, box 1, lr 1e-3 with decay),
# (a2) the same with the radius from the remainder controller (lr_kappa 0.1, TF32-off check), (a3) box 3 (sphere shares),
# (b) the rule with c := 0 (frozen-GN ridge control), (d) the reference (hyper + cap + budget 0.2) with the curvature
# diagnostics and the exact autograd check. 10 runs, 3 per GPU (jobs -> GPU i % 4). usage (inside srun): scale_lookahead.sh
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_lookahead; mkdir -p $F
B="--opt kronqf --width 4096 --depth 2 --steps 5000 --batch 256 --eval_every 250 --diag 1 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --refresh 20 --adapt_refresh 0.1 --reltol 1e-12 --lr 1e-3"
JOBS=()
for p in 2 0; do
  JOBS+=("$B --p $p --alloc2 1 --box 1")
  JOBS+=("$B --p $p --alloc2 1 --box 1 --lr_kappa 0.1 --check_every 10")
  JOBS+=("$B --p $p --alloc2 1 --box 3")
  JOBS+=("$B --p $p --alloc2 1 --box 1 --cfac 0")
  JOBS+=("$B --p $p --hyper 1 --cap 1 --rem 0.2 --curv 1 --check_curv 1")
done
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=2 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1))
done
wait; echo "ALLDONE $(date +%T)"
