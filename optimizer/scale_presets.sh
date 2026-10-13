#!/bin/bash
# The earlier algorithms as presets of the fast path, at width 4096, both targets, Adam-memory blocks of 256 + compressed
# state, 5000 steps; plus the stats_every ablation. Jobs -> GPU i % 4 (up to 12 concurrent). usage (inside srun): scale_presets.sh
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_presets; mkdir -p $F
B="--opt kronqf --width 4096 --depth 2 --steps 5000 --batch 256 --eval_every 250 --diag 1 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --eig_dtype float64 --reltol 1e-12"
JOBS=()
for p in 2 0; do
  # kronsign ks_sign (plain sign of the rotated momentum, no cap, no mask) and ks_adam (Adam rule), two lrs each
  JOBS+=("$B --p $p --rule sign --cap 0 --hyper 0 --reltol 0 --lr 5e-4 --refresh 10 --eig_adapt 1")
  JOBS+=("$B --p $p --rule sign --cap 0 --hyper 0 --reltol 0 --lr 2e-3 --refresh 10 --eig_adapt 1")
  JOBS+=("$B --p $p --rule adam --lr 3e-3 --refresh 10 --eig_adapt 1")
  JOBS+=("$B --p $p --rule adam --lr 1e-2 --refresh 10 --eig_adapt 1")
  # kronq2 (q = 0, cap, remainder controller, error-controlled refresh) and kronq3h (+ hypergradient), lr 3e-3 as tuned on L63
  JOBS+=("$B --p $p --hyper 0 --lr 3e-3 --lr_kappa 0.1 --check_every 10 --refresh 10 --adapt_refresh 0.1")
  JOBS+=("$B --p $p --hyper 1 --horizon 1 --lr 3e-3 --lr_kappa 0.1 --check_every 10 --refresh 10 --adapt_refresh 0.1")
  JOBS+=("$B --p $p --hyper 1 --horizon 100 --lr 3e-3 --lr_kappa 0.1 --check_every 10 --refresh 10 --adapt_refresh 0.1")
  # today's best (hyper, rem 0.2, adaptive) and its stats_every ablation
  JOBS+=("$B --p $p --hyper 1 --lr 1e-3 --rem 0.2 --refresh 20 --adapt_refresh 0.1")
  JOBS+=("$B --p $p --hyper 1 --lr 1e-3 --rem 0.2 --refresh 20 --adapt_refresh 0.1 --stats_every 20")
done
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=2 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1))
  if [ $((i % 12)) -eq 0 ]; then wait; fi
done
wait; echo "ALLDONE $(date +%T)"
