#!/bin/bash
# Ablation of the fast-path configuration that gave 0.064 / 1.35e-4 at width 4096 (hyper sign push, cap, output-change
# budget 0.2, blocks of 256 + compressed state, stats every step, adaptive refresh): remove one rule at a time, the
# plain sign push, the rotated Adam rule (two lrs), and full factors. Both targets, 16 runs, 4 per GPU.
# usage (inside srun): scale_ablate.sh
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_ablate; mkdir -p $F
B="--opt kronqf --width 4096 --depth 2 --steps 5000 --batch 256 --eval_every 250 --diag 1 --tf32 1 --compress 1 --mom_bf16 1 --stats_every 1 --refresh 20 --adapt_refresh 0.1 --reltol 1e-12"
JOBS=()
for p in 2 0; do
  JOBS+=("$B --p $p --block 256 --lr 1e-3 --hyper 1 --cap 1 --rem 0.2")              # a. reference (today's best)
  JOBS+=("$B --p $p --block 256 --lr 1e-3 --hyper 1 --cap 1 --rem 0")                # b. no budget
  JOBS+=("$B --p $p --block 256 --lr 1e-3 --hyper 1 --cap 0 --rem 0.2")              # c. no cap
  JOBS+=("$B --p $p --block 256 --lr 1e-3 --hyper 0 --cap 1 --rem 0.2")              # d. no hypergradient
  JOBS+=("$B --p $p --block 256 --lr 1e-3 --hyper 0 --cap 0 --rem 0")                # e. plain sign push
  JOBS+=("$B --p $p --block 256 --lr 1e-3 --rule adam --rem 0.2")                    # f. rotated Adam rule, lr 1e-3
  JOBS+=("$B --p $p --block 256 --lr 3e-3 --rule adam --rem 0.2")                    # g. rotated Adam rule, lr 3e-3
  JOBS+=("$B --p $p --block 4096 --lr 1e-3 --hyper 1 --cap 1 --rem 0.2")             # h. full factors
done
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=2 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1))
done
wait; echo "ALLDONE $(date +%T)"
