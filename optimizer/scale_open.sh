#!/bin/bash
# The open measurements of MECHANISM.md section 7: (1) the Adam rule WITH the cap and the learned weight (fair) vs the sign
# push; (2) the derived per-mode weight e_k erf(SNR_k/sqrt2)/s_k from the Kronecker SNR in place of the learned one;
# (3) batch 1024 with budgets 0.3 and 0.4 (rough, 2500 steps); (4) per-sample spread tau_k per decile in every run.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_open; mkdir -p $F
K="--opt kronqf --width 4096 --depth 2 --batch 256 --steps 5000 --eval_every 250 --diag 1 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --lr 1e-3 --rem 0.2 --cap 1"
JOBS=()
for p in 2 0; do
  JOBS+=("$K --p $p --rule adam --adam_fair 1 --hyper 1")
  JOBS+=("$K --p $p --wrule derived --hyper 0")
done
K4="--opt kronqf --width 4096 --depth 2 --batch 1024 --steps 2500 --eval_every 250 --diag 1 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --lr 1e-3 --cap 1 --hyper 1 --p 2"
JOBS+=("$K4 --rem 0.3")
JOBS+=("$K4 --rem 0.4")
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=2 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1))
done
wait; echo "ALLDONE $(date +%T)"
