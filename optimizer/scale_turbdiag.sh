#!/bin/bash
# Turbulence diagnostics (REPORT 61 item 45 follow-up): is the 64 x 64 residual learnable or noise, and is the target rough by our
# measure? On the 100000-pair cache, both targets: AdamW lr 1e-4 and minimal KronQ lr 1e-3 budget 0.15, 5000 steps, with the
# training batch loss logged at every eval (train / validation gap), --diag 1 and the measurement windows (--sa_measure 1:
# residual concentration on the true kernel's top directions at step 1 and every 250 steps, per-step held-out alignment).
# Plus the magnitude-rule test (why AdamW wins here): the Adam rule in the Kronecker basis (--rule adam) and the soft sign
# (--sign_floor 0.1), both targets. 8 runs, 4 per round, one node.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_turbdiag; mkdir -p $F
C="--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_100k.npz --width 4096 --depth 2 --batch 256 --eval_every 250 --tf32 1 --nval 2048 --steps 5000 --seed 1 --diag 1"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --lr 1e-3 --rem 0.15 --slow_adapt 1 --sa_measure 1"
JOBS=("$C $K --turb_target delta" "$C $K --turb_target direct" "$C --opt adamw --lr 1e-4 --turb_target delta" "$C --opt adamw --lr 1e-4 --turb_target direct"
      "$C $K --rule adam --fused 0 --turb_target delta" "$C $K --rule adam --fused 0 --turb_target direct" "$C $K --sign_floor 0.1 --fused 0 --turb_target delta" "$C $K --sign_floor 0.1 --fused 0 --turb_target direct")
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1)); if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait; echo "ALLDONE $(date +%T)"
