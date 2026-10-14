#!/bin/bash
# Why the refactor (--orig 1) is 60 % worse on the rough preset (0.099 vs 0.062) while it matches KronQFast on CPU in fp32 and is 10 %
# better on smooth. Hypothesis: TF32 rounding in the refactor's own products (the whole-gradient rotation, or the statistics accumulated
# in ordinary coordinates) swamps the small slow-mode components the rough target lives on. Rough preset, batch 256, 5000 steps, seed 1:
#   refactor TF32 with full fp32 in: rot / stats / refresh / all three;  refactor fully fp32;  refactor TF32 again (noise);
#   KronQFast fully fp32 and KronQFast TF32 (same-code references).
# Prediction: one part restores the rough value in TF32, and refactor fp32 ~ KronQFast fp32. Null: refactor fp32 still differs from
# KronQFast fp32 (a real difference at scale, not precision).  8 runs, 1 node.  Frozen code in $F/code.
F=$SCRATCH/l63_held_rough/runs/scale_orighp
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --qr_guard 1"
PR="--width 4096 --depth 2 --batch 256 --eval_every 250 --steps 5000 --nval 4096 --seed 1 --p 2 --lr 1e-3 --rem 0.2 $K"
SLOT0=("0 $PR --orig 1 --tf32 1 --orig_hp rot" "1 $PR --orig 1 --tf32 0")
SLOT1=("2 $PR --orig 1 --tf32 1 --orig_hp stats" "3 $PR --orig 0 --tf32 0")
SLOT2=("4 $PR --orig 1 --tf32 1 --orig_hp refresh" "5 $PR --orig 0 --tf32 1")
SLOT3=("6 $PR --orig 1 --tf32 1 --orig_hp rot,stats,refresh" "7 $PR --orig 1 --tf32 1")
echo "8 jobs" > $F/njobs
run_slot() { local g=$1; shift; for job in "$@"; do idx=${job%% *}; args=${job#* }
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 python -u "$F/code/optimizer/scale_study.py" $args --out "$F/res_j$idx.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$idx.txt"; done; }
run_slot 0 "${SLOT0[@]}" & run_slot 1 "${SLOT1[@]}" & run_slot 2 "${SLOT2[@]}" & run_slot 3 "${SLOT3[@]}" &
wait; echo "DONE $(date +%T)"
