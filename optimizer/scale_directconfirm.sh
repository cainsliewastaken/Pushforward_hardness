#!/bin/bash
# OPTIMIZER_PLAN 6.1: confirm the direct result (0.0478, one seed). Turbulence direct, 400k pairs, 20 epochs, batch 256, TF32.
#   KronQ hybrid, no budget, lr 2e-4, decay 1e-3 per step: seeds 2 and 3.
#   AdamW at the same decay per step: lr 2e-4 wd 5.0, lr 1e-3 wd 1.0 (torch AdamW decays lr x wd per step; seed 1).
# Prediction: KronQ 0.046-0.050 on both seeds; AdamW >= 0.06. Null: a KronQ seed above 0.055, or AdamW at matched decay within 5 %.
# 4 runs, one per GPU.  Frozen code in $F/code.
F=$SCRATCH/l63_held_rough/runs/scale_directconfirm
T="--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_400k.npz --turb_cpu 1 --turb_onepass 1 --width 4096 --depth 2 --eval_every 250 --tf32 1 --nval 4096 --batch 256 --steps 31240 --turb_target direct"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --qr_guard 1 --refresh_hp 1 --reorth_transport 1 --lr 2e-4 --rem 0 --wd_abs 1e-3"
JOBS=("$T $K --seed 2" "$T $K --seed 3" "$T --opt adamw --lr 2e-4 --wd 5.0 --seed 1" "$T --opt adamw --lr 1e-3 --wd 1.0 --seed 1")
echo "4 jobs" > $F/njobs
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 python -u "$F/code/optimizer/scale_study.py" ${JOBS[$g]} --out "$F/res_j$g.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$g.txt" &
done
wait; echo "DONE $(date +%T)"
