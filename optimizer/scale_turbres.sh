#!/bin/bash
# Regime of the turbulence one-step target against resolution (user's question, 2026-10-13): measurement-only KronQ runs (300 steps,
# --sa_measure windows at steps 1-40 and 250-290, --diag per-decile SNR) at 64 x 64 (100k cache), 128 x 128 (100k cache, CPU-resident)
# and 256 x 256 (8k cache, CPU-resident), both targets. Prediction: concentration on the kernel's top directions and the batch /
# held-out output-change ratio fall toward the rough signature (field rough: 6x, ~1) as the resolution rises; 64 x 64 reads 15-40x, 7-10.
# 6 runs, one per GPU, one node.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_turbres; mkdir -p $F; D=$SCRATCH/l63_held_rough/data
C="--problem turb --width 4096 --depth 2 --batch 256 --eval_every 100 --tf32 1 --nval 1024 --steps 300 --seed 1 --diag 1"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --lr 1e-3 --rem 0.15 --slow_adapt 1 --sa_measure 1 --sa_every 250 --sa_win 40"
JOBS=()
for tg in delta direct; do
  JOBS+=("$C $K --turb_cache $D/turb64_lead10_100k.npz --turb_target $tg")
  JOBS+=("$C $K --turb_cache $D/turb128_lead10_100k.npz --turb_cpu 1 --turb_target $tg")
  JOBS+=("$C $K --turb_cache $D/turb256_lead10_8k.npz --turb_cpu 1 --turb_target $tg")
done
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=8 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1)); if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait; echo "ALLDONE $(date +%T)"
