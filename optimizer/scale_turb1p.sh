#!/bin/bash
# Single-pass turbulence comparison (REPORT 61 item 47 consequence): one pass over the 400000 available 64 x 64 pairs, no sample seen
# twice (--turb_onepass 1), so memorisation is excluded by construction and the comparison measures optimization speed on the rough
# increment target (and the direct control). Batch 256 (1562 steps) and batch 1024 (390 steps). Minimal KronQ lr 1e-3 with the
# budget at the batch law (0.15 / 0.3 at 256, 0.3 / 0.6 at 1024); AdamW lr {1e-4, 3e-4, 1e-3}. Validation every 50 steps (4096
# held-out pairs from the test region); val_best and val_final. 20 runs, one node.   usage: scale_turb1p.sh
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_turb1p; mkdir -p $F
C="--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_400k.npz --turb_cpu 1 --turb_onepass 1 --width 4096 --depth 2 --eval_every 50 --tf32 1 --nval 4096 --seed 1"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --lr 1e-3"
JOBS=()
for tg in delta direct; do
  for rem in 0.15 0.3; do JOBS+=("$C $K --batch 256 --steps 1562 --turb_target $tg --rem $rem"); done
  for rem in 0.3 0.6; do JOBS+=("$C $K --batch 1024 --steps 390 --turb_target $tg --rem $rem"); done
  for lr in 1e-4 3e-4 1e-3; do
    JOBS+=("$C --opt adamw --batch 256 --steps 1562 --turb_target $tg --lr $lr")
    JOBS+=("$C --opt adamw --batch 1024 --steps 390 --turb_target $tg --lr $lr")
  done
done
echo "${#JOBS[@]} jobs" > $F/njobs
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=8 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1)); if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait; echo "ALLDONE $(date +%T)"
