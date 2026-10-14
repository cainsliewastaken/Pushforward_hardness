#!/bin/bash
# Item 54 (2026-10-13): the forgetting-rate test of the weight-decay mechanism. Claim under test: decoupled decay works on replayed data by
# erasing sample-specific components between visits, so the controlling number is the per-epoch forgetting factor f = exp(-lr wd N/B),
# not wd itself. At 400k pairs (N/B 1562, lr 1e-3): wd 0.01 f 0.98 -> nothing; wd 0.3 f 0.63 -> -35 %; wd 1.0 f 0.21 -> -43 % (items 50-53).
# Test at 100k pairs (N/B 390, 20 epochs = 7810 steps; the overfit is stronger with fewer samples, the comparison is relative):
#   KronQ increment wd 0 / 0.3 (f 0.89, predicted < 15 % gain) / 1.2 (f 0.63, predicted ~ -35 %, seeds 1 and 2) / 4.0 (f 0.21, predicted ~ -43 %
#   unless the equilibrium cap w/wd over-regularises, which would show as a higher training loss);  KronQ direct wd 0 / 1.2;
#   AdamW lr 1e-4 wd 0 / 0.3 / 1.2 (reference; predicted insensitive on the increment);  400k KronQ increment wd 3.0 (f 0.009: predicted
#   over-regularised, training loss up).  Null: the gain at 100k follows wd, not f (wd 0.3 gives ~ -35 % again).  11 runs, 1 node.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_forget; mkdir -p $F
T1="--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_100k.npz --turb_cpu 1 --turb_onepass 1 --width 4096 --depth 2 --eval_every 250 --tf32 1 --nval 2048 --batch 256 --steps 7810"
T4="--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_400k.npz --turb_cpu 1 --turb_onepass 1 --width 4096 --depth 2 --eval_every 250 --tf32 1 --nval 4096 --batch 256 --steps 31240"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --lr 1e-3 --rem 0.15 --slow_adapt 1 --sa_cap 1 --sa_kappa_max 1.0 --rem_anchor min"
JOBS=()
JOBS+=("$T4 $K --turb_target delta --seed 1 --wd 3.0")                 # the long one first
for wd in 0 0.3 1.2 4.0; do JOBS+=("$T1 $K --turb_target delta --seed 1 --wd $wd"); done
JOBS+=("$T1 $K --turb_target delta --seed 2 --wd 1.2")
for wd in 0 1.2; do JOBS+=("$T1 $K --turb_target direct --seed 1 --wd $wd"); done
for wd in 0 0.3 1.2; do JOBS+=("$T1 --opt adamw --turb_target delta --seed 1 --lr 1e-4 --wd $wd"); done
echo "${#JOBS[@]} jobs" > $F/njobs
NODE=${1:-0}; NN=${2:-1}; i=0; k=0
for job in "${JOBS[@]}"; do
  if [ $((i % NN)) -eq $NODE ]; then
    CUDA_VISIBLE_DEVICES=$((k % 4)) OMP_NUM_THREADS=8 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
    k=$((k+1)); if [ $((k % 4)) -eq 0 ]; then wait; fi
  fi
  i=$((i+1))
done
wait; echo "NODE $NODE DONE $(date +%T)"
