#!/bin/bash
# Item 53 (2026-10-13): the fair weight-decay comparison on turbulence (item 51 found KronQ wd 0.3 at 0.127 / 0.063 against AdamW's best
# 0.156 / 0.066 with AdamW's own decay tried only at 0.1, lr 3e-4). All 400k pairs, 20 epochs, batch 256:
#   AdamW lr 1e-4 / 1e-3 x wd 0.3 / 1.0 x both targets (8);  KronQ (capped preset) wd 1.0 both targets (2) and wd 0.3 seed 2 (2);
#   whitened AdamW (pca / full, lr 1e-3, wd 0) both targets (4): does the field's whitening result (item 52) carry to turbulence?
# Predictions: AdamW gains from wd 0.3 but less than KronQ did (its gap was 1.4x, KronQ's 3x); KronQ wd 1.0 over-regularises (train loss
# rises above val's gain); whitening helps AdamW on the increment and not on direct.  16 runs, 2 nodes.   usage: scale_fairwd.sh NODE NNODES
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_fairwd; mkdir -p $F
T="--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_400k.npz --turb_cpu 1 --turb_onepass 1 --width 4096 --depth 2 --eval_every 250 --tf32 1 --nval 4096 --batch 256 --steps 31240"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --lr 1e-3 --rem 0.15 --slow_adapt 1 --sa_cap 1 --sa_kappa_max 1.0 --rem_anchor min"
JOBS=()
for tg in delta direct; do
  JOBS+=("$T $K --turb_target $tg --seed 1 --wd 1.0")
  JOBS+=("$T --opt adamw --turb_target $tg --seed 1 --lr 1e-3 --wd 0.3")
  JOBS+=("$T --opt adamw --turb_target $tg --seed 1 --lr 1e-4 --wd 0.3")
  JOBS+=("$T $K --turb_target $tg --seed 2 --wd 0.3")
  JOBS+=("$T --opt adamw --turb_target $tg --seed 1 --lr 1e-3 --wd 1.0")
  JOBS+=("$T --opt adamw --turb_target $tg --seed 1 --lr 1e-4 --wd 1.0")
  JOBS+=("$T --opt adamw --turb_target $tg --seed 1 --lr 1e-3 --wd 0 --whiten pca")
  JOBS+=("$T --opt adamw --turb_target $tg --seed 1 --lr 1e-3 --wd 0 --whiten full")
done
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
