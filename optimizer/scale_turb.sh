#!/bin/bash
# Second pass (item 45 fix): 100000-pair slab (CACHE env, default turb64_lead10_100k.npz), output OUT (default scale_turb2); val_best reported.
# 2D turbulence one-step emulator (make_turb_cache.py: 64 x 64 block-mean vorticity, lead 10 = time step 0.1; Flop_Constrained data).
# Targets: delta (normalised increment: the rough target, the Euler step's residual) and direct (next state: the smooth control).
# Minimal KronQ (cap off) lr {5e-4, 1e-3, 2e-3} x budget {0.15, 0.3}; AdamW lr {1e-4, 3e-4, 1e-3} at 5000 steps and at 25000 steps
# (~equal wall, to be checked against the measured step times); seeds 1-2. 48 runs.  usage: scale_turb.sh NODE NNODES
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/${OUT:-scale_turb2}; mkdir -p $F
C="--problem turb --turb_cache $SCRATCH/l63_held_rough/data/${CACHE:-turb64_lead10_100k.npz} --width 4096 --depth 2 --batch 256 --eval_every 250 --tf32 1 --nval 2048"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --steps 5000"
JOBS=()
for seed in 1 2; do for tg in delta direct; do
  for lr in 5e-4 1e-3 2e-3; do for rem in 0.15 0.3; do JOBS+=("$C $K --turb_target $tg --seed $seed --lr $lr --rem $rem"); done; done
  for lr in 1e-4 3e-4 1e-3; do for st in 5000 25000; do JOBS+=("$C --opt adamw --turb_target $tg --seed $seed --lr $lr --steps $st"); done; done
done; done
echo "${#JOBS[@]} jobs" > $F/njobs
NODE=${1:-0}; NN=${2:-1}; i=0; k=0
for job in "${JOBS[@]}"; do
  if [ $((i % NN)) -eq $NODE ]; then
    CUDA_VISIBLE_DEVICES=$((k % 4)) OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
    k=$((k+1)); if [ $((k % 4)) -eq 0 ]; then wait; fi
  fi
  i=$((i+1))
done
wait; echo "NODE $NODE DONE $(date +%T)"
