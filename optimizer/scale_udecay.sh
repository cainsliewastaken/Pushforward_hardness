#!/bin/bash
# Item 51 (2026-10-13): the learned weight's hypergradient is a per-mode next-batch transfer test with zero mean on a non-transferring
# mode and no decay, so u random-walks there (weight norms 73 -> 127 in item 48). Test: shrink u toward a floor each step (--u_decay,
# --u_floor -2.3, w_floor 0.1), two rates (memory 3300 / 1000 steps; bias on a transferring mode at u = 0: 0.07 / 0.23 in correlation
# units). Turbulence (all 400k pairs, 20 epochs, batch 256, lr 1e-3, budget 0.15, capped budget kappa 1.0 so nothing diverges), both
# targets x 2 rates, plus the weight-decay sweep (wd 0.1, 0.3; item 50 had 0.01 only) x 2 targets: 8 turbulence runs, 2 waves of 4 GPUs.
# Field (standard preset, cap off, budget 0.2, 5000 steps, seed 1; reference 0.0622 / 1.42e-4), both targets x 2 rates: 4 runs.
# Predictions: field within seed noise; turbulence train / validation gap narrows and the matched-training-loss deficit (24-34 %) falls.
# Null: no change on turbulence = the memorising pushes do have positive next-batch agreement.   usage: scale_udecay.sh NODE NNODES
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_udecay; mkdir -p $F
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --lr 1e-3"
T="--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_400k.npz --turb_cpu 1 --turb_onepass 1 --width 4096 --depth 2 --eval_every 250 --tf32 1 --nval 4096 --seed 1 --batch 256 --steps 31240 --rem 0.15 --slow_adapt 1 --sa_cap 1 --sa_kappa_max 1.0 --rem_anchor min"
Fd="--width 4096 --depth 2 --batch 256 --eval_every 250 --tf32 1 --steps 5000 --rem 0.2 --seed 1"
JOBS=()
for tg in delta direct; do for ud in 0.0003 0.001; do JOBS+=("$T $K --turb_target $tg --u_decay $ud"); done; done
for tg in delta direct; do for wd in 0.1 0.3; do JOBS+=("$T $K --turb_target $tg --wd $wd"); done; done
for p in 2 0; do for ud in 0.0003 0.001; do JOBS+=("$Fd $K --p $p --u_decay $ud"); done; done
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
