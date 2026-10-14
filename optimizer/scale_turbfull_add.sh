#!/bin/bash
# Addendum to scale_turbfull.sh, third revision (2026-10-13): WHY the 20-epoch KronQ runs diverge at lr 1e-3 (every budget, both
# batches) when 5000-step runs do not. Two candidate mechanisms without a restoring force: (1) the learned weights drift (log w random-walks
# at rate 0.01 on signal-free modes; the cap, removed in item 43 on 5000-step evidence, was their absolute bound); (2) the budget is
# relative to the CURRENT batch residual (a worsening model is granted larger steps). Long run, lr 1e-3, budget 0.15, batch 256, 31240 steps,
# --diag 1 (u quantiles over time), both targets, seed 1:
#   a. cap back on;  b. weight clamp e^+-1 (--u_clamp 1.0);  c. budget anchored to the step-1 residual norm (--rem_anchor init);
#   d. lr 5e-4 and e. lr 2.5e-4 as the plain lower-lr controls;  f. the held-out-coupled budget (slow_adapt, kappa 1/2): predicted to be the
#   only variant that does not degrade, since the mechanism is rho fixed at 0.15 while the achievable held-out alignment c falls below rho / 2.   12 runs.   usage: scale_turbfull_add.sh NODE NNODES  (indices 100+)
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_turbfull; mkdir -p $F
C="--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_400k.npz --turb_cpu 1 --turb_onepass 1 --width 4096 --depth 2 --eval_every 250 --tf32 1 --nval 4096 --seed 1 --batch 256 --steps 31240 --diag 1"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --rem 0.15"
JOBS=()
for tg in delta direct; do
  JOBS+=("$C $K --turb_target $tg --lr 1e-3 --cap 1")
  JOBS+=("$C $K --turb_target $tg --lr 1e-3 --cap 0 --u_clamp 1.0")
  JOBS+=("$C $K --turb_target $tg --lr 1e-3 --cap 0 --rem_anchor init")
  JOBS+=("$C $K --turb_target $tg --lr 5e-4 --cap 0")
  JOBS+=("$C $K --turb_target $tg --lr 2.5e-4 --cap 0")
  JOBS+=("$C $K --turb_target $tg --lr 1e-3 --cap 0 --slow_adapt 1 --sa_kappa_max 0.5")   # f. held-out-coupled length (item 41 rule, kappa = 1/2): the mechanism's positive control
done
NODE=${1:-0}; NN=${2:-1}; i=0; k=0
for job in "${JOBS[@]}"; do
  if [ $((i % NN)) -eq $NODE ]; then
    CUDA_VISIBLE_DEVICES=$((k % 4)) OMP_NUM_THREADS=8 python -u "$HERE/scale_study.py" $job --out "$F/res_j$((100 + i)).jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$((100 + i)).txt" &
    k=$((k+1)); if [ $((k % 4)) -eq 0 ]; then wait; fi
  fi
  i=$((i+1))
done
wait; echo "NODE $NODE ADD DONE $(date +%T)"
