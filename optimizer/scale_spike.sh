#!/bin/bash
# Item 56 (2026-10-14): the spike mechanism. Item 55's diagnostic run shows the proposed step's batch output change is 3.5-4.2x the
# residual (rem_rho) while the budget is 0.03: the budget undoes 99 % of every step from ONE forward at the full step, a linear
# extrapolation far outside the Taylor radius, and the realised change is never measured. Hypothesis: the spikes (budget 0.15 at step
# ~10000; direct budget 0.03 after the collapse at 750-2000; budget 0.015 after the collapse at 1000) are where the realised change
# exceeds the budget by orders of magnitude. Three spiking configurations, shortened to cover the onset, each with:
#   (i) --rem_check 1 --diag 1 (the instrument: realised change after the rescale; prediction: >> budget at the onset, ~budget elsewhere)
#   (ii) --rem_iter 2 (remedy A: re-measure and rescale again; prediction: no spike if (i) holds)
#   (iii) --u_clamp 2.3 (remedy B: weights <= 10x; prediction: delays or removes the spike, costs some fit)
#   (iv) --clip_grad 1.0 (control: literal gradient clipping; prediction: no effect, the step is sign-based)
# Null for the mechanism: the realised change stays within ~1.5x the budget through the spike (then the spike is in directions the batch
# cannot see and a parameter-space norm is the next instrument).  12 runs of 1-3 min, 1 node.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_spike; mkdir -p $F
T="--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_400k.npz --turb_cpu 1 --turb_onepass 1 --width 4096 --depth 2 --eval_every 250 --tf32 1 --nval 4096 --batch 256 --seed 1"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --lr 1e-3 --rem_anchor min"
CFG=("--turb_target delta --wd 1.0 --rem 0.15 --steps 12500" "--turb_target direct --wd 0.3 --rem 0.03 --steps 4000" "--turb_target delta --wd 1.0 --rem 0.015 --steps 6000")
VAR=("--rem_check 1 --diag 1" "--rem_iter 2" "--u_clamp 2.3" "--clip_grad 1.0")
JOBS=()
for c in "${CFG[@]}"; do for v in "${VAR[@]}"; do JOBS+=("$T $K $c $v"); done; done
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
