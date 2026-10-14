#!/bin/bash
# The hybrid on turbulence (2026-10-14): 400k pairs, 20 epochs, batch 256, TF32 network, --qr_guard 1 --refresh_hp 1 --reorth_transport 1,
# with and without --rot_hp 1 (activation rotations fp32), on the other session's two best settings:
#   capped budget (rem 0.15 capped by the held-out alignment, anchor min) + decay: increment wd 1.0 (KronQFast 0.1109), direct wd 0.3 (0.0628 / 0.0631);
#   no budget, lr 2e-4, decay 1e-3 per step (item 57: increment 0.1148).   AdamW best: increment 0.156, direct 0.060.
# Predictions: no spikes; increment at or below the KronQFast values (the hybrid gained 5-10 % on the rough field preset, more with --rot_hp);
# direct a tie with KronQFast.  8 runs, one per GPU, 2 nodes.  usage: scale_turbhybrid.sh NODE.  Frozen code in $F/code.
F=$SCRATCH/l63_held_rough/runs/scale_turbhybrid
T="--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_400k.npz --turb_cpu 1 --turb_onepass 1 --width 4096 --depth 2 --eval_every 250 --tf32 1 --nval 4096 --batch 256 --steps 31240"
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --qr_guard 1 --refresh_hp 1 --reorth_transport 1"
CAP="--lr 1e-3 --rem 0.15 --slow_adapt 1 --sa_cap 1 --sa_kappa_max 1.0 --rem_anchor min"
NOB="--lr 2e-4 --rem 0 --wd_abs 1e-3"
JOBS=("$T $K $CAP --turb_target delta --wd 1.0 --seed 1"
      "$T $K $CAP --turb_target delta --wd 1.0 --seed 1 --rot_hp 1"
      "$T $K $NOB --turb_target delta --seed 1"
      "$T $K $NOB --turb_target delta --seed 1 --rot_hp 1"
      "$T $K $CAP --turb_target direct --wd 0.3 --seed 1"
      "$T $K $CAP --turb_target direct --wd 0.3 --seed 1 --rot_hp 1"
      "$T $K $NOB --turb_target direct --seed 1"
      "$T $K $CAP --turb_target delta --wd 1.0 --seed 2 --rot_hp 1")
echo "${#JOBS[@]} jobs" > $F/njobs
NODE=${1:-0}
for g in 0 1 2 3; do
  i=$((4 * NODE + g))
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 python -u "$F/code/optimizer/scale_study.py" ${JOBS[$i]} --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
done
wait; echo "NODE $NODE DONE $(date +%T)"
