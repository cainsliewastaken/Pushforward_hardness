#!/bin/bash
# Follow-up 9 (inside the allocation, inside shifter, 4 GPUs).
# 1. Horizon trade-off along plain gradient-descent runs (gd_tradeoff.py): Euler, dt 0.08, widths 128 and 512,
#    windows 1.28 / 1.92 / 2.56, seeds 0-2, from the line-rule 1.28 models; 4000 steps, held-out measurement every 10.
# 2. Robust evaluation of every run's final models (eval_robust.py): rms, median, 90th, 99th percentile of |e_k(T)|.
cd "$(dirname "$0")"
R=$SCRATCH/multistep_roughness
L=$R/logs/followup9; O=$R/gd_tradeoff; mkdir -p $L $O
job() { local gpu=$1 name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu python -u "$@" > $L/$name.log 2>&1; echo "exit=$? $name $(date +%T)" >> $L/exits.txt; }
ROOTS="256:$R 64:$R/pilot/w64 128:$R/pilot/w128 512:$R/pilot/w512"
gd() { job $1 gd_w$2_t$3_s$4 gd_tradeoff.py --width $2 --window $3 --seed $4 --steps 4000 --meas-every 10 --out $O/w$2_t$3_s$4.json; }
for g in 0 1 2 3; do job $g robust_part$g eval_robust.py --roots $ROOTS --part $g --nparts 4 & done
i=0; for W in 512 128; do for T in 2.56 1.92 1.28; do for s in 0 1 2; do
  g=$((i % 4)); eval "Q$g=\"\${Q$g} $W:$T:$s\""; i=$((i + 1)); done; done; done
for g in 0 1 2 3; do
  ( eval "items=\$Q$g"; set -- $items; half=$(( ($# + 1) / 2 )); k=0
    ( for it in "$@"; do k=$((k+1)); [ $k -le $half ] && gd $g ${it//:/ }; done ) &
    ( for it in "$@"; do k=$((k+1)); [ $k -gt $half ] && gd $g ${it//:/ }; done ) &
    wait ) &
done
wait
echo "all done $(date +%T)" >> $L/exits.txt
