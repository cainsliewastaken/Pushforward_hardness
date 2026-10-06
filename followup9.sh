#!/bin/bash
# Follow-up 9 (inside the allocation, inside shifter; one task per node, 4 GPUs per node, node index from SLURM_NODEID).
# 1. Horizon trade-off along plain gradient-descent runs (gd_tradeoff.py): Euler, dt 0.08, widths 128 and 512,
#    windows 1.28 / 1.92 / 2.56, seeds 0-2, from the line-rule 1.28 models; 4000 steps, held-out measurement every 10.
# 2. Robust evaluation of every run's final models (eval_robust.py): rms, median, 90th, 99th percentile of |e_k(T)|.
# 18 runs in 8 queues (queue = run index mod 8), node n takes queues 4n..4n+3, one queue per GPU; evaluation in 8 parts.
cd "$(dirname "$0")"
R=$SCRATCH/multistep_roughness; N=${SLURM_NODEID:-0}; NN=${NNODES:-2}
L=$R/logs/followup9; O=$R/gd_tradeoff; mkdir -p $L $O
job() { local gpu=$1 name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu python -u "$@" > $L/$name.log 2>&1; echo "exit=$? $name $(date +%T) node$N" >> $L/exits.txt; }
ROOTS="256:$R 64:$R/pilot/w64 128:$R/pilot/w128 512:$R/pilot/w512"
NQ=$((4 * NN))
for g in 0 1 2 3; do
  q=$((4 * N + g))
  job $g robust_part$q eval_robust.py --roots $ROOTS --part $q --nparts $NQ &
  ( i=0; for W in 512 128; do for T in 2.56 1.92 1.28; do for s in 0 1 2; do
      [ $((i % NQ)) -eq $q ] && job $g gd_w${W}_t${T}_s$s gd_tradeoff.py --width $W --window $T --seed $s --steps 4000 --meas-every 10 --out $O/w${W}_t${T}_s$s.json
      i=$((i + 1)); done; done; done ) &
done
wait
echo "node$N done $(date +%T)" >> $L/exits.txt
