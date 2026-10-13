#!/bin/bash
# Run the scaling-study job list on all nodes of an allocation: one srun task per node, SLOTS runs per GPU,
# static round-robin over the (longest-first) list. usage (inside srun, one task per node): scale_launch.sh SLOTS
SLOTS=${1:-3}
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale
NODE=${SLURM_NODEID:-0}; NN=${SLURM_NNODES:-1}; TOT=$((NN * 4 * SLOTS))
for w in $(seq 0 $((4 * SLOTS - 1))); do
  slot=$((NODE * 4 * SLOTS + w))
  ( i=0
    while IFS= read -r line; do
      if [ $((i % TOT)) -eq $slot ]; then
        CUDA_VISIBLE_DEVICES=$((w % 4)) OMP_NUM_THREADS=2 python -u "$HERE/scale_study.py" $line --out "$F/res_s$slot.jsonl" 2>&1 | grep -v -i 'pynvml\|FutureWarning'
      fi
      i=$((i+1))
    done < "$F/jobs.txt" ) > "$F/log_s$slot.txt" 2>&1 &
done
wait; echo "NODE $NODE ALLDONE $(date +%T)"
