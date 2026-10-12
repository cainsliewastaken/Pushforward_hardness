#!/bin/bash
# Item 5: hyperparameter-transfer sweep. Ground truth GD (harness logs E'K0E and E'KE), width x output-scale, depth 2 (+ depth 4 at two widths).
# usage: run_transfer.sh JOBID   (1 node, 4 GPUs; 24 runs, 3 per GPU)   cfg = H:Delta:depth:seed:n:scale:T:...
J=$1; cd $SCRATCH/l63_held_rough/runs; mkdir -p gt_transfer
NODE=$(squeue -j $J -h -o "%N" | xargs scontrol show hostnames | head -n1); echo "node $NODE"
C=""
for H in 48 96 192 384; do for s in 0.25 0.5 1 2; do T=$(python3 -c "print(int(3000/($s*$s)))"); C="$C $H:0.5:2:0:200:$s:$T:gd:sq:1:0:0:0"; done; done
for H in 96 192; do for s in 0.25 0.5 1 2; do T=$(python3 -c "print(int(3000/($s*$s)))"); C="$C $H:0.5:4:0:200:$s:$T:gd:sq:1:0:0:0"; done; done
echo "configs: $C"
srun --jobid=$J --overlap -w $NODE -N 1 --ntasks=1 --gpus=4 --gpu-bind=none shifter --image=nersc/pytorch:26.01.01 bash $(dirname "$(readlink -f "$0")")/run_design2.sh $SCRATCH/l63_held_rough/runs/gt_transfer 4 $C > gt_transfer.log 2>&1
echo "exit=$?" > gt_transfer.done; echo "transfer sweep finished $(date)"; scancel $J
