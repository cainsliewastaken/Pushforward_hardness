#!/bin/bash
# Ground-truth extension: 24 GD runs on 2 nodes (12 per node, 3 per GPU). usage: run_gt_ext.sh JOBID
# cfg = H:Delta:depth:seed:n:scale:T:OPT:LOSS:LRMULT:BATCH:WD:LAYERS   (longest runs first)
J=$1; cd $SCRATCH/l63_held_rough/runs; mkdir -p gt_ext
NODES=($(squeue -j $J -h -o "%N" | tr -d '\n' | xargs scontrol show hostnames)); echo "nodes: ${NODES[@]}"
A="256:1.0:8:0:200:1:3000:gd:sq:1:0:0:0 256:0.5:8:0:200:1:3000:gd:sq:1:0:0:0 256:1.0:6:0:200:1:3000:gd:sq:1:0:0:0 256:0.5:6:0:200:1:3000:gd:sq:1:0:0:0 256:1.0:5:0:200:1:3000:gd:sq:1:0:0:0 256:0.5:5:0:200:1:3000:gd:sq:1:0:0:0 256:1.0:4:0:200:1:3000:gd:sq:1:0:0:0 256:0.5:4:0:200:1:3000:gd:sq:1:0:0:0 256:1.0:3:0:200:0.5:12000:gd:sq:1:0:0:0 256:0.5:3:0:200:0.5:12000:gd:sq:1:0:0:0 256:1.0:2:0:200:0.5:12000:gd:sq:1:0:0:0 256:0.5:2:0:200:0.5:12000:gd:sq:1:0:0:0"
B="128:1.0:8:0:200:1:3000:gd:sq:1:0:0:0 128:0.5:8:0:200:1:3000:gd:sq:1:0:0:0 128:1.0:6:0:200:1:3000:gd:sq:1:0:0:0 128:0.5:6:0:200:1:3000:gd:sq:1:0:0:0 128:1.0:5:0:200:1:3000:gd:sq:1:0:0:0 128:0.5:5:0:200:1:3000:gd:sq:1:0:0:0 128:1.0:4:0:200:1:3000:gd:sq:1:0:0:0 128:0.5:4:0:200:1:3000:gd:sq:1:0:0:0 96:1.0:3:0:200:0.5:12000:gd:sq:1:0:0:0 96:0.5:3:0:200:0.5:12000:gd:sq:1:0:0:0 96:1.0:2:0:200:0.5:12000:gd:sq:1:0:0:0 96:0.5:2:0:200:0.5:12000:gd:sq:1:0:0:0"
srun --jobid=$J --overlap -w ${NODES[0]} -N 1 --ntasks=1 --gpus=4 --gpu-bind=none shifter --image=nersc/pytorch:26.01.01 bash $(dirname "$(readlink -f "$0")")/run_design2.sh $SCRATCH/l63_held_rough/runs/gt_ext 4 $A > gt_ext_A.log 2>&1 &
srun --jobid=$J --overlap -w ${NODES[1]} -N 1 --ntasks=1 --gpus=4 --gpu-bind=none shifter --image=nersc/pytorch:26.01.01 bash $(dirname "$(readlink -f "$0")")/run_design2.sh $SCRATCH/l63_held_rough/runs/gt_ext 4 $B > gt_ext_B.log 2>&1 &
wait; echo "exit=$?" > gt_ext.done; echo "all gt_ext runs finished $(date)"; scancel $J
