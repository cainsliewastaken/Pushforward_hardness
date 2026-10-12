#!/bin/bash
# small-step GD ground truth (LRMULT=0.25) for the two references, on nid001312 (within job 59653997)
cd $(dirname "$(readlink -f "$0")")
srun --jobid=59653997 --overlap -w nid001312 -N 1 --ntasks=1 --gpus=4 --gpu-bind=none shifter --image=nersc/pytorch:26.01.01 bash run_design2.sh $SCRATCH/l63_held_rough/runs/gt_lr 4 96:0.5:2:0:200:1:3000:gd:sq:0.25:0:0:0 96:1.0:3:0:200:1:3000:gd:sq:0.25:0:0:0
echo "lrtest finished $(date)"
