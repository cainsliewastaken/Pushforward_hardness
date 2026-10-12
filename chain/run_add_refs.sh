#!/bin/bash
# four additive-transfer reference runs on one node (one per GPU), run in the foreground of a background Bash call
cd $SCRATCH/l63_held_rough/runs/ens
i=0
for cfg in "2 0.5" "3 1.0"; do set -- $cfg; d=$1; D=$2
  for mode in gauss real32; do
    if [ $mode = real32 ]; then R=32; else R=0; fi
    ADD=1 EPS=0.03 REAL=$R CUDA_VISIBLE_DEVICES=$i srun --jobid=59653997 --overlap -w nid001309 -N 1 --ntasks=1 --gpus=4 --gpu-bind=none shifter --image=nersc/pytorch:26.01.01 python -u ../../code/ensemble_model.py 96 $d 0 $D 1.0 3000 0 2048 add_${mode}_96_${d}_${D}.npz > log_add_${mode}_96_${d}_${D}.txt 2>&1 &
    echo "launched add $mode d=$d D=$D gpu=$i pid=$!"; i=$((i+1))
  done; done
wait; echo "all add refs finished $(date)"
