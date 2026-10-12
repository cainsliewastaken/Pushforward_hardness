#!/bin/bash
# self-test: ensemble = the run's own 96 units (REAL=1 OWN=1), open- and closed-loop, additive transfer
cd $SCRATCH/l63_held_rough/runs/ens
i=0
for cfg in "2 0.5" "3 1.0"; do set -- $cfg; d=$1; D=$2
  for mode in open closed; do
    if [ $mode = open ]; then OL=$SCRATCH/l63_held_rough/runs/sweepA/late_H96_D${D}_depth${d}_s0.npy; else OL=""; fi
    OPENLOOP=$OL ADD=1 EPS=0.03 REAL=1 OWN=1 CUDA_VISIBLE_DEVICES=$i srun --jobid=59653997 --overlap -w nid001309 -N 1 --ntasks=1 --gpus=4 --gpu-bind=none shifter --image=nersc/pytorch:26.01.01 python -u ../../code/ensemble_model.py 96 $d 0 $D 1.0 3000 0 96 own_${mode}_96_${d}_${D}.npz > log_own_${mode}_96_${d}_${D}.txt 2>&1 &
    echo "launched own $mode d=$d D=$D gpu=$i pid=$!"; i=$((i+1))
  done; done
wait; echo "all selftest runs finished $(date)"
