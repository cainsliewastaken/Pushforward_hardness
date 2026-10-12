#!/bin/bash
# own-unit model WITH propagation terms: t=8 dump (compare with real), and closed loop to T=3000 for both references
cd $SCRATCH/l63_held_rough/runs/ens; mkdir -p resolve
R="srun --jobid=59653997 --overlap -w nid001312 -N 1 --ntasks=1 --gpus=4 --gpu-bind=none shifter --image=nersc/pytorch:26.01.01 python -u"
EPS=0.01 ADD=1 REAL=1 OWN=1 PROP=1 DUMP=$SCRATCH/l63_held_rough/runs/ens/resolve/model_prop_t8.pt CUDA_VISIBLE_DEVICES=0 $R ../../code/ensemble_model.py 96 2 0 0.5 1.0 8 0 96 resolve/prop8.npz > resolve/log_prop8.txt 2>&1 &
EPS=0.03 ADD=1 REAL=1 OWN=1 PROP=1 CUDA_VISIBLE_DEVICES=1 $R ../../code/ensemble_model.py 96 2 0 0.5 1.0 3000 0 96 prop_closed_96_2_0.5.npz > log_prop_closed_96_2_0.5.txt 2>&1 &
EPS=0.03 ADD=1 REAL=1 OWN=1 PROP=1 CUDA_VISIBLE_DEVICES=2 $R ../../code/ensemble_model.py 96 3 0 1.0 1.0 3000 0 96 prop_closed_96_3_1.0.npz > log_prop_closed_96_3_1.0.txt 2>&1 &
wait; echo "prop runs finished $(date)"
