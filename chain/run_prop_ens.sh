#!/bin/bash
# propagation-corrected ensembles on nid001309: 8 other inits (REAL=8), and single other inits (REAL=1, seeds 1000 / 1001 via SEEDOFF) for spread; both references
cd $SCRATCH/l63_held_rough/runs/ens
srun --jobid=59653997 --overlap -w nid001309 -N 1 --ntasks=1 pkill -f "own_closed_96|own_open_96" ; sleep 3
R="srun --jobid=59653997 --overlap -w nid001309 -N 1 --ntasks=1 --gpus=4 --gpu-bind=none shifter --image=nersc/pytorch:26.01.01 python -u"
EPS=0.03 ADD=1 REAL=8 OWN=0 PROP=1 CUDA_VISIBLE_DEVICES=0 $R ../../code/ensemble_model.py 96 2 0 0.5 1.0 3000 0 96 prop8_96_2_0.5.npz > log_prop8_96_2_0.5.txt 2>&1 &
EPS=0.03 ADD=1 REAL=8 OWN=0 PROP=1 CUDA_VISIBLE_DEVICES=1 $R ../../code/ensemble_model.py 96 3 0 1.0 1.0 3000 0 96 prop8_96_3_1.0.npz > log_prop8_96_3_1.0.txt 2>&1 &
EPS=0.03 ADD=1 REAL=1 OWN=0 PROP=1 CUDA_VISIBLE_DEVICES=2 $R ../../code/ensemble_model.py 96 2 0 0.5 1.0 3000 0 96 prop1_96_2_0.5.npz > log_prop1_96_2_0.5.txt 2>&1 &
EPS=0.03 ADD=1 REAL=1 OWN=0 PROP=1 CUDA_VISIBLE_DEVICES=3 $R ../../code/ensemble_model.py 96 3 0 1.0 1.0 3000 0 96 prop1_96_3_1.0.npz > log_prop1_96_3_1.0.txt 2>&1 &
wait; echo "prop ensemble runs finished $(date)"
