#!/bin/bash
# own-unit model: stage-resolution scan to T=25 (EPS 0.03/0.01/0.003) + dump at t=8 (EPS 0.01), and the real GD snapshot at t=8
cd $SCRATCH/l63_held_rough/runs/ens; mkdir -p resolve
R="srun --jobid=59653997 --overlap -w nid001312 -N 1 --ntasks=1 --gpus=4 --gpu-bind=none shifter --image=nersc/pytorch:26.01.01 python -u"
i=0
for e in 0.03 0.01 0.003; do
  EPS=$e ADD=1 REAL=1 OWN=1 CUDA_VISIBLE_DEVICES=$i $R ../../code/ensemble_model.py 96 2 0 0.5 1.0 25 0 96 resolve/own25_eps$e.npz > resolve/log_own25_eps$e.txt 2>&1 &
  i=$((i+1))
done
EPS=0.01 ADD=1 REAL=1 OWN=1 DUMP=$SCRATCH/l63_held_rough/runs/ens/resolve/model_t8.pt CUDA_VISIBLE_DEVICES=3 $R ../../code/ensemble_model.py 96 2 0 0.5 1.0 8 0 96 resolve/own8.npz > resolve/log_own8.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 $R ../../code/real_snapshot.py 96 2 0 0.5 8 $SCRATCH/l63_held_rough/runs/ens/resolve/real_t8.pt > resolve/log_real8.txt 2>&1 &
wait; echo "resolve runs finished $(date)"
