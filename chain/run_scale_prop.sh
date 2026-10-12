#!/bin/bash
# corrected own-init model on the scaling ground truth (runs/gt_scale): width 1024 d2, width 512 d3, depth 6 and 8 at H96, n = 800
cd $SCRATCH/l63_held_rough/runs; mkdir -p ensS
R="srun --jobid=59653997 --overlap -N 1 --ntasks=1 --gpus=4 --gpu-bind=none shifter --image=nersc/pytorch:26.01.01 python -u $(dirname "$(readlink -f "$0")")/ensemble_model.py"
launch() { # node gpu H depth D n tag
  N=$6 EPS=0.03 ADD=1 REAL=1 OWN=1 PROP=1 CUDA_VISIBLE_DEVICES=$2 $R $3 $4 0 $5 1.0 3000 0 $3 ensS/own_$7.npz > ensS/log_own_$7.txt 2>&1 &
  echo "launched $7 on $1 gpu$2"; }
# wrap launch with the node selection
N_=0
run() { node=$1; shift; (export SRUN_W=$node; launch "$node" "$@"); }
launch_on() { node=$1; gpu=$2; H=$3; d=$4; D=$5; n=$6; tag=$7
  N=$n EPS=0.03 ADD=1 REAL=1 OWN=1 PROP=1 CUDA_VISIBLE_DEVICES=$gpu srun --jobid=59653997 --overlap -w $node -N 1 --ntasks=1 --gpus=4 --gpu-bind=none shifter --image=nersc/pytorch:26.01.01 python -u $(dirname "$(readlink -f "$0")")/ensemble_model.py $H $d 0 $D 1.0 3000 0 $H ensS/own_$tag.npz > ensS/log_own_$tag.txt 2>&1 &
  echo "launched $tag on $node gpu$gpu"; }
launch_on nid001308 0 1024 2 0.5 200 H1024_D0.5_depth2_s0
launch_on nid001308 1 1024 2 1.0 200 H1024_D1.0_depth2_s0
launch_on nid001309 2 512 3 0.5 200 H512_D0.5_depth3_s0
launch_on nid001309 3 512 3 1.0 200 H512_D1.0_depth3_s0
launch_on nid001312 0 96 6 0.5 200 H96_D0.5_depth6_s0
launch_on nid001312 1 96 6 1.0 200 H96_D1.0_depth6_s0
launch_on nid001313 2 96 8 0.5 200 H96_D0.5_depth8_s0
launch_on nid001313 3 96 8 1.0 200 H96_D1.0_depth8_s0
launch_on nid001308 2 96 2 0.5 800 H96_D0.5_depth2_s0_n800
launch_on nid001309 0 96 3 1.0 800 H96_D1.0_depth3_s0_n800
wait; echo "scale prop runs finished $(date)"
