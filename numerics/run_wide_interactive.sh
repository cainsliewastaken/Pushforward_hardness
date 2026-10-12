#!/bin/bash
# Runs wide_steppers.py inside the existing interactive allocation, one width per idle GPU.
JOB=59112688
mkdir -p "$SCRATCH/hn_logs"
cd ~/Pushforward_hardness/numerics
echo "host $(hostname)"
for pair in "0 512" "1 2048"; do
  set -- $pair
  CUDA_VISIBLE_DEVICES=$1 WIDE_WIDTHS=$2 nohup srun --jobid=$JOB --overlap -N 1 --ntasks=1 \
    shifter --image=nersc/pytorch:26.01.01 python -u wide_steppers.py > "$SCRATCH/hn_logs/wide_steppers-int-w$2.out" 2>&1 &
  echo "width $2 on GPU $1: pid $!"
done
sleep 5
squeue -h -s -j $JOB -o '%i %j %M' | head
