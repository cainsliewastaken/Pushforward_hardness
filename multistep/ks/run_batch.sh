#!/bin/bash
# KS comparison batch inside an interactive allocation: 4 runs at a time, one per GPU, two waves.
# Usage (on Perlmutter, inside the allocation):  bash ks/run_batch.sh <jobid> <init final.pt for fine-tuning>
# Wave 1 (from scratch, lr 1e-3, 8000 steps, window p=20 = 10 time units): free, kappa 0.5, 0.2, 0.05.
# Wave 2 (fine-tuned from the one-step model, lr 1e-4, 8000 steps): the same four.
set -u
JOB=$1; INIT=$2
cd ~/multistep_roughness
K=$SCRATCH/multistep_roughness/ks; mkdir -p $K/logs
run() {  # gpu tag args...
  local gpu=$1 tag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu srun --jobid=$JOB --overlap -N 1 --ntasks=1 --gpus=1 --exact \
    shifter --image=nersc/pytorch:26.01.01 python -u ks/train.py --tag $tag "$@" > $K/logs/$tag.log 2>&1
}
echo "wave 1 start $(date)"
run 0 free_s0     --mode free  --p 20 &
run 1 k0.5_s0     --mode kappa --kappa 0.5  --p 20 &
run 2 k0.2_s0     --mode kappa --kappa 0.2  --p 20 &
run 3 k0.05_s0    --mode kappa --kappa 0.05 --p 20 &
wait
echo "wave 2 start $(date)"
run 0 ft_free_s0  --mode free  --p 20 --init $INIT --lr 1e-4 &
run 1 ft_k0.5_s0  --mode kappa --kappa 0.5  --p 20 --init $INIT --lr 1e-4 &
run 2 ft_k0.2_s0  --mode kappa --kappa 0.2  --p 20 --init $INIT --lr 1e-4 &
run 3 ft_k0.05_s0 --mode kappa --kappa 0.05 --p 20 --init $INIT --lr 1e-4 &
wait
echo "batch done $(date)"
