#!/bin/bash
# Submits the wide-stepper comparison: 1 node, 1 GPU, debug QOS, 30 min, account m5326.
mkdir -p "$SCRATCH/hn_logs"
cd ~/Pushforward_hardness
git log -1 --oneline
git status --short | head -5
sbatch --parsable -A m5326 --image=nersc/pytorch:26.01.01 -C gpu -q debug -N 1 -G 1 -t 00:30:00 \
  -o "$SCRATCH/hn_logs/%x-%j.out" -J wide_steppers \
  --wrap "cd ~/Pushforward_hardness/numerics && srun shifter python -u wide_steppers.py"
echo "log dir: $SCRATCH/hn_logs"
