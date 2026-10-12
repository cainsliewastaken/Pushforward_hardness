#!/bin/bash
# Submits soft_bounds.py: 1 GPU, debug QOS, 20 min. The account is passed as the first argument.
set -e
ACCOUNT="$1"
mkdir -p "$SCRATCH/pf_logs"
cd "$HOME/Pushforward_hardness/numerics"
sbatch --parsable -A "$ACCOUNT" --image=nersc/pytorch:26.01.01 -C gpu -q debug -N 1 -G 1 -t 00:20:00 \
  -o "$SCRATCH/pf_logs/%x-%j.out" -J soft_bounds \
  --wrap "cd $HOME/Pushforward_hardness/numerics && srun shifter python -u soft_bounds.py"
