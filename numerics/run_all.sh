#!/bin/bash
# Runs all numerics in one job. Submit from numerics/ with the account/QOS given on the sbatch line, e.g.
#   sbatch -A <account> -C gpu -q shared -G 1 -c 32 -t 01:00:00 --image=nersc/pytorch:26.01.01 \
#          -o $SCRATCH/pf_logs/%x-%j.out -J pf_numerics run_all.sh
set -euo pipefail
cd "$HOME/Pushforward_hardness/numerics"
echo "host $(hostname)  start $(date)"
git -C .. log -1 --oneline
git -C .. status --short
T0=$(date +%s)
for s in test_units.py sine.py lorenz.py; do
  echo "=== $s  $(date)"
  srun -n 1 shifter python -u "$s"
done
srun -n 1 shifter python -u make_report.py > /dev/null
echo "=== done $(date), wall $(( $(date +%s) - T0 )) s"
