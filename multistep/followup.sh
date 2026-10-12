#!/bin/bash
# Follow-up (run inside the allocation, inside shifter, 4 GPUs):
#  * fixed-step-count test: fine-tuned unrolled runs for the (p, dt) pairs the T_train protocols lack,
#    so p = 16 and p = 32 exist at every dt:  p16 @ dt 0.01, 0.08;  p32 @ dt 0.04, 0.08
#  * 40-point Gauss-Legendre recompute of the Bbar quantities (final checkpoints, all runs)
#  * long-horizon rollout error (T up to 5.12) for all runs
cd "$(dirname "$0")"
R=$SCRATCH/multistep_roughness
L=$R/logs/followup; mkdir -p $L
job() { local gpu=$1 name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu python -u "$@" > $L/$name.log 2>&1; echo "exit=$? $name" >> $L/exits.txt; }

# phase 1: new trainings (one per GPU) alongside the recompute of the existing 20 runs (split by dt)
NEW="0:0.08:unrollp32 1:0.04:unrollp32 2:0.08:unrollp16 3:0.01:unrollp16"
for x in $NEW; do IFS=: read g dt pr <<< "$x"
  ( job $g train_dt${dt}_${pr}_ft train.py --dt $dt --proto $pr --init ft && touch $R/runs/dt${dt}_${pr}_ft/train.ok ) &
done
g=0
for dt in 0.01 0.02 0.04 0.08; do
  runs=$(cd $R/runs && ls -d dt${dt}_* | grep -v unrollp)
  job $g gl40_dt$dt recompute_gl.py --points 40 --runs $runs &
  g=$((g+1))
done
wait
echo "phase1 done $(date)" >> $L/exits.txt

# phase 2: measure + recompute the new runs; long-horizon evaluation of everything
for x in $NEW; do IFS=: read g dt pr <<< "$x"
  ( job $g measure_dt${dt}_${pr}_ft measure_run.py --dt $dt --proto $pr --init ft && touch $R/runs/dt${dt}_${pr}_ft/measure.ok \
    && job $g gl40_dt${dt}_${pr} recompute_gl.py --points 40 --runs dt${dt}_${pr}_ft ) &
done
job 2 eval_long eval_long.py &
wait
echo "phase2 done $(date)" >> $L/exits.txt
