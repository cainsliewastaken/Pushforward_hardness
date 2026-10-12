#!/bin/bash
# Learning-rate test (inside the allocation, inside shifter): refit the three long-window fine-tunes that
# ended above their starting loss, with peak lr_ft = 1e-6 (was 1e-4). Results in runs/<tag>_lr1e-6.
# GPU 3 runs the long-horizon evaluation of all existing runs meanwhile.
cd "$(dirname "$0")"
R=$SCRATCH/multistep_roughness
L=$R/logs/followup2; mkdir -p $L
SUF=_lr1e-6
job() { local gpu=$1 name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu python -u "$@" > $L/$name.log 2>&1; echo "exit=$? $name $(date +%T)" >> $L/exits.txt; }
NEW="0:0.04:unrollp32 1:0.08:unrollp16 2:0.08:unrollp32"
for x in $NEW; do IFS=: read g dt pr <<< "$x"
  ( job $g train_dt${dt}_${pr} train.py --dt $dt --proto $pr --init ft --suffix $SUF --set lr_ft=1.0e-6 \
      && touch $R/runs/dt${dt}_${pr}_ft$SUF/train.ok \
      && job $g measure_dt${dt}_${pr} measure_run.py --dt $dt --proto $pr --init ft --suffix $SUF \
      && touch $R/runs/dt${dt}_${pr}_ft$SUF/measure.ok \
      && job $g gl40_dt${dt}_${pr} recompute_gl.py --points 40 --runs dt${dt}_${pr}_ft$SUF ) &
done
job 3 eval_long_existing eval_long.py --runs $(cd $R/runs && ls -d dt* | grep -v lr1e-6) &
wait
job 3 eval_long_new eval_long.py --runs $(cd $R/runs && ls -d dt*_lr1e-6)
echo "all done $(date +%T)" >> $L/exits.txt
