#!/bin/bash
# Pilot (run inside the allocation, inside shifter): sets the step budget and checks the width.
#  * production onestep runs (width 256, all dt, full config) -> main out_root, reused by sweep.py
#  * width check: onestep at widths 128 and 512 for dt = 0.01, 0.08 -> out_root/pilot/w<width>
#  * timing: unroll064 from scratch at dt = 0.01 for 2000 steps, then measure its final checkpoint
cd "$(dirname "$0")"
R=$SCRATCH/multistep_roughness
mkdir -p $R/logs $R/pilot
run() { local gpu=$1 log=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu python -u "$@" > $log 2>&1; echo "exit=$? $*" >> $R/pilot/exits.txt; }

for i in 0 1 2 3; do
  dt=$(echo 0.01 0.02 0.04 0.08 | cut -d' ' -f$((i+1)))
  ( run $i $R/logs/train_dt${dt}_onestep.log train.py --dt $dt --proto onestep \
      && touch $R/runs/dt${dt}_onestep/train.ok ) &
done
g=0
for w in 128 512; do
  for dt in 0.01 0.08; do
    run $g $R/pilot/w${w}_dt${dt}.log train.py --dt $dt --proto onestep --set "hidden=[$w,$w,$w]" out_root=$R/pilot/w$w &
    g=$(( (g+1) % 4 ))
  done
done
run 2 $R/logs/checks.log checks.py &
( run 3 $R/pilot/timing_unroll.log train.py --dt 0.01 --proto unroll064 --init scratch --set steps=2000 out_root=$R/pilot/timing \
  && run 3 $R/pilot/timing_measure.log measure_run.py --dt 0.01 --proto unroll064 --init scratch --steps 2000 --set out_root=$R/pilot/timing ) &
wait
