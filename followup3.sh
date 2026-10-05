#!/bin/bash
# Fixes for long-window training (inside the allocation, inside shifter, 4 GPUs), window 1.28 at dt 0.04, 0.08:
#  GPU 0/1: curriculum from the 0.32-window fine-tuned model: 0.64 window (lr 1e-5), then 1.28 window (lr 1e-6)
#  GPU 2/3: damped loss w_q = exp(-9 t_q), 1.28 window, fine-tuned from the one-step model (lr 1e-5)
# Then long-horizon evaluation of the new runs.
cd "$(dirname "$0")"
R=$SCRATCH/multistep_roughness
L=$R/logs/followup3; mkdir -p $L
job() { local gpu=$1 name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu python -u "$@" > $L/$name.log 2>&1; echo "exit=$? $name $(date +%T)" >> $L/exits.txt; }
ok() { touch $R/runs/$1/train.ok; }
g=0
for dt in 0.08 0.04; do
  ( job $g cur1_dt$dt train.py --dt $dt --proto unroll064 --init ft --from-run dt${dt}_unroll032_ft --suffix _cur --set lr_ft=1.0e-5 \
      && ok dt${dt}_unroll064_ft_cur \
      && job $g cur2_dt$dt train.py --dt $dt --proto unroll128 --init ft --from-run dt${dt}_unroll064_ft_cur --suffix _cur --set lr_ft=1.0e-6 \
      && ok dt${dt}_unroll128_ft_cur ) &
  ( job $((g+2)) damp_dt$dt train.py --dt $dt --proto unroll128g9 --init ft --set lr_ft=1.0e-5 && ok dt${dt}_unroll128g9_ft ) &
  g=$((g+1))
done
wait
job 0 eval_long_new eval_long.py --runs $(cd $R/runs && ls -d dt*_cur dt*_unroll128g9_ft 2>/dev/null)
echo "all done $(date +%T)" >> $L/exits.txt
