#!/bin/bash
# Capped multistep loss (inside the allocation, inside shifter, 4 GPUs), dt = 0.08, widths 128 and 512, 3 seeds.
# A start stops contributing once its rollout error first exceeds --cap (state units; radius of convergence measured 4.3-13).
# All runs start from the follow-up-7 line-rule 1.28 models (dt0.08_unroll128_ft_s4k_linec025), line rule c = 0.25:
#   2.56 capped at 1 and 3, 1.92 capped at 1 and uncapped, 1.28 continued uncapped (control for the extra 4000 steps).
# Reference without cap from the same start: dt0.08_unroll256_ft_s4k_linec025 (follow-up 7). Then long-horizon evaluation.
cd "$(dirname "$0")"
R=$SCRATCH/multistep_roughness
L=$R/logs/followup8; mkdir -p $L
SHORT="steps_ft=4000 warmup=200 log_every=100 n_ckpt_log=6 lr_ft=1.0e-4 line_c=0.25 $EXTRA_SET"
FROM=dt0.08_unroll128_ft_s4k_linec025
job() { local gpu=$1 name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu python -u "$@" > $L/$name.log 2>&1; local rc=$?; echo "exit=$rc $name $(date +%T)" >> $L/exits.txt; return $rc; }
root_of() { [ $1 = 256 ] && echo $R || echo $R/pilot/w$1; }
run() {  # gpu width proto suffix [--cap x]
  local g=$1 W=$2 pr=$3 sfx=$4; shift 4; local root; root=$(root_of $W)
  job $g w${W}_${pr}${sfx} train.py --dt 0.08 --proto $pr --init ft --from-run $FROM --suffix _s4k_linec025$sfx \
      --lr-rule line --set "hidden=[$W,$W,$W]" "out_root=$root" $SHORT "$@" && touch $root/runs/dt0.08_${pr}_ft_s4k_linec025$sfx/train.ok
}
( run 0 512 unroll256 _cap1 --cap 1 & run 0 512 unroll256 _cap3 --cap 3 & wait ) &
( run 1 512 unroll192 _cap1 --cap 1 & run 1 512 unroll192 "" & run 1 512 unroll128 _more & wait ) &
( run 2 128 unroll256 _cap1 --cap 1 & run 2 128 unroll256 _cap3 --cap 3 & wait ) &
( run 3 128 unroll192 _cap1 --cap 1 & run 3 128 unroll192 "" & run 3 128 unroll128 _more & wait ) &
wait
for W in 128 512; do
  root=$(root_of $W)
  job 3 eval_w$W eval_long.py --runs $FROM dt0.08_unroll256_ft_s4k_linec025 \
      $(cd $root/runs && ls -d dt0.08_unroll*_ft_s4k_linec025_* dt0.08_unroll192_ft_s4k_linec025 2>/dev/null) \
      --set "hidden=[$W,$W,$W]" "out_root=$root" $EXTRA_SET
done
echo "all done $(date +%T)" >> $L/exits.txt
