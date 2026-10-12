#!/bin/bash
# Step-size rules compared (inside the allocation, inside shifter, 4 GPUs), dt = 0.08, widths 128 and 512, 3 seeds.
# All chains start from the follow-up-6 0.32-window models (dt0.08_unroll032_ft_s4k, which hold the anchor curvature).
#  line rule (no anchor): 0.64 -> 1.28 -> 2.56 with line_c = 0.25 and 0.1 (lr schedule peak 1e-4 is only a cap)
#  sqrt / inv rules: their 0.64 and 1.28 stages exist (follow-up 6); add the 2.56 stage
# 4000 steps per stage (warmup 200), own held-out loss every 100 steps; then long-horizon evaluation.
cd "$(dirname "$0")"
R=$SCRATCH/multistep_roughness
L=$R/logs/followup7; mkdir -p $L
SHORT="steps_ft=4000 warmup=200 log_every=100 n_ckpt_log=6 lr_ft=1.0e-4 $EXTRA_SET"
job() { local gpu=$1 name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu python -u "$@" > $L/$name.log 2>&1; local rc=$?; echo "exit=$rc $name $(date +%T)" >> $L/exits.txt; return $rc; }
root_of() { [ $1 = 256 ] && echo $R || echo $R/pilot/w$1; }

line_chain() {   # gpu width c tag
  local g=$1 W=$2 c=$3 t=$4 root; root=$(root_of $W)
  local S=(--lr-rule line --set "hidden=[$W,$W,$W]" "out_root=$root" $SHORT line_c=$c)
  local prev=dt0.08_unroll032_ft_s4k
  for pr in unroll064 unroll128 unroll256; do
    job $g w${W}_${pr}_$t train.py --dt 0.08 --proto $pr --init ft --from-run $prev --suffix _s4k_$t "${S[@]}" || return 1
    touch $root/runs/dt0.08_${pr}_ft_s4k_$t/train.ok; prev=dt0.08_${pr}_ft_s4k_$t
  done
}

anchored_256() {  # gpu width rule
  local g=$1 W=$2 b=$3 root; root=$(root_of $W)
  job $g w${W}_unroll256_$b train.py --dt 0.08 --proto unroll256 --init ft --from-run dt0.08_unroll128_ft_s4k_$b --suffix _s4k_$b \
      --lr-rule $b --lr-anchor-run dt0.08_unroll032_ft_s4k --set "hidden=[$W,$W,$W]" "out_root=$root" $SHORT \
    && touch $root/runs/dt0.08_unroll256_ft_s4k_$b/train.ok
}

line_chain 0 512 0.25 linec025 &
line_chain 1 512 0.1 linec010 &
( line_chain 2 128 0.25 linec025 & line_chain 2 128 0.1 linec010 & wait ) &
( anchored_256 3 512 sqrt & anchored_256 3 512 inv & anchored_256 3 128 sqrt & anchored_256 3 128 inv & wait ) &
wait
for W in 128 512; do
  root=$(root_of $W)
  job 3 eval_w$W eval_long.py --runs $(cd $root/runs && ls -d dt0.08_*_s4k_linec* dt0.08_unroll256_ft_s4k_sqrt dt0.08_unroll256_ft_s4k_inv) \
      --set "hidden=[$W,$W,$W]" "out_root=$root" $EXTRA_SET
done
echo "all done $(date +%T)" >> $L/exits.txt
