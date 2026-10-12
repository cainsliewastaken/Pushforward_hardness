#!/bin/bash
# Curvature-scaled learning rates for the width test (inside the allocation, inside shifter, 4 GPUs), dt = 0.08.
#  Step 1 (GPU 3): curvature of every stage of the existing width chains (trainability.py), widths 64-512.
#  Step 2: short window chains (4000 steps per stage, warmup 200) at widths 64-512, from each width's one-step model:
#    0.16 (lr 1e-4) -> 0.32 (lr 1e-4, anchor: curvature recorded per model) -> then three branches for 0.64 -> 1.28:
#    fixed  : lr 1e-5, 1e-6 (the earlier protocol; checks that short runs reproduce the full-run outcomes)
#    sqrt   : per-model peak lr = 1e-4 * (lam_anchor / lam_start)^(1/2)
#    inv    : per-model peak lr = 1e-4 * (lam_anchor / lam_start)^1
#  Every stage records its own held-out loss every 100 steps (log.csv own_*). Then long-horizon evaluation.
cd "$(dirname "$0")"
R=$SCRATCH/multistep_roughness
L=$R/logs/followup6; mkdir -p $L
SHORT="steps_ft=4000 warmup=200 log_every=100 n_ckpt_log=6 $EXTRA_SET"
job() { local gpu=$1 name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu python -u "$@" > $L/$name.log 2>&1; local rc=$?; echo "exit=$rc $name $(date +%T)" >> $L/exits.txt; return $rc; }

branch() {   # gpu width root rule
  local g=$1 W=$2 root=$3 b=$4
  local S=(--set "hidden=[$W,$W,$W]" "out_root=$root" $SHORT)
  local A=(--measure-curv) L1=(lr_ft=1.0e-5) L2=(lr_ft=1.0e-6)
  if [ $b != fixed ]; then A=(--lr-rule $b --lr-anchor-run dt0.08_unroll032_ft_s4k); L1=(lr_ft=1.0e-4); L2=(lr_ft=1.0e-4); fi
  job $g w${W}_064_$b train.py --dt 0.08 --proto unroll064 --init ft --from-run dt0.08_unroll032_ft_s4k --suffix _s4k_$b "${A[@]}" "${S[@]}" "${L1[@]}" &&
    touch $root/runs/dt0.08_unroll064_ft_s4k_$b/train.ok &&
  job $g w${W}_128_$b train.py --dt 0.08 --proto unroll128 --init ft --from-run dt0.08_unroll064_ft_s4k_$b --suffix _s4k_$b "${A[@]}" "${S[@]}" "${L2[@]}" &&
    touch $root/runs/dt0.08_unroll128_ft_s4k_$b/train.ok
}

chain() {   # gpu width root
  local g=$1 W=$2 root=$3
  local S=(--set "hidden=[$W,$W,$W]" "out_root=$root" $SHORT lr_ft=1.0e-4)
  job $g w${W}_016 train.py --dt 0.08 --proto unroll016 --init ft --suffix _s4k --measure-curv "${S[@]}" && touch $root/runs/dt0.08_unroll016_ft_s4k/train.ok &&
  job $g w${W}_032 train.py --dt 0.08 --proto unroll032 --init ft --from-run dt0.08_unroll016_ft_s4k --suffix _s4k --measure-curv "${S[@]}" &&
    touch $root/runs/dt0.08_unroll032_ft_s4k/train.ok || return 1
  for b in fixed sqrt inv; do branch $g $W $root $b & done
  wait
  job $g w${W}_eval eval_long.py --runs $(cd $root/runs && ls -d dt0.08_onestep dt0.08_*_s4k*) --set "hidden=[$W,$W,$W]" "out_root=$root" $EXTRA_SET
}

# step 1: curvature of the existing (20k-step) width chains, one file per width
( for W in 64 128 256 512; do
    root=$R/pilot/w$W; [ $W = 256 ] && root=$R
    job 3 curv_w$W trainability.py --runs $(cd $root/runs && ls -d dt0.08_*_wchain) --out $root/trainability_wchain.pt \
        --set "hidden=[$W,$W,$W]" "out_root=$root" $EXTRA_SET
  done ) &
chain 0 512 $R/pilot/w512 &
( chain 1 256 $R; chain 1 128 $R/pilot/w128 ) &
chain 2 64 $R/pilot/w64 &
wait
echo "all done $(date +%T)" >> $L/exits.txt
