#!/bin/bash
# Width test and theory checks (inside the allocation, inside shifter, 4 GPUs).
#  Width test, dt = 0.08: for widths 64, 128, 256, 512 (3 hidden layers), a window chain fine-tuned from that
#  width's one-step model: 0.16 (lr 1e-4) -> 0.32 (lr 1e-4) -> 0.64 (lr 1e-5) -> 1.28 (lr 1e-6), 20k steps each,
#  then long-horizon evaluation of every stage. Runs are tagged _wchain; widths 64/128/512 live in pilot/w<width>.
#  GPU 3: checks (v) and (i) of the rollout-kernel draft on the width-256 sweep models.
cd "$(dirname "$0")"
R=$SCRATCH/multistep_roughness
L=$R/logs/followup5; mkdir -p $L
job() { local gpu=$1 name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu python -u "$@" > $L/$name.log 2>&1; local rc=$?; echo "exit=$rc $name $(date +%T)" >> $L/exits.txt; return $rc; }

chain() {   # gpu width root
  local g=$1 W=$2 root=$3
  local S=(--suffix _wchain --set "hidden=[$W,$W,$W]" "out_root=$root" $EXTRA_SET)   # EXTRA_SET: tests only
  job $g w${W}_c016 train.py --dt 0.08 --proto unroll016 --init ft "${S[@]}" lr_ft=1.0e-4 && touch $root/runs/dt0.08_unroll016_ft_wchain/train.ok &&
  job $g w${W}_c032 train.py --dt 0.08 --proto unroll032 --init ft --from-run dt0.08_unroll016_ft_wchain "${S[@]}" lr_ft=1.0e-4 && touch $root/runs/dt0.08_unroll032_ft_wchain/train.ok &&
  job $g w${W}_c064 train.py --dt 0.08 --proto unroll064 --init ft --from-run dt0.08_unroll032_ft_wchain "${S[@]}" lr_ft=1.0e-5 && touch $root/runs/dt0.08_unroll064_ft_wchain/train.ok &&
  job $g w${W}_c128 train.py --dt 0.08 --proto unroll128 --init ft --from-run dt0.08_unroll064_ft_wchain "${S[@]}" lr_ft=1.0e-6 && touch $root/runs/dt0.08_unroll128_ft_wchain/train.ok
  job $g w${W}_eval eval_long.py --runs dt0.08_onestep dt0.08_unroll016_ft_wchain dt0.08_unroll032_ft_wchain dt0.08_unroll064_ft_wchain dt0.08_unroll128_ft_wchain \
      --set "hidden=[$W,$W,$W]" "out_root=$root" $EXTRA_SET
}

chain 0 512 $R/pilot/w512 &
( chain 1 256 $R; chain 1 128 $R/pilot/w128 ) &
( job 2 w64_onestep train.py --dt 0.08 --proto onestep --set "hidden=[64,64,64]" "out_root=$R/pilot/w64" \
    && touch $R/pilot/w64/runs/dt0.08_onestep/train.ok && chain 2 64 $R/pilot/w64 ) &
( job 3 check_v theory_checks.py --part v --runs dt0.08_onestep dt0.04_onestep dt0.08_unroll032_ft dt0.04_unroll032_ft
  job 3 check_i theory_checks.py --part i --seeds 0 1 2 --runs dt0.01_onestep dt0.02_onestep dt0.04_onestep dt0.08_onestep \
      dt0.01_unroll032_ft dt0.02_unroll032_ft dt0.04_unroll032_ft dt0.08_unroll032_ft \
      dt0.01_unroll064_ft dt0.02_unroll064_ft dt0.04_unroll064_ft dt0.08_unroll064_ft ) &
wait
echo "all done $(date +%T)" >> $L/exits.txt
