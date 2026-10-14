#!/bin/bash
# Critical batch size on the field (2026-10-13; plan and predictions in crit_plan.py's caller message / XFOLD_RESULT.md successor).
# Slots are written by crit_plan.py (LPT assignment); node NODE runs slots 4*NODE .. 4*NODE+3, one GPU each, jobs one at a time.
# usage: scale_crit.sh NODE [DRY]
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_crit
NODE=${1:-0}; DRY=${2:-0}
for g in 0 1 2 3; do
  s=$((4 * NODE + g))
  ( while read -r idx args; do
      if [ "$DRY" = 1 ]; then echo "node $NODE gpu $g job $idx: ${args:0:90}"; continue; fi
      [ -f "$F/res_j$idx.jsonl" ] && continue
      CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 python -u "$HERE/scale_study.py" $args --out "$F/res_j$idx.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$idx.txt"
    done < "$F/slot_$s.txt" ) &
done
wait; echo "NODE $NODE DONE $(date +%T)"
