#!/bin/bash
# usage: run_design.sh OUTDIR NGPU cfg...   cfg = H:Delta:depth:seed:n:scale:T
OUT=$1; NG=$2; shift 2
CODE=$(dirname "$(readlink -f "$0")"); mkdir -p "$OUT"; export OMP_NUM_THREADS=2
echo "host=$(hostname) start=$(date) configs: $*"
i=0
for cfg in "$@"; do
  IFS=: read H D d s nn sc T <<< "$cfg"
  tag=H${H}_D${D}_depth${d}_s${s}; [ "$nn" != 200 ] && tag=${tag}_n${nn}; [ "$sc" != 1 ] && tag=${tag}_os${sc}
  CUDA_VISIBLE_DEVICES=$((i % NG)) L63_OUTSCALE=$sc python -u "$CODE/l63_design.py" "$nn" "$D" "$T" "$d" "$H" "$OUT" "$s" 250 > "$OUT/log_$tag.txt" 2>&1 &
  echo "launched $tag on gpu $((i % NG)) pid=$!"; i=$((i+1))
done
wait; echo "all runs finished $(date)"
