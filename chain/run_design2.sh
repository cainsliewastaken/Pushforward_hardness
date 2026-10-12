#!/bin/bash
# usage: run_design2.sh OUTDIR NGPU cfg...   cfg = H:Delta:depth:seed:n:scale:T:OPT:LOSS:LRMULT:BATCH:WD:LAYERS[:TAUS]
OUT=$1; NG=$2; shift 2; CODE=$(dirname "$(readlink -f "$0")"); mkdir -p "$OUT"; export OMP_NUM_THREADS=2
echo "host=$(hostname) start=$(date) configs: $*"; i=0
for cfg in "$@"; do
  IFS=: read H D d s nn sc T OPT LOSS LR B WD LAY TAUS <<< "$cfg"
  tag=${L63_DATA:+${L63_DATA}_}H${H}_D${D}_depth${d}_s${s}; [ "${L63_DATA:-l63}" = l63 ] && tag=H${H}_D${D}_depth${d}_s${s}; [ "$nn" != 200 ] && tag=${tag}_n${nn}; [ "$sc" != 1 ] && tag=${tag}_os${sc}; [ "$OPT" != gd ] && tag=${tag}_${OPT}; [ "$LOSS" != sq ] && tag=${tag}_${LOSS}; [ "$LR" != 1 ] && tag=${tag}_lr${LR}; [ "$B" != 0 ] && tag=${tag}_b${B}; [ "$WD" != 0 ] && tag=${tag}_wd${WD}
  CUDA_VISIBLE_DEVICES=$((i % NG)) L63_OUTSCALE=$sc L63_OPT=$OPT L63_LOSS=$LOSS L63_LRMULT=$LR L63_BATCH=$B L63_WD=$WD L63_LAYERS=$LAY L63_DATA=${L63_DATA:-l63} L63_TAUS=${TAUS:-1,3,10,30,100,300,1000,3000,10000,30000,100000} python -u "$CODE/l63_design2.py" "$nn" "$D" "$T" "$d" "$H" "$OUT" "$s" 250 > "$OUT/log_$tag.txt" 2>&1 &
  echo "launched $tag on gpu $((i % NG)) pid=$!"; i=$((i+1))
done
wait; echo "all runs finished $(date)"
