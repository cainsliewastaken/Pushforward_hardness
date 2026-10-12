#!/bin/bash
# Launch l63_late.py for a list of configs "H:Delta:depth:seed" round-robin over NGPU GPUs on this node; wait.
# usage: run_sweep_gpu.sh OUTDIR T_TOTAL NGPU NLOG NSNAP cfg1 cfg2 ...   cfg = H:Delta:depth:seed[:n]
OUT=$1; T=$2; NG=$3; NLOG=$4; NSNAP=$5; shift 5
CODE=$(dirname "$(readlink -f "$0")")
mkdir -p "$OUT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "host=$(hostname) start=$(date) T=$T ngpu=$NG configs: $*"
i=0
for cfg in "$@"; do
  IFS=: read H D d s nn <<< "$cfg"
  nn=${nn:-200}
  tag=H${H}_D${D}_depth${d}_s${s}
  [ "$nn" != 200 ] && tag=${tag}_n${nn}
  [ "${L63_DATASEED:-1}" != 1 ] && tag=${tag}_ds${L63_DATASEED}
  [ "${L63_OUTSCALE:-1}" != 1 ] && tag=${tag}_os${L63_OUTSCALE}
  CUDA_VISIBLE_DEVICES=$((i % NG)) python -u "$CODE/l63_late.py" "$nn" "$D" "$T" "$d" "$H" "$OUT" "$s" "$NLOG" "$NSNAP" > "$OUT/log_$tag.txt" 2>&1 &
  echo "launched $tag on gpu $((i % NG)) pid=$!"
  i=$((i+1))
done
wait
echo "all runs finished $(date)"
