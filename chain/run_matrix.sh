#!/bin/bash
# Launch one l63test.py process per (H, Delta, depth) in parallel, wait for all.
# usage: run_matrix.sh OUTDIR T_TOTAL LOG_EVERY THREADS "H list" "Delta list" "depth list"
OUT=$1; T=$2; LE=$3; TH=$4; HS=$5; DS=$6; DEPTHS=$7
CODE=$(dirname "$(readlink -f "$0")")
mkdir -p "$OUT"
export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH
echo "host=$(hostname) start=$(date) T=$T log_every=$LE threads=$TH H=[$HS] Delta=[$DS] depths=[$DEPTHS]"
for H in $HS; do for D in $DS; do for d in $DEPTHS; do
  tag=H${H}_D${D}_depth${d}
  python -u "$CODE/l63test.py" 200 "$D" "$T" "$d" "$H" "$LE" "$OUT" > "$OUT/log_$tag.txt" 2>&1 &
  echo "launched $tag pid=$!"
done; done; done
wait
echo "all runs finished $(date)"
