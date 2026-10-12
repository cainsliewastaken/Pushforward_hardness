#!/bin/bash
# Launch one l63test_torch.py process per (H, Delta, depth), round-robin over the visible GPUs, wait for all.
# usage: run_matrix_gpu.sh OUTDIR T_TOTAL LOG_EVERY NGPU "H list" "Delta list" "depth list"
OUT=$1; T=$2; LE=$3; NG=$4; HS=$5; DS=$6; DEPTHS=$7
CODE=$(dirname "$(readlink -f "$0")")
mkdir -p "$OUT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "host=$(hostname) start=$(date) T=$T log_every=$LE ngpu=$NG H=[$HS] Delta=[$DS] depths=[$DEPTHS]"
nvidia-smi -L
i=0
for H in $HS; do for D in $DS; do for d in $DEPTHS; do
  tag=H${H}_D${D}_depth${d}
  CUDA_VISIBLE_DEVICES=$((i % NG)) python -u "$CODE/l63test_torch.py" 200 "$D" "$T" "$d" "$H" "$LE" "$OUT" > "$OUT/log_$tag.txt" 2>&1 &
  echo "launched $tag on gpu $((i % NG)) pid=$!"
  i=$((i+1))
done; done; done
wait
echo "all runs finished $(date)"
