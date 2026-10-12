#!/bin/bash
# blind held-out batch with the propagation-corrected model: own init (REAL=1 OWN=1) and one other init (REAL=1 OWN=0) per design,
# 56 runs spread over the 4 nodes x 4 GPUs of job 59653997. T = 3000/scale^2.
cd $SCRATCH/l63_held_rough/runs; mkdir -p ensP
NODES=(nid001308 nid001309 nid001312 nid001313); i=0
for f in heldout/des_*.npy; do
  tag=$(basename $f .npy); tag=${tag#des_}
  H=$(echo $tag | sed -E 's/^H([0-9]+)_.*/\1/'); D=$(echo $tag | sed -E 's/.*_D([0-9.]+)_.*/\1/'); d=$(echo $tag | sed -E 's/.*_depth([0-9]+)_.*/\1/'); s=$(echo $tag | sed -E 's/.*_s([0-9]+).*/\1/')
  sc=1.0; T=3000; case $tag in *_os0.5) sc=0.5; T=12000;; esac
  for mode in own other; do
    OWN=1; [ $mode = other ] && OWN=0
    node=${NODES[$((i % 4))]}; gpu=$(( (i / 4) % 4 ))
    EPS=0.03 ADD=1 REAL=1 OWN=$OWN PROP=1 CUDA_VISIBLE_DEVICES=$gpu srun --jobid=59653997 --overlap -w $node -N 1 --ntasks=1 --gpus=4 --gpu-bind=none shifter --image=nersc/pytorch:26.01.01 python -u $(dirname "$(readlink -f "$0")")/ensemble_model.py $H $d $s $D $sc $T 0 $H ensP/${mode}_$tag.npz > ensP/log_${mode}_$tag.txt 2>&1 &
    echo "launched $mode $tag H=$H D=$D depth=$d seed=$s sc=$sc T=$T on $node gpu$gpu"; i=$((i+1))
  done
done
wait; echo "blind prop batch finished $(date)"
