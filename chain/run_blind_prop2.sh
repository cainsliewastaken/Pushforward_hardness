#!/bin/bash
# blind batch (own init, then other init chained in the same GPU slot) + scaling runs; device chosen INSIDE the srun step
cd $SCRATCH/l63_held_rough/runs; mkdir -p ensP ensS
NODES=(nid001308 nid001309 nid001312 nid001313); i=0
step() { node=$1; gpu=$2; shift 2; srun --jobid=59653997 --overlap -w $node -N 1 --ntasks=1 --gpus=4 --gpu-bind=none shifter --image=nersc/pytorch:26.01.01 bash -c "$*" & }
for f in heldout/des_*.npy; do
  tag=$(basename $f .npy); tag=${tag#des_}
  H=$(echo $tag | sed -E 's/^H([0-9]+)_.*/\1/'); D=$(echo $tag | sed -E 's/.*_D([0-9.]+)_.*/\1/'); d=$(echo $tag | sed -E 's/.*_depth([0-9]+)_.*/\1/'); s=$(echo $tag | sed -E 's/.*_s([0-9]+).*/\1/')
  sc=1.0; T=3000; case $tag in *_os0.5) sc=0.5; T=12000;; esac
  node=${NODES[$(( (i % 16) / 4 ))]}; gpu=$(( i % 4 ))
  M="$(dirname "$(readlink -f "$0")")/ensemble_model.py $H $d $s $D $sc $T 0 $H"
  step $node $gpu "export CUDA_VISIBLE_DEVICES=$gpu EPS=0.03 ADD=1 REAL=1 PROP=1; OWN=1 python -u $M ensP/own_$tag.npz > ensP/log_own_$tag.txt 2>&1; OWN=0 python -u $M ensP/other_$tag.npz > ensP/log_other_$tag.txt 2>&1"
  echo "slot $i: $tag on $node gpu$gpu (own then other)"; i=$((i+1))
done
for cfg in "1024 2 0.5" "1024 2 1.0" "512 3 0.5" "512 3 1.0" "96 6 0.5" "96 6 1.0" "96 8 1.0"; do set -- $cfg
  tag=H$1_D$3_depth$2_s0; node=${NODES[$(( (i % 16) / 4 ))]}; gpu=$(( i % 4 ))
  step $node $gpu "export CUDA_VISIBLE_DEVICES=$gpu EPS=0.03 ADD=1 REAL=1 OWN=1 PROP=1; python -u $(dirname "$(readlink -f "$0")")/ensemble_model.py $1 $2 0 $3 1.0 3000 0 $1 ensS/own_$tag.npz > ensS/log_own_$tag.txt 2>&1"
  echo "slot $i: scale $tag on $node gpu$gpu"; i=$((i+1))
done
wait; echo "blind+scale batch finished $(date)"
