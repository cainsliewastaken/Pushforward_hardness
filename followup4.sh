#!/bin/bash
# Trainability check (inside the allocation, inside shifter, 4 GPUs): every unrolled run, split round-robin.
cd "$(dirname "$0")"
R=$SCRATCH/multistep_roughness
L=$R/logs/followup4; mkdir -p $L
runs=($(cd $R/runs && ls -d dt*/train.ok | xargs -n1 dirname | grep -v _onestep | sort))
echo "${#runs[@]} runs" >> $L/exits.txt
for g in 0 1 2 3; do
  mine=(); for i in "${!runs[@]}"; do [ $((i % 4)) -eq $g ] && mine+=("${runs[$i]}"); done
  ( CUDA_VISIBLE_DEVICES=$g python -u trainability.py --runs "${mine[@]}" --out $R/trainability_g$g.pt > $L/g$g.log 2>&1; echo "exit=$? g$g $(date +%T)" >> $L/exits.txt ) &
done
wait
python - <<PY >> $L/exits.txt 2>&1
import torch
res = {}
for g in range(4):
    res.update(torch.load("$R/trainability_g%d.pt" % g, weights_only=False))
torch.save(res, "$R/trainability.pt"); print("merged", len(res), "models")
PY
echo "all done $(date +%T)" >> $L/exits.txt
