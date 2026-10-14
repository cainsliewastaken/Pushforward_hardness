#!/bin/bash
# Item 53 (2026-10-13): cross-fitting in the optimizer. Folds = two contiguous halves of the training set; batches alternate between them;
# a momentum per fold; diagnostic every 10 steps: per lambda decile, mean g . sign(C_same fold), g . sign(C_other fold), and a held-out
# gradient's agreement with both (the population value the learned weight's hypergradient should estimate). Rule (--xfold_rule 1): the
# hypergradient uses the push rebuilt from the other fold's momentum.
# Predictions: turbulence, slow deciles, from epoch 2: same > cross ~ val; field (fresh data): same ~ cross ~ val; rule leaves the field within
# seed noise (0.0622 / 1.42e-4) and narrows the turbulence train / validation gap (wd 0 reference 0.195 / 0.100, items 50-51).
# Null: same ~ cross on turbulence = no frozen-sample bias in the hypergradient; memorisation enters through the sign.   usage: NODE NNODES
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_xfold; mkdir -p $F
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --lr 1e-3 --xfold 1 --xfold_every 10"
T="--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_400k.npz --turb_cpu 1 --turb_onepass 1 --width 4096 --depth 2 --eval_every 250 --tf32 1 --nval 4096 --seed 1 --batch 256 --steps 31240 --rem 0.15 --slow_adapt 1 --sa_cap 1 --sa_kappa_max 1.0 --rem_anchor min"
Fd="--width 4096 --depth 2 --batch 256 --eval_every 250 --tf32 1 --steps 5000 --rem 0.2 --seed 1"
JOBS=()
for tg in delta direct; do for r in 0 1; do JOBS+=("$T $K --turb_target $tg --xfold_rule $r"); done; done
for tg in delta direct; do JOBS+=("$T $K --turb_target $tg --xfold_rule 1 --u_decay 0.0003"); done
for p in 2 0; do for r in 0 1; do JOBS+=("$Fd $K --p $p --xfold_rule $r"); done; done
for p in 2 0; do JOBS+=("$Fd $K --p $p --xfold_rule 1 --u_decay 0.0003"); done
echo "${#JOBS[@]} jobs" > $F/njobs
NODE=${1:-0}; NN=${2:-1}; i=0; k=0
for job in "${JOBS[@]}"; do
  if [ $((i % NN)) -eq $NODE ]; then
    CUDA_VISIBLE_DEVICES=$((k % 4)) OMP_NUM_THREADS=8 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
    k=$((k+1)); if [ $((k % 4)) -eq 0 ]; then wait; fi
  fi
  i=$((i+1))
done
wait; echo "NODE $NODE DONE $(date +%T)"
