#!/bin/bash
# Basis-cost tests T1 (A-only basis: --S_identity, probe at refresh steps only) and T2 (coordinates only: --S_identity
# --A_identity), both targets, one run per GPU for isolated wall; then 600-step timing runs of the reference and T1 on
# rough (one per GPU) for per-step wall next to accuracy. Reference: rough 0.061 / smooth 1.41e-4 at ~2.1 F0 per step.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_basis1; mkdir -p $F
K="--opt kronqf --width 4096 --depth 2 --batch 256 --steps 5000 --eval_every 250 --diag 1 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --lr 1e-3 --rem 0.2 --cap 1 --hyper 1"
JOBS=("$K --p 2 --S_identity 1" "$K --p 0 --S_identity 1" "$K --p 2 --S_identity 1 --A_identity 1" "$K --p 0 --S_identity 1 --A_identity 1")
i=0
for job in "${JOBS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  i=$((i+1))
done
wait
T="--opt kronqf --width 4096 --depth 2 --batch 256 --steps 600 --eval_every 300 --tf32 1 --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --lr 1e-3 --rem 0.2 --cap 1 --hyper 1 --p 2"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $T --out "$F/time_ref.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/time_ref.txt" &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $T --S_identity 1 --out "$F/time_t1.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/time_t1.txt" &
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" $T --S_identity 1 --A_identity 1 --out "$F/time_t2.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/time_t2.txt" &
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 python -u "$HERE/scale_study.py" --opt adamw --width 4096 --depth 2 --batch 256 --steps 600 --eval_every 300 --tf32 1 --lr 3e-4 --p 2 --out "$F/time_adamw.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/time_adamw.txt" &
wait; echo "ALLDONE $(date +%T)"
