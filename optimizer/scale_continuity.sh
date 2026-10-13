#!/bin/bash
# Continuity check (2026-10-12): (A) the other instance's field-problem runs re-run with THEIR script and data at the saved
# tuned settings (runs/kadam/field20k selections: expected p0 adam 0.357 / ks_sign 0.178 / ks_adam 0.161; p2 adam 11.48 /
# ks_sign 5.25 / ks_adam 5.23); (B) today's harness at the same width 512, batch 100, 6000 steps on the online field
# problem, AdamW / SOAP full / KronQ full (hypergradient) / KronQ full + budget 0.2, both targets. Jobs -> GPU i % 4.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; F=$SCRATCH/l63_held_rough/runs/scale_continuity; mkdir -p $F
D=$SCRATCH/l63_held_rough/runs/kadam
OLD="--widths 512,512 --act gelu --batch 100 --standardize none --seed 1 --steps 6000"
JOBS=(
  "OLD --data $D/field20k_p0.npz --opt adam --lr 2.5e-3 --b2 0.99 --warmup 0 $OLD"
  "OLD --data $D/field20k_p0.npz --opt kronsign --rule sign --lr 5.7e-4 --b2 0.999 --warmup 0 $OLD"
  "OLD --data $D/field20k_p0.npz --opt kronsign --rule adam --lr 3.8e-3 --b2 0.99 --warmup 0 $OLD"
  "OLD --data $D/field20k_p2.npz --opt adam --lr 1.7e-3 --b2 0.99 --warmup 100 $OLD"
  "OLD --data $D/field20k_p2.npz --opt kronsign --rule sign --lr 5.7e-4 --b2 0.999 --warmup 0 $OLD"
  "OLD --data $D/field20k_p2.npz --opt kronsign --rule adam --lr 1.1e-3 --b2 0.999 --warmup 0 $OLD"
)
NEW="--width 512 --depth 2 --steps 6000 --batch 100 --refresh 10 --eval_every 500"
for p in 0 2; do
  for lr in 1e-3 2.5e-3 5e-3; do JOBS+=("NEW --opt adamw --lr $lr --p $p $NEW"); done
  for lr in 1e-3 3e-3; do JOBS+=("NEW --opt soap --lr $lr --p $p $NEW"); done
  for lr in 5e-4 1e-3 3e-3; do JOBS+=("NEW --opt kronq --lr $lr --p $p $NEW"); done
  for lr in 1e-3 3e-3; do JOBS+=("NEW --opt kronq --lr $lr --rem 0.2 --p $p $NEW"); done
done
i=0
for job in "${JOBS[@]}"; do
  kind=${job%% *}; args=${job#* }
  if [ "$kind" = OLD ]; then
    CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=2 python -u "$HERE/train_ks.py" $args --out "$F/old_j$i.jsonl" 2>&1 | grep -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  else
    CUDA_VISIBLE_DEVICES=$((i % 4)) OMP_NUM_THREADS=2 python -u "$HERE/scale_study.py" $args --out "$F/new_j$i.jsonl" 2>&1 | grep -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
  fi
  i=$((i+1))
  if [ $((i % 12)) -eq 0 ]; then wait; fi      # at most 12 concurrent (3 per GPU)
done
wait; echo "ALLDONE $(date +%T)"
