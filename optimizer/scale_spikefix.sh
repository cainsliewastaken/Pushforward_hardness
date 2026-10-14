#!/bin/bash
# Spike fix, merged job (2026-10-14; xfold session + the other session's item 58). Cause under test: the learned-weight hypergradient ignores
# the output budget. When the budget binds it keeps a fraction of the proposed step and rescales every mode, but the hypergradient credits each
# mode with its full proposed push and omits the cross term (raising u_k shrinks all other modes), so the weights inflate together to the clamp
# (u 4.59 of 4.6 at every spike onset, proposed step 6-40x the budget). Fix: --hyper_budget 1, the derivative through the budget (zero-sum
# over modes). Compared, all instrumented (--spike_diag: umax, umean, fraction u > 4, kept fraction, probe loss, refresh change):
#   C3 rough B1024 T10000 (the other session's config): reference / fix / uniform decay 3e-4 / 1e-3 / gated decay 1e-3 (its item 58);
#   C1 smooth B1024 T10000 lr 2e-3: reference / fix / fix seed 2;  C2 smooth B512 T20000 lr 2e-3: reference / fix;
#   5000-step batch-256 presets with the fix (rough, smooth; reference 0.0622 / 1.42e-4).
# Predictions: with the fix umean stays flat, no u reaches 4.6, kept stays near the no-fix early value, no spikes on C1 / C2 / C3, presets
# within 5 %. Null: spikes persist with the fix (then the weights are not the cause), or the weights still inflate (then the derivative is
# not what drives them). Decay runs: the other session's predictions (fewer spikes if weight-norm driven).   Frozen code in $F/code.
F=$SCRATCH/l63_held_rough/runs/scale_spikefix
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0 --rem_check 1"
C3="--width 4096 --depth 2 --batch 1024 --eval_every 125 --tf32 1 --steps 10000 --seed 1 --p 2 --nval 4096 $K --lr 1e-3 --rem 0.3 --spike_diag 2000"
C1="--width 4096 --depth 2 --batch 1024 --eval_every 250 --tf32 1 --steps 10000 --p 0 --nval 4096 $K --lr 2e-3 --rem 0.3 --spike_diag 2000"
C2="--width 4096 --depth 2 --batch 512 --eval_every 250 --tf32 1 --steps 20000 --seed 1 --p 0 --nval 4096 $K --lr 2e-3 --rem 0.3 --spike_diag 2000"
PR="--width 4096 --depth 2 --batch 256 --eval_every 250 --tf32 1 --steps 5000 --seed 1 --nval 4096 $K --lr 1e-3 --spike_diag 1000 --hyper_budget 1"
SLOT0=("0 $C2" "1 $C3" "2 $PR --p 0 --rem 0.3")
SLOT1=("3 $C2 --hyper_budget 1" "4 $C3 --hyper_budget 1" "5 $PR --p 2 --rem 0.2")
SLOT2=("6 $C3 --wd_abs 3e-4" "7 $C3 --wd_abs 1e-3" "8 $C1 --seed 1")
SLOT3=("9 $C3 --wd_abs 1e-3 --wd_gate 1 --wd_gate_q0 0.5" "10 $C1 --seed 1 --hyper_budget 1" "11 $C1 --seed 2 --hyper_budget 1")
echo "12 jobs" > $F/njobs
run_slot() { local g=$1; shift; for job in "$@"; do idx=${job%% *}; args=${job#* }
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 python -u "$F/code/optimizer/scale_study.py" $args --out "$F/res_j$idx.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$idx.txt"; done; }
run_slot 0 "${SLOT0[@]}" & run_slot 1 "${SLOT1[@]}" & run_slot 2 "${SLOT2[@]}" & run_slot 3 "${SLOT3[@]}" &
wait; echo "DONE $(date +%T)"
