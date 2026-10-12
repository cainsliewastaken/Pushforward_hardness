#!/bin/bash
# Follow-up 12: checks of multistep_best_window.tex and nudged training (inside the allocation, inside shifter, 4 GPUs).
#  GPU 0: rate_checks.py (rates, surrogate, screen; Sections 10, 9, 7) on the one-step, 0.32-window and curvature-scaled
#         1.28-window models (Euler, 3 seeds), then on the nudged runs trained below.
#  GPU 1: rate_checks.py curvature vs window, free and nudged (Section 8); then follow-up 11 (check (v) at windows up to
#         2.56 and four step lengths: the unremovable fraction at two step lengths), one-step Euler model, seed 0.
#  GPU 2, 3: nudged training at a fixed learning rate (Sections 8, 9): from the 0.64-window s4k fixed models, 4000 steps,
#         window 1.28 at lr 1e-4 (free control + radii 0.01 .. 1), window 1.28 at lr 1e-6 (r0 0.1, against the fixed-lr
#         run), window 2.56 at lr 1e-4 (free control + r0 0.03, 0.3); then robust evaluation (rms, median, p90, p99).
cd "$(dirname "$0")"
R=$SCRATCH/multistep_roughness
L=$R/logs/followup12; mkdir -p $L
SHORT="steps_ft=4000 warmup=200 log_every=100 n_ckpt_log=6 $EXTRA_SET"
FROM=dt0.08_unroll064_ft_s4k_fixed
job() { local gpu=$1 name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu python -u "$@" > $L/$name.log 2>&1; local rc=$?; echo "exit=$rc $name $(date +%T)" >> $L/exits.txt; return $rc; }
trn() {   # gpu name proto suffix lr [extra args]
  local g=$1 name=$2 pr=$3 sfx=$4 lr=$5; shift 5
  job $g $name train.py --dt 0.08 --proto $pr --init ft --from-run $FROM --suffix $sfx --measure-curv "$@" --set $SHORT lr_ft=$lr \
    && touch $R/runs/dt0.08_${pr}_ft$sfx/train.ok
}
BASE="dt0.08_onestep dt0.08_unroll032_ft_s4k dt0.08_unroll128_ft_s4k_inv"

( job 0 rates_base rate_checks.py --runs $BASE --parts rates surrogate screen --out $R/rate_checks_base.pt $EXTRA_RC ) &
( job 1 curv_base rate_checks.py --runs $BASE --parts curv --seeds 0 1 2 --out $R/rate_checks_curv_base.pt $EXTRA_RC
  job 1 check_v_fu11 theory_checks.py --part v --runs dt0.08_onestep --classes euler --seeds 0 --windows 0.32 0.64 1.28 1.92 2.56 --lrs 1e-6 1e-5 1e-4 1e-3 $EXTRA_TC ) &
( trn 2 free128_lr1e-4 unroll128 _s4k_free_lr1e-4 1.0e-4
  trn 2 nudge128_r0.01 unroll128 _s4k_nudge0.01_lr1e-4 1.0e-4 --nudge 0.01
  trn 2 nudge128_r0.03 unroll128 _s4k_nudge0.03_lr1e-4 1.0e-4 --nudge 0.03
  trn 2 nudge128_r0.1 unroll128 _s4k_nudge0.1_lr1e-4 1.0e-4 --nudge 0.1
  trn 2 free256_lr1e-4 unroll256 _s4k_free_lr1e-4 1.0e-4 ) &
( trn 3 nudge128_r0.3 unroll128 _s4k_nudge0.3_lr1e-4 1.0e-4 --nudge 0.3
  trn 3 nudge128_r1 unroll128 _s4k_nudge1_lr1e-4 1.0e-4 --nudge 1.0
  trn 3 nudge128_r0.1_lr1e-6 unroll128 _s4k_nudge0.1_lr1e-6 1.0e-6 --nudge 0.1
  trn 3 nudge256_r0.03 unroll256 _s4k_nudge0.03_lr1e-4 1.0e-4 --nudge 0.03
  trn 3 nudge256_r0.3 unroll256 _s4k_nudge0.3_lr1e-4 1.0e-4 --nudge 0.3 ) &
wait
echo "training and base checks done $(date +%T)" >> $L/exits.txt
NEW=$(cd $R/runs && ls -d dt0.08_unroll128_ft_s4k_free_lr1e-4 dt0.08_unroll128_ft_s4k_nudge* dt0.08_unroll256_ft_s4k_* 2>/dev/null | tr "\n" " ")
( job 2 robust_new eval_robust.py --roots 256:$R --match _s4k_free_lr1e-4 _s4k_nudge --part 0 --nparts 2 ) &
( job 3 robust_new2 eval_robust.py --roots 256:$R --match _s4k_free_lr1e-4 _s4k_nudge --part 1 --nparts 2 ) &
( job 0 rates_new rate_checks.py --runs $NEW --parts rates surrogate --out $R/rate_checks_new.pt $EXTRA_RC ) &
( job 1 curv_new rate_checks.py --runs $NEW --parts curv --windows 1.28 2.56 --curv-r0 0.1 --out $R/rate_checks_curv_new.pt $EXTRA_RC ) &
wait
echo "all done $(date +%T)" >> $L/exits.txt
