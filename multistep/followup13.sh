#!/bin/bash
# Follow-up 13: nudging combined with the line step-size rule (inside the allocation, inside shifter, 4 GPUs).
#  Width 256, dt 0.08, Euler+direct, 3 seeds, 4000 steps per stage (warmup 200), from the 0.32-window s4k models.
#  The line rule: Adam direction u, step length min(lr schedule, line_c g.u / |Psi u|_w^2), line_c = 0.25; with
#  --nudge the curvature pass runs the nudged rollout (the loss being minimized).
#  GPU 0: free line chain 0.64 -> 1.28 -> 2.56 (control at width 256; follow-up 7 ran it at widths 128/512 only)
#  GPU 1: nudged line chain A, radii 0.01 -> 0.03 -> 0.1 (radius at the free deviation of the median starts at the window's end)
#  GPU 2: nudged line chain B, radii 0.03 -> 0.1 -> 0.3
#  GPU 3: fixed-lr nudged continuation: 2.56 at r0 = 0.1, lr 1e-4, from the follow-up-12 1.28 model (r0 = 0.03)
#  then robust evaluation of every new run (two GPUs) and the curvature/rates checks on the final 2.56 models.
cd "$(dirname "$0")"
R=$SCRATCH/multistep_roughness
L=$R/logs/followup13; mkdir -p $L
SHORT="steps_ft=4000 warmup=200 log_every=100 n_ckpt_log=6 lr_ft=1.0e-4 line_c=0.25 $EXTRA_SET"
job() { local gpu=$1 name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu python -u "$@" > $L/$name.log 2>&1; local rc=$?; echo "exit=$rc $name $(date +%T)" >> $L/exits.txt; return $rc; }

chain() {   # gpu tag r0_064 r0_128 r0_256  (r0 "none": free)
  local g=$1 t=$2 prev=dt0.08_unroll032_ft_s4k
  shift 2
  for pr in unroll064 unroll128 unroll256; do
    local r0=$1; shift
    local N=(); [ "$r0" != none ] && N=(--nudge $r0)
    job $g ${t}_$pr train.py --dt 0.08 --proto $pr --init ft --from-run $prev --suffix _s4k_$t --lr-rule line --measure-curv "${N[@]}" --set $SHORT || return 1
    touch $R/runs/dt0.08_${pr}_ft_s4k_$t/train.ok; prev=dt0.08_${pr}_ft_s4k_$t
  done
}

chain 0 line025 none none none &
chain 1 nline025A 0.01 0.03 0.1 &
chain 2 nline025B 0.03 0.1 0.3 &
( job 3 nudge256_r0.1_from128 train.py --dt 0.08 --proto unroll256 --init ft --from-run dt0.08_unroll128_ft_s4k_nudge0.03_lr1e-4 \
      --suffix _s4k_nudge0.1_lr1e-4_from128 --nudge 0.1 --measure-curv --set $SHORT \
    && touch $R/runs/dt0.08_unroll256_ft_s4k_nudge0.1_lr1e-4_from128/train.ok ) &
wait
echo "training done $(date +%T)" >> $L/exits.txt
( job 0 robust_a eval_robust.py --roots 256:$R --match _s4k_line025 _s4k_nline025A _s4k_nline025B _from128 --part 0 --nparts 2 ) &
( job 1 robust_b eval_robust.py --roots 256:$R --match _s4k_line025 _s4k_nline025A _s4k_nline025B _from128 --part 1 --nparts 2 ) &
( job 2 curv_fu13 rate_checks.py --runs dt0.08_unroll256_ft_s4k_line025 dt0.08_unroll256_ft_s4k_nline025A dt0.08_unroll256_ft_s4k_nline025B \
      --parts curv --windows 1.28 2.56 --curv-r0 0.1 --out $R/rate_checks_fu13_curv.pt $EXTRA_RC ) &
( job 3 rates_fu13 rate_checks.py --runs dt0.08_unroll256_ft_s4k_line025 dt0.08_unroll256_ft_s4k_nline025A dt0.08_unroll256_ft_s4k_nline025B \
      --parts rates surrogate --out $R/rate_checks_fu13_rates.pt $EXTRA_RC ) &
wait
echo "all done $(date +%T)" >> $L/exits.txt
