#!/bin/bash
# Follow-up 10: does multistep training help through the first-order reweighting alone, or through the higher orders
# (which contain error correction)? Euler and direct (both trained together; Euler is the focus), dt 0.08, widths 128,
# 256, 512, 3 seeds, all from the converged one-step models, 4000 steps, Adam, warmup 200, lr 1e-4, windows 0.32 and 0.64:
#   _s4k_full : the window loss L_p (all orders)
#   _s4k_fo   : its first-order part (--first-order: Jacobians at the true states held fixed; gradient -Phi^T L^T W L eps)
#   onestep _s4k_cont : the one-step loss continued for the same 4000 steps (control for the extra training)
# Then robust evaluation (rms, median, 90th, 99th percentile of |e_k(T)|) of these runs and the one-step models.
# One task per node, 4 GPUs per node, node index from SLURM_NODEID, jobs spread round-robin over all GPUs.
cd "$(dirname "$0")"
R=$SCRATCH/multistep_roughness; N=${SLURM_NODEID:-0}; NN=${NNODES:-1}; NQ=$((4 * NN))
L=$R/logs/followup10; mkdir -p $L
SHORT="steps_ft=4000 warmup=200 log_every=100 n_ckpt_log=6 lr_ft=1.0e-4 $EXTRA_SET"
root_of() { [ $1 = 256 ] && echo $R || echo $R/pilot/w$1; }
job() { local gpu=$1 name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu python -u "$@" > $L/$name.log 2>&1; echo "exit=$? $name $(date +%T) node$N" >> $L/exits.txt; }
JOBS=()
for W in 512 256 128; do
  for pr in unroll064 unroll032; do JOBS+=("$W $pr _s4k_full" "$W $pr _s4k_fo --first-order"); done
  JOBS+=("$W onestep _s4k_cont")
done
for g in 0 1 2 3; do
  q=$((4 * N + g))
  ( for i in "${!JOBS[@]}"; do
      [ $((i % NQ)) -eq $q ] || continue
      set -- ${JOBS[$i]}; W=$1 pr=$2 sfx=$3; shift 3; root=$(root_of $W)
      job $g w${W}_${pr}${sfx} train.py --dt 0.08 --proto $pr --init ft --suffix $sfx "$@" --set "hidden=[$W,$W,$W]" "out_root=$root" $SHORT
    done ) &
done
wait
echo "node$N training done $(date +%T)" >> $L/exits.txt
# evaluation: one root per GPU on node 0 (only after all training; node 1 waits for the marker files)
if [ $N -eq 0 ]; then
  while [ $(grep -c "training done" $L/exits.txt) -lt $NN ]; do sleep 10; done
  g=0; for W in 512 256 128; do root=$(root_of $W)
    job $g robust_w$W eval_robust.py --roots $W:$root --match _s4k_full _s4k_fo _s4k_cont dt0.08_onestep --part 0 --nparts 1 & g=$((g + 1)); done
  wait
fi
echo "node$N done $(date +%T)" >> $L/exits.txt
