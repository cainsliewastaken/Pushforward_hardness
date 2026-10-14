#!/bin/bash
# Spike defect, numerics test (2026-10-14). Measured at spike onsets: each basis refresh changes the network output by 3-6 % of the residual
# (it should change nothing), refreshes are not covered by the budget, and the bases are rotated and the weights transported with TF32
# products (~1e-3 relative) thousands of times. Hypothesis: accumulated precision error in the refresh is the cause (fits: low loss, many
# steps, budget slack, worst on the smooth target). Test: --tf32 0 (full fp32) vs --tf32 1, several seeds each (spikes vary between
# identical runs). Instrument --spike_diag: refresh_df and the accumulated-basis orthogonality Uacc (float64) at every refresh, probe loss.
# Configs that spiked: C1 smooth B1024 T10000 lr 2e-3 (4 seeds each), C2 smooth B512 T20000 lr 2e-3 (2 each), C3 rough B1024 T10000 (2 each).
# Predictions: fp32 lowers refresh_df ~100x and Uacc to ~1e-6, and the spikes stop (0 across fp32 seeds vs >= half of TF32 seeds spiking).
# Null: fp32 runs spike at the same rate -> precision is not the cause.  16 runs, one per GPU, 4 nodes.  usage: scale_tf32.sh NODE
F=$SCRATCH/l63_held_rough/runs/scale_tf32
K="--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0"
C1="--width 4096 --depth 2 --batch 1024 --eval_every 250 --steps 10000 --p 0 --nval 4096 $K --lr 2e-3 --rem 0.3 --spike_diag 2000"
C2="--width 4096 --depth 2 --batch 512 --eval_every 250 --steps 20000 --p 0 --nval 4096 $K --lr 2e-3 --rem 0.3 --spike_diag 2000"
C3="--width 4096 --depth 2 --batch 1024 --eval_every 125 --steps 10000 --p 2 --nval 4096 $K --lr 1e-3 --rem 0.3 --spike_diag 2000"
JOBS=()
for s in 1 2; do for t in 1 0; do JOBS+=("$C2 --seed $s --tf32 $t"); done; done          # longest first: one per GPU
for s in 1 2 3 4; do for t in 1 0; do JOBS+=("$C1 --seed $s --tf32 $t"); done; done
for s in 1 2; do for t in 1 0; do JOBS+=("$C3 --seed $s --tf32 $t"); done; done
echo "${#JOBS[@]} jobs" > $F/njobs
NODE=${1:-0}
for g in 0 1 2 3; do
  i=$((4 * NODE + g)); job="${JOBS[$i]}"
  [ -z "$job" ] && continue
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 python -u "$F/code/optimizer/scale_study.py" $job --out "$F/res_j$i.jsonl" 2>&1 | grep --line-buffered -v -i 'pynvml\|FutureWarning' > "$F/log_j$i.txt" &
done
wait; echo "NODE $NODE DONE $(date +%T)"
