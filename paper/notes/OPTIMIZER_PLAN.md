# KronQ: where it stands and the plan (2026-10-14)

Written at the end of the xfold session's spike / precision / refactor work. Results files: SPIKE_RESULT.md, XFOLD_RESULT.md,
REPORT.md items 50-59 (59 is the merge of this branch into main). Everything here is seed 1-2 unless stated.

## 1. The method (unchanged core)

Per layer, Kronecker eigenbasis of the batch-averaged factors A = avg(a a^T) and S = avg(delta delta^T) in blocks of 256; in that basis:
momentum (0.9) of the gradient, push = learned per-mode weight x sign of the momentum, learned weight by the one-step hypergradient
(memory 100); lr with linear decay; on fresh data an output budget (step scaled so the batch output change <= rem x residual); on
finite data decoupled weight decay. The step size per mode is learned from the measured response, not from the Kronecker eigenvalues.

## 2. The numerical fixes ("the hybrid"; all needed for long runs, none changes the method)

| fix | flag | why |
| QR guard | --qr_guard 1 | Cholesky-QR returned non-orthonormal rotations late in training (thousands of blocks from step ~6700); the basis broke and the loss spiked 250-1500 steps later in every smooth run, in TF32 and fp32 alike. Cost: none measurable in TF32. |
| refresh in fp32 | --refresh_hp 1 | the refresh re-expresses the weights in the new basis; in TF32 the rounding compounds over thousands of refreshes. Cost +10-19 %. |
| weights transformed at re-orthonormalisation | --reorth_transport 1 | the periodic cleanup of the basis changed the function silently. |
| activation rotations in fp32 (optional) | --rot_hp 1 | the rough field reaches the fp32 result (0.052-0.054 vs fp32 0.049) at no cost at batch 256; no effect on turbulence; cost grows with batch (unmeasured at 4096). |
The refactor with weights in ordinary coordinates (kronq_orig.py, --orig 1) equals KronQFast in fp32 but loses 60 % on the rough
field in TF32: in ordinary coordinates a slow mode is a tiny pattern over many weights, below TF32's resolution. Not adopted.

## 3. Results with the hybrid (TF32)

| problem | KronQ hybrid | previous KronQ | AdamW best |
| field rough preset (5000 steps, batch 256) | 0.0597 / 0.0565; with --rot_hp 0.0536 / 0.0517 | 0.0622 / 0.0626 | 0.39 (equal steps) |
| field smooth, batch 1024, 10000 steps | finals 4.2e-5 / 5.1e-5, 1-2 small spikes | 11 spikes per seed, finals to 5.5e-3 | |
| field rough, batch 1024, 10000 steps (best) | 0.0079 / 0.0076, no spikes | 0.0073 / 0.0078 with spikes; fp32 0.0033 | |
| turbulence increment, capped budget, wd 1.0 | 0.1114 / 0.1119 | 0.1109 | 0.1563 |
| turbulence increment, no budget, lr 2e-4, decay 1e-3/step | 0.1150 | 0.1148 | |
| turbulence direct, capped budget, wd 0.3 | 0.0631 | 0.0628 / 0.0631 | 0.0600 |
| turbulence direct, no budget, lr 2e-4, decay 1e-3/step | 0.0478 (ONE seed) | not run before | 0.0600 |

## 4. Known limitations (stated, not fixed)

- The basis is the K-FAC approximation: the factors are averaged before their product, so the eigenbasis is that of
  avg(a a^T) x avg(delta delta^T), not of the layer's averaged curvature avg(a a^T x delta delta^T). Exact for one sample; on the slow
  modes the measured coupling is 10^3-10^6 and the step mispricing 8-235x (REPORT 25 fork). The gradient's coordinates and the momentum
  are exact in any basis; what the approximate basis costs is that the per-coordinate (diagonal) rule acts in coordinates where the
  averaged curvature is not diagonal on the slow modes. An exact basis is not feasible at width 4096 (the slow end needs the whole
  spectrum); an older small-width measurement put the Kronecker basis ~3x behind the exact one per step on rough. Not pursued (user,
  2026-10-14). TO DO: THEORY.md line ~1376 and MECHANISM.md overstate this (they identify the optimizer's basis with the exact block's
  eigenbasis / check Lemma 1 in the Kronecker basis); correct the wording.
- Precision: the rough target is precision-sensitive for KronQ (fp32 2.2x better than TF32 on the long rough run); AdamW is not (fp32 =
  TF32 to 4 digits), because only a method that reaches the slow modes needs them resolved. bf16 untested.
- Runtime: 3-4x AdamW per step measured; the FLOP model allows 1.1-1.3x (section 6, item 3).

## 5. Theory-level reading of the session

- Finite data: the slow-mode push removes gradient descent's implicit regulariser; an explicit one (decoupled decay) is required; its
  controlling quantity is the per-epoch forgetting factor exp(-lr wd N / B) (other session's item 54), bracketed by a cap from the
  equilibrium w / wd. No training-only statistic times the memorisation (identifiability: signal and frozen sampling error enter only
  through their sum); turbulence has 14k-28k effective samples in 400k pairs.
- Fresh data: the budget is the noise shrinkage of the completion step that the magnitude rule lacks (other session's item 57).

## 6. Plan

1. CONFIRM THE DIRECT RESULT (running next). Direct, no budget, lr 2e-4, decay 1e-3/step, hybrid, seeds 2 and 3; AdamW at the same decay
   per step (lr 2e-4 with wd 5.0, lr 1e-3 with wd 1.0). Prediction: KronQ 0.046-0.050 on both seeds; AdamW >= 0.06. Null: a seed above
   0.055, or AdamW at matched decay within 5 % of KronQ.
2. DROPPED (user, 2026-10-14): replacing the budget by a shrunk fixed point of the learned weight (--wiener; code stashed, not on the
   branch). It needs two extra per-parameter buffers (mean sign, previous gradient): optimizer state ~2 -> ~3 floats per parameter, +50 %,
   unacceptable for large networks. It only targets fresh data without decay (the rough field, where the budget is the noise shrinkage);
   on finite data the budget is already unnecessary (lr 2e-4 + decay, item 57). Note for that regime only: keep the budget there.
3. SPEED (engineering, after 1-2): subsample the probe and the budget forward, fuse the rotated forward pass, remove the per-step GPU
   synchronisations, run the refresh on a side stream. Target 1.1-1.3x AdamW per step.
4. CRITICAL BATCH REPORT (no compute): re-summarise scale_crit (223 runs) with best-along-run values now that the spikes are explained.
5. bf16: the hybrid under bf16 autocast on the rough preset and turbulence.
6. Notes: correct THEORY.md / MECHANISM.md on the Kronecker basis (section 4).
