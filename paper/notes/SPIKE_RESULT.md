# Late-run spikes on the field: what the instrumented runs show (branch xfold, 2026-10-14)

Jobs 59767397 (scale_spikefield, crashed at the realised-change logger, onsets only) and 59767814 (scale_spikefix, 12 runs, frozen code
8927d51). Spike = an evaluation > 3x the running best after 10 % of training.

| run | final | best | spikes | mean log-weight at 30 / 60 / 90 % | weights > e^4 at end | budget kept at 30 / 90 % |
| smooth B512 T20000, reference | 81.9 | 6.8e-5 | 38 | -1.00 / -1.53 / -0.66 | 0.1 % | 0.49 / 0.51 |
| smooth B512 T20000, hyper_budget | 7.2e-5 | 7.1e-5 | 0 | -0.68 / -3.39 / -3.17 | 10 % | 0.14 / 0.79 |
| smooth B1024 T10000, reference | 5.5e-3 | 4.9e-5 | 11 | -0.46 / -1.49 / -1.01 | 0.5 % | 0.20 / 1.00 |
| smooth B1024 T10000, hyper_budget, seed 1 / 2 | 0.0227 / 1.6e-3 | 5.5e-5 / 6.0e-5 | 6 / 5 | -0.22 / -0.72 / -2.03 (seed 1) | <0.01 % | 0.05 / 0.24 |
| rough B1024 T10000, reference | 0.0125 | 0.0077 | 5 | -0.32 / -0.27 / 0.14 | 0.7 % | 1.00 / 1.00 |
| rough B1024 T10000, hyper_budget | 0.0074 | 0.0074 | 0 | -0.47 / -0.45 / 0.16 | 0.3 % | 1.00 / 1.00 |
| rough B1024 T10000, decay 3e-4 / 1e-3 / gated 1e-3 | 0.0080 / 0.0103 / 0.0075 | | 0 / 3 / 0 | | | 1.00 |
| presets B256 T5000 with hyper_budget: rough / smooth | 0.0717 / 1.63e-4 (refs 0.0622 / 1.42e-4) | 0.0628 / 1.63e-4 | 0 / 0 | | | |

1. The weight-inflation account is refuted: the mean log-weight FALLS in every reference run; only 0.1-0.7 % of weights sit near the clamp,
   in spiking and non-spiking runs alike.
2. On the rough target the budget never binds (kept = 1.00 throughout), so --hyper_budget computes nothing there: the reference and the
   "fix" are the same computation, yet one spiked 5 times and the other not at all. Spike occurrence varies between identical runs (GPU
   non-determinism), so single-run comparisons of remedies are not evidence; that also covers the decay runs (0 / 3 / 0 spikes).
3. --hyper_budget did not stop the spikes on smooth B1024 (both seeds) and costs ~15 % at the presets. Not adopted.
4. Earlier rejections stand: not Cholesky-QR (exact QR spiked), not the budget rescale (realised change 1.0-1.3x the budget at onsets).
5. What remains consistent with every observation: spikes need a low loss and many steps, are transient with recovery, appear in AdamW too
   (21 of 149 runs at lr 1e-3 / 3e-3), and occur with the budget slack. That is the profile of the chain's sharpening / edge-of-stability
   statement (THEORY.md); untested. The test: the top kernel eigenvalue times the effective step along spiking runs, several seeds each.

## TF32 on vs off (job 59768403, 4 nodes, 16 runs, frozen code 5f824ee; tf32_summary.py)
| config | TF32 on: seeds spiked, best | fp32: seeds spiked, best | median function change per refresh (on / off) |
| smooth B1024 T10000 | 4 of 4, 4.7-5.0e-5 | 4 of 4, 3.9-4.9e-5 | 5.9e-2 / 5.0e-5 |
| smooth B512 T20000 | 2 of 2, 6.5-8.4e-5 | 2 of 2, 7.7-8.5e-5 | 2.0e-2 / 3.7e-5 |
| rough B1024 T10000 | 2 of 2, 0.0073-0.0078 | 0 of 2, 0.0033-0.0034 | 1.9e-2 / 3.9e-5 |
1. Precision is not the cause on the smooth target (fp32 spikes in every seed, at the same steps), although fp32 cuts the per-refresh
   function change ~1000x. On the rough target fp32 removes the spikes AND reaches 2.2x lower loss (0.0033 vs 0.0073): TF32 sets a floor.
2. THE BASIS BREAKS FIRST, in every smooth run in both precisions: the accumulated basis error (max |U^T U - I|, float64) jumps from <1e-4
   to > 0.5 within 50-300 steps, at the same refresh where the new rotation V from Cholesky-QR has error > 1e-2, and the probe loss first
   jumps 250-1500 steps later (e.g. fp32 seed 1: basis > 0.5 at 7450, probe at 7747; TF32 seed 1: 6997, 7252). Rough fp32: the basis never
   leaves 1e-5 and nothing spikes. Rough TF32: basis failure at 5535 / 6022; the earliest probe jumps (2200) precede it and are early-training
   noise; the validation spikes are not all after it, so the rough case is less clean.
3. Cause, as measured: the refresh's Cholesky-QR returns a non-orthonormal rotation at some point late in the run (the jitter 1e-12 x mean
   diagonal lets the Cholesky succeed on an ill-conditioned Gram instead of falling back to QR); the basis loses orthonormality; the loss
   spikes a few hundred steps later. Open: the earlier --cholqr 0 run also spiked, without the basis being logged, so whether exact QR
   avoids the failure is not known. Fix to test (a numerical guard, no new constant in the method): check V^T V after Cholesky-QR and fall
   back to Householder QR in float64 when it is not orthonormal to working precision; rotations and transports in fp32 with TF32 kept for the
   network's own matmuls.

## QR guard (job 59768606, frozen code 83e58e9; same configs and seeds as scale_tf32, --qr_guard 1)
| config | unguarded (scale_tf32): seeds spiked, spikes, final | guarded: seeds spiked, spikes, final | basis error max | guard fallbacks |
| smooth B1024 fp32, seeds 1-2 | 2 of 2 (all 4 of 4), 8 / 5, 3.2e-4 / 8.5e-5 | 0 of 2, 0 / 0, 4.6e-5 / 3.8e-5 | 2e-5 (was 1.0) | 5580 / 5826, first at 7346 / 6753 |
| smooth B1024 TF32, seeds 1-4 | 4 of 4, 11 / 11 / 6 / 9, 5.5e-3 / 4.8e-4 / 2.7e-2 / 1.3e-3 | 3 of 4, 2 / 1 / 0 / 3, 4.4e-5 / 3.8e-5 / 4.1e-5 / 6.0e-5 | 4.5-5.5e-3 (was 1.0) | 3200-7300, first 6680-7090 |
| rough B1024 TF32, seeds 1-2 | 2 of 2, 3 / 1, 0.0102 / 0.0473 | 2 of 2, 1 / 1, 0.086 / 0.0102 | 5.5e-3 | 10778 / 2908, first 4854 / 5454 |
1. CONFIRMED: the catastrophic spikes are the Cholesky-QR returning non-orthonormal rotations late in training (the guard fires thousands of
   times from step ~6700 on the smooth target, ~4900 on the rough, exactly where the basis broke before). With the guard and fp32: no spikes,
   basis error 2e-5. With the guard and TF32: the catastrophic spikes are gone (finals within 1.5x of the best instead of 10-700x); small ones
   remain, with the basis error held at ~5e-3 by TF32 drift in the rotations and transports.
2. Rough TF32 still spikes (one large final spike, 0.086); rough fp32 without the guard did not spike and reached 2.2x lower loss: on the
   rough target the remaining problem is the TF32 drift, not the Cholesky failure.
3. Cost: guarded TF32 129-138 s against 135 s unguarded (no measurable cost); guarded fp32 256-259 s against 233 s (+10 %), from the
   float64 Householder fallbacks in the last third of training.
Next: the guard on by default; basis rotations, transports and the rotated-parametrisation products in fp32 with TF32 kept for the
network's own matmuls (removes the drift at a fraction of full fp32's 1.7-1.9x runtime).

## The refactor (kronq_orig.py, --orig 1; job 59769038, frozen code 70a9e8d; orig_summary.py)
CPU, fp32, width 96, 200 steps: trajectory within 2.4e-4 of KronQFast, refresh function change exactly 0, basis error 6.5e-8.
GPU, all TF32, guard on: smooth preset 1.26e-4 / 1.25e-4 (KronQFast 1.42e-4 / 1.38e-4: 10 % better); smooth B1024 T10000 seeds 1-4:
0 / 1 / 4 / 0 spikes (mild), finals 3.0e-5 - 7.8e-5, refresh change 0, basis error 1e-5 (the guard still fires 1500-9500 times);
rough preset 0.099 / 0.0999 (KronQFast 0.0622 / 0.0626: 60 % WORSE, from step 250 on: 0.44 vs 0.35, 0.19 vs 0.135 at 1000);
rough B1024 T10000 0.0111 / 0.0113, no spikes (KronQFast TF32 best 0.0073, fp32 0.0033); runtime 5000 steps: batch 4096 100 s
(KronQFast 127 s), batch 1024 62 s (59 s). Not adopted: the rough deficit is open; scale_orighp.sh tests whether TF32 in the
refactor's own products (gradient rotation, statistics in ordinary coordinates, refresh) causes it.

## Precision pairs (job 59769209; scale_precpair.sh), rough preset, batch 256, 5000 steps, seed 1
| run | 250 | 1000 | 2000 | 3000 | 4000 | 5000 |
| refactor fp32 | 0.289 | 0.133 | 0.111 | 0.080 | 0.063 | 0.054 |
| KronQFast fp32 | 0.288 | 0.132 | 0.111 | 0.078 | 0.059 | 0.049 |
| refactor TF32 | 0.441 | 0.190 | 0.135 | 0.119 | 0.107 | 0.099 |
| KronQFast TF32 | 0.347 | 0.135 | 0.117 | 0.089 | 0.071 | (end spike) |
| AdamW lr 3e-4 fp32 / TF32 | 0.593 / 0.593 | 0.502 / 0.503 | 0.454 / 0.456 | 0.423 / 0.423 | 0.402 / 0.403 | 0.390 / 0.390 |
1. In fp32 the refactor equals KronQFast (to 1 % through step 2000, 10 % at the end): its rough deficit in TF32 is precision, not a
   difference in the method. 2. KronQFast itself gains in fp32 (0.049 against the TF32 runs' 0.062-0.071). 3. AdamW is identical in fp32
   and TF32: the rough target's precision sensitivity belongs to a method that reaches the slow modes; AdamW never does.
Hybrid (KronQFast with --refresh_hp, --reorth_transport, --qr_guard; CPU check: the function change at re-orthonormalisations falls to the
level of an ordinary refresh) and --rot_hp (activation rotations in fp32): scale_hybrid.sh.

## Hybrid (job 59769295, frozen code 078a7d0; scale_hybrid.sh): weights in the basis, refresh in fp32, weights transformed at
## re-orthonormalisation, QR guard; network TF32
| run | hybrid | hybrid + activation rotations fp32 (--rot_hp) | KronQFast TF32 | KronQFast fp32 |
| rough preset seed 1 / 2 | 0.0597 / 0.0565 | 0.0536 / 0.0517 | 0.0622 / 0.0626 | 0.049 (seed 1) |
| rough B1024 T10000 seed 1 / 2, best (final) | 0.0079 (0.0101) / 0.0076 (0.0100), no spikes | not run | 0.0073 / 0.0078, 3 / 1 spikes | 0.0033 / 0.0034 |
| smooth B1024 T10000 seed 1 / 2 | 1 / 2 small spikes, finals 4.2e-5 / 5.1e-5 | | 11 / 11 spikes, finals 5.5e-3 / 4.8e-4 | |
Basis error 1e-5 - 7e-5 (was 1.0). Runtime: rough preset 66-67 s (KronQFast TF32 63 s), with --rot_hp 65-67 s (no measurable cost at batch 256);
rough B1024 T10000 with the instrument 187-191 s (KronQFast TF32 with the instrument 160 s, +19 %). The per-refresh function change is still
~5 % of the residual in TF32: it is now the rotated forward pass's own TF32 rounding (the two forwards compared use different bases), not
stored-weight error. Open: --rot_hp at batch 1024 / 10000 steps (the long run where fp32 is 2.2x better) and its cost at large batch.
