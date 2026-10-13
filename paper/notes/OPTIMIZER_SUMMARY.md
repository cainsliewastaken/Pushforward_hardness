# What worked and what did not: optimizer work, 2026-10-10 to 2026-10-12 (this conversation)

Testbed unless noted: Lorenz-63 one-step map, tanh MLP, squared loss, full batch. "Rough" = Delta 1.0, "smooth" =
Delta 0.5. Detailed numbers: REPORT.md Sections 31-58; component costs: FEASIBILITY.md.

## Worked

| finding | evidence | where |
|---|---|---|
| Sign push in a kernel (Gauss-Newton) eigenbasis is the strongest per-step rule found | exact basis: 4x Adam (rough), ~50x (smooth) per step at n = 20000 | Sec 58 |
| Kronecker (K-FAC) version of it (KronQ2) beats Adam per FLOP on L63 | ~3.1 fwd-eq/step vs Adam 3; rough 0.18-0.22 vs Adam 0.41, smooth 0.003 vs 0.029 (n = 20000, 3000 steps). Caveat: Adam lr grid of 3 values, best at the edge | q grid, Sec 58 |
| Gauss-Newton cap per mode (never move a mode past its own GN step) | 4-5x at n = 2000; free | Sec 56 |
| Momentum explained: gradient flow in rescaled time + an edge-of-stability gain | beta 0.5 and 0.9 track GF to 6 % / 30 %; at the edge the tail lifts 2-12x more; accounts for ~34x of Adam's 31x momentum gain | Sec 53 |
| Function-space Taylor remainder as a step-size detector | monotone in step size, separates good from failing settings by 10x; free if checked round-robin | Sec 56 |
| The optimal (q, lr) ridge sits at roughly constant remainder | exact rough 0.09-0.15, KronQ2 rough 0.11-0.21, KronQ2 smooth 0.012-0.035 | q grid |
| Per-layer lr set online by the remainder | rough: 0.183 with no lr tuning, better than every fixed KronQ2 grid point (0.218) | q grid |
| The fitting "wave" can be located for free | removal-weighted median eigenvalue tracks 1/t late (t lam_wave 0.4-2.8), band ~1 decade wide | Sec 54-55 |
| Error-controlled cache refresh | staleness measured as change of the first-order decrease; refresh interval 10 -> 70, same error, -15 % cost | smoke |
| Memory reductions keep accuracy at small size | blocked factors (b = 8): same error, 3.1P vs 7.3P; diagonal sides: 1.2P, small loss | smoke |
| Theory audit of the design | leverage s_k is the Jacobian singular value; normaliser = lr scale within a layer and an implicit layer balance across layers | Sec 57, chat |

## Did not work

| attempt | result | where |
|---|---|---|
| SegFlow (exact linear-flow segments, Krylov) | 45-383x fewer passes than GD, but Adam 2.4-18x cheaper in FLOPs (Lanczos cost) | Sec 31-41 |
| Krylov recycling, minibatch segments | negative | Sec 40-41 |
| Muon vs Adam on smooth field regression | Muon loses 2-8x in every setting (it wins 10-100x on L63) | Sec 42-47 |
| Roughness-supply objective, truncated spectral update, per-layer supply allocation, band-wise output-roughness preconditioning | all negative | Sec 48-52 |
| Push windows keyed to the residual Rayleigh quotient | cut the push on the residual itself; up to 40x worse | Sec 54 |
| Windows keyed to the wave | no training gain; only a validation gain that was a small-data artefact | Sec 55 |
| Front-damped / per-mode critically damped momentum | ties or losses; per-mode damping lifts the tail most but fits worst | Sec 54 |
| Derived trust-region Newton (exact GN + measured residual curvature) | loses 3-4x per step to sign push | Sec 58 |
| Low-rank trust-region Newton (32 modes) | loses badly (0.60 vs 0.10 rough) | Sec 58 |
| Global (inter-layer) GN basis for equal push | worse than per-layer bases | sweeps |
| Loss-based trust-region lr controller | throttled the lr 100x and never recovered | Sec 56 |
| Per-mode Taylor radius from \|f''\| | predicts q ~ +0.8, opposite of what works | Sec 57 |
| RMS (push proportional to drive persistence) | persistence predicts which modes get fitted, but scaling by it loses to sign | chat |
| Per-mode hypergradient (IDBD-style) | mostly an automatic lr increase; learned per-mode shape worth ~4 % (small test; large-n test running) | chat |

## Corrected claims (things I said that the data overturned)

* "q = -0.25 works everywhere": it was the fixed lr; q and lr substitute along the remainder ridge.
* "Pushing below the wave is the bad idea": the validation gains came from overfitting a 2000-point sample.
* "The second-order term is error": along deep modes it is largely useful feature learning; the residual curvature is
  negative at initialisation and positive along training (the theory gives the identity, not the sign).
* "Momentum gives a square-root speedup": not at fixed beta; it is a time rescaling plus an edge effect.

## Correction (2026-10-12, after the horizon test, REPORT Section 60)

The earlier "common thread" (deep pushes pay off only through multi-step feature learning that greedy criteria
cannot see) is NOT supported: the best q does not become more negative at longer horizons, and the one-step
hypergradient is as good as a 100-step trace. The local trust-region Newton step lost because its curvature MODEL
mispredicts the one-step gain of deep pushes; measuring that gain per mode (hypergradient) works: it matches the
tuned grid optimum (rough) and beats it by 27 % (smooth) with no q or lr tuning (Section 59).

## Open

* The remainder level along the ridge is problem-dependent (rough ~0.1-0.2, smooth ~0.02): what sets it.
* The best grid points sit at the grid edges and the two bases prefer opposite ends of the ridge.
* The Kronecker basis loses ~3x per step to the exact basis on the rough target.
* Overfitting persists on the rough target at n = 20000 for the strong optimizers (train 0.09, val 0.20).
* Per-FLOP comparison with a properly tuned AdamW, minibatch, and the field (climate-like) problem.
* Memory at scale (blocking / one-sided / sketches) at large width.
