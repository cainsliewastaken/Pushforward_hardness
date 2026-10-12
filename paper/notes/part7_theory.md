
## 36. FLOP comparison with Adam (code/adam_l63.py, segflow_generic_thresholds.py) -- the honest optimizer verdict

FLOP model in forward-equivalents: forward 1, backward 2, forward-mode product 2, reverse-mode product 3, nested second
derivative 4. One Adam or GD step = 3; one Lanczos iteration = 5. Full-batch Adam, learning rates 1e-3..1e-1 swept,
best reported; target = the saved GD run's final training error.

| design | Adam best lr | Adam steps | Adam FLOP-eq | SegFlow FLOP-eq | GD FLOP-eq | Adam/SegFlow |
|---|---|---|---|---|---|---|
| H48 d2 D0.5 | 0.03 | 700 | 2100 | 9087 | 981630 | 0.23 |
| H48 d2 D1.0 | 0.01 | 1900 | 5700 | 44507 | 1168689 | 0.13 |
| H48 d3 D0.5 | 0.01 | 1300 | 3900 | 9395 | 2105007 | 0.42 |
| H48 d3 D1.0 | 0.01 | 1800 | 5400 | 24641 | 2103648 | 0.22 |
| H160 d2 D0.5 | 0.01 | 800 | 2400 | 15093 | 2859582 | 0.16 |
| H160 d3 D1.0 | 0.01 | 1700 | 5100 | 90707 | 7134483 | 0.06 |
| H96 d3 D0.75 s0.5 | 0.01 | 3100 | 9300 | 70225 | 11645535 | 0.13 |

Adam reaches the same training error with 2.4 to 18 times fewer FLOPs than SegFlow, and 200 to 1400 times fewer than
GD. On the two non-L63 problems (FLOPs to reach E0/4, E0/16, E0/64): GELU-MLP SegFlow 5.0k/20k/43k, Adam 0.3k/5.9k/20k,
GD 6.2k/>60k/>60k; tanh-CNN SegFlow 1.1k/1.4k/1.4k, Adam 0.3k/0.3k/0.45k, GD 5.9k/11k/>15k. Same picture.

Where the loss is (H48 d2 D0.5, measured): Adam takes 969 steps of mean parameter norm 0.035 (path length 33.7);
SegFlow takes 52 segments of mean norm 0.55, max 1.64 (path length 28.4). The segments are 15x larger than Adam's
steps but cost 50x more: 8850 of SegFlow's 9087 FLOP-eq are the 30 Lanczos iterations per segment. Smaller Krylov
spaces do not work: m = 8 needs 1120 segments with 218 rejections, m = 4 does not converge in 2000 segments -- the
residual re-excites the fast modes every segment and the exponential needs all active eigenvalue scales. Adam's
diagonal preconditioner costs nothing and persists across steps; SegFlow's exact function-space preconditioner is
rebuilt from scratch every segment and discarded. A recycled-Krylov variant (keep the top Ritz pairs and their
K-images across segments, add a few new Lanczos vectors) is the one lever that could reach parity; a first
implementation (segflow.py, recycle_k) was wrong and is left as unfinished.

Conclusion: SegFlow is not an optimizer contribution. Its value is that it reproduces the gradient-flow path at 1-2% of
GD's cost (tracking 1.4% with the H2 radius rule at kappa = 0.1), which Adam does not do, and that makes the
path-wise measurements below affordable. Taylor-radius steps beat GD because GD is crippled by stiffness; Adam already
removes most of that stiffness for free.

## 37. The H2 radius rule (theory -> optimizer)

Segment radius chosen a priori from the second directional derivative along the step: 1/2 delta^2 ||f''[d,d]|| =
kappa x (predicted residual decrease at that radius), solved in the Krylov space (one nested forward product per
segment). Rejected steps: 0 at kappa <= 0.3 (vs 5-11% for the trust-region rule). Cost is not reduced (kappa = 1:
2010 pairs vs 1770 trust-region) but fidelity to the gradient-flow path becomes a dial: kappa 0.1 / 0.3 / 1.0 give
tracking medians 1.014 / 1.05 / 1.16 with ranges [0.99,1.04] / [0.94,1.11] / [0.99,1.34]. The second-order
corrector (cancel 1/2 f''[dtheta,dtheta] inside the Krylov space) gave nothing (73 vs 59 segments), consistent with
Section 30: order two barely extends the radius for the depth-2 network.

## 38. The roughness theory measured along the path (code/segflow_rough.py, rough_analysis.py; runs/rough)

Measured at mid-segment points of the kappa = 0.1 path (segment-end points carry the fresh Taylor remainder, which
inflates <E,KE> by ~40x through a 0.3% energy fraction on the top eigenvalue -- that bias was the first result and is
avoided). Check: the finite-difference slope -d ln||E||^2/dt equals 2<E,KE>/||E||^2 within a few percent at every
checkpoint after the first, so the identities of Proposition "Energy of the rough part" hold on these paths and the
numbers below are on-path. P_m = graph-Laplacian high-pass above mode m (Section 4.1 of the theory), m = 10 here.

| design | c = <E_r,KE_r>/(sigma^2 ||E_r||^2): start / mid / end | c ~ E^p | drive realised / bound (median) | sigma_10^2 start -> end | lambda_1 start -> end | H2 range |
|---|---|---|---|---|---|---|
| H48 d2 D0.5 | 0.046 / 0.0025 / 0.0007 | 1.15 | 3.5e-5 | 0.99 -> 10.2 | 10.6 -> 72 | 0.6-2.2 |
| H160 d2 D0.5 | 0.061 / 0.0009 / 0.0005 | 1.56 | 7e-6 | 3.0 -> 15.0 | 32 -> 162 | 0.9-4.2 |
| H48 d3 D0.5 | 0.051 / 0.0009 / 0.0005 | 1.21 | 1.8e-5 | 1.0 -> 14.6 | 11 -> 87 | 1.2-6.2 |
| H48 d2 D1.0 | 0.019 / 0.0001 / 0.0002 | 2.36 | 1.6e-5 | 0.99 -> 39 | 10.6 -> 95 | 0.6-17 |
| H48 d3 D1.0 | 0.018 / 0.0001 / 0.0002 | 1.50 | 2.9e-5 | 1.0 -> 132 | 11 -> 209 | 1.0-58 |
| H96 d3 D0.75 s0.5 | 0.032 / 0.0001 / 0.00003 | 0.88 | 2.7e-6 | 0.55 -> 99 | 5.6 -> 157 | 1.2-45 |

What this says about the closure of the theory:
* Assumption "efficient roughness supply" with c = O(1) fails by 1-3 orders: the kernel's Rayleigh quotient on the
  rough residual is 2-6% of the peak feature roughness at the start and 1e-4..1e-3 at the end, and c falls as a
  power 0.9-2.4 of the error. The roughness the features acquire sits mostly on rough directions that do not carry
  the residual (the residual migrates to ever smaller eigenvalues while sigma_R grows 10-200x).
* The a priori drive bound Theta_R <= sigma lambda_1^{1/2} H2 ||E|| is loose by ~1e5 (realised/bound 3e-6..4e-5).
  The reduced ignition model taken at the bounds therefore predicts nothing quantitative; with run-median constants
  (c, phi) it is off by orders of magnitude (constant c is the wrong closure).
* The exact identity does organise the curve. The effective eigenvalue the residual sits at,
  q(t) = <E,KE>/||E||^2 = -(1/2) d ln||E||^2/dt, follows q = q1 E^alpha after the lazy phase for the s = 1 designs:

| design | alpha | q1 | time to final error from the law / measured |
|---|---|---|---|
| H48 d2 D0.5 | 1.22 | 1.0e-2 | 2840 / 3090 |
| H160 d2 D0.5 | 1.38 | 1.7e-2 | 2660 / 2890 |
| H48 d3 D0.5 | 1.05 | 2.3e-2 | 2300 / 2880 |
| H48 d2 D1.0 | 0.40 | 1.2e-3 | 2650 / 3140 |
| H48 d3 D1.0 | 0.28 | 2.3e-3 | 1850 / 2690 |
| H96 d3 D0.75 s0.5 | 0.08 | 3.7e-3 | 1690 / 11900 (law fails: q not a power of E) |

  The exponent is set by the target roughness (1.0-1.4 at Delta = 0.5, 0.3-0.4 at Delta = 1.0), the prefactor by
  architecture and roughness (depth 3 doubles q1 at either Delta; width 160 vs 48 x1.7; Delta 1.0 vs 0.5 /10). The
  frozen kernel never reaches these error levels at all (its q_f(E) is undefined below E/E0 ~ 0.4), which is the held
  rough error of the theory seen directly: everything below that level is bought with eigenvalue gain.
* Not shown: that (alpha, q1) can be read from initialization. Six designs at three roughness levels, with one
  (the rich-scaled s = 0.5 design) not following the law, are not enough; and q1 does not track sigma_10(0), H2(0)
  or lambda_1(0) in any simple way across the six (e.g. q1/H2(0) = 0.005, 0.005, 0.008 at Delta = 0.5 and
  0.0006, 0.0008 at Delta = 1.0).

Cost of these measurements: 100-330 s CPU per design on a login node (the path at 1-2% of GD plus ~100 passes per
checkpoint for sigma_m, c_m, lambda_1, H2). The 28-design held-out set would take ~1-2 CPU-hours.
