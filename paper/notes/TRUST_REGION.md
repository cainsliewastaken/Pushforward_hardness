# The trust region of the step, and the first-order derivation of sign push + cap + budget (2026-10-13, ~01:00)

CORRECTION (2026-10-13 11:45, REPORT 61 item 20). The per-layer law delta_j = c_l / sqrt(alpha_j) is CONFIRMED on both
hidden layers (+-20 % over eight decades of alpha on the input layer). But the identification of the sign push with the
box branch of section 2 is wrong in magnitude: the operating push (lr eta w ~ 1e-3 per mode) is ~1000x inside every box,
so the box does not bind and the sphere is the only active constraint; pushing modes to the box (--qa -1) destroys the
fit. The optimum of section 2 under sphere x box is sparse (box pushes only on modes whose residual coordinate exceeds
c sqrt(sigma_i) / r, GN elsewhere) and has not been tested. Sections 1, 2 (the geometry and the optimisation) and 5 (the
law) stand; section 3's reading of the sign push as that optimum is withdrawn.

Written after the Taylor-radius scans (REPORT 61 item 15, COLLECTIVE_RULE.md section 2) and the Krylov test (item 17).
Status: a derivation from two MEASURED facts about the network at width 4096; the facts are measurements, the
optimisation is exact; what it predicts was checked where marked.

## 1. Two measured facts (rough target, width 4096, reference run, scans at steps 100 / 500 / 1500)

(F1) Output-space sphere. The length at which the relative Taylor remainder reaches 0.1 corresponds, for every direction
     scanned (the step, sign(g), g, two Kronecker ridges, a random direction, a single mode), to the same linear output
     change ||J d|| ~ r ||E|| with r = 0.19 at step 500 (0.13 at 1500; within a factor 1.5 across directions whose
     leverages span 250x). The first-order model holds while the batch outputs move by less than ~r ||E||, whatever the
     direction.
(F2) Parameter-space box. In parameter units the same radius spans 200x: a single mode may move ~0.01-0.1, a spread
     direction 2-5. Spread steps are allowed because they stay inside every mode's own small radius at once; the
     constraint is per coordinate, |d_k| <= delta_k, not on the Euclidean length.
On the smooth target (F1) fails (output radii 3-600 across directions, all beyond ||E||): the step there is
completion-limited, and the budget acts as a damping, not a trust region (COLLECTIVE_RULE section 2).

## 2. The first-order step under (F1) and (F2)

In the kernel eigenbasis (v_k in output space, s_k = sqrt(lambda_k) the leverage, e_k = <v_k, E>), a parameter step with
mode coordinates d_k has output response J d = sum_k s_k d_k v_k and first-order decrease sum_k e_k s_k d_k. Maximise

    sum_k e_k s_k d_k   subject to   sum_k (s_k d_k)^2 <= r^2 ||E||^2   (F1)   and   |d_k| <= delta_k   (F2).

KKT: d_k = clip( e_k / (2 mu s_k), -delta_k, delta_k ), mu >= 0 the sphere multiplier (zero if the sphere is slack).
With g_k = e_k s_k and lambda_k = s_k^2:
  * modes where the box is slack:   d_k = g_k / (2 mu lambda_k)        the Gauss-Newton (completion) step, globally scaled;
  * modes where the box binds:      d_k = delta_k sign(g_k)             a push of fixed size with the gradient's sign,
                                                                       independent of |g_k|;
  * mu from the sphere: scale the whole step so that ||J d|| = r ||E|| when it would exceed it.
The box binds on the modes with |e_k| / s_k > 2 mu delta_k: large residual coordinate and small leverage, i.e. the
held-rough modes. The gain per unit of output change on mode k is e_k, so with the output radius as the budget the
slow modes carrying the rough residual are the best use of it, and the ridge / gradient / Krylov steps, which spend the
output radius on the fast modes (small e_k after a few steps), are the worst.
Proof: separable concave objective with a single coupling constraint; the clip is the per-coordinate stationarity
condition; mu by monotonicity.  []

## 3. The optimizer is this solution

  cap  |d_ij| <= g_ij / (lam_ij lr eta L)   = the Gauss-Newton branch (Kronecker lam for lambda_k; the 1/L and the global
                                             scale play the part of 1 / (2 mu));
  sign push  D = w_ij sign(C_ij)             = the box branch (momentum sign for sign(g_k); w_ij = exp(u_ij) the estimate of
                                             delta_k, which nothing measures directly and the one-step hypergradient learns
                                             from realised gains);
  budget  ||f(theta + d) - f(theta)|| <= 0.2 ||E||   = the sphere (F1), with the constant 0.2 = the measured r;
  learning-rate decay                        = r(t) decreasing, outside this argument (minibatch annealing).
What this explains (REPORT 61 items 4-15): the width-4096 collapse without the budget (a sign step outside the sphere);
0.2 > 0.1 (the measured r); the budget as the dominant rule (7x on smooth, item 13); the cap worth little (binds on few
modes); a learned weight ~1 on convex slow modes with negligible gradient winning (the box branch; no lookahead, lift or
concavity needed); every per-mode quadratic rule losing (all assumed a parameter-space sphere: ridge, T-step allocation,
LM, lift-aware two-regime); the Adam rule losing (its normalisation is neither branch); and the Krylov step losing
(item 17: a one-step optimum for the parameter sphere, and in practice a truncated one).

## 4. Checks

  * (done) Alignment at equal output change, rough, layer 1, step 500: cos(E, J u) = 0.43 for sign(g), 0.34 for the step,
    0.39 / 0.37 for the Kronecker ridges, 0.20 for the gradient, ~0 for random and single-mode directions (scan with the
    alignment column, runs/scale_krylov/res_j5). The sign direction is the best-aligned of the seven: it carries the
    slow-mode component of E that the gradient does not. Smooth: ridge4 0.58 > sign 0.34 > ridge2 0.31 > step 0.27 >
    grad 0.08 (the residual is reachable by a small-damping ridge there).
  * (done) Krylov step (CG on the exact K, M = 1 / 3 / 8, damping 0.1-1 x Rayleigh, output radius 0.2 ||E||): alignment
    0.06-0.17 on rough and 0.07-0.3 on smooth along the run, CG residual 0.3-7 ||E|| (the solve barely reduces the batch
    residual at these M: the series converges slowly for a residual that needs the slow modes, as Thm stepkrylov says),
    error 0.48-0.56 rough (worse than AdamW 0.39) and 0.006-0.04 smooth (40x worse than the reference). Prediction (loses
    on rough) confirmed; the smooth prediction (competitive) was wrong because affordable M truncates the series far
    from the ridge: the Kronecker diagonal solve is a better ridge than 8 exact Krylov terms.
  * (DONE, REPORT 61 item 19) delta_k across the spectrum: flat in parameter units on the hidden layer (3-6, top deciles
    0.6-1; learned weight flat 0.97-1.0), proportional to 1 / leverage on the input layer (constant output radius 0.2-0.7 per
    mode; learned weight rising 0.9 -> 2.9 toward slow modes, tracking the direction but not the 100x size). The box branch
    is therefore d_k = delta sign(g_k) on the hidden layer and d_k ~ sign(g_k) / s_k (equal output change per mode, q = -1)
    on the input layer; --q_in -1 implements the latter.
  * (was) delta_k across the spectrum: scan single modes at each lambda decile (added to --radius_scan as mode_d0..d9);
    the box branch with a common push assumes delta_k roughly flat, which REPORT 57 found for the second derivative at
    width 160.
  * (to do) the learned weight u_ij against the measured delta_k decile by decile (Cor. 8(c) replaced by: u tracks the box).
  * (open) the smooth-target length: a damping with no derivation; the late-run degradation; momentum and decay.

## 5. Why the box (sketch, to be proved)
For one layer with pre-activation z = W a, a step dW changes z by dW a, and the layer's Taylor remainder is quadratic in
dz = dW a per sample and unit: ~ act''(z) dz^2. In the input eigenbasis the batch sum of dz^2 is exactly sum_ij d_ij^2
alpha_j (no cross terms, the a~_j being orthogonal on the batch), while the remainder's effect on the OUTPUT passes
through the downstream map; a single mode (i, j) concentrates dz on one output eigen-direction q_i with magnitude
d_ij sqrt(alpha_j), a spread step distributes it. The per-mode radius is delta_ij ~ c / sqrt(alpha_j) up to the
downstream factor, which is the box; its dependence on j (1 / sqrt(alpha_j)) is what the decile scan measures.
