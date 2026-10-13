# Experiments required by the theory review of 2026-10-12

Everything below is to be run at n in the thousands (user decision 2026-10-11: no n = 200 number is quoted). Each item
names the statement it decides, what is computed, what the outcomes mean, and the cost class (login node / Slurm; every
Slurm run needs the user's explicit yes, account m5326 so far). Items are ordered by what they decide, not by cost.
The reference setting unless stated: L63 one-step map, tanh MLP H96 d2 and the depth/width designs of WRITEUP S3,
gradient flow proxy (GD at eta lambda_1 <= 0.5), budget T = 3000, n >= 2000 training points with a held-out set of the same
size. Save per-snapshot: weights, Jacobian on a fixed subsample, kernel eigenpairs, residual. Log git hash, config, seed,
wall time; keep results regenerable from tables.

## A. Decides whether the cutoff bound (Thm 48) has content

A1. The transported bracket with its leakage. (THEORY 19, both CORRECTIONS.) One run with fine stages (the chain's
    per-stage eigendecompositions; stage criterion EPS = 0.01). Track eigenvectors by continuity (maximal overlap between
    consecutive stages; record every avoided crossing between a mode in u and a mode in f). Compute: the tracked unfit
    set u at T; Pi_u c(0); lambda_u(t), lambda_f(t) on the tracked subspaces; rho(t) = ||A_uf||_2 (block spectral norm);
    E_u(t), E_f(t); the leakage integrals; both sides of the bracket against ||E(T)||.
    Outcomes: (i) floor and ceiling within a factor e of ||E(T)|| with leakage a correction -> the bound is a result;
    (ii) leakage comparable to the main terms -> the bound is a tautology in this regime and the moving-cutoff
    formulation (A2) is the one to use. Also record whether the tracked eigenvalues in u are nondecreasing (the
    hypothesis that gives int lambda_u < 1). Cost: the staged model at n ~ 2000 is one 6000 x 6000 eigendecomposition per
    stage, ~1e3-1e4 stages: Slurm, 1 GPU, hours; at n = 200 it is a login-node check of the code only.
A2. Moving-cutoff formulation. Same run: u(t) = {k : lambda_k(t) t < 1} from the current sorted spectrum; P(t) jumps when a
    mode's lambda_k t crosses 1; record the jump terms (residual energy of the crossing mode at its crossing time) and
    the continuous leakage separately. Compare with A1 on the same run: which formulation has the smaller remainder.
A3. Per-stage sign condition as an operator (Cor. 43(ii), Cor. 50, Thm 46(a)). Same run: W = span of the modes carrying
    90 % of the residual energy at each stage (current basis); compute lambda_min(V_W^T (K(t_s) - K(t_{s-1})) V_W) per stage
    and lambda_min of the accumulated increment on W. Outcomes: nonnegative after the transient -> (S) holds per stage on
    the residual's subspace and Thm 46(a) applies; negative on a set of stages -> record where (expected: late, lambda_1
    drift) and whether the negative part is small relative to the lift.
A5. The transverse fraction of the lift (Theorem 51, THEORY 22) and the hypothesis of Theorem 52 (THEORY 23) with the
    chain's FINE stages: per stage, q_s = <e_s, Delta_s e_s>, the transverse fraction at creation, the sign condition of
    Prop. 53 for the principal direction, and the pure-targeted surrogate; at the coarse snapshot resolution (2026-10-12,
    chain/targeted_lift.py) the hypothesis fails (transverse 5x at creation, 28 % of stages negative on the residual). Along a run: a(t) = <E, Delta E>/||E||^2 with
    Delta = K(t) - K_0, w = Delta E - a E, tau = ||w||/(a ||E||), A(T) = int a; the exact identity for E(T), the bound (b),
    the sufficient condition (c) and the measured ||E(T)||/||E_lazy(T)||. Decides whether "feature learning beats the lazy
    bound" holds on the testbed through the targeted-lift mechanism (tau small) and what the gain e^{-A} is. Run at
    n = 200 on 2026-10-12 (chain/lazy_comparison.py; numbers in THEORY 22); to redo at large n with the chain's fine stages,
    where the integral is exact, and to bound tau from the stage targets (the one derivation that would make the lazy
    comparison unconditional).
A6. Sharpening and the edge of stability (THEORY 24, Thms 54-55). (i) Along each run, per stage: the frozen-eigenvector
    change phi_{v_1}(theta(t_s)) - phi_{v_1}(theta(t_{s-1})) (one Jacobian-vector product with v_1 per stage end), its first-order
    part 2 <g_1, S(v_1) d> and the rotation gap lambda_1(t_s) - phi_{v_1}(theta(t_s)) >= 0; the sign of the first-order part along
    the run is the single hypothesis of Thm 54 (expected: negative until t ~ 0.3-1 for raw residuals, positive after).
    (ii) At the edge of stability (numerics/eos runs, GD with eta lambda_1 ~ 2): per step d_k, the period decomposition
    d_k = delta_k + (-1)^k d; the three terms 2 nabla phi . delta, 2 |S(v_1) d|^2 and 2 <g_1, T(v_1)[d, d]> (the last by a nested
    forward-over-forward-over-forward jvp along d), the rotation term 4 ||dK v_1||^2 / gap, and the measured change of
    lambda_1 over the period against the two-sided bound of Thm 55. Decides whether self-stabilisation is the third-derivative
    term (expected) and how tight the bound is for the top mode. Login node for (i) on saved snapshots; (ii) needs the
    EoS runs' parameter trajectories.
A4. Thm 36 first-order check. Same run: evaluate ||E_lazy(T)||^2 - 2 int <E_lazy(2T-u), Delta(u) E_lazy(u)> du and the
    remainder bound against ||E(T)||^2. Expected: the remainder bound is useless (D large) and the first-order term is
    sign-correct; this is a sanity check on the withdrawn all-orders claim, not a result.

## B. The roughness function (corrected 2026-10-12 after user review)

The roughness function to use at any point is the Krylov functional of the current state, phi_gamma = E_gamma/||E_gamma||
with E_gamma = gamma (K + gamma)^-1 E (Part I: Thm stepsup / stepkrylov / slow(c)); a fixed Laplacian high-pass is a local
complexity function and gives a weaker bound. The earlier B1 ("rank-to-roughness law as the bridge") is withdrawn: no
input-space notion of roughness is needed for the moving frame to be a roughness statement. What remains to measure:
B1. The dataset in the coordinates of the kernel at large n: rho(gamma) = ||Y_gamma|| (slow part of the labels under K_0)
    for the L63 targets and the field targets, by the Krylov series (no eigendecomposition), and its dependence on width
    and depth at initialisation (the quantity through which depth enters Cor. depth of the held-rough section).
B2. The one-time floor along the run: ||E_gamma(t)|| at the current state, gamma = 1/t and gamma = lambda_eff(t), against
    ||E(t)|| (the exact first-order floor; measured at n = 200 in C1 below), and the Krylov values F_M(s) at the actual
    step length (numerics/report.md found the series trivial at the GD step; redo at the chain's stage length).

## C. The floors of the held-rough section in Krylov form (C1 run on the n = 200 snapshots, 2026-10-12)

C1. chain/krylov_floor.py, on runs/sweepA (72 GD runs, T = 3000, n = 200; K(t) at 30 geometric snapshot times, E(t) at
    400 log times). Per run and time: the path length ell(t) = int |theta_dot|; the time-0 floor F_0(ell) - 1/2 H_2bar ell^2
    (Thm time0, F_0 by the ridge closed form, H_2bar = running max of the logged H_2 along the gradient, a proxy below the
    ball maximum, so the floor reported is slightly optimistic); the path floor <phi, E_0> - cost_t(phi) with phi the Krylov
    functionals of (K_0, E_0) on a gamma grid (Thm path; cost from the snapshot kernels, log-interpolated); the energy-form
    floor of the earlier draft for the same phi (Prop rough-energy (b), eps = 1); the one-time floor ||E_{1/t}(t)||/||E(t)||
    at the current state; the cost split cost_t / (ell ||phi||_{K_0}) (how much of the charge is the lift); the leak/main
    ratio of the energy form. Checks: <phi, E(t)> >= <phi, E_0> - cost_t(phi) holds to the interpolation error (viol <= 0.3 %);
    sum lam e^2 = |g_E|^2 to 4 digits.
    RESULTS (n = 200, 36 runs of sweepA: H in {32, 96, 256}, depth 1-4, Delta in {0.1, 0.5, 1.0}, T = 3000; median [10 %, 90 %] over runs;
    every floor divided by the measured ||E(t)||):
ratio of each floor to the measured ||E(t)||, median [10 %, 90 %] over runs
   t     F0(ell)/E        floor_t0/E       floor_path/E     floor_energy/E   slow_now=||E_{1/t}||/||E||  lift_cost=cost/(ell sig0)  leak/main (energy form)
    10     1.05 [   0.99,   1.38]      -11 [  -45.2, -0.288]    0.938 [  0.384,   0.97]        0 [      0,      0]    0.919 [  0.676,  0.953]     1.94 [   1.24,   3.35] 1.08e+03 [    161,2.14e+04]
    30     1.24 [   1.04,   2.55]      -52 [   -244,   -4.2]    0.709 [ -0.357,  0.956]        0 [      0,      0]    0.782 [  0.655,  0.961]      5.4 [   1.85,   65.5] 3.94e+03 [    406,4.95e+04]
   100     1.87 [    1.1,   5.43]     -267 [-1.01e+03,  -17.9]   -0.191 [  -4.68,  0.856]        0 [      0,      0]    0.737 [  0.674,  0.946]     30.8 [   4.31,    187]  5.6e+04 [1.19e+03,9.74e+28]
   300     3.96 [   1.31,   10.4] -1.15e+03 [-3.19e+03,  -75.8]    -4.43 [  -24.4,  0.405]        0 [      0,      0]    0.744 [  0.674,  0.934]     73.1 [   6.27,    386] 3.69e+12 [2.55e+04,1.69e+300]
  1000     9.53 [   1.98,   24.9] -3.38e+03 [ -2e+04,   -486]    -17.3 [   -214,  -3.08]        0 [      0,      0]    0.717 [  0.588,  0.874]      189 [   7.24,    728] 3.24e+124 [1.75e+16,5.83e+300]
  3000     16.7 [   3.41,   71.3] -1.03e+04 [-6.85e+04,-1.25e+03]    -82.7 [   -784,  -8.11]        0 [      0,      0]    0.699 [  0.595,  0.865]      277 [   8.29,    859] 5.76e+299 [1.3e+73,1.58e+301]
    * The time-0 floor F_0(ell) - 1/2 H_2bar ell^2 is halved by the H_2 term at ell = 0.61 [0.41, 1.50], t = 1.2 [0.3, 5.8] of 3000,
      and is negative from t ~ 10 on: it is a statement about the first unit of path length only. F_0(ell) alone (the frozen
      first-order floor, which is NOT a bound once the kernel moves) exceeds the error by 1.9x at t = 100, 4x at 300 and 17x at T:
      the whole content of the late error is the motion of the kernel.
    * The path floor with a functional fixed at the start (Thm path, actual kernels along the run) holds 0.94 of the error at
      t = 10 and 0.71 at t = 30, and is empty from t ~ 100 (median -0.19), because the cost along the best fixed functional is
      lifted to 5x its first-order value by t = 30, 31x by t = 100 and 280x by T (the "lift_cost" column): the lift along the
      slow part is the feature learning, and a floor charged by it loses content exactly when the lift arrives.
    * The energy-form floor of the earlier draft (Prop rough-energy (b), fixed rank-one P) is identically zero: its leak term
      is 10^3 times the main term already at t = 10 (REPORT 0 found 4-12x for the cross term itself at t = 0). The signed form
      has no such term.
    * The one-time floor at the current state, ||E_{1/t}(t)||, holds 0.92 of the error at t = 10 and 0.70-0.74 from t = 100 on
      (10-90 %: 0.59-0.95): the Krylov functional of the current state is the floor that stays informative, and what it does
      not capture (the component of the error removed faster than 1/t) is 25-30 % late.
    * Checks: sum lam e^2 = |g_E|^2 to 4 digits in every run; the path-floor inequality holds to the snapshot interpolation
      (max violation 0.000 of <phi, E_0>).
    Conclusion for the section: keep the a priori floor as an exact statement whose content is the first unit of path length;
    state the one-time Krylov floor as the informative one along the run; hand the late error to the chain's cutoff bound.
    To redo at large n: the same with the chain's fine stages (cost integral exact), H_2bar over the ball (not the path),
    and the current-state slow part at every stage; decision: the time-0 floor is a statement about the first unit of path
    length only, and the content of the late error is in the cost (the lift), which only the chain brackets.
C2. Trajectory bracket on the Krylov functional: Theorem 13 applied to v = phi_gamma(t) (the accumulated feature change
    along the current slow part vs its initial features), per stage; the Krylov-form of THEORY 20(ii).
C3. The cutoff rate gamma*(t) (Def. cutoff, rewritten) against the rate at which the measured error's spectrum in the
    current kernel basis drops; rho(gamma) against the measured error spectrum; H_2bar and rho vs depth at init.

## D. Chain verifications that the tex quotes and that are n = 200 now

D1. Theorem 13 bracket at T (D_v / l_v and lower bound / actual by rank), Theorem 11 bulk/core at init for tanh, ReLU,
    GELU, softplus, Theorem 11' along the path, the stage table of section 8 (Cross and Q per mode per stage, the sign
    change of Cross at t ~ 0.3-1), the 12.2 richness-expansion check at s = 2-16 to second order, the 15/propagator
    numbers (Phi_max in three directions), the seed spread (two seeds per design; include depth 1 as the control without
    the hidden-state channel, THEORY 6 scope note), Prop. 9b visibility numbers, Prop. 9d retention table, section 13
    validation split signs. All at n >= 2000 with a held-out set. Cost: Slurm, the WRITEUP S3 design set, GPU-hours.
D2. Lemma 15 init-feature sign along the path at large n (section 8 MEASURED table): the sign of the first term of the
    cross term after the transient, per mode; the raw-vs-centred comparison of 12.1.
D3. Euler vs direct from a shared initialisation at Delta in {0.1, 0.25, 0.5, 1} (section 11): E_D(t), E_E(t), the stage
    lifts and their targets; checks Thm 20-22 and the conditional Cor. 23.
D4. Theorem 10 ambiguity made quantitative: train pairs (theta_0, theta_0 + N) with N of the null-space form at a
    hidden-hidden interface, several norms; spread of E(T) over N vs the seed spread.

## E. Adam and preconditioned methods (THEORY 10.4 CORRECTION, section 21)

E1. The dP term. Under Adam (and under KronQ with its refreshes): per stage, v^T Psi dP Psi^T v / n against Cross_P + Q_P on
    the residual-carrying modes; sign and size along training. Outcomes: dP term small after the transient -> the
    tail-lift structure transfers; comparable -> the lift and the metric decay compete and the chain's monotonicity
    statements are not available for Adam.
E2. Cross_s sign per stage under Adam (listed in 10.6, not run): needs the Adam harness to log P.
E3. Energy identity with momentum: log E^T K_P g_bar along an Adam run; fraction of steps with the wrong sign.

## F. Optimizer line (section 21; after the scaling study)

F1. Per-layer vs global basis at equal push, isolated (the measured fact without an explanation): n = 20000, the same
    lr grid, with the residual-space overlap between layers' pushed modes logged (the cap split's cost).
F2. Remainder level: what sets the ~0.1 (exact basis) vs ~0.02 (Kronecker, smooth) optimum; log the per-step Q lift on
    the residual-carrying modes against the remainder along the (q, lr) ridge.
F3. Tuned baselines for the scaling study's claim: AdamW / Muon / SOAP with warmup, cosine schedule, weight decay, lr
    scaled with width, 5-point lr grids at widths 1024 and 4096. Required before any "beats tuned X" statement.
F4. The hypergradient as a stage quantity (21(c)1): per mode, split the one-step loss change into first-order progress
    and the Q-lift part; test whether the learned weights track the first-order part alone.

## Cost summary (to be refined before each request)

Login node: C1 was run on the n = 200 saved snapshots on 2026-10-12 (chain/krylov_floor.py, ~minutes).
Slurm, GPU: A1-A4 (one fine-staged run at n ~ 2000, hours on 1 GPU); B1-B2 + C1-C3 + D1-D2 (the design set at n >= 2000,
GD to T = 3000, tens of GPU-hours in total); D3-D4 (a few GPU-hours); E1-E3 (reuse Adam runs with P logged); F1-F4
(the optimizer harness, each a 20-40 min node job as in REPORT 58-60).
