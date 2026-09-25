# Experiment plan

Paper: `paper/learned_steppers.tex`. Predictions P1--P8 are in its Section 10.
Goal: test the predictions, and find at least one problem where a Taylor method of order 2
(or exponential Taylor order 2) has lower injected error and rollout error than the Euler step
and the direct step.

## 1. Systems

| ID | System | Role | Notes |
|----|--------|------|-------|
| L63 | Lorenz-63 | cheap sanity check | compute L_k, M_k exactly; fit capacity exponent alpha with MLPs |
| KS | 1D Kuramoto--Sivashinsky, periodic | dissipative, stiff linear part | reuse setup of arXiv:2608.16084 (L=100, 1024 modes) and a cheaper 128-mode version for sweeps |
| KF | 2D forced Kolmogorov flow, vorticity form | main candidate for an order-2 win | Re ~ 1000, 64^2 and 128^2, drag term as in Kochkov et al. 2021; pseudo-spectral |
| KdV | 1D Korteweg--de Vries (or NLS) | conservative control | purely imaginary test-mode rates; history-based order-2 should not blow up |

All systems are written as semilinear `u_t = A u + B(u)` with `A` the linear operator (diagonal in
Fourier space) and `B` the nonlinear term. The solver must expose, as differentiable functions:
`f(u) = A u + B(u)`, `g(u) = Df(u) f(u)` (via a Jacobian-vector product), `exp(hA)`,
`phi1(hA)`, `phi2(hA)` with `phi1(z) = (e^z - 1)/z`, `phi2(z) = (e^z - 1 - z)/z^2`
(use the contour-integral evaluation of Kassam and Trefethen 2005 for small |z|).
Recommended: JAX for the solver and all derivative quantities. Training can use the existing
PyTorch code if that is faster to reuse; decide with the user.

## 2. Methods (fixed IDs)

`h` is the time step, `N` is the network, `A` and `B` as in Section 1.

| ID | Step map | Slot types | Paper reference |
|----|----------|-----------|-----------------|
| `direct` | `N(u)` | none | Taylor order 0 |
| `euler` | `u + h N(u)` | none | Taylor order 1 |
| `rk4` | classical RK4 with N as vector field | none | Runge--Kutta wrapper, Eq. (rk) |
| `ieuler` | `u' = u + h N(u')`, fixed-point solve | none | implicit Euler wrapper |
| `t2_exact` | `u + h f(u) + (h^2/2) N(u)` | exact slot | Taylor order 2, exact lower slot |
| `t2_net` | `u + h F1(u) + (h^2/2) N(u)`, F1 = frozen network from `euler` | state-based slot | Rule 2 test |
| `t2_hist` | `u + (u - u_prev) + (h^2/2) N(u)`; training uses true `u_prev`, rollout uses predicted | history-based slot | parasitic root e^{-z} |
| `exp_direct` | `exp(hA) u + N(u)` | none | exponential known part |
| `exp_euler` | `exp(hA) u + h phi1(hA) N(u)` | none | exponential known part |
| `exp_t2` | `exp(hA) u + h phi1(hA) B(u) + h^2 phi2(hA) N(u)` | exact slot | exponential Taylor order 2 |
| `rkn_exact` (optional) | RKN4 for `x'' = N(x)` with `v_j = f(x_j)` | exact slot | Section 11.5 (a) |

Implement methods as (known part, slot weight, network) triples where possible, so that the
hardness of every method can be computed by the same code (Section 4).

## 3. Sweeps

- Time steps: 6--8 values per system, chosen so that `h * |lambda|` on the energetic modes spans
  roughly 0.01 to 2. Report the chosen values and the resulting `z_k = h lambda_k` ranges.
- Network sizes W: 3--4 values per architecture (same architecture across methods).
- Architectures: FNO (primary), MLP or U-Net (secondary).
- Seeds: 3.
- Noise: nu in {0, two nonzero levels} on the observed next state, for Corollary 2 (noise window).
- Estimate the total run count and GPU-hours per system before launching; show the user.

## 4. Measurements that need no training

1. Measured hardness `Lip(Phi_h - Q_h^ex)` for every affine method and every h, from data.
   Two estimators: (a) finite-pair ratios over nearest-neighbor pairs on the attractor, reported
   as the max and the 99th percentile; (b) spectral norm of the Jacobian `D(Phi_h - Q_h^ex)(u)`
   by power iteration with JVP/VJP through the differentiable solver, sampled over the attractor.
2. `M_k`, `L_k` for k = 1, 2, 3 (estimate `L_k` with estimator (b) applied to `x^(k)`).
3. Test-mode rates `lambda_k` of the linear operator, and per-mode hardness from Table 3 of the
   paper at each h.

## 5. Metrics per trained run

- Injected error on held-out true states: sup norm and RMS; per-wavenumber spectrum.
- Rollout error vs. time; correlation time (Pearson < 0.8, as in the eigenanalysis paper).
- `|lambda_max|` and sigma = ln|lambda_max| / h of the step-map Jacobian (reuse the diagnostic from
  arXiv:2608.16084).
- For `t2_hist`: measured growth rate of the component `u_j - u_{j-1} - h f(u_j)` (parasitic mode).
- Long rollouts: energy spectrum and one-point PDF vs. reference.

## 6. Figures (targets for the paper)

1. Per-mode hardness curves on the real and imaginary axes (analytic), crossovers marked
   (Proposition 4), with `z_k = h lambda_k` overlaid for KS and KF at the chosen h.
2. Measured hardness vs. h (Section 4.1), log-log, expected slopes 0 / 1 / 2 for
   `direct` / `euler` / `t2_exact` on resolved modes; include `exp_*` methods.
3. Injected error vs. h for trained methods at fixed W; expected slopes 0, alpha, 2 alpha (P1);
   mark the window where order 2 wins.
4. Injected error vs. measured hardness, all methods and h, one panel per W: tests whether the
   capacity model collapses the data onto one curve per W (Assumption 3, P8); fit alpha and rho(W).
5. Per-wavenumber injected error vs. per-mode hardness prediction (P3, P4, P5).
6. Rollout error decomposed into injection x amplification (Eq. (compare) of the paper), using
   `|lambda_max|` from the eigen diagnostic.
7. Controls: `t2_hist` parasitic growth on KS vs. KdV (P6); `t2_net` vs. `t2_exact` (Rule 2, P7);
   `rk4` representability floor at large h on KS (P5).

## 7. Validation before experiments

- Solver: convergence in dt and resolution; KS and KF statistics against published values.
- `g = Df f`: compare with finite differences in time of true trajectories.
- `phi1`, `phi2`: compare with series at small z and closed form at large z.
- Hardness estimators: check on a linear system where `Lip(Phi_h - Q_h^ex) = max_k |e^{z_k} - q(z_k)|`
  is known exactly.
- Each method with the network replaced by its exact target (where computable) should reproduce
  `Phi_h` to solver precision.

## 8. Milestones

1. Solver + derivative quantities + validation (Section 7) for L63 and KS.
2. L63: measured hardness and a small W sweep to check the capacity model.
3. KS: Figure 2 (no training), then the method x h x W sweep, Figures 3--5.
4. KF: same pipeline; this is the main candidate for the order-2 win.
5. KdV control and Figure 7.
6. Figure 6 and final plots.
Report back to the user at the end of each milestone with numbers and plots before continuing.
