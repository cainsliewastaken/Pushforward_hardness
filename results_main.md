# multistep_roughness: results

Definitions, models, losses and departures from the original brief: `README.md`.
Figures: `results/figs/` (symlink to `$SCRATCH/multistep_roughness/figs`). The full generated tables are
appended below (sections "Tables" and "Follow-up tables"); numbers quoted here are taken from them.
Ranges are min-max over the 3 seeds; "better" / "worse" means the ranges are disjoint.

## What was run

| Stage | What | Allocation (m5326, interactive GPU node, 4 A100) |
|---|---|---|
| Sweep | 2 classes x 4 dt x 3 seeds; one-step, L_p with T_train 0.32 / 0.64, each from scratch and fine-tuned (20 runs x 6 models) | 59349089, 3 h 08 min (+ pilot 59348699, 10 min; failed launch 59348389, 0.5 min) |
| Follow-up 1 | fixed step count p = 16, 32 at every dt (4 fine-tuned runs); 40-point quadrature recompute; long-horizon evaluation | 59358259, 45 min |
| Follow-up 2 | the three long-window fine-tunes refit at peak lr 1e-6 | 59389991, 23 min |
| Follow-up 3 | curriculum 0.32 -> 0.64 -> 1.28 and damped loss (gamma = 9) at dt 0.04, 0.08 | 59393779, 25 min |
| Follow-up 4 | trainability check (curvature of each unrolled run's own loss) | 59396927, 7 min |
| Follow-up 5 | checks (v) and (i) of the rollout-kernel draft; width test (window chain at widths 64-512, dt = 0.08) | 59398009, 35 min |

Total about 5.6 GPU-node hours. Code: git history of `~/multistep_roughness` (sweep at `2e57626`; training code unchanged from `7b7aa80` for the sweep runs).

## Identity checks

| Check | Max relative error |
|---|---|
| State Jacobian (forward mode) vs autograd | 1.3e-15 |
| Psi^(p) g vs central FD of the rollout, random parameters | 1.1e-8 |
| Psi^(p) g vs FD, trained models, FD step h = 1e-5 | 4.3e-5; a step scan on a trained Euler model at T = 1.28 gives 2.7e-5 / 2.7e-7 / 1.5e-9 / 2.6e-8 / 2.4e-7 for h = 1e-3 ... 1e-7 (truncation, then roundoff): FD error, not an identity failure |
| g^(Sigma)(p) vs FD of L_p (directional) | 1.9e-6 (random parameters), 8.2e-5 (trained, h = 1e-5; not step-scanned) |
| e(t_p) = sum_i P(t_p,t_i)(eps_i + q_i), random parameters, 10-point Gauss-Legendre | 1.5e-11 |
| Same, final checkpoints of every run, 40-point GL | 1.1e-7 |
| Same, intermediate checkpoints, 10-point GL | up to 4.2e-2 (52% above 1e-8) |

The reconstruction is exact; its error is quadrature in Bbar. The brief's 5-point rule assumed small errors; where |e| is a sizable fraction of the attractor the integrand varies over the segment. Worst case: 5.8e-2 / 2.6e-3 / 1.3e-6 with 10 / 20 / 40 points. Final-checkpoint values of the Bbar quantities (rhobar, q) use 40 points; they agree with the 10-point values to three digits, so plots 4-5 were not affected.

## Summary

Unrolled training helps when the training window is short in physical time, at every dt. Fine-tuning a converged one-step model on L_p over a window of 0.16-0.32 time units (0.15-0.3 Lyapunov times) lowers the rollout error at all horizons for Euler models at all four dt, including dt = 0.08 with only 4 unrolled steps (T = 0.32: 0.0020-0.0022 vs 0.0026-0.0030 one-step; T = 5.12: 2.2-2.5 vs 3.2-4.1). A window of 0.64 can help further, less reliably across seeds: the best Euler models are the 0.64-window fine-tune at dt = 0.02 (T = 1.28: 0.069-0.094, against 0.32-0.37 one-step) and, at dt = 0.08, the 0.64 window approached in stages at a lower learning rate (0.063-0.090, against 0.12-0.23 one-step and 0.10-0.13 for the 0.32 window). Windows of 1.28 and longer could not be trained to any benefit: at lr 1e-4 the models end with a training loss up to 16 times their starting value and are worse at every horizon; at lr 1e-6, after a curriculum, or with a damped loss they are at best equal to the one-step model. The limit is the physical window, not the number of steps or dt: at the same window, p = 16 and p = 32 fail identically, and the curvature of the training loss is the same at every dt for Euler models (about 2.6e3, 1e5, 4e7, 6.5e9 for windows 0.32, 0.64, 1.28, 2.56). The impression from the first sweep that "unrolling only helps at small dt" came from judging at T = 1.28 only, where the seed spread is largest.

Of the brief's predictions, three held and one could not be tested. (1) R(T = 1.28) is flat in dt for Euler models trained on L_p (0.12-0.14 for T_train = 0.32 from scratch), and Delta t n.grad H n grows roughly in proportion to dt (n.grad H n = -1.4 to -2.2 per unit time); the regime where Delta t n.grad H n is order one is not reached at these step sizes (at most 0.27), so "R rises toward zero" was not tested. (3) c_Sigma decays in physical time at every dt (to below about 0.3 by T = 0.5), roughly collapsing, with a large seed spread. (7) The rollout kernel collapses in physical time for Euler (lambda_max(K^(p)) about 470 at T = 0.08 and 2.2e7-2.5e7 at T = 1.28 for all dt) and scales as dt^-2 for direct models, as the rollout-kernel draft predicts. Plot 4 shows no monotone relation between sum ln|rhobar| and R across models. Euler beats direct in rollouts at T = 1.28 with disjoint ranges in all 5 protocols at dt = 0.01, 4 of 5 at dt = 0.02 and 3 of 5 at dt = 0.04 (overlapping at dt = 0.08), with a larger gap after unrolled training (dt = 0.01, |e(1.28)|: one-step 0.31-0.44 vs 0.76-1.43; T_train 0.32 from scratch 0.09-0.17 vs 1.39-1.46).

The draft's one-step theorem (check v) shows that long windows do not lack good steps: on a batch, the first-order optimal step of a given length removes 70-90% of the error at window 1.28 (and is realized to three digits at small lengths), while Adam's first step and a gradient-descent step of the same length increase the error by factors of 1.4-700, because the gradient is dominated by the one sharp direction of the nearly rank-one kernel. The failures of long-window training are a failure of gradient-following step rules, which points to Gauss-Newton/Levenberg-Marquardt steps (not yet tested in training). The draft's rate bound (check i) holds in every case but is informative only for windows up to about 0.16. A width test (64-512) did not show the best window growing with width (0.64 at widths 64-128, 0.32 at 256-512), but the 0.64 stage failed to descend at the larger widths at the common learning rate, so the question remains open.

## Main sweep, by plot (final checkpoints, held-out starts)

**Plot 1, R(T = 1.28) vs dt** (`fig1_R_vs_dt.pdf`). The injected part I is small (0.005-0.03 for Euler), so R is about 95% of |e(1.28)|. Euler, unrolled from scratch with T_train 0.32: 0.131 / 0.122 / 0.143 / 0.136 for dt = 0.01 / 0.02 / 0.04 / 0.08 (flat). Euler one-step: 0.35 / 0.34 / 0.15 / 0.16 (larger at small dt). Direct one-step: 0.98 / 0.61 / 0.23 / 0.16. Direct from scratch with T_train 0.64 is the worst setting at every dt (I up to 1.0 at dt = 0.01).

**Plot 2, contraction along the error** (`fig2_contraction_vs_dt.pdf`). Euler, T_train 0.32 from scratch: mean rho - 1 = Delta t n.grad H n = -0.014 / -0.031 / -0.069 / -0.175, i.e. -1.4 / -1.5 / -1.7 / -2.2 per unit time. Euler one-step: +0.0015 / -0.0005 / -0.021 / -0.111 (not proportional to dt; slightly expanding at dt = 0.01). For Euler models the learned value matches the truth map's rho^f - 1 along the same direction in every protocol (e.g. -0.0144 vs -0.0158), and rhobar matches rho to three digits: unrolled training changes where the error points, not the learned Jacobian along it. Direct models trained by unrolling at small dt contract more than the truth (dt = 0.01, T_train 0.32 from scratch: -0.049 vs -0.020).

**Plot 3, c_Sigma(p) vs T** (`fig3_cSigma_vs_T.pdf`, per-horizon cosines `fig3b`). Starts at 1, falls below about 0.3 by T = 0.5 and fluctuates around 0-0.3 afterwards, for every dt and protocol. Euler one-step at T = 0.32: 0.44 / 0.34 / 0.39 / 0.62; at T = 0.64: 0.03 / -0.15 / 0.18 / 0.27.

**Plot 4, sum ln|rhobar| vs R(T)** (`fig4_sumlnrho_vs_R.pdf`). No monotone relation across models and horizons. sum ln|rhobar| decreases about linearly with T (more steeply for unrolled-trained models), while R is set mostly by the injected errors and their direction.

**Plot 5, nonlinear part** (`fig5_nonlinear_ratio_vs_T.pdf`). |sum P q| / |e| < 0.01 up to T = 0.64 for every Euler model (direct up to 0.26) and 0.04-0.72 at T = 1.28 (direct up to 1.9 at dt = 0.01). The first-order (linearized) picture holds for windows up to about 0.6 and fails by 1.28.

**Plot 6, R_e(H)** (`fig6_Re_H.pdf`). R_e(H) equals max (1 - rho)/dt to 2-3 digits in every model: the Hessian contribution is negligible in the maximum. Euler: 14-19 at every dt and protocol. Direct: 75 / 38 / 22 / 18, about 1/dt. One-step and unrolled models at the same dt agree. R_e(H - f_dt), the variation of the one-step error map along the error: Euler 5-11 at every dt; direct 64 / 29 / 14 / 6.
M6 (spectral norm of M; upper bound on sigma_e): Euler 6.7e3 / 4.7e3 / 3.2e3 / 2.0e3, direct 7.1e4 / 2.5e4 / 8.5e3 / 3.1e3; the largest row norm (lower bound): Euler 115-160, direct 870 / 380 / 185 / 110. The spectral norm is not normalized by the number of rows, which grows as 1/dt, so its dt trend for Euler is mostly that count.

**Plot 7, rollout kernel** (`fig7_Klam_vs_T.pdf`). Euler: lambda_max(K^(p)) is the same curve in T at every dt in every protocol, about 470 at T = 0.08, 1.5e3 at 0.16, 1e4 at 0.32, 1e6 at 0.64, 2.2e7 at 1.28. Direct at T = 0.08: 9.2e4 / 2.2e4 / 4.6e3 / 1.1e3, a factor 4 per halving of dt. tr K / lambda_max falls from 4-7 at T = 0.08 to about 1.1-2 at T >= 0.64 (Euler about 1.5): the kernel is nearly rank one at long horizons. |P^T ehat| grows to 1e3-2e3 at T = 1.28.

**Width check** (held-out one-step loss, one-step training). dt = 0.01, Euler: 2.0e-8 / 1.8e-8 / 1.3e-8 for widths 128 / 256 / 512; direct 6.7e-7 / 6.3e-7 / 3.0e-7. dt = 0.08, Euler: 1.0e-6 / 5.9e-7 / 3.4e-7; direct 2.0e-6 / 1.05e-6 / 4.5e-7. Euler stays better (disjoint) at every width, but width 256 is not converged: absolute errors here are conditional on the network size.

## Follow-up experiments

**Fixed step count** (p = 16 and 32 at every dt, fine-tuned, lr 1e-4). At p = 32, dt = 0.01 (window 0.32) improves on one-step at every horizon while dt = 0.08 (window 2.56) is 65 times worse at T = 0.32 (0.17-0.20 vs 0.0026-0.0030). Equal windows with different step counts fail equally (window 1.28: p = 32 at dt = 0.04 and p = 16 at dt = 0.08 give T = 0.32 errors 0.080 and 0.078). The step count is not the controlling variable; the physical window is.

**Why long windows fail: their own loss is not optimized.** The held-out training loss L_p of each fine-tune, before and after: windows 0.32-0.64 end below their start (end/start 0.2-0.9), windows 1.28-2.56 end up to 16 times above it at lr 1e-4 (8 of 9 Euler runs above 1.8, one at 1.0), after transient increases of 40-4600 times. These are optimization failures, not a trade of short-horizon for long-horizon accuracy: the same models are also worse at T = 2.56 and 5.12 (e.g. Euler dt = 0.08, window 1.28: 2.7-3.3 at T = 2.56 vs 0.59-0.98 one-step).

**Learning rate 1e-6** (the three long-window fine-tunes). Removes the blow-ups (largest increase 1-13 times) but 6 of 9 Euler runs still end above their starting loss, and the models are worse than one-step at short horizons (Euler dt = 0.08, window 1.28: T = 0.32 0.0035-0.0043) and not better at long ones.

**Curriculum** (from the 0.32-window model: window 0.64 at lr 1e-5, then 1.28 at lr 1e-6). Stage 1 descends in 5 of 6 Euler runs and gives the best Euler models of the study (dt = 0.08 at T = 1.28: 0.063-0.090; dt = 0.04: 0.062-0.15, overlapping the 0.32-window model). Stage 2 ends above its start in 5 of 6 Euler runs and is worse than stage 1 at every horizon.

**Damped loss** (w_q = exp(-9 t_q), window 1.28, lr 1e-5; gamma = 9 equalizes the contribution of each horizon given the measured error growth). Lowers the starting curvature of the loss from about 4e7 to 3e4 and avoids the damage, but the models equal the one-step model at every horizon (Euler dt = 0.08: T = 0.32 0.0028-0.0032, T = 5.12 3.5-3.9). Damping makes the long window trainable by turning it back into a short-window objective.

**Long horizons** (`fig8_long_horizon.pdf`, T up to 5.12, about 4.6 Lyapunov times). Errors at T = 5.12 are 2-5 (Euler) and 3-16 (direct) against a saturated-error scale of 21, so all comparisons remain meaningful. Ordering at T = 5.12 for Euler, dt = 0.08: 0.32 window 2.2-2.5, curriculum 0.64 2.2-3.4, one-step 3.2-4.1, long windows 3.0-10.

**Trainability check** (`fig9_trainability.pdf`). The largest Gauss-Newton eigenvalue of each fine-tune's own loss at the start: for Euler, about 570 / 2.6e3 / 1e5 / 4e7 / 6.5e9 for windows 0.16 / 0.32 / 0.64 / 1.28 / 2.56, the same within 10% at every dt. For direct it is 10-1000 times larger at the same window and grows as dt^-2 (window 0.32: 1.5e4 / 6e4 / 2.7e5 / 1.0e6 for dt = 0.08 ... 0.01). Whether a fine-tune descended is predicted only moderately by any single number (126 fine-tuned models, 55% descended): lambda_max alone classifies 83% correctly (threshold about 1e6), lr x lambda_max 79%, the window 79%. The gradient-descent stability limit lr x lambda_max < 2 does not apply to Adam: many runs with lr x lambda_max of 10-300 descended. The size of the transient increases does not correlate with lr x lambda_max (correlation 0.18).

## Follow-up 5: checks (v) and (i) of the draft, and network width

**Check (v): one training step on the rollout loss** (`theory_checks.py --part v`; one-step and 0.32-window models, width 256, dt = 0.04 and 0.08, seed 0, both classes; windows 0.32 / 0.64 / 1.28; 128 held-out starts; the weighted rollout kernel formed explicitly, N up to 12288). For each step length s = lr sqrt(P) of Adam's first step at lr = 1e-6, 1e-5, 1e-4: the first-order optimum F(s) of the draft's one-step theorem, the realized error after the first-order optimal step of that length, after Adam's first step, and after a gradient-descent step of that length (all relative to the starting error).

| One-step model, Euler, dt = 0.08 | F(s) | optimal step, realized | Adam | gradient descent |
|---|---|---|---|---|
| window 0.32, lr 1e-6 | 0.85 | 0.85 | 1.65 | 9.2 |
| window 0.64, lr 1e-6 | 0.40 | 0.40 | 2.0 | 12 |
| window 1.28, lr 1e-6 | 0.16 | 0.16 | 4.0 | 26 |
| window 1.28, lr 1e-5 | 0.11 | 0.11 | 30 | 91 |
| window 1.28, lr 1e-4 | 0.06 | 0.18 | 118 | 316 |

Across all 24 cases: (1) the realized error after the first-order optimal step equals F(s) to about three digits at the two smaller step lengths and departs from it at the largest (nonlinearity); (2) F(s) is smaller for longer windows (at lr 1e-6: 0.83-0.89 at window 0.32, 0.25-0.58 at 0.64, 0.10-0.31 at 1.28), so on the batch a good step exists and removes more of the error the longer the window; (3) Adam's first step and the gradient-descent step of the same length increase the error, by 1.4-14 times at lr 1e-6 and 13-700 times at lr 1e-4, more for longer windows and for direct models. The gradient direction is dominated by the top eigendirection of the kernel (tr K / lambda_max = 1.1-1.3 at window 1.28); the optimal step, A^T (M + gamma)^-1 b, is a Gauss-Newton / Levenberg-Marquardt step that is small along it. Caveat: the optimal step was computed and evaluated on the same 128 starts (133k parameters, at most 12288 constraints); whether such steps generalize to other starts was not tested.

**Check (i): the rate bound for Euler** (`--part i`; 36 Euler models: one-step, 0.32- and 0.64-window fine-tuned, all dt, 3 seeds; 128 held-out starts). No violation: measured lambda_max(K^(q)) <= (dt sum_i e^{ell (q-i-1) dt} s_i)^2 <= e^{2|ell| dt} phi(ell, tau)^2 max_i s_i^2 in every case, with equality at q = 1 (tau = 0.08 at dt = 0.08), as it must be. The middle bound exceeds the measurement by 1-4 at tau = 0.08, 20-80 at 0.16, 6e2-1e7 at 0.32, 1e5-2e17 at 0.64 and 3e12-2e39 at 1.28. ell, the largest one-step expansion of grad H over the visited states, is 16-42 (local expansion of the Lorenz flow), while lambda_max(K^(q)) grows only about like e^{9 tau}: the log-norm bound is informative only for tau up to about 0.16.

**Width test** (`followup5.sh`; dt = 0.08, widths 64 / 128 / 256 / 512 with 3 hidden layers, both classes, 3 seeds; from each width's one-step model a window chain 0.16 (lr 1e-4) -> 0.32 (lr 1e-4) -> 0.64 (lr 1e-5) -> 1.28 (lr 1e-6), 20k steps per stage, learning rates as tuned at width 256). Euler, |e(T)| at T = 0.32 and 0.64; "up" marks a stage whose own loss ended above its start in at least 2 of 3 seeds.

| width | one-step | 0.16 | 0.32 | 0.64 | 1.28 |
|---|---|---|---|---|---|
| 64 | 0.0052-0.0054 / 0.018-0.025 | 0.0041-0.0043 / 0.016-0.021 | 0.0034-0.0036 / 0.011-0.015 | 0.0035-0.0039 / 0.0085-0.012 | 0.0040-0.0052 / 0.011-0.026 |
| 128 | 0.0040-0.0043 / 0.013-0.021 | 0.0031-0.0034 / 0.011-0.017 | 0.0026-0.0028 / 0.0083-0.012 | 0.0026-0.0029 / 0.0049-0.0075 | up |
| 256 | 0.0026-0.0030 / 0.012-0.013 | 0.0022-0.0024 / 0.0094-0.012 | 0.0018-0.0020 / 0.0071-0.0092 | up | up |
| 512 | 0.0017-0.0020 / 0.0071-0.011 | 0.0014-0.0016 / 0.0043-0.0062 | 0.0012-0.0014 / 0.0034-0.0049 | up | 0.0021-0.0028 / 0.0072-0.017 |

The best window does not grow with width in this run: it is 0.64 at widths 64-128 and 0.32 at widths 256-512. But the 0.64 stage descended in every seed at widths 64 and 128 and went up in 2 of 3 seeds at 256 and 512, with the same learning rate at every width; the 1.28 stage went up in 2 of 3 seeds at widths 128 and 256 and in 1 of 3 at widths 64 and 512, and is never the best window. The comparison is therefore confounded by optimization: if the curvature of the rollout loss grows with width (measured only at width 256), a fixed learning rate fails at long windows first for the larger networks, which would hide or reverse the effect being tested. Not supported, not refuted. Width itself lowers the error at every window (one-step T = 0.32: 0.0053 at width 64 to 0.0018 at 512), and the best window improves on one-step by a similar fraction at every width (about 30% at T = 0.32).

## Relation to the roughness and multistep theory

The rollout-kernel draft (`multistep_rollout_kernel.tex`) lists five numerical checks; checks (i) and (v) are in Follow-up 5 above, (ii) and parts of (iii) and (iv) below.

1. **Continuous-time limit, Euler (check ii).** The draft proves that the Euler rollout features are a forward-Euler discretization of the learned sensitivity equation, so K^(p) depends on the horizon tau = p dt, not on p, and that nothing forces convergence for direct models. Measured: lambda_max(K^(p)) of Euler models is the same curve in T for dt = 0.01 ... 0.08 in every protocol (plot 7), and so is the curvature of the training loss (trainability check). For direct models both scale as dt^-2, the "features carry no factor dt" effect of the draft's direct-step section: at fixed tau the bound grows with q = tau/dt, and (tau/dt)^2 is what is measured.
2. **Rate of rollout training grows with the horizon (Theorem "rate" and the remark on safe learning rates).** The draft bounds lambda_max by phi(l, tau)^2 max s_i^2, exponential in tau for expanding learned dynamics, and notes that the safe gradient-descent learning rate then shrinks exponentially in the horizon. Measured: the curvature grows by about 40 from window 0.32 to 0.64, 400 from 0.64 to 1.28 and 160 from 1.28 to 2.56, and training fails from window 1.28 on. Not confirmed: the gradient-descent threshold itself; training used Adam, whose per-parameter step does not follow lr x lambda_max (runs with lr x lambda_max up to 300 descended). The scaling of the difficulty with tau is confirmed; a quantitative stability threshold for Adam is not.
3. **Long-horizon remark (check iv, partly).** The draft conjectures lambda_max(K^(tau)) = exp(2 lambda_1 tau + o(tau)) with the kernel becoming nearly rank one along the leading Lyapunov direction. Measured: tr K / lambda_max falls to about 1.5 by T = 0.64, and the growth rate of the curvature per unit window falls from about 11 (0.32-0.64) to about 4 (1.28-2.56), toward 2 lambda_1 = 1.8 of Lorenz-63, consistent with finite-time exponents converging to the asymptotic one. The alignment of the top eigenvector with the Lyapunov direction was not measured.
4. **Short horizons (Proposition "short", check iii, indirectly).** For h tau << 1 the draft gives K^(q) = tau_q^2 (Kbar^(q) + E). Measured: lambda_max grows from 470 to 1.5e3 between T = 0.08 and 0.16 (a factor 3.2 for tau^2 = 4) and the training-loss curvature from 570 to 2.6e3 between windows 0.16 and 0.32 (4.5): close to tau^2 at the shortest horizons, with exponential growth taking over by 0.32-0.64. Kbar^(q) itself was not formed.
5. **Error identity and the propagated injections (eigenanalysis expansion).** e(t_p) = sum_i P(t_p,t_i)(eps_i + q_i) holds exactly (1e-11 with converged quadrature). The nonlinear part is below 1% of e up to T = 0.64 for Euler and 4-70% at T = 1.28 (plot 5): the linear theory, and the kernel picture built on it, describe windows up to about half a Lyapunov time, which is also where unrolled training works.
6. **What unrolling changes.** The gradient of the q-th term is a sum of one-step feature gradients weighted by the propagator transposed, so unrolled training penalizes one-step errors along directions the flow amplifies. Measured: for Euler models the learned Jacobian along the error matches the truth map's in every protocol (rho - 1 = rho^f - 1 to about 10% wherever |rho - 1| > 0.01; direct models trained by unrolling at small dt contract more than the truth), while the error direction moves into contracting directions (Euler dt = 0.01: mean rho - 1 from +0.0015 one-step to -0.014 ... -0.028 unrolled). This is the mechanism by which the short-window losses improve long-horizon error.
7. **Roughness of the error map and Euler vs direct.** With J = D Phi_h + D E, the variation of the error map along the error, R_e(H - f_dt), measures how far the learned Jacobian is from the truth's in the direction that matters. Euler: 5-11 at every dt; direct: 64 / 29 / 14 / 6, about 1/dt. Per step in state units the direct model's Jacobian error at dt = 0.01 is about 7 times the Euler model's; its rollout error at T = 1.28 is about 3 times larger after one-step training and 10 times larger after unrolled training, converging by dt = 0.08. Same direction and dt trend as the one-step Euler-vs-direct result. In addition, the direct model's rollout loss is 10-1000 times sharper than the Euler model's at the same window (item 1), so at a common learning rate it is also harder to train by unrolling.
8. **Roughness and the near-rank-one kernel.** With a step size limited by lambda_max, an error component along an eigendirection with eigenvalue lambda_k decreases at a rate of order lambda_k / lambda_max. At long windows almost the whole spectrum is in one eigenvalue (tr K / lambda_max about 1.5), so every component off that direction is slow; this is the roughness-theory reading of the lr 1e-6 runs (no blow-up, no progress). Check (v) (Follow-up 5) sharpens this: for a step rule that follows the gradient (Adam, gradient descent) the top direction dominates the step and the step overshoots, while the first-order optimal step of the same length, which is small along that direction, removes most of the error on the batch. The slow progress of the lr 1e-6 runs is therefore a property of the step rule, not of the loss alone.

Not explained by the theory: the best window (about 0.3-0.6 Lyapunov times, found empirically), why one-step models at small dt leave their error in neutral directions, and the transient increases of 20-1000 times at the start of every fine-tune (Adam with a fresh optimizer state on a sharp loss; outside the gradient-flow analysis).

## Limitations

- 3 seeds; several comparisons rely on ranges that are disjoint by a small margin.
- One width (256, not converged), one optimizer family (Adam), one learning rate per stage in the follow-ups.
- The curriculum and damping were tested at dt = 0.04 and 0.08 only.
- Lorenz-63 is 3-dimensional; the training windows (up to 0.6 Lyapunov times usefully) and evaluation horizons (up to 4.6) are short compared with long-run statistics, which were not measured.
- M6 and M7 were measured at final checkpoints only; intermediate checkpoints have M1-M5 on 1024 starts with 10-point quadrature.
