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
| Follow-up 6 | curvature of the width chains; short width chains (4000 steps per stage) with fixed and curvature-scaled learning rates | 59400741, 20 min |

Total about 5.9 GPU-node hours. Code: git history of `~/multistep_roughness` (sweep at `2e57626`; training code unchanged from `7b7aa80` for the sweep runs).

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

Unrolled training helps when the training window is short in physical time, at every dt. Fine-tuning a converged one-step model on L_p over a window of 0.16-0.32 time units (0.15-0.3 Lyapunov times) lowers the rollout error at all horizons for Euler models at all four dt, including dt = 0.08 with only 4 unrolled steps (T = 0.32: 0.0020-0.0022 vs 0.0026-0.0030 one-step; T = 5.12: 2.2-2.5 vs 3.2-4.1). A window of 0.64 can help further, less reliably across seeds: the best Euler models are the 0.64-window fine-tune at dt = 0.02 (T = 1.28: 0.069-0.094, against 0.32-0.37 one-step) and, at dt = 0.08, the 0.64 window approached in stages at a lower learning rate (0.063-0.090, against 0.12-0.23 one-step and 0.10-0.13 for the 0.32 window). With a fixed peak learning rate, windows of 1.28 and longer could not be trained to any benefit (lr 1e-4: training loss up to 16 times its starting value; lr 1e-6, curriculum or damped loss: at best equal to one-step); with the peak learning rate scaled by the measured curvature of the loss (follow-up 6), the 1.28 window trains at every width and gives the best models at T >= 0.64 from width 128 on (dt = 0.08, width 512, T = 1.28: 0.033-0.037 against 0.054-0.057 for the 0.32 window and 0.069-0.11 one-step). The difficulty is set by the physical window, not the number of steps or dt: at the same window, p = 16 and p = 32 fail identically, and the curvature of the training loss is the same at every dt for Euler models (about 2.6e3, 1e5, 4e7, 6.5e9 for windows 0.32, 0.64, 1.28, 2.56). The impression from the first sweep that "unrolling only helps at small dt" came from judging at T = 1.28 only, where the seed spread is largest.

Of the brief's predictions, three held and one could not be tested. (1) R(T = 1.28) is flat in dt for Euler models trained on L_p (0.12-0.14 for T_train = 0.32 from scratch), and Delta t n.grad H n grows roughly in proportion to dt (n.grad H n = -1.4 to -2.2 per unit time); the regime where Delta t n.grad H n is order one is not reached at these step sizes (at most 0.27), so "R rises toward zero" was not tested. (3) c_Sigma decays in physical time at every dt (to below about 0.3 by T = 0.5), roughly collapsing, with a large seed spread. (7) The rollout kernel collapses in physical time for Euler (lambda_max(K^(p)) about 470 at T = 0.08 and 2.2e7-2.5e7 at T = 1.28 for all dt) and scales as dt^-2 for direct models, as the rollout-kernel draft predicts. Plot 4 shows no monotone relation between sum ln|rhobar| and R across models. Euler beats direct in rollouts at T = 1.28 with disjoint ranges in all 5 protocols at dt = 0.01, 4 of 5 at dt = 0.02 and 3 of 5 at dt = 0.04 (overlapping at dt = 0.08), with a larger gap after unrolled training (dt = 0.01, |e(1.28)|: one-step 0.31-0.44 vs 0.76-1.43; T_train 0.32 from scratch 0.09-0.17 vs 1.39-1.46).

The draft's one-step theorem (check v) shows that long windows do not lack good steps: on a batch, the first-order optimal step of a given length removes 70-90% of the error at window 1.28 (and is realized to three digits at small lengths), while Adam's first step and a gradient-descent step of the same length increase the error by factors of 1.4-700, because the gradient is dominated by the one sharp direction of the nearly rank-one kernel. The failures of long-window training are a failure of gradient-following step rules at a fixed learning rate; scaling Adam's learning rate with the curvature (follow-up 6, lr down to about 1e-8 at window 1.28) is enough to make them train. Gauss-Newton/Levenberg-Marquardt steps were not tested in training. The draft's rate bound (check i) holds in every case but is informative only for windows up to about 0.16. With a fixed learning rate a width test (64-512) did not show the best window growing with width; the curvature turned out to be nearly width-independent and the failures seed-dependent. With curvature-scaled rates the gain of the 1.28 window over the 0.32 window is within the seed spread at width 64 and clear from width 128 on: weak support for longer windows paying off more for larger networks.

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

## Follow-up 6: learning rate scaled by the curvature

**Curvature vs width** (`trainability.py` on the 20k-step width chains, dt = 0.08). The largest Gauss-Newton eigenvalue of each stage's own loss at its start barely depends on width: Euler, window 0.64: 1.3e5 / 1.1e5 / 9.6e4 / 1.3e5 for widths 64 / 128 / 256 / 512; window 1.28: 3.0e7 / 3.2e7 / 4.2e7 / 7.0e7 (direct: 7.5e5-1.2e6 and 3.7e7-8.1e7). At most a factor 2.5 over an 8-fold width range, so curvature growing with width does not explain the earlier failures at large width.

**Short chains** (`followup6.sh`; 4000 steps per stage, warmup 200, own held-out loss logged every 100 steps; widths 64-512, dt = 0.08, both classes, 3 seeds). Windows 0.16 and 0.32 at lr 1e-4, the 0.32 stage recording each model's curvature as the anchor; then 0.64 and 1.28 in three branches: fixed lr (1e-5, 1e-6, the earlier protocol), lr = 1e-4 (lambda_anchor / lambda)^(1/2) ("sqrt") and lr = 1e-4 (lambda_anchor / lambda) ("inv"), per model.
- The fixed branch reproduces the earlier failures only in part: its 0.64 stage went up in 1 of 3 Euler seeds at widths 64, 256 and 512 and in none at 128; its 1.28 stage went up in at least 2 of 3 seeds at widths 128 and 512. Single-seed failures at a fixed learning rate are frequent and do not follow width; the earlier "fails at large width" pattern was not a width effect.
- With either curvature-scaled rule the 1.28 window trains at every width. Euler, own loss end/start at window 1.28: sqrt 0.27-0.84 (12 of 12 seeds descended), inv 0.40-1.17 (11 of 12). The inv rule gives peak lr of 5e-9 to 1e-8 at window 1.28 and still lowers the loss with no transient at all (largest increase 1.0-1.5 times): consistent with check (v), where the parameter distance needed to halve the 1.28-window error is about 1e-5.
- The 1.28-window models are then the best at T >= 0.64 from width 128 on, with no cost at T = 0.32. Euler, |e(T = 1.28)|:

| width | one-step | 0.32 window | 1.28, fixed lr | 1.28, sqrt | 1.28, inv |
|---|---|---|---|---|---|
| 64 | 0.22-0.26 | 0.15-0.23 | 0.13-0.19 | 0.11-0.16 | 0.10-0.23 |
| 128 | 0.14-0.33 | 0.13-0.21 | 0.092-0.19 | 0.10-0.12 | 0.10-0.13 |
| 256 | 0.12-0.23 | 0.11-0.12 | 0.078-0.19 | 0.041-0.061 | 0.043-0.063 |
| 512 | 0.069-0.11 | 0.054-0.057 | 0.09-0.45 | 0.033-0.037 | 0.033-0.048 |

  At T = 0.32 the curvature-scaled 1.28-window models equal the 0.32-window models (width 256: 0.0023-0.0026 vs 0.0021-0.0024; width 512: 0.0015-0.0017 vs 0.0014-0.0017); at T = 5.12 the unrolled models overlap one another (all below one-step at width 256, overlapping it at the other widths).
- Width: the gain of the 1.28 window over the 0.32 window at T = 1.28 is within the seed spread at width 64, disjoint or touching at width 128, and disjoint at widths 256 and 512 (about 1.5-2.3 times). This is weak support for the hypothesis that longer windows pay off more for larger networks; with 3 seeds and short runs it is suggestive, not conclusive.
- Direct models behave the same way (1.28 window, sqrt: width 256 0.037-0.084 vs 0.32 window 0.078-0.19; width 512 0.044-0.061 vs 0.099-0.15).

Consequence for the earlier conclusions: windows of 1.28 (about 1.2 Lyapunov times) are trainable with Adam once the peak learning rate is scaled with the curvature of the loss; the earlier "1.28 cannot be trained" was a learning-rate effect. The rescaling picture of the theory note (curvature ~ e^{2 Lambda tau}, needed parameter distance shrinking accordingly) predicts the right learning rate within the tested range. Not tested: windows beyond 1.28, other dt, full-length schedules with the scaled rates.

## Relation to the roughness and multistep theory

The rollout-kernel draft (`multistep_rollout_kernel.tex`) lists five numerical checks; checks (i) and (v) are in Follow-up 5 above, (ii) and parts of (iii) and (iv) below.

1. **Continuous-time limit, Euler (check ii).** The draft proves that the Euler rollout features are a forward-Euler discretization of the learned sensitivity equation, so K^(p) depends on the horizon tau = p dt, not on p, and that nothing forces convergence for direct models. Measured: lambda_max(K^(p)) of Euler models is the same curve in T for dt = 0.01 ... 0.08 in every protocol (plot 7), and so is the curvature of the training loss (trainability check). For direct models both scale as dt^-2, the "features carry no factor dt" effect of the draft's direct-step section: at fixed tau the bound grows with q = tau/dt, and (tau/dt)^2 is what is measured.
2. **Rate of rollout training grows with the horizon (Theorem "rate" and the remark on safe learning rates).** The draft bounds lambda_max by phi(l, tau)^2 max s_i^2, exponential in tau for expanding learned dynamics, and notes that the safe gradient-descent learning rate then shrinks exponentially in the horizon. Measured: the curvature grows by about 40 from window 0.32 to 0.64, 400 from 0.64 to 1.28 and 160 from 1.28 to 2.56, and training at a fixed learning rate fails from window 1.28 on. Not confirmed: the gradient-descent threshold itself (Adam runs with lr x lambda_max up to 300 descended). Confirmed in the form that matters: scaling Adam's peak learning rate by (lambda_anchor / lambda)^(1/2) or (lambda_anchor / lambda) makes the 1.28 window train at every width (follow-up 6), so the scaling of the safe learning rate with the curvature carries over to Adam even though the threshold does not.
3. **Long-horizon remark (check iv, partly).** The draft conjectures lambda_max(K^(tau)) = exp(2 lambda_1 tau + o(tau)) with the kernel becoming nearly rank one along the leading Lyapunov direction. Measured: tr K / lambda_max falls to about 1.5 by T = 0.64, and the growth rate of the curvature per unit window falls from about 11 (0.32-0.64) to about 4 (1.28-2.56), toward 2 lambda_1 = 1.8 of Lorenz-63, consistent with finite-time exponents converging to the asymptotic one. The alignment of the top eigenvector with the Lyapunov direction was not measured.
4. **Short horizons (Proposition "short", check iii, indirectly).** For h tau << 1 the draft gives K^(q) = tau_q^2 (Kbar^(q) + E). Measured: lambda_max grows from 470 to 1.5e3 between T = 0.08 and 0.16 (a factor 3.2 for tau^2 = 4) and the training-loss curvature from 570 to 2.6e3 between windows 0.16 and 0.32 (4.5): close to tau^2 at the shortest horizons, with exponential growth taking over by 0.32-0.64. Kbar^(q) itself was not formed.
5. **Error identity and the propagated injections (eigenanalysis expansion).** e(t_p) = sum_i P(t_p,t_i)(eps_i + q_i) holds exactly (1e-11 with converged quadrature). The nonlinear part is below 1% of e up to T = 0.64 for Euler and 4-70% at T = 1.28 (plot 5): the linear theory, and the kernel picture built on it, describe windows up to about half a Lyapunov time, which is also where unrolled training works.
6. **What unrolling changes.** The gradient of the q-th term is a sum of one-step feature gradients weighted by the propagator transposed, so unrolled training penalizes one-step errors along directions the flow amplifies. Measured: for Euler models the learned Jacobian along the error matches the truth map's in every protocol (rho - 1 = rho^f - 1 to about 10% wherever |rho - 1| > 0.01; direct models trained by unrolling at small dt contract more than the truth), while the error direction moves into contracting directions (Euler dt = 0.01: mean rho - 1 from +0.0015 one-step to -0.014 ... -0.028 unrolled). This is the mechanism by which the short-window losses improve long-horizon error.
7. **Roughness of the error map and Euler vs direct.** With J = D Phi_h + D E, the variation of the error map along the error, R_e(H - f_dt), measures how far the learned Jacobian is from the truth's in the direction that matters. Euler: 5-11 at every dt; direct: 64 / 29 / 14 / 6, about 1/dt. Per step in state units the direct model's Jacobian error at dt = 0.01 is about 7 times the Euler model's; its rollout error at T = 1.28 is about 3 times larger after one-step training and 10 times larger after unrolled training, converging by dt = 0.08. Same direction and dt trend as the one-step Euler-vs-direct result. In addition, the direct model's rollout loss is 10-1000 times sharper than the Euler model's at the same window (item 1), so at a common learning rate it is also harder to train by unrolling.
8. **Roughness and the near-rank-one kernel.** With a step size limited by lambda_max, an error component along an eigendirection with eigenvalue lambda_k decreases at a rate of order lambda_k / lambda_max. At long windows almost the whole spectrum is in one eigenvalue (tr K / lambda_max about 1.5), so every component off that direction is slow; this is the roughness-theory reading of the lr 1e-6 runs (no blow-up, no progress). Check (v) (Follow-up 5) sharpens this: for a step rule that follows the gradient (Adam, gradient descent) the top direction dominates the step and the step overshoots, while the first-order optimal step of the same length, which is small along that direction, removes most of the error on the batch. The slow progress of the lr 1e-6 runs is therefore a property of the step rule, not of the loss alone.

Not explained by the theory: the best window (0.3-0.6 Lyapunov times at a fixed learning rate; at least 1.2 Lyapunov times with curvature-scaled rates, longer windows untested), why one-step models at small dt leave their error in neutral directions, and the transient increases of 20-1000 times at the start of every fine-tune (Adam with a fresh optimizer state on a sharp loss; outside the gradient-flow analysis).

## Limitations

- 3 seeds; several comparisons rely on ranges that are disjoint by a small margin.
- One width (256, not converged), one optimizer family (Adam), one learning rate per stage in the follow-ups.
- The curriculum and damping were tested at dt = 0.04 and 0.08 only.
- Lorenz-63 is 3-dimensional; the training windows (up to 0.6 Lyapunov times usefully) and evaluation horizons (up to 4.6) are short compared with long-run statistics, which were not measured.
- M6 and M7 were measured at final checkpoints only; intermediate checkpoints have M1-M5 on 1024 starts with 10-point quadrature.

---

# Tables (generated by aggregate.py)

Over the 3 seeds: mean [min, max]. Final checkpoint, held-out starts.

## Identity checks

- Pre-sweep, nontrivial random parameters, all classes and dt: state Jacobian (forward mode) vs autograd: max rel. error 1.33e-15
- Pre-sweep, nontrivial random parameters, all classes and dt: Psi^(p) g vs central FD of the rollout: max rel. error 1.06e-08
- Pre-sweep, nontrivial random parameters, all classes and dt: g^(Sigma)(p) vs FD of L_p: max rel. error 1.85e-06
- Pre-sweep, nontrivial random parameters, all classes and dt: reconstruction e(t_p) = sum P (eps + q), 10-pt GL: max rel. error 1.51e-11
- Trained models (final checkpoints, T = 0.32 and 1.28): Psi g vs FD max 4.28e-05; g^(Sigma) vs FD max 8.16e-05
- Reconstruction, intermediate checkpoints (10-point GL, p <= p(1.28)): max rel. error 4.15e-02 (quadrature error where |e| is attractor-sized; share > 1e-8: 0.52)
- Reconstruction, final checkpoints (40-point GL, recompute_gl.py): max rel. error 1.12e-07

### Plot 1: R(T=1.28)

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.353 [0.306, 0.431] | 0.131 [0.0855, 0.162] | 0.225 [0.182, 0.297] | 0.162 [0.138, 0.21] | 0.122 [0.101, 0.159] | 0.981 [0.728, 1.38] | 1.19 [1.17, 1.21] | 1.69 [1.44, 1.87] | 0.324 [0.306, 0.336] | 1.32 [1.15, 1.41] |
| 0.02 | 0.336 [0.314, 0.358] | 0.122 [0.0704, 0.154] | 0.281 [0.176, 0.374] | 0.169 [0.125, 0.231] | 0.0597 [0.048, 0.0761] | 0.61 [0.515, 0.745] | 0.552 [0.343, 0.658] | 1.3 [0.845, 2.01] | 0.281 [0.187, 0.388] | 0.627 [0.586, 0.657] |
| 0.04 | 0.154 [0.134, 0.179] | 0.143 [0.125, 0.169] | 0.272 [0.185, 0.366] | 0.103 [0.0853, 0.138] | 0.159 [0.0821, 0.296] | 0.228 [0.19, 0.291] | 0.529 [0.347, 0.644] | 0.616 [0.549, 0.664] | 0.149 [0.115, 0.192] | 0.406 [0.216, 0.651] |
| 0.08 | 0.161 [0.117, 0.224] | 0.136 [0.0906, 0.217] | 0.216 [0.191, 0.24] | 0.112 [0.0985, 0.121] | 0.111 [0.091, 0.131] | 0.16 [0.102, 0.229] | 0.202 [0.157, 0.275] | 0.343 [0.295, 0.417] | 0.135 [0.0717, 0.192] | 0.12 [0.12, 0.121] |

### Plot 1 (context): |e(T=1.28)|

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.361 [0.313, 0.44] | 0.139 [0.0945, 0.168] | 0.256 [0.213, 0.33] | 0.169 [0.145, 0.218] | 0.142 [0.12, 0.18] | 1.02 [0.763, 1.43] | 1.43 [1.39, 1.46] | 2.69 [2.46, 2.85] | 0.369 [0.353, 0.38] | 1.48 [1.32, 1.57] |
| 0.02 | 0.344 [0.322, 0.366] | 0.131 [0.0797, 0.163] | 0.311 [0.204, 0.403] | 0.176 [0.131, 0.238] | 0.0785 [0.0694, 0.0939] | 0.631 [0.535, 0.765] | 0.642 [0.411, 0.765] | 1.6 [1.13, 2.35] | 0.301 [0.205, 0.408] | 0.691 [0.645, 0.734] |
| 0.04 | 0.161 [0.141, 0.186] | 0.15 [0.131, 0.178] | 0.306 [0.223, 0.395] | 0.109 [0.0916, 0.144] | 0.173 [0.0957, 0.31] | 0.238 [0.202, 0.302] | 0.566 [0.388, 0.68] | 0.717 [0.657, 0.769] | 0.16 [0.126, 0.203] | 0.435 [0.245, 0.677] |
| 0.08 | 0.166 [0.122, 0.229] | 0.143 [0.0972, 0.224] | 0.248 [0.223, 0.27] | 0.117 [0.104, 0.126] | 0.119 [0.0998, 0.14] | 0.167 [0.108, 0.236] | 0.218 [0.173, 0.291] | 0.391 [0.337, 0.472] | 0.142 [0.0776, 0.197] | 0.132 [0.132, 0.133] |

### Plot 1 (context): I(T=1.28)

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.00781 [0.00692, 0.00852] | 0.0085 [0.00664, 0.00979] | 0.0303 [0.027, 0.033] | 0.00721 [0.00657, 0.0077] | 0.0198 [0.0188, 0.0213] | 0.0381 [0.0356, 0.043] | 0.237 [0.211, 0.253] | 1 [0.893, 1.1] | 0.0456 [0.0393, 0.0498] | 0.155 [0.142, 0.164] |
| 0.02 | 0.00786 [0.00657, 0.00887] | 0.00906 [0.00883, 0.00936] | 0.0298 [0.0284, 0.0315] | 0.00721 [0.00633, 0.00799] | 0.0188 [0.0173, 0.0214] | 0.0208 [0.0191, 0.0226] | 0.0898 [0.0674, 0.111] | 0.309 [0.289, 0.347] | 0.0198 [0.0186, 0.0214] | 0.0642 [0.0574, 0.0763] |
| 0.04 | 0.0067 [0.00643, 0.00685] | 0.00753 [0.00629, 0.00887] | 0.0337 [0.0287, 0.0386] | 0.0062 [0.00604, 0.00638] | 0.0139 [0.0136, 0.0144] | 0.0105 [0.00916, 0.0119] | 0.0377 [0.0362, 0.0402] | 0.1 [0.088, 0.108] | 0.0106 [0.00967, 0.0111] | 0.0283 [0.0266, 0.0295] |
| 0.08 | 0.0054 [0.00511, 0.00559] | 0.00704 [0.00655, 0.0075] | 0.0322 [0.03, 0.0344] | 0.00523 [0.00502, 0.00545] | 0.0087 [0.0084, 0.00886] | 0.00647 [0.00563, 0.00749] | 0.0158 [0.0154, 0.016] | 0.0486 [0.0416, 0.0549] | 0.00627 [0.00555, 0.00737] | 0.0117 [0.0107, 0.0127] |

### Plot 2: mean(rho - 1) over i < p(1.28) (Euler: = dt n.gradH.n)

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.00147 [0.00109, 0.00185] | -0.0144 [-0.0193, -0.00918] | -0.0237 [-0.0277, -0.0174] | -0.00283 [-0.00319, -0.0026] | -0.0281 [-0.0291, -0.0264] | -0.00286 [-0.00344, -0.00213] | -0.0488 [-0.051, -0.0471] | -0.0834 [-0.103, -0.0569] | -0.00699 [-0.00924, -0.00345] | -0.0358 [-0.0419, -0.0281] |
| 0.02 | -0.000503 [-0.000931, -0.000169] | -0.0309 [-0.0326, -0.0292] | -0.042 [-0.053, -0.032] | -0.00977 [-0.0104, -0.00938] | -0.0641 [-0.0753, -0.0534] | -0.0103 [-0.014, -0.00833] | -0.0553 [-0.0595, -0.0501] | -0.0935 [-0.101, -0.0848] | -0.0221 [-0.0273, -0.0173] | -0.0549 [-0.065, -0.0397] |
| 0.04 | -0.0208 [-0.0211, -0.02] | -0.0688 [-0.0748, -0.0655] | -0.0993 [-0.102, -0.0963] | -0.0338 [-0.0388, -0.0301] | -0.103 [-0.128, -0.0575] | -0.0359 [-0.042, -0.0271] | -0.0733 [-0.0868, -0.0546] | -0.136 [-0.139, -0.133] | -0.0454 [-0.0502, -0.0421] | -0.0797 [-0.095, -0.0662] |
| 0.08 | -0.111 [-0.122, -0.103] | -0.175 [-0.177, -0.173] | -0.235 [-0.245, -0.222] | -0.14 [-0.143, -0.135] | -0.233 [-0.252, -0.205] | -0.142 [-0.156, -0.129] | -0.177 [-0.185, -0.166] | -0.269 [-0.272, -0.264] | -0.166 [-0.171, -0.158] | -0.202 [-0.229, -0.184] |

### Plot 2 (context): mean(rhobar - 1)

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.00147 [0.00109, 0.00185] | -0.0144 [-0.0193, -0.00918] | -0.0237 [-0.0277, -0.0174] | -0.00283 [-0.00319, -0.0026] | -0.0281 [-0.0291, -0.0264] | -0.00286 [-0.00344, -0.00212] | -0.0488 [-0.051, -0.0471] | -0.0833 [-0.103, -0.0569] | -0.00699 [-0.00924, -0.00344] | -0.0358 [-0.0419, -0.0281] |
| 0.02 | -0.000504 [-0.000932, -0.000169] | -0.0309 [-0.0326, -0.0292] | -0.042 [-0.053, -0.032] | -0.00977 [-0.0104, -0.00938] | -0.0641 [-0.0753, -0.0534] | -0.0103 [-0.014, -0.00833] | -0.0553 [-0.0595, -0.0502] | -0.0934 [-0.101, -0.0847] | -0.0221 [-0.0273, -0.0173] | -0.0549 [-0.065, -0.0396] |
| 0.04 | -0.0208 [-0.0211, -0.02] | -0.0688 [-0.0748, -0.0655] | -0.0993 [-0.102, -0.0963] | -0.0338 [-0.0388, -0.0301] | -0.103 [-0.128, -0.0575] | -0.0359 [-0.0419, -0.0271] | -0.0733 [-0.0868, -0.0546] | -0.136 [-0.139, -0.133] | -0.0454 [-0.0502, -0.0421] | -0.0797 [-0.095, -0.0663] |
| 0.08 | -0.111 [-0.122, -0.103] | -0.175 [-0.177, -0.173] | -0.235 [-0.245, -0.222] | -0.14 [-0.143, -0.135] | -0.233 [-0.252, -0.205] | -0.142 [-0.156, -0.129] | -0.177 [-0.185, -0.166] | -0.269 [-0.272, -0.264] | -0.166 [-0.171, -0.158] | -0.202 [-0.229, -0.184] |

### Plot 2 (context): truth map mean(rho^f - 1)

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.00068 [0.000422, 0.000958] | -0.0158 [-0.0206, -0.0101] | -0.0281 [-0.0327, -0.0221] | -0.00385 [-0.00423, -0.00358] | -0.0303 [-0.0312, -0.0287] | 0.00203 [0.000836, 0.0027] | -0.0198 [-0.0216, -0.0185] | -0.0343 [-0.04, -0.0271] | -0.0009 [-0.00337, 0.00265] | -0.0158 [-0.0196, -0.012] |
| 0.02 | -0.00203 [-0.00292, -0.00149] | -0.0335 [-0.0353, -0.0318] | -0.0486 [-0.0619, -0.0376] | -0.0117 [-0.0121, -0.011] | -0.0687 [-0.08, -0.0587] | -0.00519 [-0.00819, -0.0036] | -0.0354 [-0.0376, -0.0323] | -0.0538 [-0.0595, -0.0497] | -0.0145 [-0.0179, -0.0115] | -0.0357 [-0.0412, -0.0267] |
| 0.04 | -0.0242 [-0.025, -0.0234] | -0.0719 [-0.0779, -0.0686] | -0.111 [-0.114, -0.104] | -0.0377 [-0.0429, -0.0337] | -0.111 [-0.135, -0.0638] | -0.0295 [-0.0354, -0.0211] | -0.0624 [-0.0768, -0.0439] | -0.113 [-0.116, -0.111] | -0.0377 [-0.043, -0.0338] | -0.0661 [-0.0782, -0.0558] |
| 0.08 | -0.12 [-0.132, -0.11] | -0.179 [-0.18, -0.176] | -0.252 [-0.261, -0.239] | -0.151 [-0.156, -0.145] | -0.247 [-0.264, -0.22] | -0.137 [-0.149, -0.125] | -0.168 [-0.175, -0.161] | -0.25 [-0.252, -0.247] | -0.159 [-0.164, -0.152] | -0.192 [-0.22, -0.173] |

### Plot 3: c_Sigma at T = 0.08, 0.16, 0.32, 0.64, 1.28 (seed means)

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.757 / 0.646 / 0.439 / 0.0289 / 0.0692 | 0.459 / 0.422 / 0.443 / 0.0165 / 0.0332 | 0.44 / 0.277 / 0.0846 / -0.0294 / 0.0165 | 0.627 / 0.32 / 0.161 / -0.0787 / -0.111 | 0.772 / 0.418 / 0.153 / -0.0397 / -0.0634 | 0.883 / 0.754 / 0.266 / 0.0339 / -0.000594 | 0.387 / 0.25 / 0.121 / 0.0657 / -0.0757 | 0.428 / 0.507 / 0.467 / 0.214 / 0.0618 | 0.83 / 0.728 / 0.585 / 0.24 / 0.243 | 0.532 / 0.277 / 0.0532 / 0.0694 / 0.0476 |
| 0.02 | 0.883 / 0.656 / 0.336 / -0.153 / 0.0288 | 0.314 / 0.0895 / -0.177 / -0.1 / 0.3 | 0.582 / 0.311 / 0.19 / 0.172 / 0.18 | 0.691 / 0.51 / 0.31 / -0.148 / -0.101 | 0.778 / 0.427 / 0.247 / -0.00975 / -0.0166 | 0.86 / 0.706 / 0.404 / 0.189 / 0.306 | 0.523 / 0.264 / 0.141 / 0.13 / 0.201 | 0.692 / 0.523 / 0.466 / 0.396 / 0.152 | 0.577 / 0.474 / 0.292 / 0.25 / -0.0198 | 0.718 / 0.431 / 0.0759 / 0.0686 / -0.0234 |
| 0.04 | 0.872 / 0.577 / 0.393 / 0.177 / 0.1 | 0.741 / 0.45 / 0.274 / 0.0994 / 0.0341 | 0.794 / 0.428 / 0.272 / 0.00361 / 0.212 | 0.881 / 0.58 / 0.327 / 0.0491 / 0.0208 | 0.9 / 0.717 / 0.347 / 0.103 / 0.241 | 0.897 / 0.782 / 0.562 / -0.0161 / 0.0975 | 0.797 / 0.628 / 0.566 / 0.179 / 0.335 | 0.665 / 0.471 / 0.417 / 0.168 / 0.0122 | 0.949 / 0.793 / 0.399 / 0.274 / 0.0787 | 0.93 / 0.727 / 0.23 / 0.292 / 0.0156 |
| 0.08 | 1 / 0.849 / 0.619 / 0.266 / 0.239 | 1 / 0.801 / 0.504 / 0.237 / 0.0862 | 1 / 0.751 / 0.705 / 0.23 / -0.0942 | 1 / 0.752 / 0.493 / 0.0645 / 0.023 | 1 / 0.779 / 0.456 / 0.106 / 0.224 | 1 / 0.865 / 0.37 / 0.0653 / 0.0785 | 1 / 0.808 / 0.383 / -0.0183 / 0.00875 | 1 / 0.831 / 0.362 / 0.141 / 0.267 | 1 / 0.821 / 0.455 / 0.0423 / -0.0614 | 1 / 0.725 / 0.408 / 0.293 / 0.253 |

### Plot 3b: per-horizon cos(g1, g^(q)) at the same T (seed means)

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.64 / 0.503 / 0.231 / 0.0189 / 0.0626 | 0.303 / 0.455 / 0.315 / -0.0314 / 0.0597 | 0.305 / 0.26 / -0.0497 / -0.0687 / 0.0147 | 0.502 / 0.0561 / 0.094 / -0.103 / 0.0752 | 0.568 / 0.262 / -0.0678 / -0.0491 / -0.0377 | 0.847 / 0.638 / 0.115 / 0.0317 / 0.0202 | 0.345 / 0.156 / 0.0726 / 0.0644 / 0.015 | 0.299 / 0.464 / 0.0247 / 0.264 / 0.131 | 0.791 / 0.644 / 0.5 / 0.212 / 0.145 | 0.37 / 0.0204 / 0.0422 / 0.0716 / 0.0387 |
| 0.02 | 0.813 / 0.412 / 0.196 / -0.19 / 0.202 | -0.00114 / 0.156 / -0.336 / -0.0594 / 0.289 | 0.284 / 0.161 / -0.0317 / 0.15 / 0.183 | 0.56 / 0.306 / 0.186 / -0.181 / -0.027 | 0.581 / 0.249 / -0.113 / -0.00116 / -0.0248 | 0.801 / 0.605 / 0.338 / 0.163 / 0.305 | 0.484 / 0.161 / 0.105 / 0.122 / 0.0975 | 0.644 / 0.408 / 0.148 / 0.289 / 0.181 | 0.474 / 0.412 / 0.218 / 0.246 / -0.0265 | 0.594 / 0.178 / 0.0171 / 0.0841 / -0.189 |
| 0.04 | 0.795 / 0.46 / 0.31 / 0.151 / 0.146 | 0.528 / 0.228 / 0.181 / 0.0906 / -0.0397 | 0.423 / -0.056 / -0.0353 / 0.00657 / 0.197 | 0.808 / 0.342 / 0.217 / 0.0319 / -0.00253 | 0.796 / 0.588 / 0.188 / 0.09 / 0.224 | 0.845 / 0.715 / 0.465 / -0.0599 / 0.0487 | 0.513 / 0.448 / 0.34 / 0.166 / 0.298 | 0.429 / 0.153 / 0.0741 / 0.0907 / 0.0183 | 0.91 / 0.601 / 0.289 / 0.273 / 0.0466 | 0.883 / 0.454 / 0.133 / 0.291 / -0.143 |
| 0.08 | 1 / 0.726 / 0.505 / 0.245 / 0.218 | 1 / 0.466 / 0.28 / 0.196 / -0.00953 | 1 / 0.418 / 0.464 / 0.163 / -0.0209 | 1 / 0.55 / 0.3 / 0.0497 / -0.0173 | 1 / 0.601 / 0.177 / 0.0863 / 0.261 | 1 / 0.724 / 0.235 / 0.0578 / 0.0424 | 1 / 0.463 / 0.148 / -0.0549 / 0.0989 | 1 / 0.581 / 0.199 / 0.2 / 0.251 | 1 / 0.634 / 0.243 / 0.0269 / -0.148 | 1 / 0.468 / 0.353 / 0.288 / 0.213 |

### Plot 4: sum_i ln|rhobar_i| at T = 0.08, 0.16, 0.32, 0.64, 1.28 (seed means; full scatter in tables/plot4_scatter.csv)

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | -0.235 / -0.339 / -0.335 / -0.181 / 0.044 | -0.352 / -0.645 / -1.1 / -1.56 / -2.01 | -0.326 / -0.643 / -1.26 / -2.2 / -3.22 | -0.267 / -0.442 / -0.582 / -0.583 / -0.505 | -0.494 / -0.957 / -1.74 / -2.72 / -3.81 | -0.473 / -0.695 / -0.813 / -0.725 / -0.602 | -1.01 / -1.85 / -3.31 / -5.26 / -7.25 | -1.23 / -2.33 / -4.47 / -7.94 / -12.2 | -0.492 / -0.762 / -1.05 / -1.18 / -1.15 | -0.815 / -1.53 / -2.73 / -4.18 / -5.37 |
| 0.02 | -0.193 / -0.311 / -0.358 / -0.281 / -0.333 | -0.307 / -0.626 / -1.15 / -1.71 / -2.33 | -0.257 / -0.561 / -1.15 / -2.02 / -3.02 | -0.241 / -0.445 / -0.651 / -0.731 / -0.923 | -0.459 / -0.97 / -1.85 / -3.04 / -4.55 | -0.364 / -0.623 / -0.843 / -0.881 / -1.02 | -0.52 / -1.06 / -1.94 / -3.02 / -4.25 | -0.676 / -1.43 / -2.78 / -4.8 / -7.24 | -0.368 / -0.658 / -1.04 / -1.4 / -1.83 | -0.527 / -1.09 / -2.05 / -3.21 / -4.3 |
| 0.04 | -0.176 / -0.41 / -0.666 / -0.829 / -1.43 | -0.227 / -0.607 / -1.21 / -1.9 / -2.98 | -0.21 / -0.58 / -1.32 / -2.45 / -3.98 | -0.186 / -0.475 / -0.889 / -1.22 / -1.84 | -0.287 / -0.78 / -1.62 / -2.69 / -4.2 | -0.229 / -0.552 / -0.942 / -1.27 / -1.99 | -0.249 / -0.658 / -1.32 / -2.08 / -3.22 | -0.33 / -0.919 / -1.99 / -3.58 / -5.58 | -0.223 / -0.555 / -1.02 / -1.52 / -2.32 | -0.28 / -0.748 / -1.53 / -2.51 / -3.59 |
| 0.08 | 0 / -0.509 / -1.35 / -2.56 / -5.03 | 0 / -0.604 / -1.71 / -3.2 / -5.98 | 0 / -0.559 / -1.7 / -3.59 / -6.7 | 0 / -0.526 / -1.47 / -2.79 / -5.4 | 0 / -0.69 / -1.84 / -3.72 / -6.78 | 0 / -0.684 / -1.74 / -3.04 / -5.6 | 0 / -0.665 / -1.85 / -3.41 / -6.2 | 0 / -0.719 / -2.16 / -4.43 / -7.96 | 0 / -0.687 / -1.87 / -3.36 / -6.04 | 0 / -0.742 / -1.98 / -3.77 / -6.73 |

### Plot 5: |sum P q| / |e(t_p)| at T = 0.08, 0.16, 0.32, 0.64, 1.28 (seed means)

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 8.96e-05 / 0.000256 / 0.000738 / 0.00681 / 0.715 | 0.000238 / 0.000378 / 0.00121 / 0.00217 / 0.215 | 0.000943 / 0.002 / 0.0049 / 0.00808 / 0.269 | 0.000113 / 0.00035 / 0.000945 / 0.00216 / 0.337 | 0.000126 / 0.000446 / 0.0011 / 0.00364 / 0.131 | 0.000552 / 0.00128 / 0.00307 / 0.0786 / 1.6 | 0.011 / 0.0241 / 0.0241 / 0.0446 / 1.93 | 0.018 / 0.0451 / 0.09 / 0.257 / 1.75 | 0.000553 / 0.00129 / 0.00259 / 0.031 / 0.158 | 0.00317 / 0.00591 / 0.00887 / 0.0482 / 1.24 |
| 0.02 | 6.77e-05 / 0.000224 / 0.000644 / 0.00451 / 0.685 | 0.000212 / 0.000307 / 0.000896 / 0.00203 / 0.17 | 0.000969 / 0.00214 / 0.00412 / 0.00718 / 0.208 | 8.22e-05 / 0.000282 / 0.000807 / 0.00196 / 0.353 | 0.000117 / 0.000416 / 0.00102 / 0.00343 / 0.0406 | 0.0002 / 0.00062 / 0.00173 / 0.0207 / 0.437 | 0.00226 / 0.00327 / 0.00521 / 0.0167 / 0.568 | 0.00405 / 0.00972 / 0.0218 / 0.0362 / 0.668 | 0.000241 / 0.000681 / 0.00207 / 0.0097 / 0.251 | 0.000749 / 0.00178 / 0.00334 / 0.0219 / 0.576 |
| 0.04 | 3.32e-05 / 0.000137 / 0.000672 / 0.00341 / 0.159 | 0.000149 / 0.000473 / 0.000953 / 0.00193 / 0.188 | 0.000962 / 0.00172 / 0.0036 / 0.00634 / 0.162 | 3.89e-05 / 0.000173 / 0.000758 / 0.00204 / 0.12 | 8.37e-05 / 0.000291 / 0.000952 / 0.00212 / 0.17 | 9.59e-05 / 0.00029 / 0.00159 / 0.00988 / 0.229 | 0.000573 / 0.00164 / 0.00424 / 0.00867 / 0.513 | 0.00168 / 0.00557 / 0.0124 / 0.0197 / 0.376 | 7.84e-05 / 0.000285 / 0.00199 / 0.00479 / 0.144 | 0.000212 / 0.000653 / 0.00165 / 0.0103 / 0.346 |
| 0.08 | 0 / 0.000102 / 0.000566 / 0.00356 / 0.204 | 0 / 0.00033 / 0.0016 / 0.00144 / 0.178 | 0 / 0.00148 / 0.0044 / 0.00798 / 0.124 | 0 / 9.87e-05 / 0.000661 / 0.00185 / 0.138 | 0 / 0.000161 / 0.000492 / 0.00138 / 0.0995 | 0 / 0.000299 / 0.00119 / 0.00617 / 0.117 | 0 / 0.000411 / 0.0024 / 0.00441 / 0.167 | 0 / 0.003 / 0.0076 / 0.0131 / 0.215 | 0 / 0.000285 / 0.00136 / 0.00224 / 0.154 | 0 / 0.000565 / 0.00159 / 0.00463 / 0.0674 |

### Plot 5 (context): |sum P eps| at T = 0.08, 0.16, 0.32, 0.64, 1.28 (seed means)

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.00124 / 0.00207 / 0.00381 / 0.0196 / 0.506 | 0.0022 / 0.0027 / 0.00319 / 0.013 / 0.136 | 0.00987 / 0.0109 / 0.0115 / 0.0168 / 0.275 | 0.00127 / 0.0019 / 0.00272 / 0.00942 / 0.201 | 0.00338 / 0.00393 / 0.00474 / 0.0118 / 0.144 | 0.00634 / 0.0114 / 0.0274 / 0.178 / 2.1 | 0.0355 / 0.0384 / 0.0479 / 0.222 / 2.93 | 0.143 / 0.146 / 0.153 / 0.24 / 4.83 | 0.00778 / 0.0115 / 0.0212 / 0.122 / 0.366 | 0.0419 / 0.0425 / 0.0505 / 0.174 / 2 |
| 0.02 | 0.00123 / 0.00209 / 0.00378 / 0.0215 / 0.473 | 0.0022 / 0.00263 / 0.00315 / 0.0118 / 0.124 | 0.01 / 0.0106 / 0.0122 / 0.0175 / 0.327 | 0.00125 / 0.00189 / 0.00261 / 0.00989 / 0.212 | 0.00331 / 0.00377 / 0.00443 / 0.00735 / 0.0788 | 0.00348 / 0.00586 / 0.0115 / 0.0734 / 0.678 | 0.0175 / 0.0189 / 0.023 / 0.107 / 0.809 | 0.0672 / 0.0664 / 0.0728 / 0.112 / 1.68 | 0.00389 / 0.00564 / 0.0085 / 0.0438 / 0.323 | 0.0172 / 0.0178 / 0.0217 / 0.0719 / 0.747 |
| 0.04 | 0.00113 / 0.00197 / 0.00325 / 0.0155 / 0.168 | 0.00217 / 0.00242 / 0.00286 / 0.0111 / 0.145 | 0.01 / 0.0108 / 0.012 / 0.0183 / 0.312 | 0.00114 / 0.00177 / 0.00238 / 0.00906 / 0.115 | 0.00296 / 0.00347 / 0.00408 / 0.00992 / 0.189 | 0.00202 / 0.00336 / 0.00589 / 0.0406 / 0.226 | 0.00939 / 0.011 / 0.0131 / 0.0553 / 0.651 | 0.0304 / 0.0297 / 0.0358 / 0.0484 / 0.773 | 0.0022 / 0.0033 / 0.00434 / 0.0232 / 0.153 | 0.00787 / 0.00853 / 0.0112 / 0.0379 / 0.474 |
| 0.08 | 0.00114 / 0.0019 / 0.0028 / 0.0126 / 0.153 | 0.00224 / 0.00278 / 0.00271 / 0.00924 / 0.129 | 0.0109 / 0.0117 / 0.0128 / 0.0182 / 0.242 | 0.00119 / 0.00179 / 0.00215 / 0.0085 / 0.114 | 0.00245 / 0.00297 / 0.00358 / 0.00722 / 0.117 | 0.00161 / 0.00258 / 0.00383 / 0.0156 / 0.168 | 0.00513 / 0.00627 / 0.00718 / 0.0255 / 0.229 | 0.017 / 0.0184 / 0.0214 / 0.0279 / 0.393 | 0.00174 / 0.00254 / 0.00314 / 0.00997 / 0.15 | 0.00376 / 0.00478 / 0.00579 / 0.013 / 0.132 |

### Plot 5 (context): |sum P q| at T = 0.08, 0.16, 0.32, 0.64, 1.28 (seed means)

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 1.12e-07 / 5.33e-07 / 2.8e-06 / 0.000165 / 0.265 | 5.28e-07 / 1.04e-06 / 3.85e-06 / 2.84e-05 / 0.0313 | 9.28e-06 / 2.2e-05 / 5.62e-05 / 0.000136 / 0.0764 | 1.44e-07 / 6.74e-07 / 2.58e-06 / 2.05e-05 / 0.0597 | 4.27e-07 / 1.75e-06 / 5.21e-06 / 4.29e-05 / 0.0193 | 3.5e-06 / 1.44e-05 / 8.48e-05 / 0.0148 / 1.82 | 0.000394 / 0.000933 / 0.00116 / 0.00999 / 2.76 | 0.00258 / 0.00664 / 0.014 / 0.0582 / 4.71 | 4.3e-06 / 1.49e-05 / 5.69e-05 / 0.00437 / 0.0584 | 0.000134 / 0.000251 / 0.000448 / 0.00909 / 1.82 |
| 0.02 | 8.43e-08 / 4.67e-07 / 2.39e-06 / 0.000104 / 0.236 | 4.78e-07 / 8.07e-07 / 2.78e-06 / 2.43e-05 / 0.0247 | 9.72e-06 / 2.27e-05 / 5e-05 / 0.000129 / 0.0687 | 1.03e-07 / 5.34e-07 / 2.09e-06 / 1.94e-05 / 0.0664 | 3.88e-07 / 1.56e-06 / 4.54e-06 / 2.55e-05 / 0.00323 | 6.98e-07 / 3.63e-06 / 1.99e-05 / 0.0017 / 0.281 | 4.04e-05 / 6.26e-05 / 0.000121 / 0.00184 / 0.408 | 0.000269 / 0.000645 / 0.00157 / 0.00412 / 1.18 | 9.39e-07 / 3.84e-06 / 1.76e-05 / 0.000426 / 0.0811 | 1.31e-05 / 3.18e-05 / 7.21e-05 / 0.0016 / 0.408 |
| 0.04 | 3.78e-08 / 2.7e-07 / 2.2e-06 / 5.32e-05 / 0.0255 | 3.27e-07 / 1.16e-06 / 2.77e-06 / 2.16e-05 / 0.028 | 9.67e-06 / 1.85e-05 / 4.33e-05 / 0.000117 / 0.053 | 4.47e-08 / 3.07e-07 / 1.82e-06 / 1.88e-05 / 0.0142 | 2.46e-07 / 1e-06 / 3.91e-06 / 2.11e-05 / 0.0429 | 1.92e-07 / 9.81e-07 / 9.12e-06 / 0.000441 / 0.0602 | 5.63e-06 / 1.83e-05 / 5.6e-05 / 0.000481 / 0.318 | 5.16e-05 / 0.000166 / 0.000446 / 0.000953 / 0.27 | 1.72e-07 / 9.56e-07 / 8.51e-06 / 0.00012 / 0.0258 | 1.68e-06 / 5.57e-06 / 1.83e-05 / 0.000502 / 0.174 |
| 0.08 | 0 / 1.95e-07 / 1.58e-06 / 4.49e-05 / 0.037 | 0 / 9.27e-07 / 4.33e-06 / 1.36e-05 / 0.0343 | 0 / 1.72e-05 / 5.63e-05 / 0.000143 / 0.0307 | 0 / 1.79e-07 / 1.4e-06 / 1.57e-05 / 0.0164 | 0 / 4.77e-07 / 1.75e-06 / 9.96e-06 / 0.012 | 0 / 7.73e-07 / 4.59e-06 / 0.000116 / 0.021 | 0 / 2.58e-06 / 1.75e-05 / 0.000113 / 0.0412 | 0 / 5.52e-05 / 0.000165 / 0.000364 / 0.0877 | 0 / 7.3e-07 / 4.31e-06 / 2.3e-05 / 0.0267 | 0 / 2.71e-06 / 9.16e-06 / 6.01e-05 / 0.00891 |

### Plot 6: R_e(H), T = 1.28

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 18.8 [18.6, 19.1] | 17.9 [17.1, 18.3] | 17.6 [15.7, 19.2] | 19.1 [18.7, 19.5] | 18.8 [18.6, 18.9] | 75.1 [73.6, 76.8] | 78.3 [60.5, 87.8] | 71.5 [66.4, 77.4] | 78.6 [76.3, 82.1] | 79.9 [76.6, 84.9] |
| 0.02 | 17.2 [16.9, 17.6] | 16.5 [15.9, 17.5] | 16.2 [15, 17.4] | 17.8 [17.6, 17.9] | 17.5 [17.4, 17.7] | 37.5 [34.3, 40.8] | 38.2 [32.9, 43] | 41.2 [40.7, 41.9] | 40.8 [35.2, 45.7] | 41 [34.5, 48.2] |
| 0.04 | 14.5 [14.2, 14.7] | 14.2 [13.9, 14.6] | 16.4 [14.3, 19] | 14.7 [14.3, 15.1] | 14.4 [14.1, 14.8] | 21.8 [20.7, 22.6] | 22.3 [21.1, 23.5] | 21.8 [20.1, 22.8] | 23 [21.4, 24.1] | 22.5 [21.7, 23.8] |
| 0.08 | 18.5 [18.4, 18.6] | 18.2 [17.6, 18.8] | 19.2 [18.9, 19.6] | 18.3 [17.9, 18.6] | 18.3 [18.1, 18.5] | 17.8 [17.6, 18] | 18.1 [17.6, 18.4] | 19.3 [19.2, 19.3] | 17.6 [17.5, 17.7] | 18.1 [17.5, 18.9] |

### Plot 6: max (1 - rho)/dt, T = 1.28

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 18.8 [18.6, 19.1] | 17.9 [17.1, 18.3] | 17.6 [15.7, 19.2] | 19.1 [18.7, 19.5] | 18.8 [18.6, 18.9] | 75.1 [73.6, 76.8] | 78 [60.6, 87.7] | 71.5 [66.4, 77.6] | 78.6 [76.3, 82.1] | 79.8 [76.5, 84.9] |
| 0.02 | 17.2 [16.9, 17.6] | 16.5 [15.9, 17.5] | 16.2 [15, 17.4] | 17.8 [17.6, 17.9] | 17.5 [17.4, 17.7] | 37.6 [34.3, 40.8] | 38.2 [32.9, 43.1] | 41.3 [40.8, 42] | 40.8 [35.2, 45.7] | 40.9 [34.4, 48.1] |
| 0.04 | 14.2 [13.7, 14.6] | 13.4 [13.3, 13.6] | 16.4 [14.3, 19] | 14.6 [14.3, 15.1] | 14.2 [14, 14.4] | 21.8 [20.7, 22.6] | 22.3 [21.1, 23.5] | 21.9 [20.1, 22.8] | 23 [21.4, 24.1] | 22.5 [21.7, 23.8] |
| 0.08 | 18.5 [18.4, 18.6] | 18 [17, 18.8] | 19.1 [18.9, 19.5] | 18.3 [17.9, 18.6] | 18 [17.1, 18.5] | 17.3 [17.1, 17.6] | 18 [17.4, 18.4] | 19.3 [19.2, 19.3] | 17.4 [17.2, 17.6] | 17.9 [17.1, 18.8] |

### Plot 6 (context): R_e(H - f_dt), T = 1.28

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 9.22 [8.21, 9.72] | 8.03 [7.08, 9.04] | 11 [9.73, 12.3] | 8.59 [8.15, 9.11] | 8.98 [8.7, 9.4] | 64.2 [62.3, 65.5] | 65.2 [48.5, 78.1] | 62.1 [52.5, 70.3] | 68.5 [67.1, 70.5] | 69.3 [65.2, 75.4] |
| 0.02 | 7.4 [7.03, 8.12] | 8.37 [7.14, 9.38] | 9.93 [8.85, 10.6] | 7.7 [7.24, 8.07] | 7.84 [7.2, 8.48] | 28.9 [24.9, 32.8] | 28.9 [23.8, 33.9] | 32.8 [31.8, 33.4] | 30.9 [25.2, 34.3] | 31.2 [24.7, 38.7] |
| 0.04 | 8.01 [7.52, 8.68] | 6.03 [5.59, 6.33] | 8.87 [8.24, 9.88] | 7.62 [6.94, 8.72] | 8.19 [6.95, 9.34] | 13.6 [12.4, 14.6] | 14.6 [13.1, 16.2] | 15.4 [13.9, 16.9] | 14.7 [13.3, 16.5] | 14.7 [13.4, 16.6] |
| 0.08 | 5.68 [5.27, 6.34] | 5.2 [4.82, 5.89] | 7.28 [7.19, 7.46] | 6.35 [5.77, 7.15] | 6.19 [5.86, 6.69] | 6.02 [5.44, 6.68] | 7.48 [6.92, 7.86] | 7.65 [7, 8.21] | 6.73 [6.65, 6.81] | 7.16 [6.36, 7.57] |

### M6: spectral norm of M (upper bound on sigma_e), T = 1.28

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 6.73e+03 [6.65e+03, 6.81e+03] | 6.82e+03 [6.73e+03, 6.91e+03] | 7.83e+03 [7.78e+03, 7.95e+03] | 6.62e+03 [6.57e+03, 6.71e+03] | 6.81e+03 [6.7e+03, 6.94e+03] | 7.07e+04 [6.94e+04, 7.25e+04] | 6.93e+04 [6.73e+04, 7.19e+04] | 7.78e+04 [7.37e+04, 8.24e+04] | 7.02e+04 [6.87e+04, 7.18e+04] | 6.71e+04 [6.56e+04, 6.83e+04] |
| 0.02 | 4.73e+03 [4.65e+03, 4.82e+03] | 4.7e+03 [4.67e+03, 4.74e+03] | 5.68e+03 [5.67e+03, 5.72e+03] | 4.65e+03 [4.6e+03, 4.68e+03] | 5e+03 [4.77e+03, 5.43e+03] | 2.48e+04 [2.38e+04, 2.53e+04] | 2.63e+04 [2.57e+04, 2.74e+04] | 3.35e+04 [3.17e+04, 3.6e+04] | 2.41e+04 [2.31e+04, 2.48e+04] | 2.32e+04 [2.24e+04, 2.41e+04] |
| 0.04 | 3.22e+03 [3.21e+03, 3.23e+03] | 3.3e+03 [3.28e+03, 3.35e+03] | 3.89e+03 [3.85e+03, 3.94e+03] | 3.2e+03 [3.19e+03, 3.2e+03] | 3.3e+03 [3.14e+03, 3.41e+03] | 8.47e+03 [8.07e+03, 8.76e+03] | 9.57e+03 [9.09e+03, 9.83e+03] | 1.17e+04 [1.12e+04, 1.19e+04] | 8.29e+03 [8e+03, 8.44e+03] | 7.95e+03 [7.76e+03, 8.31e+03] |
| 0.08 | 2e+03 [1.98e+03, 2.02e+03] | 2.19e+03 [2.16e+03, 2.2e+03] | 2.72e+03 [2.65e+03, 2.79e+03] | 1.99e+03 [1.97e+03, 2e+03] | 1.97e+03 [1.94e+03, 2.01e+03] | 3.11e+03 [3.1e+03, 3.13e+03] | 3.44e+03 [3.23e+03, 3.59e+03] | 4.16e+03 [3.97e+03, 4.3e+03] | 3e+03 [2.95e+03, 3.05e+03] | 2.85e+03 [2.71e+03, 2.99e+03] |

### M6: largest row norm of M (lower bound on sigma_e), T = 1.28

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 119 [113, 123] | 135 [130, 139] | 156 [140, 171] | 118 [115, 121] | 119 [118, 120] | 867 [832, 900] | 926 [715, 1.27e+03] | 947 [867, 994] | 886 [836, 943] | 914 [846, 976] |
| 0.02 | 118 [115, 121] | 128 [125, 132] | 147 [142, 155] | 116 [111, 120] | 121 [121, 122] | 383 [368, 406] | 404 [379, 439] | 582 [549, 619] | 393 [383, 400] | 413 [384, 445] |
| 0.04 | 121 [120, 122] | 124 [120, 128] | 160 [140, 173] | 120 [119, 122] | 120 [118, 122] | 185 [182, 189] | 242 [237, 246] | 321 [306, 336] | 185 [168, 209] | 190 [178, 210] |
| 0.08 | 115 [113, 118] | 132 [130, 134] | 160 [147, 171] | 117 [114, 120] | 115 [107, 126] | 110 [107, 114] | 151 [133, 170] | 192 [183, 199] | 109 [107, 111] | 115 [113, 117] |

### Plot 7: lambda_max(K^(p)) at T = 0.08, 0.16, 0.32, 0.64, 1.28 (seed means)

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 461 / 1.48e+03 / 1.17e+04 / 1.36e+06 / 2.16e+07 | 501 / 1.67e+03 / 6.15e+03 / 5.58e+05 / 1.91e+07 | 1.06e+03 / 3.35e+03 / 9.5e+03 / 2.88e+05 / 2.17e+07 | 460 / 1.49e+03 / 1.15e+04 / 1.31e+06 / 2.3e+07 | 471 / 1.52e+03 / 1.11e+04 / 1.22e+06 / 2.46e+07 | 9.17e+04 / 3.57e+05 / 5.33e+06 / 1.08e+09 / 3.38e+09 | 1.14e+05 / 4.19e+05 / 1.96e+06 / 3.96e+08 / 2.88e+10 | 2.14e+05 / 7.66e+05 / 2.37e+06 / 2.85e+08 / 5.89e+09 | 9.24e+04 / 3.55e+05 / 5.05e+06 / 1.1e+09 / 2e+09 | 9.52e+04 / 3.62e+05 / 4.87e+06 / 9.92e+08 / 2.37e+09 |
| 0.02 | 476 / 1.48e+03 / 1.13e+04 / 1.25e+06 / 2.18e+07 | 521 / 1.67e+03 / 5.96e+03 / 5.07e+05 / 1.91e+07 | 1.17e+03 / 3.48e+03 / 9.4e+03 / 2.79e+05 / 2.1e+07 | 476 / 1.49e+03 / 1.12e+04 / 1.21e+06 / 2.31e+07 | 487 / 1.54e+03 / 1.08e+04 / 1.13e+06 / 2.5e+07 | 2.2e+04 / 8.64e+04 / 1.31e+06 / 2.85e+08 / 5.76e+08 | 3.32e+04 / 1.18e+05 / 4.2e+05 / 7.65e+07 / 4.55e+08 | 7.41e+04 / 2.64e+05 / 7.62e+05 / 5.94e+07 / 1.23e+09 | 2.22e+04 / 8.55e+04 / 1.24e+06 / 2.55e+08 / 4.26e+08 | 2.29e+04 / 8.66e+04 / 1.17e+06 / 2.26e+08 / 4.27e+08 |
| 0.04 | 490 / 1.49e+03 / 1.04e+04 / 1.1e+06 / 2.32e+07 | 561 / 1.7e+03 / 5.71e+03 / 4.72e+05 / 1.81e+07 | 1.29e+03 / 3.61e+03 / 9.04e+03 / 2.48e+05 / 2.06e+07 | 490 / 1.5e+03 / 1.03e+04 / 1.07e+06 / 2.36e+07 | 500 / 1.54e+03 / 9.99e+03 / 1e+06 / 2.42e+07 | 4.64e+03 / 1.7e+04 / 2.52e+05 / 5.02e+07 / 1.01e+08 | 7.77e+03 / 2.66e+04 / 9.12e+04 / 1.69e+07 / 1.32e+08 | 1.51e+04 / 5.3e+04 / 1.56e+05 / 9.25e+06 / 1.67e+08 | 4.7e+03 / 1.7e+04 / 2.4e+05 / 4.7e+07 / 8.93e+07 | 4.85e+03 / 1.74e+04 / 2.25e+05 / 4.28e+07 / 8.44e+07 |
| 0.08 | 490 / 1.36e+03 / 7.44e+03 / 8.04e+05 / 2.49e+07 | 680 / 1.74e+03 / 5.45e+03 / 3.94e+05 / 1.98e+07 | 1.54e+03 / 3.87e+03 / 8.01e+03 / 2.19e+05 / 1.97e+07 | 490 / 1.36e+03 / 7.37e+03 / 7.72e+05 / 2.24e+07 | 495 / 1.38e+03 / 7.21e+03 / 7.18e+05 / 2.05e+07 | 1.1e+03 / 3.41e+03 / 4.69e+04 / 8.29e+06 / 1.83e+07 | 1.68e+03 / 4.94e+03 / 1.65e+04 / 2.85e+06 / 2.05e+07 | 3.01e+03 / 9.37e+03 / 2.67e+04 / 2.08e+06 / 3.24e+07 | 1.11e+03 / 3.45e+03 / 4.52e+04 / 8.04e+06 / 1.87e+07 | 1.14e+03 / 3.54e+03 / 4.33e+04 / 7.65e+06 / 1.88e+07 |

### M7: tr K / lambda_max at T = 0.08, 0.16, 0.32, 0.64, 1.28 (seed means)

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 5.49 / 4.99 / 2.53 / 1.47 / 1.49 | 5.94 / 4.98 / 4 / 2 / 1.69 | 4.75 / 4.22 / 3.51 / 2.77 / 1.8 | 5.52 / 5 / 2.55 / 1.48 / 1.49 | 5.54 / 5.02 / 2.63 / 1.51 / 1.47 | 4.58 / 3.95 / 1.58 / 1.07 / 1.42 | 3.75 / 3.16 / 2.47 / 1.13 / 1.18 | 3.6 / 3.05 / 2.85 / 1.2 / 1.32 | 4.52 / 3.92 / 1.6 / 1.06 / 1.77 | 4.44 / 3.87 / 1.62 / 1.07 / 1.88 |
| 0.02 | 5.59 / 5.18 / 2.62 / 1.52 / 1.53 | 5.7 / 4.87 / 3.92 / 2.03 / 1.7 | 4.49 / 4.14 / 3.5 / 2.8 / 1.78 | 5.62 / 5.18 / 2.65 / 1.54 / 1.52 | 5.63 / 5.12 / 2.73 / 1.56 / 1.51 | 4.83 / 3.99 / 1.57 / 1.06 / 1.65 | 4.17 / 3.4 / 3.01 / 1.19 / 1.72 | 3.88 / 3.16 / 2.83 / 1.32 / 1.59 | 4.76 / 3.99 / 1.59 / 1.06 / 1.81 | 4.67 / 3.94 / 1.62 / 1.07 / 1.99 |
| 0.04 | 5.83 / 5.32 / 2.83 / 1.63 / 1.56 | 5.86 / 5.06 / 4.09 / 2.1 / 1.76 | 4.46 / 4.21 / 3.65 / 2.98 / 1.93 | 5.85 / 5.33 / 2.85 / 1.65 / 1.56 | 5.85 / 5.27 / 2.92 / 1.67 / 1.56 | 5.45 / 4.49 / 1.65 / 1.07 / 1.85 | 4.84 / 3.78 / 3.23 / 1.2 / 1.64 | 4.56 / 3.49 / 2.88 / 1.45 / 1.92 | 5.39 / 4.46 / 1.67 / 1.07 / 1.96 | 5.28 / 4.36 / 1.72 / 1.08 / 1.99 |
| 0.08 | 6.38 / 5.75 / 3.33 / 1.74 / 1.54 | 6.34 / 5.74 / 4.35 / 2.28 / 1.8 | 4.3 / 4.05 / 3.75 / 2.96 / 1.85 | 6.38 / 5.76 / 3.34 / 1.76 / 1.58 | 6.38 / 5.71 / 3.39 / 1.79 / 1.63 | 6.76 / 5.56 / 1.79 / 1.09 / 2.08 | 6.02 / 4.73 / 3.57 / 1.23 / 1.96 | 6.03 / 4.5 / 3.41 / 1.4 / 1.82 | 6.71 / 5.5 / 1.82 / 1.09 / 2.04 | 6.62 / 5.39 / 1.85 / 1.09 / 2.07 |

### M7: |P^T ehat| / |ehat| at T = 0.08, 0.16, 0.32, 0.64, 1.28 (seed means)

| dt | euler / one-step | euler / L_p, T=0.32 (scratch) | euler / L_p, T=0.64 (scratch) | euler / L_p, T=0.32 (fine-tuned) | euler / L_p, T=0.64 (fine-tuned) | direct / one-step | direct / L_p, T=0.32 (scratch) | direct / L_p, T=0.64 (scratch) | direct / L_p, T=0.32 (fine-tuned) | direct / L_p, T=0.64 (fine-tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 2.78 / 5.11 / 27.8 / 794 / 2.44e+03 | 2.75 / 4.55 / 18.8 / 757 / 2.22e+03 | 2.9 / 4.78 / 15.9 / 846 / 1.62e+03 | 2.73 / 4.59 / 20.9 / 913 / 2.56e+03 | 2.66 / 4.41 / 19.5 / 1.14e+03 / 2.14e+03 | 3.04 / 6.77 / 44.2 / 1.05e+03 / 2.77e+03 | 2.8 / 4.63 / 18 / 716 / 1.16e+04 | 2.46 / 3.95 / 10.7 / 356 / 3.7e+03 | 2.95 / 5.49 / 32.2 / 1.09e+03 / 1.36e+03 | 3.14 / 4.52 / 18.4 / 992 / 1.87e+03 |
| 0.02 | 1.97 / 3.44 / 18.9 / 634 / 1.72e+03 | 1.97 / 3.22 / 10.5 / 370 / 1.02e+03 | 2.09 / 3.45 / 10.4 / 639 / 1.28e+03 | 1.97 / 3.26 / 11.9 / 514 / 1.8e+03 | 1.88 / 2.82 / 7.96 / 356 / 1.3e+03 | 2.02 / 3.88 / 25.2 / 754 / 1.76e+03 | 1.91 / 3.2 / 11.4 / 405 / 1.01e+03 | 1.83 / 2.89 / 7.48 / 397 / 2.35e+03 | 2 / 3.38 / 17.5 / 629 / 1.32e+03 | 2.13 / 3.17 / 14.2 / 571 / 1.12e+03 |
| 0.04 | 1.39 / 2.22 / 12.8 / 483 / 959 | 1.39 / 2.12 / 8.96 / 441 / 1.03e+03 | 1.46 / 2.27 / 7.28 / 357 / 1.05e+03 | 1.39 / 2.22 / 8.98 / 359 / 738 | 1.35 / 1.97 / 7.25 / 394 / 1.09e+03 | 1.43 / 2.52 / 13.3 / 458 / 1.24e+03 | 1.39 / 2.13 / 6.64 / 352 / 1.34e+03 | 1.37 / 1.96 / 6.04 / 248 / 1.05e+03 | 1.42 / 2.28 / 10.4 / 362 / 1.12e+03 | 1.45 / 2.26 / 10.1 / 390 / 870 |
| 0.08 | 1 / 1.47 / 7.7 / 279 / 993 | 1 / 1.41 / 4.84 / 125 / 932 | 1 / 1.46 / 3.81 / 126 / 486 | 1 / 1.46 / 5.73 / 244 / 980 | 1 / 1.49 / 3.76 / 143 / 666 | 1 / 1.5 / 7.5 / 224 / 707 | 1 / 1.41 / 4.48 / 205 / 697 | 1 / 1.42 / 3.75 / 161 / 577 | 1 / 1.43 / 4.56 / 179 / 705 | 1 / 1.46 / 4.66 / 251 / 679 |


Diverged (non-finite) model checkpoints: 0
---

# Follow-up tables (generated by followup_report.py)

Rollout error |e(T)| on 4096 held-out starts, final checkpoints, min-max over 3 seeds. Saturated-error scale (RMS distance of independent attractor states): 21.2.

### dt = 0.01

| run | protocol | class | T=0.32 | T=0.64 | T=1.28 | T=2.56 | T=5.12 |
|---|---|---|---|---|---|---|---|
| dt0.01_onestep | one-step | euler | 0.0035-0.0041 | 0.014-0.027 | 0.31-0.44 | 0.51-0.77 | 3.1-3.5 |
| dt0.01_onestep | one-step | direct | 0.021-0.032 | 0.15-0.21 | 0.76-1.4 | 2.4-2.9 | 7.4-8.6 |
| dt0.01_unrollp16_ft | window 0.16 (p=16), fine-tuned from one-step | euler | 0.0028-0.0029 | 0.011-0.015 | 0.17-0.25 | 0.53-0.79 | 2.8-2.9 |
| dt0.01_unrollp16_ft | window 0.16 (p=16), fine-tuned from one-step | direct | 0.014-0.022 | 0.08-0.14 | 0.66-1.3 | 1.9-2.5 | 6.4-8.1 |
| dt0.01_unroll032_ft | window 0.32 (p=32), fine-tuned from one-step | euler | 0.0027-0.0028 | 0.0089-0.01 | 0.14-0.22 | 0.59-0.75 | 2.3-2.7 |
| dt0.01_unroll032_ft | window 0.32 (p=32), fine-tuned from one-step | direct | 0.015-0.028 | 0.088-0.17 | 0.35-0.38 | 1.9-2.5 | 6.8-8.4 |
| dt0.01_unroll032_scratch | window 0.32 (p=32), scratch | euler | 0.0029-0.0034 | 0.011-0.014 | 0.095-0.17 | 0.59-0.88 | 2.4-3 |
| dt0.01_unroll032_scratch | window 0.32 (p=32), scratch | direct | 0.045-0.05 | 0.2-0.25 | 1.4-1.5 | 2.8-3.7 | 11-12 |
| dt0.01_unroll064_ft | window 0.64 (p=64), fine-tuned from one-step | euler | 0.0046-0.0048 | 0.011-0.012 | 0.12-0.18 | 0.61-1.1 | 3.1-3.8 |
| dt0.01_unroll064_ft | window 0.64 (p=64), fine-tuned from one-step | direct | 0.049-0.051 | 0.14-0.24 | 1.3-1.6 | 3.3-3.6 | 10-11 |
| dt0.01_unroll064_scratch | window 0.64 (p=64), scratch | euler | 0.011-0.012 | 0.015-0.018 | 0.21-0.33 | 0.94-1.3 | 4.2-4.5 |
| dt0.01_unroll064_scratch | window 0.64 (p=64), scratch | direct | 0.14-0.16 | 0.22-0.25 | 2.5-2.8 | 5.2-5.6 | 15-16 |

### dt = 0.02

| run | protocol | class | T=0.32 | T=0.64 | T=1.28 | T=2.56 | T=5.12 |
|---|---|---|---|---|---|---|---|
| dt0.02_onestep | one-step | euler | 0.0035-0.004 | 0.018-0.026 | 0.32-0.37 | 0.73-0.78 | 3.2-3.6 |
| dt0.02_onestep | one-step | direct | 0.01-0.013 | 0.048-0.1 | 0.53-0.77 | 1.5-1.9 | 5.2-5.9 |
| dt0.02_unroll032_ft | window 0.32 (p=16), fine-tuned from one-step | euler | 0.0025-0.0028 | 0.0094-0.011 | 0.13-0.24 | 0.58-0.84 | 2.5-3 |
| dt0.02_unroll032_ft | window 0.32 (p=16), fine-tuned from one-step | direct | 0.0082-0.0089 | 0.042-0.045 | 0.21-0.41 | 0.98-1.1 | 3.8-4.7 |
| dt0.02_unroll032_scratch | window 0.32 (p=16), scratch | euler | 0.0029-0.0033 | 0.0093-0.014 | 0.08-0.16 | 0.54-0.88 | 3.1-3.7 |
| dt0.02_unroll032_scratch | window 0.32 (p=16), scratch | direct | 0.019-0.025 | 0.083-0.12 | 0.41-0.77 | 1.5-2 | 6.4-7.6 |
| dt0.02_unroll064_ft | window 0.64 (p=32), fine-tuned from one-step | euler | 0.0042-0.0047 | 0.0065-0.0081 | 0.069-0.094 | 0.62-0.67 | 2.8-3.1 |
| dt0.02_unroll064_ft | window 0.64 (p=32), fine-tuned from one-step | direct | 0.021-0.023 | 0.063-0.078 | 0.64-0.73 | 2.1-2.2 | 7.5-7.8 |
| dt0.02_unroll064_scratch | window 0.64 (p=32), scratch | euler | 0.012-0.012 | 0.016-0.02 | 0.2-0.4 | 1.1-1.3 | 4.4-5 |
| dt0.02_unroll064_scratch | window 0.64 (p=32), scratch | direct | 0.066-0.085 | 0.098-0.13 | 1.1-2.4 | 3.2-3.9 | 11-12 |

### dt = 0.04

| run | protocol | class | T=0.32 | T=0.64 | T=1.28 | T=2.56 | T=5.12 |
|---|---|---|---|---|---|---|---|
| dt0.04_onestep | one-step | euler | 0.003-0.0034 | 0.013-0.017 | 0.14-0.19 | 0.64-1.2 | 3-3.8 |
| dt0.04_onestep | one-step | direct | 0.0052-0.0065 | 0.025-0.049 | 0.2-0.3 | 0.72-1.1 | 3.7-4.4 |
| dt0.04_unroll032_ft | window 0.32 (p=8), fine-tuned from one-step | euler | 0.0022-0.0026 | 0.0078-0.011 | 0.092-0.14 | 0.59-1 | 2.8-3.4 |
| dt0.04_unroll032_ft | window 0.32 (p=8), fine-tuned from one-step | direct | 0.0041-0.0046 | 0.018-0.028 | 0.13-0.2 | 0.74-0.99 | 3.4-4.1 |
| dt0.04_unroll032_scratch | window 0.32 (p=8), scratch | euler | 0.0027-0.003 | 0.011-0.011 | 0.13-0.18 | 0.58-0.75 | 3.1-3.3 |
| dt0.04_unroll032_scratch | window 0.32 (p=8), scratch | direct | 0.012-0.014 | 0.051-0.059 | 0.39-0.68 | 1.4-2 | 4.9-6.4 |
| dt0.04_unroll064_ft | window 0.64 (p=16), fine-tuned from one-step | euler | 0.004-0.0042 | 0.0084-0.013 | 0.096-0.31 | 0.61-1.4 | 3.1-5 |
| dt0.04_unroll064_ft | window 0.64 (p=16), fine-tuned from one-step | direct | 0.0096-0.013 | 0.019-0.06 | 0.25-0.68 | 1.1-2.4 | 4.7-7.1 |
| dt0.04_unroll064_ft_cur | window 0.64 (p=16), fine-tuned from dt0.04_unroll032_ft, lr 1e-05 | euler | 0.0023-0.0026 | 0.0071-0.0096 | 0.062-0.15 | 0.59-0.88 | 2.5-3.5 |
| dt0.04_unroll064_ft_cur | window 0.64 (p=16), fine-tuned from dt0.04_unroll032_ft, lr 1e-05 | direct | 0.0053-0.0059 | 0.017-0.035 | 0.21-0.34 | 0.92-1.9 | 3.6-6.2 |
| dt0.04_unroll064_scratch | window 0.64 (p=16), scratch | euler | 0.011-0.013 | 0.017-0.02 | 0.22-0.4 | 1.3-1.5 | 4.6-4.8 |
| dt0.04_unroll064_scratch | window 0.64 (p=16), scratch | direct | 0.035-0.037 | 0.047-0.05 | 0.66-0.77 | 1.9-2.5 | 6.8-8.8 |
| dt0.04_unroll128_ft_cur | window 1.28 (p=32), fine-tuned from dt0.04_unroll064_ft_cur, lr 1e-06 | euler | 0.0034-0.0075 | 0.015-0.057 | 0.18-0.43 | 0.78-2.2 | 3.6-8.1 |
| dt0.04_unroll128_ft_cur | window 1.28 (p=32), fine-tuned from dt0.04_unroll064_ft_cur, lr 1e-06 | direct | 0.0059-0.0082 | 0.025-0.037 | 0.22-0.57 | 0.75-1.8 | 3.8-6.1 |
| dt0.04_unroll128g9_ft | window 1.28 (p=32), fine-tuned from one-step, lr 1e-05, damped gamma=9 | euler | 0.0034-0.0042 | 0.015-0.019 | 0.22-0.32 | 0.63-1.7 | 3-6.2 |
| dt0.04_unroll128g9_ft | window 1.28 (p=32), fine-tuned from one-step, lr 1e-05, damped gamma=9 | direct | 0.0051-0.0059 | 0.017-0.023 | 0.16-0.32 | 1.1-1.4 | 4-5 |
| dt0.04_unrollp32_ft | window 1.28 (p=32), fine-tuned from one-step | euler | 0.08-0.083 | 0.11-0.11 | 0.23-0.4 | 2.8-3.1 | 10-10 |
| dt0.04_unrollp32_ft | window 1.28 (p=32), fine-tuned from one-step | direct | 0.18-0.2 | 0.22-0.24 | 0.47-0.58 | 5.1-5.2 | 14-14 |
| dt0.04_unrollp32_ft_lr1e-6 | window 1.28 (p=32), fine-tuned from one-step, lr 1e-06 | euler | 0.0044-0.0054 | 0.017-0.021 | 0.14-0.27 | 1.3-1.5 | 4.4-5.2 |
| dt0.04_unrollp32_ft_lr1e-6 | window 1.28 (p=32), fine-tuned from one-step, lr 1e-06 | direct | 0.0088-0.011 | 0.037-0.058 | 0.33-0.37 | 1.3-1.7 | 5.3-6.7 |

### dt = 0.08

| run | protocol | class | T=0.32 | T=0.64 | T=1.28 | T=2.56 | T=5.12 |
|---|---|---|---|---|---|---|---|
| dt0.08_onestep | one-step | euler | 0.0026-0.003 | 0.012-0.013 | 0.12-0.23 | 0.59-0.98 | 3.2-4.1 |
| dt0.08_onestep | one-step | direct | 0.0036-0.004 | 0.0098-0.021 | 0.11-0.24 | 0.85-0.89 | 3.5-3.8 |
| dt0.08_unroll032_ft | window 0.32 (p=4), fine-tuned from one-step | euler | 0.002-0.0022 | 0.0077-0.0097 | 0.1-0.13 | 0.52-0.65 | 2.2-2.5 |
| dt0.08_unroll032_ft | window 0.32 (p=4), fine-tuned from one-step | direct | 0.0029-0.0034 | 0.0086-0.012 | 0.078-0.2 | 0.67-1.1 | 2.9-3.9 |
| dt0.08_unroll032_scratch | window 0.32 (p=4), scratch | euler | 0.0026-0.003 | 0.0077-0.01 | 0.097-0.22 | 0.57-1.1 | 3-3.4 |
| dt0.08_unroll032_scratch | window 0.32 (p=4), scratch | direct | 0.0066-0.0076 | 0.021-0.029 | 0.17-0.29 | 1.2-1.4 | 4.4-5 |
| dt0.08_unroll064_ft | window 0.64 (p=8), fine-tuned from one-step | euler | 0.0035-0.0037 | 0.0065-0.0081 | 0.1-0.14 | 0.61-0.85 | 2.6-3.5 |
| dt0.08_unroll064_ft | window 0.64 (p=8), fine-tuned from one-step | direct | 0.0057-0.0059 | 0.011-0.015 | 0.13-0.13 | 0.77-1.1 | 3.7-4.8 |
| dt0.08_unroll064_ft_cur | window 0.64 (p=8), fine-tuned from dt0.08_unroll032_ft, lr 1e-05 | euler | 0.0021-0.0023 | 0.0052-0.0081 | 0.063-0.09 | 0.46-0.59 | 2.2-3.4 |
| dt0.08_unroll064_ft_cur | window 0.64 (p=8), fine-tuned from dt0.08_unroll032_ft, lr 1e-05 | direct | 0.0032-0.0035 | 0.0065-0.0093 | 0.098-0.13 | 0.56-0.8 | 2.9-3.9 |
| dt0.08_unroll064_scratch | window 0.64 (p=8), scratch | euler | 0.013-0.013 | 0.017-0.02 | 0.22-0.27 | 1-1.6 | 4.7-5.1 |
| dt0.08_unroll064_scratch | window 0.64 (p=8), scratch | direct | 0.02-0.023 | 0.024-0.031 | 0.34-0.47 | 1.5-2 | 6-6.9 |
| dt0.08_unroll128_ft_cur | window 1.28 (p=16), fine-tuned from dt0.08_unroll064_ft_cur, lr 1e-06 | euler | 0.0029-0.0035 | 0.01-0.016 | 0.095-0.17 | 0.52-1 | 3-3.4 |
| dt0.08_unroll128_ft_cur | window 1.28 (p=16), fine-tuned from dt0.08_unroll064_ft_cur, lr 1e-06 | direct | 0.0037-0.0046 | 0.0097-0.014 | 0.091-0.15 | 0.66-1.5 | 3.1-5.2 |
| dt0.08_unroll128g9_ft | window 1.28 (p=16), fine-tuned from one-step, lr 1e-05, damped gamma=9 | euler | 0.0028-0.0032 | 0.011-0.015 | 0.15-0.19 | 0.5-0.87 | 3.5-3.9 |
| dt0.08_unroll128g9_ft | window 1.28 (p=16), fine-tuned from one-step, lr 1e-05, damped gamma=9 | direct | 0.0042-0.0045 | 0.015-0.02 | 0.075-0.34 | 0.6-0.97 | 3.4-4.6 |
| dt0.08_unrollp16_ft | window 1.28 (p=16), fine-tuned from one-step | euler | 0.075-0.081 | 0.091-0.099 | 0.17-0.25 | 2.7-3.3 | 8.9-9.7 |
| dt0.08_unrollp16_ft | window 1.28 (p=16), fine-tuned from one-step | direct | 0.12-0.12 | 0.14-0.14 | 0.31-0.57 | 4-4.2 | 11-12 |
| dt0.08_unrollp16_ft_lr1e-6 | window 1.28 (p=16), fine-tuned from one-step, lr 1e-06 | euler | 0.0035-0.0043 | 0.015-0.016 | 0.14-0.26 | 0.69-1.5 | 3.2-5.7 |
| dt0.08_unrollp16_ft_lr1e-6 | window 1.28 (p=16), fine-tuned from one-step, lr 1e-06 | direct | 0.0051-0.0059 | 0.016-0.034 | 0.22-0.46 | 1.1-1.3 | 3.8-4.6 |
| dt0.08_unrollp32_ft | window 2.56 (p=32), fine-tuned from one-step | euler | 0.17-0.2 | 0.16-0.17 | 0.25-0.47 | 0.94-1.4 | 5.1-6.3 |
| dt0.08_unrollp32_ft | window 2.56 (p=32), fine-tuned from one-step | direct | 0.35-0.38 | 0.34-0.36 | 0.43-0.49 | 1.2-1.6 | 5.7-7 |
| dt0.08_unrollp32_ft_lr1e-6 | window 2.56 (p=32), fine-tuned from one-step, lr 1e-06 | euler | 0.0042-0.0068 | 0.014-0.043 | 0.19-0.41 | 0.59-1 | 3.1-4.7 |
| dt0.08_unrollp32_ft_lr1e-6 | window 2.56 (p=32), fine-tuned from one-step, lr 1e-06 | direct | 0.0069-0.012 | 0.019-0.085 | 0.2-0.87 | 0.74-2.2 | 3.9-7.4 |

## Trainability: curvature of each unrolled run's own loss

lam0 = largest Gauss-Newton eigenvalue of the run's training loss (its window and weights) at the start of training, on 512 held-out starts; descended = (own loss at the end) / (at the start) < 1. Fine-tuned runs only (scratch runs start from a random initialization).

| run | protocol | class | lam0 (median) | lr x lam0 (median) | end/start per seed | largest spike per seed |
|---|---|---|---|---|---|---|
| dt0.01_unrollp16_ft | window 0.16 (p=16), fine-tuned from one-step | euler | 5.7e+02 | 0.057 | 0.81, 0.87, 0.9 | x24, x11, x16 |
| dt0.01_unrollp16_ft | window 0.16 (p=16), fine-tuned from one-step | direct | 1.3e+05 | 13 | 0.95, 0.88, 0.85 | x69, x83, x1.2e+02 |
| dt0.01_unroll032_ft | window 0.32 (p=32), fine-tuned from one-step | euler | 2.5e+03 | 0.25 | 0.62, 0.71, 0.81 | x17, x1.4e+02, x17 |
| dt0.01_unroll032_ft | window 0.32 (p=32), fine-tuned from one-step | direct | 1e+06 | 1e+02 | 0.76, 0.62, 0.96 | x4.7e+02, x1.9e+02, x1.6e+02 |
| dt0.02_unroll032_ft | window 0.32 (p=16), fine-tuned from one-step | euler | 2.7e+03 | 0.27 | 0.63, 0.65, 0.75 | x1.2e+02, x91, x6.8e+02 |
| dt0.02_unroll032_ft | window 0.32 (p=16), fine-tuned from one-step | direct | 2.8e+05 | 28 | 0.83, 0.75, 0.59 | x5.4e+02, x2.1e+02, x2.1e+02 |
| dt0.04_unroll032_ft | window 0.32 (p=8), fine-tuned from one-step | euler | 2.7e+03 | 0.27 | 0.69, 0.72, 0.7 | x81, x3.2e+02, x34 |
| dt0.04_unroll032_ft | window 0.32 (p=8), fine-tuned from one-step | direct | 6.2e+04 | 6.2 | 0.84, 0.81, 0.71 | x3.7e+02, x2.3e+02, x2.7e+02 |
| dt0.08_unroll032_ft | window 0.32 (p=4), fine-tuned from one-step | euler | 2.6e+03 | 0.26 | 0.76, 0.8, 0.72 | x2.8e+02, x56, x4.7e+02 |
| dt0.08_unroll032_ft | window 0.32 (p=4), fine-tuned from one-step | direct | 1.5e+04 | 1.5 | 0.72, 0.74, 0.85 | x1.4e+02, x1.4e+02, x2.6e+02 |
| dt0.01_unroll064_ft | window 0.64 (p=64), fine-tuned from one-step | euler | 1e+05 | 10 | 0.19, 0.82, 1.3 | x2.6e+02, x1.7e+03, x2.3e+03 |
| dt0.01_unroll064_ft | window 0.64 (p=64), fine-tuned from one-step | direct | 5.5e+07 | 5.5e+03 | 1.7, 0.51, 1.3 | x1.1e+03, x6e+02, x1.4e+03 |
| dt0.02_unroll064_ft | window 0.64 (p=32), fine-tuned from one-step | euler | 1.1e+05 | 11 | 0.32, 0.19, 0.52 | x9.8e+02, x1e+03, x7.7e+02 |
| dt0.02_unroll064_ft | window 0.64 (p=32), fine-tuned from one-step | direct | 1.5e+07 | 1.5e+03 | 2.5, 0.41, 0.97 | x8.2e+03, x1.1e+03, x2.7e+03 |
| dt0.04_unroll064_ft | window 0.64 (p=16), fine-tuned from one-step | euler | 1e+05 | 10 | 0.31, 0.54, 0.55 | x4.9e+02, x3.3e+03, x1.8e+03 |
| dt0.04_unroll064_ft | window 0.64 (p=16), fine-tuned from one-step | direct | 3.2e+06 | 3.2e+02 | 1.2, 0.95, 2 | x6.1e+03, x3.1e+03, x1.2e+03 |
| dt0.04_unroll064_ft_cur | window 0.64 (p=16), fine-tuned from dt0.04_unroll032_ft, lr 1e-05 | euler | 1e+05 | 1 | 1.5, 0.67, 0.58 | x70, x51, x86 |
| dt0.04_unroll064_ft_cur | window 0.64 (p=16), fine-tuned from dt0.04_unroll032_ft, lr 1e-05 | direct | 3e+06 | 30 | 1.8, 2.4, 2.1 | x1.2e+02, x1e+02, x1.3e+02 |
| dt0.08_unroll064_ft | window 0.64 (p=8), fine-tuned from one-step | euler | 9.8e+04 | 9.8 | 0.45, 0.45, 0.31 | x7.5e+02, x8.7e+02, x3.8e+02 |
| dt0.08_unroll064_ft | window 0.64 (p=8), fine-tuned from one-step | direct | 7.2e+05 | 72 | 0.61, 0.22, 3.6 | x2.3e+03, x4.8e+03, x1.7e+04 |
| dt0.08_unroll064_ft_cur | window 0.64 (p=8), fine-tuned from dt0.08_unroll032_ft, lr 1e-05 | euler | 9.7e+04 | 0.97 | 0.77, 0.57, 0.39 | x49, x45, x20 |
| dt0.08_unroll064_ft_cur | window 0.64 (p=8), fine-tuned from dt0.08_unroll032_ft, lr 1e-05 | direct | 6.9e+05 | 6.9 | 0.95, 0.39, 1.3 | x1.1e+02, x24, x1.5e+02 |
| dt0.04_unroll128_ft_cur | window 1.28 (p=32), fine-tuned from dt0.04_unroll064_ft_cur, lr 1e-06 | euler | 4.5e+07 | 45 | 0.4, 7.8, 1.1e+02 | x5, x17, x1.1e+02 |
| dt0.04_unroll128_ft_cur | window 1.28 (p=32), fine-tuned from dt0.04_unroll064_ft_cur, lr 1e-06 | direct | 1.5e+08 | 1.5e+02 | 5.9, 0.63, 1.3 | x5.9, x2.5, x8.2 |
| dt0.04_unroll128g9_ft | window 1.28 (p=32), fine-tuned from one-step, lr 1e-05, damped gamma=9 | euler | 3.2e+04 | 0.32 | 1.2, 1.3, 1.5 | x64, x31, x22 |
| dt0.04_unroll128g9_ft | window 1.28 (p=32), fine-tuned from one-step, lr 1e-05, damped gamma=9 | direct | 3.6e+05 | 3.6 | 0.67, 0.42, 0.28 | x53, x31, x35 |
| dt0.04_unrollp32_ft | window 1.28 (p=32), fine-tuned from one-step | euler | 4e+07 | 4e+03 | 2.5, 9.4, 14 | x1.3e+03, x2.9e+03, x2.5e+03 |
| dt0.04_unrollp32_ft | window 1.28 (p=32), fine-tuned from one-step | direct | 1.8e+08 | 1.8e+04 | 9.1, 7.7, 6.5 | x2.6e+03, x8.3e+02, x8.5e+02 |
| dt0.04_unrollp32_ft_lr1e-6 | window 1.28 (p=32), fine-tuned from one-step, lr 1e-06 | euler | 4e+07 | 40 | 0.79, 6.5, 2.5 | x3.9, x6.5, x2.5 |
| dt0.04_unrollp32_ft_lr1e-6 | window 1.28 (p=32), fine-tuned from one-step, lr 1e-06 | direct | 1.8e+08 | 1.8e+02 | 1.1, 1.3, 1.1 | x9, x5.6, x1.8 |
| dt0.08_unroll128_ft_cur | window 1.28 (p=16), fine-tuned from dt0.08_unroll064_ft_cur, lr 1e-06 | euler | 4.1e+07 | 41 | 3.7, 10, 1.4 | x20, x16, x9.7 |
| dt0.08_unroll128_ft_cur | window 1.28 (p=16), fine-tuned from dt0.08_unroll064_ft_cur, lr 1e-06 | direct | 4.5e+07 | 45 | 2.4, 4.1, 1.2 | x78, x12, x12 |
| dt0.08_unroll128g9_ft | window 1.28 (p=16), fine-tuned from one-step, lr 1e-05, damped gamma=9 | euler | 3.2e+04 | 0.32 | 1.3, 0.9, 0.81 | x28, x1.1e+02, x47 |
| dt0.08_unroll128g9_ft | window 1.28 (p=16), fine-tuned from one-step, lr 1e-05, damped gamma=9 | direct | 8.6e+04 | 0.86 | 0.71, 0.74, 1.9 | x24, x36, x1.2e+02 |
| dt0.08_unrollp16_ft | window 1.28 (p=16), fine-tuned from one-step | euler | 4.1e+07 | 4.1e+03 | 3.3, 7.9, 1.3 | x2.1e+03, x3.7e+03, x6e+02 |
| dt0.08_unrollp16_ft | window 1.28 (p=16), fine-tuned from one-step | direct | 4.6e+07 | 4.6e+03 | 4.5, 12, 11 | x5.6e+02, x3.7e+03, x1.8e+03 |
| dt0.08_unrollp16_ft_lr1e-6 | window 1.28 (p=16), fine-tuned from one-step, lr 1e-06 | euler | 4.1e+07 | 41 | 1.9, 2.2, 0.49 | x3, x14, x1 |
| dt0.08_unrollp16_ft_lr1e-6 | window 1.28 (p=16), fine-tuned from one-step, lr 1e-06 | direct | 4.6e+07 | 46 | 2, 3.8, 1.4 | x2, x8.5, x4.9 |
| dt0.08_unrollp32_ft | window 2.56 (p=32), fine-tuned from one-step | euler | 6.5e+09 | 6.5e+05 | 5, 16, 0.35 | x3.3e+02, x3.1e+02, x35 |
| dt0.08_unrollp32_ft | window 2.56 (p=32), fine-tuned from one-step | direct | 2.1e+09 | 2.1e+05 | 0.49, 4.6, 7.1 | x37, x5.2e+02, x1e+03 |
| dt0.08_unrollp32_ft_lr1e-6 | window 2.56 (p=32), fine-tuned from one-step, lr 1e-06 | euler | 6.5e+09 | 6.5e+03 | 6.5, 9.7, 0.043 | x12, x15, x1 |
| dt0.08_unrollp32_ft_lr1e-6 | window 2.56 (p=32), fine-tuned from one-step, lr 1e-06 | direct | 2.1e+09 | 2.1e+03 | 0.66, 8.1, 28 | x1.2, x22, x28 |

Best single threshold separating runs that descended (fine-tuned, all classes): lr x lam0: accuracy 0.79 at 27.8; lam0: accuracy 0.83 at 1.1e+06; window: accuracy 0.79 at 0.64 (n = 126, 0.55 descended).
