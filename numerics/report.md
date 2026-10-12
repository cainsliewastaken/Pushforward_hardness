# Numerical checks: hardness theory for learned pushforward operators

Code, CSVs and figures are in `numerics/`. Run on Perlmutter, Shifter image `nersc/pytorch:26.01.01`, 1 A100 (interactive QOS, account m5326).

| Part | Script | Output | Wall time |
|---|---|---|---|
| **R** Replication of the earlier paper (Overleaf `old_files/rough_slow_kernel.tex`) | `replicate_rsk.py` | `results_rsk/` | 6.2 min |
| 1 Unit tests of the one-step results | `test_units.py` | `results/unit_*.csv` | 5 s |
| 2 Sine targets | `sine.py`, `plot_sine_FM.py` | `results/sine_*` | 85 s |
| 3 Lorenz-63 direct vs Euler | `lorenz.py` | `results/lorenz_*` | 36 min |
| Tables | `make_report.py` | `results/summary.md` (full pass/fail table) | — |

Total compute was about 45 min of one GPU, under the 2 h budget. Nothing had to be shrunk.

---

## R. Replication of the earlier paper's experiments

The script follows the paper's Appendix "Experimental details": same networks, data sizes, optimizers, step counts, thresholds and rough subspaces. Random draws cannot be the same (different seeds and code), so agreement is statistical. Where I had to guess a detail, it is listed under "Differences" below.

### Table `c` (increment correlation of Lorenz-63)

| h | c (old) | c (new) | Eq. (c-G) old / new | κ old / new | 2nd-order old / new |
|---|---|---|---|---|---|
| 0.005 | 0.9977 | 0.9977 | 0.9977 / 0.9976 | 1.998 / 1.998 | 0.9977 / 0.9977 |
| 0.01 | 0.9907 | 0.9907 | 0.9907 / 0.9907 | 1.993 / 1.993 | 0.9907 / 0.9907 |
| 0.02 | 0.9635 | 0.9634 | 0.9635 / 0.9634 | 1.973 / 1.973 | 0.9627 / 0.9627 |
| 0.05 | 0.7946 | 0.7948 | 0.7946 / 0.7947 | 1.844 / 1.844 | 0.7669 / 0.7670 |
| 0.1 | 0.3947 | 0.3951 | 0.3946 / 0.3953 | 1.480 / 1.481 | 0.0674 / 0.0678 |
| 0.2 | −0.2045 | −0.2053 | −0.2045 / −0.2044 | 0.670 / 0.670 | −2.730 / −2.729 |
| 0.3 | −0.5441 | −0.5465 | −0.5430 / −0.5460 | −0.130 / −0.139 | −7.393 / −7.390 |

τ_v = 0.0732 (old 0.073). **Replicated** to 3–4 digits.

### Table `floor` (sine targets, 300000 GD steps, λc = 0.3 λ₁, high-pass k = 8, m = 24; means over 3 inits)

| Target | ‖V‖ old/new | W(V)^½ old/new | W(f)^½ old/new | floor old/new | slow error old/new |
|---|---|---|---|---|---|
| sin x | 0.515 / 0.528 | 0.006 / 0.006 | 0.006 / 0.006 | 0.000 / 0.000 | 0.000 / 0.001 |
| sin 5x | 0.755 / 0.731 | 0.027 / 0.026 | 0.027 / 0.026 | 0.000 / 0.000 | 0.012 / 0.013 |
| sin 10x | 0.700 / 0.688 | 0.056 / 0.058 | 0.059 / 0.059 | −0.003 / −0.001 | 0.013 / 0.013 |
| sin 20x | 0.685 / 0.695 | 0.155 / 0.123 | 0.135 / 0.081 | 0.019 / 0.042 | **0.267 / 0.415** |
| sin 40x | 0.689 / 0.700 | 0.484 / 0.486 | 0.029 / 0.029 | 0.455 / 0.457 | 0.648 / 0.670 |
| sin 60x | 0.691 / 0.704 | 0.623 / 0.657 | 0.019 / 0.014 | 0.604 / 0.643 | 0.654 / 0.681 |
| sin 90x | 0.706 / 0.726 | 0.677 / 0.701 | 0.024 / 0.012 | 0.653 / 0.690 | 0.688 / 0.713 |
| sin 120x | 0.684 / 0.686 | 0.647 / 0.651 | 0.027 / 0.015 | 0.620 / 0.636 | 0.660 / 0.663 |
| random | 1.064 / 0.970 | 0.989 / 0.916 | 0.017 / 0.027 | 0.972 / 0.889 | 1.035 / 0.934 |

Other claims in the old text, checked against the new run:

| Claim | Old | New |
|---|---|---|
| Floor below slow error at all 81 checkpoints | yes | yes |
| Budget identity Σλᵢ W(vᵢ) = ω² agreement | 1.6e-14 | ≤ 3.7e-15 |
| ‖P_f P_A‖ | ≤ 0.015 | ≤ 0.0121 |
| σ/√λc | 0.024–0.055 | 0.019–0.049 |
| ω/√λc | 0.030–0.089 | 0.024–0.076 |
| Leak term at 300000 steps | < 1e-9 | < 6e-10 |
| m = 24 is the smallest grid value with σ/√λc ≤ 0.1 in all 27 runs | yes | yes (max 0.077 at m = 24, 0.131 at m = 16) |

The differences in ‖V‖, W(V) and the random labels come from the random inputs and labels. The one qualitative difference is **sin 20x**. Its training is still in transition at 300000 steps:

| sin 20x error after | 3000 steps | 30000 steps | 300000 steps |
|---|---|---|---|
| old | 0.66 | 0.62 | 0.27 |
| new | 0.68 | 0.63 | 0.41 |

The new runs are slower at this target. Its floor is still below the slow error in every run. For sin 10x the old trajectory 0.44 → 0.12 → 0.013 matches the new one (0.43 → 0.11 → 0.013).

### Table `ceiling` (bound on the slow target, 300000 steps)

| Target | limit old/new | m = 24 old/new | best m old/new | best high-pass bound old/new | best cluster (k, p) old/new | best cluster bound old/new |
|---|---|---|---|---|---|---|
| sin x | 0.076 / 0.080 | −0.007 / −0.004 | 96 / 96 | −0.002 / −0.001 | (5,2) / (5,2) | −0.000 / −0.000 |
| sin 5x | 0.753 / 0.723 | 0.026 / 0.025 | 2 / 2 | 0.716 / 0.660 | (100,0) / (100,0) | 0.704 / 0.641 |
| sin 10x | 0.692 / 0.677 | 0.053 / 0.056 | 4 / 4 | 0.636 / 0.625 | (100,1) / (100,1) | 0.647 / 0.628 |
| sin 20x | 0.683 / 0.693 | 0.152 / 0.122 | 4 / 4 | 0.651 / 0.669 | (40,0) / (100,1) | 0.659 / 0.666 |
| sin 40x | 0.687 / 0.697 | 0.482 / 0.485 | 4 / 4 | 0.670 / 0.682 | (100,1) / (50,1) | 0.669 / 0.681 |
| sin 60x | 0.690 / 0.699 | 0.622 / 0.656 | 2 / 8 | 0.672 / 0.680 | (100,1) / (100,2) | 0.674 / 0.679 |
| sin 90x | 0.703 / 0.725 | 0.675 / 0.700 | 4 / 8 | 0.686 / 0.716 | (50,1) / (100,2) | 0.690 / 0.714 |
| sin 120x | 0.683 / 0.684 | 0.645 / 0.649 | 4 / 8 | 0.674 / 0.667 | (100,2) / (50,1) | 0.673 / 0.665 |
| random | 1.050 / 0.966 | 0.983 / 0.914 | 8 / 8 | 1.014 / 0.946 | (50,2) / (100,2) | 1.021 / 0.945 |

Pattern **replicated**:
- The limit equals the slow target.
- The m = 24 bound is small for smooth targets.
- The best bound uses a small m and lies within 0.01–0.06 of the limit (new: 0.009–0.063).

The best-m and best-cluster choices differ in a few rows. The bounds there are within about 0.01 of each other.

### Tables `targets`, `steppers`, `split` (Lorenz-63; medians over 5 inits, widths 32 and 128)

Targets. Direct size is 14.7–14.95 (old 14.6–14.9), and Euler size matches at every h (e.g. 1.105 vs 1.13, 10.35 vs 10.27, 21.45 vs 21.29).

Roughness at m = 300:

| | old | new |
|---|---|---|
| Identity | 0.33–0.35 | 0.32–0.35 |
| Direct target at h = 0.01 | 0.33 | 0.33 |
| Euler target at h = 0.01 | 0.04 | 0.040 |
| Euler rougher than direct | from h = 0.15 | from h = 0.15 |
| Both targets at h ≥ 0.5 | 5.7–7.1 | 5.8–6.8 |

**Replicated.**

| h | width | val. error direct old/new | val. error Euler old/new | slow target direct old/new | slow target Euler old/new | Mann–Whitney p (new) |
|---|---|---|---|---|---|---|
| 0.01 | 32 | 0.035 / 0.037 | 0.009 / 0.009 | 0.186 / 0.195 | 0.045 / 0.046 | 0.008 |
| 0.01 | 128 | 0.025 / 0.025 | 0.009 / 0.008 | 0.149 / 0.155 | 0.048 / 0.043 | 0.008 |
| 0.02 | 32 | 0.041 / 0.045 | 0.020 / 0.019 | 0.197 / 0.208 | 0.085 / 0.092 | 0.008 |
| 0.02 | 128 | 0.035 / 0.030 | 0.019 / 0.016 | 0.169 / 0.166 | 0.097 / 0.087 | 0.008 |
| 0.03 | 32 | 0.044 / 0.047 | 0.026 / 0.027 | 0.217 / 0.223 | 0.133 / 0.132 | 0.008 |
| 0.03 | 128 | 0.035 / 0.034 | 0.025 / 0.022 | 0.192 / 0.196 | 0.146 / 0.136 | 0.008 |
| 0.05 | 32 | 0.053 / 0.052 | 0.039 / 0.046 | 0.256 / 0.273 | 0.207 / 0.219 | 0.008 |
| 0.05 | 128 | 0.040 / 0.042 | 0.039 / 0.037 | 0.248 / 0.237 | 0.235 / 0.208 | **0.008 (old 0.22)** |
| 0.07 | 32 | 0.064 / 0.059 | 0.060 / 0.053 | 0.318 / 0.319 | 0.289 / 0.306 | 0.31 |
| 0.07 | 128 | 0.054 / 0.045 | 0.051 / 0.042 | 0.300 / 0.303 | 0.310 / 0.298 | 0.42 |
| 0.1 | 32 | 0.087 / 0.075 | 0.082 / 0.080 | 0.382 / 0.397 | 0.402 / 0.411 | 0.22 |
| 0.1 | 128 | 0.064 / 0.051 | 0.065 / 0.052 | 0.382 / 0.363 | 0.401 / 0.388 | 0.55 |
| 0.15 | 32 | 0.126 / 0.120 | 0.139 / 0.129 | 0.411 / 0.500 | 0.536 / 0.599 | 0.095 |
| 0.15 | 128 | 0.091 / 0.070 | 0.098 / 0.076 | 0.437 / 0.385 | 0.545 / 0.476 | 0.31 |
| 0.2 | 32 | 0.202 / 0.205 | 0.219 / 0.225 | 0.544 / 0.573 | 0.710 / 0.744 | 0.69 |
| 0.2 | 128 | 0.122 / 0.113 | 0.131 / 0.120 | 0.489 / 0.456 | 0.620 / 0.579 | 0.42 |
| 0.3 | 32 | 0.472 / 0.471 | 0.541 / 0.517 | 1.049 / 0.960 | 1.158 / 1.094 | 0.42 |
| 0.3 | 128 | 0.282 / 0.257 | 0.279 / 0.269 | 0.888 / 0.734 | 0.873 / 0.785 | 0.69 |
| 0.5 | 32 | 2.984 / 2.913 | 3.081 / 3.091 | 3.921 / 3.663 | 3.975 / 3.888 | 0.15 |
| 0.5 | 128 | 2.118 / 2.206 | 2.135 / 2.279 | 3.296 / 3.109 | 3.371 / 3.122 | 0.31 |
| 0.75 | 32 | 8.079 / 8.315 | 8.070 / 8.193 | 8.541 / 8.480 | 8.499 / 8.489 | 0.31 |
| 0.75 | 128 | 7.841 / 7.809 | 7.830 / 7.994 | 8.469 / 8.294 | 8.492 / 8.315 | 0.31 |

The old paper's conclusion is **replicated**:
- For h ≤ 0.03 the Euler stepper is better at both widths, with disjoint ranges (p = 0.008).
- For h ≥ 0.07 the two steppers cannot be distinguished (all p ≥ 0.095).
- Nowhere is the direct stepper significantly better.

One difference: at **h = 0.05, width 128** the Euler advantage is significant here (p = 0.008, disjoint ranges) but was not in the old run (p = 0.22). With 5 inits each, 0.008 is the smallest p the test can give.

Table `split` (identity part, increment part and cosine for the direct network):
- The identity part is 0.13–0.23 for h ≤ 0.1 (old 0.14–0.20).
- The cosine is between −0.06 and +0.11 for h ≤ 0.02 (old −0.02 to +0.08), and turns to −0.24 … −0.72 for 0.05 ≤ h ≤ 0.3 (old −0.34 … −0.70).

**Replicated.** Full numbers are in `results_rsk/C_table_split.csv`.

Other claims in the old appendix:
- ‖P_f V‖ ≥ 98% of ‖V‖ for h ≤ 0.3: new minimum 99.8%.
- ≥ 80% at every h: new minimum 81%.
- Theorem slow (d) gives no positive bound for any rough subspace tried: new, the largest bound over m ∈ {3, 10, 30, 100, 300, 1000, 2000} and all 220 runs is −0.75.
- The training error varied by less than 0.4% over the last 500 steps: new maximum 0.64%.

### Figures

`results_rsk/fig_floor.png`, `results_rsk/fig_clusters.png` and `results_rsk/fig_stepper.png` redraw the three old figures, with panels as in the old captions. They show the same shapes as the originals.

In `fig_clusters` (a), I used m ∈ {8, 24, 64}. The old caption says "three values of m" without naming them.

### Differences from the old setup (guesses or unavoidable)

1. **Lorenz data.** The old paper used one long trajectory: 2000 time units for h ≤ 0.2, 10⁴h for larger h. I integrated an ensemble of 20 independent trajectories of the same total length (40 for Table `c`), each after a 50-unit spin-up. This samples the same invariant measure and is much faster.
2. **Details the old text doesn't state:**
   - The loss is the mean over all entries of the squared error on the scaled target.
   - First-layer weights and biases are drawn N(0, 1).
   - The two steppers share their initialization.
   - Π is taken as 1/(√v̂ + ε) at the final step. The old text says "Adam's final preconditioner", but the learning rate is 0 at the end of the cosine schedule and only ratios λ/λ₁ enter.
3. **Clipping at width 128.** Clipping was active in 1.4–20% of steps (median over inits), against 5–56% in the old run. At width 32 it is 0.7–1.3% (old 0.7–1.9%). So the width-128 runs are clipped less often than before, which the old text notes as a sensitivity.
4. **Not replicated:** the Adam minibatch check of Theorem `momentum` (Appendix "Check of Theorem momentum"), which isn't a table or figure.

---

## 1–3. Checks of the new paper's results

These follow the spec you first gave me. The full pass/fail table, with every check, the smallest margin and the worst case, is in `results/summary.md`. Condensed:

| Result checked | Cases | Result | Smallest rel. margin / largest rel. diff |
|---|---|---|---|
| Thm stepsup (d): closed form = direct min (SLSQP) = dual max | 36 | PASS | diff 3e-7 (SLSQP tolerance) / 1e-14 |
| Thm stepsup (a): b_R(s) ≤ F(s) (high-pass, random projectors, Lipschitz) | 108 | PASS | +0.22 |
| Thm stepsup (b): rank-one φ* attains F; (e) first-order bound | 35 | PASS | diff 9e-15 |
| Prop krough (a): R(u)/σ_R ≤ ‖u‖_{K+}, equality for K⁺u | 40 | PASS | +0.25 / diff 1e-15 |
| Prop krough (b): Krylov values nondecreasing, reach ‖u‖_{K+} at M = q_u | 12 | PASS | rounding level |
| Prop krough (c): CG bound Eq. (cg) | 4 | PASS | +0.005 |
| Thm stepkrylov (a)–(d): monotone, reach F at q_E, two-sided bound, rate | 66 (unit) + 729 (sine) | PASS | rounding level (≈ −2e-15) |
| Prop stepexact on a nonlinear net with real steps (averaged Jacobian; κ̄ over the ball sampled) | 54 | PASS | +3.6e-4 |
| Sine: ‖E_new‖ ≥ F(s) with the step-averaged Jacobian | 81 | PASS | −2e-14·‖E‖ (within tol.; see obs. 5) |
| Sine: slow error ≥ slow-error floor (Thm floor), every λc | 81 | PASS | +0.022 |
| Lorenz: Thm dirEustep (a) lower bound ≤ exact s_D; (c) exact s_E ≤ upper bound | 171 | PASS | +0.18 / +0.85 |
| Lorenz: Cor anytime (c) bound ≤ exact s_D | 360 | PASS | +0.012 |
| Lorenz: ‖E_D(t)‖ ≥ G(t) − ‖Φ_h − id‖ (G approximated) | 2160 | PASS | +0.040 |
| Lorenz: where 2‖Φ_h − id‖ < G(t), measured ‖E_E‖ < ‖E_D‖ | 127 | PASS | +0.62 |
| Lorenz: ‖E_E(t)‖ ≤ ‖Φ_h − id‖ for all t | 30 | 20/30 | −2e-16 (rounding at t = 0; see obs. 6) |
| Sanity: GD step = −lr·∇, quadrature identity (sine) | 162 | 146/162 | diff 4e-8 / 2e-7 (cancellation; obs. 5) |

Figures:
- `results/fig_sine_FM.png`: F_M(s)/F(s) against M. The supplementary `_x100` and `_x10000` versions use longer steps.
- `results/fig_sine_kinvM.png`: Krylov values of the error.
- `results/fig_lorenz_id_krylov.png`: ‖id‖_{K₀+,M} against M.
- `results/fig_lorenz_G.png`: G(t) and 2‖Φ_h − id‖ against the measured errors, for each h.
- `results/fig_lorenz_G0.png`: G₀(t) with the kernel held fixed.

### Observations

1. **With s equal to the actual step, the Krylov series is trivial.** A full-batch GD step at lr 0.02 lowers the error by at most about 2e-4 relative. F₁ already gives 100% of the certified decrease ‖E‖ − F(s) (to 1%) at every checkpoint, and F − F_M converges at M = 1. The series becomes informative only for much longer steps. At 10⁴ × the actual step, the median M to reach 1% is 2 and the maximum is 11, with κ up to about 6e3. The smooth targets (sin x, sin 5x, sin 10x) converge slowest.
2. **The kernel-roughness Krylov values of the identity grow without bound.**
   - At initialization, ‖id‖_{K₀+,M} rises from about 16.6 at M = 1 to 170–230 at M = 50.
   - The limit is ∞, because K₀ has rank 99. With the output layer at zero, only output-layer features are active.
   - The identity has a null-space part d₀ = 0.013–0.031. That value depends on the rank cutoff at the 1e-8 level (0.021–0.034 there) but is stable between 1e-12 and 1e-16.
3. **The direct-vs-Euler step-length conclusion doesn't hold here in the paper's convention.**
   - Theorem dirEustep (b) never applies: d₀ ≈ 0.02 is far below (1+ϱ)‖Φ_h − id‖ ≥ 1.67.
   - The sufficient condition via (a) and (c) never holds: 0 of 15 runs at ϱ = 0.5. The (c) upper bound on s_E is 14–90× larger than the exact s_E.
   - Exact step lengths:
     - In the paper's convention (c_E = h, network target f_h), the Euler step needs the *longer* step at every h: s_E ≈ 240–310 against s_D ≈ 27–47.
     - With c_E = 1 (network target x_{t+1} − x_t, unnormalized), s_D > s_E for h ≤ 0.1 (3/3 seeds) and in 1/3 at h = 0.2.
     - With normalized targets, s_E > s_D at every h.

   The conclusion therefore depends on the output scaling, as the Remark "Normalized targets" warns.
4. **The integration theorem is informative only early in training.**
   - With the Riemann-sum approximation of the path cost, 2‖Φ_h − id‖ < G(t) holds up to about step 50–300 (raw targets) or step 2–12 (normalized) for h ≤ 0.1, and never at h = 0.2.
   - After that the certified G(t) is 0 and the upper value is 0.03–0.09, so the bound is vacuous from then on.
   - Where the condition holds, the measured Euler error is smaller, in all 127 cases.
   - The measured Euler error is also smaller at the end of every run (30/30), which the theorem does not predict.
   - G₀(t) with the kernel held fixed never goes below d₀ ≈ 0.02 (Corollary (c)). That is below 2‖Φ_h − id‖, so it gives no bound either.
5. **Checks at the edge of float64 precision.**
   - At 30000 and 300000 GD steps on the sine targets, a step changes E by about 1e-8 relative. The identity E_new = E + Ψ̄g and the step = −lr·∇ sanity check then differ by up to 2e-7 relative. That's cancellation in E_new − E and in the gradient sum, not quadrature error: the 32- and 64-node averages differ by 3e-16.
   - For the same reason the main check ‖E_new‖ ≥ F̄(s) has margins of −2e-14·‖E‖ at late checkpoints (−2e-6 relative to the decrease). It passes within tolerance but is not informative there.
   - An earlier version computed F_M through eigh of QᵀKQ and gave false violations of 4e-7·‖E‖. I replaced it with an SVD of Qᵀ√K, and all Krylov checks now pass at rounding level.
6. **The Euler training loss does increase under Adam, but never above its start.**
   - Adam raises the Euler loss on 94–636 of the 6000 steps (mean over seeds), by up to 5% relative in one step.
   - It never exceeds its initial value, so the bound ‖E_E(t)‖ ≤ ‖Φ_h − id‖ used in the integration theorem holds.
   - The 10 "failures" of that check are rounding at t = 0 (−2e-16), where the two sides are equal.
7. **The slow-error floor (Thm floor) is far from vacuous for rough targets.** For sin 40x and rougher, and for random labels, the floor is 0.46–0.89, within 0.02–0.21 of the slow error. It is zero or slightly negative (no information) for sin x to sin 10x, where the network output is as rough as the target.

### Setup notes and choices for parts 1–3

- ϱ = 0.5 (0.1 also in the CSVs).
- Numerical rank cutoff at λ < 1e-12 λ₁, with sensitivity reported.
- Lorenz targets are uncentred, as in your spec. The replication above uses the old paper's centred targets.
- G(t) is computed by duality: the certified lower value comes from the residual direction of a FISTA solution, which also gives an upper value. The path cost is a Riemann sum with Jacobians at steps 0–35 (dense) and then every 50, left and right rules.
- G₀ is plotted against gradient-flow time; I did not map it onto Adam steps.
- Supplementary step lengths (100× and 10⁴× the actual step) are labelled as such.
- Nothing was shrunk from the requested sizes.
