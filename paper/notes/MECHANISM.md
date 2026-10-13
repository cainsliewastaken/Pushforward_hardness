# Why the sign push works: the batch drive's sampling noise in the kernel basis (2026-10-13)

Written after the measurements of REPORT 61 items 21-26 and a rereading of THEORY.md sections 10, 13, 21 and the
one-time theorems. Status: a derivation from the paper's objects (Theorem 2's per-sample outer products, Corollary 24a's
batch drive) plus ONE assumption about the sampling, stated in section 2 and checked against the measured per-decile
signal-to-noise. It accounts for the ranking of the rules on both targets, the failure of the exact ridge (Krylov), the
learned per-mode weights' shape, and the role of the cap. What it does not account for is listed in section 7.

## 0. Setting (paper's objects)

Population residual E in R^N (N = n ny; e_k = <v_k, E> its coordinates in the kernel eigenbasis), kernel K = J J^T / n with
eigenpairs (lambda_k, v_k), leverage s_k = sqrt(n lambda_k), parameter direction of mode k: u_k = J^T v_k / s_k (unit).
Per-sample gradient contribution (Theorem 2, data-indexed form): z_x = J_x^T E_x, so the population gradient is
g = (1/n) sum_x z_x and a batch gradient is g_B = (1/n_B) sum_{x in B} z_x (Corollary 24a: the batch drive). A step d
changes the population residual to first order by J d, with coordinates s_k d_k on v_k (d_k = <u_k, d>).

## 1. The signal and the noise of the batch drive, mode by mode

Project z_x on u_k: <u_k, z_x> = <J_x u_k, E_x> = s_k <v_k(x), E_x>, where v_k(x) in R^ny is the part of the eigenvector
on sample x (so sum_x |v_k(x)|^2 = 1). Hence the coordinate of the batch gradient on mode k is

    g_B,k = s_k (1/n_B) sum_{x in B} p_k(x),      p_k(x) = <v_k(x), E_x>  (the mode's per-sample projection of the residual),

with population mean (over x) m_k = e_k / n and per-sample variance tau_k^2 = Var_x p_k(x). Under sampling without
structure (batches drawn uniformly, fresh each step, as in the harness):

    E[g_B,k] = s_k e_k / n,        Var[g_B,k] = s_k^2 tau_k^2 / n_B,        SNR_k = E / sqrt(Var) = (e_k / n) sqrt(n_B) / tau_k.

LEMMA 1 (noise follows leverage). The sampling noise of the batch gradient on mode k has standard deviation proportional
to the mode's leverage s_k, and its signal is proportional to s_k e_k; the signal-to-noise ratio SNR_k = e_k sqrt(n_B) /
(n tau_k) does not contain lambda_k.  Proof: the two displays.  []

ASSUMPTION (S). The per-sample spread tau_k of the mode projections is of one size across modes, tau_k ~ tau (the
eigenvectors v_k are delocalised over samples, |v_k(x)|^2 ~ 1/n, and the per-sample residual is not aligned with any
one of them). Under (S), SNR_k ~ e_k sqrt(n_B) / (n tau): FLAT in lambda_k and proportional to the residual coordinate.
Check (REPORT 61 item 21, table C; half-batch SNR per lambda decile in the Kronecker basis, both layers): rough at
step 250: 1.17-1.35 across all ten deciles; smooth: 1.02-1.05 on deciles 1-9 and 1.8-2.3 on the top decile, where the
smooth residual's coordinates are largest; rough at step 250 the slow input modes carry SNR ~ 0.8 (item 24, sig 0.41)
where the held-rough residual's coordinates are largest. Both targets agree with SNR flat in lambda and rising with e_k.
Per-batch SNR at n_B = 256 is ~ 0.2 on a typical mode (item 24: persistent fraction 1-3 %).

## 2. The three rules' expected output response under the noise

Write the step on mode k as d_k and the first-order population decrease of the loss as (2/N) sum_k e_k s_k d_k. Take the
noise Gaussian per mode (a sum of n_B per-sample terms), sigma_k = s_k tau / sqrt(n_B).

(GD / Adam-in-coordinates, magnitude rule)  d_k = eta g_B,k:
    E[s_k d_k] = eta lambda_k' e_k      (lambda_k' = s_k^2 / n),     Var[s_k d_k] = eta^2 s_k^2 sigma_k^2 = eta^2 s_k^4 tau^2 / n_B.
    The output response is weighted by lambda on the signal and by lambda^2 on the noise: the noise sits on the fast modes.
(exact ridge, gamma -> 0: the Krylov / Gauss-Newton direction, scale rho)  d_k = rho g_B,k / lambda_k':
    E[s_k d_k] = rho e_k,                Var[s_k d_k] = rho^2 n tau^2 / n_B   for EVERY mode.
    The output noise is flat across all N modes: total noise energy N rho^2 n tau^2 / n_B against signal energy rho^2 ||E||^2.
(sign push, per-mode length delta)  d_k = delta sign(g_B,k):
    E[s_k d_k] = s_k delta erf(SNR_k / sqrt 2) ~ 0.8 s_k delta SNR_k = 0.8 delta s_k e_k sqrt(n_B) / (n tau)   (small SNR),
    E[(s_k d_k)^2] = s_k^2 delta^2   exactly, whatever the noise.
    The output response has signal weighted by s_k e_k x SNR_k ~ s_k e_k^2 and noise energy weighted by lambda_k: the
    noise follows the kernel's own spectrum and is BOUNDED per mode (|d_k| = delta), it cannot concentrate on the fast
    modes (GD) nor spread at full strength over all modes (ridge).

PROPOSITION 2 (expected first-order gain per unit of output change, the quantity the budget fixes). With the output
change ||J d|| held at the same value for the three rules (the budget), the expected first-order population decrease is
proportional to the alignment

    A_GD   = sum_k lambda_k' e_k^2                / sqrt( sum_k lambda_k'^2 (e_k^2 + n tau^2 / n_B) ),
    A_ridge = ||E||^2                              / sqrt( ||E||^2 + N n tau^2 / n_B ),
    A_sign = (0.8 sqrt(n_B) / (n tau)) sum_k s_k e_k^2 / sqrt( sum_k lambda_k' ) ,

each divided by ||E||. Proof: ratio of the expected signal response <E, E[J d]> to the root expected output energy, with
the per-mode moments above; cross-mode noise terms vanish in expectation.  []

Reading. (a) The ridge is defeated by the N in its denominator: it inverts the leverage, so every one of the N modes
carries output noise n tau^2 / n_B, and N n tau^2 / n_B >> ||E||^2 at batch 256 (item 17: the exact-K Krylov step's
alignment 0.06-0.17 and its CG residual ~ ||E||: the step was noise). (b) GD is defeated by lambda'^2 on the fast modes'
noise, and its signal is lambda-weighted, which is nothing on a rough residual (AdamW's held-out alignment 0.01-0.07,
item 22). (c) The sign push's denominator is the trace of the kernel, independent of the residual and of the noise,
and its numerator weights mode k by s_k e_k^2: compared with GD's lambda_k' e_k^2 = s_k^2 e_k^2 / n it favours mode k by
n / s_k, i.e. the LOW-leverage modes with large residual coordinates, which is where a rough target's residual sits
(the held-rough part). On a smooth target the residual sits on the top modes, the three numerators agree up to constants,
and the rules tie there up to their noise handling: measured, sign 1.41e-4, Adam rule 2.2e-4, SOAP 1.1-1.4e-4, uniform sign
2.5e-4 (items 13, 22). On rough: sign 0.061, Adam rule 0.111, uniform sign 0.089-0.12, AdamW 0.39, Krylov 0.48-0.56.

## 3. The optimal per-mode push, and what the learned weight is

Maximise the expected gain sum_k e_k s_k delta_k erf(SNR_k / sqrt 2) subject to the output energy sum_k lambda_k' delta_k^2
<= R^2 (the budget, in expectation). KKT: delta_k ~ e_k erf(SNR_k / sqrt 2) / s_k ~ e_k^2 / (s_k tau) sqrt(n_B) / n for
small SNR: the push should be largest on modes with a large residual coordinate and small leverage, and small on the top
modes. The one-step hypergradient's expected update on the log weight is proportional to g_k x E[D_k] ~ s_k e_k x erf(SNR_k)
~ s_k e_k^2: the same ranking. Measured (item 19): learned weights 2-3x on the slow input modes early in a rough run
(where e_k is large and s_k small), 0.9 on the top decile, flat ~1 on the hidden layer where the residual coordinates
are spread. The learned weight is the online estimate of e_k erf(SNR_k) / s_k, which nothing else in the optimizer
measures; that is why it is worth 1.5-1.8x (items 13, 22) and why no rule built from the gradient's sign statistics alone
(items 23, 26) replaced it: those saw SNR_k but not e_k / s_k.

## 4. The cap

On the top modes s_k delta exceeds the completion length e_k early (large leverage, small residual coordinate), and the
sign push overshoots: the measured period-4 oscillation on the top decile (item 24, R(2) = -0.1 to -0.5). The cap
d_k <= g_k / lambda_k is the completion step there (Kronecker-priced, hence imperfect); it is worth 1.3x (item 13), and
the learned weight's 0.9 on the top decile is the hypergradient doing the same job slowly.

## 5. The budget

The budget fixes ||J d|| on the batch at r ||E_B||. Under the noise the population step is at its break-even point
(item 22: gn_v = |first_v| within 5-10 % for every rule): the population first-order gain equals the quadratic cost of
the population output change, so the budget is the largest step whose expected population loss change is not positive.
Its constant is therefore set by the ratio of population signal to total output change, i.e. by the noise level, and
rises with n_B (item 22: held-out / batch gain ratio 0.3-0.4 at 256, 0.5-0.6 at 1024). On the rough target this length
coincides with the output-space Taylor radius (item 15, 0.19 ||E||); the two limits happen to agree there and the
remainder is 6 % (item 21), so the first-order model is not what binds. On smooth the Taylor radius is far beyond and
the budget is purely the noise limit (a damping).

## 6. Predictions and the record

  * SNR flat in lambda, proportional to the residual coordinate: measured (section 1).
  * Ranking at equal output change on a rough residual: sign > Kronecker-ridge-like rules > magnitude rules > exact
    ridge: measured (items 17, 21, 22).
  * Smooth residual: the rules tie up to noise handling: measured (SOAP = sign = full KronQ within 1.3x).
  * The learned weight's shape e_k erf(SNR_k) / s_k: measured in direction (item 19); its size not compared.
  * Noise reduction helps sign but not GD: batch 1024 gave sign 1.24x per step and AdamW nothing (item 22). The model
    predicts sign's expected gain ~ sqrt(n_B) at fixed delta (2x for 4x batch), and the budget then moves the operating
    point; 1.24x is below that, which is the first quantitative gap (section 7).
  * Full factors vs blocks (fork, item 25): the basis changes which modes are "fast" and how well lambda_k is priced;
    the mechanism above is basis-independent, the cap and the SNR per mode are not.

## 7. What this does not explain (the open list for the paper)

UPDATE after the open measurements (REPORT 61 item 27, 2026-10-13 14:00): Lemma 1 and assumption (S) are confirmed
directly (tau_k flat within 1.5x over ten deciles; per-batch SNR 0.6-0.9 on every decile). Item 2 below is RESOLVED
(budget re-set to the batch's break-even: 4x batch = 2.35x per step). Section 3's per-mode optimum is REFUTED (the
derived weight diverges on rough; the learned profile is flat and better). Item 1 is SHARPENED: with cap and learned
weights on both, the sign push beats the Adam rule 1.8x on rough and 1.15x on smooth; at momentum SNR ~3 both directions
are nearly deterministic, so the difference is the magnitude profile across modes (flat vs ~SNR-dependent), not noise
handling. Section 2's account of why magnitude rules and the exact ridge lose stands; its account of why the sign in
particular wins does not, and that is the one open question before the paper.

  1. Sign push vs the Adam rule in the same basis, 1.45x on rough: both have |d_k| bounded and E[d_k] ~ SNR_k; the model
     puts them at parity. Candidate: the Adam rule's magnitude m / sqrt(v) carries the noise of v (SNR ~ 0.2 makes v
     noise-dominated), so its effective per-mode length fluctuates by O(1) step to step while the sign's is fixed.
  2. The size of the batch-size effect (1.24x measured vs ~2x predicted at fixed delta).
  3. The output-space Taylor radius on the rough target (why 0.19 ||E||, and why it is isotropic in output space).
  4. The late phase (the end-of-run degradation seen in some variants).
  5. Assumption (S) is checked at the decile level only; the per-mode tau_k (the spread of the mode's per-sample
     projection) has not been measured directly, and the Gaussian noise per mode is assumed.
Each of 1, 2 and 5 is one measurement with the existing diagnostics (mechanism.py: per-mode tau_k from the per-sample
projections; the per-step length of the Adam rule's effective step).

## 8. Where it goes in the paper

Section 21's "measurements organised by the identities" becomes: the kernel's eigenbasis (Theorem 2), the batch drive
(Corollary 24a), the sampling lemma (Lemma 1 + (S)), the three alignments (Proposition 2), the per-mode optimum
(section 3) = the step rule, the cap (one-step theorem on the fast modes), the budget (break-even under noise). The
per-mode noise lemma is the "statistics of {g_B} over the sampling" that Corollary 24a left undone, and it is short.
