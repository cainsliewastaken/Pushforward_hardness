# The step sequence of the sign-of-momentum rule under batch noise, and the horizon of the length rule (2026-10-13)

Written after REPORT 61 items 31 (exact held-out line search loses), 38-39 (kappa from the lag-1 reversal), 41 (slow
adaptation of the budget loses on every target). Status: a derivation from MECHANISM.md's objects (Lemma 1: the batch
drive's signal and noise per mode) and the rule as implemented (kronq_fast.py: C <- b1 C + (1 - b1) g, D = w sign(C),
|D| <= |g| / (lam lr)), with Gaussian noise and independent modes as the two assumptions. It gives the statistics of the
net output displacement over W steps, hence kappa as a function of the horizon W, and says which measurement decides.
Nothing here is fitted; the constants are b1 = 0.9 and the per-decile quantities already measured.

## 0. The question, and why one step is the wrong unit

For squared error the held-out loss after W steps is an identity in the net output change D_W = sum_t df_t:

    N (L_W - L_0) = 2 <E_0, D_W> + ||D_W||^2.                                                         (0.1)

The one-step rule (W = 1) was tested as the exact line search on a fresh batch and lost (item 31); the lag-1 two-stage
rule kappa = 1 / (2 (1 - r)) was tested as the slow adaptation and lost (item 41) while reproducing the drift of r
(item 38). On the fixed reference run the per-step quantities late in training are c_1 = cos(-E, df) ~ 0.03 and rho_1 =
||df|| / ||E|| ~ 0.05-0.1 (items 22, 38), i.e. kappa_1 = rho_1 / (2 c_1) ~ 1-1.7: by one-step accounting every late step
is at or past the break-even, yet the run improves 2.2x between steps 1000 and 5000. So the per-step accounting does not
describe the sequence. What follows is the statistics of D_W under the rule.

Identity used throughout: kappa_W := (||D_W|| / ||E||) / (2 cos(-E, D_W)) = ||D_W||^2 / (2 <-E, D_W>), so kappa_W = 1 is the
break-even of (0.1) and kappa_W = 1/2 the optimum, for a common scaling of the controllable part of the steps.

## 1. Per-mode model (MECHANISM.md section 1, plus the momentum)

Mode k of the kernel eigenbasis at the window start (leverage s_k, residual coordinate e_k). The batch gradient coordinate
at step t is

    g_t = mu_k + sigma_k xi_t,     mu_k = s_k e_k / n,   sigma_k = s_k tau / sqrt(n_B),   xi_t white N(0, 1),
    q_k := mu_k / sigma_k = e_k sqrt(n_B) / (n tau)     (the per-batch SNR of Lemma 1; measured ~ 0.1-0.2 on a typical mode,
                                                         ~ 0.8 on the slow input modes early on rough; items 21b, 24).

Whiteness of the gradient noise is measured: R_g(1) = 0.01-0.10, R_g(tau >= 2) = 0 on every decile (item 24a).

Momentum C_t = b1 C_{t-1} + (1 - b1) g_t is the EMA of a white sequence, hence Gaussian AR(1):

    E C = mu_k,   Var C = sigma_k^2 (1 - b1) / (1 + b1),   R_C(tau) = b1^tau,
    Q_k := E C / sqrt(Var C) = q_k sqrt((1 + b1) / (1 - b1)) = 4.36 q_k   (b1 = 0.9).                     (1.1)

Step on mode k (uncapped): D_t = w_k sign(C_t); its output change on v_k is -l_k sign(C_t), l_k := lr w_k s_k.

## 2. The sign of a Gaussian AR(1)

With a := Q_k the standardised mean of C:

    m_k := E sign C = erf(Q_k / sqrt 2),         Var sign C = 1 - m_k^2,                                  (2.1)
    Cov(sign C_t, sign C_{t+tau}) / (1 - m_k^2) =: R_s(tau) = (2 / pi) arcsin(b1^tau) + O(Q_k^2).            (2.2)

(2.2) is Sheppard's formula for the orthant probability of a bivariate normal with correlation b1^tau; the correction for
the mean is of the order of the signal fraction m_k^2 and is dropped for the noise-dominated modes. Numbers for b1 = 0.9:

    tau        1      2      3      5      10     20     40
    R_s(tau)   0.713  0.601  0.520  0.402  0.227  0.078  0.009

The mean sign (2.1) with the momentum gain (1.1):

    q_k         0.1    0.2    0.5    1.0
    m_k         0.34   0.62   0.97   1.00      (without momentum, erf(q / sqrt 2): 0.08  0.16  0.38  0.68)

So the momentum turns a per-batch SNR of 0.2 (signal fraction of the gradient q^2 / (1 + q^2) ~ 4 %, item 24's "1-3 %")
into a sign that is right 81 % of the time (m = 0.62): the signal fraction of the STEP, m_k^2, is ~ 0.1-0.4 on a typical
mode, ten times the gradient's. This is the mechanism by which the sign push beats the magnitude rules under noise, stated
for the sequence rather than for one step (MECHANISM.md Proposition 2 is the one-step version).

## 3. Uncapped modes: the net displacement over W steps

S_W := sum_{t=1}^W sign(C_t).  E S_W = W m_k.  Var S_W = (1 - m_k^2) W Gamma_W with

    Gamma_W := 1 + 2 sum_{tau=1}^{W-1} (1 - tau / W) R_s(tau),
    W          1     2     3     5     10    20    40    80    inf
    Gamma_W    1.00  1.71  2.35  3.46  5.56  8.11  10.4  11.8  13.3                                           (3.1)

Gamma_W is the variance inflation of the noise part of the step sequence relative to W independent steps. It is the
momentum's doing: the sign of an EMA persists for ~ 1 / (1 - b1) = 10 steps, so over a horizon of ten steps the NOISE part
of the uncapped steps adds almost coherently (Gamma_10 = 5.6) and saturates at 13 for long horizons. This is reinforcement,
not cancellation, and it is the opposite of what the lag-1 reversal r > 0 reports: that reversal belongs to the capped
modes (section 4). Per uncapped mode, over the horizon W, in output units:

    gain      <-E, D_W>_k = e_k l_k W m_k,
    cost      ||D_W||_k^2 = l_k^2 [ W^2 m_k^2 + W Gamma_W (1 - m_k^2) ].                                   (3.2)

Both the signal cost (W^2 m^2) and the noise cost (W Gamma_W) grow faster than the gain (W): the per-step cost over a
horizon W is l_k^2 [W m_k^2 + Gamma_W (1 - m_k^2)] against l_k^2 at W = 1. The one-step accounting UNDERSTATES the
uncapped modes' cost, by W m^2 + Gamma_W (1 - m^2): a factor 3-6 at W = 10 for m^2 = 0.1-0.4.

The horizon over which (3.2) holds is bounded by the coherence of the direction: the kernel moves 2-3 % per step on rough
(item 21e), so v_k rotates away in ~ 30-50 steps, and e_k itself is being removed. Beyond that D_W stops growing as W.

## 4. Capped modes: the Newton regime and the telescoping sequence

The cap D <= |g| / (lam lr) binds when the sign push would exceed the Gauss-Newton completion step; on those modes the step
IS the completion step on the batch, d_k = g_k / lam'_k (lam'_k = s_k^2 / n), with output change

    s_k d_k = n g_k / s_k = e_k + nu_t,     nu_t := (n tau / sqrt(n_B)) xi_t    (flat across modes: MECHANISM section 2, ridge).

The step removes the population coordinate e_k (one shot) and leaves the batch noise nu_t as the new residual coordinate;
the next completion step removes that and leaves nu_{t+1}:

    df_t,k = -(e_k 1[t = 1]) - nu_t + nu_{t-1}      (nu_0 = 0),                                               (4.1)

a differenced white sequence. Its lag-1 reversal in the plan's definition is -<df_{t+1}, df_t> / ||df_t||^2 = 1/2 in the
limit, and the sum TELESCOPES: D_W,k = -e_k - nu_W, so

    gain      <-E, D_W>_k = e_k^2  (once),          cost      ||D_W||_k^2 = e_k^2 + nu^2   for every W >= 1.           (4.2)

The capped modes' cost over a horizon is that of ONE step whatever W; the one-step accounting OVERSTATES their cost by W.
Measured: r_fast on the true kernel's top-32 directions rises to 0.3-0.48 late on rough (items 38, 41), approaching the
1/2 of (4.1); the sign of the step comes from C while the cap magnitude comes from g, so the sign lags and the telescoping
is incomplete, which is why 0.45 and not 0.5. On smooth r_fast is negative: there the top-32 modes are not capped and
section 3 applies to them too.

## 5. kappa over the horizon, and what decides it

Summing (3.2) and (4.2) over modes (independence assumed), with U the uncapped and Cp the capped set:

    kappa_W = [ sum_U l_k^2 (W m_k^2 + Gamma_W (1 - m_k^2)) + (1 / W) sum_Cp (e_k^2 + nu^2) ]
              / [ 2 sum_U e_k l_k m_k + (2 / W) sum_Cp e_k^2 ].                                                 (5.1)

The two parts move in opposite directions with W. Whether kappa_W falls or rises from kappa_1 is decided by one measurable
quantity: the share of the step's output energy sum lam D^2 that sits on capped modes (the per-decile profile prof_eshare
and the cap-active fraction capfrac, both in the --diag output). Prediction for the fixed reference run, which improves
at kappa_1 ~ 1-1.7: its output energy must sit mostly on capped, telescoping modes, so that kappa_W drops below 1 within
a few steps; then the per-step rule and the lag-1 rule both misprice the steps and the slow adaptation of item 41 cut the
steps to half their useful length for no reason. If instead the energy sits on uncapped modes, kappa_W RISES with W
(as it did in the width-128 CPU check: 0.22 -> 0.58 from W = 1 to 10, where nothing is capped) and the improvement at
kappa_1 > 1 is not explained by this model.

The rule that follows, if (5.1) is confirmed: control the window-level kappa at the coherence horizon W* (10-20 steps, the
momentum's persistence and the kernel's coherence time) rather than the per-step kappa_1 or the lag-1 r. Only the uncapped
part scales with the lr / budget, so budget_new = budget x m / kappa_W* applies to that part; the capped part is set by the
batch noise nu and the batch size, not by the length. The per-step c_t of item 39 and the r of item 38 are then diagnostics
of the two mode classes, not control variables.

## 6. What the model leaves out, and how each shows up in the measurement

(a) Mode coupling on the slow deciles (item 25c, 10^3-10^6): the per-mode attribution of gain and cost fails there. The
    output-space sums in (0.1) and the measured kappa_W do not depend on the basis; only the prediction (5.1) does.
(b) Sign from C, magnitude from g on capped modes: incomplete telescoping (r 0.45 vs 0.5).
(c) Learned weights w_k move at rate 0.01 per step: constant over a 40-step window.
(d) Gaussian noise (sum of 256 per-sample terms): fine. The first-order df_t = J d_t on the fresh batch: remainder 0.2-6 %.
(e) Independence across modes for the variance of ||D_W||^2: the cross terms between modes average out only if the noise
    is uncorrelated across modes, which Lemma 1's per-sample structure does not guarantee for correlated eigenvectors.

## 7. The measurement (scale_measure.sh, --sa_measure 1 --diag 1; fixed reference, rough / smooth field, coordinates raw)

Per window of 40 steps every 250: kappa_W for W = 1, 2, 5, 10, 20, 40 from the net first-order displacement on a fixed
fresh batch (sa_kappaW*), per-step c and lag-1 r, the batch / held-out output-change ratio (the budget acts on the batch
change; (0.1) is held-out), the lag-1 sign agreement of consecutive steps per lambda decile (tests R_s(1) = 0.71 on
uncapped modes against the telescoping sign on capped ones), the per-decile output-energy share and cap-active fraction.
Decides between the two cases of section 5 and gives W*.

## 8. Outcome of the measurement (REPORT 61 item 42, job 59761481)

Section 2 is confirmed exactly: the lag-1 sign agreement of consecutive steps is 0.65-0.75 on every decile, layer, target
and stage (0.71 predicted). Section 4 is moot in the reference configuration: the cap is active on <= 0.3 % of modes and
carries <= 8 % of the step's output energy. Section 5 is refuted in both of its cases: kappa_W = 1.00 +- 0.05 for every W
from 1 to 40, from step 500 to 5000, on both field targets, while the step length falls 100x under the lr decay. The net
output displacement is additive (||D_W||^2 ~ W): the per-mode coherence of section 3 does not reach the output, the mode
coupling (REPORT 61 item 25 (fork) c) decorrelates the responses, and the horizon is not the variable. What the flat
kappa_W = 1 does say: the run holds gain = cost on held-out data at every horizon and every length, keeping 1-2 % of its
gross gain as progress, and the alignment of its steps is proportional to their length. A length rule that takes c as a
property of the direction cannot work, which is why items 31, 39 and 41 all lost. The candidate controller that holds
kappa at 1 is the learned per-mode weight (the one-step hypergradient drives <g_{t+1}, D_t> to zero per mode, the
one-step optimum along the mode); the test is the same measurement with uniform weights (--hyper 0).
