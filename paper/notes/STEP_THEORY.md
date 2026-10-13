# What the theory says about the optimizer's step (2026-10-12, evening)

Written while the ablation of the fast-path configuration ran (optimizer/scale_ablate.sh). Purpose: go over the
rules in the current algorithm (the inventory of twelve in the chat of 2026-10-12, and KRONQ_STEP_NOTE.md) and say which
of them the one-time theory (Thm stepsup, Thm stepkrylov, Prop rates, Thm time0 of training_theory_2026-10-07.tex and
held_rough_error_lean.tex) and the chain (THEORY.md 8-24) already decide, what the other optimizers are in the same
language, and which statements can be proved. Companion to THEORY.md section 21, which lists what is NOT covered; this
note is the positive side. Every symbol is defined here once.

## 1. Objects

* theta in R^P the parameters, Yhat(theta) in R^N the outputs on the batch (N = m ny), E = Yhat - Y the residual.
* J = dYhat/dtheta (N x P), K = J J^T / n the training kernel (N x N), eigenpairs (lambda_k, v_k), e_k = <v_k, E>.
  Leverage of a mode: s_k = sqrt(n lambda_k) = the singular value of J on v_k; a parameter step of length d along the
  right singular vector of v_k moves the residual coordinate e_k by -s_k d.
* A step d in R^P changes the residual to E + J d + r(d), r the Taylor remainder, ||r(d)|| <= (1/2) H2bar(|d|) |d|^2
  (Thm 16 / Prop. ell; H2bar the largest rms second derivative of the outputs on the ball of radius |d|). The Taylor
  radius is that ball: a SPHERE in the (rescaled, Pi^{1/2}) parameter metric by definition of the bound. Whether the
  remainder is actually isotropic over directions is a measurement (REPORT 57: the second derivative along mode k goes
  as s_k^0.14-0.19, flat to a factor of a few, at width 160; not yet measured at width 4096 - the ridge-shape diagnostic of
  kronq_fast does this, and tonight's numbers are void because of the TF32 bug).
* The per-layer Kronecker eigenbasis of the optimizer is the eigenbasis of each layer's block of K (Thm 2 / Thm 17(ii)
  with the factorisation E[delta delta^T] (x) E[a a^T]); its eigenvalues lam_ij approximate lambda_k; its mispricing of a
  coherent step (the product of averages for the average of products) was measured at 30-380x (REPORT 61 item 4).

## 2. The one-step optimum is the ridge step, and the Krylov series is conjugate gradient on it

Thm stepsup: the smallest first-order error after a step of length <= s is F(s), attained by the step whose residual is
E_gamma = gamma (K + gamma)^{-1} E, with gamma set by s. In parameter space that step is

    d*(s) = - J^T (K + gamma I)^{-1} E / n = - (J^T J / n + gamma I)^{-1} J^T E / n,

the Levenberg-Marquardt step with damping gamma(s): per mode d_k = e_k s_k / (n lambda_k + ... ) i.e. the mode is fitted
completely when lambda_k >> gamma (the Gauss-Newton step e_k / s_k) and proportionally to lambda_k / gamma when
lambda_k << gamma (a gradient step). The roughness function attaining the bound is the rank-one functional along E_gamma.

Thm stepkrylov: restricting the step's residual direction to the Krylov space V_M = span{E, K E, ..., K^{M-1} E} gives
F_M(s) increasing to F(s). The step whose residual lies in V_M is a polynomial of degree M-1 in K applied to E, i.e.
M-1 conjugate-gradient iterations on (K + gamma) x = E. So:
* M = 1 is the gradient step with a line search (the best multiple of J^T E);
* M = q_E recovers the ridge step exactly;
* each Krylov term costs one application of K, i.e. one jvp plus one vjp on the batch (about one forward-backward).
This is the matrix-free form of the ridge step: no factor statistics, no eigendecomposition, exact on the batch. Its
cost at M terms is about M F0 (F0 = one forward-backward). The Kronecker eigenbasis is a way to approximate the same
solve with a diagonal; at blocks of 256 the fast path pays about 1.1 F0 per step for it plus 0.67 F0 for the probe.

Consequence 1 (direction, fast modes). For the modes above the cutoff gamma the theory's direction is the ridge step,
and the fast path's cap (step at most |g_ij| / lam_ij) is its Kronecker approximation. The division of the cap by the
number of layers L is a heuristic for the overlap of different layers' modes in residual space; the exact replacement
is the one-dimensional ridge along the chosen direction D on the batch, alpha = g.D / (||J D||^2 / n + gamma), which the
code has as the exact-GN line search (--gnls; one jvp). It costs 0.33 F0 and removes the L.

Consequence 2 (length). The remainder bound says the first-order model is valid while (1/2) H2bar |d|^2 is small
against the first-order gain |d| ||E_hat||_K; the a priori optimum of gain minus remainder is at |d| = ||E_hat||_K / H2bar,
where the relative remainder rf = ||r|| / ||J d|| is 1/2. The measured optimum is rf ~ 0.1 (REPORT 60) and the theory does
not predict that level (THEORY 21(a)); but it does say that rf is the quantity to hold, and that it is one forward plus
one jvp per check. The output-change budget (--rem 0.2: ||J d|| <= 0.2 ||E||) controls the first-order TERM, not the
remainder; it exists because the Kronecker model mispriced ||J d||. Its principled replacement is the pair (exact line
search for the direction's length, remainder controller for the radius), both already in the code, with the constant
0.2 replaced by the remainder level kappa. The decisive comparison is rem 0.2 against gnls + lr_kappa, both at Adam
memory, once the TF32 fix is in (it was not in any run before 22:40 on 2026-10-12).

## 3. The other optimizers in this language (Thm 17: a preconditioner P gives the kernel K_P = J P J^T / n)

| method | basis in which the rule acts | rule on a coordinate | kernel-theory reading |
|---|---|---|---|
| GD | parameters | d = lr g | M = 1 Krylov term; spectral bias, rate lambda_k per unit time |
| Adam | parameters | d = lr m / (sqrt(v) + eps) ~ lr sign(g) for consistent coordinates | sign (l_inf) steepest descent in the coordinate basis; K_P with a negative dP term (Thm 17 correction) |
| Muon | per-layer SVD of the gradient G = Delta g A^T / n | all singular values set to 1 | sign descent on the modes of the GRADIENT (the residual seen through the layer), not of the kernel; w^3 per step |
| SOAP (b256) | per-layer eigenbasis of E[G G^T], E[G^T G] | Adam rule on rotated coordinates | sign descent with consistency weight |m_k| / sqrt(v_k) in the eigenbasis of the gradient covariance = K weighted by the residual's coordinates |
| KronQ fast, --rule adam | per-layer eigenbasis of E[delta delta^T], E[a a^T] (probe = true Gauss-Newton factors) | Adam rule | SOAP's rule in the KERNEL's eigenbasis instead of the gradient covariance's |
| KronQ fast, sign push + cap | same | w_ij sign(m_ij), at most g_ij / lam_ij | sign descent below the cutoff, ridge (completion) above it |
| KronQ --alloc / LM / trust-region | same | g / ((lam + gamma) T_rem) | the ridge step itself (one-step or T-step, frozen lam) |

Two facts of the record read off this table. (i) Every method that works at width 4096 (Adam, SOAP, Muon, the sign
push) is a sign-type rule in some basis; every first-order-optimal rule (GD, LM, the trust-region ridge of REPORT 58,
the T-step allocation of REPORT 61 item 7) loses per step. (ii) The bases differ in whether they are built from K alone
(ours) or from K weighted by the residual (SOAP, Muon): the Krylov functional says the optimal direction depends on
both K and E through (K + gamma)^{-1} E, so a residual-weighted basis is not wrong, it is a different approximation of
the same solve. Which is better at equal FLOPs is not decided by theory; the ablation's f/g (rule adam in our basis)
against SOAP b256 isolates the basis from the rule.

## 4. Why first-order-optimal rules lose: the lift, and the two-regime structure of sign + cap

CORRECTION (same night, LOOKAHEAD_THEORY.md): the claim below that a first-order-optimal rule cannot start a
zero-leverage mode, and P2 in section 6, hold only without cross-lift; under gradient flow a dead direction's leverage
changes at rate up to L_phi |theta_dot| through the motion along the gradient (Lemma 1 there). The regime of a mode is
decided by its residual curvature c_k = <E, f_uu> (the non-Gauss-Newton term), not by its leverage: Theorem 7 there
replaces the two-regime model of this section, and P2 is withdrawn.

The chain's stage identity: over one step, K_{t+1} - K_t = Cross + Q with Q = Delta_J Delta_J^T / n >= 0, Delta_J the
change of the Jacobian along the step, second order in the step; Cross first order and sign-indefinite. The spike
theorem (THEORY 7, 7.1) says a direction is learned only by lifting its leverage. The one-step theorems of section 2 are
first-order optimal: they allocate to mode k in proportion to its leverage s_k. A direction with s_k ~ 0 (a held-rough
direction: large e_k, tiny lambda_k) gets d_k ~ 0 from every such rule, and the lift it would get from motion along it,
delta lambda_k ~ c_k d_k^2 (Q's diagonal; c_k the squared second derivative of the outputs along the mode, the quantity
REPORT 57 measured flat in lambda), is then ~ 0 too. Gradient flow does lift such a mode, through d_k ~ lr s_k e_k:
lambda_k grows at rate ~ c_k lr^2 e_k^2 lambda_k, i.e. exponentially from lambda_k(0), with a time constant that is the
spike theorem's. A constant push d on the mode lifts it linearly, lambda_k(t) ~ lambda_k(0) + c_k d^2 t, independent of
lambda_k(0).

Model (to be stated as a proposition; assumptions named): per-mode dynamics with no interaction between modes,
e_k' = -s_k d_k, lambda_k' = c_k d_k^2 (Q only; Cross dropped), budget sum_k d_k^2 <= R^2 per step (sphere), T_rem steps
left, objective min sum_k e_k(T)^2. Then
* modes with lambda_k large enough to be completed within the budget are fitted at the ridge/completion rate
  d_k = e_k / (s_k T_rem) (the cap, the q = -1 regime of KRONQ_STEP_NOTE);
* modes below the cutoff are in the lift regime: the marginal value of d_k is dominated by the future leverage it buys,
  which under the lift law is maximised by a CONSTANT push for the remaining steps, with the sign of e_k's gradient
  and a magnitude set by the budget shared among lift-regime modes - a sign push with a common weight;
* the cutoff between the regimes is set by R, T_rem and c_k, not by lambda_k alone.
This is the structure of the fast path's rule (sign push below, cap above), and the learned per-mode weight exp(u_ij)
is the online estimate of the regime and of the weight. It explains the measured record: the ridge and LM rules lose
3-4x per step (REPORT 58), the frozen-lambda allocation loses to the hypergradient (REPORT 61 item 7), and the learned u
rises toward slow modes and peaks before the slowest decile (the lift regime ends where c_k d^2 T_rem can no longer
bring the mode to completion). Gradient-flow spectral bias and the Euler/direct results of the paper are the same
mechanism seen from the first-order side.

What the one-step hypergradient misses (THEORY 21(c)1): it measures -lr g_t,k D_{t-1,k}, the current gradient along the
previous step, which contains the lift only through the change of g; the future value of the lift is not in it. A
multi-step value would need the lift rate c_k, which is measurable per mode from the stage identity (Q's diagonal in
the mode basis) and is the quantity the allocation should use.

## 5. The inventory of rules against the theory

| rule (chat inventory) | theory's verdict |
|---|---|
| 1 Kronecker GN factors from the probe | derived: the kernel is the Gauss-Newton Gram (Thm 1-2); the probe gives the true factors; SOAP's gradient covariance is the residual-weighted alternative. Keep. |
| 2 statistics every step, EMA 0.95 | the kernel moves by a relative ~rf per step (stage identity), so the EMA horizon should be ~1/rf ~ 10 steps, not 20; testable (beta_stats 0.9). |
| 3 refresh by the change of the predicted decrease | reasonable; the threshold is a tuned constant. The basis is stale when the accumulated Cross + Q on the residual-carrying modes is O(1); same criterion, not derived further. |
| 4 blocks of 256 | nothing in the theory. The matrix-free ridge (section 2, CG on the batch) needs no basis at all for the fast modes; the blocks are only needed for the sign part. Open. |
| 5 momentum 0.9 | standard; in the chain it is a memory term in the drive (Thm 17). Not derived. |
| 6 sign push | derived in structure (section 4: the lift regime), not in its weight. |
| 7 hypergradient weight | a greedy one-step estimate of the per-mode value (21(c)1); the multi-step value needs c_k. |
| 8 live mask 1e-12 | inert; remove. |
| 9 cap, divided by L | the ridge/completion regime; replace the L by the exact line search (0.33 F0). |
| 10 lr 1e-3 with linear decay | the decay anneals minibatch noise (Cor 24a identity; not treated). The per-step RADIUS should be set by rf (section 2), the decay then acts on the sign part only. Not derived. |
| 11 output-change budget 0.2 | a first-order trust region that patches the Kronecker mispricing; replace by exact line search + remainder controller (section 2). The 0.2 is then gone; kappa remains, measured ~0.1, unpredicted. |
| 12 bf16 state, rank-one normaliser | engineering; no theory content. |

## 6. Statements to prove (for the paper's optimizer section) and what to measure

Proofs available with the existing tools:
* P1 (ridge = LM; Krylov = CG). Thm stepsup(c) in parameter space is the LM step; Thm stepkrylov's V_M step is the
  (M-1)-th CG iterate on (K + gamma) x = E. Algebra; worth stating because it makes the fast-mode part of the optimizer
  a matrix-free solve on the batch, with no factor statistics.
* P2 (WITHDRAWN, see the correction in section 4; replaced by LOOKAHEAD_THEORY Theorem 7) (first-order-optimal rules do not start a zero-leverage mode). For any rule whose step is a function of J^T E
  through a positive map (GD, LM, ridge, any K_P flow with P fixed), d_k = 0 when s_k = 0, and under the lift law the
  lift of that mode is zero; the spike theorem's rate for gradient flow is the exponential one above. Direct from the
  definitions and Thm 17.
* P3 (two-regime allocation under the lift law). The model of section 4 as a proposition: closed-form regimes and
  cutoff for constant c_k; the sign push with a common weight below the cutoff and completion above it. Needs only the
  per-mode model; the assumptions (no interaction, Cross dropped, flat c_k, sphere budget) are named and each is a
  measurement in the list below.
* P4 (what the output-change budget is). ||J d|| <= rem ||E|| is a bound on the first-order term; under the Kronecker
  model with mispricing factor kappa_mis = ||J d||_true / ||J d||_Kron, the sign push's true first-order change exceeds
  the model's by kappa_mis, and the budget is equivalent to dividing the push by kappa_mis. Trivial, but it says the
  budget is a measurement of kappa_mis, which the line search makes explicit.

Measurements the proofs need (width 4096, n in the thousands, both targets):
* M1 the lift rate c_k per mode (Q's diagonal in the mode basis) against lambda_k: flat or not (REPORT 57 at 160 only).
* M2 the ridge-shape diagnostic with the TF32 fix: remainder along step / gradient / random / ridge directions at equal
  length; whether the Taylor radius is a sphere in the Kronecker mode coordinates.
* M3 rem 0.2 against gnls + lr_kappa(0.1) at Adam memory with the fix: the budget constant against the remainder level.
* M4 the learned u profile against the P3 cutoff computed from measured c_k, R, T_rem: whether the hypergradient finds
  the two-regime solution.
* M5 the matrix-free ridge for the fast modes (CG, M = 2-3 on the batch) plus the Kronecker sign push below the cutoff,
  at equal FLOPs: whether the factor statistics are needed for the fast part at all.
* M6 SOAP b256 against --rule adam in our basis (ablation f/g): basis against rule.
