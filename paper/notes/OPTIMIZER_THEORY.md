# The optimizer in the theory's terms: what is derived, what is measured, what is refuted (2026-10-13)

The consolidated statement after the 2026-10-12/13 sessions. Every claim carries a pointer to REPORT.md section 61
(items 1-32), to the notes that derive it (MECHANISM.md, TRUST_REGION.md, LOOKAHEAD_THEORY.md, LIFT_GRADIENT.md,
STEP_THEORY.md, COLLECTIVE_RULE.md), or to THEORY.md. Numbers are for the online L63 field problem at width 4096,
depth 2, batch 256, 5000 steps, Adam-memory KronQ (blocks of 256, compressed state), single seed; the method's
configuration is the "fast path" of optimizer/kronq_fast.py (stats every step, subspace iteration, adaptive refresh
with floor 5, sign push, cap, learned per-mode weight, output budget 0.2, lr 1e-3 with linear decay). This note is
meant to be reusable on the next problems: section 6 lists what to re-measure there.

## 0. The method from first principles (2026-10-13, after items 37-42): three ingredients and a safety

What the optimizer is for: a rough target's residual sits on the kernel's slow modes (THEORY.md, held rough error), and
gradient descent moves mode k's residual coordinate by lr lam_k e_k per step, so the slow modes are never reached within
the budget. The nudge must move every live mode by a fraction of its residual coordinate that does not scale with lam_k.
Three things determine such a nudge, and each has a theory statement and a status:

  1. WHICH MODES (basis). The per-layer Kronecker eigenbasis of the Gauss-Newton factors (Theorem 2's per-sample outer
     products under the Kronecker factorisation). Status: derived up to the factorisation; the factorisation is wrong on
     the slow deciles (coupling 10^3-10^6, item 25 (fork) c) and right within 2x on the fast ones. Measured as necessary:
     without it the same rule collapses on rough (T2, item 37).
  2. DIRECTION. The sign of the momentum per mode. Theory: Lemma 1 (noise follows leverage) and Prop. 2 (a bounded flat
     profile beats the lambda-weighted and the flat-noise rules under batch noise), comparative statements; and the sign
     process (SIGN_PROCESS.md 2): the momentum raises the per-batch SNR 4.36x and the sign is right 81 % of the time at
     q = 0.2; lag-1 agreement 0.71, CONFIRMED on every decile (item 42b). Status: derived as a comparison, confirmed as a
     process; the momentum horizon (10 steps) is an empirical constant whose scale the theory names (coherence time
     30-50 steps from the 2-3 % per step kernel motion, item 21e) but does not set.
  3. MAGNITUDE. Per mode, lr x w_k with w_k learned so that the next batch gradient and the last step are uncorrelated.
     Theory: the fixed point of that rule is the completion step on the population gradient times the mean sign,
     w_k lam_k lr = m_k |grad_k|, FOR A MODE WHOSE RESPONSE IS ITS OWN (REPORT 61 item 42, correction): a secant estimate
     of the per-mode completion distance that never reads the Kronecker lam (which is why the method survives the
     curvature mispricing that sinks the ridge and Adam-in-basis rules). Status: the weakest of the three. The derivation
     needs decoupled modes (false on the slow deciles, exactly where the rough residual lives) and convergence off the
     clamp (medians of u are 0.0-0.7, so the bulk is off the clamp; the tails are unmeasured). Its one discriminating
     test (the ratio w lam lr / |grad| per decile against m_k on the fast deciles) has not been run. What is measured:
     the learned weights are worth 2x over uniform ones (items 13, 22), the median weight is ~1-2 x lr, so the gain is in
     the per-mode SHAPE, not in the overall length, which lr sets. The averaging horizon of the weights (100 steps) is
     an empirical constant like the momentum's.
  4. SAFETY (the envelope). An output-space trust region, batch output change <= rem ||E_B||, with the lr schedule. Theory:
     the Taylor radius of the chain (THEORY.md): the expansion about the current state holds inside it, and a slow mode's
     radius is small because its direction rotates quickly under the step (the rotation remainder; near-degenerate slow
     eigenvalues rotate fastest, which is the coupling of item 25 (fork) c seen from the chain's side). The radius is a
     deterministic function of the state, not a noise quantity; it is the ceiling, and the noise-shrunk completion step
     of ingredient 3 is the working point below it. Measured: at the working length the remainder is 0.2-6 % of the
     first-order term (item 21a), so the ceiling is not binding there; at lr 1.5e-3 the budget is the difference between
     0.07 and divergence (item 40), so the ceiling is where the budget acts. Status: the per-mode radius of a slow mode
     is not computable from the Kronecker model (the gaps are degenerate; the object that has a finite radius is the
     near-degenerate SUBSPACE, and the equal-magnitude sign push on such a subspace is a subspace push); the output-space
     constant 0.2 and the lr are empirical, set by the grid (item 40), and item 42's correction says why no local rule
     can set them: the quantity a local rule reads (gain minus cost over a window) is 1-2 % of the terms it sits between.
     The fixed budget is not safe on every problem: on 2D turbulence (400k pairs, 20 epochs, lr 1e-3) every run at budget
     0.15 diverges after step 1000-5000 (item 48) while the same budget on the field is the working point; the ceiling is
     problem-dependent and below 0.15 there. A held-out-coupled cap, budget_t = min(rem, kappa 2 c_t) with c_t the
     held-out alignment of the step (slow_adapt.py --sa_cap 1), removes the divergence (item 50), but as a rule it is not
     what it claims: on turbulence c_t falls to 0.003-0.013 within 2500 steps and the budget sits on its 0.03 floor for the
     rest of the run, and on the smooth field the cap binds and costs 1.5-3x against the uncapped run, because the batch
     output change overstates the held-out change 9-145x (items 32, 50) and the per-step quadratic does not set the useful
     length. Status: a problem-dependent ceiling exists and the method has no rule for it; the cap is a safety that
     substitutes a small constant for the divergence. On turbulence KronQ with it reaches 0.19 (increment; AdamW 0.156) and
     0.079 best / 0.095 final (direct; AdamW 0.066).
  5. REGULARISER (items 51-53, finite data). The push removes gradient descent's implicit regulariser: the slow modes it reaches are
     where a finite sample's own structure and the leverage noise sit (Lemma 1), so on replayed data the fit goes to directions
     that do not transfer (train 0.063 / validation 0.195 on the turbulence increment; at matched training loss 24-34 % worse
     than AdamW). No per-step, per-mode test sees it (the hypergradient already is one; a shrink on it did nothing on turbulence
     and cost 1.6-1.8x on the field). Explicit decoupled weight decay does: at wd 0.3-1.0 the training loss is unchanged and the
     validation falls 35-43 %, to 0.111-0.127 against AdamW's 0.156 (which gains nothing from decay there), reproduced on a
     second seed; on the smooth target a tie (0.063 vs 0.060). Whitened AdamW overfits the same way (0.216 / 0.122, training
     loss far below unwhitened): reaching the slow directions is the gain and the memorisation at once, and the decay separates
     them. The decay strength has a target-dependent ceiling (a transient spike at wd 1.0 on direct). Status: measured, one
     constant, theory statement at the level of "implicit regulariser removed, explicit one needed"; the value is not derived.
     On the field (fresh data) the input-side basis alone is most of the rough gain: whitened AdamW 0.109 vs KronQ 0.062 vs
     AdamW 0.39 (item 52), and the moving hidden-layer bases carry the remaining 1.75x (frozen bases collapse).

Not ingredients: the cap (active on 0.3 % of modes, item 42b; it is ingredient 3 computed from one batch instead of the
sequence, and the sequence estimate is always smaller), the regime switch (retracted, item 41), the kappa rules (items
38-42), the per-layer and per-mode budgets (items 32, 39), the lift (LIFT_GRADIENT.md), and the speed engineering
(blocks, bf16 state, Cholesky-QR, fused update, adaptive refresh), which changes wall time and not the method.

Conjecture (stated, untested): sign(C) x learned w is a per-mode Wiener-shrunk Gauss-Newton step, m_k^2 in place of
Q_k^2 / (1 + Q_k^2) (within 15 % at the measured Q), with the completion distance from a secant estimate. If the ratio
test above holds on the fast deciles, the method is "shrunk secant-Newton in the Kronecker basis", and the three
ingredients are one.

What this section does NOT claim: that the 3.8x on the field problem generalises. It is one problem, one width, one
architecture; the raw-coordinate image is a tie with tuned AdamW (item 41). Breadth before any further component.

## 1. The method, stated

Per layer, with the eigenbases of the Gauss-Newton factors A = E[a a^T] (inputs) and S = E[delta delta^T] (backpropagated
probe), a mode is q_i p_j^T, its Kronecker curvature lam_ij = sigma_i alpha_j (2 / n_y), its gradient g_ij. One step:
  1. momentum C <- 0.9 C + 0.1 g (in the mode basis);
  2. direction D_ij = w_ij sign(C_ij), capped at the completion length |g_ij| / (lam_ij lr L);
  3. step Wt <- Wt - lr(t) D, lr(t) linear decay to 0;
  4. the whole step scaled so that the batch output change is <= 0.2 ||E_B||;
  5. w_ij <- w_ij exp(-beta_u hg_ij / rms(hg)), hg_ij = -lr g_ij D_ij(previous step)  (the one-step hypergradient).
Statistics every step, bases refreshed when the predicted first-order decrease changes by 10 % (floor 5 steps), by two
warm-started subspace iterations. Cost 2.1-2.5 forward-backwards per step, state 1.96 x parameters.

Results, 3 seeds, isolated, fused update (item 36): rough 0.062 (seeds 0.060-0.065; SOAP b256 0.091, AdamW 0.39 at
equal steps, ~0.21 at equal wall), smooth 1.38e-4 (AdamW 8.5e-4 at equal steps, ~1.1e-4 at equal wall; SOAP unstable
across seeds, 1.3e-4 to 1.4e-3). Wall per step, clean: 12.7 ms (rough) / 8.4 ms (smooth) against AdamW 2.2 ms and SOAP
28.5 ms: 5.8x AdamW, 2.2x faster than SOAP. At equal wall: ~3.4x better than AdamW on rough, level on smooth. The eager
path shows a 1.8x seed outlier on rough (numerical sensitivity, item 35); the fused path's seeds agree within 8 %.

## 2. The mechanism (MECHANISM.md; items 21, 22, 27, 28)

Objects: E in R^N the population residual, K = J J^T / n the kernel with eigenpairs (lambda_k, v_k), leverage
s_k = sqrt(n lambda_k), e_k = <v_k, E> the residual coordinates; the batch drive g_B = (1/n_B) sum_B z_x with
z_x = J_x^T E_x (Theorem 2's per-sample form; Corollary 24a).

LEMMA 1 (sampling noise follows leverage; derived, confirmed). The batch gradient's coordinate on mode k is
s_k x (batch mean of p_k(x)), p_k(x) = <v_k(x), E_x>. Hence its signal is s_k e_k / n, its noise s_k tau_k / sqrt(n_B),
and SNR_k = e_k sqrt(n_B) / (n tau_k) contains no lambda_k. Under (S): tau_k ~ tau across modes (delocalised
eigenvectors), SNR_k is flat in lambda and proportional to e_k. MEASURED: tau_k / s_k flat within 1.5x over ten
deciles on both layers (item 27); per-batch SNR 0.6-0.9 on every decile (items 21, 27), higher on the deciles where the
residual sits (top decile on smooth, slow input modes early on rough: item 24).

PROPOSITION 2 (expected gain per unit output change of the three rule types; derived, ranking confirmed). With the
output change ||J d|| fixed (the budget), the expected first-order population decrease is proportional to
  GD / Adam-in-coordinates:  sum lambda_k e_k^2 / sqrt(sum lambda_k^2 (e_k^2 + n tau^2 / n_B))   (signal lambda-weighted, noise lambda^2 on the fast modes),
  exact ridge (Krylov / GN): ||E||^2 / sqrt(||E||^2 + N n tau^2 / n_B)                          (flat noise on all N modes),
  bounded flat-profile step:  (0.8 sqrt(n_B) / n tau) sum s_k e_k^2 / sqrt(sum lambda_k)        (signal s_k e_k^2, noise bounded, ~ lambda_k).
On a rough residual (e_k large where s_k small) the third is largest by a factor ~ n / s_k on those modes; on a smooth
residual (top modes) the three agree up to constants. MEASURED: held-out alignment per step AdamW 0.01-0.07, Krylov
0.06-0.17 on the batch, sign / Kronecker-ridge / Adam-rule 0.10-0.12 held-out (items 17, 21, 22); the rankings above on
both targets.

THE SIGN IS NOT SPECIAL (items 27, 28). With the cap and the learned weight applied to both, the sign push and the Adam
rule in the same basis have the same per-decile output-energy profile within 10 %; the Adam rule's |m| / sqrt(v) is a
flat ~0.15 (v remembers larger past gradients), so its step is 2-3x shorter where the budget does not bind (rough) and
equal where it does (smooth: the two tie at 1.15x). The sign is the simplest bounded flat-profile step that reaches
the budget every step. Its margin over SOAP's rule is the length rule, not the direction.

THE LEARNED WEIGHT (items 13, 19, 22, 27, 28). Worth 1.5-1.8x. Profile: 2-6x on the slow input modes early in a rough
run (where the persistent signal is, item 24), 0.9 on the top decile, ~1 on the hidden layer. The model's closed form
e_k erf(SNR_k / sqrt 2) / s_k is WRONG (diverges on rough, 3x worse on smooth, item 27); so are the sign-statistics
rules (items 23, 26) and the derived input-side weighting (item 20). The one-step hypergradient is the rule to keep; its
form is measured, not derived.

THE CAP (items 13, 24). Worth 1.3x; the completion step on the top modes, which otherwise overshoot (period-4
oscillation of the top decile's gradient, R(2) = -0.1 to -0.5).

THE LENGTH (items 15, 22, 27, 29-32). The per-step held-out loss change of a step with relative output size rho and
held-out alignment c is -2 c rho + rho^2 (times ||E||^2): optimum rho = c, break-even rho = 2c. Measured: every rule's
step sits at its break-even on held-out data (gain = quadratic cost within 10 %); c ~ 0.1 early on rough, so the budget
0.2 sits AT the break-even; the per-step optimum (held-out line search, rho = c) is lr-insensitive but 2-6x WORSE
because the run's progress is the accumulated drift of many steps, not the per-step gain (the same lesson as the
lookahead line: greedy one-step optimality under-steps). The length is bracketed c <= rho <= 2c and the per-step theory
does not fix it inside; the free rule (batch budget + schedule) puts it at 1.5-1.9 c early and below c late, and every
costed alternative (per-layer pre-activation radius, held-out budget, held-out line search) is not better. The
constant rises with the batch (break-even set by the noise): batch 1024 wants 0.3-0.4 and then gains 2.35x per step.
On the rough target the output-space Taylor radius (0.19 ||E||, item 15) coincides with this break-even; the Taylor
radius is not what binds (the reference runs 20x beyond the per-layer rf = 0.1 radius on the hidden layer, item 30).

## 3. Trust-region geometry (TRUST_REGION.md; items 15, 19, 20)

Measured at width 4096: the first-order radius (rf = 0.1) is a sphere in OUTPUT space on the rough target (radius ~0.19
||E|| for seven directions whose leverages span 250x) and nothing like a sphere in parameter space (200x between
concentrated and spread directions). Per layer it is a sphere in pre-activation space: the single-mode radius is
delta_j = c_l / sqrt(alpha_j), confirmed to +-20 % over eight decades of the input eigenvalue alpha_j on the input layer
and over the measurable range on the hidden layer (DERIVED: the layer's remainder is quadratic in dz = dW a, and the
batch sum of dz^2 is exactly sum d_ij^2 alpha_j in the input eigenbasis, no cross terms). These are statements about the
network and belong in the paper. They do NOT explain the step: the method operates far inside the per-mode boxes
(push ~1e-3 vs box 3-200) and far beyond the first-order radius on the hidden layer; the box-edge rule and the
per-layer budget both lose (items 20, 30).

## 3b. Component ablation (item 37, single seed, Cholesky-QR fused configuration)
Momentum essential (off: 1.9x rough / 6x smooth worse; 0.9 near-optimal, 0.95 lags 3x on rough); statistics EMA 0.95
near-optimal (0.99 = 2.2x worse on rough); hypergradient rate 0.01 at its optimum (+-1.2x either side); the 1/L in the
cap inert (removed: the cap is the Gauss-Newton step exactly); lr decay needed (none: 1.25x / 1.5x worse; cosine =
linear); blocks 128 worse on both, 512 better on smooth (0.79x) at 1.6x cost; the probe Gauss-Newton S factor 1.9x
better on rough, the empirical gradient covariance 1.28x better on smooth. Target-dependent choices: block size and the
output-side statistic; everything else is at its optimum and has a number.

## 4. Refuted, each by its own experiment (do not revisit without new evidence)

* Lift as the mechanism, second order (no concave modes: fraction 1e-5 to 1e-3, residual curvature ~1 % of GN and
  positive; the two-regime rule loses in every form: items 14, 19) and first order (the sharpening-gradient rule raises
  the slow residual's leverage 20x and the error is worse: LIFT_GRADIENT.md). The per-step loss change is first-order +
  Gauss-Newton to within 0.2-6 % (item 21).
* The Euclidean parameter-space sphere as the trust region, and with it the frozen ridge, T-step allocation, LM,
  Krylov step (items 7, 14, 17; REPORT 58).
* The per-mode box as the binding constraint (item 20); the per-layer pre-activation radius as the length (item 30).
* Lookahead as the reason the sign push wins (sections 4-5 of LOOKAHEAD_THEORY, withdrawn); the collective lift
  objective (COLLECTIVE_RULE section 1, withdrawn).
* Per-mode momentum windows (item 23: the gradient per mode is white with a ~1 % persistent part except the slow input
  modes early on rough, item 24); sign-statistics step rules (item 26); the derived per-mode weight (item 27); input-side
  weighting (item 20).
* Noise as help (plateau / saddle escape): no negative curvature anywhere; larger batch helps the sign push 1.24x per
  step at fixed budget and 2.35x with the budget raised, and AdamW not at all (items 22, 27).
* The exact Krylov step at affordable M (M <= 8: alignment 0.06-0.17, worse than AdamW on rough: item 17).
* Costed length rules: held-out budget (= reference, +0.67 F0), held-out line search (2-6x worse), per-layer budget
  (1.8-6x worse).

## 5. Open (stated as such in the paper)

0. REPRODUCIBILITY (item 35): the rough-target result at width 4096 varies by up to 1.5x between numerically equivalent
   implementations (0.060-0.092) while the smooth target is reproducible to 2 %; all rough comparisons below 1.5x in this
   note are single-seed and unresolved; the paper's rough table needs 3+ seeds per entry.

1. The learned weight's form (why the flat-ish profile with 2-6x on the slow input modes beats every closed form).
2. The position of the length inside the bracket c <= rho <= 2c (a multi-step statement the per-step theory lacks).
3. The output-space Taylor radius on rough (why isotropic in output space, why 0.19 ||E||).
4. The late phase (end-of-run degradation seen in some variants: item 16).
5. Basis choice: full factors 3x better on smooth, blocks 1.9x better on rough (fork, item 25): a rule that picks or
   mixes is untested.
6. Equal-FLOP comparison with the fast path (the cost model is in scale_jobs_flops.py; not rerun since item 9), and the
   batch-1024 + raised-budget configuration at equal FLOPs.

## 6. What to measure first on a new problem

In this order, each one run with the existing diagnostics (mechanism.py --mech, --radius_scan, --acf, --diag prof_*):
  a. the per-decile SNR and tau_k (Lemma 1 / (S)): if SNR is not flat in lambda the rankings of Prop. 2 change;
  b. where the residual sits (fast or slow modes): decides whether the rough-target story applies at all;
  c. the held-out alignment c per step and the break-even 2c: sets the budget constant for that problem and batch;
  d. the top-decile oscillation (cap needed?) and the learned-weight profile;
  e. per-step wall against AdamW in isolation, in the fast configuration only.
Candidates: the paper's L63 emulator at large n (also experiments A-F), a deeper MLP on the same field, a 2D turbulence
emulator (the climate-relevant rough regime).
