# The sharpening gradient: kernel basis or parameter space, the derivative of the kernel modes, and a model test (2026-10-13)

Written in the theory fork of the 2026-10-12 session (the other fork runs the width-4096 experiments; this note and
optimizer/lift_model.py are the only files it touches). Companion to LOOKAHEAD_THEORY.md (per-mode model, Theorem 7)
and STEP_THEORY.md. Sections 1-4 are the theory as discussed; section 5 is the model built to test it; section 6 is
what the model found, which is a negative result for the first-order lookahead term and a restriction of the theory.

## 1. Is the kernel eigenbasis the right place to look?

Only for the first-order part. The kernel K = J J^T lives in function space; its eigenvectors are residual directions and
the parameter directions they correspond to are the row space of J, of dimension at most the number of batch outputs
(m ny ~ 10^5 at width 4096) out of P ~ 3 x 10^7 parameters. The rest of parameter space is the null space of the current
Jacobian: motion there changes the outputs by nothing at first order and the kernel basis cannot see it. Yet the lift of
a residual direction v by a step d is M_v d, with M_v the Hessian of <v, f>, and the null-space part of d contributes to
it exactly as the row-space part does. The directions that can lift the slow residual are, generically, directions the
kernel basis does not contain. Late training is lift-driven (the residual sits on the slow modes by then), so the
object governing late learning is a parameter-space object.

Parameter space on its own has no structure (coordinates are arbitrary; Adam's basis is a poor one). The structure is
the residual Hessian M_E = grad^2 <E, f>: its large eigen-directions are "the untrained parts that can be bent into the
residual" (for an MLP: neurons whose pre-activation sits where act'' is large and whose downstream weights project on
the residual). The loss Hessian is J^T J + M_E; the kernel eigenbasis diagonalises the first term, nothing says it
diagonalises the second. The right viewpoint is the pair (J, M_E): J for what the current features can fit, M_E for
what the next features will be. The roughness floors stay in function space, where the error is measured; the two
spaces are linked by J at first order and M_E at second.

What this exposes in the per-mode rule of LOOKAHEAD_THEORY: it uses only the diagonal of M_E in the Kronecker basis
(assumption A1). Whether the concave directions are single modes or combinations is the off-diagonal of M_E in that
basis, measurable by Hessian-vector products on pairs of modes, or by the Rayleigh quotient of M_E along the actual
step against the sum of the per-mode diagonals. If it is large, the basis to rotate into is the one diagonalising
lambda + c, not lambda.

## 2. The derivative of the kernel modes with respect to the parameters

For an eigenpair (lambda_k, v_k) of K, the leverage of v_k is ||J^T v_k||, and J^T v_k is a backward pass with v_k as the
output cotangent (the probe is this with a random +-1 vector). First-order perturbation theory:

    grad_theta lambda_k = grad_theta ||J^T v_k||^2 = 2 M_{v_k} (J^T v_k)                       (the sharpening gradient S_{v_k})

one more backward through the backward graph (double backprop), about two forward-backward costs per direction. Its
inner product with a step is the Cross term of the stage identity on v_k. For a direction with no leverage it vanishes,
so the lift of such a direction is second order only (Q; the curvature c of LOOKAHEAD_THEORY measures it on the
residual). For a slow direction with some leverage it is nonzero and is the first-order lift, which the per-mode model
drops (A1, no cross-lift). Rotated into the Kronecker basis, its component on mode u_ij is the current gradient weighted
against the u_ij-th column of M_E: the off-diagonal information of M_E without forming M_E.

The eigenvector derivative is the same computation with the other eigenvectors as cotangents, divided by the gaps
lambda_k - lambda_j: the rotation rate of the moving frame (THEORY 18-19). Summed over stages it is the slow-fast
coupling whose spectral norm is the rotation remainder rho of Theorem 48. Worth logging for the paper (never measured
per stage); it does not inform the step.

## 3. In the chain's terms

* Stage identity: on a direction v, Cross = 2 <J^T v, dJ^T v> and Q = ||dJ^T v||^2 with dJ the full change of the
  Jacobian over the stage. To first order in the step Cross = <S_v, d>, to second order Q = ||M_v d||^2, and Cross has
  its own second-order part <J^T v, T[d, d]> (T the third derivative of f) of the same order as Q. (The model's exact
  check below confirms the exact identity to 1e-4 and the first-order approximations to 5-7 % at the step lengths used.)
* Feature-sum bracket (Thm 13): the leverage at T is ||Psi_0^T v + D_v||^2 with D_v the accumulated dJ^T v; S_v . d
  summed over stages is its first-order part; Lemma 1 of LOOKAHEAD_THEORY bounds it.
* Sharpening and the sign input: under gradient descent d = -eta J^T E, so Cross on the residual direction equals
  -2 eta <J^T E, M_E J^T E>: GD sharpens the kernel along the residual iff the residual Hessian is negative along the
  gradient (the curvature-sign quantity of THEORY 21(c)2 and the induction of THEORY 24). A step chosen with S . d >= 0
  enforces at first order, on that one direction, the per-stage positivity that Cor. 50 / Thm 46 carry as a
  hypothesis; not on the whole residual subspace, so those theorems stay conditional for the optimizer's path, on a
  measured quantity instead of an assumed one.
* Dyson / richness: the first-order Dyson term in the stage increments is Cross on the residual, i.e. S . d.
* Preconditioning: S is the gradient of the plain kernel's leverage (the geometry); with a per-mode preconditioner the
  effective kernel K_P has the dP term (Thm 17 correction), accounted separately.

## 4. The lookahead-gradient rule, as derived, and the summary that was agreed

In the single-direction model of LOOKAHEAD_THEORY Prop. 4 (direction v with residual component e, leverage s, path
Lambda_rem left, not yet completable: s Lambda_rem < e), the value of extra leverage is d(removal)/ds = Lambda_rem, so
the future loss decrease per unit step along w is (2 e / N) Lambda_rem <S, w> / (2 s), against the immediate decrease
-<grad L, w>. Steepest descent on the sum gives

    d = -rho g~ / |g~|,   g~ = grad L - kappa S,   kappa = e Lambda_rem / (N s),   kappa = 0 once s Lambda_rem >= e.

The summary agreed in the discussion, with the two corrections made there: (i) some parameter directions raise the
kernel's leverage on the slow residual, in two ways (first order S, needs some leverage; second order c, works for dead
directions); (ii) "hill" (lambda + c < 0: the loss falls at second order either way) and "lifts the slow residual" are
different properties and the step uses both; (iii) a rule that never increases the error is a greedy monotone rule,
and the theory's content is that greedy monotone rules lose: the best plan trades error now for error at T. The
guarantee is on what no rule can beat (LOOKAHEAD_THEORY Thm 2), not on monotone descent. On momentum (asked in the
discussion): heavy-ball momentum is a Krylov method (its iterate is a degree-t polynomial in K applied to the residual;
Chebyshev-type acceleration), so it approximates the higher Krylov terms of the ONE-STEP series, not the lift terms
(it is linear in the gradients and never sees M_E); what it does by accident is take up to 1/(1 - beta) larger steps
on persistent (slow) modes, and since the lift is second order in the step, that is the same mechanism as the sign
push's. The lookahead theory proposes no other mechanism; it proposes the accounting.

## 5. The model: optimizer/lift_model.py

Exact parameter space of a small MLP (width 128, depth 2, the field problem of scale_study.py), everything by autograd
(torch.func), no Kronecker basis, no blocks. Per step: g = grad L; the direction lifted v = E_hat ('full') or the
normalised least-squares Krylov residual r_M = E - K x_M, x_M in span{E, K E, ..., K^(M-1) E} minimising ||r|| ('slow',
M = 5: the part no M-term Krylov step removes, Thm stepkrylov; ||r|| <= ||E|| by construction; NOTE: conjugate
gradients' residual is NOT monotone in this norm and came out 9x larger than E, which invalidated a first set of runs);
s^2 = ||J^T v||^2 (one vjp); S = grad s^2 (double backprop). Rules at the SAME per-step Euclidean length rho_t (linear
decay), so only the direction differs: gd (-g), momentum (heavy ball 0.9), sign (-sign g, l_inf at the same l_2 length),
lookahead (g~ above), liftonly (+S, control). --check 1 verifies at four steps: the exact stage identity
s^2(theta + d) - s^2(theta) = 2 <J^T v, dJ^T v> + ||dJ^T v||^2 (two vjps), its first-order approximations <S, d> and
||M_v d||^2 (jvp of the vjp), and the residual curvature d^T M_E d (jvp of the gradient) against central differences.
Login-node runs: ~5 s for the check, ~30 s per 800-step comparison run.

## 6. Results (2026-10-13, login node; runs/lift_model/{check,cmp,cmp2,cmp3}.jsonl)

Checks (width 64, rho 0.01, gd, slow direction): identity error 1e-4 of the terms at every step; <S, d> within 7 % of
the exact Cross, ||M d||^2 within 5 % of the exact Q; d^T M_E d against the quarter-step central difference within 0.5 %.
At rho 0.3 none of the second-order approximations hold (the leverage changes by 400x in ten steps): the expansion
is local, as it says. Under GD at init the slow residual's leverage FALLS at every step (Cross < 0) with positive
curvature along the step: the unfavourable-sign case of Prop. 4.

Comparison (width 128, batch 64, 800 steps, validation MSE at the end; rough target initial 0.657, smooth 0.826):

| rule (direction lifted: slow) | rough, rho 0.03 | rough, rho 0.1 | smooth, rho 0.03 |
|---|---|---|---|
| gd | 0.6345 | 0.632 | 0.2431 |
| momentum (0.9) | 0.631 | 0.6355 | 0.1116 |
| sign | 0.6348 | 0.6309 | - |
| lookahead, kappa as derived | 0.6459 | 0.638 | 0.2769 |
| lookahead, kappa x 0.3 | 0.6421 | - | - |
| liftonly (control) | 1092 (diverged) | - | - |

Slow leverage s^2 along r_M, first -> last (rough, rho 0.03): gd 80 -> 19, momentum 80 -> 24, sign 80 -> 24, lookahead 80 -> 450-530;
smooth: gd 760 -> 694, momentum 760 -> 376, lookahead 760 -> 1300; liftonly 80 -> 3.5e6 with the error up 1000x. So the
baselines LOWER the slow residual's leverage over the run (de-sharpening), the lookahead raises it 20-25x relative to them,
and the error is nonetheless worse. The alignment cos(g, S) is +0.1 to +0.3 through the runs on both targets: the sharpening
direction ascends the loss at first order. (With the whole residual as the lifted direction, 'full', the lookahead at
rho 0.1 drove s^2 to 3e6 and the error to 1.3: sharpening the fast modes, i.e. instability.)

Reading. (a) The first-order lookahead term does what it is built to do, it raises the slow residual's leverage several
fold, and the error at T is worse, not better, at every kappa tried. (b) Lift alone is sensitivity, not learning: the
control raises leverage by 4e4 and destroys the fit. (c) The derivation's assumption that the lift direction is
orthogonal to the descent direction is false here (cos +0.1 to +0.3), so the lift is paid for in error now and the
model's horizon value does not recover it within 800 steps. (d) A second omission: the per-step radius is fixed in
the model, but in a real optimizer it is set by the remainder, which grows with the sharpness the lift creates; the
lift's payoff per step is leverage x radius, and the radius shrinks. (e) On the rough target at this size nothing
learns in 800 steps (0.657 -> 0.63 for every rule), so it is not a discriminating testbed for the rough problem; on
the smooth target momentum beats gd 2.2x at the same step length, which is the Krylov acceleration, not lift.

Verdict for the theory. The first-order lookahead (S as a gradient correction) is withdrawn as a step rule. What
survives is the second-order criterion of LOOKAHEAD_THEORY Theorem 7: on a mode with lambda + c < 0 the lift and the
loss decrease are the same motion (the self-lift case of Prop. 4), so there is no error-now to pay and no separate
radius to buy; those are the only directions on which investing in leverage is free, and the sign push with the
learned weight is a stochastic search for them. Prop. 4's "lift then reduce" needs, to be made right, the first-order
cost <g, w_phi> of the lift direction and the dependence of the radius on sharpness; both are missing from the model
in LOOKAHEAD_THEORY and are named there now as the reason the S-term fails.

Consequences for the width-4096 code (the other fork): do NOT add S to the step. S rotated into the Kronecker basis
and binned by lambda-decile next to c and the learned weight remains a worthwhile diagnostic (Corollary 8(c)
prediction, and whether the learned weight tracks the hills rather than the lifts); the eigenvector-derivative sum is
Theorem 48's remainder per stage and is worth one measurement for the paper. The experiment of LOOKAHEAD_THEORY
section 4 (the two-regime rule against the reference) stands as planned; its decision criteria are unchanged.

Open. Whether a radius-aware, cost-aware lookahead (value of leverage net of the error-now and of the radius
shrinkage) has any regime where it beats the hill criterion is a question for the model at a size where the rough
target learns (width 256-512, 5000 steps, which needs a GPU node); it is not the next thing to run.
