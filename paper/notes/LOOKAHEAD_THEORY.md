# Looking ahead with the chain: what can be proved about a multi-step update (2026-10-12, night)

Written at the user's request ("create the theory, make it as airtight as possible; I'm not sure how far the chain can
be used to look ahead"). Companion to STEP_THEORY.md (one-step theorems, the inventory of rules) and KRONQ_STEP_NOTE.md
(the frozen-curvature allocation). Everything here is stated for a general differentiable network and for every update
rule unless an assumption is named in the statement. Section 1 is exact. Section 2 is exact within a named model.
Section 3 lists what the chain cannot give. Section 4 is the rule that follows and the experiment that tests it.

One correction first. STEP_THEORY.md section 4 and its P2 claimed that a first-order-optimal rule "does not start a
zero-leverage mode" and that gradient flow lifts such a mode only exponentially from its initial leverage. That is
true only if a direction's leverage can change only through motion along that direction (no cross-lift). It is false in
general: under gradient flow the leverage of a direction phi changes at rate <= L_phi |theta_dot| whatever its current
value (Lemma 1 below), through the motion along the gradient. P2 is withdrawn; the correct statement is Proposition 4
and Theorem 7 below, and the quantity that decides the regime of a mode is not its leverage but its residual curvature.

## 0. Setting and notation

* theta in R^P; outputs on the batch f(theta) in R^N (N = m ny); labels Y; residual E(theta) = f(theta) - Y.
* J(theta) = df/dtheta (N x P), with the 1/sqrt(n) absorbed so that K = J J^T is the training kernel.
* Steps: theta_{t+1} = theta_t + d_t, t = 0..T-1. Path length Lambda_t = sum_{s<t} |d_s|, Lambda = Lambda_T. For a flow,
  Lambda(t) = int_0^t |theta_dot|. All norms Euclidean.
* For a unit residual direction phi in R^N, h_phi(theta) = <phi, f(theta)> is a scalar function of theta;
  its gradient is J^T phi, its Hessian M_phi(theta) = grad^2_theta h_phi (P x P). Define
    leverage       ell(phi; theta) = ||J(theta)^T phi|| = sqrt(phi^T K phi)   (= ||phi||_K of the paper),
    lift constant  L_phi = sup over the region visited of ||M_phi(theta)||_op.
  ell(phi) is the output change per unit parameter step achievable along phi; L_phi bounds how fast it can change.
* For a unit parameter direction u in R^P: response J u in R^N, leverage ell(u) = ||J u|| = sqrt(u^T J^T J u),
  f_uu = d^2 f / d u^2 in R^N (the second derivative of the outputs along u), gradient on u: g_u = <E, J u>.

## 1. Exact statements (every update rule, every step size)

LEMMA 1 (leverage is Lipschitz in the path length). For every unit phi and every theta, theta',
    | ell(phi; theta') - ell(phi; theta) | <= L_phi |theta' - theta|,
and along any path, ell_t(phi) <= ell_0(phi) + L_phi Lambda_t. The same holds for a parameter direction u with
L_u = sup ||d/dtheta (J u)||_op = sup ||d^2 f[u, .]||_op.
Proof. J^T phi = grad h_phi, and the mean-value inequality for the gradient of a C^2 function gives
||grad h_phi(theta') - grad h_phi(theta)|| <= sup_seg ||M_phi|| |theta' - theta|. The reverse triangle inequality gives the
first display; summing over steps gives the second. For u: J u = d/dtheta f[u] is the gradient of nothing in general,
but d/dtheta (J u) = d^2 f[u, .] and the same mean-value inequality applies entrywise in R^N.  []

This is the quantitative form of "a direction is learned only by lifting its leverage" (spike theorem, THEORY 7): the
lift of any direction per unit path is at most L_phi, for every rule. The step that attains it moves along the top
right-singular vector of M_phi with the right sign; a step that reduces <phi, E> at the maximal rate moves along
J^T phi. These two directions need not coincide (Proposition 4).

THEOREM 2 (floor along a direction, stage form and a priori form). For every path and every unit phi:
  (a) stage form (the chain), exact up to the per-step remainder of the scalar h_phi:
        <phi, E_T> >= <phi, E_0> - sum_t ell_t(phi) |d_t| - (1/2) L_phi sum_t |d_t|^2;
  (b) a priori form:
        <phi, E_T> >= <phi, E_0> - ell_0(phi) Lambda - (1/2) L_phi Lambda^2,
      and the same bound from any intermediate stage s: <phi, E_T> >= <phi, E_s> - ell_s(phi) Lambda_{s->T} - (1/2) L_phi Lambda_{s->T}^2.
  Both hold with |.| on the left (two-sided).
Proof. <phi, E_{t+1}> - <phi, E_t> = h_phi(theta_t + d_t) - h_phi(theta_t) = <J_t^T phi, d_t> + r_t with
|r_t| <= (1/2) L_phi |d_t|^2 (Taylor with the Lagrange remainder for the scalar h_phi), and |<J_t^T phi, d_t>| <= ell_t |d_t|.
Summing gives (a). For (b) insert Lemma 1, ell_t <= ell_0 + L_phi Lambda_t, and use the identity
sum_t Lambda_t |d_t| + (1/2) sum_t |d_t|^2 = (1/2) Lambda^2 (expand (sum_t |d_t|)^2). The step-size term cancels exactly.  []

Remarks. (i) Part (b) is the time-0 floor of the paper (Thm time0, F_0(ell) - (1/2) H2bar ell^2) for the rank-one
roughness function along phi, with the constant L_phi of the scalar h_phi in place of H2bar; it is not new and the
chain adds nothing to it (the chain is an exact description, its truncations are the appendix's business). What the
chain adds is (a): the floor with the ACTUAL leverages ell_t(phi) along the run, which are observable (for a kernel
eigen-direction, ell_t^2 = lambda_t), so the floor can be carried forward from any stage with the current numbers.
(ii) The a priori form (b) is loose whenever Lambda is large, because it allows every step to lift phi at the maximal
rate L_phi in the best direction. For our runs one sign step has |d| ~ lr sqrt(P_layer) ~ 4 at width 4096, so (b) is
uninformative after a few steps (C1 of EXPERIMENTS.md found the same for the paper's floor: informative at t ~ 1 of
3000). The informative lookahead is therefore on the achievable side (section 2), not on the bound side.

COROLLARY 3 (least path to remove a direction; feasibility from the current stage). The component <phi, E_s> can be
removed by stage T only if ell_s Lambda_{s->T} + (1/2) L_phi Lambda_{s->T}^2 >= <phi, E_s>, i.e.
    Lambda_{s->T} >= ( sqrt(ell_s^2 + 2 L_phi <phi, E_s>) - ell_s ) / L_phi.
For a direction with no leverage (ell_s = 0): Lambda_{s->T} >= sqrt(2 <phi, E_s> / L_phi), a square-root law in the
roughness demanded; for ell_s^2 >> 2 L_phi <phi, E_s> it is the lazy value <phi, E_s> / ell_s.
Proof. Theorem 2(b) from stage s with <phi, E_T> = 0.  []

PROPOSITION 4 (when the one-step-optimal direction is also the lookahead-optimal one). Fix phi with <phi, E> > 0. The
rate of decrease of <phi, E> per unit step is maximal along -J^T phi / ell (first order); the rate of increase of
ell(phi) per unit step is maximal along the top right-singular vector w_phi of M_phi (to first order in the step,
d ell = <J^T phi, M_phi d> / ell for ell > 0, and d ell = ||M_phi d|| for ell = 0). Consider the model in which the
two directions are orthogonal and a step of length rho is split as a along -J^T phi / ell and b along w_phi, a^2 + b^2
<= rho^2, with ell' = ell + L_phi b per step and <phi, E>' = <phi, E> - ell a. Over a path Lambda the removal is
maximised by lifting first and reducing after: lift for a path Lambda_1 = (L_phi Lambda - ell_0)_+ / (2 L_phi), then
reduce, giving removal (ell_0 + L_phi Lambda)^2 / (4 L_phi) when L_phi Lambda > ell_0 and ell_0 Lambda otherwise. The
one-step-optimal rule (a = rho, b = 0 at every step) removes ell_0 Lambda. The two coincide iff L_phi Lambda <= ell_0.
If instead the two directions coincide (the step along -J^T phi also lifts phi: <J^T phi, M_phi J^T phi> < 0, i.e.
the scalar h_phi is concave along its own gradient), reduction and lift happen together and the bound of Theorem 2(b)
is attained; the one-step rule is then lookahead-optimal.
Proof. The first two claims are Cauchy-Schwarz and the definition of the operator norm. In the split model, removal
= int ell dLambda_a with ell nondecreasing and concave in the lift spent; by an exchange argument any schedule is
improved by moving its lift earlier, so the optimum is lift-then-reduce; the one-parameter maximisation gives
Lambda_1. In the coincident case the two rates are attained by the same step, which is the equality case of both
inequalities in Theorem 2.  []

Reading. The lookahead-optimal step differs from the one-step-optimal step exactly when the path left is long compared
with ell_0 / L_phi: a direction with little leverage and much remaining time should be lifted before it is reduced, and
the amount to invest, (L Lambda - ell_0) / (2L), grows with the time left. Whether a given architecture lets the reducing
step also lift (the coincident case) is the sign of <J^T phi, M_phi J^T phi>, the curvature of h_phi along its own
gradient; it is the curvature-sign quantity of THEORY 21(c)2 (REPORT 57: negative at init on the raw residual,
positive after), and it is the mechanism of sharpening (THEORY 24). No closed-form sign is available (21(c)2).

Remark (cross-lift). Lemma 1 bounds the lift from ALL motion. Under gradient flow theta_dot = -J^T E the leverage of a
direction phi with ell(phi) = 0 changes at rate ||M_phi J^T E|| <= L_phi ||J^T E||, which is not zero: gradient flow
does lift dead directions, through motion along the gradient. This is the cross-lift. The statement withdrawn above
ignored it. The per-mode model of section 2 ignores it too and says so.

## 2. The achievable side: the per-mode second-order model and the T-step allocation

The optimizer controls parameter directions, not residual directions, so the model is stated for a set of orthonormal
parameter directions u_k (the modes: for KronQ, u_ij = q_i p_j^T per layer). For each mode three numbers at the current
point:
    g_k = <E, J u_k>            the gradient on the mode (first-order gain per unit step),
    lambda_k = ||J u_k||^2      the Gauss-Newton curvature on the mode (leverage squared),
    c_k = <E, f_{u_k u_k}>      the residual curvature on the mode: the projection of the mode's second derivative
                                on the residual, the term the Gauss-Newton model drops.

LEMMA 6 (loss and lift along one mode). Along theta + s u_k,
    ||E(s)||^2 = ||E||^2 + 2 s g_k + s^2 (lambda_k + c_k) + O(s^3),
    J(s) u_k  = J u_k + s f_{u_k u_k} + O(s^2),  so  lambda_k(s) = lambda_k + 2 s <J u_k, f_{uu}> + s^2 ||f_{uu}||^2 + O(s^3).
Proof. Taylor expansion of f along u_k; the second line is the stage identity of the chain restricted to the mode
(Cross = 2 s <J u, f_uu>, Q = s^2 ||f_uu||^2).  []
So c_k is the part of the mode's lift that the residual sees: the loss curvature along the mode is lambda_k + c_k, not
lambda_k. A mode with lambda_k + c_k < 0 is a direction along which the loss is concave; moving along it in either
direction lowers the loss at second order whatever g_k is. These are the modes on which "investing in leverage" pays
within the current expansion.

MODEL (named assumptions). (A1) The modes do not interact: the loss change over the horizon is the sum over modes of
the per-mode quadratic of Lemma 6 with (g_k, lambda_k, c_k) frozen at the current point. (A2) The per-mode quadratic is
valid for |s| up to a per-mode radius r_k (the Taylor radius along the mode); beyond it the model says nothing.
(A3) The step budget is a sphere: sum_k d_k^2 <= rho^2 per step, with rho set by the remainder level (the controller).
(A4) T steps remain and the per-step pushes are constant in time on each mode (for a convex quadratic even spreading is
optimal by convexity; for a concave mode the push is at the radius anyway), so the total path on mode k is
Lambda_k = T d_k and the budget is sum_k Lambda_k^2 <= (T rho)^2 =: Lambda^2.

THEOREM 7 (the T-step allocation under (A1)-(A4)). Let S_k = sign(-g_k) and write the per-mode decrease as
    D_k(Lambda_k) = 2 |g_k| Lambda_k - (lambda_k + c_k) Lambda_k^2  for  0 <= Lambda_k <= min(T r_k, Lambda).
The allocation maximising sum_k D_k subject to sum_k Lambda_k^2 <= Lambda^2 is: there is mu >= 0 (zero if the budget is
slack) such that
  (i) convex modes, lambda_k + c_k + mu > 0:   Lambda_k = min( |g_k| / (lambda_k + c_k + mu),  T r_k ),
      the ridge step with the FULL curvature lambda_k + c_k and damping mu; a mode is completed (its gradient zeroed)
      when mu is negligible against lambda_k + c_k;
  (ii) concave modes, lambda_k + c_k + mu <= 0:  Lambda_k = min(T r_k, Lambda): the full radius, whatever |g_k|, with
      the sign S_k (either sign if g_k = 0);
  (iii) mu is the unique value making sum_k Lambda_k^2 = Lambda^2 when the unconstrained allocation exceeds the budget.
Proof. D_k is concave on convex modes and convex on concave modes. Lagrangian: sum_k D_k - mu (sum_k Lambda_k^2 - Lambda^2).
On a convex mode the stationarity condition 2|g_k| - 2 (lambda_k + c_k + mu) Lambda_k = 0 gives (i), with the box
constraint active when the stationary point exceeds T r_k. On a concave mode the Lagrangian is convex in Lambda_k, so
its maximum over the box is at an endpoint, and D_k(Lambda_k) - mu Lambda_k^2 > 0 at the upper endpoint whenever
lambda_k + c_k + mu <= 0 (both terms are nonnegative there), which gives (ii). Monotonicity of the total in mu gives
(iii).  []

COROLLARY 8 (what the existing rules are). (a) The frozen-Gauss-Newton ridge / T-step allocation of KRONQ_STEP_NOTE is
Theorem 7 with c_k = 0: all modes convex, Lambda_k = |g_k| / (lambda_k + mu); it allocates to a mode in proportion to
|g_k|, which is sqrt(lambda_k) times the residual coordinate, so slow modes get almost nothing. (b) The sign push with
the cap is Theorem 7 with c_k unknown: the cap |g_k| / lambda_k is regime (i) with c_k = 0 and mu = 0 (fast modes, where
lambda_k >> |c_k|), and the push at a common length is regime (ii) applied to every other mode, i.e. it assumes every
slow mode is concave. (c) The one-step hypergradient on the per-mode weight w_k = exp(u_k), dL/du_k = -lr g_t,k
D_{t-1,k}, has expectation proportional to the realised per-step decrease on the mode, which by Lemma 6 is
2|g_k| d_k - (lambda_k + c_k) d_k^2 for a push d_k; its sign for a slow mode (|g_k| small) is the sign of -(lambda_k + c_k).
So the hypergradient is a stochastic estimator of the regime: it raises w_k on concave modes and lowers it on convex
slow ones. This is why the learned weights beat both the frozen ridge and the plain sign push, and why the learned
profile peaks before the slowest decile: the slowest modes are the ones where c_k does not make the mode concave.

COROLLARY 9 (receding horizon = the chain). Theorem 7 is a plan from the current stage. Over the next stage the numbers
change by the stage identity: g_k by the step and the remainder, lambda_k by Cross_k + Q_k (Lemma 6), c_k by the third
derivative. Re-solving at every stage with the current (g_k, lambda_k, c_k, T_rem) is the chain applied to the plan;
the plan's error over one stage is the change of these numbers over the stage, which is first order in the step for
g_k and lambda_k (Cross) and second order for Q. Nothing in the chain bounds the change of c_k without a third
derivative; it is to be measured (section 3).

PROPOSITION 10 (c_k for the Kronecker modes from factor statistics). For a layer with pre-activation z = W a,
activation h = act(z), and downstream map F from h to the outputs, and the mode u_ij = q_i p_j^T:
    c_ij = <E, f_{uu}> = (1/m) sum_x sum_a (q_i)_a^2 (delta_x)_a act''(z_xa) (p_j . a_x)^2  +  <E, F''[h_eps, h_eps]>,
where delta_x = F'^T E_x is the backpropagated residual into the layer (the same delta as in the S factor, with the
true residual in place of the probe), and h_eps = act'(z) (.) q_i (p_j . a). The first term is, for all (i, j) at once,
    C^(1) = (1/m) (Q (.) Q)^T [ (delta (.) act''(z))^T (A~ (.) A~) ],
with A~ = the rotated inputs (p_j . a_x), already computed by the fast path's forward, (Q (.) Q) the entrywise square of
the S eigenbasis (block-diagonal with blocks), and (delta (.) act''(z)) one backward pass with the residual. The second
term is the downstream curvature along the mode's induced activation change; it vanishes for the last hidden layer
(F linear) and is dropped for earlier layers (assumption A5, to be measured against a Hessian-vector product on a
sample of modes).
Proof. Differentiate f(W + s u) twice at s = 0: z changes by s q_i (p_j . a), h by s act'(z) q_i (p_j . a) + (1/2) s^2
act''(z) q_i^2 (p_j . a)^2, and f by F' of that plus F'' of the first-order change twice; take the inner product with E
and use F'^T E = delta. The factorised form is the sum rearranged.  []
Cost: one matmul of the squared rotation (per block, 256^2 x 256) and one of the squared rotated inputs per layer per
statistics update, plus one backward with the true residual (which the probe backward can share). Memory: one o x i
array per layer, or its EMA. No eigendecomposition.

## 3. Assumptions, their checks, and what the chain cannot give

* (A1) no interaction / no cross-lift. The lift of mode k from motion along other modes is dropped. Lemma 1 says it is
  real and up to L_k per unit path; under the sign push the path is spread over all modes and the cross-lift on any one
  mode is a sum of many terms of mixed sign. Check: compare the measured change of lambda_k over a refresh interval with
  the self-lift prediction 2 d_k <J u_k, f_uu> + d_k^2 ||f_uu||^2 summed over the interval; the residual is the cross-lift.
* (A2) per-mode radius r_k. The third-order term of Lemma 6 is not bounded here. The controller holds the remainder of
  the WHOLE step at kappa; whether the per-mode quadratic holds to the same radius is the ridge-shape measurement
  (STEP_THEORY M2, now valid after the TF32 fix).
* (A3) sphere budget. The Taylor radius is a ball in the rescaled metric by definition of the remainder bound; whether
  the remainder is isotropic over mode directions is the same measurement.
* (A4) constant pushes over the horizon. Exact for the model; the chain (Corollary 9) says the plan is re-made every
  stage anyway, so only the first step of each plan is ever taken.
* (A5) downstream curvature dropped in c_k for the non-final hidden layers.
* Rotation of the modes over the horizon (the moving frame, THEORY 18-19) is ignored by the plan; the transport at each
  refresh carries the state, and the rotation remainder is the chain's named error term.
* What the chain cannot give: an achievability guarantee. Section 1 bounds what no rule can do (and is loose late in
  training, Remark (ii)); section 2 says what the best plan is inside a model whose constants are measured, not derived.
  The gap between them is the cross-lift and the third derivative. A sharper bound than Theorem 2 would need a bound on
  the lift from below, which is architecture-specific (the sign of the residual curvature, 21(c)2), and the paper does
  not claim one.

## 4. The rule, and the experiment that tests it

Rule (one stage; all per layer, in the current Kronecker eigenbasis; names as in kronq_fast.py):
  1. Statistics as now (A, S from the probe), plus C^(1) of Proposition 10 from the residual backward (EMA, bf16 ok).
  2. Per mode: g = rotated momentum (as now), lam = Kronecker or EKFAC curvature (as now), c = C^(1).
  3. Radius rho from the remainder controller (TF32 off in the check), T_rem = steps left.
  4. Solve Theorem 7 with Lambda = T_rem rho and r_k = rho: concave modes (lam + c <= 0) get the full push rho with the
     momentum's sign; convex modes get min(|g| / (lam + c + mu), T_rem rho) / T_rem with mu from the budget (bisection
     as in solve_gamma, over the convex modes only, with the concave modes' allocation subtracted from the budget).
  5. Step = sum of the per-mode pushes; no cap (it is regime (i)), no output-change budget (the radius is rho), no
     hypergradient (replaced by the measured c), no live mask (lam + c decides).
What this costs against the fast path: the C^(1) products (about 0.1 F0 with blocks of 256) and the residual backward
(shared with the probe) in place of the hypergradient state (saves one o x i array per layer).

Experiment (width 4096, both targets, Adam memory, 5000 steps; one node, about 40 min):
  (a) the rule above against the ablation's reference (hyper + cap + budget 0.2);
  (b) the rule with c := 0 (frozen-GN ridge with the radius from the controller), to isolate c;
  (c) the reference with the sign push restricted to modes with measured lam + c <= 0 (the regime test alone);
  (d) diagnostics: the fraction of modes with lam + c <= 0 per lambda decile along the run, and the learned u_k of the
      reference binned by the sign of lam + c (Corollary 8(c) predicts u rises where lam + c < 0).
Decision: if (a) matches or beats the reference on both targets, the tuned rules 6, 7, 9, 11 of the inventory are
replaced by a derived one with two measured constants (kappa, and c per mode); if (d) holds but (a) loses, the model's
constants are right and its horizon handling (A4, Corollary 9) is what is missing.
