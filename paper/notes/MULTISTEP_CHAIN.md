# Multistep training through the chain (written 2026-10-14)

Purpose. The multistep theory (`multistep_training_theory.tex`, 2026-10-06; on the laptop under
`NERSC/multistep/`, not in this repository) states every result at ONE training time ("nothing integrates over a run"),
cannot say how long a window should be ("certificates only, no converse"), and reaches the off-data error only
through a cross-kernel at one instant. This note uses the chain (THEORY.md sections 8-20) to answer three questions:
 Q1  when unrolled training beats one-step training at the target horizons, and by how much;
 Q2  the best window and per-horizon weights, from measurable quantities;
 Q3  the rollout error coming from the off-data part of the model error, which one-step training does not train.
Sources used: THEORY.md sections 0-24, the multistep draft above, `multistep/README.md`, `multistep/results_main.md`.
No other notes were used (user decision, 2026-10-14).

Labels as in THEORY.md: THEOREM (exact, proof given), PROPOSITION (exact under the stated assumptions),
MODEL (closed form in a reduced model, assumptions stated), CHECK (numerical verification run for this note),
CONSISTENT (agrees with a measurement already in results_main.md; not a test), PREDICTION (testable, not run).

## 0. Setting and notation

Learned one-step map G~(x; theta) (Euler x + h H, or direct), true flow map S_h, model error as a function on ALL of
state space E(x; theta) = G~(x; theta) - S_h(x). Starts k = 1..n, steps i = 0..p-1. True states x_{k,i} = u_T,k(t_i);
predicted states u^_{k,i} = u_p,k(t_i); rollout error e_k(t_q) = u_T,k(t_q) - u_p,k(t_q), e(t_0) = 0; one-step errors
eps_{k,i} = u_T(t_{i+1}) - G~(x_{k,i}) = -E(x_{k,i}).
Pi fixed positive diagonal preconditioner; rescaled velocity g^ = Pi^{-1/2} theta_dot.
Features OF THE MAP (slot weight included): Phi(x) = d_theta G~(x) Pi^{1/2} (d x P); operator features
Xi^(k)(x) = d_theta grad^k G~(x) Pi^{1/2}. One-step kernel as a function of two states: k_theta(x, y) = Phi(x) Phi(y)^T (d x d).
On the training states (N = n p d entries): Phi_X, K = Phi_X Phi_X^T / n. Operator errors Delta_k = grad^k E at the
training states; cross kernels K_k0 = Xi^(k)_X Phi_X^T / n (the draft's K_k0 with c_p absorbed into the features).
Learned Jacobians J^_i = grad G~(u^_i) and J_i = grad G~(x_i); propagators P^(t_q, t_i) = J^_{q-1} ... J^_{i+1} and P along the
true states; P(t_{i+1}, t_i) = I. True-flow Jacobians J^S_i = grad S_h(x_i), propagators P_S.
(L z)_q = sum_{i<q} P(t_q, t_i) z_i (block lower triangular, identity blocks next to the diagonal, hence invertible).
Window loss L_w = (1/2n) sum_k sum_q w_q |e_k(t_q)|^2, target V_v the same with weights v (on held-out starts H when stated).
Convention change from the draft: Wbar_w = diag(w_q) WITHOUT the 1/n, and the propagation Gram (new name, section 8)
      M_w = L^T Wbar_w L        (block diagonal over starts, PSD, invertible when w_q > 0 for all q <= p).
One-step loss L_1 = (1/2n) sum_k sum_{i<p} |E(x_{k,i})|^2, on the same true states as the window.

## 1. The exact reduction: unrolled training is one-step kernel training at the predicted states

THEOREM M1 (adjoint form, exact). For any weights w and any theta, the rescaled gradient of L_w is
      g_w = -(1/n) sum_k sum_{i<p} Phi(u^_{k,i})^T lambda^w_{k,i},
with the ADJOINTS lambda^w_{k,p-1} = w_p e_k(t_p),  lambda^w_{k,i} = w_{i+1} e_k(t_{i+1}) + J^_{k,i+1}^T lambda^w_{k,i+1}
(equivalently lambda^w_{k,i} = sum_{q>i} w_q P^_k(t_q, t_i)^T e_k(t_q)). Under preconditioned gradient flow on L_w the learned
map changes, as a function on all of state space, by
      d/dt G~(x) = (1/n) sum_{k,i} k_theta(x, u^_{k,i}) lambda^w_{k,i}        for every x.
One-step training is the same formula with (points, drive) = (x_{k,i}, eps_{k,i}); for p = 1, w_1 = 1 the two coincide.
Proof. d_theta u_p(t_q) = sum_{i<q} P^(t_q, t_i) d_theta G~(u^_i) (the draft's features lemma, Psi^ = L^ Phi^); insert into
grad L_w = -(1/n) sum_k sum_q w_q d_theta u_p(t_q)^T e(t_q) and exchange the sums; the recursion is
P^(t_q, t_i) = P^(t_q, t_{i+1}) J^_{i+1}. Then d/dt G~(x) = Phi(x) g^ with g^ = -g_w.  []
CHECK (2026-10-14; Perlmutter login node, Shifter nersc/pytorch:26.01.01, CPU, seconds, nothing saved): Lorenz-63 RK4
truth, h = 0.02, tanh MLP 3-32-32-3, G~ = S_h + s N(x; theta), n = 8 starts, p = 6, uniform w: adjoint formula against
the autograd gradient of L_w, relative error 2.6e-16 at s = 1e-1, 1e-2, 1e-3.

What transfers verbatim. THEORY.md section 9 states the chain for any update dW_l = -(1/n) sum_x delta_l(x) g(x) a_l(x)^T
with a per-point drive g. Theorem M1 puts unrolled training in exactly that form: data index = (start, step), points =
the predicted states, drive g = -lambda^w. Therefore, with Psi = Phi at the visited predicted states and lambda in place
of the residual, the following hold for unrolled training without change: Thm 2 (kernel = sum over blocks of activation
Gram (.) backprop Gram), Thm 3 (dW_l = (1/n) int sum_{k,i} delta_l(u^_{k,i}) lambda_{k,i} a_l(u^_{k,i})^T: the row space of the
weight change lies in the span of the activations at the PREDICTED states), Thm 4 (unit dynamics), Thm 11 and the spike
theorem (each stage core has rank <= d and is built from the drive's data-by-output structure: under unrolling from
lambda, under one-step training from eps), the stage identity, Thms 12-13 (feature sum, bracket at T), Thm 14 with
||g|| = ||lambda||, Thm 16 (stage error and Taylor radius), Cor 24a (minibatches), Thm 24 / Cor 44 (validation).
What does NOT transfer to lambda: Thm 1. lambda is not the gradient of a loss of the outputs at fixed points (it depends
on the outputs at all later steps, and the points move), so the energy identity, the Dyson series, the moving frame and
Thm 48 do not hold for lambda. They hold instead for the rollout error with the rollout kernel: under gradient flow on
L_w, e_dot = -Psi^ Psi^T Wbar_w e / n exactly (Thm 1 form, self-adjoint in the Wbar_w inner product), and
      K_roll = L^ K^_1 L^^T Wbar_w,   K^_1 = one-step kernel at the predicted states,
so a stage increment of the rollout kernel splits exactly into (a) the stage increment of the one-step kernel at the
predicted states (Cross + Q of the stage identity, plus the motion of the points), transported by the learned
propagators, and (b) the change of the learned propagators (Jacobian learning).

THEOREM M2 (held-out rollout error at every time, exact). For held-out starts H and target weights v, under gradient
flow on L_w on the training starts,
      d/dt V_v = -(1/(n |H|)) sum_{a in H, i} sum_{b, j} lambda^v_{a,i}^T k_theta(u^_{a,i}, u^_{b,j}) lambda^w_{b,j},
and under one-step training the same with (u^_{b,j}, lambda^w_{b,j}) -> (x_{b,j}, eps_{b,j}). Integrated,
V_v(T) - V_v(0) = -int_0^T (the same) ds, and k_theta(s) = k_theta(0) + [k_theta(s) - k_theta(0)] splits every term into lazy
transfer and feature-learning transfer exactly as in Cor 44.
Proof. d/dt V_v = -(1/|H|) sum_a sum_q v_q e_a(t_q) . d_theta u_p,a(t_q) theta_dot = -(1/|H|) sum_{a,i} lambda^v_{a,i}^T Phi(u^_{a,i}) g^
(the computation of Theorem M1 on the held-out rollout); insert g^.  []
Reading. At every instant one-step and unrolled training differ ONLY in the drive field they push through the same
kernel k_theta: one-step pushes the map at the true states by the one-step errors, unrolled training pushes it at the
predicted states by the adjoints. The held-out rollout error responds through the kernel between its own predicted
states and the training points, weighted by its own adjoints on the other side. The draft's Prop "Alignment with the
held-out target" is this formula at one instant; the chain makes it a statement over the run.

## 2. First order in the one-step errors: every window in one set of coordinates

PROPOSITION M3 (span of the unrolled update). Phi(u^) = Phi(x) - Xi^(1)(x)[e] + O(|e|^2) and lambda^w = M_w eps + O(eps^2), so
      g^_w = (1/n) Phi_X^T lambda^w  -  (1/n) Xi^(1)_X^T (lambda^w (x) e)  +  O(|e|^2 |lambda|),
where (lambda (x) e) is the tensor with pairing <lambda (x) e, grad G~> = lambda^T grad G~ e at each (start, step). The first
term is the draft's first-order gradient Phi^T L^T W L eps (up to the 1/n convention). The unrolled update therefore lies
in the span of the one-step value features at the true states up to O(eps^2); at leading order the part outside it is
-(1/n)(I - P_0) Xi^(1)_X^T (lambda^w (x) e), with P_0 the orthogonal projector onto range(Phi_X^T).
Proof. Taylor expansion of Phi(x - e) in Theorem M1; lambda^w = M_w eps + O(eps^2) is the draft's first order of the error,
e = L eps + O(eps^2), inserted into the adjoint.  []
CHECK (same run as M1): the fraction of g_w outside span(Phi_X^T) is 1.07e-4, 1.07e-5, 1.07e-6 at s = 1e-1, 1e-2, 1e-3
(rollout rms error at t_p 0.16, 0.016, 0.0016): linear in the one-step error, as stated.

Consequences (all at first order in eps; M_w taken at its current value, since its change, Jacobian learning, has rate
O(eps) and enters the error at second order):
 (i)   Residual dynamics at the true states: one-step E_dot = -K E; unrolled on w E_dot = -K M_w E (E = -eps at the data).
 (ii)  Kernel-whitened coordinates (new name): x = K^{-1/2} E on range(K) (the part of E in ker K is moved by neither rule
       at this order). For EVERY window, x_dot = -A(w) x with
            A(w) = K^{1/2} M_w K^{1/2} = sum_q w_q A_q,   A_q = K^{1/2} L_q^T L_q K^{1/2}  (per-horizon terms),
       symmetric, PSD and LINEAR in the weights; one-step training is A = K. Any target is V_v = (1/2n) x^T A(v) x.
       The nonzero spectrum of A(w) is that of K M_w, which is the Gauss-Newton curvature of L_w (the quantity of the
       trainability check in results_main.md).
 (iii) Training on the target (w = v): z = M_v^{1/2} E obeys z_dot = -Kt_v(t) z with Kt_v = M_v^{1/2} K(t) M_v^{1/2}, Thm 1's form
       with a PSD kernel whose time dependence is that of the one-step kernel. So the chain's error-at-T statements hold
       for the rollout target with Kt_v in place of K: Thm 35 (Dyson), Thm 36, Lemma 45 (moving frame), Thm 46, Thm 48
       (cutoff bound with the leakage as named remainder), Cor 49 (crossing condition; in Thm 13's bracket take the
       direction u = M_v^{1/2} zeta, not normalised).
 (iv)  CONSISTENT: results_main.md reports that for Euler models "unrolled training changes where the error points, not the
       learned Jacobian along it" (rho - 1 = rho^f - 1 to about 10 %; the error moves into contracting directions), at
       windows <= 0.64 where the nonlinear part is < 1 %. Prop M3 says why: at first order the unrolled update lies in the
       same value-feature span as one-step training, so the Jacobian is reached through the same cross kernel K_10 as under
       one-step training (section 5); what changes is the drive, M_w eps instead of eps, which reweights the one-step errors
       towards the patterns the propagators amplify.

## 3. Q1: when unrolling helps, and by how much

Fair comparison. Multiplying a loss by a constant rescales time, so rules are compared at equal stability: gradient
descent at learning rate c / lambda_max of the rule's own Gauss-Newton curvature, 0 < c <= 1 (the curvature-scaled "inv"
rule of follow-up 6 is the nearest experiment). Write Ahat = A / lambda_max(A) and nu_m = its eigenvalues (the
normalised spectrum).

PROPOSITION M4 (rate at one instant, exact at first order). Per normalised step the target changes by
      one-step:    -c E^T M_v K E / (n lambda_max(K)),
      unrolled, w = v:  -c ||K^{1/2} M_v E||^2 / (n lambda_max(K M_v))  <= 0,
and the one-step rate has no sign (the draft's Prop "Norms do not determine the sign"). Unrolling on the target is
faster at this instant iff  ||K^{1/2} M_v E||^2 / lambda_max(K M_v)  >  E^T M_v K E / lambda_max(K).
Reading. After one-step training to time t_1 the residual sits at the front, lambda_j t_1 ~ 1 (THEORY Prop 8a; Thm 48), so
the right side is of order E^T M_v E / (t_1 lambda_max(K)): continued one-step training is as slow on the rollout target as
it is at its front. The left side is large when M_v carries that slow residual onto directions K already fits fast. This
MIXING of the held rough error into fast kernel directions by the propagation Gram is what unrolling exploits; the
draft's Prop normsign showed the size of the one-step error alone cannot decide.

THEOREM M5 (the gain in cutoff form; first order, frozen K and L, i.e. lazy). After S gradient-descent steps at
normalised learning rate c,
      V_v^(w)(S)   = (1/2n) x_0^T (I - c Ahat(w))^S A(v) (I - c Ahat(w))^S x_0          (exact),
      V_v^(one)(S) = (1/2n) E_0^T (I - c Khat)^S M_v (I - c Khat)^S E_0                   (exact).
For w = v, in the eigenbasis of A(v) (eigenvalues alpha_m, coefficients xhat_m of x_0), with the cutoff set
u(S) = {m : c S alpha_m / alpha_max < 1} and c <= 1/2,
      e^{-4} U  <=  2n V_v^(v)(S)  <=  U + e^{-2} F,     U = sum_{m in u} alpha_m xhat_m^2,  F = sum_{m not in u} alpha_m xhat_m^2:
the target error after S steps is, within the factor e^4, the target energy of the initial residual on the slow part
of the window's normalised curvature. For one-step training the same bookkeeping in the eigenbasis of K gives the
target metric M_v applied to the residual one-step training leaves, i.e. its slow part E_u (cutoff in K) plus cross
terms between slow and fit modes that decay at least like (1 - c lambda_l / lambda_1)^S on the fit side.
Proof. Iterate x_{s+1} = (I - c Ahat) x_s; for w = v the matrices commute; (1 - y)^{2S} >= e^{-4 S y} >= e^{-4} for y <= 1/2 and
S y < 1, and (1 - y)^{2S} <= e^{-2 S y} <= e^{-2} for S y >= 1.  []
Q1, answered at this level: unrolling on the target helps (to within e^4) when the target-weighted held rough error of
one-step training, ||E_u^(one)(S)||^2_{M_v}, exceeds U, the target energy on the slow part of A(v); "by how much" is the
ratio of the two. Both are computable at the start of a fine-tune from K, L (hence M_v) and E at the true states.

With feature learning (the chain).
 - Targeting. By Theorem M1 the stage cores of unrolled training are driven by lambda ~ M_v eps, so the lift (Q >= 0 on the
   driven directions, section 8 (ii); Thm 13's bracket) is aimed at the target-weighted residual. One-step training lifts
   along eps. A target direction zeta (unit, z coordinates) is fit when the crossing condition of Cor 49 holds:
   (||D_u(t)|| -/+ ||Phi_0^T u||)^2 t / n = 1 brackets its crossing time, u = M_v^{1/2} zeta, D_u = sum_s dPhi_s^T u. Unrolling
   helps along zeta when this crossing time is earlier under w = v than under one-step training: a comparison of two
   accumulated coherent drives along the same u.
 - Theorem M5 is the frozen-kernel (lazy) statement. With the kernel moving, Thm 48 in the z coordinates of section 2
   (iii) brackets V_v^(v)(T) with the leakage across the cutoff as the named remainder, and Thm 46(a) gives "at least the
   frozen rate" under per-stage positivity of the increment of Kt_v on the residual's subspace, which is the same open
   hypothesis as THEORY Cor 43(ii) (now for Kt_v).

## 4. Q2: the best window and weights

THEOREM M6 (best weights when the curvatures commute; first order, frozen). Suppose A(v) and every candidate A(w) share
an eigenbasis (diagonal model: modes j with one-step kernel eigenvalue lambda_j and propagation-Gram eigenvalue
m_j(w) = sum_q w_q m_{j,q}, so alpha_j(w) = lambda_j m_j(w)). For gradient descent at learning rate c / alpha_max(w), c <= 1,
      V_v(S; w) = (1/2n) sum_j alpha_j(v) xhat_j^2 (1 - c r_j(w))^{2S},     r_j(w) = alpha_j(w) / max_l alpha_l(w)  in (0, 1],
      V_v(S; w) >= (1/2n) (1 - c)^{2S} sum_j alpha_j(v) xhat_j^2,
with equality iff r_j(w) = 1 for every j with alpha_j(v) xhat_j != 0. The best weights make the window's normalised
curvature FLAT on the modes that carry the target error, and such weights are best for every budget S at once. If no w
is flat there, maximising the worst rate min_{j in supp} r_j(w) is a LINEAR PROGRAM in w (alpha_j(w) is linear in w):
      maximise t  subject to  sum_q w_q lambda_j m_{j,q} >= t  (j in supp),   sum_q w_q lambda_l m_{l,q} <= 1  (all l),   w >= 0.
Proof. 1 - c r is in [0, 1] and decreasing in r for c <= 1, so each term is smallest at r = 1.  []
Without commutation the same criterion reads: maximise t subject to V_T^T A(w) V_T >= t I and A(w) <= I, w >= 0, with V_T a
basis of the target-carrying subspace. Because A(w) is linear in w this is a semidefinite program. It is the natural
criterion (it maximises the worst normalised rate on that subspace) but it is NOT proved optimal without commutation:
the cross terms of Theorem M5 enter. The exact greedy rule for one step at a time, without commutation, is to maximise
E^T M_v K M_w E / lambda_max(K M_w) (Prop M4 for w != v).

MODEL M7 (two groups of modes: the crossover window). Group A is held (slow in K: lambda_A) and amplified
(m_A(tau) = c_A(tau) e^{2 Lambda_A tau}). Group B is fit (lambda_B >> lambda_A) and less amplified
(m_B(tau) = c_B(tau) e^{2 Lambda_B tau}, Lambda_B < Lambda_A). Here tau is the window length with uniform weights, Lambda
are finite-time growth rates over the window, and c_j(tau) varies slowly. Then r_A = r_B = 1 exactly where
lambda_A m_A(tau) = lambda_B m_B(tau), i.e. at
      tau_x = [ ln(lambda_B / lambda_A) + ln(c_B / c_A) ] / (2 (Lambda_A - Lambda_B)),
and by Theorem M6 tau_x is the best window for every budget. Shorter windows leave A slow
(r_A = e^{-2 dLambda (tau_x - tau)}), longer ones freeze B (r_B = e^{-2 dLambda (tau - tau_x)}).
Reading: the best window is set by two quantities. The numerator is the log of the one-step kernel's eigenvalue ratio
between the error one-step training has fit and the error it still holds, a roughness quantity: how slow the held
rough error is for the current features. The denominator is twice the gap in finite-time growth between the two, a
dynamics quantity. With the front law (THEORY Prop 8a, Thm 48: after one-step training to time t_1 the held error sits
at lambda_A t_1 ~ 1),
      tau_x ~ [ ln(lambda_B t_1) + ln(c_B / c_A) ] / (2 dLambda).
PREDICTION: the best window grows LOGARITHMICALLY with how far one-step training has gone. That means the time t_1, or
anything else that pushes the front to slower modes (more data, more width: more is fit and the remainder is rougher).
Each e-fold in t_1 lengthens it by 1 / (2 dLambda).
CONSISTENT (not tests):
 - Follow-up 6: the gain of the 1.28 window over the 0.32 window grows from width 64 to 512.
 - Curriculum: lengthening the window in stages helped.
 - Damping w_q = exp(-gamma t_q), with gamma chosen to equalise HORIZONS, gave one-step performance. Theorem M6 requires
   flatness across MODES. Equalising horizons by undoing the growth removes the amplification that separates A from B,
   which returns A(w) to K's ordering.
 - Long windows at a fixed learning rate made no progress (results_main item 8). At window 1.28, A(w) is nearly rank one
   (tr / lambda_max = 1.1-1.5), so r_j ~ 0 for every mode but the top. This is the cutoff of Theorem M5 in the normalised
   spectrum.
Order of magnitude only (not a fit): the measured curvature grows about like e^{11 tau} early and towards 2 Lambda_1 = 1.8
later, so dLambda between about 1 and 5; with lambda_B / lambda_A between 1e2 and 1e6, tau_x is between about 0.5 and several
time units. The measured best windows (0.32-0.64 at a fixed learning rate, >= 1.28 with curvature-scaled rates) lie in
that range. Both inputs are measurable (section 7).
Weights beyond windows. Since A(w) = sum_q w_q A_q, the program of Theorem M6 chooses per-horizon weights directly; a window
of p steps is the special case w_q = 1/p for q <= p and 0 beyond.

## 5. Q3: the off-data part of the model error, and a floor on what one-step training leaves in it

THEOREM M8 (true-propagator form, exact). e_k(t_q) = sum_{i<q} P_S^sec(t_q, t_i) [ eps_{k,i} - O_{k,i} ], with the OFF-DATA
error O_{k,i} = E(u^_{k,i}) - E(x_{k,i}) = -Deltabar_{1,k,i} e_k(t_i), where P_S^sec are products of the secant Jacobians of
the TRUE flow B^S_i = int_0^1 grad S_h(x_i - tau e_i) dtau and Deltabar_1 is the secant Jacobian error of the learned map.
Proof. e(t_{i+1}) = S_h(x_i) - S_h(u^_i) - E(u^_i) (the draft's Lemma "Rollout error: true separation plus model error")
= B^S_i e(t_i) + eps_i - O_i.  []
The rollout error is the on-data error eps and the off-data error O, both carried by the true dynamics. At first order
e^(1)(t_q) = sum_i P_S(t_q, t_i) eps_i + sum_j P_S(t_q, t_j) Delta_{1,j} e^(1)(t_j) (a rearrangement of the draft's Prop
"Learned and true propagators"). The derivative of the target with respect to the Jacobian error is the tensor
      phi*_v = ( lambda^v_j (x) e(t_j) )_j        (adjoint (x) error; new name: rollout test tensor),
since d e(t_q) / d J_j [delta] = P(t_q, t_j) delta e(t_j) for q > j. This is the direction in which the Jacobian error
matters for the target.

THEOREM M9 (what one-step training does to an operator error; the untransferable subspace; exact). Under one-step
gradient flow, for every test tensor phi on the training states and every k >= 1,
      <phi, Delta_k(T)> = <phi, Delta_k(0)> - int_0^T <K_k0(0)^T phi, E(s)> ds - int_0^T <[K_k0(s) - K_k0(0)]^T phi, E(s)> ds
(lazy transfer + feature-learning transfer: Cor 44 with the k-th operator in place of the validation outputs), and by
the feature-sum identity (Thm 12),
      [K_k0(s) - K_k0(0)]^T phi = (1/n) ( DPhi(s) Xi_0^(k)T phi + Phi_0 DXi^(k)(s)^T phi + DPhi(s) DXi^(k)(s)^T phi ),
DPhi(s) = Phi_X(s) - Phi_X(0) and likewise DXi. Define the UNTRANSFERABLE SUBSPACE (new name)
      N_k = ker K_k0(0)^T = { phi : Phi_0 Xi_0^(k)T phi = 0 },
the test tensors whose operator features have no overlap with the value features at initialisation. For phi in N_k the
lazy term vanishes identically, and
      |<phi, Delta_k(T)> - <phi, Delta_k(0)>| <= (1/n) int_0^T [ ||DPhi(s)|| ||Xi_0^(k)T phi|| + (||Phi_0|| + ||DPhi(s)||) ||DXi^(k)(s)^T phi|| ] ||E(s)|| ds,
with ||DPhi(s)|| <= sup ||d_theta Phi_X|| l(s) and ||DXi(s)^T phi|| <= sup ||d_theta Xi^T phi|| l(s), l(s) = int_0^s |g^| the path
length (l(s) <= sqrt(s/2) ||E_0||, the main text's Prop "ell"). Taking phi = Pi_{N_k} Delta_k(0) / ||Pi_{N_k} Delta_k(0)||:
      ||Delta_k(T)||  >=  ||Pi_{N_k} Delta_k(0)||  -  (feature-learning transfer above)                      (rough-label floor).
The untransferable part of the initial operator error is lost only through feature learning, at a rate first order in
the accumulated feature change times the residual. In the lazy regime it is exactly frozen. With the output layer
initialised at zero (as in the multistep runs) Delta_k(0) = -grad^k U, so the floor is the part of the operator TARGET
that the value features cannot see (grad U = h grad f + O(h^2) for Euler, I + O(h) for direct; the draft's Prop "Local
complexity of the operator targets").
Proof. Delta_k_dot = Xi^(k)_X g^ with g^ = -(1/n) Phi_X^T E; integrate; expand the product of the summed features; bound
by Cauchy-Schwarz and the mean value inequality along the path.  []

COROLLARY M10 (unrolled training reaches N_k only through feature learning at first order; its own channel is second
order). Under gradient flow on L_w, Delta_k_dot = Xi^(k)_X g^_w with g^_w from Prop M3. The first term is Theorem M9 with
drive -lambda^w (= -M_w eps at first order) in place of E: the SAME transfer operator K_k0 with a different drive, so N_k is
untouched by it in the lazy regime. The second term moves Delta_k along -(1/n) Xi^(k)_X Xi^(1)_X^T phi*_w: unrolled training trains the Jacobian along the
rollout test tensor "adjoint (x) error" directly, with strength second order in the one-step errors. The UNTRANSFERABLE
FRACTION (new name)
      chi_w = ||(I - P_0) Xi^(1)_X^T phi*_w||^2 / ||Xi^(1)_X^T phi*_w||^2   in [0, 1]
is the fraction of the target's Jacobian-error gradient that one-step training cannot reach without feature learning.
chi_w near 1 together with a large Jacobian-error part of the rollout error (Theorem M8) is the regime where unrolling
is NECESSARY rather than only faster.
Size of the second-order channel relative to the first: |e| ||Xi^(1)|| / ||Phi||, i.e. |e| over the scale on which the
features vary (the draft's radius r). CONSISTENT: the draft measured the higher-order part of the gradient
|g_w - grad^(1)_w| / |grad^(1)_w| at about 1e-4 for windows <= 0.32, 1e-3 at 0.64, 0.01-0.18 at 0.96-1.44. So the direct
off-data channel is negligible for windows <= 0.64 and opens near 1-1.5, where long windows began to pay off once the
learning rate was scaled (a coincidence of ranges, not a causal test).

PROPOSITION M11 (Euler versus direct in these terms; from THEORY Thms 20-22). Same network, outputs scaled as in
`multistep/README.md`. The direct map's features carry no factor h and Euler's carry h, so K_D = kappa h^{-2} K_E with kappa
the ratio of the fixed output scales, and at equal propagators A_D(w) = kappa h^{-2} A_E(w).
 (i)   The rollout Gauss-Newton curvature of direct is h^{-2} times Euler's at the same window. CONSISTENT: measured
       growth like dt^-2 (direct / Euler at window 0.32 rises 64-fold from dt = 0.08 to 0.01).
 (ii)  The normalised spectra are equal. At stability-normalised steps Euler and direct therefore differ at first order
       only through the residual (the identity phase, Thm 21), the learned propagators L, and feature learning.
 (iii) As h -> 0 direct training is lazy (Thm 22), so by Theorem M9 its untransferable Jacobian error is frozen at
       ||Pi_{N_1} grad U||. Its Jacobian target contains the h-independent part I; whether the untransferable part of I is
       nonzero is a property of the features at initialisation (measurable, section 7). Euler's Jacobian target is O(h),
       and Euler trains rich. CONSISTENT: the per-step Jacobian error along the error, h R_e(H - f_h) from results_main,
       is 0.64 / 0.58 / 0.56 / 0.48 for direct at h = 0.01 / 0.02 / 0.04 / 0.08 (about h-independent), and for Euler
       about 0.05-0.11 at h = 0.01 rising to 0.4-0.9 at h = 0.08 (about proportional to h).

## 6. Status

   M1 adjoint form, M2 held-out rate         exact, any order, any time; M1 checked to 1e-16
   transfer list after M1                     exact (section 9 setting with data index (start, step)); Thm 1-based
                                              statements hold for e with K_roll, not for lambda
   M3 span                                    exact expansion; O(eps^2) remainder bounded under the draft's Assumption
                                              "Analytic extension" (its Thm "Bound on g_w - grad^(1)_w"); checked (linear scaling)
   section 2 (i)-(iii)                        first order in eps, M_w frozen (its change enters at second order)
   M4                                         exact at first order
   M5                                         exact for frozen K, L; with feature learning = Thm 48 in z coordinates,
                                              conditional on the leakage and on per-stage positivity (as THEORY Cor 43(ii))
   M6                                         exact when the curvatures commute; the SDP is the natural criterion,
                                              optimality without commutation NOT proved
   M7                                         two-group model; the front-law input is a measured regularity (n = 200 caveat
                                              of THEORY.md applies)
   M8, M9, M10                                exact (M9's bound with path suprema as constants)
   M11                                        (i)-(ii) exact at equal propagators; (iii) uses the lazy-training theorem (Thm 22)
Open. (a) Second order in eps: the motion of the points and of L. Theorem M1 is exact there, but the common coordinates
of section 2 are not. (b) Adam: every first-order statement holds with K_P at fixed P, but along the path the dP term
of the CORRECTION to Thm 17 enters. (c) Errors beyond the radius: nothing. (d) The statistics of minibatch drives.
(e) Optimality of the SDP without commutation. (f) The front law itself (open in THEORY.md).

## 7. Measurements that would decide (none run; each needs explicit permission, account, QOS, GPUs and time first)
 1. Inputs of Q1/Q2 on the existing Lorenz-63 checkpoints ($SCRATCH/multistep_roughness), one-step Euler models at
    dt = 0.04 and 0.08, a few hundred starts. Measure:
     - the spectrum of K at the true states;
     - the per-horizon A_q;
     - the commutation defect ||[A(v), A(w)]|| / (||A(v)|| ||A(w)||);
     - the target-carrying subspace;
     - U and the one-step held energy of Theorem M5;
     - tau_x and the LP/SDP weights.
    Then compare with the measured best windows. Dense eigendecompositions of a few-thousand-entry kernel: minutes.
 2. The logarithmic law: fine-tune from the saved log-spaced one-step checkpoints (several t_1) at several windows with
    curvature-scaled rates, and locate the best window per t_1. Needs training: GPU, interactive allocation.
 3. chi_w and ||Pi_{N_1} Delta_1(0)|| for Euler and direct across dt (one-step models; one QR of Phi_X^T with P = 133k).
 4. Theorem M5's bracket against the check-(v) data (the realised first-order-optimal steps).

## 8. New names (proposed; to be added to THEORY.md once approved, per the repository rule)
adjoint lambda^w (already used as "adjoint pass" in multistep/README.md), propagation Gram M_w, kernel-whitened coordinates x,
per-horizon terms A_q and window curvature A(w), rollout test tensor phi*, off-data error O, untransferable subspace N_k,
untransferable fraction chi_w, crossover window tau_x.
