# What can be proved, what can be stated under assumptions, and what the data say

> **NOTE (2026-10-11, user decision): every experiment in this document was run with n = 200 training points
> (exceptions: four n = 400/800 ground-truth designs used only for the gain-exponent check). At n = 200 the training set
> is nearly fit within the budget, the residual-adapted late phase is short, the depth-6-8 "floor" is a sample-size
> artefact, and the negative lazy-transfer sign in the validation split is overfitting of a 200-point sample. This is a
> toy version of a toy problem. ALL EXPERIMENTS MUST BE REDONE at a sample size that leaves a genuine late stage
> (n in the thousands at least), before any measured number here is quoted. The theorems carry n as a symbol and are
> unaffected; the verifications, bulk ratios, exponents, signs, transfer fractions and the ranker's fits are all
> n = 200 statements until redone. The exact staged model does not scale to large n (eigendecomposition per stage);
> the ground-truth GD harness does, so the redo is training-only.**

Status 2026-10-10 (night). Companion to WRITEUP.md (sections 2-10) and REPORT.md. Every statement is labelled
THEOREM (exact, proof given), PROPOSITION (exact under the stated assumptions), or FINDING (measured; sample given).

## 0. Setting and notation
Network f(x; theta) in R^{ny}, tanh MLP with L weight matrices W_0..W_{L-1} and biases; "depth d" means d hidden layers,
L = d+1. Data x_1..x_n, targets y_i. Loss L(theta) = (1/2n) sum_i ||f(x_i) - y_i||^2. Gradient flow theta' = -grad L.
Stacked residual E in R^{n ny}, E_{(i,c)} = f_c(x_i) - y_{i,c}. Jacobian J = d f / d theta (n ny x P). Kernel K = J J^T / n.
Per layer l: incoming activations a_l(x) in R^{H_{l-1}} (a_0 = x), augmented with 1 for the bias; preactivations z_l(x);
backprop signals delta_l^c(x) = d f_c(x) / d z_l(x) in R^{H_l}; for hidden units, delta_q^c = tanh'(z_q) m_q^c with
m_q^c(x) = (W_{l+1}^T delta_{l+1}^c(x))_q the incoming backward signal.

## 1. Exact statements

THEOREM 1 (residual dynamics). Under gradient flow with the squared loss, E' = -K(t) E. Hence
d/dt ||E||^2 = -2 E^T K E <= 0 and E(T) = Texp(-int_0^T K(s) ds) E(0) (time-ordered exponential).
Proof. theta' = -(1/n) J^T E; E' = J theta' = -(1/n) J J^T E = -K E.  For a general loss l(f, y): f' = -K grad_f l.  []

THEOREM 2 (layer decomposition). K = (1/n) sum_l A_l (.) D_l, where A_l[i, j] = a_l(x_i) . a_l(x_j) + 1 (n x n) and
D_l[(i, c), (j, e)] = sum_q delta_{l, q}^c(x_i) delta_{l, q}^e(x_j), with (.) the Hadamard product over the data index.
Proof. d f_c(x) / d W_l[q, p] = delta_{l,q}^c(x) a_{l,p}(x) and d f_c(x)/d b_l[q] = delta_{l,q}^c(x). Summing products of
these over (q, p) and over the bias gives the layer-l block of J J^T; the sum over p of a_p(x_i) a_p(x_j) plus 1 is A_l.  []

THEOREM 3 (data-indexed weight change; coherence). W_l(t) - W_l(0) = -(1/n) int_0^t sum_i sum_c delta_{l}^c(x_i, s)
E_c(x_i, s) a_l(x_i, s)^T ds. Hence at every time the weight change lies in the span of the n ny outer products
(backprop signal at a data point) x (incoming activation at that data point), its rank is at most min(n ny, H_l, H_{l-1}),
and its row space is contained in the span of the incoming activations on the data.
Proof. The gradient of L with respect to W_l is (1/n) sum_i sum_c delta_l^c(x_i) E_c(x_i) a_l(x_i)^T; integrate.  []
Remark. This is the "coherence" used throughout: every layer's change is driven by the same residual E through its own
backprop factor; the product structure is what makes the kernel change low-rank per layer (FINDING F4 in WRITEUP) and
composed across layers (F5).

THEOREM 4 (unit dynamics; the exact init-only model). For a hidden unit q in layer l,
   z_q'(x) = sum_p W_l'[q, p] a_{l,p}(x) + b_q'            (coherent, given by Theorem 3 with the unit's own delta_q)
          + sum_p W_l[q, p] a_{l,p}'(x)                     (propagation of the change of the layer below),
   m_q^c'(x) = sum_r W_{l+1}'[r, q] delta_{l+1, r}^c(x)     (coherent, Theorem 3 for layer l+1)
          + sum_r W_{l+1}[r, q] delta_{l+1, r}^c'(x)        (propagation of the change of the layer above).
For l = 0 the forward propagation term vanishes (a_0 = x is fixed); for the last hidden layer the backward propagation
term vanishes (delta at the output is the identity). Together with Theorem 1 and the table rule of Theorem 2 this is a
closed system in (E, {z_q, m_q}, {W_l for hidden-hidden l}) on the data; its solution is the gradient-flow solution.
Proof. Differentiate z_q = W_l[q, :] a_l + b_q and m_q = W_{l+1}[:, q] . delta_{l+1}.  []
FINDING 4a. The coherent terms alone (omitting both propagation terms) reproduce the real network's unit states at
t = 8 (reference H96 d2 Delta 0.5) only to 11 % (z) and 28 % (m) of their displacement; with the propagation terms the
staged integration reproduces them to 1.5e-3 relative, every kernel eigenvalue to 0.1 %, and ||E|| to four digits;
it tracks E(t) and the gains to 1-2 % through t = 670 and E(T) at depth 8 (0.0063 vs 0.0063). The stage criterion is
converged (budgets 0.03/0.01/0.003 agree to 3 digits). Ground truth is gradient flow (4x smaller step: 4-digit agreement).

PROPOSITION 5 (Hellmann-Feynman rate and the lift bound). For a simple eigenpair (lambda_k, v_k) of K,
lambda_k' = v_k^T K' v_k, and K' = (1/n) sum_l (A_l' (.) D_l + A_l (.) D_l'), where A_l', D_l' are linear in the unit
velocities of Theorem 4, which are linear in E. Hence |lambda_k'| <= C(t) ||E(t)|| with C(t) depending on the current
features, and lambda_k(T) - lambda_k(t0) <= int_{t0}^T C(s) ||E(s)|| ds.
Proof. Standard first-order perturbation; linearity by inspection of Theorems 3-4.  []
Remark. C(t) is not bounded a priori (features grow), so this is structural: the lift is driven by the integrated
residual (the "drive" of the reduced models), which is the reason no quantity at init fixes the late kernel.

## 2. Statements under assumptions (what an early- or init-based predictor would need)

PROPOSITION 6 (frozen kernel as an upper bound). Assume on [t0, T] the eigenvectors of K are constant and every eigenvalue
is nondecreasing. Then ||E(T)|| <= ||exp(-(T - t0) K(t0)) E(t0)||, mode by mode.
Proof. In the fixed basis E_k' = -lambda_k(t) E_k, so |E_k(T)| = |E_k(t0)| exp(-int lambda_k) <= |E_k(t0)| exp(-(T-t0) lambda_k(t0)).  []
FINDING 6a (82 GD runs, squared loss). The bound holds in 82/82 runs for t0 <= 10 % of T (min ratio prediction/truth
1.025) and in 79/82 at t0 = 30 % (min 0.969): the fixed-basis assumption is mildly violated late. As a predictor it is
loose: median 16x at t0 = 0.3 %, 9.7x at 1 %, 4.4x at 3 %, 2.2x at 10 %, 1.3x at 30 %.
Remark. Without the fixed-basis assumption the statement is false in general (rotating kernels can beat the frozen
bound), so no proof is offered; the data show the violations are small and late.

PROPOSITION 7 (gain-law extrapolation). Under the fixed-basis assumption, if lambda_k(t) = lambda_k(t0) (t/t0)^{beta_k} on
[t0, T], then ||E(T)||^2 = sum_k E_k(t0)^2 exp(-2 lambda_k(t0) t0 ((T/t0)^{beta_k + 1} - 1)/(beta_k + 1)).  (Direct.)
FINDING 7a. With beta_k measured between t0/3 and t0: median prediction/truth 9.6x (t0 = 0.3 %), 4.6x (1 %), 1.5x (3 %),
1.12x (10 %), 1.01x (30 %), but the 10-90 % band is 0.4-3.7x even at 10 %; pairwise ordering of designs within 1.1x is
59-65 % (chance 50 %), within 1.1-1.3x 65-76 %. The naive predictor "compare E(t0)" orders pairs equally well.

PROPOSITION 8 (self-similar front). Assume lambda_k(0) ∝ k^{-a} and lambda_k(t) = lambda_k(0) g(t) with g ∝ t^{beta} for
the modes that matter. The residual front, defined by lambda_{k*}(t) t = const, moves as k*(t) ∝ t^{(1 + beta)/a}.  (Direct.)
FINDING 8a. Measured front (rank where half the residual energy sits): d log k*/d log t = 0.19, 0.22, 0.18, 0.12 per
decade (medians; 10-90 % 0.05-0.35), i.e. nearly constant and predictable: k*(T) extrapolated from t0 = 1-30 % of T is
within 10 % of the measured one in the median. The error is NOT recoverable from k*: E(T) estimated as the target's
energy beyond the front spans 0.02-35x of the truth (t0 = 1-3 %) and 0.7-3.9x (30 %), because the target spectrum in the
kernel basis is steep and the lifted eigenvectors rotate out of the init basis.

## 3. The negative statement, and its status

CONJECTURE 9 (no early predictor below the seed spread). For this class (tanh MLPs, gradient flow, squared loss), the
late training error E(T) is not a function of the early history {K(t), E(t) : t <= t0} for t0 <= 0.1 T to within the
initialization-to-initialization spread of E(T).
Evidence (all GD runs, squared loss):
 (i) Seed spread. 17 designs with two seeds each: E(T) ratio median 1.155, range 1.009-1.464 (train); validation 1.01-1.38.
 (ii) Early agreement. The same seed pairs agree in E(t) to |log ratio| 0.022 (median) for t <= 0.1 T, diverge to 0.076 at
      0.3 T and 0.144 at T. Their rank-100 gains differ from the start (|log ratio| 0.18 at 0.003 T, 0.36 at 0.01 T), but
      the early tail gain does not predict which seed ends lower (53-65 % of 17 pairs, chance 50 %).
 (iii) Predictors. Frozen kernel, gain-law extrapolation, front extrapolation, and E(t0) itself all order designs within
      1.1x at 55-65 % and within 1.1-1.3x at 65-76 % for t0 <= 0.1 T (82 runs, 158 and 251 pairs).
 (iv) Mechanism (Theorems 4-5). The late kernel is the integral of a drive (the residual) through init-specific propagation
      factors W_l, W_{l+1}; the early residual is insensitive to those factors (ii), so they are not identifiable from it.
What would make it a theorem: a construction of two initializations with identical (K, E) histories on [0, t0] to O(eps)
and E(T) ratios bounded below by a constant; the propagation term of Theorem 4 is the natural place to build it. Not done.

## 4. What the diagnostic can therefore be (consequence, not theorem)
- With one random initialization and the subsample: the staged system of Theorem 4 (own-init model) is exact up to the
  stage discretization (converged) and the subsample assumption; cost per stage one (n ny)^2 eigendecomposition plus
  H^2 n per hidden interface, ~1e4 stages, independent of parameter count and dataset size. It resolves designs down to
  the seed spread, which is the resolution limit of any single-run comparison (Finding 9 i).
- Without an initialization: the coherent theory (Theorems 2-3, Findings F1-F9) gives the gain exponent from the target
  roughness, the composition across layers and the ordering across roughness classes; its error is ~2x at H = 96 and the
  init-specific part decreases with width (mean-field limit).
- From early training: nothing beyond "compare E(t0)" at the resolutions tested (Finding 9 iii); the front rate is a
  genuine regularity (Finding 8a) but carries the position of the residual, not its size.
- At 1-2 % resolution: only the seed-averaged error is defined; predicting it needs the own-init model on several inits,
  or training several times. Validation error is outside all of the above.

## 5. Why the late error cannot be approximated from early training: the precise statement (added 2026-10-11)

The strict version ("E(T) is not a function of the early history") is FALSE, and the reason is itself a theorem:

PROPOSITION 9a (determinism). For a tanh network under gradient flow, theta(t) is real-analytic in t (analytic vector
field, analytic dependence of ODE solutions), hence so are K(t) and E(t). Two trajectories that agree on any interval
[0, t0] agree for all t (identity theorem). So the early history determines E(T) exactly, and the obstruction to
predicting it is not identifiability but conditioning: extrapolating an analytic function from an interval is ill-posed.

PROPOSITION 9b (visibility). Let B be the set of modes (eigenbasis of K(t0)) that carry a fraction phi of ||E(T)||^2.
(i) Their share of the loss-decrease rate at t0 is s_B(t0) = sum_B lambda_k E_k^2 / sum_all lambda_k E_k^2 <= max_B lambda_k(t0) / lambda_eff(t0).
(ii) In the fixed-basis model, E_k(t) = E_k(0) exp(-Lambda_k(t)) with Lambda_k(t0) = int_0^{t0} lambda_k, so on [0, t0] the
band components are constant to relative order Lambda_k(t0) ~ lambda_k(t0) t0; the early residual history constrains
lambda_k only to relative precision delta / (lambda_k(t0) t0), where delta is the relative precision of the history.
Predicting E_k(T) to relative precision eps needs Lambda_k(T) ~ 1 to precision eps, hence delta <= eps lambda_k(t0) t0.
Proof: (i) is the energy identity (Theorem 1) restricted to B; (ii) is the closed form of the fixed-basis solution.  []
MEASURED (81 GD runs, band = modes carrying the central 50 % of ||E(T)||^2):
   t0 = 1 % of T:  lambda_k(t0) t0 median 1.4e-6 (10-90 % 4e-9 .. 8e-4);  share of the loss-decrease rate 8e-7.
   t0 = 10 % of T: lambda_k(t0) t0 median 1.0e-2 (8e-5 .. 4e-2);          share of the loss-decrease rate 7e-4.
So at 1 % of the budget the modes that will carry half of the final error contribute one part in 10^6 of what is
happening; resolving them would need the early history to 6 relative digits, against a seed-to-seed variation of 2 %.
At 10 % of the budget the requirement is 3 digits. The kernel spectrum at t0 gives lambda_k(t0) directly, but not its
future growth, which is set by the init-specific propagation term (Theorem 4) and differs between seeds by 18-45 %
in the rank-100 gain from the first percent of training (section 3, ii).

PROPOSITION 9c (conditioning on the growth exponents). In the fixed-basis model with lambda_k(t) = lambda_k(t0)(t/t0)^{beta_k},
   d log|E_k(T)| / d beta_k = -2 lambda_k(t0) t0 [ (T/t0)^{beta_k+1} ((beta_k+1) log(T/t0) - 1) + 1 ] / (beta_k+1)^2
                            ~ -2 Lambda_k(T) log(T/t0)   for (T/t0)^{beta_k+1} >> 1.
For the modes that carry the late error Lambda_k(T) ~ 1 (the residual sits where lambda_eff t ~ 0.6-1.4, section 2),
so an absolute error d beta in the exponent gives a relative error ~ 2 log(T/t0) d beta in E(T).  (Direct.)
MEASURED: 2 Lambda_k(T) log(T/t0) median 3.8 (t0 = 1 % T) and 1.9 (10 % T). With the exponent estimated from the
interval [t0/3, t0] (noise delta -> d beta ~ delta / log 3), the end-to-end amplification is only 2-4x: the dominant
obstruction is 9b (the modes are invisible early), not 9c (the extrapolation is only mildly unstable once seen).

PROPOSITION 9d (rotation; what the fixed-basis model misses). Measured same-band energy fraction of the current
eigenvectors in the init basis (1 = no rotation), bands 1-10 / 10-50 / 50-200 / 200-600:
   early (t ~ 10): depth 1: .94 .91 .84 .94;  depth 3: .81 .77 .81 .94;  depth 8: .72 .57 .64 .90   (monotone in depth);
   mid-training (t/T ~ 0.4-0.7): every depth 2-8: .6-.8 / .46-.51 / .51-.55 / .83-.86.
Depth sets how fast the eigenvectors rotate (the spikes compose faster), not how far: by mid-training about half the
energy of the ranks 10-200, where the late residual lives, has left the init basis at every depth. The rotation is
low-rank and coherent (the spike directions, section 2), which is why the ORDERING of designs survives it while the
absolute late error does not.

Closure (what is proved vs measured). Proved: 9a, 9b(i)-(ii), 9c in the fixed-basis model. Measured: the sizes of the
two factors (1e-6 / 1e-3 visibility; 2-4x exponent amplification), the rotation (50 % in the relevant band), and the
end-to-end consequence: seed pairs that agree to 2 % in E(t) for t <= 0.1 T end 15 % apart, and every early predictor
tried orders designs within 1.1x at chance. What is NOT proved: a lower bound on the conditioning for actual networks
(it would need a bound on the propagation term's contribution to the tail growth in terms of init quantities), i.e.
the statement that NO early functional, not just the ones tried, resolves the late error below the seed spread.

## 6. Non-identifiability from the initialization kernel (added 2026-10-11)

THEOREM 10. Fix the data and a tanh network with a hidden-hidden interface l (weights W_l: H_{l-1} -> H_l). Let
A = [a^{(l-1)}(x_i)] in R^{H_{l-1} x n} be the incoming activations and Delta = [delta^{(l)}_c(x_i)] in R^{H_l x n ny} the
outgoing backprop signals at initialization. For every N = U V^T with U^T Delta = 0 and V^T A = 0, the initialization
theta_0' = theta_0 + N (added to W_l) has, on the data: the same preactivations z and backward signals m in every
layer, hence the same output, residual E_0, loss, gradient, kernel K_0, and the same first-order (coherent) weight
velocity (Theorem 3); it differs only in the propagation terms of Theorem 4, z' - z = N a_dot^{(l-1)}, m' - m = N^T delta_dot^{(l)},
which are nonzero in general. Such N exist whenever H_{l-1} > rank A and H_l > rank Delta (e.g. H > n ny); the
family has dimension (H_{l-1} - rank A)(H_l - rank Delta) and contains perturbations of any norm.
Consequently E(T) is not a function of (K_0, E_0, target): a predictor of the late error that uses only the
initialization kernel and residual is ill-defined, with an ambiguity equal to the spread of E(T) over the family.
Matching also the first k time derivatives of (K, E) at t = 0 (hence every init quantity used in sections 12-13 of
WRITEUP, including H2) requires in addition U^T [Delta_dot .. Delta^{(k)}] = 0 and V^T [A_dot .. A^{(k)}] = 0, which is
possible when the widths exceed (k+1) n ny up to rank; the two trajectories then agree to order t^{k} and differ at
order t^{k+1}, in the tail first (Proposition 9b), where the lift later amplifies the difference.
Proof. Forward: z^{(l)} = W_l a^{(l-1)} + b, and N a^{(l-1)}(x_i) = U (V^T A)_i = 0, so all forward states on the data are
unchanged, hence all activations and all delta's. Backward: m^{(l-1)} = W_l^T delta^{(l)}, and N^T delta = V (U^T Delta) = 0.
K_0 depends only on the unit states (Theorem 2), the gradient only on states and E_0 (Theorem 3). The velocities of
the states add the terms N a_dot and N^T delta_dot (Theorem 4); a_dot(x_i) and delta_dot(x_i) are not in general in the
column spans of A and Delta, so these are nonzero.  []
VERIFIED numerically (code in the session log; H = 512, depth 2, n = 50, interface l = 1): dim ker A^T = 462,
dim ker Delta^T = 362; with ||N||_F = ||W_1||_F: max |z' - z| = 5e-15, max |delta' - delta| = 4e-16, ||K_0' - K_0||/||K_0|| = 2e-16,
||E_0' - E_0|| = 7e-15, ||grad' - grad||/||grad|| = 9e-16; after one GD step (eta lam1 = 1e-3) the kernel increments differ by
1 % of their size: the two networks are indistinguishable at init and begin to differ through the propagation term.
What remains to make the ambiguity quantitative: train pairs (theta_0, theta_0 + N) to T and measure the spread of
E(T) over N (not yet run; a few GPU-minutes per pair at this size).
Remark. Theorem 10 also explains Proposition 9b from the other side: the information that separates the two
networks is carried by the weights' action outside the span of the data activations, which no data-indexed quantity
at init can see, and which enters the data-indexed dynamics only as the activations move.

## 7. Spikes form whenever the network learns: the linear-response theorem (added 2026-10-11)

Setting: squared loss, gradient flow, hidden layer l whose outgoing weights are W_{l+1}; U(t) = int_0^t E ds (n x ny);
A = incoming activations on the data (n x H_{l-1}); S[i,q] = sigma'(z_q(x_i)); M[i,q] = sum_c W_{l+1}[c,q] (W_{l+2}^T ... delta)_c  — for
the LAST hidden layer M = U W_out (n x H_l, rank <= ny). By Theorem 3 the weight change is dW_l = -(1/n) (S (.) M)^T A.

THEOREM 11 (spike + bulk decomposition, first order). Write S = h0 + (S - h0) with h0 = E[sigma'(z)] over the preactivation
distribution. Then dW_l = dW_l^core + dW_l^bulk with
   dW_l^core = -(h0/n) M^T A    (rank <= rank M <= ny; for the last hidden layer its left singular space is spanned by
                                 the rows of W_out, its right singular space by A^T U: the spike),
   dW_l^bulk = -(1/n) ((S - h0) (.) M)^T A.
If the preactivation weights of layer l are independent of W_{l+1} (true at initialization), then
   E_init ||dW_l^bulk||_F^2 / E_init ||dW_l^core||_F^2 = Var(sigma'(z)) / E[sigma'(z)]^2 =: rho(sigma_z),
evaluated at the preactivation scale sigma_z. For tanh: rho = 0.05 (sigma_z 0.5), 0.16 (0.8), 0.27 (1.0), 0.44 (1.3), 0.92 (2.0).
Proof. Expand sigma' = h0 + (sigma' - h0); the core term is h0 M, of rank <= rank M. For the bulk, E||(S-h0)(.)M||_F^2 =
sum_{i,q} E[(S_iq - h0)^2] E[M_iq^2] = Var(sigma') ||M||_F^2 by independence of S_iq (a function of layer-l input weights)
and M_iq (a function of W_{l+1} and the residual) for each (i,q); the same factorisation holds after multiplying by A
(a function of the layers below). The core has E||h0 M^T A||^2 = h0^2 E||M^T A||^2.  []
"Provided the network is learning": the spike is nonzero iff U != 0, and in a fixed basis |U_k| lambda_k = |E_k(0)|(1 - e^{-lambda_k T})
>= sqrt(2 Delta L_k) tanh(lambda_k T / 2)^{1/2}, so every mode in which a loss decrease Delta L_k has been achieved
contributes at least that much drive; learning forces the spike.
VERIFIED at initialization (first-order dW_l of the last hidden layer, exact split; 15 networks: H 96-1024, depth 2-3,
Delta 0.5/1.0, seeds 0-2): predicted rho 0.028-0.085 vs measured bulk/core energy 0.026-0.096 (within 10 % in every
case); the first-order dW has 99.8-100 % of its energy in 3 singular values; rank(dW^core) = 3 exactly.
Along training (snapshots, last hidden layer, top-3 energy of the cumulative dW): 0.98 at t = 0.3, 0.91 at t = 4,
0.78-0.80 at t = 420 (Delta 0.5), 0.65 (Delta 1.0): the bulk grows as the preactivation distribution broadens and as
the shifts leave the linear-response regime; rougher targets (larger drive) leave it sooner.
Composition (corollary, same proof layer by layer): for layer l < L-1, M_l = (W_{l+1}^T (S_{l+1} (.) M_{l+1}))-type, so the core of
layer l is the image of the core of layer l+1 through W_{l+1}^T, aligned with what the layer above reads (the cross-layer
coherence), and the bulk ratio compounds: 1 + rho_l(1 + rho_{l+1}(...)) - 1 across the stack.
Scope and limits. (i) First order in the shift and independence at init: along the trajectory the coherent updates
correlate layers, and rho must be evaluated at the current preactivation scale; the measured decay of the core fraction
is exactly this. (ii) rank(core) <= ny: the spike is low-rank because the OUTPUT dimension is small relative to width;
with many outputs the core's rank is rank(U) = rank of the residual's data-output cross-structure, a property of the
task, not of the network. (iii) Kernel-space spikes follow from weight-space spikes to first order in the feature change.
(iv) The propagation term (Theorem 4) is not part of dW_l and does not affect this decomposition; it enters the states.

## 8. Expansion at an arbitrary time and chaining (added 2026-10-11)

Everything in sections 1-7 was stated at t = 0. Nothing in the identities refers to the initialization: Theorems 1-4
hold at every t, and Theorem 11 holds at every t with S, M, A, U evaluated at t (the independence used for the bulk
RATIO is the only init-specific ingredient; the decomposition itself is an identity). The stagewise formulation:

STAGE IDENTITY (exact). Over any stage [t, t + alpha], with Psi = J(t) and dPsi = J(t + alpha) - J(t),
   K(t + alpha) - K(t) = (Psi dPsi^T + dPsi Psi^T)/n  +  dPsi dPsi^T / n  =:  Cross + Q,
and for any direction v the quadratic form changes by  [ 2 (Psi^T v).(dPsi^T v) + ||dPsi^T v||^2 ] / n.
Q is positive semidefinite. Cross is indefinite in general; it is bounded by |v^T Cross v| <= 2 sqrt(lambda_v) ||dPsi^T v|| / sqrt(n),
so the form increases on every direction where ||dPsi^T v|| > 2 ||Psi^T v|| = 2 sqrt(n lambda_v): the stage's feature
change exceeds twice the existing feature norm in that direction. In the tail (lambda_v small) this holds as soon as the
coherent feature change has any appreciable component along v: the tail is lifted by Q at every stage, monotonically.

TAYLOR RADIUS. dPsi = alpha Psi_dot(t) + O(alpha^2), with ||Psi(t+alpha) - Psi(t) - alpha Psi_dot(t)|| <= (alpha^2/2) sup ||Psi_ddot||;
Psi_ddot involves the second derivative of the map along the path (the H2 quantity of WRITEUP section 12 is its norm
along the descent direction). The stage length within which the first-order structure (Theorem 11 at time t, the
targeted lift Q_l ~ sigma_w^2 Sigma'_l(t) (.) v_U v_U^T with v_U = A_{l-1}(t) int_t^{t+alpha} E / n) is accurate is therefore
alpha(t) ~ ||Psi_dot|| / ||Psi_ddot||, computable from current quantities; the staged exact model (section 1, Theorem 4)
is the constructive realisation of the chaining and converges to 3 digits across a 10x range of stage budgets.

CHAINING (what composes). (i) Q-lifts are PSD stage by stage, so by Weyl the tail eigenvalues are nondecreasing along
the trajectory to the extent Cross is nonnegative there (measured: it is, see below). (ii) The lifted subspace at T is
contained in the span of the stage targets {Sigma'(t_s) (.) v_U(t_s)}, the smoothed residual directions along the path;
rotation of the kernel eigenvectors is the entry of new stage targets as the residual front moves (Prop. 9d), and the
effective rank of the cumulative dW is the number of effectively distinct stage cores (rank <= ny each), which grows with
how far the front travels through the spectrum, i.e. with target roughness (measured erank 5 -> 9 from Delta 0.5 to 1.0).
(iii) Coherence at time t: the core of layer l's stage change is aligned with the CURRENT W_{l+1}, which itself contains
the accumulated core aligned with the layer below through the same U; this makes the coherent part of Cross a positive
form (accumulated core x stage core), and it vanishes at t = 0 where no change has accumulated.

MEASURED (reference H96 d2 Delta 0.5, GD; per-mode increments in the eigenbasis of K(t), normalised by lambda_k):
   stage 8->10:    Cross/lambda: +0.11 (k=1) +0.17 (k=100) +0.23 (k=200) +0.20 (k=400);  Q/lambda: 0.006, 0.07, 0.59, 2.8;  99.7 % of modes increase
   stage 30->38:   Cross: +0.12, +0.23, +0.31, +0.60;  Q: 0.015, 0.23, 9.8, 871;  99.8 % increase
   stage 100->130: Cross: +0.04, +0.13, +0.31, +0.23;  Q: 0.004, 0.22, 1.5, 46;   100 % increase
   stage 300->400: Cross: -0.02 (k=1), +0.12, +0.13, +0.21;  Q: 0.003, 0.13, 0.67, 3.7;  99.7 % increase (lambda_1 drifts down once its drive is spent)
   Whenever the inequality ||dPsi^T v|| > 2||Psi^T v|| held, the mode increased (100 % of such modes, all stages).
   At initialization the Cross term is NEGATIVE on 96-97 % of modes (stage 0 -> 0.05, two seeds), crosses zero at t ~ 0.3-1,
   and is positive on 80-95 % of modes from t ~ 1 on: the positivity of Cross is created by the accumulated learned
   weights, as (iii) predicts; the first stages' lift is pure Q, tail first.
Status. The stage identity, the inequality and the Taylor-radius bound are exact. The structure of dPsi at time t
(Theorem 11 evaluated at t) is exact as a decomposition; its bulk ratio needs the current preactivation statistics.
The sign of the coherent Cross term via the accumulated core is derived under the first-order/mean-field step and
confirmed by the sign change at t ~ 0.3-1; the negative Cross at t = 0 is measured, not yet derived.

## 9. The general setting: what the identities need (added 2026-10-11)

Model class. f(x; theta) is any composition of PARAMETER-LINEAR BLOCKS (z_l = W_l a_l, where a_l is whatever input the
block receives: dense, convolution (data index = (sample, position)), attention projections (index = (sample, token)),
normalisation gains) and PARAMETER-FREE MAPS between them (pointwise activations, softmax, normalisation), differentiable
almost everywhere in theta; ReLU qualifies (its derivative exists off a measure-zero set, and gradient flow spends
measure-zero time there for generic data). Training: any update of the form dW_l = -(1/n) sum_i delta_l(x_i) g(x_i) a_l(x_i)^T
with g the per-sample loss gradient (gradient flow, GD, SGD on a subset, momentum via the integral); NOT Adam.
Loss: any; the drive is g = grad_f l (= E for squared loss). Width, depth, initialisation, time: arbitrary.

What holds verbatim (exact): Theorem 1 (f' = -K g), Theorem 2 (K = sum over blocks of incoming Gram (.) outgoing
backprop Gram, where the backprop signal delta_l = (d f / d z_l) is whatever the chain rule gives through the parameter-
free maps), Theorem 3 (data-indexed weight change; row space in the span of incoming activations on the data), Theorem 4
(z_l' = W_l' a_l + W_l a_l', m' = W_{l+1}'^T delta + W_{l+1}^T delta': coherent + propagation), the stage identity and
inequality of section 8, and the Taylor-radius bound where J is differentiable (for ReLU the Jacobian is piecewise
constant in theta and the stage error comes from activation-pattern changes instead of a second derivative).

What generalises by substitution. In Theorem 11 "sigma'(z_q(x_i))" becomes the local Jacobian of the parameter-free
map following block l, applied to the backprop signal: for pointwise activations it is the diagonal S[i,q] = sigma'(z_q(x_i)),
for softmax/normalisation a dense per-sample Jacobian. The split S = h0 + (S - h0) with h0 the PER-UNIT mean over the
data (no Gaussian reference, no Hermite expansion needed) gives core = diag(h0) M (rank <= rank M <= ny) and bulk, exactly;
the bulk/core energy ratio is sum_q Var_i(S_iq) <M^2>_q / sum_q h0_q^2 <M^2>_q, a dispersion of the local derivative,
computable from the network's own state. Hermite expansions are one way to evaluate it for near-Gaussian preactivations;
they are not part of the statement.

Where the activation enters, measured at init (first-order dW of the last hidden layer, standard init, 3 outputs):
   activation      mean sigma'   bulk/core (pred / meas, coefficients)   bulk/core in dW   energy in top-3 sv of dW
   tanh (gain 1)      0.81-0.86        0.02-0.05 / 0.02-0.05               0.01-0.02           0.999-1.000
   tanh (gain 2)      0.50             0.32 / 0.33                         0.09                0.996
   softplus           0.50-0.52        0.01-0.03 / 0.01-0.03               0.00-0.01           1.000
   gelu               0.50-0.51        0.16-0.49 / 0.14-0.50               0.27-0.29           0.987-0.992
   relu               0.50-0.54        0.33-0.50 / 0.33-0.54               0.16-0.27           0.982-0.994
The prediction (independence of the derivative pattern and the drive across data, within a unit) matches the measured
coefficient ratio for every activation. For ReLU the local derivative is a 0/1 pattern with mean ~1/2, so the bulk carries
30-50 % of the coefficient energy (vs 2-5 % for tanh at gain 1); after multiplication by the incoming activations the bulk
is still subdominant (16-27 % of dW energy) and 98-99 % of the first-order weight change lies in 3 singular values.
So the spike core forms for every activation; the activation sets the size of the bulk around it, through the dispersion
of its derivative, and nothing else in the theory depends on the activation.

What is genuinely restricted. (i) rank(core) <= rank of the drive's data-by-output structure (<= ny): "low rank" is a
small-output or low-rank-drive property. (ii) Adam-type preconditioning breaks the outer-product form of dW (measured in
WRITEUP section 9). (iii) Attention and normalisation keep Theorems 1-4 and the stage identity but replace the per-unit
diagonal by a dense per-sample Jacobian, so the core/bulk split must be redone with that Jacobian; not done.

## 10. Training through time: the chained expansion and what it proves at time T (written 2026-10-11)

Setting: section 9 (any parameter-linear/parameter-free composition, a.e. differentiable; any loss with per-sample
gradient g; width, depth, init, time arbitrary). Notation: N = n ny data-output entries; Psi(t) = J(theta(t)) in R^{N x P};
GD/flow kernel K = Psi Psi^T / n; stages 0 = t_0 < t_1 < ... < t_S = T; dPsi_s = Psi(t_s) - Psi(t_{s-1}); for a unit vector
v in R^N write lam_v(t) = v^T K(t) v and l_v = ||Psi_0^T v|| = sqrt(n lam_v(0)).

### 10.1 Exact statements at T

THEOREM 12 (feature-sum identity).  Psi_T = Psi_0 + sum_s dPsi_s, hence for every v:
      v^T K_T v = (1/n) || Psi_0^T v + D_v ||^2,     D_v := sum_s dPsi_s^T v   (the total feature change along v).
Proof: telescoping sum.  []
Remark. D_v is the history: by Theorem 4, dPsi_s = int_{t_{s-1}}^{t_s} Psi_dot, and Psi_dot is linear in the unit velocities,
which split exactly into coherent and propagation parts; so D_v = D_v^coh + D_v^prop exactly, and by Theorem 11 (at each
stage, with the stage's own S, M, A) D_v^coh = D_v^core + D_v^bulk exactly. D_v^core is explicit in the residual history:
each stage contributes the first-order image of the rank-<= ny core -(h0(t_s)/n) M_s^T A_s on Psi^T v.

THEOREM 13 (two-sided bracket on the kernel at T).  For every unit v,
      (1/n) ( ||D_v|| - l_v )_+^2   <=   v^T K_T v   <=   (1/n) ( ||D_v|| + l_v )^2 ,
and, splitting D_v,
      (1/n) ( ||D_v^core|| - ||D_v^bulk|| - ||D_v^prop|| - l_v )_+^2   <=   v^T K_T v .
Proof: triangle inequality in Theorem 12.  []
Corollary (eigenvalues). For any k-dimensional subspace V_k, lam_k(K_T) >= min_{v in V_k} (1/n)(||D_v|| - l_v)_+^2
(Courant-Fischer). Taking V_k = span of the top-k eigenvectors of K_0 gives a lower bound on the k-th eigenvalue at T from
the accumulated feature change on the initial modes; taking V_k inside the span of the stage targets gives the lift of
the visited tail: on directions the residual has driven, l_v is negligible and lam_v(T) >= (||D_v^core|| - remainders)^2 / n.
This is the precise form of "the kernel at T is lifted where the residual has been, by the square of the accumulated
coherent drive".

THEOREM 14 (upper bound from the drive).  Where J is differentiable along the path, with Psi' the derivative of J in
theta and theta_dot = -(1/n) Psi^T g (GD; for Adam see 10.4),
      ||K_dot(t)||_2 <= (2/n^2) ||Psi(t)||_2^2 ||Psi'(t)||_2 ||g(t)|| =: C(t) ||g(t)||,
hence by Weyl  |lam_k(K_T) - lam_k(K_0)| <= int_0^T C(s) ||g(s)|| ds  for every k, and v^T K_T v - v^T K_0 v <= the same.
For ReLU-type networks Psi' = 0 inside activation-pattern cells and the integral is replaced by the sum of the jumps of K
at pattern changes, each bounded by the corresponding feature jump. C(t) is finite along any finite trajectory.
Proof: K_dot = (Psi_dot Psi^T + Psi Psi_dot^T)/n, Psi_dot = Psi'[theta_dot], submultiplicativity, Weyl's inequality.  []
Together, Theorems 13-14 bracket every lifted eigenvalue at T between two functionals of the residual history
{g(s)}: a lower bound through the accumulated coherent feature change, an upper bound through the integrated drive.

### 10.2 The stage identity and the sign lemma (what upgrades the tail bracket to global monotonicity)

STAGE IDENTITY (exact, section 8).  K(t_s) - K(t_{s-1}) = Cross_s + Q_s,  Q_s = dPsi_s dPsi_s^T / n >= 0,
Cross_s = (Psi(t_{s-1}) dPsi_s^T + dPsi_s Psi(t_{s-1})^T)/n,  |v^T Cross_s v| <= 2 l_v(t_{s-1}) ||dPsi_s^T v|| / n.

LEMMA 15 (sign of the cross term; STATUS: derived under the first-order structure, not proved in general).
Write Psi(t_{s-1})^T v = Psi_0^T v + sum_{r<s} dPsi_r^T v. Then
      v^T Cross_s v = (2/n) [ (Psi_0^T v).(dPsi_s^T v)  +  sum_{r<s} (dPsi_r^T v).(dPsi_s^T v) ].
(a) The second term is the inner product of the stage's feature change with the accumulated change. Its core-core part
    is sum_{r<s} <core_r, core_s>: by Theorem 11 each core is the image of the same drive structure through the layer's
    backprop directions, so consecutive cores driven by residuals of persistent direction have positive inner product;
    in the first-order (zeroth-Hermite) form the stage lift is Sigma'(t) (.) (v_U v_U^T), and the accumulated-core x stage-core
    term is Sigma'-weighted (v_U^{acc} . v_U^{s})^2-type, >= 0 whenever the integrated residual keeps its direction.
(b) The first term (initial features against the stage change) has no sign in general. MEASURED: negative on 96-97 % of
    modes at t = 0 (two seeds), crossing zero at t ~ 0.3-1, and the full Cross_s positive on 80-100 % of modes from t ~ 1
    through t = 400 on the reference run; the exceptions are modes whose drive has been exhausted (lambda_1 late).
Consequence if Lemma 15 holds from a stage s_0 on (in the form v^T Cross_s v >= -eps_s for all v):
      K(t_s) >= K(t_{s-1}) - eps_s I  for s > s_0,  hence  lam_k(K_T) >= lam_k(K(t_{s_0})) - sum_s eps_s  for every k (Weyl),
i.e. every eigenvalue is nondecreasing after the transient up to the accumulated eps, and the lower bound of Theorem 13
holds for all of K_T, not only on the visited tail. This is the single lemma that separates "tail theorems" from
"global monotonicity"; its proof requires controlling the init-feature term (b), which is where the propagation term
and the random initialisation enter.

### 10.3 The chain itself (how the history is produced, and the error it carries)

THEOREM 16 (stage error / Taylor radius).  Where J is C^2 along the path, dPsi_s = alpha_s Psi_dot(t_{s-1}) + R_s with
||R_s|| <= (alpha_s^2 / 2) sup_{[t_{s-1}, t_s]} ||Psi_ddot||, and Psi_ddot = Psi''[theta_dot, theta_dot] + Psi'[theta_ddot] with
theta_ddot = -(1/n)(Psi_dot^T g + Psi^T g_dot), g_dot = (grad^2 l) f_dot = -(grad^2 l) K g. Choosing alpha_s so that
alpha_s ||Psi_ddot|| <= eps ||Psi_dot|| makes each stage's first-order increment accurate to relative eps and the chain's
total error <= eps sum_s ||dPsi_s|| (first order) — the exact staged model of section 1 is this construction with the
stage criterion EPS, converged to 3 digits across a 10x range. Within a stage, the first-order increment has the structure
of Theorem 11 evaluated at t_{s-1}; the induction carries three invariants: the current preactivation statistics (hence
h0, the bulk ratio), the accumulated core/bulk/prop decomposition of the weights (inherited by summation, no hypothesis),
and the residual. The invariants are bookkeeping; the only hypotheses are those of section 9.  []

### 10.4 Adam

THEOREM 17 (preconditioned flow).  Let theta_dot = -P(t) grad L with P(t) diagonal, positive (Adam: P = (sqrt(v) + eps)^{-1}
entrywise, with v the second-moment estimate; momentum replaces g by its exponential average, a memory term in the drive).
Then (i) f_dot = -K_P g with K_P = Psi P Psi^T / n >= 0 (this is why the plain-kernel regression fails on the Adam runs while
the structure survives); (ii) K_P = sum_blocks sum_{q,p} P_l[q,p] (delta_q delta_q^T) (.) (a_p a_p^T) / n, exact, no Gram
factorisation unless P_l is rank one; (iii) dW_l = -(1/n) int P_l (.) (Delta_l g A_l^T): the data-indexed outer-product form,
multiplied ENTRYWISE by P_l; (iv) the stage identity, Theorem 12 and Theorem 13 hold verbatim for K_P with Psi replaced by
P_T^{1/2}-weighted features: v^T K_{P,T} v = (1/n)||P_T^{1/2}(Psi_0^T v + D_v)||^2, and Theorem 14 with the extra factor ||P||.
Proof: chain rule with the preconditioner; (iv) is algebra.  []
LEMMA 18 (sign of low rank; STATUS: needed for the Adam spike statement, measured, not proved).  For the core, the gradient
block G = Delta g A^T / n has rank <= ny, and Adam's P is built from G itself (v ~ time-average of G^2 entrywise), so
P (.) G is a normalised-sign pattern of a low-rank matrix. For rank one, sign(u w^T) = sign(u) sign(w)^T exactly (rank one);
for rank r > 1 the sign pattern has a dominant rank-r component and a tail. MEASURED (WRITEUP section 9): under Adam the
effective rank of the hidden-layer change is 6-13 vs 5-9 for GD at the same designs, with the same gain law.
With Lemma 18 the whole of sections 7-8 and 10.1-10.2 transfers to Adam with K_P.

### 10.5 The residual at T

THEOREM 19 (exact representation and what the bracket gives).  f_T - f_0 = -int_0^T K(s) g(s) ds exactly, and for the
squared loss E(T) = Texp(-int_0^T K) E(0). Energy: ||E(T)||^2 = ||E(0)||^2 - 2 int_0^T E^T K E ds. If Lemma 15 holds from s_0 on,
K(s) >= K(t_{s_0}) - sum eps for s >= t_{s_0}, so the energy decreases at least as fast as under the frozen kernel K(t_{s_0})
in the sense of the integrand: E^T K(s) E >= E^T K(t_{s_0}) E - (sum eps)||E||^2 pointwise in s. This is a statement about the
rate along the actual residual; converting it into a closed-form bound on ||E(T)|| needs the fixed-basis assumption
(Prop. 6), which is violated by ~50 % in the relevant band (Prop. 9d). The honest chain of implications is therefore:
history of g  ->  (Thms 12-14) bracket on K_T and on K(s) along the path  ->  (Lemma 15) monotone K  ->  (Thm 19) a lower
bound on the instantaneous decay rate at every time, hence on the loss decrease, with the fixed-direction statement
(how much of E(T) survives in a given mode) left open to the extent the eigenvectors rotate.

### 10.6 Status table
   Thm 12 feature sum               exact, any model in section 9, GD and Adam
   Thm 13 bracket at T              exact; remainders D^bulk, D^prop explicit, computable along the chain
   Thm 14 drive upper bound         exact where J is C^1; ReLU via jumps
   Lemma 15 cross-term sign         derived for the core-core part; init-feature part measured (negative at t=0, positive after); NOT proved
   Thm 16 stage error               exact (C^2 along the path); the chain's convergence verified numerically
   Thm 17 Adam identities           exact
   Lemma 18 sign of low rank        rank one exact; general measured (erank 6-13); NOT proved
   Thm 19 residual at T             exact representation; rate bound conditional on Lemma 15; mode-wise bound needs fixed basis
Verification still to run (login node, existing code): the bracket of Thm 13 against K(T) on the reference run (needs
Psi_T: one GD run to T with the Jacobian saved); the sign of Cross_s per stage under Adam (needs the Adam harness to log P).
VERIFIED (Thm 13 bracket, reference H96 d2 Delta 0.5, Psi_0 and Psi_T from the saved endpoint weights, v = eigenvectors of K_0):
   rank:                 1       10      30      60      100     200     300     500
   ||D_v|| / l_v:        1.5     3.4     4.3     19      28      157     1100    31000
   lower bound / actual: 0.07    0.46    0.60    0.90    0.93    0.99    0.998   1.000
The total feature change exceeds the initial features in all 600 directions (even rank 1, by 1.5x). The lower bound is
essentially exact on the visited tail (ranks >= 60: 90-100 % of v^T K_T v; median 0.998 over all directions) and loose
only on the top ~30 modes, where initial features and change are comparable and aligned (the cross term carries the
rest). So "the kernel at T on the residual's history equals the squared accumulated feature change there" is not only a
bound but the value, and the content moves entirely to the decomposition of D_v into core, bulk and propagation along
the chain, which is the next computation (needs the accumulators in the exact staged model; not yet run).

## 11. Euler versus direct, in the scaled normalisation (written 2026-10-11)

Setting. Flow Phi_Delta on data x_i (normalised). Direct target y_D = Phi_Delta(x); Euler target y_E = (Phi_Delta(x) - x)/Delta.
Same architecture, same initialisation theta_0, outputs centred at init (or small). Direct predictor f_D(x); Euler predictor
x + Delta f_E(x). Physical one-step errors: e_D = f_D - y_D,  e_E = Delta (f_E - y_E) = Delta E_E. Gradient flow on the squared loss
of each network's OWN target, same learning rate. Decompose y_D = x + Delta y_E, so E_D = (f_D - x) - Delta y_E.

THEOREM 20 (direct = Euler with the drive scaled by Delta).  Write the direct residual on the increment as E_D = Delta E~.
Then E~' = -K_D E~ and E_E' = -K_E E_E with K_D(0) = K_E(0), and the weight velocities are
      theta_dot_D = -(Delta/n) J^T E~        versus        theta_dot_E = -(1/n) J^T E_E .
At equal normalised residual the direct network moves its weights Delta times slower. By Theorem 11 (at every stage) the
direct network's stage core on the increment is Delta times, and its stage lift Q is Delta^2 times, the Euler network's.
Proof. Chain rule: grad of (1/2n)||E_D||^2 = (1/n) J^T E_D = (Delta/n) J^T E~; the residual equation follows from Theorem 1
with K the kernel of the same map; the stage statements are Theorem 11 with U_D = Delta U~.  []
This is exact and holds for any model in the class of section 9, for GD/SGD/momentum (for Adam the preconditioner
partly undoes the scaling; see 10.4 — the comparison must be redone with K_P).

THEOREM 21 (identity phase of the direct network).  The direct residual also contains -(x - (f_D - x)|_{t=0}), the identity,
Delta-independent. If the coordinate functions lie in the leading eigenspace of K_0 (measured for the kernels here: the
identity is fit on the time scale 1/lambda_top << T), this component is removed early, with a Delta-INDEPENDENT drive
U_id = O(||x||/lambda_top) and a positive semidefinite lift Q_id targeted at the smoothed identity direction (section 8),
plus a Delta-independent shift of the units (propagation). After this phase the direct network solves the increment problem
from the state (K_0 + Q_id + cross, theta_0 + d theta_id) with the scaled drive of Theorem 20.  (Exact given the eigenspace
statement, which is a property of K_0 and the data.)

THEOREM 22 (the limit Delta -> 0 is the lazy limit for direct, and only for direct).  Fix T. As Delta -> 0, the direct
network's increment-phase weight motion is O(Delta) (Theorem 20), so by the lazy-training theorem (gradient flow stays
in the linearised regime over a fixed horizon when the parameter motion is small relative to the curvature scale of the
map) its normalised residual E~ converges to kernel regression with the FROZEN kernel K_D(t_1) = K_0 + (identity-phase
change). Hence the normalised direct error converges to the frozen-kernel error of the increment target, which is the
quantity the static roughness bound controls: the held-rough bound becomes TIGHT for direct as Delta -> 0. The Euler
network's drive is Delta-independent; its error is the full feature-learning error, below the frozen-kernel value by the
feature-learning gain on the increment target (Theorem 19's rate comparison, conditional on Lemma 15), so the same
bound is loose for Euler by exactly that gain, which grows with T and with the increment's roughness (gain law).
Proof. Theorem 20 for the scaling; lazy-training conditions (Lipschitz Jacobian along the trajectory, finite T) for the
convergence to the linearised flow; Theorems 12-13 for the lift; the rate comparison for the inequality.  []

COROLLARY 23 (physical error).  e_E = Delta E_E and e_D = Delta E~ (increment part) + identity remainder. Under the rate
comparison (Lemma 15 + comparison across the two trajectories, which needs persistence or a fixed basis — the one
remaining hypothesis), ||E_E(t)|| <= ||E~(t)|| for all t, hence ||e_E(t)|| <= ||e_D(t)|| at every time, with the gap equal to the
feature-learning gain on the increment and additionally the identity remainder early. Equality is approached only when
Delta -> 1 (same drive) — and there the two targets have comparable roughness and the static theory applies.

What this changes in the roughness paper's argument. The static argument ("Euler's target is smoother, so a kernel fits
it better") is the frozen-kernel shadow of the real mechanism: dividing by Delta multiplies the drive by 1/Delta and
the stage lifts by 1/Delta^2, so Euler trains in the rich regime on the increment while direct trains lazily on it. The
roughness bound is exact for direct in the limit and the Euler advantage is the feature-learning gain the bound cannot
see. Both halves are stated with the same objects (kernel, residual, roughness in the kernel basis).

What is exact vs conditional. Thm 20: exact. Thm 21: exact given the eigenspace statement (measurable). Thm 22: exact
(lazy-training theorem) for direct's limit; the inequality for Euler is conditional on the rate comparison (Lemma 15).
Cor. 23: conditional on the cross-trajectory comparison (persistence / fixed basis). Numerically all of it is decidable
with the exact staged model (same theta_0 for both, any Delta), which gives E_D(t), E_E(t), the stage lifts and their
targets exactly; not yet run (needs the Euler target in the harness: Y_E = (Y - X)/Delta in normalised units).

## 12. Candidate applications beyond Euler/direct, spikes and EoS (literature scan 2026-10-11; status = what the framework can prove)

(4) Progressive sharpening and the initial sharpness-reduction phase.  Stated open: "why neural networks exhibit
    progressive sharpening" (Damian-Nichani-Lee 2022, arXiv 2209.15594), "underlying mechanisms remain poorly
    understood" (Yoo et al. ICML 2025), "extending to deeper nonlinear networks is open" (Wang et al. NeurIPS 2022);
    warmup papers (Kalra-Barkeshli NeurIPS 2024) observe an early sharpness-REDUCTION phase followed by sharpening
    depending on init/parameterisation, without a mechanism.  Framework: lambda_1' = v_1^T (Cross + Q) v_1 per stage;
    Q >= 0 always; Cross negative at t = 0 (measured, 96-97 % of modes) and positive once the accumulated core exists
    (Lemma 15 core part, derived) -> sharpness reduction then progressive sharpening, for any depth/activation,
    with depth entering through the composition of cores (measured 5x -> 86x). Missing: derivation of the t = 0 sign.
(5) Adam's rotation dependence.  Stated open: "what properties of a rotation benefit or hinder [adaptive methods]"
    (arXiv 2410.19964), "rigorous analysis of Adam's implicit bias ... remains open" (NeurIPS 2025).  Framework:
    Theorem 17 (K_P exact) + Lemma 18: the entrywise preconditioner acts on a rank-<= ny gradient block, so the update
    is a normalised sign pattern of a low-rank matrix whose effective rank depends on the coherence of the core's
    singular vectors with the coordinate axes — a rotation-dependent statement of exactly the requested kind.
    Missing: Lemma 18 for rank > 1 (sign-pattern rank bounds).
(6) Why hyperparameters transfer across width and depth.  Stated open by Bordelon et al. ICLR 2024: "it remains an
    open problem theoretically why hyperparameters transfer ... even when the actual predictions/losses differ".
    Framework (finite width, no limit): the memory equation splits the dynamics into a lazy term (K_0, scaling known
    through Theorem 2) and a feature-learning term (stage cores, scaling through Theorems 3-4 and 11); transfer of the
    optimal scale/learning rate <=> size-invariance of the ratio of the two terms at matched normalised time, a
    criterion computable from one Jacobian per size with the exact staged model. Missing: the computation.
(7) Grokking delay.  Kumar et al. ICLR 2024 state they "do not provide a sharp theoretical characterization of the ...
    training time to transition"; arXiv 2609.26679 (2026) says a rigorous lazy-to-rich account is missing and builds a
    residual-driven NTK-growth system — our mechanism.  Framework: output scale s divides the drive by s and the stage
    lifts by s^2 (the Delta argument of section 11 with s in place of 1/Delta), so the time for the accumulated coherent
    lift along the task direction to exceed the initial features there (Theorem 13 bracket) scales as s^2 — a concrete
    delay law. Caveat: our theory is about the training kernel; "task direction = generalising direction" is their
    modelling step, not ours.
(8) Minimum LoRA rank.  Stated open (Zeng-Lee, arXiv 2310.17513): "what is the minimum rank of the LoRA adapters
    required".  Framework: Theorem 11 + chain: rank of the gradient-flow update <= ny x (number of distinct drive
    directions), exact; sharp for small-output tasks, vacuous for vocabulary-size outputs unless the integrated drive
    is itself low rank (a task property). Honest scope: regression / few-class fine-tuning.
Not addressable here: anything about test error (validation is outside the identities), and SGD-noise effects.

### 12.1 Results on items 4, 6, 7 (2026-10-11)
(4) Initial sharpness-reduction phase: MECHANISM VERIFIED. Along the residual, v^T Cross v at t = 0 is proportional to
    -E.f''[g,g] (exact), and E.f''[g,g] > 0 in 5/5 initialisations (H 96-256, depth 2-3), so the kernel's form along the
    residual decreases at first. Mode-wise, the cross term is negative on 90-91 % of K_0 eigen-directions for the raw
    residual E = f_0 - y and on 48-55 % (no systematic sign) for the CENTRED residual E = -y, with the magnitude along E
    20-40x smaller. So the reduction phase is the network unlearning its own initial function f_0; it is absent for
    centred initialisations (f_0 subtracted), which is the init/parameterisation dependence reported in the warmup
    literature. The term-level split (f_0-part vs y-part of E.f''[g,g]) is NOT clean (both parts positive in most cases),
    so the derivation stands at the level "f_0 in the residual produces the systematic sign", established by the
    raw-vs-centred comparison, not by a closed-form sign of a single term. Together with Lemma 15 (core part): reduction
    -> crossover (t ~ 0.3-1 here) -> progressive sharpening.
(6) Adam's rotation dependence: LEMMA 18 DERIVED AND VERIFIED (synthetic). sign(G) for G = U W^T of rank R: core energy
    fraction in the top-R singular directions = 1.000 (R=1, exact), 0.81 (R=2), 0.75 (R=3), 0.72 (R=4), 0.69 (R=8), 0.68
    (R=16) for Gaussian factors, approaching the Hermite value 2/pi = 0.637 for Gaussian entries; with SPARSE
    (coordinate-aligned) factors the fraction stays 0.95-1.00; heavy-tailed factors behave like Gaussian. With the
    time-averaged second moment (Adam's v over a window of drifting gradients) the fraction drops further to 0.52-0.54.
    Statement: Adam's entrywise normalisation is a nonlinearity applied to a low-rank gradient block; its core/bulk split
    follows the same mean-plus-fluctuation argument as Theorem 11, with the core fraction set by the first Hermite
    coefficient of the normaliser under the ENTRY distribution of the gradient. Rotations that align the gradient's
    singular vectors with the coordinate axes (sparse entries) preserve the low-rank core; delocalising rotations push
    it to ~2/pi and inflate the bulk. This is the rotation-dependent property asked for in arXiv 2410.19964, and it
    predicts the measured erank increase under Adam (6-13 vs 5-9 for GD).
(7) Grokking delay vs output scale: PARTIAL. Prediction from the drive scaling (U ~ tau/s, lift ~ U^2): the lazy time
    tau* = s^2 t* at which a slow mode's gain reaches a threshold scales ~ s. Measured on H96 runs at s = 0.25/0.5/1
    (gain at rank 100 reaching 3 and 10, rank 200 reaching 10; three designs): tau* ratios 1 : 2.6-2.9 : 7.8-14, i.e.
    tau* ~ s^{1.5-2.0}, not s^1. The direction and the power-law form are right; the exponent needs the full chain
    (the one-line argument ignores how the drive's spectral position and the propagation term change with s).

## 13. Validation error: the swap of the left Jacobian stack (added 2026-10-11)

THEOREM 24 (validation dynamics, exact).  With J_val the Jacobian on validation points, the validation outputs obey
f_val' = -(1/n) J_val J_tr^T g_tr = -K_vt g_tr, K_vt = J_val J_tr^T / n (the cross kernel). For the squared loss,
E_val' = -K_vt(t) E_tr(t), so  -d/dt (1/2)||E_val||^2 = E_val^T K_vt(t) E_tr(t).  Everything in the chain applies with the
left factor replaced: K_vt(t_s) - K_vt(t_{s-1}) = (dJ_val J_tr^T + J_val dJ_tr^T + dJ_val dJ_tr^T)/n, and the coherent part of
dJ_val is the SAME weight increment (data-indexed on the training points, Theorem 3) evaluated on validation inputs:
its activation part is the cross activation Gram A_{val,tr} applied to the drive. Under the assumption that the unit
statistics on validation and training points agree (same preactivation distribution, same bulk ratio), the stage core
on validation points is the training core transported by A_{val,tr}. No other change.  Proof: chain rule.  []
Consequence (the overfitting split, exact): the validation descent rate splits as
      E_val^T K_vt(0) E_tr   (lazy transfer: the frozen init cross-kernel acting on the current training residual)
    + E_val^T [K_vt(t) - K_vt(0)] E_tr   (feature-learning transfer: the lifted cross-kernel).
MEASURED (snapshots of six GD runs, rates normalised by |E_val||E_tr|; training rate for scale):
 - Early (t <~ 5): both parts positive and the validation rate is 70-110 % of the training rate: training steps reduce
   validation error almost as fast as training error.
 - Late: the LAZY part turns NEGATIVE (H128 d6: from t = 28; H256 d8: from t = 13; H256 d4 Delta 1: from t = 16;
   H96 d3: at t = 420): once the training residual lives in sample-specific rough modes, the frozen init cross-kernel maps
   it into directions that INCREASE validation error — the static "held rough" overfitting mechanism, seen directly.
 - The LIFTED part stays POSITIVE in the same window (e.g. H128 d6 at t = 28-840: lifted +0.037..+0.0013 vs lazy
   -0.011..-0.0005; H256 d8 at t = 13-390: +0.088..+0.0063 vs -0.038..-0.0042), so the net validation rate remains
   slightly positive and validation error keeps falling slowly; where the lifted part also flips (H256 d4 Delta 1,
   t = 155-466) validation error rises (1.06 -> 1.45): overfitting proper.
 - By band of the training residual the lifted contributions have large alternating signs (e.g. -0.79 / -0.20 / +1.08),
   i.e. the lift transfers coherently through some bands and overfits through others; the sign is decided by whether
   the lifted direction is shared by validation points (smooth under A_{val,tr}) or sample-specific.
Statement this supports (to be proved under the equal-statistics assumption): feature learning generalises better than
the frozen kernel in the late phase because the lift is built from A_{val,tr}-smoothed drives (shared directions),
while the lazy tail is sample-specific. This is the dynamic counterpart of the roughness paper's static bound, and the
"cross-kernel coherence" of a lifted direction is its roughness measured between the two samples.
(7, continued) Lazy-side runs (H96 d2 Delta 0.5 at s = 2, 4, 8, 16; runs/gt_lazy) and fixed-direction analysis.
 - Drive saturation is NOT the cause of the lazy-side exponent: at s = 8, 16 the drive grows as tau^0.92-0.95 (lazy
   learning curve |E|^2 ~ tau^-0.14), so U ~ tau, yet the sorted-gain delay exponent is ~2 and the fixed-direction excess
   at matched tau scales as s^-1.5 (not s^-2) with tau-growth 0.6-0.8 (not 2).
 - The stage identity explains the whole pattern: along a fixed direction v the excess is [2 (Psi'v).(dPsi'v) + ||dPsi'v||^2]/(n lam_v).
   On the lazy side the FIRST-order cross term dominates (||dPsi'v|| << ||Psi'v||): excess ~ (U/s) x alignment, where the
   alignment of the feature change with the existing features is itself built by the accumulated core (~U/s) — hence the
   intermediate exponents (-1.5 in s, 0.6-0.8 in tau). On the rich side the second-order term and the Hermite bulk dominate:
   exponents -2.4 to -3.5 in s. For the deep tail (rank >= 200) the fixed-direction excess is NEGATIVE early at every s >= 2
   (-0.2 to -0.3 at rank 300 until tau ~ 40-130): the init-tail eigen-directions lose kernel energy before the lift arrives
   (the rotation of Prop. 9d), so they are not "task directions" in the grokking sense either.
 - Conclusion: a universal delay exponent in the output scale does not exist in this testbed under any natural
   observable; the clean law tau* ~ s holds only when the direction has substantial initial features, small change and a
   fixed alignment, which is not the grokking regime. What the theory gives rigorously is the decomposition of the
   delay into first-order (alignment-weighted), second-order and bulk contributions (Theorem 13 + section 8), each with its
   own s-dependence, and the monotone power-law-like growth of the delay with s that the literature reports.
   Strengthening beyond this = attribution, not a law: run the exact staged model with per-direction accumulators for the
   cross, Q and bulk parts along a chosen task direction to show which term carries the s-dependence in each regime.

### 12.2 Item 7 resolved by the chain: the lift as a series in 1/s (2026-10-11)

THEOREM 25 (scale expansion of the lift).  With centred outputs f_s = s (N(theta) - N(theta_0)) and lazy time tau = s^2 t,
the residual equation dE/dtau = -K_N E has no s in it and the weight velocity is dtheta/dtau = -(1/(s n)) J_N^T E.
Hence, for any fixed direction v and fixed tau, the lift is a power series in epsilon = 1/s:
      v^T K_N(tau) v / v^T K_N(0) v - 1 = a_1(tau)/s + a_2(tau)/s^2 + O(1/s^3),
      a_1 = 2 (Psi_0^T v).(dPsi^(1)^T v) / (n lam_v),       dPsi^(1)^T v = Hess_{v.f}(theta_0) [d theta^(1)],
      a_2 = ||dPsi^(1)^T v||^2 / (n lam_v)  +  2 (Psi_0^T v).(dPsi^(2)^T v) / (n lam_v)  (+ the term from the second-order weight
            change, omitted below),     d theta^(1)(tau) = -(1/n) J_0^T U_lazy(tau),   U_lazy(tau) = int_0^tau E_lazy = V diag((1-e^{-w tau})/w) V^T (-y),
      dPsi^(2)^T v = (1/2) d^2/de^2 grad(v.f)(theta_0 + e d theta^(1)).
All coefficients are functionals of the INITIAL Jacobian (first and second directional derivatives along the lazy
drive) and of the closed-form lazy residual history; no training at any scale is needed. The grokking delay for a
threshold g* is the inversion tau*(s) of this series; it is a power law only asymptotically, with exponent set by the
tau-growth of a_1 (if a_1 != 0) or of a_2, and in general it is a predicted function, not an exponent.
Proof. Expand dPsi in the stage identity in powers of the weight displacement, which is O(1/s) at fixed tau; the
residual history is s-independent at leading order; group terms.  []
VERIFIED (H96 d2 Delta 0.5, centred init, v = init eigenvectors, measured runs at s = 1, 2, 4, 8, 16; two terms, without
the second-order-weight-change part of a_2):
   rank 100: prediction/measurement 0.88-0.95 while the excess is < 0.06; 0.70-0.85 for excess 0.06-0.5; the ratio falls
   below 0.5 only once the measured excess exceeds ~1 (s = 1 beyond tau ~ 4, s = 2 beyond tau ~ 13, s = 4 beyond tau ~ 130).
   rank 200: the EARLY NEGATIVE excess (-0.007 .. -0.01, the initial decrease along init-tail directions) is predicted with
   ratio 0.8-1.3 at s = 2-16, including its sign change at tau ~ 4-13; the late lift (10^2-10^3, far outside the
   perturbative range) is not.
   The coefficient a_1 changes sign along tau (positive early at rank 100, negative after tau ~ 40; negative throughout
   at rank 200): the first-order cross term is what makes the early lift non-monotone and non-universal.
So the accurate claim for item 7 is: at output scale s the lift of any direction is predicted from the initialisation
to within 10-20 % for as long as the lift is below order one (the perturbative regime, whose extent in tau grows with s),
by two init-computable coefficients; the delay to a small-lift threshold is the inversion of that prediction. Beyond the
perturbative range the exact staged model (section 1) is required; there is no closed exponent, by the mechanism of
12.1 (7, continued). Extending the expansion to third order (Psi''' and the second-order weight change) is the direct way
to push the validity range; it is bookkeeping, not new theory.
Full second order (code/scale_expansion2.py: adds the second-order weight change from the moving Jacobian and the
feedback of the lifted kernel on the lazy residual, both via variation of constants in the init eigenbasis):
   rank 100, prediction/measurement with ALL O(1/s^2) terms: s = 16: 1.01, 1.03, 1.13, 0.95 (tau 2.6-43); s = 8: 1.00, 1.00, 1.05,
   1.16, 0.95 (tau 0.7-42); s = 4: 0.98, 1.00, 1.00, 1.06, 1.09, 0.84 (tau 0.2-44); s = 2: 0.92, 1.00, 1.00, 0.99, 0.98, 0.79 (tau 0.1-13).
   rank 200 (negative early excess included): 0.86-1.17 at the first two snapshots for every s; 0.6-0.9 while the lift < 1.
   The two-term truncation was 0.7-0.95 in the same range: the missing terms were the 10-20 %.
   Breakdown is sharp: once the measured lift exceeds ~1 (s = 16 beyond tau ~ 130, s = 8 beyond tau ~ 130, s = 4 beyond tau ~ 130,
   s = 2 beyond tau ~ 13) the second-order residual-feedback term grows negative and the truncated series fails
   (uncancelled third-order terms): the radius of the expansion in 1/s is where the lift along the direction reaches order one,
   exactly as Theorem 13's inequality (feature change vs existing features) says.
Status of Theorem 25: proved; verified to within 0-16 % at full second order throughout its range on five output scales;
range = lift below order one, extent in tau growing with s.

### 12.3 The richness expansion to all orders, its radius, and the chain as its resummation (written 2026-10-11)

Setting: the class of section 9 with analytic parameter-free maps (piecewise-analytic for ReLU, within an activation-
pattern cell). Let epsilon be any parameter that multiplies the weight velocity at fixed residual dynamics in a suitable
time: the inverse output scale 1/s (lazy time tau = s^2 t), the Euler step Delta of section 11, a muP-type multiplier, or
the inverse-width factor of mean-field scaling. In that time the system is
      dE/dtau = -K_N(theta) E,        dtheta/dtau = -(epsilon/n) J_N(theta)^T E,     theta(0) = theta_0, E(0) = -y (centred).

THEOREM 26 (all-orders richness expansion).  For each fixed tau there is R(tau) > 0 such that (theta(tau), E(tau)) and
every analytic observable of them (the kernel form along any v, the loss, the per-layer shares) are given by convergent
power series in epsilon for |epsilon| < R(tau):
      v^T K_N(tau) v / v^T K_N(0) v - 1 = sum_{k>=1} a_k(tau) epsilon^k .
The coefficients are determined recursively from the initialisation and the lazy history alone: a_0 is kernel regression
with K_N(theta_0); theta^(1)(tau) = -(1/n) J_0^T U_lazy(tau); a_1 = 2 (Psi_0^T v).(J'[theta^(1)]^T v)/(n lam_v); and a_k involves
directional derivatives of J at theta_0 up to order k along the lower-order displacements theta^(1..k-1) together with the
lower-order residual corrections E^(1..k-1), which obey linear inhomogeneous equations driven by the lower-order kernel
corrections (variation of constants in the init eigenbasis). No quantity along a trained path enters at any order.
Proof. The right-hand side is analytic in (theta, E, epsilon); analytic dependence of ODE solutions on parameters gives
analyticity in epsilon at fixed tau with a positive radius; the recursion is the Taylor expansion of the ODE in epsilon.  []
Remark (one theorem, several questions). With epsilon = 1/s this is the delay/onset law of item 7 (section 12.2, verified to
0-16 % at second order); with epsilon = Delta it is the Euler-vs-direct statement of section 11 (direct = Euler at
epsilon -> Delta epsilon, so direct's lift is the SAME series evaluated at a smaller argument); with epsilon the inverse width
factor it is the finite-width correction to mean-field training; and the feature-learning/lazy ratio of item 5 is
epsilon a_1(tau; H) + epsilon^2 a_2(tau; H) + ..., so hyperparameter transfer across width (or depth) is the statement that
epsilon(H) a_k(tau; H) is H-independent at the orders that matter — a finite-width criterion computable from one Jacobian
per size, no limit taken.

THEOREM 27 (the radius).  R(tau) is bounded below by the stage criterion of Theorem 16 applied to a single expansion
point: the series is valid while the induced feature change stays within the Taylor radius of J at theta_0,
|epsilon| ||theta^(1)(tau)|| ||Psi''|| / ||Psi'|| < 1, and along a given direction v while ||dPsi^T v|| < ||Psi_0^T v|| (Theorem 13's
inequality: the feature change smaller than the existing features). Where the second condition fails the lift along v is of
order one and the series around theta_0 cannot represent it: no finite or infinite number of terms recovers the event,
because the singularity of the map epsilon -> trajectory lies inside the evaluation point.  (Proof: Theorem 16 + 13.)
MEASURED: the truncated series fails sharply where the measured lift reaches ~1 (12.2), and the direction that is "fit" in
the grokking sense (lift >> 1) lies beyond R(tau) by construction.

THEOREM 28 (the chain is the resummation).  Re-expanding at each stage boundary t_s around theta(t_s) (section 8, Theorem 16)
gives a sequence of convergent expansions each within its own radius; their composition is the exact trajectory for
any epsilon and any tau (section 10, exact stage identity + Theorem 16 error control). Hence: the all-orders series of
Theorem 26 is the chain with a single expansion point; the chain is the series continued past R(tau) by moving the
expansion point along the path. Within R(tau) the two agree; beyond it only the chain is defined, and its coefficients
must be evaluated along the path rather than at the initialisation.  []
Consequence for the paper: state the richness expansion at all orders (Theorem 26), exhibit a_1, a_2 and their accuracy
(12.2), state the radius as a theorem (27), and hand every statement about the event itself — the plateau end, the
grokking time, the late error — to the chain (28), where it is a computation, not a formula. This is complete rather
than truncated, and it applies verbatim to Euler-vs-direct, to the output-scale and width/depth families, and to Adam
with K_P in place of K_N.

### 7.1 The spike theorem along the trajectory without independence (added 2026-10-11)

What fails after initialisation is only the bookkeeping, not the rank. MEASURED (reference, t = 0.3): the cumulative
change is still 98 % in three singular directions, but the per-unit-mean split of Theorem 11 (core = mean sigma' x drive)
gives bulk/core = 0.9 against a dispersion prediction of 0.05: the local derivative S = sigma'(Z) and the drive M have
become correlated across the data (the units have adapted to the residual), so the mean of sigma' is no longer the right
rank-ny approximant of S. Independence was a convenience valid at t = 0; the rank statement never used it.

THEOREM 11' (correlation-free rank bound, any time).  For the last hidden layer the instantaneous update is
dW = -(1/n)(S (.) M)^T A with rank M <= ny EXACTLY (M = E W_out). For any rank-r approximation S_r of S with remainder S~ = S - S_r,
      rank( (S_r (.) M)^T A ) <= ny r,          || (S~ (.) M)^T A ||_F <= ||S~||_max ||M||_F ||A||_2 / n,
so the energy of dW outside an (ny r)-dimensional subspace is bounded by the entrywise size of the spectral tail of S.
S itself is constrained without any assumption on correlations: Z = A_{l-1} W_l^T + b, so every column of Z lies in the
column space of the layer below's activations on the data, whose Gram spectrum the chain tracks (steep: ~k^-5 here), and
sigma' applied entrywise raises the numerical rank only through the Hermite coefficients of sigma' (He_k(Z) has rank at most
that of degree-k polynomials in the columns of Z; for the first hidden layer with d_in = 3 that is C(3+k, k)). Hence
rank_delta(S_l(t)) is controlled at every t by (i) the activation's Hermite decay at the current preactivation scale and
(ii) the numerical rank of the preactivation matrix Z_l(t), both chain quantities. For deeper layers M_l is the layer above's
(S (.) M) propagated through W, so the bound composes: rank(dW_l) <= ny prod_{l' >= l} rank_delta(S_{l'}).
Proof: Hadamard rank inequality rank(X (.) Y) <= rank X . rank Y; the entrywise Frobenius bound; the column-space inclusion
for Z; Hermite expansion of sigma'.  []
This replaces the independence assumption by two computable spectra and holds along the whole path, for any
activation, any width. The sharpness of the bound (how close ny r_S is to the measured effective rank) is being
measured along the reference trajectory (code/rank_trajectory.py); numbers to follow.
Remark on the testbed. L63 at n = 200 is nearly fit by T, so its late stage has small residuals; the late lifts are large
(tail gains 10^2-10^4) but carry little loss. A target strongly misaligned with K_0 (or a larger n) would exercise the
late-stage regime more; nothing in Theorems 11'/26-28 depends on the testbed.

## 14. Is training chaotic? Why late-stage prediction is hard, provably (written 2026-10-11; numbers pending the perturbation test)

PROPOSITION 29 (gradient flow is not chaotic).  For gradient flow theta' = -grad L on a C^2 loss, L is a strict Lyapunov
function (dL/dt = -||grad L||^2 < 0 off critical points), so there are no periodic orbits, no recurrent sets and no strange
attractors: every bounded trajectory converges to the critical set. Sustained exponential separation of nearby trajectories
on an attractor — the defining property of chaos — is impossible. What remains possible is FINITE-TIME sensitivity: the
linearised flow map has expansion factors exp(int_0^T -lambda_min(grad^2 L) dt) along transiently negative-curvature
(saddle) directions, finite for finite T, and algebraic amplification through the kernel lift (below).  (Standard.)
Discrete GD with eta lambda_1 > 2 (edge of stability) can be genuinely chaotic (period-doubling of the top mode); in the
regime used here (eta lambda_1 <= 1, verified equal to the 4x-smaller-step trajectory to four digits) it is not.

PROPOSITION 30 (what is hard is conditioning, with three provable components).  Let O_[0,t0] be any observable of the
history up to t0 (losses, kernels, outputs on the data). The map O_[0,t0] -> E(T) is deterministic and analytic (Prop. 9a)
but its inverse problem is ill-conditioned by the product of three factors, each a theorem of this document:
 (i) HIDDEN STATE (Theorem 10): the weights' action outside the span of the data activations is invisible to every
     data-indexed observable at init, and enters the data-indexed dynamics only through the propagation term as the
     activations move. Two networks identical in every observable can differ there by any amount.
 (ii) VISIBILITY SUPPRESSION (Prop. 9b): the modes that carry the late error contribute to any early observable at relative
     order lambda_k(t0) t0 (measured 1e-6 at t0 = 0.01 T, 1e-2 at 0.1 T), because they have not yet moved.
 (iii) AMPLIFICATION BY THE LIFT (Theorems 12-13, section 8): the same modes are later lifted by factors 10^2-10^4 (gain law),
     coherently across layers (composition), with the lift quadratic in the accumulated drive (Q ~ U^2) and compounding
     across stages. A relative difference d in a tail feature at t0 becomes a difference of order (gain) x d in the late
     kernel there, and the late error is exponentially sensitive to the integrated kernel (E_k ~ exp(-int lambda_k)).
Putting (ii) and (iii) together: to predict E(T) to relative precision eps from the early history, the tail components
must be known to relative precision ~ eps x lambda_k(t0) t0 / (sensitivity), i.e. ~ 1e-7 at t0 = 0.01 T for eps = 0.1, five
orders of magnitude below the natural initialisation-to-initialisation variation of the early history (2e-2 measured) and
below single precision. This is an information/conditioning impossibility, not a dynamical one: the late state is a
smooth function of the early state whose relevant derivatives are enormous exactly in the directions that are
unobservable early. The difficulty is intrinsic to the spectral structure (ten decades of eigenvalues, the slow modes
being both the invisible ones and the ones feature learning targets) and does not go away with more data, more width or
a better model; it goes away only by carrying the hidden state forward, which is what the chain (and training) does.
PERTURBATION TEST (code/lyapunov.py, running): perturb theta_0 by 1e-7 relative in a random direction, in the tail-feature
direction and in the top-feature direction, train all copies, track ||dtheta(t)||/||dtheta(0)|| and ||dE||/||E|| on a log grid.
Prediction from Props 29-30: no exponential growth; algebraic growth of the tail perturbation by up to the gain
(10^2-10^4) and much less for the top; the random direction in between. Numbers to be appended.

> CORRECTION (2026-10-11, after user review): statements in this document and in the conversation that the statistical
> (coherent/ensemble) chain "works to about 2x" refer to ONE case (H = 96, depth 2, n = 200). At depth 3 the Gaussian ensemble
> was off by 5x on the rank-100 gain and 2.3x on the error, and the statistical chain was never run at width above 96.
> The claim that the ensemble error and the seed spread shrink with width (e.g. ~1/sqrt(H)) is a CONJECTURE with no
> measurement behind it; the depth-3 results point the other way for depth. Both are to be measured in the redo
> (seed spread of E(T) vs width and depth; own-init vs other-init chain gap vs width and depth), at large n.
> What IS established about predictability is the mechanism: the late state depends on initialisation-specific hidden
> state (Theorem 10) that no data-indexed observable sees early (Prop. 9b) and that feature learning amplifies (gain law);
> a statistical description predicts the distribution over initialisations, never the draw.

### 12.4 Item 5 result (transfer sweep, 24 GD runs, n = 200; runs/gt_transfer, code/transfer_analysis.py): INCONCLUSIVE
Widths 48/96/192/384 at depth 2 and 96/192 at depth 4, output scales 0.25/0.5/1/2, budget 3000/s^2.
- The training-error optimum is NOT bracketed: the smallest scale (richest) is best at every width and depth
  (E(T) at s = 0.25: 0.0090/0.0074/0.0045/0.0041 for widths 48-384 at depth 2), so "optimal s vs width" cannot
  distinguish standard from muP-like transfer in this range. On TRAINING error at fixed budget, more feature learning
  simply wins here; the transfer question needs a regime where richness has a cost (test error, or a budget/stability
  constraint), which the training-error criterion does not have.
- The feature-learning/lazy ratio R(tau) = (E'KE - E'K0E)/E'K0E does not collapse under either labelling: at fixed s it
  GROWS with width (s = 1, tau = 3000: 4.1, 9.5, 16.3, 18.3 for widths 48-384), and rows with equal s*sqrt(H/96) differ
  by 2x. So the finite-width criterion of Theorem 26's remark is not satisfied by either parameterisation at these
  widths and n; the feature-learning share increases with width in standard parameterisation.
Status: no claim on item 5 from this sweep. Redo at large n with a criterion that has an optimum.
PERTURBATION TEST RESULT (code/lyapunov2.py, runs/lyapunov_full.txt; H96 d2 Delta 0.5, full run to T = 3000, 1e-7 relative
perturbations of theta_0; separation ||dtheta(t)||/||dtheta(0)|| and ||dE(t)||/||E(t)||):
   t:          0.1     1      10     100    1000   3000
   random:     1.00    1.00   1.09   1.9    4.2    6.1      (dE/E: 2e-8 -> 1.4e-6)
   tail(300):  1.00    1.03   1.7    5.2    13.5   21.5     (dE/E: 1e-8 -> 7.4e-6)
   top(1):     0.08    0.06   0.15   0.72   3.2    4.9      (dE/E: 2e-7 -> 1.7e-6; contracted first, then regrown)
Growth is algebraic throughout (roughly t^{0.2-0.4} after t ~ 10), with no exponential phase anywhere in 3.5 decades
of time; the tail-feature perturbation is amplified most (x21 in weights, x700 in the residual relative to its start),
the top-feature perturbation is contracted by 16x while the top mode is fit and then regrows through the lift: exactly
the ordering of Prop. 30 (ii)-(iii). The total finite-time amplification of a weight perturbation over the whole run is
one order of magnitude; of a residual perturbation two to three orders. Combined with the visibility factor 1e-6 this is
the conditioning statement of Cor. (section 14) with measured numbers: not chaos. [n = 200]

## 15. Error propagation through the chain (written 2026-10-12)

Setting as in sections 8-10. A truncated chain replaces the exact stage map by its order-k Taylor expansion about the
stage's initial state; write e_s for the local error committed at stage s (the Taylor remainder of order k+1) and ask how
it appears at time T. Three exact structures decide this.

PROPOSITION 31 (the propagator is the variational flow; it is algebraic, not exponential).  A perturbation d theta of the
state at time s evolves under gradient flow by the variational equation d(d theta)/dt = -H(t) d theta with H = grad^2 L
= J^T J / n + (1/n) sum_i E_i grad^2 f_i (Gauss-Newton part + residual-curvature part). Hence the propagator Phi(T, s) =
Texp(-int_s^T H) satisfies
      ||Phi(T, s)||_2 <= exp( int_s^T lambda_max(-H(u))_+ du ) <= exp( int_s^T (1/n) ||E(u)|| H_2(u) du ),
since the Gauss-Newton part is positive semidefinite and only the residual-curvature part, of size <= ||E|| H_2 / n, can
expand. The right-hand side is finite for finite T and its integrand DECAYS with the residual, so the expansion of a
perturbation is a finite factor accumulated mostly early, not an exponential in T.  (Proof: standard Gronwall; the split
of H is Theorem 1's Jacobian structure.)  MEASURED [n = 200, full run, Prop. 29]: ||Phi|| along the worst (tail-feature)
direction = 21 in weights over the whole run; 6 in a random direction; the induced residual perturbation grows 10^2-10^3
relative to its start; growth ~ t^{0.2-0.4} after t ~ 10.

PROPOSITION 32 (how a stage error reaches T: additive feature offset + damped residual error + feedback).  Let the truncation
at stage s produce an error d theta_s (equivalently d Psi_s in the features) and no other error. Then at T:
 (i) FEATURE OFFSET (additive, undamped): Psi_T = Psi_0 + sum_r d Psi_r, so the feature error is carried to T as a permanent
     offset Phi(T, s) d Psi_s, and the kernel error at T is dK_T = (Psi_T dPsi^T + dPsi Psi_T^T + dPsi dPsi^T)/n with
     dPsi the propagated offset — linear in the offset for offsets smaller than the features, quadratic beyond.
 (ii) RESIDUAL ERROR (damped, driven by the kernel error): d E' = -K dE - dK E, so dE(T) = -int_s^T Texp(-int_u^T K) dK(u) E(u) du:
     errors in the residual are contracted by K in every direction the kernel has already lifted and persist only in the
     slow modes; they are sourced by the kernel error acting on the residual, i.e. by (i) times E(u), which is small late.
 (iii) FEEDBACK (second order): the wrong residual drives wrong later feature changes, d(dPsi_r) ~ Psi'[J^T dE] for r > s;
     this is the term the propagator of Prop. 31 bounds, and it is what distinguishes the chain from a sum of independent
     stage errors. It is of order ||dE|| x (drive per stage), so it inherits the smallness of (ii).
Proof: differentiate the stage identity and Theorem 1 along the trajectory; (i) is the telescoping sum of section 10; (ii) is
variation of constants for Theorem 1; (iii) is Theorem 4 linearised.  []

THEOREM 33 (global error of the order-k chain).  Let the stages satisfy alpha_s ||Psi^{(k+1)}|| <= eps ||Psi'|| ... (the
order-k stage criterion: the Taylor remainder of order k+1 is at most eps times the first-order increment). Then the local
feature error of stage s is ||e_s|| <= eps ||dPsi_s||, and by Prop. 32(i)-(ii) and Prop. 31,
      ||Psi_T^{chain} - Psi_T|| <= eps  sum_s ||Phi(T, t_s)||  ||dPsi_s||  <=  eps  Phi_max  sum_s ||dPsi_s||,
      ||E_T^{chain} - E_T||  <=  (2 eps Phi_max / n) ||Psi_T||  sum_s ||dPsi_s||  x  int_0^T ||E(u)|| du  +  O(eps^2),
with Phi_max = max_s ||Phi(T, t_s)|| (measured 6-21). Because the stage errors are proportional to the stage INCREMENTS and
accumulate additively, their sum is eps times the TOTAL feature change along the path, not eps times the number of stages:
the chain is stable in the sense of numerical analysis (consistency of order k per stage + an algebraic propagator =>
global error O(eps), uniformly in the number of stages). The residual error inherits an extra factor int ||E||, the
integrated drive, which is finite and dominated by the early phase. As eps -> 0 (shorter stages at fixed k, or higher k at
fixed stage length) the truncated chain converges to the exact trajectory; this is the convergence observed numerically
(3 digits across a 10x range of stage budgets with k = 1) and the precise sense in which "the chain holds".
Proof: triangle inequality over stages with Prop. 32(i); Prop. 31 for the propagator; Prop. 32(ii) for the residual.  []

COROLLARY 34 (orders versus stages).  The stage criterion can be met either by many short stages at low order or by few
long stages at high order: with alpha the stage length, the order-k remainder is O(alpha^{k+1} ||Psi^{(k+1)}||), so the number
of stages needed for a target eps scales as S_k ~ eps^{-1/k} int (||Psi^{(k+1)}|| / ||Psi'||)^{1/k} dt. Both ends converge by
Theorem 33; they differ only in cost. For PROOFS the k = 1, S -> infinity end suffices (it is the ODE), and nothing in
sections 7-14 needs more than first order per stage plus the propagator bound. For COMPUTATION the chain is not a cheap
object: long stages need high-order directional derivatives of the Jacobian at every expansion point (k nested
Jacobian-vector products), short stages need ~1e4 of them; either way its cost is that of integrating the dynamics, as
the user noted. Its value is the error accounting above, which makes every truncated or reduced version of it (the
coherent closure, the richness series, the ranker) a controlled approximation with a named error term rather than a model.

Remark (where truncations do not propagate benignly). Theorem 33's constants are finite but not small when (a) the
residual-curvature part of H is large early (then Phi grows: the "sharpness-reduction" phase is exactly where
lambda_max(-H) > 0), and (b) the stage error lands in a slow mode, where Prop. 32(ii) gives no damping and the offset of
(i) is later amplified by the lift of that mode (gain 10^2-10^4). This is the chain's own statement of Prop. 9b-30: errors
(or ignorance) in the slow modes early are the ones that survive and are amplified; errors in fast modes are forgotten.
PROPAGATOR BOUND CHECK (code/propagator_bound.py, runs/propagator_bound.txt; reference run): the bound exp(int ||E|| H2 / n) of
Prop. 31 is valid but useless (exp(8.4) at t = 130, exp(15.7) at t = 620) against a measured propagator of 6-21 over the whole
run: it bounds the negative curvature of H by its worst case ||E|| H2 / n ~ 1e-2, while the measured growth x21 over
3000 time units corresponds to an integrated negative curvature ln 21 ~ 3, i.e. an AVERAGE lambda_max(-H) ~ 1e-3 (1e-5 of
lambda_1). The direct power-iteration estimate of lambda_max(-H) along the path returned 0 (it cannot resolve a negative
eigenvalue five orders of magnitude below the top one in 40 iterations); a Lanczos/inverse-iteration estimate would be
needed to verify the sharp form exp(int lambda_max(-H)_+) = 21 directly. Status: Prop. 31 as an inequality stands; its
first (H2) form is loose by exp(13); the sharp form is consistent with the measurement but not independently computed.

## 16. The error at T term by term: the Dyson expansion in the stage increments (written 2026-10-12)

Write K(t) = K_0 + Delta(t), Delta(t) = sum_{s: t_s <= t} (Cross_s + Q_s) (the accumulated stage increments; in the limit of
short stages, Delta(t) = K(t) - K_0 exactly). The residual equation E' = -(K_0 + Delta) E has the exact interaction-picture form
      E(T) = e^{-K_0 T} E_0  -  int_0^T e^{-K_0 (T-u)} Delta(u) E(u) du ,
and iterating gives the Dyson series in the increments,
      E(T) = sum_{m>=0} (-1)^m  int_{T>u_1>...>u_m>0}  e^{-K_0(T-u_1)} Delta(u_1) e^{-K_0(u_1-u_2)} Delta(u_2) ... Delta(u_m) e^{-K_0 u_m} E_0 .

THEOREM 35 (term-wise facts, exact).
 (a) The series converges absolutely for any bounded Delta: ||term_m|| <= (int_0^T ||Delta||)^m / m! ||E_0||, since ||e^{-K_0 t}|| <= 1.
     Hence E(T) is an entire function of the increments and the series IS the exact solution, not an approximation.
 (b) Every term is a product of lazy propagators and increments; every increment is a sum over stages of Cross_s + Q_s,
     each built from the stage's feature change dPsi_s, which by Theorem 11 (at that stage) is a rank-<= ny core aligned with
     the stage's backprop directions and driven by the stage's residual, plus bulk. Therefore every term of the series,
     and hence E(T), lies in the span generated by the lazy flow of E_0 and the stage targets; the late residual cannot
     contain anything that neither the initial kernel nor the history of drives has produced.
 (c) The m = 1 term, paired with the lazy solution, is
      < e^{-K_0 T} E_0 , term_1 >  =  - int_0^T  < e^{-K_0 (2T - u)} E_0 ... >  — precisely:  -int_0^T < E_lazy(2T-u), Delta(u) E_lazy(u) > du
     in the fixed-basis sense (since e^{-K_0 (T-u)} is symmetric and commutes with e^{-K_0 T}). Its sign is determined by
     two properties at each u: (S) Delta(u) >= 0 on the relevant subspace, and (P) the alignment of E_lazy(u) with
     E_lazy(2T-u) in the modes Delta touches (persistence of the lazy residual's direction).

THEOREM 36 (the integrated rate comparison, term by term; conditional on (S) and (P)).  Suppose for u in [t_0, T]:
 (S)  Delta(u) >= 0  (Lemma 15: the accumulated stage increments are positive semidefinite after the transient), and
 (P)  for the modes k with Delta(u)-components, E_lazy,k(u) E_lazy,k(2T-u) >= 0 — automatically true in a fixed basis
      (both are E_{0,k} e^{-lambda_k (.)}), and the content of the persistence condition when the basis rotates.
Then the first-order term reduces the energy, < E_lazy(T), term_1 > <= 0, and by the same argument applied recursively
(each additional increment is paired with lazy propagators on both sides) every odd term reduces it and every even term
is bounded by the square of the previous odd one; summing,
      ||E(T)||^2  <=  ||E_lazy(T)||^2  -  2 int_0^T < E_lazy(2T-u), Delta(u) E_lazy(u) > du  +  O(||Delta||^2),
with an explicit, nonnegative first correction. In the fixed-basis case (P) is automatic and the statement is exact to
all orders: E_k(T) = E_{0,k} exp(-lambda_k T - int_0^T Delta_k(u) du) <= E_lazy,k(T) mode by mode (Prop. 6 recovered as the
resummed series). With rotation, (P) is the hypothesis, and the term-wise form shows it is needed at every order and
nowhere else.
Proof. (a) is the standard Dyson bound; (b) is Theorem 11 inserted into each increment; (c)-(36) are the inner products
written out with e^{-K_0 t} symmetric; the resummation in the fixed-basis case is the scalar ODE per mode.  []

What this adds to section 10. Theorem 19's rate comparison (L' <= -E^T K(t_0) E under (S)) is the differential form; Theorem 36
is the same statement integrated to T and expanded term by term, which (i) identifies (S) and (P) as the complete set of
hypotheses, order by order, (ii) gives the first correction explicitly in terms of the lazy residual and the increments,
and (iii) recovers Prop. 6 (frozen kernel bound) as the exactly resummed fixed-basis case. The two lemmas remain the
open items; the method shows there is no third.

## 17. Term-wise corollaries (written 2026-10-12): statements that become unconditional once the right series is used

Throughout, Delta = K(T) - K_0 (or the accumulated increments to any time), with eigenpairs (lambda_k, v_k) of K_0.

COROLLARY 37 (Euler vs direct, order by order).  By Theorem 26 the lift of any direction under the direct target is the
richness series evaluated at Delta epsilon, so its k-th term is Delta^k times Euler's: inside the common radius,
      g_v^{direct}(tau) - 1 = sum_k a_k(tau) Delta^k epsilon^k  <=  sum_k a_k(tau) epsilon^k = g_v^{Euler}(tau) - 1   term by term
whenever the a_k are nonnegative, and in general |direct term_k| = Delta^k |Euler term_k|. No sign lemma is needed for the
lift comparison; the error comparison follows from Theorem 36 under (S) and (P).  (Immediate.)

COROLLARY 38 (sign-definite second-order sharpening and level repulsion).  For a simple top eigenvalue, the Rayleigh-
Schrodinger series of lambda_1(K_0 + Delta) in Delta is
      lambda_1 + v_1^T Delta v_1 + sum_{k>1} |v_1^T Delta v_k|^2 / (lambda_1 - lambda_k) + O(Delta^3),
and the second-order term is >= 0 for ANY symmetric Delta (all denominators positive). For the smallest eigenvalue the
second-order term is <= 0. Hence, independently of the sign of the cross term (Lemma 15), every coupling between the top
mode and the rest pushes lambda_1 UP at second order, and every coupling pushes the bottom of the spectrum DOWN:
progressive sharpening of the top mode has an unconditional second-order component, and the deep tail has an
unconditional second-order depression (level repulsion), which is the measured early DECREASE of the kernel's form along
init-tail directions (12.2, rank 200-300) before the first-order lift Q overtakes it.  (Standard perturbation theory.)

COROLLARY 39 (rotation lives where the spectrum is dense).  To first order the eigenvector of mode k rotates by
      dv_k = sum_{j != k} [ v_j^T Delta v_k / (lambda_k - lambda_j) ] v_j ,
coupling over gap. With the init spectrum ~ k^{-a} (a ~ 5 here) the relative gaps are (lambda_k - lambda_{k+1})/lambda_k ~ a/k:
O(1) at the top, ~1 % at rank 100, ~0.5 % at rank 200. For a coupling of fixed relative size the rotation angle therefore
grows like k/a with rank: the top band keeps its basis, the mid band mixes, the tail reorganises completely — the
measured retention 0.6-0.9 / 0.5 / 0.85 (Prop. 9d; the last is high only because the deep tail is "nothing mixing with
nothing"). Since rank in the init basis is roughness (F-law), this is the statement that rotation of the lifted
subspace is concentrated in the rough modes, i.e. the fixed-basis assumption of Prop. 6-7 fails exactly where the roughness
theory says the action is. The same series gives the mixing between two directions as |coupling|/|gap|, a computable
criterion for when a direction keeps its identity along training.  (Standard.)

COROLLARY 40 (the seed spread is bounded by the propagation term).  Two initialisations with the same unit statistics
(Theorem 10's family, or two draws of the same distribution) have Dyson series (Theorem 35) that differ only through the
propagation part of the stage features, dPsi_s^{prop} (Theorem 4). Hence
      ||E_T(theta_0) - E_T(theta_0')||  <=  Phi_max  x  (2/n)||Psi_T||  x  sum_s ||dPsi_s^{prop}(theta_0) - dPsi_s^{prop}(theta_0')||  x  int_0^T ||E||  +  O(2nd order),
by Prop. 32/Theorem 33 applied to the difference. The irreducible uncertainty of ANY statistical description of a single
run is therefore bounded by the propagator times the accumulated propagation-part mismatch: an explicit, measurable,
width- and depth-dependent quantity. The question "does the statistical chain become exact at large width" is the
question "does sum_s ||dPsi_s^{prop}|| shrink with width", an inequality to measure rather than a conjecture to argue.
[n = 200 inputs for the bound: Phi_max = 6-21; the propagation part is 11-28 % of the unit displacement at t = 8, width 96.]

Remark (what remains conditional). Corollaries 37-40 need no sign lemma. Everything about the ERROR at T beyond the lazy
bound (Theorem 36) still needs (S) and (P); Corollary 38 shows that at second order the top eigenvalue's growth does
not, and Corollary 39 shows where (P) is expected to fail.

COROLLARY 41 (targeting to all orders in the richness parameter).  In Theorem 26 the k-th coefficient a_k(tau) is built from
directional derivatives of J at theta_0 along the displacements theta^(1..k-1), each of which is -(1/n) J_0^T U^(j) with U^(j) the
j-th order correction of the integrated residual; so the k-th order feature change lies in the span of
{ J^{(m)}_0 [J_0^T U^(j_1), ..., J_0^T U^(j_m)] } with j_1 + ... + j_m <= k, all of which are functions of the lazy drive history
through the init Jacobian and its derivatives. Hence every term of the lift, and the resummed lift for |epsilon| < R(tau),
lies in the span generated by the init features and the drive history: the targeting of section 8 holds to all orders
inside the radius, not only at first order.  (Induction on k through the recursion of Theorem 26.)

COROLLARY 42 (visibility and amplification without the fixed-basis assumption).  Differentiate the Dyson series (Theorem 35)
with respect to a component c of the early state that lives in a slow mode of K_0 (a tail feature or tail residual
component). In the lazy term e^{-K_0 T} E_0 the dependence is through e^{-lambda_k T} ~ 1: order one at T, but its
contribution to any early observable is through e^{-lambda_k t_0} - 1 ~ lambda_k t_0 (Prop. 9b, now a statement about the
first Dyson term with no basis assumption). In the m-th term the dependence on c enters through Delta(u) (whose stage
cores are driven by the residual, hence by c with weight lambda_k-independent) and through the propagators; by Prop. 31-32
each such term is bounded by the propagator times the derivative of the increments with respect to c, which is the
lift's sensitivity to the drive along mode k — the amplification of Prop. 30(iii), measured as the gain. Thus
      | d E(T) / d c |  >=  (gain along k) x O(1)        while        | d (early observable) / d c |  <=  lambda_k t_0 x O(1),
term by term, in the actual (rotating) basis: the conditioning statement (Cor. in section 14) survives rotation, with the
rotation only redistributing WHICH late components carry the sensitivity (Cor. 39), not its size.  (Term-wise
differentiation of an absolutely convergent series.)

Remark (validation).  Theorem 35 holds verbatim for E_val with K_0 -> K_vt(0) on the left factor and Delta -> the cross-kernel
increments; the m = 0 term is lazy transfer, m >= 1 feature-learning transfer, and Theorem 36's sign analysis applies with
(S) and (P) stated for the cross-kernel. Not written out separately.

COROLLARY 43 (the sign condition is discharged on the support of E(T); the obstruction is rotation alone).  By Theorem 13, on
every direction v with ||D_v|| > 2 ||Psi_0^T v|| the accumulated increment satisfies v^T Delta v >= ||D_v||^2/n - 2 l_v ||D_v||/n > 0,
i.e. (S) holds there without any lemma. MEASURED (reference at T, n = 200): the inequality holds on 100 % of the init
directions beyond rank 10 (40 % within the top 10), and 100 % of the energy of E(T) lies in directions where it holds
(E(T) is 93 % in ranks > 200, 45 % in ranks > 400). So for the error at T the hypothesis (S) of Theorem 36 is satisfied on
the whole support of the late residual, and what remains is (P) alone.
(P) in its accumulated first-order form FAILS completely: the increment Delta = K_T - K_0 is far outside first-order
eigenvector perturbation theory (max_j |v_j^T Delta v_k| / |lambda_k - lambda_j| = 16 at the top, 160 at ranks 11-50, 2e3 at
51-100, 6e4 at 101-200, 1e7-6e9 in the tail), as it must be when gains are 10^2-10^4. Corollary 39's rotation formula is
therefore a PER-STAGE statement (small increment, current gaps), and persistence has to be stated and proved stage by
stage with the current spectrum — which is exactly what the chain provides and a single expansion about K_0 cannot.
Consequence: the only open hypothesis for "feature learning beats the lazy bound at T" is the per-stage rotation of the
residual's directions in the rough tail, where the current gaps are smallest. This is the frontier of the theory, stated
as a single quantity: the stage-by-stage ratio |coupling| / |current gap| on the modes carrying the residual.

COROLLARY 44 (validation, term by term).  Theorem 35 holds for E_val with the left factor replaced by the cross kernel:
      E_val(T) = E_val,lazy(T) - int_0^T e^{-K_vt(0)...} ... — precisely, E_val' = -K_vt(0) E - [K_vt(t) - K_vt(0)] E, so
      E_val(T) = E_val(0) - int_0^T K_vt(0) E(u) du - int_0^T [K_vt(u) - K_vt(0)] E(u) du ,
with E(u) itself given by the Dyson series of Theorem 35. The first integral is lazy transfer (depends on the training
residual history and the INITIAL cross kernel only), the second is feature-learning transfer (the lifted cross kernel, whose
coherent part is the training stage core transported by A_{val,tr}). The sign analysis of Theorem 36 applies to the
second term with (S) and (P) stated for the cross-kernel increments: under equal unit statistics on the two samples, the
cross-kernel core is the training core transported, so (S) transfers whenever the transported direction has positive
overlap with the validation residual — the "shared direction" condition of section 13.  (Direct.)
MEASURED (per-stage rotation on the residual-carrying modes, reference run, consecutive snapshots; n = 200):
   stage 0->0.3:   coupling/gap median 0.35, residual band (ranks 19-141) retained 0.99,  increment/eigenvalue 0.00
   stage 0.3->0.8: 0.77, retained 1.00, 0.05
   stage 0.8->4.4: 26,   retained 0.92, 3.0
   stage 4.4->44:  780,  retained 0.76, 61
   stage 44->428:  200,  retained 0.55, 27
With stages this coarse the increments on the residual band are 3-60x the eigenvalues; in the chain's fine stages the
per-stage coupling/gap is proportionally small and Cor. 39 applies per stage, but the ACCUMULATED rotation is what it is:
the subspace carrying the residual at t = 44 is only 55 % retained at t = 428. So (P) in the init basis is FALSE for the
late residual on this testbed, not merely unproved, and Theorem 36 cannot be made unconditional by a per-stage argument
in a fixed basis. The residual lives in the rough band where the current gaps are smallest, and the targeted lift creates
the very directions the residual then moves into.
CONSEQUENCE / NEXT STEP (the moving-frame formulation). The correct persistence statement is relative to the CURRENT
eigenbasis: in the eigenbasis of K(t) the residual sits at the front (ranks where lambda_k(t) t ~ 1) and the front moves slowly
(d log k*/d log t ~ 0.2, Prop. 8a), so the residual's rank profile is persistent even though its init-basis direction is
not. The Dyson expansion should therefore be organised in the instantaneous eigenframe (an adiabatic / interaction
picture with the moving basis), where the increment has a diagonal part (the lift of the current modes: Q and the diagonal
cross term, both with controlled sign by Thm 13 + Cor. 38) and an off-diagonal part (the rotation, coupling/gap per stage),
and the energy inequality of Theorem 36 is written mode by mode in the moving frame with the off-diagonal part as a
named remainder bounded by the per-stage coupling/gap. In that frame (P) becomes "the residual's rank profile changes
slowly", a measured regularity, and the fixed-basis theorems (Prop. 6-7) are recovered as the adiabatic limit. This is a
derivation, not a computation, and it is the one piece that would make the error-at-T statement unconditional.

## 18. The moving-frame identity: persistence replaced by the front (written 2026-10-12)

LEMMA 45 (moving-frame dynamics, exact).  Let K(t) = V(t) Lambda(t) V(t)^T with V orthogonal and smooth where the eigenvalues
are simple, and c(t) = V(t)^T E(t) the residual in the instantaneous eigenframe. Then
      c' = -Lambda(t) c - A(t) c,      A = V^T V'  antisymmetric,
hence  d/dt ||c||^2 = -2 c^T Lambda c  exactly: the rotation A redistributes energy among the current modes and does not
change the norm; the energy decays only through the current eigenvalues on the current rank profile c_k(t)^2.
Proof: differentiate c = V^T E with E' = -K E and V^T V = I (so V^T V' is antisymmetric).  []

THEOREM 46 (rate comparison without persistence).  Suppose the sorted eigenvalues of K are nondecreasing on [t_0, T] —
which follows from the per-stage increments being positive semidefinite on the relevant subspace (Weyl), i.e. from (S),
discharged by Theorem 13 on every direction whose feature change exceeds twice its initial features (measured: the whole
support of the late residual). Then at every t in [t_0, T]
      -d/dt ||E||^2 = 2 sum_k lambda_k(t) c_k(t)^2  >=  2 sum_k lambda_k(t_0) c_k(t)^2 :
the instantaneous descent rate is at least the frozen-kernel rate evaluated on the CURRENT rank profile. No hypothesis on
eigenvector persistence enters: the moving frame absorbs the rotation into the profile.  (Lemma 45 + Weyl.)  []

What this does and does not give. It is the exact accounting of feature learning for the error: the lifted eigenvalues
help at every instant; what changes besides is the rank profile c_k(t)^2 — which feature learning pushes to higher ranks
(more modes fit, what remains is rougher) — so a closed comparison with the lazy trajectory ||E_lazy(T)|| needs the
evolution of the profile, not of the eigenvectors. That evolution is the front of Prop. 8 (lambda_eff t ~ 0.6-1.4,
d log k*/d log t ~ 0.2, measured), i.e. the residual sits at the modes with lambda_k(t) t ~ 1 and the profile moves slowly in
rank. The persistence hypothesis (P) of Theorem 36 is thereby replaced by a statement about the front — a regularity of
the profile in the moving frame, which the data support and which is what section 13 of the held-rough paper already
calls the moving cutoff — and the fixed-basis theorems (Props 6-7) are the special case A = 0. Proving the front law from
the chain (the trajectory integral of the stage targets, section 8 (ii)) is the remaining derivation for an unconditional
error bound at T; in the moving frame it is a one-dimensional statement about where the residual sits, not a
600-dimensional statement about which way it points.

COROLLARY 47 (extreme eigenvalues to all orders; what the eigenvalue series does and does not need).
 (a) The Rayleigh-Schrodinger series of a simple eigenvalue lambda_k(K_0 + Delta) is a power series in Delta with radius of
     convergence at least half the isolation distance of lambda_k from the rest of the spectrum (Kato): it is NOT a harmonic
     series; it converges geometrically for ||Delta|| < gap_k / 2 and in general not beyond. For the top eigenvalue the gap is
     O(lambda_1) and the series converges for every stage increment of the chain; for a tail eigenvalue the gap is ~ a lambda_k / k
     and convergence needs stage increments that small — the chain supplies them by taking stages short enough, and at
     coarse resolution the series fails exactly where Cor. 43 measured coupling/gap >> 1.
 (b) The SIGN conclusion of Cor. 38 does not need the series at all. By the variational principle, for ANY symmetric Delta,
        lambda_max(K_0 + Delta) >= v_1^T (K_0 + Delta) v_1 = lambda_1 + v_1^T Delta v_1 ,
        lambda_min(K_0 + Delta) <= lambda_min + v_min^T Delta v_min ,
     so the sum of ALL orders beyond the first is >= 0 for the top eigenvalue and <= 0 for the bottom one, exactly, with no
     convergence question: the top is pushed up and the bottom down by the totality of the higher-order terms, whatever
     their individual sizes. Cor. 38 is the second-order term of this exact inequality. The same inequality holds stage by
     stage and composes along the chain (each stage's top eigenvalue is at least the previous one's plus the first-order
     term), which is the all-orders form of "sharpening has an unconditional component".
 (c) For interior eigenvalues neither sign is forced; their all-orders behaviour is what the chain tracks and what the
     rotation remainder of section 18 measures.

COROLLARY 50 (interior eigenvalues, all orders).  If a stage increment is positive semidefinite on a subspace containing the
relevant eigenvectors, Weyl's monotonicity (A <= B  =>  lambda_k(A) <= lambda_k(B) for every k) gives lambda_k nondecreasing for
EVERY sorted index, with no series. With (S) discharged on the support of the late residual (Cor. 43), all sorted
eigenvalues on that support are nondecreasing along the chain, exactly. The eigenvalue-series question is therefore
closed: extreme eigenvalues by the variational principle (Cor. 47), interior ones by monotonicity, both to all orders.

## 19. The cutoff bound: a two-sided, unconditional bound on the error at T with a named rotation remainder (2026-10-12)

Setting: Lemma 45's moving frame, K(t) = V(t) Lambda(t) V(t)^T, c = V^T E, sorted eigenvalues nondecreasing on [t_0, T] (Cor. 50,
on the support of the residual). Fix the index set u = { k : lambda_k(T) T < 1 } (the modes still unfit at T; its complement f
are the modes with lambda_k(T) T >= 1), and let k* = min f be the CUTOFF RANK at T. Transport the index set along the eigenvector
flow: P(t) = V(t) Pi_u V(t)^T, with Pi_u the fixed coordinate projector onto u. Define the cross-front rotation rate
      rho(t) = max_{k in u, j in f} |A_kj(t)|,     A = V^T V'   (per Cor. 39, A_kj = v_j^T K' v_k / (lambda_k - lambda_j)).

THEOREM 48 (energy above the cutoff, two-sided).  With E_u = P E and E_f = (I - P) E,
   d/dt ||E_u||^2 = -2 E_u^T K E_u + L(t),    d/dt ||E_f||^2 = -2 E_f^T K E_f - L(t),    |L(t)| <= 2 rho(t) ||E_u|| ||E_f||,
(the leakage L = E^T P' E is the only coupling between the two subspaces and it is antisymmetric: what leaves one enters
the other). Consequently, writing lambda_u(t) = max_{k in u} lambda_k(t) <= 1/T x (1 + ...) ... precisely lambda_u(t) <= lambda_{k*-1}(t) and
lambda_f(t) = min_{k in f} lambda_k(t) = lambda_{k*}(t):
   (floor)    ||E(T)||  >=  ||E_u(T)||  >=  ||Pi_u c(t_0)|| exp(-int_{t_0}^T lambda_u)  -  int_{t_0}^T rho ||E_f|| ,
   (ceiling)  ||E(T)||^2 <=  ||Pi_u c(t_0)||^2  +  2 int_{t_0}^T rho ||E_u|| ||E_f||  +  ||E_f(T)||^2 ,
              with  ||E_f(T)|| <= ||(I - Pi_u) c(t_0)|| exp(-int_{t_0}^T lambda_f)  +  int_{t_0}^T rho ||E_u|| .
At t_0 = 0, Pi_u c(0) is the initial residual's energy on the init modes with ranks >= k*: "the energy of the initial error
above the cutoff", with the cutoff rank set by the LIFTED spectrum at T through lambda_{k*}(T) T = 1.
Proof. Differentiate ||P E||^2 with E' = -K E and [P, K] = 0 (P is a spectral projector of K(t)); the term E^T P' E is the
leakage, and P' = sum_{k in u, j in f} A_kj (v_j v_k^T + v_k v_j^T) (eigenvector perturbation with a FIXED index set, so no
crossing terms), giving the bound on |L|. Gronwall on each subspace with lambda_u, lambda_f as the extreme eigenvalues of K on
them gives the floor and ceiling; the two leakage integrals are the same quantity with opposite sign.  []

What the theorem says. The error at T is bracketed by the energy of the initial residual above the lifted cutoff, from
below after decay at the slow rate of the unfit subspace (which is < 1/T by definition of u, so the decay factor over
[t_0, T] is at least t_0/T-ish) and from above with no decay, plus the fit part (which decays at the front rate and is
small) and plus/minus the LEAKAGE: the rotation of the residual across the front, rho, weighted by the residual on BOTH
sides. The leakage is the only place where eigenvector rotation enters, and it is suppressed by ||E_f||, the fit residual,
which decays fast: rotation among unfit modes (the dense tail, where Cor. 39 says rotation is largest) costs nothing,
because it moves energy within P, not across it. This is why the measured 45-55 % retention of the residual's init-basis
subspace (Cor. 43) is compatible with a sharp energy statement: what matters is not whether the residual keeps its
direction but whether it crosses the front, and crossing requires coupling to a mode that is being fit, whose residual
component is already small. No persistence hypothesis appears; (S) enters only through Cor. 50 for the monotone
spectrum, and it is discharged on the support of the residual.

How the chain closes it. The cutoff rank k*(T) is determined by the lifted sorted spectrum at T, which Theorem 13 brackets
two-sidedly in terms of the accumulated feature change along the init directions (and the gain law summarises); the
cross-front rotation rate rho is the per-stage coupling/gap across rank k* (Cor. 39 with current gaps), a chain quantity;
Pi_u c(0) is the initial residual's spectrum in the init basis. So Theorem 48 is the statement "E-dot -> E(T)" the chain was
built for: exact identities at every stage, summed, with every input either computable at initialisation (the init
spectrum of the residual), bracketed by the stage sums (the cutoff rank), or a named remainder (the leakage), and with the
held-rough floor of the static theory recovered as the case where the cutoff is taken from the FROZEN spectrum
(lambda_{k*}(0) T = 1) — the lift moves the cutoff up in rank, and that is the whole effect of feature learning on the
error, in one scalar: k*(T).
MEASURED [n = 200, reference]: the residual's central band at t = 44 is ranks 51-180 and at t = 428 ranks 77-148, i.e. the
front and the residual move together (lambda_eff t = 0.6-1.4, Prop. 8a), which is the regime in which the leakage is a
correction: per-stage coupling/gap across the front is what Cor. 43's stage table measured (0.35-0.77 at the earliest
stages, far larger at the coarse late stages), to be remeasured with fine stages at the redo.

COROLLARY 49 (the grokking / plateau-end time, bracketed by the chain).  For a direction v (or an index band) the time T_v at
which it leaves the unfit set is the solution of lambda_v(T_v) T_v = 1. By Theorem 13,
      (||D_v(t)|| - l_v)_+^2 / n  <=  lambda_v(t)  <=  (||D_v(t)|| + l_v)^2 / n ,
so T_v is bracketed by the solutions of  (||D_v(t)|| -/+ l_v)^2 t / n = 1, with D_v(t) the accumulated coherent feature change
along v, driven by the residual history (Theorem 11 per stage, Cor. 41 to all orders inside the radius), plus the bulk and
propagation remainders. Before T_v the direction's residual component is protected by Theorem 48 (it is in u, decays at
most at rate 1/t, and loses energy only by leakage); at T_v it crosses the front and is fit at the front rate. This is
the delayed-feature-learning event as a chain statement: an exact crossing condition, a two-sided bracket on its time
from the accumulated drive, and the event's sharpness set by the leakage rate rho at the crossing. Inside the richness
radius the bracket is the inversion of the series of Theorem 26 (section 12.2); outside it, it is a chain computation.
What is NOT given is a closed form for D_v(t) over a whole run — that is the self-consistent chain — and the literature's
timings (weight decay ~ 1/(eta lambda_wd); two-layer toys) are this bracket evaluated in settings where D_v is explicit.
CORRECTION TO THEOREM 48 (same day). The index set u must be transported by CONTINUITY of the eigenvector flow, not by
sorted rank: eigenvalues cross massively during training (init directions of sorted rank 300 are lifted 10^6-fold and
overtake; Cor. 43), so "ranks >= k* at T" and "ranks >= k* at 0" label different modes. The statement and proof hold
with P(t) = V(t) Pi_u V(t)^T defined by following each eigenvector continuously (through crossings this is the eigenvector
flow, well defined where the spectrum is simple); then Pi_u c(0) is the initial residual's energy on the init-time
eigenvectors that EVOLVE into the unfit set at T — not the energy above sorted rank k* at t = 0. A check using sorted
ranks at both ends (reference, n = 200) gives ||Pi_u c(0)|| = 8.24 (lifted cutoff k* = 301) against ||E(T)|| = 0.567: the
inequality holds but by 14x, which measures the mislabelling, not the leakage. Computing the correctly transported
Pi_u c(0) requires tracking the eigenvectors along the run (the chain's per-stage eigendecompositions do this; the five
saved snapshots cannot), and is deferred to the redo. The earlier line comparing with a "frozen-cutoff floor" is
withdrawn: the held-rough floor uses the sigma_R-based moving cutoff, not the frozen spectrum, and was not evaluated.
What survives unchanged: the energy identity with the leakage as the only coupling, its bound by rho, the two-sided form,
the absence of any persistence hypothesis, and Cor. 49's crossing condition (which is basis-free: lambda_v(t) t = 1 along a
fixed direction v uses v^T K v, not a sorted rank). What the correction teaches: in a regime where ranks reorder, the right
bookkeeping of "which modes are fit" is the continuously transported label, i.e. the chain, and the sorted spectrum alone
is insufficient — the same lesson as Cor. 43 and section 18, now at the level of the bound's input.

## 20. Consequences for the held-rough section of the paper (reconciliation, 2026-10-12)

Mapping of the existing section (fixed projector seminorms P, sigma_R^2 = lambda_max(PKP), H_2, Props rough-energy /
a-priori floor / cutoff-floor, Assumption "efficient roughness supply", Prop two-sided band, Prop reduced ignition,
Prop 20) onto the chain:
 (i) Prop rough-energy (a)-(c) = Theorem 1 projected on a FIXED P; its leak is the static off-diagonal block. Theorem 48 is
     the same identity with the projector TRANSPORTED along the eigenvector flow, where the leak becomes the cross-front
     rotation. The fixed-P statements remain true and are the A = 0 reading.
 (ii) Prop 15(b) (rate of sigma_R, Theta_R <= sigma_R lambda_1^{1/2} H_2 ||E||) = Prop. 5 (Hellmann-Feynman lift bound) with
     constants; the a-priori floor's Sigma(s) = sigma_0 + a s is its worst-case integral. Theorem 13 replaces the worst case by a
     two-sided TRAJECTORY bracket on every eigenvalue of PKP in terms of the accumulated feature change on P's range.
 (iii) Assumption "efficient roughness supply" (<E_r,KE_r> >= c sigma_R^2 |E_r|^2 with c of order one) is FALSE along training
     and must be dropped. MEASURED (four runs, Laplacian high-pass above mode 20 and 60; n = 200):
        c = 0.02-0.09 at t = 0 and DECAYS to 0.001-0.005 by t ~ 400-2700, while sigma_R^2 is lifted 6x (H96 d2) to 50x
        (H128 d6, H256 d8) and |E_r|/|E| -> 0.95-1.00.
     Mechanism (chain): the lift is targeted at the smoothed residual and raises the TOP of the rough subspace by the gain
     law, while the residual itself moves to the slowest directions of that subspace (the front, lambda_eff t ~ 1 in the
     CURRENT basis); so c(t) ~ lambda_eff(t)/sigma_R(t)^2 -> 0. The correct closure is the front law in the moving frame
     (section 18-19), not a constant fraction of sigma_R^2. Prop two-sided band inherits the error; Theorem 48 is its
     replacement, with the leakage as the named remainder instead of c.
 (iv) The "held rough error" in the Laplacian sense is not held: lambda_eff t on the Laplacian-rough residual grows to 5-250
     (the kernel has lifted the Laplacian-rough directions and fits them), and the whole residual becomes Laplacian-rough
     (|E_r|/|E| -> 1) while decaying. What is held is the residual's energy on the modes still unfit in the CURRENT kernel
     basis (Theorem 48, transported cutoff). Roughness must be measured against the lifted kernel, i.e. in the moving frame;
     the fixed seminorm is a valid floor whose content shrinks as sigma_R grows.
 (v) Definition cutoff (from the a-priori growth bound I_m(t)) -> the transported cutoff of Theorem 48 (from the actual lifted
     spectrum, continuity-labelled). Prop cutoff-floor remains a valid a-priori inequality; the chain version is two-sided.
 (vi) Prop 20 (analyticity radius about theta_0) = Theorem 27 at one expansion point; Theorem 28 (the chain) is its continuation.
 (vii) Prop reduced ignition -> Theorem 26 (richness expansion; onset of the lift as a convergent series) + Cor. 49 (the
     crossing condition and its bracket); the reduced two-variable model is subsumed and should be cut.
Net: keep Part I and the exact fixed-projector floor; replace the closure, the band, the ignition model and Prop 20's role
by the chain section; the paper's "closure beyond the lazy regime" becomes the chain, and its one measured hypothesis
(constant c) is replaced by a theorem plus a named remainder.
