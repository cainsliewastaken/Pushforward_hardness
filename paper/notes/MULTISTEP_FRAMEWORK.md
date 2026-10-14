# A theory of multistep training: why it helps, when it stops, how it relates to roughness
(started 2026-10-14; separate paper for now, to be folded into the roughness paper if it stays thin)

Scope decisions (user, 2026-10-14). Separate paper. Updates discrete and generic where possible:
theta_{s+1} = theta_s + Pi^{1/2} d_s for an arbitrary rule d_s. Gradient descent with a frozen kernel where an operator is
needed. Gradient flow only where a statement needs the moving kernel (the chain's Thm 46/48, Cor 49).
Builds on MULTISTEP_CHAIN.md (notation sections 0-2, Theorems M1-M11), which stays as the record of the first pass.
Labels: THEOREM / PROPOSITION / COROLLARY (proved, hypotheses stated), MODEL, CHECK, CONSISTENT, PREDICTION, CONJECTURE.

## 0. The organising statement

The roughness theory and multistep training are about one object, the one-step error eps, measured in two metrics.
 - Roughness theory: how much of eps must remain after training with the current features, and where it sits (the
   slow part for the current kernel K; the front).
 - The rollout target weighs eps by the propagation Gram M_v = L^T Wbar_v L: by how much the dynamics, through the
   learned propagators, accumulates and amplifies each pattern of one-step errors along a trajectory.
 - One-step training removes eps in the order of K and is blind to M_v. Unrolled training removes it in the order of
   A(w) = K^{1/2} M_w K^{1/2} (first order in eps; MULTISTEP_CHAIN section 2).
 - No rule beats the roughness floor on eps (the draft's Prop "The floor holds during multistep training"). A rule
   decides only WHICH part of the unavoidable error remains. Unrolling pays when it can keep the remainder in patterns
   the dynamics damps; Theorem B2 turns this into a lower bound valid for every rule.
So: multistep training helps when the held rough error of one-step training is fast in the rollout metric (B1). It
stops helping when the rollout metric itself becomes the problem:
 - ill-conditioned (C1);
 - nonlinear (C2);
 - sample-specific (C3).
Section D states the common form: a roughness theory whose metric is a parameter.

Discrete exact layer used throughout. For any rule and any sampled function F(theta) (one-step errors at the data,
rollout errors, operator errors), F(theta_{s+1}) - F(theta_s) = Dbar_s d_s exactly, with
Dbar_s = int_0^1 d_theta F(theta_s + t Pi^{1/2} d_s) Pi^{1/2} dt (the step-averaged features of the roughness paper). For a
scalar target V, V_{s+1} - V_s = <gbar_{V,s}, d_s> with the step-averaged gradient. No expansion is involved.

## B. Why unrolling helps

Setting for B1 (first order in eps; K and L frozen, i.e. lazy; gradient descent on each rule's own loss at learning rate
c / lambda_max of that loss's Gauss-Newton curvature, 0 < c <= 1/2). In kernel-whitened coordinates x = K^{-1/2} E (E = -eps at
the training states) every rule is x_{s+1} = (I - c Ahat) x_s, with Ahat = A / lambda_max(A): A = K for one-step training,
A = A(w) for window w. The target is V_v = (1/2n) x^T A(v) x = (1/2n) E^T M_v E.

THEOREM B1 (no anisotropy, no gain; anisotropy aligned with the held error, gain).
(a) If the propagation Gram is uniform to relative accuracy delta < 1, ||M_w - mbar I|| <= delta mbar, then for every S
        ||x_S^(w) - x_S^(one)||  <=  (2 c S delta / (1 - delta)) ||x_0||,
    and |sqrt(2n V_v^(w)(S)) - sqrt(2n V_v^(one)(S))| <= ||A(v)||^{1/2} times the same. With normalised learning rates,
    unrolling cannot differ from one-step training by more than O(c S delta): any gain needs ANISOTROPY of the
    propagation Gram over the residual.
(b) Without any commutation assumption, let u = {j : c S lambda_j / lambda_1 < 1} be the one-step slow set (cutoff in K),
    U its spectral subspace, E_{0,u} and E_{0,f} the parts of E_0 on U and on its complement, and
    mu_U = lambda_min(Pi_U M_v Pi_U), the SMALLEST amplification of any pattern in the one-step slow subspace. Then
        sqrt(2n V_v^(one)(S))  >=  e^{-2} mu_U^{1/2} ||E_{0,u}||  -  e^{-1} ||M_v||^{1/2} ||E_{0,f}||,
    while training on the target itself (w = v) gives (MULTISTEP_CHAIN Thm M5)
        2n V_v^(v)(S)  <=  U_v + e^{-2} F_v,
    with U_v and F_v the energies of A(v)^{1/2} x_0 on the slow and fit parts of A(v)'s normalised spectrum. Unrolling on
    the target is therefore better whenever the right side of the first inequality, squared, exceeds U_v + e^{-2} F_v.
    In words: whenever the dynamics amplifies EVERY pattern one-step training leaves slow (mu_U large), and the
    target-weighted error is mostly fast for A(v).
Proof. (a) lambda_max(A(w)) = lambda_max(K^{1/2} M_w K^{1/2}) lies in [(1 - delta) mbar lambda_1, (1 + delta) mbar lambda_1]. Writing
Ahat(w) - Khat = K^{1/2}(M_w - mbar I)K^{1/2} / lambda_max(A) + K (mbar / lambda_max(A) - 1 / lambda_1) gives
||Ahat(w) - Khat|| <= 2 delta / (1 - delta). Both iteration matrices are symmetric with spectrum in [1 - c, 1], so
||B_w^S - B_1^S|| <= S ||B_w - B_1||; the target bound follows from ||A(v)^{1/2} y|| <= ||A(v)||^{1/2} ||y||.
(b) E_S = (I - c Khat)^S E_0. Each mode in u keeps at least e^{-2} of its amplitude, since (1 - y)^S >= e^{-2 S y} for y <= 1/2.
Each mode outside u keeps at most e^{-1}. E_{S,u} lies in U, so ||M_v^{1/2} E_{S,u}|| >= mu_U^{1/2} ||E_{S,u}||; finish with the
triangle inequality.  []
Two kinds of anisotropy feed (b), and both are present in the check below.
 - DYNAMICAL anisotropy. The finite-time growth rates differ between directions (Lyapunov spectrum).
 - TEMPORAL anisotropy. Even with no growth, L sums the injections along a trajectory. Temporally coherent error
   sequences are amplified about (number of steps)^2 times more than cancelling ones. The one-step error is a smooth
   function of state, sampled at spacing h along a trajectory. So its held part is temporally coherent at every scale
   coarser than the distance moved in one step, and that is the pattern the summation amplifies most.
CHECK (2026-10-14, Lorenz-63 true tangent propagators, h = 0.04, 40 starts on the attractor, median over starts):
lambda_2 / lambda_1 of the uniform-window propagation Gram M_w is 0.49 / 0.41 / 0.11 / 0.071 / 0.022 at windows 0.16 /
0.32 / 0.64 / 1.28 / 2.56. So M_w is strongly anisotropic already at the shortest window (temporal), and more so at long
windows (dynamical).

THEOREM B2 (the rollout floor for every training rule: roughness floor times amplification). At first order in eps
(V_v = (1/2n) E^T M_v E), for ANY update rule d_s, any S and any test vector phi on the training states,
        sqrt(2n V_v(S))  >=  [ <phi, E_0> - cost_S(phi) ]_+  /  ||M_v^{-1/2} phi||,      cost_S(phi) = sum_{s<S} ||Phibar_s^T phi|| |d_s|,
with Phibar_s the step-averaged value features. The supremum over phi is the roughness theory's signed floor (main text:
Thm path, discrete form; the Krylov functional is the optimal phi at first order in the step), converted to the rollout
target by the factor 1 / ||M_v^{-1/2} phi||. That factor is large when the dynamics amplifies the floor's direction and
small when it damps it.
Proof. Exactly, <phi, E_{s+1} - E_s> = <Phibar_s^T phi, d_s>, so <phi, E_S> >= <phi, E_0> - cost_S(phi). By Cauchy-Schwarz,
<phi, E_S> = <M_v^{-1/2} phi, M_v^{1/2} E_S> <= ||M_v^{-1/2} phi|| sqrt(2n V_v(S)).  []
Reading. This is the precise sense in which multistep training "chooses where the unavoidable error goes".
 - No rule can do better than the worst-amplified direction that the path cannot yet pay to remove.
 - One-step training leaves its remainder wherever K is slow; B1(b) measures how far that is from this floor.
 - The floor is the same for every rule. The rules differ only in how close they come to it.
Status: B1 and B2 are proved at first order in eps; B1 also needs frozen K and L. B2 holds for any rule, any path, with no
frozen-kernel assumption.

Open in B. (i) A NECESSARY-and-sufficient gain statement without commutation: B1(a) and (b) bracket it from the two
sides. The candidate quantity is the commutation defect ||[K, M_v]|| on the slow subspace. (ii) B1 with the kernel
moving: Thm 48 in z coordinates (MULTISTEP_CHAIN 2(iii)) for w = v; nothing yet for one-step versus w. (iii) How much of
the gain is the temporal and how much the dynamical anisotropy: measurable by replacing L with the step-summation
operator (no growth) in M_v.

## C. When unrolling stops helping: three horizons

### C1. Conditioning (proved under a dominated-splitting hypothesis)

Hypothesis H (rank-one dominated propagation along the training trajectories). There are Lambda_1 and gamma > 0,
Lambda_2 = Lambda_1 - gamma, unit vectors xi_q and vectors eta_i (|eta_i| <= C_eta), and C_R, such that for all i < q
        P(t_q, t_i) = e^{Lambda_1 (t_q - t_i)} xi_q eta_i^T + R_{q,i},     ||R_{q,i}|| <= C_R e^{Lambda_2 (t_q - t_i)}.
For the learned propagators this follows from the same property of the true tangent flow plus MULTISTEP_CHAIN's
learned-versus-true bound (the draft's Prop "Learned and true propagators"), when the Jacobian errors are small
against the gap gamma.
CHECK (true L63 tangent flow, as above): sigma_2 / sigma_1 of the single-horizon block L_p, median over 40 starts, is
0.65 / 0.64 / 0.26 / 0.21 / 0.10 at tau = 0.16 / 0.32 / 0.64 / 1.28 / 2.56, against e^{-0.9 tau} = 0.87 / 0.75 / 0.56 /
0.32 / 0.10. Rank-one dominance holds at long windows, with gamma near the asymptotic gap Lambda_1 - Lambda_2 = 0.9. At
short windows the second direction is not suppressed (constants C_R of order one).
Define, per start, the LEADING ADJOINT PATTERN of horizon q (new name): a^(q)_i = e^{-Lambda_1 t_i} eta_i for i < q, 0 otherwise.
Then <a^(q), u> = sum_{i<q} e^{-Lambda_1 t_i} eta_i^T u_i is the error sequence u projected on the unstable adjoint direction,
discounted so that EARLY errors count most. Let Acal_w be the span of the a^(q) over the horizons with w_q > 0 (all starts),
sigma_2(q)^2 = sum_{i<q} e^{2 Lambda_2 (t_q - t_i)}, s_w = sum_q w_q sigma_2(q)^2, and
G = Acal^T K Acal, the n x n kernel between the leading adjoint patterns of horizon p of different starts, with largest
eigenvalue g_max.

THEOREM C1 (collapse of the normalised curvature onto the leading adjoint patterns). Under H, for any window weights
w with w_p > 0 and any direction x (kernel-whitened coordinates) with K^{1/2} x orthogonal to Acal_w,
        x^T Ahat(w) x  <=  C_R^2 s_w lambda_1(K) ||x||^2  /  ( w_p ( e^{2 Lambda_1 t_p} g_max / 2 - C_R^2 sigma_2(p)^2 lambda_1(K) ) )_+ .
For uniform weights the right side is of order e^{-2 gamma tau} times factors polynomial in tau/h, tau = t_p. Under gradient
descent at a stable learning rate, every component of the error outside the leading adjoint patterns therefore trains
at a relative rate that falls like e^{-2 gamma tau}. As tau grows, a long window trains essentially n numbers, one per start
(the discounted projection of that start's early errors on its unstable adjoint direction), and their joint
conditioning is that of the n x n matrix G.
Proof. Per start, L_q u = e^{Lambda_1 t_q} xi_q <a^(q), u> + sum_{i<q} R_{q,i} u_i, and by Cauchy-Schwarz
|sum_{i<q} R_{q,i} u_i| <= C_R sigma_2(q) |u|. For u orthogonal to every a^(q) with w_q > 0 the first term vanishes, so
u^T M_w u = sum_q w_q |L_q u|^2 <= C_R^2 s_w |u|^2. Summing over starts with u = K^{1/2} x gives the numerator, using
||K^{1/2} x||^2 <= lambda_1(K) ||x||^2. For the denominator, take a unit y whose image K^{1/2} y attains
max sum_k <a_k^(p), (K^{1/2} y)_k>^2 = g_max. Keeping only horizon p and using (alpha - beta)^2 >= alpha^2/2 - beta^2 gives
lambda_max(A(w)) >= y^T A(w) y >= w_p (e^{2 Lambda_1 t_p} g_max / 2 - C_R^2 sigma_2(p)^2 lambda_1(K)).  []
COROLLARY C1' (long-window limit of the curvature's shape). For a single horizon,
tr A / lambda_max(A) -> tr G / lambda_max(G) as tau -> infinity, up to O(e^{-2 gamma tau}) corrections. PREDICTION: the measured
near-rank-one rollout kernel (tr K^(p) / lambda_max = 1.1-1.5 at T >= 0.64, results_main plot 7) is the statement that the
leading adjoint patterns of different starts are nearly collinear for the network's features. G is computable from the
one-step kernel and the adjoint Lyapunov vectors, so this is testable.
What C1 says about WHEN unrolling stops helping.
 - The binding quantity is gamma, the gap between the first and second finite-time growth rates along the data, not
   Lambda_1. Past the crossover window of MULTISTEP_CHAIN M7, every target component outside the leading adjoint
   patterns slows by e^{-2 gamma (tau - tau_x)} relative to the best window.
 - For Lorenz-63, gamma is about 0.9. Conditioning alone then costs about e^{-1.8} in rate per extra time unit of window.
   That is a slow loss: it explains why curvature-scaled learning rates made the 1.28 window trainable and best
   (follow-up 6), and predicts that conditioning is NOT what limits L63 windows near 1.5.
 - The fixed-learning-rate failures are a different, cruder effect: the absolute curvature grows like e^{2 Lambda_1 tau}
   (measured 2.6e3 -> 4e7 from 0.32 to 1.28), and C1's normalisation removes it.
 - For systems with a small gap gamma (many near-neutral directions, e.g. turbulence), C1 predicts the opposite: the
   collapse is slow, and conditioning lets windows stay useful longer.

### C2. Concentration and nonlinearity (proved under the stated hypotheses, 2026-10-14)
Safe side (proved in the draft, Cor "Forward Euler, uniform constants"): the higher orders stay below the first-order
envelope up to tau_nl = (1/Lambda) ln( Lambda min(r/4, 1/kappa_r) / (C etabar e^{Lambda h}) ), logarithmic in the inverse
one-step error.
Breaking side (written 2026-10-14). The draft's horizon uses constants uniform over starts, i.e. the WORST start.
What breaks first is set by how the growth is distributed over starts, and two effects have to be separated:
 - concentration: the loss and gradient end up carried by a few fast-growing starts. This happens already at first
   order.
 - nonlinearity: those starts leave the analytic radius.

Per-start growth. For start k and window tau = t_p, let G_k(tau) >= 0 be a per-start growth factor such that
G_k(tau) <= theta implies start k is admissible on [0, tau]. One choice is the draft's recursion of Thm "The higher-order
part of the error" run with per-start propagator norms. Its first-order envelope is
etabar h sum_{i<q} ||P_k(t_q, t_i)||, with threshold theta = r/4 / etabar up to the draft's constant kappa_r.
Starts are drawn i.i.d. from the data distribution. The FINITE-TIME GENERALISED LYAPUNOV EXPONENTS are
        L_tau(q) = (1/tau) ln E[ G(tau)^q ],
and the INTERMITTENCY RATE is delta_tau = L_tau(4) - 2 L_tau(2).

THEOREM C2a (the higher-order part of the gradient splits by admissibility; exact). Let A(tau) be the admissible starts
and b_A the draft's bound (its Thm "Bound on g_w - grad^(1)_w") summed over A(tau) only. Then
        |g_w - grad^(1)_w|  <=  b_A  +  (1/n) | sum_{k not in A(tau)} ( g_{w,k} - grad^(1)_{w,k} ) |.
The first term is controlled to all orders. The second has no first-order control: for those starts the series in the
one-step errors need not converge. Removing the non-admissible starts restores the draft's bound exactly.
Proof. The gradient is an average over starts; apply the draft's per-start bound on A(tau).  []

PROPOSITION C2b (concentration of the window loss and gradient; first order in eps). Assume the per-start loss
factorises as l_k = G_k^2 X_k and the first-order per-start gradient norm as |grad^(1)_{w,k}| = G_k^2 Y_k, with X_k, Y_k > 0
independent of G_k and with finite second moments. This holds under C1's hypothesis H up to its remainder: X_k is the
squared projection <a_k, eps_k>^2, and Y_k carries the feature factor. Then, almost surely as n -> infinity,
        n_eff^loss / n  =  (sum_k l_k)^2 / (n sum_k l_k^2)  ->  e^{-tau delta_tau} (E X)^2 / E[X^2],
and the same with Y for the gradient. Moreover delta_tau >= 0, with equality iff G(tau) is almost surely constant over
starts. The loss-carrying share of the starts therefore decays exponentially in the window at the intermittency rate.
Define the CONCENTRATION HORIZON of a batch of n_b starts by n_b e^{-tau delta_tau} = 1:
        tau_conc = ln(n_b) / delta_tau.
Beyond it, the batch loss and the batch gradient are each carried by O(1) starts. The minibatch gradient is then a
few-sample estimate whose relative variance is of order 1/n_eff, and the population loss is a tail statistic.
Proof. The ratio of sample means converges by the strong law (finite E l^2). Independence factorises the moments,
giving (E G^2)^2 / E G^4 = e^{-tau delta_tau}. q -> ln E G^q is convex (Hoelder) and 0 at q = 0, so
2 ln E G^2 <= ln E G^0 + ln E G^4 = ln E G^4. Strict convexity holds unless G is degenerate.  []
For log-normal growth, ln G ~ N(tau Lambda, D tau), one has L_tau(q) = q Lambda + q^2 D / 2 and delta = 4D.

THEOREM C2c (when the first start leaves the radius; rigorous bound). For every q > 0,
        P( G(tau) > theta )  <=  exp( tau L_tau(q) - q ln theta )                                 (Markov),
and over a batch of n_b starts, P(some start is not admissible on [0, tau]) <= n_b inf_{q>0} exp( tau L_tau(q) - q ln theta ).
So all starts of the batch are admissible, and the draft's all-orders bounds hold for the whole batch, with probability
at least 1 - beta, up to the window tau_nl(n_b, beta) solving
        inf_q [ tau L_tau(q) - q ln theta ] = ln(beta / n_b),
with ln theta = ln(r / etabar) + const.
For log-normal growth this gives tau_nl ~ ln(theta)/Lambda - sqrt(2 D ln(theta) ln(n_b/beta)) / Lambda^{3/2} at leading order.
The typical start leaves at ln(theta)/Lambda. The first start of a batch leaves earlier by a term growing like
sqrt(ln n_b): bigger batches and more intermittent dynamics break earlier.
Proof. Markov's inequality applied to G^q, then the union bound.  []

PROPOSITION C2d (the starts that leave carry more than their share of the gradient). Under the factorisation of C2b,
the share of the first-order gradient norm carried by non-admissible starts satisfies
        E[ G^2 Y 1{G > theta} ] / E[ G^2 Y ]  =  P_2( G > theta )  >=  P( G > theta ).
Here P_2 is the growth-tilted law, dP_2 = G^2 dP / E G^2. For log-normal growth the tilt shifts the mean of ln G by
2 D tau. The gradient-weighted share therefore reaches 1/2 at
        tau_nl^grad = ln(theta) / (Lambda + 2D),
much earlier than the typical exit ln(theta)/Lambda when D is comparable to Lambda.
Proof. Y is independent of G and cancels. G^2 and 1{G > theta} are both nondecreasing in G, so by Chebyshev's
association inequality E[G^2 1{G > theta}] >= E[G^2] P(G > theta).  []

What C2 says about WHEN unrolling stops helping. Three horizons follow from the growth statistics, in order:
 1. Concentration, tau_conc = ln(n_b)/delta: the batch is carried by a few starts. This happens at first order.
 2. Gradient-weighted exit, ln(theta)/(Lambda + 2D): most of the gradient comes from starts outside the radius, and the
    draft's step series and quadratic model lose control.
 3. Typical exit, ln(theta)/Lambda: the median start goes nonlinear.
The first two are where the window loss stops being a good proxy for the typical rollout error, which is what
"better" was about. Both grow only logarithmically: (1) in the batch size, (2) in the inverse one-step error.
CHECK (2026-10-14, true Lorenz-63 tangent flow, RK4 step 0.004, 20,000 starts after 20 time units on the attractor;
G = ||P(tau, 0)||, growth of a single injection at the start of the window, as a proxy for the per-start envelope):
   tau          0.16   0.32   0.64   1.28    1.92    2.56
   mean FTLE    3.70   3.17   2.35   1.67    1.41    1.27
   D = Var(ln G)/tau  1.39   1.87   1.85   1.03   0.69   0.54
   delta_tau    4.3    6.1    7.6    7.2     (4.2)   (3.2)
   RMS/median of G  1.24  2.31  9.52  62.3  (43)    (36)
   n_eff/n      0.50   0.14   7.7e-3 1.0e-4 (3e-4)  (3e-4)
Values in parentheses are limited by the sample. Beyond tau ~ 1.3, E G^4 is set by the largest of the 20,000 starts,
so n_eff/n saturates near a few / 20,000.
With a batch of n_b = 1024 starts, tau_conc ~ ln(1024)/7 ~ 1.0. About 8 starts carry the batch at tau = 0.64, and about one
at 1.28. Growth at short windows is far from its asymptotic rate (mean FTLE 3.2 at 0.32 against Lambda_1 = 0.9), so the
asymptotic exponents would badly underestimate these effects. Finite-time quantities are the relevant ones.
CONSISTENT (draft, section "Lorenz-63 measurements"; results_main):
 - (i) The RMS rollout error over starts is 3-5 times the median at horizon 0.64 and about 20 times at 1.28, at every
   width. The growth-factor ratio above is 9.5 and 62. It is larger than the error ratio, as expected: the error sums
   injections at many times, which averages the growth. The rise with horizon is the same.
 - (ii) Removing the 1 % of starts with the largest first-order error (10 of 1024) brings the higher-order part of the
   gradient below 0.2, and removing 5 % brings it below 0.05, up to window 2.48. This is C2a combined with C2d: those
   starts carry a share of the gradient far above their number.
 - (iii) Ranked by RMS error, windows 1.28-2.56 are best at their own horizons; ranked by median or 90th percentile,
   short windows (0.32-0.64) are best at every horizon up to 2.56. This is C2b: past tau_conc the window loss is a tail
   statistic, so training on it improves the few fastest-growing starts and not the typical one. "Better" depends on
   the statistic exactly as predicted.
PREDICTIONS:
 - (a) The median-ranked best window depends on the batch only through ln(n_b)/delta, and shifts later for larger batches
   only logarithmically.
 - (b) Reweighting starts by 1/G_k^2 (normalising each start's rollout loss by its own growth) removes the concentration.
   It should move the median-ranked best window towards the RMS-ranked one, at the price of a loss that no longer
   matches the RMS target.
 - (c) For systems with weaker intermittency (smaller delta) unrolling stays useful longer under the median criterion.
 - (d) Two logarithmic laws bound the useful window: MULTISTEP_CHAIN M7, from roughness and one-step training, and C2c,
   from nonlinearity. Together with C2b and C1 there are four candidate horizons. Which binds first is a property of
   the system and the data, computable from the one-step kernel and the finite-time growth statistics.

### C3. Generalisation (to be proved; statement)
By MULTISTEP_CHAIN Thm M2, the held-out rollout error changes through the one-step kernel between held-out and
training predicted states, weighted by the adjoints on both sides. Under C1's limit both adjoint fields lie along the
leading adjoint patterns.
CONJECTURE C3. The held-out alignment decays once the window exceeds the time over which the projections
<a_k, eps_k> decorrelate between nearby starts. At first order in eps this is a statement about the cross version of
G (held-out patterns against training patterns), and the lazy-transfer / feature-learning-transfer split of THEORY
Cor 44 applies. Measured support: the cosine between the one-step and window gradients falls from 1 to about 0.3 by
T = 0.5 at every dt (results_main plot 3).

## D. Roughness with the metric as a parameter (plan)
Write the roughness paper's floor, step bound, Krylov functional and held rough error for a general metric M (M = I:
one-step; M = M_v: rollout target). B2 is the floor part. What is still needed:
 - the slow part E_gamma and its Krylov certificate in the M metric, i.e. for the kernel K M;
 - the operator-target version (MULTISTEP_CHAIN M9: the untransferable subspace) as the off-data part;
 - the fixed-horizon floor the draft left open, through the roughness of S_tau measured in the leading adjoint
   patterns (C1).

## Paper skeleton (separate paper)
 1. Setting; one-step versus unrolled as two drives through one kernel (M1, M2).
 2. First-order reduction; kernel-whitened coordinates; A(w) linear in the weights (M3).
 3. Why unrolling helps: B1, B2; temporal and dynamical anisotropy.
 4. The best window and weights: MULTISTEP_CHAIN M6 (flatness, LP/SDP), M7 (crossover window, logarithmic law).
 5. When it stops: C1 (conditioning), C2 (nonlinearity), C3 (generalisation).
 6. Relation to roughness: D, with the off-data floor (M8-M10) and Euler versus direct (M11).
 7. Lorenz-63 evidence (existing runs), and the measurements of section E.

## E. Measurements to plan (none run; each needs a yes with account, QOS, GPUs and time)
 1. G and its spectrum on one-step L63 checkpoints; compare tr G / lambda_max(G) with the measured tr K^(p) / lambda_max.
 2. mu_U, U_v and F_v of B1(b) at the start of the existing fine-tunes; compare with the measured gains.
 3. The split of the gain into temporal and dynamical parts (B open (iii)).
 4. For C2: per-start first-order envelopes against the radius estimates, and the window at which the gradient
    concentrates.

## New names (proposed, not yet in THEORY.md)
temporal / dynamical anisotropy, leading adjoint pattern a^(q), the pattern kernel G, the conditioning horizon
(C1), the nonlinear horizon tau_nl (from the draft), the generalisation horizon (C3).
