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

### C3. Generalisation: the sample-size horizons (proved under the stated hypotheses, 2026-10-14)
By MULTISTEP_CHAIN Thm M2 the held-out rollout error changes through the one-step kernel between held-out and training
states, weighted by the adjoints on both sides. At first order in eps, for one step (or one instant of gradient flow),
this becomes a SAMPLING question. The training gradient is an average of i.i.d. per-start gradients whose sizes have
the heavy-tailed law of C2, so whether a step on the training starts lowers the error on the start distribution
depends on how well n starts estimate a tail-dominated mean. There are two separate failure modes, variance and
missing tail, plus a third for the evaluation itself.

Setting. Starts are i.i.d. from the data distribution. Per-start first-order window gradients are g_k, the population
window gradient is gbar_w = E g_k, and the sample gradient is g = (1/n) sum_k g_k. The target is the population held-out
error Vbar_v = E V_{v,k}, with population gradient gbar_v. A step d = -eta g changes Vbar_v by -eta <gbar_v, g> to first order in
the step. Population coherence: rho_w = |gbar_w| / E|g_k| (how much the per-start gradients cancel). Population
alignment: cos theta = <gbar_v, gbar_w> / (|gbar_v| |gbar_w|) (the question of sections B and M6, assumed > 0 here).

THEOREM C3a (variance horizon). For 0 < eta' < 1,
        P( <gbar_v, g> <= (1 - eta') <gbar_v, gbar_w> )  <=  ( E|g_k|^2 / (E|g_k|)^2 )  /  ( n eta'^2 rho_w^2 cos^2 theta ).
Under the factorisation |g_k| = G_k^2 Y_k of C2b, E|g|^2 / (E|g|)^2 = e^{tau delta_tau} E[Y^2] / (E Y)^2. So a step on n training
starts lowers the population target error by at least (1 - eta') of its expected amount, with probability >= 1 - beta,
for windows up to
        tau_var = [ ln n + ln( beta eta'^2 rho_w^2 cos^2 theta (E Y)^2 / E[Y^2] ) ] / delta_tau,
i.e. ln(n)/delta at leading order. This is C2b's concentration horizon with the number of distinct training starts in
place of the batch size.
Proof. E<gbar_v, g> = <gbar_v, gbar_w>. Var<gbar_v, g> = Var_k<gbar_v, g_k>/n <= |gbar_v|^2 E|g_k|^2 / n. Apply Chebyshev and
divide by <gbar_v, gbar_w>^2 = |gbar_v|^2 rho_w^2 (E|g_k|)^2 cos^2 theta.  []

THEOREM C3b (missing-tail horizon). Let m_2 be the median of G under the growth-tilted law P_2 (dP_2 = G^2 dP / E G^2), and
p_tail = P(G >= m_2). Under the factorisation, the starts with G >= m_2 carry at least half of the population mass E[G^2 Y]
(C2d). A training set of n starts contains none of them with probability (1 - p_tail)^n >= 1 - n p_tail. Under a large
deviation principle for ln G / tau (Gaertner-Ellis, with tau L_tau(q) -> the limiting cumulant function),
        p_tail = exp( -tau kappa_tau + o(tau) ),     kappa_tau = 2 L_tau'(2) - L_tau(2) = KL(P_2 || P) / tau
(kappa = 2D for log-normal growth, against delta = 4D). For windows beyond
        tau_bias = ln(n) / kappa,
the training set typically contains none of the starts that carry half of the population gradient. The sample
gradient then estimates only the bulk part of a tail-dominated population gradient. A step can still lower the typical
(median) held-out error, but nothing guarantees it lowers the population RMS error, which is carried by starts the
training set has not seen.
Proof. The share statement is C2d at threshold m_2. The probability is independence plus Bernoulli's inequality. The
exponent is the Legendre transform at the tilted mean, since d/dq ln E G^q at q = 2 is E_{P_2} ln G.  []

COROLLARY C3c (what an empirical RMS can measure). The same argument applies to an evaluation set of |H| held-out
starts. Beyond ln(|H|)/kappa, an empirical RMS rollout error at horizon tau is typically below the population RMS, and it
is carried by O(1) starts. Rankings by RMS at such horizons are few-start statistics.

CHECK (2026-10-14, true L63 tangent flow, 200,000 starts, G = ||P(tau, 0)||; "population" = all 200,000):
   tau                                       0.32    0.64    0.96    1.28    1.92
   delta_tau                                 6.25    8.23    9.20    7.95    6.12
   kappa_tau                                 4.02    6.21    7.63    7.28    5.97
   p_tail = P(G >= tilted median)            5.3e-2  2.5e-3  7.0e-5  1.5e-5  5e-6
   ln(n)/kappa for n = 1024 / 4096 / 131072  1.7/2.1/2.9  1.1/1.3/1.9  0.91/1.09/1.54  0.95/1.14/1.62  1.2/1.4/2.0
   subsample mean of G^2 / population mean (median over 200 draws):
     n = 1024                                0.99    0.97    0.35    0.035   0.003
     n = 4096                                1.00    0.96    0.60    0.12    0.012
     n = 32768                               1.00    1.00    0.80    0.68    0.038
For each n the collapse of the subsample mean begins at the predicted tau_bias:
 - n = 1024: between 0.64 and 0.96;
 - n = 4096: near 1.0-1.1;
 - n = 32768: between 1.28 and 1.92 (predicted about 1.4-1.7).
p_tail falls with tau at about the rate kappa, with sub-exponential prefactors not small at these tau.
Caveats:
 - At tau >= 1.28 the 200,000-start "population" is itself only a few tilted-typical starts deep (p_tail * 200,000 = 3), so
   the true ratios there are smaller still.
 - G is the growth of a single injection at the start of the window. An actual rollout error sums injections over the
   window and has a milder tail, so these horizons are lower bounds for the trained models' horizons.
CONSISTENT (results_main; the multistep runs used 131,072 training starts, fresh batches of 1024, and 4096 held-out
starts):
 - The held-out set of 4096 has ln(4096)/kappa ~ 1.1-1.3. The RMS comparisons at T = 1.28 (e.g. "the 1.28 window is best
   at T >= 0.64") are at or beyond C3c's horizon, so they are few-start statistics. results_main already notes that the
   seed spread is largest at T = 1.28.
 - The 131,072 training starts give tau_bias ~ 1.5-1.6 and tau_var ~ 1.4-1.5. Beyond that, longer windows cannot be expected
   to improve the population RMS error by training on more steps; only more starts (logarithmically) or a less
   tail-dominated loss can. This range coincides with the measured jump of the gradient's higher-order part
   (1.52-1.60) and with the longest window that trained usefully (1.28). The coincidence is noted, not explained.
What remains for C3. rho_w and cos theta are population quantities and decide whether the population window gradient
helps at all; they belong with B and M6. Measured proxy: the cosine between the one-step and window gradients falls
from 1 to about 0.3 by T = 0.5 (results_main plot 3). Under C1, per-start gradients point along each start's leading
adjoint pattern with the sign of <a_k, eps_k>. rho_w is then the coherence of those signed patterns over the attractor:
a computable quantity that has not been computed.

### Summary of section C: the horizons, in the order they bind for Lorenz-63 (batch 1024, 131,072 training starts,
4096 held-out starts; numbers from the checks above, single-injection proxy)
 - C2b batch concentration, ln(n_b)/delta: about 0.8-1.1.
 - C3c evaluation reliability, ln|H|/kappa: about 1.1-1.3.
 - C3a/b training-set sampling, ln(n)/delta and ln(n)/kappa: about 1.3-1.6.
 - C2c/d nonlinearity, gradient-weighted ln(theta)/(Lambda + 2D) then typical ln(theta)/Lambda: depends on the one-step
   error through theta; to be measured on the trained models.
 - C1 conditioning: a slow penalty e^{-1.8 tau} past the crossover, not binding here.
 - MULTISTEP_CHAIN M7 crossover (roughness): where the benefit peaks; to be measured.
Every horizon is logarithmic: in the batch size, the training-set size, the evaluation-set size, or the inverse one-step
error. The intermittency rates delta and kappa of the finite-time growth set the scale of all the sampling horizons.

## D. Roughness with the metric as a parameter (written 2026-10-14)

THEOREM D1 (the roughness theory in any metric). Let M be symmetric positive definite on the space of sampled
functions (N entries), <a, b>_M = a^T M b. Measure the first-order step E + Phi g in ||.||_M. Every statement of the
roughness paper that uses only a linear map from steps to sampled functions and an inner product then holds verbatim
in (R^N, <.,.>_M), with the kernel K replaced by K M:
 - The single-step value F_M(s) = min_{|g| <= s} ||E + Phi g||_M.
 - The slow part E_{M,gamma} = gamma (K M + gamma)^{-1} E and the identity F_M(s_gamma) = ||E_{M,gamma}||_M.
 - The supremum over all roughness functions equals F_M, attained by the rank-one functional u -> <E_{M,gamma}, u>_M,
   i.e. by the vector M E_{M,gamma}.
 - The Krylov certificate in span{E, K M E, (K M)^2 E, ...}.
 - The local-complexity floor, with the norm constant and the peak feature roughness computed in <.,.>_M.
 - The signed path floor (B2 is its rollout-target form).
K M is self-adjoint and PSD for <.,.>_M, with the spectrum of M^{1/2} K M^{1/2}. A degenerate M (PSD, e.g. a single
horizon) gives the same statements on the quotient by ker M.
Proof. z = M^{1/2} E is an isometry from (R^N, <.,.>_M) to Euclidean R^N. It maps the features to M^{1/2} Phi, the kernel
to M^{1/2} K M^{1/2}, and every seminorm R to R o M^{-1/2}. Apply the paper's theorems in z and map back:
M^{-1/2} gamma (M^{1/2} K M^{1/2} + gamma)^{-1} M^{1/2} = gamma (K M + gamma)^{-1}. The paper's statements are finite-dimensional
Hilbert-space statements (the draft's Thm "Each operator error obeys the roughness theory" uses the same fact).  []
Instances:
 - M = I: one-step training.
 - M = M_v: the rollout target at first order in eps (MULTISTEP_CHAIN section 2).
 - M = M_{delta_p}: a single horizon (D3).
 - The operator targets of MULTISTEP_CHAIN M9 (Jacobian and higher): the same theory on tensor-valued sampled functions
   with Frobenius M.
Reading. Under gradient descent on the target's own loss (w = v), E evolves by -K M_v E, so the M-geometry slow part
keeps the paper's meaning, "what training on this loss removes slowly". Under one-step training it is only a spectral
filter, as the draft noted for multistep training in the Euclidean metric. "Rough" is therefore relative to a metric.
Multistep training changes the metric in which the one-step target's roughness is measured, from K to K M_v.

PROPOSITION D2 (metric sandwich; the most one step of unrolling can gain). For M positive definite,
        lambda_min(M)^{1/2} F_I(s)  <=  F_M(s)  <=  lambda_max(M)^{1/2} F_I(s),
and the step that is best in the Euclidean metric, g_I(s), has ||E + Phi g_I(s)||_M <= kappa(M)^{1/2} F_M(s). For one
first-order step of length s, the best step for the rollout target beats the best step for the one-step loss, measured
on the rollout target, by at most sqrt(cond M_v). Together with B1(a): there is no gain without anisotropy of M_v, and
at most sqrt(cond M_v) per step with it.
Proof. lambda_min(M) |y|^2 <= y^T M y <= lambda_max(M) |y|^2 applied to the minimisers.  []
This is loose for long windows (cond M_v grows like p^2 from accumulation and e^{2 Lambda tau} from growth). It says only
that room exists, and how much at most.

PROPOSITION D3 (a local-complexity floor at a fixed horizon). This answers the draft's open item "no local-complexity
floor is known for the error at a fixed horizon". First order in eps, learned propagators frozen. For each start take
the SVD of its horizon block L_{k,p} (d x pd):
        L_{k,p} = G_k xi_k alpha_k^T + R_k,     G_k = sigma_1(L_{k,p}),  |xi_k| = |alpha_k| = 1,  xi_k orthogonal to range(R_k),
||R_k|| = sigma_2(L_{k,p}). Let s_k = alpha_k^T eps_k: the start's one-step error sequence projected on its leading adjoint
pattern. Exactly, by Pythagoras,
        ||e(tau)||_rms^2 = (1/n) sum_k G_k^2 s_k^2 + (1/n) sum_k |R_k eps_k|^2  >=  ||s||^2_{D_G},    D_G = diag(G_k^2)/n.
s is a sampled function of the start with features Phi_s = Acal^T Phi (each start's one-step features projected on its
adjoint pattern) and target Acal^T U (the method's one-step target U projected the same way). D1 with the metric D_G
therefore gives, for any update rule and any S:
 - the signed floor ||e_S(tau)||_rms >= [ <phi, s_0>_{D_G} - sum_{s<S} ||Phibar_{s,s}^T D_G phi|| |d_s| ]_+ for every phi with ||phi||_{D_G} = 1;
 - for every roughness function R (norm constant and peak feature roughness in D_G),
        ||e(tau)||_rms  >=  ||s_{D_G,gamma}||_{D_G}  >=  ( R(Acal^T U) - R(Acal^T c_p N) - sigma_R Upsilon(gamma)^{1/2} ) / C_R.
The fixed-horizon target is the scalar function x_0 -> sum_i alpha_i(x_0)^T U(x_i): the one-step target transported along the
trajectory from x_0 and projected on the leading adjoint pattern. It is measured in the metric that weights each start
by its squared growth.
Proof. SVD; xi_k is orthogonal to range(R_k), so the two parts of e_k(tau) = G_k xi_k s_k + R_k eps_k are orthogonal. s_S - s_0 =
sum Acal^T Phibar_s d_s exactly at first order with Acal frozen. Apply D1 on R^n with metric D_G.  []
No hypothesis H is needed here: the SVD is exact, and H (C1) is only needed to say that the dropped part
|R_k eps_k| is small (sigma_2/sigma_1 = 0.21 at tau = 1.28 and 0.10 at 2.56 on L63, C1's check).

PROPOSITION D4 (growth does not make the fixed-horizon target rougher; it makes the metric intermittent). For
positive weights G_k, the minimum-norm step that fits the fixed-horizon target exactly at first order,
min { |g| : Acal^T Phi g = -s }, does not depend on G. Equivalently the RKHS norm of D_G^{1/2} s under the kernel
D_G^{1/2} K_s D_G^{1/2} equals s^T K_s^{-1} s, with K_s = Acal^T K Acal. What does depend on G is how the target's energy is spread
over the spectrum of D_G^{1/2} K_s D_G^{1/2}, i.e. the slow part at finite gamma and the stable learning rate
(lambda_max grows with max_k G_k^2).
Proof. The constraint is the same linear system with or without the weights.
(D_G^{1/2} s)^T (D_G^{1/2} K_s D_G^{1/2})^{-1} (D_G^{1/2} s) = s^T K_s^{-1} s.  []
CHECK (2026-10-14, true L63 tangent flow, 2,000-4,000 starts on the attractor, top singular vectors and value of P(tau, 0)):
   tau                                              0.16    0.32    0.64    0.96    1.28    1.92
   median |d alpha|/|dx|  (adjoint direction field)     0.027   0.036   0.060   0.063   0.072   0.079
   median |d xi|/|dx|     (forward direction field)     0.056   0.076   0.17    0.27    0.33    0.68
   median |d ln G|/|dx|   (growth field)                0.023   0.044   0.12    0.17    0.28    0.60
The last row is identical at separations 1e-7, 1e-3 and 1e-1, so the field is smooth at those scales with a shrinking
length scale. The adjoint direction field, which builds the projected target, stays smooth (its Lipschitz constant
roughly triples and levels off). The growth field sharpens at about 1.9 per time unit, the known sharpening of
finite-time-Lyapunov ridges.
Earlier conjecture, now refuted: that the long-horizon target becomes rough through the adjoint direction field. It
does not on L63.
What D3-D4 say about the relation to roughness.
 - The projected fixed-horizon target is about as smooth as the one-step target, so the distance the parameters must
   move (D4) does not blow up with the horizon.
   CONSISTENT: check (v) of results_main, where the first-order optimal step removes 70-90 % of the error at window 1.28
   while gradient and Adam steps of the same length increase it. A good nearby step exists; gradient rules cannot take it.
 - What blows up is the metric: its weights G_k^2 are heavy-tailed across starts (the statistical face, C2 and C3) and
   concentrated on thinner and thinner ridges of the growth field (the spatial face, this check). This is the same
   object in two views.
 - Spatially: once the ridge width ~ 1/|grad ln G| falls below the spacing of the training starts on the attractor
   (~ n^{-1/d_A}, d_A ~ 2.06 for L63), the starts no longer resolve where the weight sits. This is the spatial form of C3's
   ln(n)/kappa. PREDICTION: tau_res ~ [ln(n)/d_A + ln(L_A |grad ln G|_0^{-1})] / r_G with r_G ~ 1.9 on L63, compared with
   C3b's horizons from the same n. Not computed.

The relation to roughness theory, in one paragraph. Roughness is relative to a metric (D1). One-step training sees the
one-step target's roughness in K. The rollout target sees it in K M_v, and multistep training on that target removes
error in that order. Unrolling helps when the one-step held rough error, which is slow in K, is fast in K M_v and
amplified by M_v (B1, B2): a pure roughness statement. It stops helping when the rollout metric degrades, not when the
target gets rougher. The projected target stays about as smooth as the one-step target (D3, D4, check). The rollout
metric becomes ill-conditioned (C1), and intermittent across starts both statistically (C2, C3) and spatially (D4's
check). The off-data part (MULTISTEP_CHAIN M8-M10) is the roughness theory applied to the operator targets, the
Jacobian first.

Still open in D.
 - The local-complexity floor of D3 for a fixed seminorm on the start set needs R(Acal^T U) bounded below for a
   concrete R. That requires the regularity of alpha, which is measured (smooth on L63) but not proved.
 - D4's resolution horizon is a prediction.
 - D1 is first order in the step, like the paper.

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
