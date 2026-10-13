# Open problems in training dynamics that the theory can address (screened 2026-10-12)

Sources: Exa searches on 2026-10-12 for stated open problems in training dynamics (sharpening / EoS, plasticity,
forgetting, warmup and schedules, PINN spectral bias, scaling laws, loss spikes, plateaus). For each: the problem as its
authors state it, what the theory gives EXACTLY today (which theorem of THEORY.md / the main text), what would be new,
and the grade: A = a theorem we can write now from existing results; B = a theorem plus one measurement or one of the
open lemmas; C = a reformulation whose predictive content needs the chain computation. Nothing here claims the lazy
comparison (user decision 2026-10-12).

## A. Provable now

### A1. Weight decay lowers the edge-of-stability plateau (Saether, Kolic, Poggio, Beneventano, arXiv 2605.16622)
Stated: weight decay "robustly slows progressive sharpening" and, in MLPs, makes the sharpness stabilise "significantly
below 2/eta"; mechanism identified empirically as "the global alignment of the parameter vector and the sharpness gradient".
Theory: Theorem 55 (period expansion at the edge) already contains a drift term 2 nabla phi_{v1} . delta. With weight decay
the per-step drift is delta = -eta lambda_wd theta, so the per-period change of the top eigenvalue acquires
-2 eta lambda_wd <nabla phi_{v1}, theta>. For a network whose blocks are homogeneous of degree p in theta (ReLU/linear
blocks, no biases), phi_v(c theta) = c^{2p} phi_v(theta), so by Euler's identity <nabla phi_v, theta> = 2p phi_v = 2p lambda_1
along v_1: the drift lowers the sharpness by the FIXED FRACTION 4 p eta lambda_wd per period, independent of the data.
THEOREM to write: at the edge, lambda_1(theta_{k+2}) - lambda_1(theta_k) <= [self-stabilisation term] + 2|S(v1)d|^2 - 4 p eta lambda_wd lambda_1
+ rotation; hence the plateau sits where the lift balances the decay, below 2/eta by a computable margin, and the margin
grows with lambda_wd linearly at first order. Their "alignment of theta with the sharpness gradient" is exactly Euler's
identity. Non-homogeneous blocks (tanh, biases): the same with <nabla phi, theta> measured. Cost: a page; check on
numerics/eos runs with weight decay (A6 of EXPERIMENTS.md).

### A2. Catastrophic forgetting is low-rank beyond the NTK regime (arXiv 2606.18024; ICLR 2026 NTK forgetting paper)
Stated: forgetting under new-task training "concentrates in a small number of old-task NTK eigenmodes"; "extending the
mechanism to ... evolving representations ... remains open".
Theory: Theorem 24 (validation dynamics) with the OLD task as the validation set: E_old' = -K_{old,new}(t) g_new exactly,
with the kernel moving. Theorem 11' (spike theorem, any time, no independence): the coherent part of each stage's feature
change has rank <= n_y r_s, so the change of the cross kernel over a stage, and hence the forgetting increment, has rank
bounded by n_y times the numerical rank of the derivative pattern; summed over stages the forgetting vector lies in a
subspace of dimension <= n_y x (number of effectively distinct stage targets), which the chain measures as the effective
rank of the cumulative change (5-13 here). This is their result with the representation evolving.
THEOREM to write: "Forgetting is low rank along the whole fine-tuning path", rank bound explicit, plus the split of the
forgetting rate into lazy transfer (initial cross kernel) and lifted transfer (Theorem 24's two terms), with the lifted
part the transported stage core. Cost: half a page; it is Theorem 24 + Theorem 11' assembled. Measurement: effective rank
of the forgetting vector along a fine-tuning run (any model; the identity is architecture-free).

### A3. Loss of plasticity as a floor (Spectral Collapse, arXiv 2509.22335; Barriers for Learning, arXiv 2510.00304)
Stated: "there remains no unifying consensus on what drives loss of plasticity"; spectral collapse of the Hessian/NTK at
new-task initialisation precedes the failure.
Theory: at the start of task 2 the network has kernel K_T from task 1. Part I's floor with the Krylov functional on
task 2's labels under K_T, ||Y_2,gamma||_{K_T} (the slow part of the new labels under the inherited kernel), is a LOWER
BOUND on task-2 error after any step of length s (Thm stepsup) and after any path (Thm path): plasticity for task 2 is
exactly the roughness of Y_2 as K_T sees it, one Krylov series to compute. Spectral collapse is the statement that this
slow part is large. The mechanism side: by Theorem 2 the kernel is a sum of (input Gram) (.) (backprop Gram), and by
Theorem 11 the backprop factor carries sigma'(z); saturated or dead units make sigma' -> 0 on the data, which removes both
the lift (core -> 0) and the kernel's bulk on the affected block, exactly. Task-1 training lifts the kernel only on
task-1's drive directions (Theorem 13: lifted where the residual has been), so the inherited kernel is spiky along task 1
and unlifted elsewhere, and task 2's slow part is large unless Y_2 lies in task 1's lifted span.
THEOREM to write: "Plasticity floor": task-2 error after path length ell is at least F_0^{(K_T)}(ell) - 1/2 H_2 ell^2, with F_0
the ridge value of task 2's labels under K_T, and the decomposition of K_T into init + task-1 lift (Theorem 12). Grade A for
the floor and the decomposition; B for "collapse is caused by saturation" (the sigma' -> 0 identity is exact, that it
happens is measured).

### A4. Adam's loss spikes from the preconditioner term (Adaptive Preconditioners Trigger Loss Spikes in Adam, ICML 2026;
Adam stability phase diagram, arXiv 2609.18314)
Stated: loss spikes in Adam are "linked to optimizer dynamics, yet how its two momentum timescales govern them remains
unclear".
Theory: the CORRECTION to Theorem 17: over a stage K_P changes by Cross_P + Q_P + Psi dP Psi^T / n, and the third term is
the preconditioner's own change. Theorem 54's device (variational principle with the current top eigenvector) gives
lambda_1(K_P(t_s)) >= lambda_1(K_P(t_{s-1})) + v_1^T [Cross_P + Q_P] v_1 + v_1^T Psi dP Psi^T v_1 / n: a jump of P on
coordinates where the top mode has features raises the effective sharpness by at least that quadratic form, exactly,
and the step becomes unstable when eta lambda_1(K_P) crosses 2. For Adam, dP > 0 on a coordinate exactly when its second
moment estimate decays faster than the gradient (a quiet coordinate that wakes up), which is the trigger in the ICML paper.
THEOREM to write: "Preconditioner-induced sharpness jump": the lower bound above, with dP written in terms of the
second-moment decay rate beta_2 and the gradient history, giving the condition on (beta_2, eta) for a spike; the momentum
timescale beta_1 enters through the drive. Grade A for the inequality; the phase boundary needs the measured gradient
statistics. Cost: a page.

### A5. Spectral bias in PINNs and neural operators is dynamical (arXiv 2602.19265; Wang-Teng-Perdikaris NTK view)
Stated: "its interaction with optimization dynamics and physics-based loss formulations remains poorly understood";
"the fundamental reasons behind the practical difficulties of training fully-connected PINNs still remain unclear";
second-order methods "substantially alter the spectral learning order".
Theory: every identity holds for any loss with a per-sample gradient (Thm 1: f' = -K grad l), so with the PDE residual
r = N[f] on collocation points as the "output" and Psi_r its Jacobian, the kernel K_r = Psi_r Psi_r^T / n governs the
residual flow, Part I's floor with the Krylov functional applies to the PDE residual, the one-time floor ||r_gamma|| is
the exact first-order reachable PDE residual, the cutoff rate gamma*(t) says which modes of the residual are held, and
Theorem 17 says what a preconditioner does: K_P = Psi_r P Psi_r^T / n, i.e. the "spectral learning order" is the
spectrum of K_P. Spectral bias is the statement that the differential operator makes rough modes of the residual slow
under K_r, which the budget inequality lambda_k R(v_k)^2 <= sigma_R^2 makes precise for R the Krylov functional.
THEOREM to write: the PINN floor and cutoff in Krylov form, with the derivative counting (N of order m multiplies the
roughness of the output's rough modes by the m-th power of their wavenumber) entering through R(Y) for the residual; the
preconditioned version for GN / spectral-aware losses. Grade A as a transfer of proved statements; what is new is the
object (the PDE-residual kernel) and the computable floor; the main-text theorems carry over verbatim.

## B. A theorem plus one measurement or one open lemma

### B1. Why warmup (arXiv 2510.03164: "theoretical foundations remain poorly understood"; Sun 2026: early sharp-to-flat phase)
Theory: Theorem 54's first-order term at t = 0 along the residual is -(2 alpha / n^2) <E, N''[g, g]>, exactly, and the
measured positivity of <E, f''[g,g]> for raw residuals is the sharpness-reduction phase: lambda_1 falls while the network
unlearns its initial output. Warmup keeps eta lambda_1(t) below 2 while lambda_1 is still high; the length of warmup needed
is the duration of the reduction phase, which the identity ties to ||f_0|| (the raw-minus-centred difference).
PREDICTION (new, testable): centred initialisation (f_0 subtracted) removes the reduction phase (measured here: the
cross term loses its systematic sign) and should remove the need for warmup at the same peak learning rate. Grade B: the
identity is exact, the prediction needs the centred-vs-raw warmup experiment (a few GPU-minutes at small scale).

### B2. Plateaus and saddle-to-saddle escape in deep nonlinear networks (arXiv 2605.01288: "exact theory of modewise
escape times ... remains open"; ICLR 2026 simplicity-bias paper)
Theory: a plateau is the held error of Part I (the floor at the current state), and an escape is the cutoff crossing of
Cor. 49: a direction leaves the unfit set when lambda_v(t) t = 1, with the time bracketed two-sidedly by the accumulated
feature change (Theorem 13), and the onset of the lift given by the richness expansion inside its radius (Theorem 26).
Their escape-time law tau* ~ eps^{-(r-2)} for small init is the inversion of that bracket in a regime where D_v is
explicit. Grade B: the bracket is a theorem; a closed escape time needs the chain computation of D_v (open), and the
"simple first" ordering is the rank ordering of the current kernel, which is Part I's budget inequality.

### B3. Neural scaling laws in training time (Spectral Reach, arXiv 2605.31244: "including feature learning and the
nature of the power-law into the picture remains an open question"; optimizer dependence, arXiv 2605.29387)
Theory: the loss at time t is at least the energy of the labels above the cutoff rate (Prop. cutoff-floor), and with a
power-law roughness spectrum rho(gamma) ~ gamma^beta the a priori cutoff gives a power law in t whose exponent is set by
beta and the path-length growth; the lift (feature learning) moves the cutoff by the gain law (measured exponent p), and
the optimizer enters through K_P (Theorem 17). Grade C for the exponent (the a priori floor is vacuous past the first unit
of path length, C1; the front law from the chain is the open derivation), A for the statement that the exponent is a
property of the labels' roughness spectrum in the LIFTED kernel's coordinates, which is what "feature learning
corrections to kernel scaling" means here.

## C. Reformulations (worth a remark, not a section)

- Data ordering / curriculum: by Theorem 12 the kernel at T is the squared accumulated feature change along the drive
  history, so two orders of the same data with the same final residual have different kernels; the stage targets are
  built from the residual in the order it is driven. Exact, but predictive only through the chain.
- Spikes and mechanistic circuits (arXiv 2603.28964, "spectral edge"): their rolling-window Gram gap dynamics are the
  per-stage spectrum of our stage increments; the "k* <= 3 active modes" is the rank bound n_y of Theorem 11'.
- Golden-path hypothesis (SGD does not change which features are learned): Cor. 24a makes SGD the validation identity
  with the batch as training set; the question becomes whether the batch drives share the full-set drive's direction.

## What to do first
A1 (weight decay at the edge) and A4 (Adam spikes) are one-page theorems that reuse Theorem 54/55; A2 (low-rank
forgetting) and A3 (plasticity floor) are assembled from Theorems 24, 11' and Part I and speak to a large audience; A5
(PINNs) is the transfer with the most direct connection to the paper's scientific-ML readership and to the time-stepping
part. B1 is the cheapest new prediction to test. None of these needs the open lemmas (per-stage sign, leakage, radius).
