# Experiments required by the theory review of 2026-10-12

Everything below is to be run at n in the thousands (user decision 2026-10-11: no n = 200 number is quoted). Each item
names the statement it decides, what is computed, what the outcomes mean, and the cost class (login node / Slurm; every
Slurm run needs the user's explicit yes, account m5326 so far). Items are ordered by what they decide, not by cost.
The reference setting unless stated: L63 one-step map, tanh MLP H96 d2 and the depth/width designs of WRITEUP S3,
gradient flow proxy (GD at eta lambda_1 <= 0.5), budget T = 3000, n >= 2000 training points with a held-out set of the same
size. Save per-snapshot: weights, Jacobian on a fixed subsample, kernel eigenpairs, residual. Log git hash, config, seed,
wall time; keep results regenerable from tables.

## A. Decides whether the cutoff bound (Thm 48) has content

A1. The transported bracket with its leakage. (THEORY 19, both CORRECTIONS.) One run with fine stages (the chain's
    per-stage eigendecompositions; stage criterion EPS = 0.01). Track eigenvectors by continuity (maximal overlap between
    consecutive stages; record every avoided crossing between a mode in u and a mode in f). Compute: the tracked unfit
    set u at T; Pi_u c(0); lambda_u(t), lambda_f(t) on the tracked subspaces; rho(t) = ||A_uf||_2 (block spectral norm);
    E_u(t), E_f(t); the leakage integrals; both sides of the bracket against ||E(T)||.
    Outcomes: (i) floor and ceiling within a factor e of ||E(T)|| with leakage a correction -> the bound is a result;
    (ii) leakage comparable to the main terms -> the bound is a tautology in this regime and the moving-cutoff
    formulation (A2) is the one to use. Also record whether the tracked eigenvalues in u are nondecreasing (the
    hypothesis that gives int lambda_u < 1). Cost: the staged model at n ~ 2000 is one 6000 x 6000 eigendecomposition per
    stage, ~1e3-1e4 stages: Slurm, 1 GPU, hours; at n = 200 it is a login-node check of the code only.
A2. Moving-cutoff formulation. Same run: u(t) = {k : lambda_k(t) t < 1} from the current sorted spectrum; P(t) jumps when a
    mode's lambda_k t crosses 1; record the jump terms (residual energy of the crossing mode at its crossing time) and
    the continuous leakage separately. Compare with A1 on the same run: which formulation has the smaller remainder.
A3. Per-stage sign condition as an operator (Cor. 43(ii), Cor. 50, Thm 46(a)). Same run: W = span of the modes carrying
    90 % of the residual energy at each stage (current basis); compute lambda_min(V_W^T (K(t_s) - K(t_{s-1})) V_W) per stage
    and lambda_min of the accumulated increment on W. Outcomes: nonnegative after the transient -> (S) holds per stage on
    the residual's subspace and Thm 46(a) applies; negative on a set of stages -> record where (expected: late, lambda_1
    drift) and whether the negative part is small relative to the lift.
A4. Thm 36 first-order check. Same run: evaluate ||E_lazy(T)||^2 - 2 int <E_lazy(2T-u), Delta(u) E_lazy(u)> du and the
    remainder bound against ||E(T)||^2. Expected: the remainder bound is useless (D large) and the first-order term is
    sign-correct; this is a sanity check on the withdrawn all-orders claim, not a result.

## B. Decides whether the paper's roughness claim survives the moving frame

B1. Rank-to-roughness law F2 at large n, along training, with a held-out sample. For each snapshot and each eigenvector
    v_k(t) of K(t): its graph-Laplacian roughness on the training inputs, and on the held-out inputs via the network's
    own features (the eigenvector evaluated off-sample as (J_val J_tr^T / n) v_k / lambda_k). Plot roughness vs rank for
    K_0 and K(T); test whether the function of rank is the same, and whether it is the same on the held-out sample.
    Outcomes: (i) same law on both samples -> Thm 48 is a roughness statement, F2 enters the paper as a measured law
    with its own section; (ii) law holds on training inputs but the lifted eigenvectors are rougher or sample-specific
    on held-out inputs -> the cutoff bound is a training-error statement about rank only, and the paper's roughness
    claim has to be restricted to the frozen-kernel part. Cost: Slurm, GD runs at n = 2000-20000, the WRITEUP S3 designs.
B2. Derivation attempt (no compute): the lifted directions as images of Gram-smoothed residual directions through a
    bounded number of Hermite orders (THEORY 20(viii)). If B1 gives (i), this is the theorem to write; if (ii), it is
    the explanation of the off-sample roughness.

## C. Decides whether the held-rough section's a priori floor has content

C1. Sigma(t) vs sigma_R(t) (held-rough measurement item (13)). On the saved runs (and the new large-n runs): for the
    manifold high-pass at m = 20, 60, 200: sigma_m(t)^2 = lambda_max(P_m K(t) P_m) per snapshot; the a priori Sigma_m(t)
    with a = lambda_1^{1/2} H_2 ||E_0|| (H_2 from one nested AD at init, and its max over the path); the realised drive
    Theta_R per snapshot. Report the ratio Sigma/sigma along t and the time at which exp(-(2+eps) I_m(t)) drops below 1/2
    against the time at which ||P_m E||^2 actually halves. Outcomes: ratio of order 1-10 -> the floor is informative;
    ratio of order e^10 -> the floor and the cube-root ignition time are vacuous at the times of interest and the
    trajectory bracket (C2) replaces them in the tex. Cost: login node on saved runs (one top eigenvalue per snapshot;
    n = 200 snapshots are available now, and the number decides the shape of the section even before the redo);
    large-n: with the B1 runs.
C2. Trajectory bracket on PKP (THEORY 20(ii)): Theorem 13 applied to v in the range of P, i.e. the accumulated feature
    change D_v on the rough subspace versus l_v, per snapshot; gives the two-sided bracket on sigma_m(t). Same runs as C1.
C3. Measurement items (14)-(17) of the held-rough section at large n: c(t) = <E_r, K E_r>/(sigma_R^2 |E_r|^2); ||E_r||^2 against
    Eq. rough-floor-path; m*(t) against the mode at which the measured roughness spectrum of the error drops;
    sigma_R(theta_0), lambda_1, H_2 vs depth at init.

## D. Chain verifications that the tex quotes and that are n = 200 now

D1. Theorem 13 bracket at T (D_v / l_v and lower bound / actual by rank), Theorem 11 bulk/core at init for tanh, ReLU,
    GELU, softplus, Theorem 11' along the path, the stage table of section 8 (Cross and Q per mode per stage, the sign
    change of Cross at t ~ 0.3-1), the 12.2 richness-expansion check at s = 2-16 to second order, the 15/propagator
    numbers (Phi_max in three directions), the seed spread (two seeds per design; include depth 1 as the control without
    the hidden-state channel, THEORY 6 scope note), Prop. 9b visibility numbers, Prop. 9d retention table, section 13
    validation split signs. All at n >= 2000 with a held-out set. Cost: Slurm, the WRITEUP S3 design set, GPU-hours.
D2. Lemma 15 init-feature sign along the path at large n (section 8 MEASURED table): the sign of the first term of the
    cross term after the transient, per mode; the raw-vs-centred comparison of 12.1.
D3. Euler vs direct from a shared initialisation at Delta in {0.1, 0.25, 0.5, 1} (section 11): E_D(t), E_E(t), the stage
    lifts and their targets; checks Thm 20-22 and the conditional Cor. 23.
D4. Theorem 10 ambiguity made quantitative: train pairs (theta_0, theta_0 + N) with N of the null-space form at a
    hidden-hidden interface, several norms; spread of E(T) over N vs the seed spread.

## E. Adam and preconditioned methods (THEORY 10.4 CORRECTION, section 21)

E1. The dP term. Under Adam (and under KronQ with its refreshes): per stage, v^T Psi dP Psi^T v / n against Cross_P + Q_P on
    the residual-carrying modes; sign and size along training. Outcomes: dP term small after the transient -> the
    tail-lift structure transfers; comparable -> the lift and the metric decay compete and the chain's monotonicity
    statements are not available for Adam.
E2. Cross_s sign per stage under Adam (listed in 10.6, not run): needs the Adam harness to log P.
E3. Energy identity with momentum: log E^T K_P g_bar along an Adam run; fraction of steps with the wrong sign.

## F. Optimizer line (section 21; after the scaling study)

F1. Per-layer vs global basis at equal push, isolated (the measured fact without an explanation): n = 20000, the same
    lr grid, with the residual-space overlap between layers' pushed modes logged (the cap split's cost).
F2. Remainder level: what sets the ~0.1 (exact basis) vs ~0.02 (Kronecker, smooth) optimum; log the per-step Q lift on
    the residual-carrying modes against the remainder along the (q, lr) ridge.
F3. Tuned baselines for the scaling study's claim: AdamW / Muon / SOAP with warmup, cosine schedule, weight decay, lr
    scaled with width, 5-point lr grids at widths 1024 and 4096. Required before any "beats tuned X" statement.
F4. The hypergradient as a stage quantity (21(c)1): per mode, split the one-step loss change into first-order progress
    and the Q-lift part; test whether the learned weights track the first-order part alone.

## Cost summary (to be refined before each request)

Login node now: C1 on the n = 200 saved snapshots (seconds per snapshot; decides the shape of the held-rough section).
Slurm, GPU: A1-A4 (one fine-staged run at n ~ 2000, hours on 1 GPU); B1 + C1-C3 + D1-D2 (the design set at n >= 2000,
GD to T = 3000, tens of GPU-hours in total); D3-D4 (a few GPU-hours); E1-E3 (reuse Adam runs with P logged); F1-F4
(the optimizer harness, each a 20-40 min node job as in REPORT 58-60).
