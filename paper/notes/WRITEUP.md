# Predicting late-stage training from initialization: theory, model, assumptions, evidence

> **NOTE (2026-10-11, user decision): every experiment in this document was run with n = 200 training points
> (exceptions: four n = 400/800 ground-truth designs used only for the gain-exponent check). At n = 200 the training set
> is nearly fit within the budget, the residual-adapted late phase is short, the depth-6-8 "floor" is a sample-size
> artefact, and the negative lazy-transfer sign in the validation split is overfitting of a 200-point sample. This is a
> toy version of a toy problem. ALL EXPERIMENTS MUST BE REDONE at a sample size that leaves a genuine late stage
> (n in the thousands at least), before any measured number here is quoted. The theorems carry n as a symbol and are
> unaffected; the verifications, bulk ratios, exponents, signs, transfer fractions and the ranker's fits are all
> n = 200 statements until redone. The exact staged model does not scale to large n (eigendecomposition per stage);
> the ground-truth GD harness does, so the redo is training-only.**


Status document, 2026-10-10. Code: `$SCRATCH/l63_held_rough/code/`; runs: `$SCRATCH/l63_held_rough/runs/`;
per-step log of the work: `REPORT.md` (sections 0-30). This file is the consolidated statement.

## 1. Problem and setting

Goal: for a network design (width, depth, parametrisation; later optimizer, loss, architecture) and a dataset,
predict from initialization how good the network will be at a training budget T, in the feature-learning regime,
accurately enough to rank designs and to give the training error to a modest factor. Validation error is out of scope
by decision; the theory is a theory of the training error.

Testbed: Lorenz-63 one-step map x -> flow(x, Delta), n = 200 training points on the attractor (standardised), targets
of increasing roughness Delta in {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}; tanh MLPs, 3 outputs, widths 32-512, depths 1-4,
output scale s (lazy for large s, rich for small s); full-batch gradient descent with step <= 0.5/lam1 as a gradient-flow
proxy; squared loss; budget T = 3000 (kernel time ~10^5). Ground truth: ~190 trained runs with full spectra logged.

Notation: N = 3n stacked outputs; Psi (N x d) Jacobian; K = Psi Psi^T / n the kernel on the data; E = Yhat - Y;
eigenpairs (lam_k, v_k) of K in descending order; c_k = (v_k . E)^2 / n the error energy on mode k; per layer l with
weights W_l (fan_in -> fan_out): input activations a_l(x), preactivations z_l(x), backprop signal delta_l(x) (per output),
clouds A_l = [a_l(x_i)], Delta_l = [delta_l(x_i)], Grams G_l = A_l^T A_l (+1 for the bias feature), D_l = Delta_l^T Delta_l.

## 2. Exact results (gradient flow; any loss l via grad l in place of E; any pointwise activation)

(T1) Energy identity: d/dt ||E||^2 = -2 <E, K E>;  Edot = -K grad l.
(T2) Layer decomposition: K = sum_l G_l (Hadamard) D_l / n (bias included via +1 in G).  Verified to 1e-15.
(T3) Data-indexed weight change (coherence): dW_l(T) = -int_0^T Delta_l(t) U(t) A_l(t)^T dt with U = diag(grad l / n).
     Its rank is at most (points x outputs); every layer's change is the backprop cloud times the activation cloud,
     weighted by where the error has sat. No direction is free.
(T4) Coherence maps (T3 inserted into the forward/backward passes), for an increment dU:
       z_l(x) <- z_l(x) - sum_i delta_l(x_i) dU_i G_l(x_i, x)                 (forward shift, per unit)
       m_l(x) <- m_l(x) - sum_i a_{l+1}(x_i) dU_i D_{l+1}(x_i, x)              (backward shift, per unit; delta_l = sigma'(z_l) m_l)
     Exact per unit; verified: the forward-shift covariance matches the real network to 3 digits.
(T5) Per-mode eigenvalue rate (Hellmann-Feynman): d lam_k/dt = 2 |g| sqrt(lam_k) sqrt(q_k) (-cos_k), with q_k the energy of
     the second-order features along the gradient on mode k, cos_k their alignment. Verified to ratio 1.00.
(T6) Per-layer share of the loss decrease: <E, K^(l) E> / <E, K E>, additive, each term >= 0 (benefit of each layer's
     change is exact). Validation analogue: d(E_val.E_val)/dt = -2 <E_val, K_vt E> (cross kernel), also additive.

## 3. Empirical facts (all on the testbed; sample sizes in REPORT.md)

(F1) Training is kernel regression with the CURRENT kernel: ||E(t)||^2 ~= sum_k <v_k(t),E0>^2 exp(-2 lam_k(t) t)
     within 0.64-1.13 (5-95 %) over 86 runs and all times; any fixed or earlier basis fails by 2-20x on rough targets.
(F2) Rank is roughness: the graph-Laplacian roughness of the k-th eigenvector is the same function of rank for K0 and
     K(T) (ranks > 100 at all times; all ranks at init) across 42 runs. The spectrum is a law lam(rho).
(F3) Feature learning lifts the tail: lam_k(T)/lam_k(0) ~ k^p, p a saturating function of the target's coefficient
     decay in K0's basis (p = 0 at Delta 0.1; 2 at 0.5; 3.2 at 1.0; 3.4 at 2.0) times richness (0.45 lazy .. 4.4 rich);
     nearly independent of width and depth. Smooth targets never lift in any regime; lazy runs never lift.
(F4) The lifted tail is Q = dPsi dPsi^T / n (the cross term lifts nothing); relative to K0's tail Q is generically
     oriented (randomly rotated Q reproduces the lifted spectrum within 1.5x to rank 400, rough targets).
(F5) dW_l is low rank in practice (4 directions carry 63-69 % of the lift, 8 carry 89-91 %); top singular values rise
     from the MP edge to 3-9x the edge; spikes of adjacent layers are coherent: randomly rotating each layer's spikes
     (same spectra) cuts the lift 5-15x; the gradient chain at the same amplitude reproduces it within 1.1-3x.
(F6) The lift composes multiplicatively across layers (hidden-only 27, output-only 26, input-only 5, all 337); the
     roughness enters through the backprop factors (delta_0 roughness 0.025 -> 0.2-0.7) while activations stay smooth.
(F7) Hermite-type response: at init along the coherent direction, the exponent of lam_k vs amplitude is linear in
     log(lam1/lam_k(0)) (residual 0.04-0.16 over 380 ranks, 14 designs); slopes 0.13-0.18.
(F8) Late training: the residual's effective eigenvalue equals the current spectrum at the residual's mean rank
     (within 1.5-2x) while that rank drifts 100 -> 500; all late descent goes through the lifted tail (share 1.00).
(F9) The amplitude law A_l^2 = sum_k q_lk J_k^2 (J_k integrated residual per mode, q_lk = v_k^T K^(l) v_k at init)
     reproduces measured amplitudes within 1.5-2.5x with no fitting (over by 1.7x in the rich regime).

## 4. What does NOT work (established negatives)

(N1) Anything evaluated once at initialization: lazy kernel, init spectral scores (ranking 0.6-0.8, scale-blind),
     a spike on the initial weights in any init-derivable direction (10-50x short), integrated residual times
     initial features (no lift). Prop. 20 of the paper, in weight space: the rough features must be built first.
(N2) Scalar-amplitude closures (one amplitude per layer driving a response law): bistable (starve or run away) even
     with per-design coefficients. The response is local in roughness (a front), not global.
(N3) Random-rotation (isotropic) closure for the residual's basis change: wrong anisotropy (held-out ranking 0.38).
(N4) Two-point Gaussian tables (unit covariances) with coherent shift covariances: forward right, backward coherent
     term 1-9 % of exact; the term is a 4-point statistic of one unit's preactivation process.
(N5) The paper's held-rough-error test with rank-one Krylov functionals: cross term 4-12x the diagonal; the leak
     bound lam1 ||E_s||^2 is 100-700x the true cross term. Empty, by construction of the functional.

## 5. The model (statistical, init-only): virtual Gaussian unit ensemble with coherent maps

State. Per layer l: NV virtual units, each a pair of functions on the data, (z_q(.), m_q(.)); the residual vector E;
the integrated residual u. No weights, no directions. Layers couple only through the Grams G_l, D_l and u.
Initialization. Units sampled as independent Gaussian processes: z ~ GP(0, Zc_l), m ~ GP(0, D_{l+1}/fan_out_l), with the
covariances taken from the untrained network on the data (or, in the wide limit, from the standard recursion).
Tables. A_{l+1} = fan_out_l E_q[sigma(z_q) sigma(z_q)^T] + 1,  D_l = fan_out_l E_q[delta_q delta_q^T], delta_q = sigma'(z_q) m_q.
Kernel. K = sum_l A_l (Hadamard) D_l / n, assembled as the measured initial spectrum by rank times the ensemble's lift
ratio (positive by construction).
Stage. (i) residual decays by exact kernel regression in the current K over the stage; (ii) du = int E dt / n;
(iii) apply (T4) to every virtual unit with du and the CURRENT Grams; (iv) recompute tables. Stage length adaptive:
budget on the RMS increment of u (EPS), so stages follow the path length.
Cost. Independent of parameter count and dataset size: O(stages x depth x (NV n^2 + n^3)); seconds to minutes per
design on one GPU at n = 200-1000.

### Assumptions, explicitly
(A1) Gradient flow (full-batch GD with small step) on the given loss. Adam/SGD/weight decay change the descent rule
     and therefore (T3)-(T4); not yet derived or tested.
(A2) Pointwise nonlinearity after a linear map (dense/conv). Attention and normalisation layers: maps not derived.
(A3) Gaussian unit ensemble: units of a layer are exchangeable and conditionally independent given the Grams, at init
     and along training (mean-field/DMFT-type assumption; rigorous for wide 2-layer nets, derived for deep).
(A4) Within a stage the kernel is frozen (piecewise kernel regression); convergence in the stage budget is checked
     numerically, not assumed.
(A5) The data subsample of n points represents the task down to the n-th roughness mode; finer modes are invisible.
(A6) Finite-width tail: the ensemble's initial tail differs from the real network's beyond rank ~200-300 (tables match
     to 4-6 %); the model's lift ratios there are not trusted; the measured initial spectrum is used as the base.

## 6. Validation of the model

Checkpoint (fixed shift, exact targets from the shifted network): tables within 0.5-2x layer by layer; lift at ranks
50-100 within 1.0-1.5x (vs 1-9 % for two-point tables).
Dynamics, reference H96 d2 Delta 0.5 (measured E(T) 0.039, lift100 338): fixed stages 80/200/400/800 -> E 0.150/0.085/
0.030/0.025, lift100 381/417/485/606; adaptive (EPS 0.02, 17678 stages) -> E 0.025, lift100 1533. The error converges
to 0.6-0.8x of measured; the kernel at rank 100 is over-lifted 1.4-4.5x at convergence (A3/A6), the error being
insensitive to the deep tail. Rough reference H96 d3 Delta 1.0 (measured 0.035, 3444): fixed 400/800/1600 -> E 0.43/0.37/
0.22, lift100 1028/1531/2294 (not converged; adaptive run pending).
Blind, 28 held-out designs (seeds 1-2, widths 48/160, depths 2-3, Delta 0.5/0.75/1.0, s=1 and 0.5), 400 fixed stages:
  Delta 0.5: median pred/meas 1.15 (0.62-1.76), Spearman 0.76;  Delta 0.75: 5.3 (3.2-10.6), 0.91;  Delta 1.0: 5.3 (2.9-15), 0.74;
  pairwise design decisions 87 % (90 % for pairs > 1.3x apart). The 5x on rough targets is stage resolution (see above).
For comparison, the staged propagation with the real network's weights (model v4, re-linearised every delta = 0.25 of
displacement): blind median 1.04 (0.87-1.34), Spearman 0.95, 100 % of pairs > 1.3x. It is the numerical upper bound on
what the statistical model can reach, at ~100 Jacobians per design.

## 7. Break conditions (where the vision stops)
(B1) Subsample resolution (A5): absolute late error below the n-th mode's scale is invisible; design ranking at a
     fixed budget is unaffected until the budget reaches that scale. Testable here: predict n=400/800 runs from 200.
(B2) Architecture (A2): transformers need the maps for attention/normalisation.
(B3) Optimizer/loss (A1): Adam (destroys outer-product structure), SGD noise (adds a bulk), weight decay (caps
     amplitudes), cross-entropy (restricts output directions): each a one-place change, none run.
(B4) Ansatz (A3): narrow layers; strong saturation. Internal checks: init self-test (shift the real init network and
     the ensemble, compare), convergence in EPS and NV, init-to-init spread.
(B5) Validation error: out of scope; cross-kernel extension (T6) exists in principle.

## 8. Test program (compute-heavy part)
S1 Convergence and accuracy of the ensemble model at converged staging on the existing blind set (login GPU).
S2 Real-init ensemble: build the virtual units from K random initializations of the real architecture (exact
   finite-width statistics), compare with the Gaussian ensemble; this is also the route to arbitrary architectures.
S3 Scaling ground truth: n = 400, 800 (subsample-to-full prediction), depth 6 and 8, width 1024, 3 seeds each.
S4 Optimizer ground truth: Adam (several lr), GD with momentum, SGD (batch 32/64), weight decay; same designs.
   Diagnostics as in sweep A/C (spectra over time, dW spectra, per-layer blocks) to see which of (F1)-(F9) survive.
S5 Loss ground truth: L1/Huber and a classification variant (cross-entropy) on the same data.
S6 Other problems: a second chaotic system (Rossler), a synthetic rough regression on a 2-D manifold, and a small
   convolutional image task (architecture transfer; needs the conv maps).
S7 Model extensions in the same order: preconditioned maps for Adam; grad l for losses; conv maps; then blind tests.

## 9. Test-program results (2026-10-10; 4-node interactive session, job 59653997)

Ground truth: `l63_design2.py` (optimizer/loss/data variants), diagnostics `survival.py` -> `runs/survival_table.csv`.
Columns: ckr = current-kernel-regression ratio (squared-loss formula), p = eigenvalue-gain exponent (ranks 20-300),
lift100 = sorted gain at rank 100, erank = effective rank of the hidden-layer spikes.

S4 optimizers (28 runs, H96, Delta 0.5/1.0, depth 2/3, T = 3000 GD-time units; Adam/momentum at the listed lr):
- Momentum (0.1) and minibatch SGD (b = 32, 64): indistinguishable from full-batch GD. ckr 0.55-1.05, p 1.7-3.3,
  lift100 216-3273, erank 4.8-9.9, same gain law. (F1)-(F9) survive unchanged; the noise bulk of (B3) is not visible
  at these batch sizes.
- Adam (lr 0.03, 0.1): the spike structure and the gain law survive (erank 6-13, p 2.8-4.0, lift100 224-2192), but
  current-kernel regression with the plain kernel fails (ckr 1.4-6.2, i.e. the residual follows the preconditioned
  kernel), and the training error is 0.002-0.012 (vs 0.02-0.16 for GD). The model needs the preconditioned coherence
  maps (S7) before Adam can be predicted.
- Weight decay: 1e-3 weakens the gain (p 0.8-2.6) and dilutes the spikes (erank 23-42); 1e-2 over T = 3000 destroys
  them (erank 48-55, lift100 2-7, training error 0.45-1.07). Decay enters the maps as a shrink of z and m per unit time.

S5 losses (16 runs): Huber keeps everything (p 1.8-3.1, lift100 214-2339, erank 4.8-8.5). L1 keeps the spikes and the
gain law (p 2.7-4.0, erank 7.5-12) but the top eigenvalue SHRINKS (lam1(T)/lam1(0) = 0.38-0.87) while the tail lifts;
the squared-loss ckr formula does not apply (0.12-0.8). Both are the same model with grad l(E) in place of E (T2).

S3 scaling (15 runs): depth 6 and 8, width 512 and 1024: ckr 0.74-1.03, p 1.3-2.3 at Delta 0.5; lam1 gain grows with
depth (13-22 at depth 6-8, 86 at depth 6 Delta 1.0) = multiplicative composition (F5). Subsample: n = 400 and 800 give
the same exponent as n = 200 (d2 Delta 0.5: 1.93/1.94 vs 1.98; d2 Delta 1.0: 3.0/2.9 vs 3.2) and the same lift100
(331/396 vs 338): the gain law transfers from a 200-point subsample to the full set, which is what (B1) needs.

S6 other data (8 runs, GD, squared loss): Rossler one-step map: small gains (p 0.26-0.65, lift100 5-13), erank 4-5;
synthetic rough 2-D manifold regression: p 0.77-1.84, lift100 49-373, erank 9-12. ckr 0.88-1.30 throughout. The
exponent is set by the target (F3), not by the architecture; consistency with the roughness law is to be checked
with `rough_law.py` on these runs.

S2 real-init ensemble and the kernel transfer: building the units from 32 real initializations does NOT remove the
over-lift (d2 Delta 0.5: gain100 1761 vs Gaussian 1539 vs measured 338; d3 Delta 1.0: 41901 vs 16052 vs 3444), so
the Gaussian idealization is not the cause. The init tails of ensemble and real net agree within 1.2-2x. Transferring
the ensemble's lift ADDITIVELY by rank (lam_k = lam_k^meas(0) + [lam_k^ens(t) - lam_k^ens(0)], env ADD=1) instead of
multiplicatively brings the tail to gain100 559/579 (d2) and 4267/4562 (d3) vs 338/3444 measured (1.3-1.7x), but the
error is then 0.017/0.018-0.021 vs 0.039/0.035 measured (about 2x too low). The error bias is therefore not a tail
problem; an open-loop test (maps driven by the measured residual history, env OPENLOOP=<late run>) that compares the
model kernel's action on the real residual with the measured one is running to locate it.

Open-loop result (ensemble of 2048 Gaussian units, additive transfer, maps driven by the measured residual of the
real run; compare the model kernel's action on the real residual, lam_eff = E'KE/E'E, with the measured one):
- d2 Delta 0.5: matches to t ~ 3 (3.7e-2 vs 3.7e-2), then the model kernel keeps acting on the real residual at
  ~1.6e-2 while the real kernel's action falls to 2.8e-3 (t = 224) and 2.9e-4 (t = 3000); the model ends at 5.1e-2,
  i.e. 180x the measured value, with gain1 25 (meas 5.0) and gain100 6263 (meas 338). Driven by the true residual,
  the ensemble lifts the kernel IN THE RESIDUAL'S OWN DIRECTIONS far more than the real network does. The closed loop
  hides most of this: its residual dies 2x faster, so its integrated drive is smaller (gain1 4.6, gain100 559).
- d3 Delta 1.0: the opposite sign. With the true drive the ensemble under-lifts (gain100 73 vs 3444, lam_eff 0.4x
  measured); closed loop reaches gain100 4267 only because its own residual direction is self-consistent.
Conclusion: the closed-loop agreement of the lift (section 6) is partly compensation. For depth 2 the per-unit maps
are exact given the drive, so the open-loop excess must come from the kernel assembly from a different draw of units
(mean-field tables vs this network's 96 units) or from the staged integration. Self-test running: ensemble = the
run's own 96 units (env REAL=1 OWN=1), open and closed loop, both references. If it reproduces the measured kernel,
the limitation is init-specificity of the late residual directions (a finite-width effect the ensemble cannot carry);
if it does not, the maps or the integration are at fault and fixable.

## 10. The cause of the error, found and fixed (2026-10-10, evening)

Facts established by direct test (all on H96 d2 Delta 0.5 unless stated; `runs/ens/resolve/`):
1. Ground truth is gradient flow: rerunning both references with a 4x smaller step reproduces E(t), lam1(t) and the
   rank-100 gain to four digits at every logged time (`runs/gt_lr/`). The finite-step explanation is excluded.
2. The staged integration is converged: the own-unit model at stage budgets EPS = 0.03, 0.01, 0.003 gives identical
   E(25) = 1.0725/1.0722/1.0721 and identical gains.
3. "depth 2" in this code has TWO hidden layers (weight shapes 96x3, 96x96, 3x96). The coherent maps carry each unit's
   change from its own weight update but omit the change that propagates through the fixed hidden-hidden weights:
   forward W*da (from the layer below), backward W^T*d(delta) (from the layer above). At t = 8 these terms are 11 % of
   the preactivation displacement and 28 % of the backward-signal displacement; the omission compounds over three
   decades of t into the 2x error at T and the wrong eigenvalue gains.
4. With the propagation terms (env PROP=1: each ensemble member keeps its own hidden-hidden INIT weights, block-
   diagonal over members, updated by the coherent increments; z and m of layers >= 1 are recomputed through them), the
   own-unit model (REAL=1 OWN=1) matches the real network at t = 8 to ||E|| 1.2569 vs 1.2569, every sorted eigenvalue
   to 0.1 %, every unit state to 1e-4 relative, and tracks it through t = 224 (E 0.267 vs ~0.27, gain1 5.86 vs 5.85,
   gain100 100 vs ~100). `code/real_snapshot.py`, `code/compare_t8.py`.

Consequences for the model and its claims:
- The coherent part (spikes, gain law, layer coupling through the residual, Hermite response) is statistical and
  cheap, as stated in sections 2-5. The propagation part is the random, init-specific part of the dynamics. It needs,
  per ensemble member, the inter-layer weights at INITIALIZATION (drawn, never trained): the model stays init-only
  with no weights along the path, but it is no longer weight-free. The infinite-width (DMFT) description of this term
  is a Gaussian increment per unit with covariance from the Gram of the activation change; at H = 96 the finite-width,
  init-specific realisation matters, which is why a different draw of units cannot stand in for this network's late
  error better than ~2x (section 9, open-loop result).
- The earlier closed-loop "agreement" of the Gaussian ensemble (section 6) was compensation between the missing
  propagation (too little kernel change per unit drive) and the ratio transfer (too much tail); both are now removed.
- Cost: per stage one (n ny)^2 eigendecomposition plus H^2 n per hidden interface, ~1e4 stages for three decades of t,
  independent of parameter count and dataset size. On this testbed that is slower than training (10-20 min vs 1-7
  min); the advantage exists only at scale and rests on the subsample assumption (A5), which the n = 400/800 ground
  truth supports for the gain law.
- Running: own-init model to T = 3000 on both references; 8-member and 1-member other-init ensembles (init-specificity
  and spread); the 28-design blind batch with the corrected model, own init and one other init per design (`runs/ensP/`).

## 11. Closing results and conclusion (2026-10-10, night)

Reference designs, corrected own-init model (REAL=1 OWN=1 PROP=1 ADD=1, EPS 0.03) vs measured training run:
  H96 d2 Delta 0.5: E(T) 0.0394 vs 0.0393; gain1 4.99 vs 5.00; gain100 338.0 vs 338.0; gain200 1476 vs 1471.
  H96 d3 Delta 1.0: E(T) 0.0336 vs 0.0346; gain1 10.92 vs 11.11; gain100 3470 vs 3444.
  H96 d8 Delta 0.5: E(T) 0.0063 vs 0.0063.
Member-averaged kernels are biased: an 8-init ensemble sharing one residual gives E(T) 0.0132 (d2) and 0.0071 (d3),
3-5x too low, because averaging kernels over members removes the init-specific fluctuation and over-lifts. Single
other-init members track the own init to a few % (prop1 d2 at t = 671: 0.124 vs own 0.131 vs measured 0.129).
Error bars must therefore come from single-init runs, never from a member-averaged kernel.

Early-training predictors (82 GD runs; `code/early_predict.py`, `code/front_predict.py`, `runs/early_predict.csv`):
frozen kernel from t0 (an upper bound under a fixed eigenbasis; holds in 82/82 runs for t0 <= 0.1 T), rank-wise
gain-law extrapolation, residual-front extrapolation, and E(t0) itself all order designs within 1.1x at 55-65 % and
within 1.1-1.3x at 65-76 % for t0 <= 0.1 T. The front moves at a nearly constant rate (0.12-0.22 per decade) and its
position at T is predictable to 10 %, but the error is not recoverable from it. Seed-to-seed spread of E(T) at fixed
design: median 1.155, range 1.01-1.46 (17 designs); seed pairs agree in E(t) to 2 % for t <= 0.1 T and diverge late.
Full statements and proofs: `report/THEORY.md`.

Conclusion.
1. The exact init-only model (Theorem 4 of THEORY.md) is gradient flow restricted to the subsample, integrated in
   ~1e4 stages. It is exact to 1-3 % on every design tested, including depth 8, but it has no practical advantage over
   simply training on the same subsample for a few minutes, and on this testbed it is slower than full training.
   It is kept as the validation tool and as the proof that the staged picture is complete, not as the deliverable.
2. No init-only or early-training quantity forecasts the late training error below a factor of about 2 (init-only)
   or below the seed spread (early training). The information that decides fine comparisons is in the late,
   init-specific kernel dynamics (the propagation term of Theorem 4).
3. The deliverable is the instantaneous (E-dot) diagnostic, where the theory is exact: the per-layer and per-mode
   decomposition of the descent rate (Theorems 1-2), the split of the second derivative into lazy decay and
   feature-learning acceleration (Theorems 3-5), the gain exponent from the target's roughness in the init basis,
   and the front position; all from one Jacobian on the subsample, all verified across optimizers, losses, depths,
   widths and datasets (section 9).
4. The question "cheap diagnostic for large networks" reduces to one experiment that does not involve this model:
   does training on a small subsample order designs as the full data does? If yes, the diagnostic is subsample
   training plus the E-dot decomposition, and the theory says why (the gain law and the front depend on target and
   architecture, not on n; supported by n = 200/400/800 for the gain law). If no, nothing on a subsample can work.
   This needs a handful of full-size training runs once.
5. Design resolution below the seed spread (15 %) is only defined for the seed-averaged error and needs several runs
   per design under any method.

## 12. Init-only adaptability ranker (added after the user's point that E-dot alone ignores kernel adaptability)

Quantities at init, one Jacobian plus one directional second derivative on the subsample (seconds): H2 = operator norm
of the second derivative of the network map along the descent direction (how fast the Jacobian, hence the kernel, can
change), and k90 = rank holding 90 % of the target's energy in the init kernel basis (target roughness).
60 GD runs at output scale 1 (n = 200, depths 1-8, widths 32-1024, Delta 0.5-1.0), leave-one-out:
  log E(T) = -16.8 - 1.29 log H2 + 2.59 log k90:  Spearman 0.93, median |pred/meas| 1.38, 90th pct 1.92;
  pairs ordered correctly: 97 % (true gap > 2x), 75 % (1.3-2x), 67 % (1.1-1.3x), 53 % (< 1.1x).
  H2 alone: Spearman 0.84; target alone: 0.45; Delta alone: -0.01 (Delta is not visible without the architecture).
Within a roughness class H2 alone ranks with Spearman -0.90/-0.93 (Delta 0.5/1.0), better than depth at intermediate
gaps because it also orders width. The Hellmann-Feynman "adaptation happening now" at init (per-mode rates on the
initial residual) carries no information (Spearman ~0): what matters is the CAPACITY to adapt (H2), not the adaptation
at t = 0. Open: the normalisation across output scales (the Delta 0.75 class, which mixes scales, is not ranked by raw
H2). Status: a 3-parameter fitted law, LOO-validated on 60 designs, resolution ~1.3-2x; the "educated guess" tier.
Improved (same 60 designs, LOO): log E(T) = -3.21 - 1.27 log H2 + 4.86 log E_lazy(T) + 0.50 log lam1(0), where
E_lazy(T) is the EXACT frozen-init-kernel error at the budget (from lam0 and the initial residual; 1.5-150x above the
real error): Spearman 0.97, median |pred/meas| 1.24, 90th pct 1.90; pairs correct 99 % (> 2x), 85 % (1.3-2x), 68 %
(1.1-1.3x), 54 % (< 1.1x); excluding same-design seed pairs changes nothing. Next features (no allocation): H2 along the
residual the init kernel cannot fit by T (slow-mode H2), per-layer H2, output-scale normalisation; more depth-4-8 and
width-256-1024 ground truth would cost allocation.
Slow-mode and per-layer H2 (code/init_features.py -> runs/init_features.csv; recomputed H2 matches the saved value to
3 digits): 53 unique scale-1 designs, LOO. Best: H2 summed over HIDDEN layers along the slow-mode direction + E_lazy +
lam1: Spearman 0.974, median 1.25x, 90th pct 1.80x, pairs 100 % (> 2x), 89 % (1.3-2x), 70 % (1.1-1.3x), 54 % (< 1.1x).
Descent-direction H2 + E_lazy + lam1: 0.970 / 1.30x / 100-85-67-54 %. The slow-mode direction alone is not better
than the descent direction (0.946); the per-layer split helps slightly; the scalar second derivative is worse (0.863).
Output-scale normalisation NOT solved: fitting on scale 1 and predicting the 9 rich/lazy runs (s = 0.25, 0.5, 4) gives
median errors 3.2-4.3x with any of H2/s, lam1*T, gnorm/s (ordering among them Spearman 0.8). The rich regime needs
its own ground truth or a richness feature; see the proposed ground-truth extension.
More eigenvalues and higher order (code/init_features2.py -> runs/init_features2.csv; 53 scale-1 designs, LOO):
adding spectrum-shape features (lam at ranks 10/50/100/200, tail exponent, E_lazy at T/10 and T/100), an adaptability
spectrum (H2 along the residual restricted to rank bands 1-10, 10-50, 50-200, 200-600), or third-order directional
derivatives does not move the ranker beyond noise: every variant sits at Spearman 0.95-0.98, median 1.23-1.35x, and
67-72 % on pairs 1.1-1.3x apart; greedy selection over 25 features reaches 0.978 at 4 features, within LOO noise of
the 3-feature model. The per-band H2 falls from the top band to the tail (e.g. 11.9 / 8.2 / 7.3 / 3.8 at depth 8),
i.e. the init-time adaptability is largest where the residual is already fast; the tail bands alone rank worse
(0.93-0.94). Conclusion: the init-only ranker is saturated at ~1.25x median / ~1.3x pairwise resolution with these
53 designs; the remaining variance is in the late, init-specific dynamics (section 10-11), not in higher init moments.

## 13. Ground-truth extension (job 59657931, 2 nodes x 4 GPUs, 3 h, m5326): deep/wide and rich-regime designs
24 launched (`runs/gt_ext/`): depths 4/5/6/8 x widths 128/256 x Delta 0.5/1.0 (scale 1) and the rich regime
(output scale 0.5, T = 12000) at depths 2/3 x widths 96/256 x Delta 0.5/1.0. H256 d8 Delta 1.0 was stopped (lam1 > 9000
at t = 40, step size forced below 1e-4, est. 7 h); H128 d8 Delta 1.0 and H256 d6 Delta 1.0 may not finish in time.
Cost note: deep/wide GD is expensive precisely because lam1 grows 20-100x and the stable step shrinks with it.
Interim (18 finished):
- TRUE out-of-sample test: ranker fit on the previous 53 scale-1 designs, applied to the 10 new deep/wide ones:
  Spearman 0.89, median error 1.4-1.5x, max 2.1-2.7x; pairs 100 % (> 2x), 91 % (1.3-2x). Systematic: deep Delta 0.5
  designs are predicted 0.4-0.6x too LOW (measured error saturates with depth, 0.0094 -> 0.0050 from depth 4 to 8,
  while log H2 keeps growing); the log-linear law needs a saturating form in H2. Refit on all 63: LOO 0.978, 1.30x.
- Rich regime: predictions are biased by a clean power of the output scale (3-4x high at s = 0.5, 6-8x at s = 0.25,
  0.3-0.5x at s = 4). Adding log(scale) as a feature resolves it: joint LOO over all 80 designs (scale 0.25-4) with
  H2 + E_lazy + lam1 + scale: Spearman 0.981, median 1.26x, 90th pct 1.9x, pairs 100 % (> 2x), 91 % (1.3-2x),
  70 % (1.1-1.3x), 56 % (< 1.1x). The "normalisation problem" of section 12 was a missing s^c factor.
Saturating forms (added (log H2)^2, 1/H2, (log E_lazy)^2, H2 x E_lazy interaction; LOO over 80 designs): none improves
on the log-linear law (Spearman 0.974-0.981, median 1.23-1.26x); the extra terms that fix the deep Delta 0.5 designs
in-sample overfit out-of-sample (max error 2.4x vs 1.9x). Out-of-sample on the 18 new designs with the scale term:
log-linear Spearman 0.89, median 1.29x, max 1.94x. The deep Delta 0.5 under-prediction (0.5-0.6x) is a floor effect:
at depth >= 6 the measured error (0.004-0.007) is set by the budget on the last subsample modes, i.e. break condition
(B1), which a smooth law in H2 cannot see; it needs E_lazy evaluated with the lifted tail, not a new feature.
Final (21 of 24 finished; H128 d8 Delta 1.0 and H256 d6 Delta 1.0 were at t = 600 after 2 h and were stopped with the
launcher; allocation released at 2 h 01 min of 3 h). Feature tables: runs/init_features_ext2.csv; fit: code/refit_ranker.py.
- Out-of-sample (fit on the previous 53 scale-1 designs, predict the 13 new deep/wide ones, depths 4-8, widths 128-256):
  Spearman 0.89-0.90, median error 1.28-1.50x, max 2.1-2.7x; pairs 100 % (> 2x), 89 % (1.3-2x), 75 % (1.1-1.3x).
- All 83 designs (scales 0.25-4, depths 1-8, widths 32-1024), LOO, log E(T) ~ log H2 + log E_lazy + log lam1 + log scale:
  Spearman 0.981, median 1.28x (1.24x with hidden-layer slow-mode H2), 90th pct 1.8-1.9x, max 3.1x;
  pairs correct 100 % (> 2x), 92 % (1.3-2x), 69 % (1.1-1.3x), 57 % (< 1.1x).
- Known residual: deep designs near the subsample floor (B1) are predicted 0.4-0.6x low; the floor is a testbed property
  (n = 200, fixed budget), not a depth property, and should be modelled as a lower bound from the lifted tail.
Status of the deliverable: an init-only, seconds-per-design ranker with ~1.3x median error and ~1.3x pairwise resolution,
validated out of sample on deeper and wider networks than it was fit on, and across output scales with one scale term.
Not validated: large n (the subsample-to-full-data question), other architectures, validation error.
