# Late-stage training of a Lorenz-63 one-step map: what the kernel does and an approximation that works

Session of 2026-10-09 on Perlmutter (job 59609484, 2 nodes x 4 A100, interactive, m5326).
All code in `$SCRATCH/l63_held_rough/code`, runs in `$SCRATCH/l63_held_rough/runs/sweep{A,B,C,D}`,
figures in `runs/fig*`. Numpy original: `l63test.py`; torch port: `l63test_torch.py` (bit-for-bit checked
against numpy); late-stage diagnostics: `l63_late.py` (CUDA-graph GD step, 0.28 ms/step).

## 0. Why the original experiment (held rough error along Krylov functionals) was empty
(see chat) The rank-one Krylov functionals have cross term <phi,K E_s> 4-12x the diagonal term sigma^2 <phi,E>
from t=0; the section assumes the opposite. The leak bound lam1 ||E_s||^2 is 100-700x the true cross term.
Not a code problem.

## 1. Setup
Lorenz-63, n=200 samples, Delta in {0.1, 0.5, 1.0, 2.0} (target roughness), tanh MLP width H in {32, 96, 256, 512},
depth 1-4, full-batch GD with eta <= 0.5/lam1 as gradient-flow proxy, T = 3000 (10000 for sweep D).
Kernel K(t) = Psi Psi^T / n (600 x 600). Everything is measured against the CURRENT kernel; nothing is frozen.
Roughness basis: symmetric normalized kNN graph Laplacian on the standardized inputs (k=10).

(sections 2-6 filled in below as results come in)

## 2. An approximation that works: current-kernel regression (CKR)
Decompose the target (sweep A) or the initial error E0 (sweeps B-D) in the eigenbasis {v_k(t), lam_k(t)} of the
kernel AT TIME t and damp each component with the current eigenvalue over the elapsed time:

    ||E(t)||^2  ~=  S(t) := sum_k <v_k(t), E0>^2 exp(-2 lam_k(t) t)                       (CKR, instantaneous)
                 or  sum_k <v_k(t), E0>^2 exp(-2 int_0^t lam_k(s) ds)  (modes labelled by rank; "adiabatic")

Sweep A (36 runs, H in {32,96,256}, depth 1-4, Delta in {0.1,0.5,1.0}, T=3000): sqrt(S)/||E|| stays within
0.6-1.6 over the WHOLE trajectory (figA/ckr_ratio.png), median late-stage ratio 0.8-1.2 for most runs
(figA/summary.csv, columns S_inst_ratio, S_adia_ratio). It holds mode by mode for the modes that carry energy
(per-run figures, panel "per-mode current-kernel regression test").

By contrast, every prediction that carries a basis forward from an earlier time fails: from t0 = time at which
||E|| has halved, frozen-spectrum prediction is off by 2-10x at T (pF/E), and even the moving-spectrum-by-rank
prediction from t0 is off by 2-7x (pM/E). The rotation of the eigenbasis is essential, and the CKR captures it
by always re-expanding in the current basis. This is the "no frozen kernel" statement made quantitative:
the system at time t looks as if it had been trained for time t with the kernel it has NOW.

Late-stage facts (all runs): the residual has zero energy on the top-10 current modes; its energy-weighted mean
rank is 90-480 of 600; lam_eff := <e,Ke>/||e||^2 ~ alpha/t with alpha the late power-law exponent of ||E||.
The residual's graph-Laplacian roughness index is 0.9-1.3 against 0.08/0.46/0.57 for the target at
Delta=0.1/0.5/1.0, and the top kernel eigenvectors stay smooth (0.05). Late training = fitting modes that are
far rougher than the target, deep in the kernel tail.

## 3. What the kernel does: the tail is lifted by orders of magnitude, as a power law in rank
- Smooth target (Delta=0.1): no tail lift (lam_k(T)/lam_k(0) = 2-10, flat in k); alpha ~= 0.5 for every
  architecture; late training is kernel regression with the initial spectrum.
- Rough targets: lam_k(T)/lam_k(0) ~ k^p with p ~= 2 (Delta=0.5) and p ~= 3 (Delta=1.0), the SAME slope for all
  widths and depths (figA/scale_vs_k.png); lam_100 lifts by 10^2-10^3 (Delta=0.5) and 10^3-10^4 (Delta=1.0),
  lam_400 by up to 10^6. The lifted spectrum is much flatter than the initial one.
- In time, tail eigenvalues rise as power laws, lam_k(t) ~ t^gamma_k with gamma_k increasing with k
  (figA/tail_trajectories.png; tail_exponents.py table), and the lift at rank k scales with the integrated drive
  C(t) = int H2 ||E|| ds (Prop. 15 bound) as C^p_k with p ~= 1 at k=10 and p ~= 1.6-2.1 at k=100, 2.1-2.8 at
  k=200, consistently across H and depth.
- Tail mass M_k = sum_{j>=k} lam_j vs path length ell = int |g_E|: exponent 2.0-2.4 at k=50, 2.4-2.9 at k=100,
  2.9-3.7 at k=200 (tail_mass.py). Prefactor of the k=100 lift in units of (H2bar ell)^2: 1e-5..1e-4 (Delta=0.1),
  3e-4..1.4e-3 (Delta=0.5), 6e-4..6e-3 (Delta=1.0): set by target roughness, weakly by architecture.

## 4. Mechanism (theory link): the lifted tail is the Gram matrix of the feature displacement
K = Psi Psi^T/n, so K(t) - K(0) = [Psi0 dPsi^T + dPsi Psi0^T]/n + dPsi dPsi^T/n =: X + Q. Psi0 has no weight on
the tail directions of K(0), so there only the quadratic term Q survives; it is second order in the displacement
(hence the ell^2 scaling at moderate rank), and Prop. 16 says dPsi is built from the second-order features along
g_E, i.e. from the curvature H2 along the residual path. Direct check (sweep C, L63_QUAD=1 logging): spectrum of
K0 + Q matches K(T) in the tail, K0 + X does not (table in section 4b).

## 4b. RMT view
dK's top eigenvectors are smooth (roughness 0.02-0.3) and NOT aligned with Y or E0 (alignment < 0.08); its
effective rank is 10-50 and tr(dK)/tr(K0) up to 30. Random-rotation test (rmt_check.py): spec(K0 + R dK R^T)
with R Haar-random reproduces the lifted tail of K(T) to within ~1.5x out to rank 400 for rough targets at
depth >= 2, i.e. the tail lift is a bulk (free-convolution-like) effect of Q's spectrum, not an aligned
rank-one push; for smooth targets dK is aligned with the top of K0 and random rotation over-predicts the tail.
The K0-diagonal part of dK alone over-predicts the deep tail (it ignores interlacing).
Init tail (RMT of the finite-width kernel): lam_100(0) grows with width (1.3e-5, 1.3e-4, 2.7e-4 for H=32,96,256
at depth 1) and with depth (1.3e-5 -> 1.5e-4 from depth 1 to 4 at H=32).

## 5. Architecture dependence (figA/arch_summary.png)
- alpha (late ||E|| ~ t^-alpha): ~0.5 for smooth target regardless of architecture; for rough targets it grows
  with depth (0.15 at depth 1 -> 0.75-1.6 at depth 3-4) and weakly with width. Depth-1 nets on rough targets
  stall (alpha 0.12-0.15, ||E(T)|| ~ 0.8).
- Final error falls with depth and (less) width; the relative tail lift lam_100(T)/lam_100(0) is nearly
  independent of depth >= 2 and of width at fixed Delta: the architecture sets the INITIAL tail (RMT), the target
  roughness sets the LIFT, and the product sets lam_k(T) and hence, through CKR, the late error.

## 4c. Direct mechanism table (sweep C, quad_check.py; H=96, T=3000)
At T, for every run and every rank k in {10,50,100,200,400}: lam_k(K0+Q) = lam_k(K(T)) within 5-15 %,
lam_k(K0+X) = lam_k(K0) (no lift at all), and Q alone reproduces K(T) in the tail. tr Q / |tr X| = 23-110 at T
(3-12 at t=21). So the late kernel is K(t) ~= K0 + Q(t) with Q = dPsi dPsi^T / n the Gram matrix of the feature
displacement; the cross term only matters at the very top and early.

## 6. How Q grows (q_shape.py)
- Q(t) converges to a fixed shape: relative Frobenius distance of Q(t)/trQ(t) to its final shape falls from ~1
  at t<10 to 0.1-0.17 at t=1500 (T=3000); top-10 eigenspace overlap reaches 0.98. Amplitude trQ(t) keeps
  growing slowly (trQ ~ ell^1.3 late; trQ/ell^2 decreases, i.e. the early growth is quadratic in path length and
  the late growth sub-quadratic).
- Consequence tested (analyze_late, "late-t1" columns): freezing the kernel at t1 = T/6 ~= 500 and predicting
  to T=3000 works for Delta=0.1 (ratio 0.9-1.04), is fair for Delta=0.5 (1.2-2.1x) and fails for Delta=1.0 at
  depth >= 2 (3-7x); the moving-spectrum-by-rank version from t1 is 1.0-1.5x / 2-4x. For rough targets the
  eigenvectors keep rotating late, and only the CKR (re-expansion in the current basis) stays within 20 %.

## 7. The kernel's roughness law (figA/roughness_law.png, q_shape.py)
- The roughness index of the k-th eigenvector is the same function of k for K0, K(T) and Q(T), for every run:
  0.06 (k<=10), 0.25-0.44 (10-50), 0.56-0.63 (50-100), 0.88-0.90 (100-200), 1.10-1.16 (200-400). Rank is
  roughness on this dataset (data geometry, Weyl-like), independent of architecture and training.
- Therefore the lift lam_k(T)/lam_k(0) ~ k^p is a change of the kernel's eigenvalue-vs-roughness law
  lam(rho): at t=0 it is the same steep curve for all architectures (8 decades from rho=0.1 to 1.2, with
  width/depth only shifting it up by < 1 decade); at T it is flatter by an amount set by the target roughness
  (about 5 decades for Delta=0.5, 4 decades for Delta=1.0 over the same range), nearly the same for all widths
  and depths. The learned kernel is a lower-order "Sobolev" kernel whose order is set by the target.

## 8. Negative result: no one-scalar reduction (ckr_shape.py, sweep C)
Ktilde(t) = K0 + q(t) Qhat with Qhat the FINAL shape of Q and q(t) = tr Q(t) fails in mid-training
(CKR ratio 0.2-0.5 for t in 10-700) and only converges to the fully-frozen late kernel at the end (0.5-0.75).
The exact K(t) gives 0.95-0.99 early and 0.6-0.9 late. So the eigenvectors of Q reorganise continuously and the
late dynamics cannot be reduced to an amplitude on a fixed learned kernel; the current kernel is the state.
(E0 and Y as CKR source give identical ratios: 0.769 vs 0.770 at t=127, 0.811 vs 0.813 at T.)

## 9. Exponents (lift_exponent.py; all runs with T=3000, seeds and n=400 included)
Fits over ranks 20-300: lam_k(T)/lam_k(0) ~ k^p; lam_k(0) ~ k^-a0; lam_k(T) ~ k^-aT; target energy in the
kernel basis yV2_k ~ k^-b at t=0 (bY0) and at T (bYT).
  Delta  |  p (lift)      | a0 (init)   | aT (final)  | bY0 (target in K0 basis) | bYT (target in K(T) basis)
  0.1    | -0.1 .. 0.1    | 5.2 .. 7.2  | 5.3 .. 7.3  | 3.5 .. 4.9               | 4.2 .. 5.5
  0.5    | 1.5 .. 2.4     | 5.2 .. 7.2  | 3.5 .. 5.0  | 0.8 .. 1.35              | 1.8 .. 3.7 (grows with depth)
  1.0    | 2.8 .. 3.6     | 5.2 .. 7.2  | 2.3 .. 3.2  | 1.0 .. 1.2               | 0.8 .. 1.4
  2.0    | 3.1 .. 3.4     | 5.8 .. 7.2  | 2.4 .. 4.1  | 0.45 .. 0.7              | 0.45
(H=32 depth 1 excluded: its init tail is at the numerical floor, a0 = 14.7.) Seed spread of p: +-0.2;
n=400 gives the same p (1.93, 3.03). Width 32-512 and depth 2-4 change p by < 0.5 at fixed Delta.
Reading: the target's roughness measured IN THE KERNEL'S OWN BASIS (bY0: how fast the target's coefficients decay
along the initial kernel's eigenmodes) determines how much the kernel's spectral law is flattened
(a0 -> aT, equivalently p = a0 - aT) and the flattening saturates near aT ~ 2.3-2.5 for the roughest targets.
Depth and width hardly enter p, but they enter the error through two other channels: the initial tail level
(RMT, section 4b) and the re-ordering of the basis towards the target (bYT - bY0 grows with depth at Delta=0.5),
which together with CKR give the faster late decay (alpha) of deeper nets.
The classical frozen-kernel exponent alpha = (b-1)/(2a) evaluated with the final (aT, bYT) under-predicts the
measured alpha by 3-4x (e.g. 0.19 vs 0.76 for H96 d2 Delta=0.5) because the tail keeps rising as t^gamma_k;
a time-dependent-spectrum version is the natural next step (not done here).

## 10. Persistence to T=10000 (sweep D, H=96 depth 2)
Delta=0.5: alpha 0.78 (0.76 at T=3000), p 2.05 (1.99), CKR ratio 0.89 (0.80): the late power law and the
estimate persist over another half decade, and the lift has essentially saturated by t~3000 (gamma_k at T=10000
falls to 0.15-0.34).
Delta=1.0: alpha steepens to 1.26 (0.97), p 3.37 (3.18), CKR instantaneous 0.62 / by-rank 0.86: the roughest
target keeps lifting its tail (gamma_k 0.3-0.8) and the instantaneous form of the estimate degrades slowly
while the by-rank integrated form stays within 15 %.
Intermediate roughness (sweep F): p = 0.5-0.65 (Delta=0.25), 2.7-3.0 (0.75), 3.3-3.4 (1.5); aT = 5.1-5.2,
2.8-2.9, 2.3-2.45. Data-seed replicates (sweep E, ds2/ds3) reproduce p to +-0.1.

## 11. Lazy vs rich (sweep G: output scale s, N_s = s (N - N_0), T = 3000/s^2 so kernel time s^2 t is matched)
H=96 depth 2.   s = 16 (lazy)  |  4  |  1  |  0.25 (rich)
  Delta=0.5:  p = 0.44 | 1.29 | 1.99 | 2.60 ;  aT = 5.4 | 4.5 | 3.8 | 3.2 ;  ||E(T)|| = 0.71 | 0.23 | 0.039 | 0.0074
  Delta=1.0:  p = 0.46 | 1.75 | 3.18 | 3.63 ;  aT = 5.4 | 4.1 | 2.6 | 2.2 ;  ||E(T)|| = 0.98 | 0.78 | 0.12 | 0.0092
So p -> 0 in the lazy limit at fixed target roughness: the tail lift IS the feature learning, and its exponent
is set jointly by the target's roughness in the kernel basis (bY0) and the richness of the parametrisation.
The CKR estimate holds in every regime (late ratio 0.84-1.05), i.e. the description "kernel regression with the
current kernel" is uniform across lazy and rich training; what changes between regimes is only how far the
kernel's spectral law flattens.
The 'saturation' of aT near 2.3 seen at s=1 (section 9) is therefore a property of the standard
parametrisation, not a universal floor: the rich run flattens further (aT = 2.2 at Delta=1.0, 3.2 at Delta=0.5).
RMT check across regimes (rmt_check.py on sweep G): lazy (s=16) dK is a small change aligned with the top of K0
(tr dK/tr K0 = 0.2-0.3, random rotation over-predicts the tail 5-20x); rich (s=0.25) dK is large
(tr dK/tr K0 = 10, eff. rank 34) and randomly rotating it reproduces the tail within 1.3-4x to rank 200:
the free-convolution-like tail lift is a feature-learning phenomenon.

## 11b. Rich regime across depth (sweep H, s=0.25, T=48000; lazy s=16 at depth 3 for contrast)
  depth 1 rich:  Delta=0.5 p=3.44 aT=3.7 ||E(T)||=0.085 (0.25 at s=1);  Delta=1.0 p=4.35 aT=2.8 ||E(T)||=0.24 (0.79 at s=1)
  depth 2 rich:  Delta=0.1 p=0.20 (smooth target: still no lift even when rich), ||E(T)||=0.0026
  depth 3 lazy:  Delta=0.5 p=0.63 ||E(T)||=0.49;  Delta=1.0 p=0.54 ||E(T)||=0.95  (no fit without the lift)
So: (i) richness raises p at every depth, including depth 1, which in the standard parametrisation barely
learns rough targets; (ii) the lift still needs a rough target: smooth targets lift nothing in any regime;
(iii) without the lift (lazy) rough targets are not fitted at any depth. The lift is necessary and sufficient
for late-stage progress on rough targets, and it is what the CKR estimate tracks.

## 12. Synthesis: the late-stage model that the data support
State variable: the current kernel K(t) (600 x 600 on the training set), nothing frozen.

(a) Error from kernel (works, 86 runs incl. lazy/rich, median ratio 0.94, 5-95 % 0.64-1.12 over all times):
      ||E(t)||^2 ~= sum_k <v_k(t), E0>^2 exp(-2 lam_k(t) t)        [current-kernel regression, CKR]
    i.e. training looks, at every t, like kernel regression for time t with the kernel it has now. The hard
    cutoff version (energy of E0 on current modes with lam_k(t) t < 1) is equally good. Anything that fixes a
    basis at an earlier time (initial kernel, or kernel frozen at T/6, or final shape with a running amplitude)
    fails by 2-20x for rough targets.

(b) Kernel from features (mechanism, exact decomposition + sweep C):
      K(t) = K0 + X(t) + Q(t),   Q = dPsi dPsi^T / n,   X = (Psi0 dPsi^T + dPsi Psi0^T)/n
    The tail of K(t) is Q(t) (K0+Q matches K(T) at every rank; K0+X does not lift anything).
    Q is the Gram matrix of the feature displacement; Prop. 16 of the paper says dPsi is built from the
    second-order features along g_E, i.e. from H2 and the residual path. Its size grows as ell^2 (path length)
    at moderate rank and super-quadratically deeper in the tail; its eigenvectors keep rotating (section 8).

(c) Spectrum from roughness (the architecture/roughness link):
    - rank <-> roughness of eigenvectors is a fixed map (same for K0, K(T), Q(T), all architectures): the kernel
      spectrum IS a law lam(rho) relating eigenvalue to feature roughness.
    - At init lam(rho) is a steep law common to all widths/depths (a0 = 5.2-7.2 in rank), shifted up by width
      and depth by < 1 decade (finite-width tail, the RMT part).
    - Training flattens it: lam_k(T)/lam_k(0) ~ k^p, with p a monotone, saturating function of the target's
      roughness measured in the kernel's own basis (bY0), multiplied by the richness of the parametrisation
      (p -> 0.5 lazy, 2-3.6 standard, 2.6-4.4 rich); p is nearly independent of width and depth at fixed Delta.
    - For rough targets the lifted tail is reproduced by a randomly rotated dK (free-convolution-like), i.e.
      Q's eigenbasis is generic with respect to K0's tail; for smooth targets / lazy training dK is aligned
      with the top of K0 and nothing reaches the tail.

(d) Where depth and width enter: not through p, but through (i) the initial tail level, (ii) how fast the
    kernel re-orders its basis towards the target (bYT - bY0 grows with depth), and (iii) the step count the
    dynamics need (lam1 grows 2-30x). Deeper nets reach the same lifted spectral law sooner and fit the rough
    residual faster (alpha grows with depth), but the late-stage spectral law they end up with is the same.

What is NOT closed: a formula for p (or for Q's spectrum) from (target, architecture) alone. The data say it is
a function of bY0 and richness with a saturating shape; the mechanism says Q = dPsi dPsi^T/n with dPsi driven by
the rough residual through the second-order features. A reduced model would need the roughness spectrum of the
second-order features nabla^2 N [g_E] as a function of the residual's roughness -- that is the next measurement.

## 13. Next steps (in order of expected value)
1. Measure the roughness spectrum of the second-order features along g_E (the columns of Psi_{g_E}) and test
   whether the per-rank growth rate d lam_k/dt is proportional to the energy of those features at rank k times
   ||E||. This would turn (b)+(c) into a closed ODE for the spectrum, with the CKR giving ||E||.
2. Replace Delta by the actual bY0 (target decay in the kernel basis) as the roughness variable in the paper;
   it collapses the Delta=0.1..2.0 data and is defined for any dataset.
3. Test the CKR and the lift law on a second dataset/activation (the code takes any (X, Y); activation is tanh
   in both the numpy and torch scripts and would need the derivative tables changed).
4. For the paper's section on held rough error: the right projector is the tail of the CURRENT kernel; the
   leak then is the inter-mode transfer measured in section (A) of analyze_late (transfer fraction per log
   interval 0.5-3 %), not lam1 ||E_s||^2.

## 14. Files
code/   l63test.py (numpy original, patched for H/log_every/outdir), l63test_torch.py (GPU port, OUTSCALE),
        l63_late.py (diagnostics; env L63_NOGRAPH, L63_QUAD, L63_DATASEED, L63_OUTSCALE), run_sweep_gpu.sh,
        analyze_late.py (per-run figures + summary.csv), cross_runs.py, arch_summary.py, rmt_check.py,
        tail_exponents.py, tail_mass.py, quad_check.py, q_shape.py, ckr_shape.py, lift_exponent.py,
        rough_law.py, final_figs.py, compare_np_torch.py, gpu_timing.py.
runs/   test1 (original 4 runs), sweepA (36: H x depth x Delta), sweepB (seeds, n=400, H=512, Delta=2),
        sweepC (8, with Q/X snapshots), sweepD/D2 (T=10000/5000), sweepE (data seeds, n=100, H=256 Delta=2),
        sweepF (Delta 0.25/0.75/1.5), sweepG (output scale), sweepH/H2 (rich/lazy across depth).
        Each late_*.npy holds: t, E, lam (all 600 eigenvalues per log), eE2/yV2/e0V2 (energies per current mode),
        rough_E, rough_top, drive_e, H2_g, gnorm, Evec, pathlen, K snapshots (30, float32), top-20 eigvecs,
        and for sweep C the Q and X snapshots.
figures figA/*.png (per run + cross-run), figB, figC, figD, figG, figFinal/final_summary.png.
  depth 3 rich:  Delta=0.5 p=2.19 aT=3.4 ||E(T)||=0.0033;  Delta=1.0 p=3.53 aT=2.06 ||E(T)||=0.0027
  depth 4 rich:  Delta=0.5 p=1.96 aT=3.5 ||E(T)||=0.0023
  (standard s=1 at the same depths: p=1.8/1.6 (Delta=0.5), 3.2 (Delta=1.0); ||E(T)||=0.018/0.012, 0.035)
  Rich training at depth 3-4 raises p only modestly over s=1 but lowers the final error 5-13x: at matched
  kernel time the richer dynamics spend their path on re-ordering the basis towards the target (bYT 3.3-3.5
  vs 3.0-3.4) rather than on a flatter law. The spectral law at depth >= 3 is already nearly the same for
  s=0.25 and s=1; depth 1-2 is where richness changes p the most.

## 15. The second-order features (sweep S, L63_SOF=1; sof_check.py, sof_figs.py, figS/)
Exact per-mode rate (Hellmann-Feynman, verified: predicted/finite-difference = 1.00, corr 1.000 over ranks):
    d lam_k/dt = -2 |g_E| <v_k, Psi_g Psi^T v_k>/n = 2 |g_E| sqrt(lam_k) sqrt(q_k) (-cos_k),
    q_k = ||v_k^T Psi_g||^2 / n  (energy of the second-order features along g_E on current mode k, q_k <= H2^2),
    cos_k = alignment of that second-order feature with the mode's own feature direction Psi^T v_k.
Measured structure (H=96; standard and rich):
- At t=0 the second-order features are SMOOTH: energy-weighted mean rank of q is 6, fraction beyond rank 100 is
  4e-4 to 8e-4, for every run; the residual sits at mean rank 120-160.
- They roughen as the residual roughens, lagging behind it: mean rank of q at T is 31 (Delta=0.5 d2), 23 (d3),
  92 (Delta=1.0 d2), 72 (d3), 48 (rich Delta=0.5), 138 (rich Delta=1.0); tail fraction (k>100) 0.05-0.07
  (Delta=0.5), 0.25-0.32 (Delta=1.0), 0.16 / 0.52 (rich). The residual's mean rank at T is 330-570.
  So the lift of the tail is fed by second-order features that are much rougher than the initial features
  (Prop. 16's mechanism) but still smoother than the residual: the kernel chases the residual's roughness
  from above, which is why lam_k rises as a power of k rather than jumping to the residual's modes.
- The alignment is not small: median -cos_k = 0.2-0.5 for k <= 100 and 0.1-0.25 for k > 100 during the
  active phase, decaying to 0.05-0.15 at the end. The 1e-4 'drive/bound' ratio of the first experiment came
  from the bound's slack (|g| <= lam1^{1/2}||E|| and q_k <= H2^2 are both loose by 10-100x on the tail),
  not from a small cosine.
- By-rank integration of the exact rate (figS, panel 4) reproduces lam_k(t) for ranks 10-400 over the whole
  run in the Delta=1.0 depth-2 case inspected (dotted on solid); the q_k/H2^2 spectrum flattens from 10^-12
  at rank 400 (t=0) to 10^-2 (t=T), i.e. the roughness supply itself spreads over the whole spectrum.
Closure proposal: the data support  d sqrt(lam_k)/d ell = sqrt(q_k) (-cos_k)  with ell the path length, and
q_k a roughness spectrum that is slaved to (and smoother than) the residual's; modelling q_k(t) from the
residual spectrum eE2_k(t) is the remaining step to a closed late-stage model.

# Part II (2026-10-10): design comparison from initialization / a few steps  (job 59642585, 1 node, 4 GPUs)

## 16. Offline ranking tests on the 92 saved runs (rank_test.py, rank_fewstep.py)
Question: can a score computed WITHOUT training to T rank designs (width, depth, output scale) by ||E(T)||?
- Init only (K0 basis, target decomposition): lazy score and lifted score rank width x depth at fixed rough
  target with Spearman 0.7-0.9 (T=300..3000), the init tail lam_100(0) alone 0.6-0.8; for the smooth target
  the lazy score fails (rho ~ 0) but lam_100(0) still ranks (0.7). Init-only scores are BLIND to the output
  scale (identical score for s=0.25..16 while E(T) spans 0.007-0.7).
- Few steps: freeze the measured kernel at kernel time tau_f = lam1(0) t_f and score
  S_f = sum_k eE2_k(t_f) exp(-2 lam_k(t_f)(T - t_f)). Across all 28 designs per target INCLUDING scale:
    tau_f:        0.1   1    3    10   30   100  300  1000
    Delta=0.5:   0.60 0.65 0.70 0.84 0.94 0.96 0.93 0.90
    Delta=1.0:   0.42 0.49 0.56 0.64 0.89 0.95 0.94 0.90
  tau_f = 30-100 is 75-250 GD steps at eta = 0.4/lam1, i.e. ~1 % of the budget. The scale (richness) becomes
  visible in the kernel between tau = 10 and 30, which is when the smooth part of the error has been removed
  and the tail starts to lift. Before that, nothing at initialization distinguishes lazy from rich.

## 17. Layer sweep (runs/layers, runs/layers2; layer_analysis.py, spike_to_kernel.py)
- Tail attribution: the hidden-layer kernel blocks carry 55-75 % of sum_{k>100} lam_k and gain share over
  training (depth 3: layer 2's share 0.25 -> 0.46); the input layer's share falls (0.25 -> 0.07-0.15).
- Weight changes are low rank (effective rank of dW_l 2-8); the top singular value of W_l rises from the
  Marchenko-Pastur edge (ratio 0.9-1.0 at init) to 3-9x the edge; the Hill tail exponent of the hidden W_l
  spectra falls from 3.5 to 2.1-2.6 (1.7-1.8 rich) as the tail lifts.
- The top spike direction's feature (linear projection, output pre-activation, and tanh of it) stays SMOOTH
  (roughness 0.05-0.1): the roughness does not come from a single rough spike feature.
- LINK TO THE KERNEL: whole-kernel lift lam_100(t)/lam_100(0) vs hidden spike amplitude a = top sv of dW_l,
  over 43 run-snapshot points: log-log slope 2.26 (Spearman 0.86); Delta=0.5: 1.90, Delta=1.0: 2.70.
  Per layer block: slopes 2.2-3.4. Hill alpha: Spearman -0.86 with the lift. Mechanism predicts 2 (Q quadratic
  in the displacement); the excess at Delta=1.0 is the saturation of tanh' as amplitudes pass the bulk edge.
  So: spike amplitude (weight-side, HT-SR-type statistic) -> tail lift ~ a^(2..2.7) -> lam_T(rho) -> error at
  budget. The first link, a law for a_l(t), is the open modelling problem.

## 18. Are the spikes beneficial? (layer_benefit.py)
Exact: dW_l = -int dL/dW_l dt = -(1/n) sum_i int delta_l(x_i) a_l(x_i)^T dt (residual-aligned by construction);
d||E||^2/dt = -2 sum_l <E, K^(l) E>, each term >= 0, so the training benefit of each layer's change is exact and
additive. Measured (H=96, depths 2-4, Delta 0.5/1.0, s=1 and 0.25):
- hidden-layer blocks carry 35-65 % of <E,KE> at all stages; input layer 5-30 %; output layer 8-45 %.
- the share of <E,KE> on modes k>100 rises from 0 (tau<=100) to 1.00 (tau>=1e4) in every run: ALL late-stage
  training descent goes through the lifted tail (lazy runs, which do not lift it, stall).
- validation: d(E_val.E_val)/dt = -2 <E_val, K_vt E> (also additive over layers). Normalised, it is +0.03..+0.08
  while smooth modes are fitted and 0.000..-0.003 once the tail share reaches 1: the lifted rough features are
  localised at the training points on this dataset (n=200). Benefit for the loss we care about is governed by
  the sign of this cross-kernel inner product, which the theory does not fix and the data can measure.

## 19. Init-only test of the static spike theory (virtual_spike.py)  -- NEGATIVE
At initialization, add a spike of amplitude a to the weights along the direction the rough part of E0 drives
(gradient of P_{k>100} E0), rank-1 per layer or the full multi-layer direction, and read lam_100(K)/lam_100(K0):
  H96 d2 Delta=0.5: rank-1 spike, a=4: 1.5 / 2.9 / 3.1 (layers 0/1/2); a=8: 2.1 / 2.9 / 5.3.
                    full rough direction a=4: 7.2, a=8: 38;  smooth direction a=4: 6.8, a=8: 26.
  H96 d3 Delta=0.5: rank-1 a=8: 1.7 / 2.3 / 3.7 / 6.4; full rough a=8: 43.
Training reached lifts of 150-300 at hidden spike amplitudes 3-6 (section 17). A static spike on the initial
weights at matched amplitude under-predicts the lift by 10-50x. The lift is a property of the trained path, not
of a perturbation of the initial network; this is Prop. 20 again (no expansion about theta_0 reaches the rough
stretch). Control in progress (dw_probe.py): the trained dW itself, scaled and rank-truncated.

## 20. What the trained change is (dw_probe.py, H96 d2 Delta=0.5, T=3000, full lift_100 = 337)
- rank truncation of each layer's dW: r=1 -> 29, r=2 -> 129, r=4 -> 211, r=8 -> 299, r=16 -> 328: the end state
  IS low rank (4 spikes/layer = 63 % of the lift, 8 = 89 %), but a single spike is far from enough.
- static scaling alpha*dW: alpha=0.1/0.25/0.5/0.75/1 -> 2.7/12/57/160/337; at matched amplitude this
  reproduces the lift along the training path (amp 4.2 -> 57 vs 52 measured). The DIRECTION of the trained change
  carries the lift; the same amplitude along the init rough-gradient direction gives 7 (section 19).
- layers: hidden-only dW -> 27, output-only -> 26, input-only -> 5, input+hidden -> 40, all -> 337: the kernel's
  response to the spikes is multiplicative across layers (composition), not additive.
- biases only: 2.2.
Dynamic test (lowrank_train.py): GD with each layer gradient truncated to rank r per step tracks full GD's lift
and loss for r >= 2 at early times (r=4: lift 8.8 vs ~10 at t=24; r=1 lags); late-time comparison in progress.
Conclusion so far: the spike theory is right about the state (few directions per layer) and about the dynamics
living in those directions; the init-only prediction fails because the spike DIRECTIONS are set along the path by
the sequence of residuals, and the kernel response to them is a product over layers. A theory of the rich regime
from initialization has to predict those directions (or their statistics relative to K0's tail), not amplitudes.

## 21. Where the roughness enters (runs/layers2, factor roughness per hidden layer)
Energy-weighted graph-Laplacian roughness of the layer factors (kernel features are delta_l (x) a_l):
  activations a_l: 0.06-0.12 at all times (smooth);  tanh'(z_l): 0.025-0.09 (0.29 in the rich Delta=1 run);
  backprop factor delta_l: 0.025 at init -> 0.15-0.67 at T, largest at the INPUT layer (delta_0 = 0.18-0.67),
  i.e. the longest downstream product. Delta=1.0 > Delta=0.5; rich > standard; deeper -> rougher delta_0.
The roughness that lifts the kernel is produced by the composition of downstream derivative factors and weights
along the backward pass, not by any one layer's features: consistent with the multiplicative per-layer lift of
section 20. A per-layer statistical theory must model the backward product, and roughness compounds with depth.
Second case (H96 d3 Delta=1.0, lift_100 = 2948): rank truncation r=1/2/4/8/16 -> 50/511/2032/2695/2806
(69 % at r=4, 91 % at r=8, 1.7 % at r=1); scaled change alpha=0.1..1 -> 4.8/56/490/1475/2948 (~alpha^3.3);
hidden-only 127, output-only 47, input-only 7, input+hidden 198, all 2948. Same structure as the first case.

## 22. Can the spike directions be known at initialization? (spike_directions.py) -- NO
Final dW_l of two trained cases (H96 d2 Delta=0.5; H96 d3 Delta=1.0) expressed in the basis of INITIAL per-mode
gradient features G_k = Psi_l^T v_k (k = K0 modes):
- overlap^2 with the initial full-gradient direction: <= 0.005 in every layer.
- fraction of dW_l in the span of G_1..G_m: m=10: 0.00-0.03; m=50: 0.03-0.28; m=100: 0.06-0.54; m=300: 0.16-1.00.
- hidden layers (H x H): only 0.55-0.66 (depth 2) and 0.25-0.56 (depth 3) of dW_l lies in the span of ALL 600
  initial gradient features; the rest is in directions invisible to K0, created along the path. (Input/output
  layers have < 600 parameters, so their span is trivially complete; even there ~300 modes are needed.)
The spike directions are path-generated, mostly in tail-mode and K0-invisible subspaces. Together with sections
19-21: the dynamic low-rank (spike) description of training is correct, but its directions are not functions of
initialization-time quantities in any low-dimensional way. An init-only theory of the rich regime through spike
directions is not available on this problem (Prop. 20 in weight space).

# Part III (2026-10-10, job 59645654, 36 min of 2 nodes): the init-only reduced model and its blind test

## 23. The model (code/reduced_model2.py, DRIVE=1; closures in code/closures.npz)
Inputs, all from one Jacobian at initialization on the n=200 training points: K0 spectrum lam0_k and eigenvectors;
E0 energies c0_k; per-layer per-mode energies q_lk = v_k^T K^(l) v_k (code/init_blocks.py); H2 at init.
State: c_k (residual energy per initial mode), J_k (integrated residual per mode), lam_x,k (drive-accumulated
eigenvalue per mode). Derived each step:
  P = prod_{hidden l} A_l,  A_l^2 = top-16 sum_k q_lk J_k^2            (rank-16 amplitude of dW_l; exact up to
                                                                       orthogonality of per-mode features)
  lift_k = 1 + exp(b0 + b1 log P + b2 log(k/100) + b3 (depth-2) + b4 log frough)   (fitted curve, 32 runs)
  lam_k = lam0_k lift_k ;  dJ_k/dt = sqrt(c_k lift_k)
  drive: |g| = sqrt(sum_k (lam_k+lam_x,k) c_k); lam_eff = sum (lam+lam_x) c / sum c;
         rate = 2 |g| sqrt(lam_eff) sqrt(r_q) H2 cos_e,  r_q = min(0.5, 4e-4 (P/0.1)^1.2),  cos_e = 0.25
         d lam_x,k/dt = rate * c_k / sum c                            (rank-one kernel growth along the residual)
  dc_k/dt = -2 (lam_k + lam_x,k) c_k ;   ||E(t)|| = sqrt(sum_k c_k)
Constants: b (lift curve) fitted on 32 spike-logging runs; r_q and cos_e read off the second-order-feature
measurements (sections 15, 21); nothing fitted to errors. Cost per design: one Jacobian + seconds of numpy.

## 24. Results
In-sample (47 designs whose runs entered the lift fit), fixed basis: Spearman 0.92 all scales, 0.95-0.98
within target (s=1); level ~10-17x high. Random-rotation (Lorentzian) flow for the residual: WORSE (held-out
Spearman 0.38, level 37x): the residual's basis rotation is anisotropic (toward the residual), not random.
BLIND held-out (28 designs: seeds 1-2, widths 48/160, depths 2-3, Delta 0.5/0.75/1.0, s=1 and 0.5), drive closure:
   Delta=0.5 (8):  Spearman 0.93 (0.3T: 0.93), median pred/meas 2.8
   Delta=0.75 (12, never seen): 0.99 (0.97), 1.6
   Delta=1.0 (8):  0.93 (0.98), 0.9
   all 28: 0.82 (0.84), median 1.5, 10-90 % 0.73-3.2
Fixed-basis model on the same held-out set: Spearman 0.90/0.91/0.98 per target, level 19x.
Predicted rank-16 amplitudes vs measured (in-sample, no fitting): within 1.5-2.5x (standard, lazy), 2-3x high (rich).

## 25. Status and what remains
Established: an initialization-only model, with every design-specific input derived from the E-dot identities
and two statistical closures, predicts the final training error of unseen designs to within a factor ~1.5-3 and
ranks them within a target at Spearman 0.93-0.99 on this problem (Lorenz-63, tanh MLP, GD, squared loss).
Untested dimensions, each a small run: another dataset; Adam (may change the spike structure, not only the
amplitude law); other losses (enter through grad l in the amplitude law). Known approximations: per-mode
feature orthogonality (rank of dW over-predicted), global lift curve (factor ~3.7 scatter per snapshot),
cos_e constant.

# Part IV: the model that works -- piecewise-linearised gradient flow along the gradient chain (code/reduced_model4.py)

## 26. Coherence (code/coherence_test.py, runs/probe)
Spikes of adjacent layers are coherent because every layer is driven by the same residual through the same backward
chain. Test on two trained runs: the kernel lift at rank 100 is 337 (depth 2, Delta 0.5) / 2948 (depth 3, Delta 1.0)
for the trained dW; 62 / 177 when each layer's spikes are randomly rotated (same per-layer spectra, coherence
destroyed); 309 / 902 for the coherent gradient-generated chain dW_l = -sum_k J_k Psi_l^T v_k/sqrt(n) at the same
amplitude, 245 / 898 with 16 modes only. The kernel-relevant direction is the gradient chain; it is computable
from initialization. (Section 22's "directions invisible to K0" measured parameter-space overlap, which is not the
quantity that matters for the kernel.) Statistical closures for spec(Q) (section 25 attempts: regression, universal
roughness distribution) are unnecessary and were inferior.

## 27. Model v4
State: residual vector E, accumulated weight change dtheta. From the current Jacobian Psi (features) and K = Psi Psi^T/n
with eigenpairs (lam_k, v_k): between re-linearisations, a_k = v_k.E/sqrt(n) decays exactly as a_k e^{-lam_k dt} and
dtheta accumulates -Psi^T int E dt / n (closed form per step). When ||dtheta - dtheta_last|| > delta (delta = 0.25),
re-evaluate Psi and the true output error at theta_0 + dtheta (one Jacobian) and re-expand. No fitted inputs.
Cost: 70-170 Jacobians per design (13-15 s on one A100), i.e. 10^4 times fewer feature evaluations than training.
References: H96 d2 D0.5: E(T) 0.0487 vs 0.0393 measured, lift_100 338 vs 338; H96 d3 D1.0: 0.0333 vs 0.0346, 3542 vs 3444.

## 28. Blind test (28 held-out designs: seeds 1-2, widths 48/160, depths 2-3, Delta 0.5/0.75/1.0, s=1 and 0.5)
E(T) predicted/measured: median 1.04, 10-90 % 0.87-1.34, extremes 0.68-1.65. Spearman 0.95 pooled
(0.90 / 0.97 / 0.93 per target). Spectrum gain at rank 100: median 1.01 (0.90-1.08). Pairwise design decisions:
94 % of 108 pairs, 100 % of the 92 pairs whose true errors differ by more than 1.3x. Seed-to-seed spread of the
true error for a fixed design: median 1.17x -- the model is at the noise floor. Per-design table: runs/v4/.
Scope validated: Lorenz-63 one-step map, n=200, tanh MLPs, gradient descent, squared loss, training error.
Untested: other datasets, optimizers (Adam enters through the update rule; the scheme generalises directly),
other losses (grad l replaces E in the accumulated gradient), validation error (excluded by choice).

# Part V: the statistical (unit-averaged) model and where it breaks (code/stat_model.py, spiked_ntk.py, coherence_test.py)

## 29. Facts established on the way
- Response law at initialization along the coherent direction: the exponent of lam_k vs amplitude is linear in
  log(lam1/lam_k(0)) for every rank (residual 0.04-0.16 over 380 ranks, 14 designs), slope 0.13-0.18, as the Hermite
  picture predicts; the onset is linear in the same variable. Coefficients vary +-25 % across designs.
- A scalar-amplitude closure (one amplitude per layer driving the whole spectrum) is bistable (starves or runs
  away) even with each design's own measured coefficients: the response must be local in roughness (a front).
- Integrated residual with INITIAL features produces no lift (1.6x vs 338-3444): the rough displacement is made by
  features that exist only after earlier lift. Any propagation from initialization must be staged (Prop. 20).
- Unit-averaged Gaussian recursion (2-point tables + coherent shift covariance): forward shift covariance exact
  (ratio 1.00), forward activation tables 0.4-1.0 of exact, derivative covariance 0.7-1.2 of exact, kernel lift
  within ~2x to rank 200; but the backward coherent term (dW_{l+1} carried through the backward pass) is 1-9 % of
  exact in the input layer, and the recursion's own tail is indefinite beyond rank ~300.
- Decomposition on the exact shifted network: the backward change is 1.06-1.43 x the dW-in-backward part; it is a
  4-point statistic of a single unit's preactivation process (tanh' tanh' tanh tanh at four points), which 2-point
  Gaussian tables factorise away.

## 30. Conclusion of Part V
The MP/Gaussian abstraction of the units is correct and sufficient for everything 2-point: initial spectrum,
rank-roughness map, forward shift variance, and the free-convolution lift of the generic part. The coherent
composition of spikes through the layers -- the mechanism that creates rough features -- is a 4-point statistic of the
unit ensemble and is not captured by 2-point tables. Faithful statistical propagation therefore needs either the
4-point Gaussian integrals of the activation (a derivation not attempted here) or a sampled unit ensemble, which is a
random network of the same architecture propagated in stages (the v4 scheme with generated rather than stored weights).
Validated deliverables remain: v4 (blind 1.04x median, Spearman 0.95), the init-only ranking (0.9), the mechanism.

# Part VI: SegFlow -- a training method built on the Taylor radius (code/segflow.py, segflow_l63.py, segflow_generic_test.py)

## 31. The idea

Full-batch GD on the one-step map takes 0.3M-5M steps because its step size is pinned by lambda_1 while the residual sits
in modes with lambda_k/lambda_1 ~ 1e-4..1e-5 (Sections 3, 7). Inside the Taylor radius the residual dynamics are linear,
r(t) = exp(-K t) r_i, and can be integrated exactly for any t. SegFlow does exactly that, matrix-free:

1. At theta_i run m Lanczos iterations on K = J J^T/n started from the residual r_i (one Jacobian-vector product and one
   vector-Jacobian product per iteration; K is never formed). Because the residual lives in a narrow band of the
   spectrum at any time -- it migrates from the large eigenvalues to the small ones as training proceeds -- m = 30 vectors
   represent exp(-Kt) r_i to 1e-14 (checked against the exact 600x600 kernel at init for t*lambda_1 up to 300).
2. In the Ritz basis, r(t) = exp(-K t) r_i and dtheta(t) = -J^T phi(Kt) t r_i / n with phi(z) = (1-e^{-z})/z; the
   displacement norm is exact in the Krylov space (||J^T Q w||^2 = n w^T T w).
3. Take the largest t with ||dtheta(t)|| <= delta (the trust radius = Taylor radius), one more vjp for dtheta, one forward
   for the new residual.
4. Adapt delta by the standard actual/predicted-decrease ratio rho (grow x1.6 if rho > 0.75, halve if rho < 0.25,
   reject if the loss increases). No learning rate anywhere.

Section 30 of the previous part showed why the *direction* cannot be improved inside the radius: at a fixed
displacement the flow step, the Levenberg-Marquardt step and the Krylov-optimal step differ by < 0.2% in residual.
The whole gain is in step length: one radius-0.25 segment is worth 7 GD steps at init and 300-600 GD steps by
t ~ 10-30, and the Taylor series in the weights diverges beyond ~15% of the trained path (Section 30), so
re-linearising is the only way to go further -- which is what the segments do.

## 32. Results on the L63 designs (login-node CPU, 8 threads; GD numbers are the saved GPU runs of Section 28)

Stopping rule: run until the training error reaches the GD run's E(T). "passes ratio" counts one Lanczos iteration
(jvp + vjp) as one GD step (forward + backward), which is slightly unfavourable to SegFlow.

| design | GD steps | GD wall (GPU) | GD E(T) | SegFlow E | Lanczos iters (jvp+vjp pairs) | segments | SegFlow wall (CPU, 8 thr) | passes ratio GD/SegFlow | val E GD | val E SegFlow |
|---|---|---|---|---|---|---|---|---|---|---|
| H160_D0.5_depth2_s1 | 953194 | 278 s | 0.0500 | 0.0489 | 2940 | 98 | 11 s | 324x | 0.538 | 0.483 |
| H160_D1.0_depth3_s1 | 2378161 | 1023 s | 0.0244 | 0.0242 | 17670 | 589 | 87 s | 135x | 1.417 | 1.428 |
| H48_D0.5_depth2_s1 | 327210 | 181 s | 0.0535 | 0.0524 | 1770 | 59 | 30 s | 185x | 0.495 | 0.316 |
| H48_D0.5_depth3_s1 | 701669 | 497 s | 0.0219 | 0.0216 | 1830 | 61 | 6 s | 383x | 0.294 | 0.303 |
| H48_D1.0_depth2_s1 | 389563 | 219 s | 0.1544 | 0.1534 | 8670 | 289 | 20 s | 45x | 1.300 | 1.397 |
| H48_D1.0_depth3_s1 | 701216 | 363 s | 0.0687 | 0.0686 | 4800 | 160 | 15 s | 146x | 1.433 | 1.548 |
| H96_D0.75_depth3_s1_os0.5 | 3881845 | 1344 s | 0.0078 | 0.0077 | 13680 | 456 | 61 s | 284x | 1.163 | 1.140 |

Krylov size: m = 30 / 60 / 100 on H48 D0.5 depth 2 needed 1770 / 2760 / 4500 pairs for the same target (segments
59 / 46 / 45): the segment count barely depends on m, so the smallest m that still represents the residual wins.
Filter: with the Levenberg-Marquardt filter 1/(lambda + 1/t) in place of the flow filter, the same design reached the
target in 1500 pairs (50 segments) with validation error 0.391 vs 0.316 (flow) vs 0.495 (GD) -- one design, not a
conclusion.
Tracking: at equal flow time t SegFlow's training error is 1.0-1.3x GD's (median over the path), i.e. the segments
follow the GD trajectory, they do not jump to the Gauss-Newton solution; validation errors are within +-10% of GD's
in all 7 designs (better in 3).

## 33. Generality (code/segflow_generic_test.py; equal pass budgets, no tuning of SegFlow)

| problem | P | passes | SegFlow | GD (eta = 0.4/lambda_1) | Adam (lr 1e-3) |
|---|---|---|---|---|---|
| GELU MLP 1->64->64->64->1, rough 1-D target (sin + steps + kink), n = 256 | 8513 | 3000 | 0.0883 (111 segments) | 0.162 | 0.0904 |
| tanh CNN (2 conv + 2 linear), 8x8 inputs -> scalar, n = 200 | 17113 | 2000 | 0.00000 (14 segments, exact fit) | 0.0494 | 0.00028 |

The optimizer takes a torch.func-differentiable map theta -> outputs and targets; nothing in it is specific to L63 or
to tanh MLPs. Loss: squared error only (the exact linear flow needs a quadratic loss; a generalised Gauss-Newton
version for other losses would replace K by J H_loss J^T/n).

## 34. Prior work (Exa search, 2026-10-10)

* Trust-region Gauss-Newton with Krylov solves for neural-network least squares: Mizutani & Demmel, NeurIPS 2000
  ("Krylov-dogleg" steps, CG on the normal equations, matrix-free). Same cost structure as SegFlow; dogleg/LM step
  instead of the flow step; no notion of following the GD path.
* Hessian-free optimisation (Martens, ICML 2010), Krylov Subspace Descent (Vinyals & Povey 2011), subsampled
  Gauss-Newton/natural gradient with LM damping (Ren & Goldfarb 2019), K-FAC and Practical Gauss-Newton (Botev et al.
  2017): the matrix-free second-order lineage. Reported gains are per-step; wall-clock and minibatch noise are why
  they lost to Adam at scale ("Why Transformers need Adam", NeurIPS 2024).
* Gram-Gauss-Newton (Cai et al. 2019): Gauss-Newton solved in the n x n NTK Gram space, kernel-ridge variant = LM,
  minibatch version, quadratic convergence for wide nets; learning rate fixed to 1, no trust region.  Function-space
  Gauss-Newton / natural gradient for PINNs (Jnini, Vella, Zeinhofer 2024; "dual natural gradient" 2025) solve the same
  n x n system for physics losses.
* Closest in spirit to the "residual migrates down the spectrum" design principle: Jiang et al., ICLR 2026
  (arXiv 2601.03162), "On the convergence behavior of preconditioned gradient descent toward the rich learning regime":
  Gauss-Newton/LM flatten the mode-wise rates (remove spectral bias) in the lazy regime, lose effectiveness in the
  rich regime, and generalise worse because they stick to the NTK subspace; they recommend GN first, Adam after.
  SegFlow differs in two ways that bear on exactly that criticism: (i) the step is exp(-Kt) along the gradient
  flow, parametrised by time and limited by the Taylor radius, so the iterate follows the GD path (tracking 1.0-1.3x
  above) instead of jumping to the Gauss-Newton point; (ii) the trust radius is re-set from the measured linearisation
  error every segment, so the method continues into the rich regime (all L63 designs here are rich: hidden-layer
  pre-activations move by 2.5 std) rather than stalling when the linear model breaks.
* Exponential integrators: the segment step is the exponential Rosenbrock-Euler step (Hochbruck & Ostermann 2005,
  2009; Krylov exp of Hochbruck & Lubich 1997; KIOPS 2018) applied to the function-space residual ODE with the
  Gauss-Newton Jacobian; the search found no prior use of this as a neural-network training step. Mechanically it is
  equivalent to an LM step at the same radius (Section 30), so the novelty is the time parametrisation and the
  radius rule, not the linear algebra.

## 35. Caveats

* Full batch only. Minibatch versions (GGN, SGN) exist in the literature but change the method.
* Cost per segment is 2m+2 passes; at n = 600 residual entries and m = 30 that is negligible against the 1e3-1e4 GD
  steps a late segment replaces, but for large n the Lanczos basis (n x m) must fit in memory and m may need to grow.
* The radius adaptation is the textbook rule; the second-derivative bound from the paper (along_dir_second, opnorm in
  l63test_torch.py) was not used. The rejected-step rate is 5-10%.
* Validation errors on this task are large for every method (0.3-1.5 RMS); the comparison says SegFlow does not
  generalise worse than GD here, not that either generalises well.

## 36. FLOP comparison with Adam (code/adam_l63.py, segflow_generic_thresholds.py) -- the honest optimizer verdict

FLOP model in forward-equivalents: forward 1, backward 2, forward-mode product 2, reverse-mode product 3, nested second
derivative 4. One Adam or GD step = 3; one Lanczos iteration = 5. Full-batch Adam, learning rates 1e-3..1e-1 swept,
best reported; target = the saved GD run's final training error.

| design | Adam best lr | Adam steps | Adam FLOP-eq | SegFlow FLOP-eq | GD FLOP-eq | Adam/SegFlow |
|---|---|---|---|---|---|---|
| H48 d2 D0.5 | 0.03 | 700 | 2100 | 9087 | 981630 | 0.23 |
| H48 d2 D1.0 | 0.01 | 1900 | 5700 | 44507 | 1168689 | 0.13 |
| H48 d3 D0.5 | 0.01 | 1300 | 3900 | 9395 | 2105007 | 0.42 |
| H48 d3 D1.0 | 0.01 | 1800 | 5400 | 24641 | 2103648 | 0.22 |
| H160 d2 D0.5 | 0.01 | 800 | 2400 | 15093 | 2859582 | 0.16 |
| H160 d3 D1.0 | 0.01 | 1700 | 5100 | 90707 | 7134483 | 0.06 |
| H96 d3 D0.75 s0.5 | 0.01 | 3100 | 9300 | 70225 | 11645535 | 0.13 |

Adam reaches the same training error with 2.4 to 18 times fewer FLOPs than SegFlow, and 200 to 1400 times fewer than
GD. On the two non-L63 problems (FLOPs to reach E0/4, E0/16, E0/64): GELU-MLP SegFlow 5.0k/20k/43k, Adam 0.3k/5.9k/20k,
GD 6.2k/>60k/>60k; tanh-CNN SegFlow 1.1k/1.4k/1.4k, Adam 0.3k/0.3k/0.45k, GD 5.9k/11k/>15k. Same picture.

Where the loss is (H48 d2 D0.5, measured): Adam takes 969 steps of mean parameter norm 0.035 (path length 33.7);
SegFlow takes 52 segments of mean norm 0.55, max 1.64 (path length 28.4). The segments are 15x larger than Adam's
steps but cost 50x more: 8850 of SegFlow's 9087 FLOP-eq are the 30 Lanczos iterations per segment. Smaller Krylov
spaces do not work: m = 8 needs 1120 segments with 218 rejections, m = 4 does not converge in 2000 segments -- the
residual re-excites the fast modes every segment and the exponential needs all active eigenvalue scales. Adam's
diagonal preconditioner costs nothing and persists across steps; SegFlow's exact function-space preconditioner is
rebuilt from scratch every segment and discarded. A recycled-Krylov variant (keep the top Ritz pairs and their
K-images across segments, add a few new Lanczos vectors) is the one lever that could reach parity; a first
implementation (segflow.py, recycle_k) was wrong and is left as unfinished.

Conclusion (supersedes the optimizer framing of Sections 31-33): SegFlow is not an optimizer contribution. Its value is that it reproduces the gradient-flow path at 1-2% of
GD's cost (tracking 1.4% with the H2 radius rule at kappa = 0.1), which Adam does not do, and that makes the
path-wise measurements below affordable. Taylor-radius steps beat GD because GD is crippled by stiffness; Adam already
removes most of that stiffness for free.

## 37. The H2 radius rule (theory -> optimizer)

Segment radius chosen a priori from the second directional derivative along the step: 1/2 delta^2 ||f''[d,d]|| =
kappa x (predicted residual decrease at that radius), solved in the Krylov space (one nested forward product per
segment). Rejected steps: 0 at kappa <= 0.3 (vs 5-11% for the trust-region rule). Cost is not reduced (kappa = 1:
2010 pairs vs 1770 trust-region) but fidelity to the gradient-flow path becomes a dial: kappa 0.1 / 0.3 / 1.0 give
tracking medians 1.014 / 1.05 / 1.16 with ranges [0.99,1.04] / [0.94,1.11] / [0.99,1.34]. The second-order
corrector (cancel 1/2 f''[dtheta,dtheta] inside the Krylov space) gave nothing (73 vs 59 segments), consistent with
Section 30: order two barely extends the radius for the depth-2 network.

## 38. The roughness theory measured along the path (code/segflow_rough.py, rough_analysis.py; runs/rough)

Measured at mid-segment points of the kappa = 0.1 path (segment-end points carry the fresh Taylor remainder, which
inflates <E,KE> by ~40x through a 0.3% energy fraction on the top eigenvalue -- that bias was the first result and is
avoided). Check: the finite-difference slope -d ln||E||^2/dt equals 2<E,KE>/||E||^2 within a few percent at every
checkpoint after the first, so the identities of Proposition "Energy of the rough part" hold on these paths and the
numbers below are on-path. P_m = graph-Laplacian high-pass above mode m (Section 4.1 of the theory), m = 10 here.

| design | c = <E_r,KE_r>/(sigma^2 ||E_r||^2): start / mid / end | c ~ E^p | drive realised / bound (median) | sigma_10^2 start -> end | lambda_1 start -> end | H2 range |
|---|---|---|---|---|---|---|
| H48 d2 D0.5 | 0.046 / 0.0025 / 0.0007 | 1.15 | 3.5e-5 | 0.99 -> 10.2 | 10.6 -> 72 | 0.6-2.2 |
| H160 d2 D0.5 | 0.061 / 0.0009 / 0.0005 | 1.56 | 7e-6 | 3.0 -> 15.0 | 32 -> 162 | 0.9-4.2 |
| H48 d3 D0.5 | 0.051 / 0.0009 / 0.0005 | 1.21 | 1.8e-5 | 1.0 -> 14.6 | 11 -> 87 | 1.2-6.2 |
| H48 d2 D1.0 | 0.019 / 0.0001 / 0.0002 | 2.36 | 1.6e-5 | 0.99 -> 39 | 10.6 -> 95 | 0.6-17 |
| H48 d3 D1.0 | 0.018 / 0.0001 / 0.0002 | 1.50 | 2.9e-5 | 1.0 -> 132 | 11 -> 209 | 1.0-58 |
| H96 d3 D0.75 s0.5 | 0.032 / 0.0001 / 0.00003 | 0.88 | 2.7e-6 | 0.55 -> 99 | 5.6 -> 157 | 1.2-45 |
| H160 d3 D1.0 | 0.022 / 0.0004 / 0.0008 | 0.78 | 8.8e-6 | 3.4 -> 150 | 38 -> 725 | 2.0-77 |

What this says about the closure of the theory:
* Assumption "efficient roughness supply" with c = O(1) fails by 1-3 orders: the kernel's Rayleigh quotient on the
  rough residual is 2-6% of the peak feature roughness at the start and 1e-4..1e-3 at the end, and c falls as a
  power 0.9-2.4 of the error. The roughness the features acquire sits mostly on rough directions that do not carry
  the residual (the residual migrates to ever smaller eigenvalues while sigma_R grows 10-200x).
* The a priori drive bound Theta_R <= sigma lambda_1^{1/2} H2 ||E|| is loose by ~1e5 (realised/bound 3e-6..4e-5).
  The reduced ignition model taken at the bounds therefore predicts nothing quantitative; with run-median constants
  (c, phi) it is off by orders of magnitude (constant c is the wrong closure).
* The exact identity does organise the curve. The effective eigenvalue the residual sits at,
  q(t) = <E,KE>/||E||^2 = -(1/2) d ln||E||^2/dt, follows q = q1 E^alpha after the lazy phase for the s = 1 designs:

| design | alpha | q1 | time to final error from the law / measured |
|---|---|---|---|
| H48 d2 D0.5 | 1.22 | 1.0e-2 | 2840 / 3090 |
| H160 d2 D0.5 | 1.38 | 1.7e-2 | 2660 / 2890 |
| H48 d3 D0.5 | 1.05 | 2.3e-2 | 2300 / 2880 |
| H48 d2 D1.0 | 0.40 | 1.2e-3 | 2650 / 3140 |
| H48 d3 D1.0 | 0.28 | 2.3e-3 | 1850 / 2690 |
| H96 d3 D0.75 s0.5 | 0.08 | 3.7e-3 | 1690 / 11900 (law fails: q not a power of E) |
| H160 d3 D1.0 | -0.50 | 2.2e-3 | 914 / 2820 (law fails: q rises as E falls) |

  The exponent is set by the target roughness (1.0-1.4 at Delta = 0.5, 0.3-0.4 at Delta = 1.0), the prefactor by
  architecture and roughness (depth 3 doubles q1 at either Delta; width 160 vs 48 x1.7; Delta 1.0 vs 0.5 /10). The
  frozen kernel never reaches these error levels at all (its q_f(E) is undefined below E/E0 ~ 0.4), which is the held
  rough error of the theory seen directly: everything below that level is bought with eigenvalue gain.
* The power law is not universal: it holds for the five s = 1 designs of width 48 and the width-160 depth-2 design,
  and fails for the rich-scaled s = 0.5 design (q nearly constant in E) and for the wide deep rough design
  H160 d3 D1.0, where q increases as E falls (the eigenvalue gain outruns the migration of the residual; lambda_1
  grows 19x and sigma_10^2 44x on that run). The exact identity holds on all seven; the closure q(E) does not.
* Not shown: that (alpha, q1) can be read from initialization. Seven designs at three roughness levels, two of them
  outside the law, are not enough; and q1 does not track sigma_10(0), H2(0)
  or lambda_1(0) in any simple way across the six (e.g. q1/H2(0) = 0.005, 0.005, 0.008 at Delta = 0.5 and
  0.0006, 0.0008 at Delta = 1.0).

Cost of these measurements: 100-360 s CPU per design on a login node (the path at 1-2% of GD plus ~100 passes per
checkpoint for sigma_m, c_m, lambda_1, H2). The 28-design held-out set would take ~1-2 CPU-hours.

## 39. Segments in Adam's metric (code/segflow.py, precond='adam') and why it does not close the gap

Segments integrate theta' = -P J^T r/n exactly in the linear model, P = diag(1/(sqrt(v_hat)+eps)) from an exponential
average of squared gradients at segment starts, normalised to unit mean; the kernel becomes J P J^T/n and the
displacement bound is exact in the P-metric. Result on H48 d2 D0.5: 8165 FLOP-eq at m = 30 (vs 9087 plain, vs Adam
2100-2900); m = 10 and m = 5 get worse (11.7k, 28.4k; 32 and 115 rejections). H48 d3 D1.0: 41.3k (vs 24.6k plain).

Diagnostics along a real Adam run (lr 0.01, H48 d2 D0.5), stiffness lambda_1/lambda_k of the plain kernel and of the
Adam-metric kernel, and the linearisation error of Adam's own steps:

| Adam step | E | K: l1/l101 | K_P: l1/l101 | K: l1/l301 | K_P: l1/l301 | Adam step norm | linearisation error of the step |
|---|---|---|---|---|---|---|---|
| 0 | 1.84 | 102251 | - | 1.2e9 | - | 0.52 | 3.2% |
| 50 | 1.11 | 15827 | 9789 | 5.8e7 | 3.5e7 | 0.11 | 0.9% |
| 200 | 0.27 | 1459 | 851 | 9.3e5 | 4.4e5 | 0.039 | 0.2% |
| 500 | 0.093 | 909 | 362 | 3.6e5 | 1.5e5 | 0.008 | 0.1% |
| 1000 | 0.052 | 683 | 241 | 2.4e5 | 8.6e4 | 0.004 | <0.1% |

Three facts. (i) Adam's diagonal improves the function-space conditioning by only 2-3x, so the number of active
eigenvalue scales -- and the Krylov size -- does not shrink; the premise of the Adam-metric segment is false here.
(ii) Adam's steps are far inside the Taylor radius (linearisation error 0.1-3%); its speed owes nothing to leaving the
linear model. (iii) The kernel's own stiffness collapses along Adam's path (l1/l101 from 1e5 to 1.5e3 in 200 steps,
to ~7e2 by 1000) -- the eigenvalue gain of Parts II-V happens early along Adam's path because Adam moves the weights
quickly from the start (step norm 0.5 at step 0 against GD's eta|g| ~ 0.08), whereas on the gradient-flow path the
same gain arrives late in time. Both paths have similar length (34 vs 28) and end with similarly lifted kernels; the
difference is FLOPs per unit path length: Adam ~86, SegFlow ~270 (30 Lanczos iterations x 5 per segment of ~0.55).

Conclusion: the only route below Adam's cost per unit displacement is memory across segments (reuse of the Krylov
information), since neither the exactness of the step, nor the direction, nor a diagonal metric changes the ~30 active
scales. That is Krylov recycling (unfinished here) and it can at best reach parity on these problems. Pending:
whether the comparison inverts at error targets 3-30x below GD's final error (code/deep_target.py), where a
parameter-diagonal preconditioner may stall on the rough tail.

## 40. Closing the optimizer line (negative results, recorded for the project log)

* Momentum across segments (extrapolate along the last accepted step, strength 0.6): H48 d2 D0.5 9087 -> 5083 FLOP-eq,
  H48 d3 D1.0 24641 -> 11705; zero rejected steps. Adam: 2100-2900 and 5400. Final gap: 1.8-2.4x.
* What Adam's speed is (ablation, steps to GD's final error, H48 d2 D0.5): Adam 700; heavy-ball momentum GD without
  normalisation 1650; RMSprop (normalisation without momentum) 12200; plain GD at the largest stable step 51500;
  sign descent and layer-normalised GD never. Momentum is the main ingredient (31x over GD), normalisation adds 2.4x on
  top of momentum and is harmful alone. Adam puts more displacement into the hidden layer (15 vs 9 for GD) and less
  into the output layer (4 vs 7.5).
* Deep tail (targets 3x, 10x, 30x below GD's final error): with learning-rate decay Adam stays 1.2-4x cheaper than
  SegFlow at every target on both designs (e.g. 193k vs 518k, 80k vs 329k FLOP-eq at E/30). Without decay Adam stalls
  at E/30; the inversion only exists against untuned Adam.
* Krylov recycling: every variant (stale K-images of recycled Ritz pairs; deflated Lanczos; fast-pairs only; with
  refresh) made segments 5-10x shorter and total cost 2-4x higher, because the recycled pairs are not accurate
  eigenpairs when the residual barely excites them.
* Why the Krylov size cannot shrink (measured at segment starts along the momentum run): from mid-training on, the
  residual's Ritz values above lambda_1/20 number 16-21 and hold 20-27% of the residual energy and 90-97% of <r,Kr>.
  Even with the five fastest directions removed, Krylov size 10 reproduces the segment to only 30-50% and size 20 to
  5-20%. The residual at a segment start spans ~20 eigenvalue scales within a factor 20 of lambda_1, where the trained
  kernel's spectrum is dense; "a narrow band plus a small fast transient" is not what the residual looks like. The
  30 matvecs per segment are intrinsic to an exact function-space step on this kernel.
* Verdict: no variant of the Taylor-radius step beats tuned Adam on FLOPs here. The method remains a faithful
  gradient-flow surrogate (Section 37) and nothing more on the optimizer side.

## 41. The explicit layer-structured kernel and the width scan: where the Taylor-radius step beats Adam

Krylov on the implicit operator costs 5 forward-equivalents per matvec because each product is a forward and a reverse
pass. For a layered network the kernel has a closed form that needs no Jacobian: K = sum_l (D_l D_l^T) o (A_{l-1}
A_{l-1}^T (x) 1_{dout x dout}) / n, with A_{l-1} the layer inputs (bias column appended) and D_l the per-output backprop
signals at layer l (N x H_l). Cost ~ 2 N^2 sum_l H_l flops, i.e. ~ n dout^2 / H forward-equivalents (measured 97 at
H = 48, 30 at 160, 12 at 512, 6.6 at 1536; exact to 1e-15 against the Jacobian kernel). The exponential on the explicit
600 x 600 kernel is then nearly free (dense Lanczos 2 N^2 m flops, or a full eigendecomposition ~ 10 N^3 ~ 2 fwd-eq at
width 1536), so the Krylov size can be whatever the residual needs.

Width scan at n = 200, Delta = 0.5, depth 2, common target E = 0.05, SegFlow = explicit kernel + momentum 0.6 across
segments, Adam = full batch, best of lr 3e-3/1e-2/3e-2 (FLOP-eq in forward passes: fwd 1, bwd 2, jvp 2, vjp 3, kernel
and dense Lanczos counted as flops/(2nP)):

| width | P | SegFlow FLOP-eq (m=30) | (m=60) | (m=200) | segments | kernel fwd-eq/segment | mean segment |dtheta| | Adam FLOP-eq | Adam steps | Adam/SegFlow (best m) |
|---|---|---|---|---|---|---|---|---|---|---|
| 48 | 2.7k | 3337 | - | - | 33 | 97 | 0.95 | 2175 | 725 | 0.65 |
| 160 | 27k | 1322 | - | 1676 | 37-39 | 30-41 | 0.70-0.78 | 1875 | 625 | 1.42 |
| 512 | 266k | 1239 | 1303 | 1380 | 78-81 | 12-13 | 0.27 | 2175 | 725 | 1.76 |
| 1536 | 2.37M | 1400 | 946 | 956 | 89-132 | 6.6 | 0.13-0.18 | 2325 | 775 | 2.46 |

At width 1536 the residual needs more than 30 Krylov vectors (m = 60 and 200 agree; m = 30 under-resolves and doubles
the segment count); with the explicit kernel this costs nothing. The win grows with width because the kernel cost
falls as 1/H while Adam's step count is flat, but it grows slower than 1/H because the segment count rises (33 -> 89):
the mean segment displacement, i.e. the Taylor radius in parameter norm found by the trust region, shrinks from
0.95 to 0.18 while the path length only falls from 28 to 15. Adam at width 1536 takes 775 steps of a cheap kind; the
segments take 89 steps at ~10 fwd-eq each.

Scope of the win: full batch, squared loss, dataset small enough to hold the N x N kernel (N = n dout; the kernel cost
n dout^2/H forward-equivalents must stay below Adam's ~3 per step times the step ratio). Minibatch segments (explicit
batch kernel, Section 40 and runs/segflow) lose to tuned minibatch Adam at every budget on n = 4000, b = 200,
H = 256: they fit each batch exactly and stall at the batch-noise floor (0.05 vs Adam 0.029 at 3000 fwd-eq).

On the roughness basis as an alternative to Krylov: the graph-Laplacian high-pass basis nearly diagonalises the
kernel's tail (the fixed rank-roughness map of Part II holds for ranks > ~100 at all times) but not the top ~50 modes,
which re-orient toward the target as they are learned. The step acts exactly on that band, so a fixed roughness basis
cannot replace the eigen-solve on the band; with the explicit kernel the eigen-solve is cheap anyway, and the roughness
spectrum sigma_m(t)^2 = lambda_max(P_m K P_m) becomes a free by-product of every segment.

# Part VIII: per-layer spectral preconditioning and target roughness (code/muon_roughness.py, muon_field.py)

## 42. Why this direction

Function-space steps carry a kernel of size (batch x dout)^2 and cannot reach field-to-field problems (Section 41).
The parameter-space optimizers that beat tuned AdamW at scale are the matrix-preconditioned ones (Muon, SOAP, Shampoo:
~1.3-1.4x at <= 0.5B parameters, ~1.1x at 1.2B against well-tuned AdamW; the 2x claims rest on undertuned
baselines -- Wen et al. 2025, Qiu et al. 2025, Liu et al. 2025). Muon orthogonalises each layer's update so every
singular direction moves equally. The spike theory (Parts II-V) says the roughness supply enters through the small
singular directions of the per-layer update, so the theory predicts: the advantage of spectral preconditioning over
Adam grows with target roughness.

## 43. One-step map, H = 160, n = 200, full batch, lr swept for both, linear lr decay to zero at a fixed step budget

Final training error (RMS) at the budget; Muon = Nesterov momentum 0.95 + Newton-Schulz(5) orthogonalisation + Jordan's
shape factor, biases by Adam; best lr per optimizer per cell.

| budget | depth | Delta 0.25 | 0.50 | 0.75 | 1.00 | 1.25 |
|---|---|---|---|---|---|---|
| 1000 steps | 2 | Adam 0.020 / Muon 0.002 | 0.050 / 0.003 | 0.090 / 0.003 | 0.279 / 0.004 | 0.350 / 0.006 |
| 1000 steps | 3 | 0.015 / 0.001 | 0.023 / 0.003 | 0.071 / 0.001 | 0.115 / 0.001 | 0.115 / 0.003 |
| 3000 steps | 2 | 0.008 / 0.001 | 0.012 / 0.001 | 0.011 / <0.0005 | 0.035 / <0.0005 | 0.043 / 0.001 |
| 3000 steps | 3 | 0.005 / <0.0005 | 0.005 / <0.0005 | 0.007 / <0.0005 | 0.007 / <0.0005 | 0.007 / <0.0005 |

At the 1000-step budget Adam's error grows 17x (depth 2) and 8x (depth 3) from the smoothest to the roughest target
while Muon's stays within 0.001-0.006; the Adam/Muon ratio grows from ~10 to ~60-115 with roughness. At 3000 steps
Muon is at the measurement floor everywhere and Adam's roughness penalty persists at depth 2 only. Without a decaying
schedule (fixed lr, Section runs/segflow/muon_roughness.log) Muon stalls at 0.05-0.13 late in training while still
reaching the first error level sooner than Adam on rough targets (steps ratio 1.2-1.5 at Delta >= 0.75, < 1 at
Delta <= 0.5): the schedule is Muon's decisive hyperparameter. Accounting: steps (forward + backward), as in the
optimizer literature; Newton-Schulz on 160 x 160 matrices costs ~10 forward passes of this tiny problem per step and
would be negligible at climate batch sizes.

## 44. Field problem (16 x 16 periodic grid, dout = 256, Gaussian-random-field inputs with (1+k^2)^-2 spectrum,
target tanh(2 N[(-Lap)^p u]), n = 2000, batch 100, MLP 256-512-512-256 GELU)

Without weight decay Muon fails: on the smooth target (p = 0) its test error explodes (16-314 RMS at lr >= 0.01) while
the train error does not, and on p = 1 it is 2.5x worse than Adam. Mechanism: orthogonalisation moves the input
layer equally in every input direction, including the high-wavenumber directions in which smooth input fields carry
no variance, so those weights grow without a loss signal -- the weight-growth failure that Liu et al. fix with
weight decay. Runs with decoupled weight decay 0.01 and a decaying schedule for both optimizers are in
runs/segflow/muon_field_wd.log.

## 45. Field problem: Muon loses to Adam in every setting tested

Training RMSE (test in brackets) at the end of the budget, best lr per optimizer; decoupled weight decay 0.01 and a
linear schedule for both unless noted.

| setting | target p | Adam | Muon | Muon/Adam |
|---|---|---|---|---|
| minibatch 100, 6000 steps, no WD | 0 (smooth) | 0.026 (0.045) | 0.390 (0.585) | 15 |
| minibatch 100, 6000 steps, no WD | 1 | 0.232 (0.64) | 0.596 (0.62) | 2.6 |
| minibatch 100, 6000 steps, WD | 0 | 0.021 (0.044) | 0.159 (0.180) | 7.6 |
| minibatch 100, 6000 steps, WD | 1 | 0.194 (0.66) | 0.394 (0.44) | 2.0 |
| minibatch 100, 6000 steps, WD | 2 (87% rough energy) | 0.440 (1.08) | 0.563 (0.85) | 1.3 |
| full batch 2000, 2000 steps, WD | 0 | 0.029 (0.049) | 0.161 (0.174) | 5.5 |
| full batch 2000, 2000 steps, WD | 1 | 0.233 (0.63) | 0.425 (0.48) | 1.8 |

At lr >= 0.01 Muon's test error diverges (2-300 RMS) while its train error does not. The full-batch row rules out
minibatch-noise amplification as the cause: with no gradient noise Muon is still 5.5x worse on the smooth target and 1.8x worse on the rough one.
The remaining mechanism is the input-layer degeneracy: the inputs are smooth fields whose high-wavenumber directions
carry no variance, orthogonalisation moves the 512 x 256 input layer equally in all 256 input directions, and the
weights along the empty directions grow with no loss signal to stop them (weight decay 0.01 slows but does not stop
it). Flattening the update spectrum helps only along directions the data actually excites; a flattening that stops at
the data's own spectrum (damped: sigma -> sigma/sqrt(sigma^2 + mu^2) with mu at the data/noise floor, i.e. the
Shampoo/K-FAC damping) is the principled form, untested here. On p = 2 neither optimizer fits (n = 2000 is too small for
87% rough energy) and neither generalises.

Standing verdict for climate-type training (minibatch, smooth high-dimensional inputs, high-dimensional outputs):
Adam is better than everything tested in this project. The roughness-dependent advantage of spectral flattening
(Section 43) is established full-batch on a 3-output problem only.

## 46. How much flattening? Spectral power scan (code/spectral_power.py)

Per-layer update U diag((sigma/sigma_max)^q) V^T of the Nesterov-momentum gradient (exact SVD), q = 1 momentum GD,
q = 0 Muon, q < 0 toward Gauss-Newton; 1000-step budget, linear lr decay, lr swept per cell; final training error.

| depth | Delta | q=1 | q=0.5 | q=0.25 | q=0 (Muon) | q=-0.25 | q=-0.5 | best q |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.25 | 0.0143 | 0.0088 | 0.0055 | 0.0018 | 0.0028 | diverges | 0 |
| 2 | 0.50 | 0.0288 | 0.0191 | 0.0075 | 0.0037 | 0.0031 | diverges | -0.25 |
| 2 | 0.75 | 0.0335 | 0.0104 | 0.0112 | 0.0026 | 0.0106 | diverges | 0 |
| 2 | 1.00 | 0.0388 | 0.0129 | 0.0147 | 0.0035 | 0.0228 | diverges | 0 |
| 2 | 1.25 | 0.1668 | 0.0160 | 0.0287 | 0.0043 | 0.0295 | diverges | 0 |
| 3 | 0.25 | 0.0090 | 0.0026 | 0.0017 | 0.0012 | 0.0007 | diverges | -0.25 |
| 3 | 0.50 | 0.0126 | 0.0048 | 0.0027 | 0.0019 | 0.0004 | diverges | -0.25 |
| 3 | 0.75 | 0.0236 | 0.0047 | 0.0019 | 0.0011 | 0.0009 | diverges | -0.25 |
| 3 | 1.00 | 0.0437 | 0.0064 | 0.0025 | 0.0024 | 0.0005 | diverges | -0.25 |
| 3 | 1.25 | 0.0590 | 0.0092 | 0.0054 | 0.0038 | 0.0011 | diverges | -0.25 |

Two findings. (i) The prediction that the optimal exponent shifts with roughness is NOT borne out: the best q is ~0 at
depth 2 and -0.25 at depth 3 for every Delta; the optimum is set by depth (stability -- q = -0.5 diverges everywhere, the
Gauss-Newton limit overshoots the Taylor radius as in Section 30), not by the target. (ii) The qualitative claim holds:
the penalty for too little flattening grows with roughness -- momentum GD (q = 1) degrades 12x (depth 2) and 6.6x
(depth 3) from Delta = 0.25 to 1.25, Muon (q = 0) only 2.4x and 3x. Roughness sets how much one loses by not
flattening, not where the optimum sits.

## 47. Closing statement of Part VIII

Established: on the full-batch one-step map, per-layer spectral flattening (Muon with a schedule) beats tuned Adam by
one to two orders of magnitude in final error at equal step budgets, by a factor that grows with target roughness
(Section 43), and the optimal flattening exponent is depth-set, not roughness-set (Section 46). Not established:
any advantage on field-to-field regression with smooth high-dimensional inputs, where Muon loses to Adam by 2-8x in
every configuration tested, full batch included, because orthogonalisation moves the input layer along data-empty
directions (Section 45). No method built in this project beats tuned Adam in the minibatch, high-dimensional-output
regime. The theory's contribution to optimizer design is predictive (how much is lost by not flattening, as a
function of roughness), not a new algorithm.

## 48. Making the closure true by construction: the roughness-supply objective (code/supply_step.py, supply_field.py)

Idea: gradient flow violates the efficient-supply assumption (c ~ 0.01, Section 38) because the gradient aims
displacement at the fitted modes. Add a term that raises the kernel's Rayleigh quotient on the current residual,
q = <r,Kr>/||r||^2 = ||J^T r||^2/(n||r||^2), i.e. kernel-target alignment toward the residual as a training signal:
Adam(W) on F = L - beta L.detach() log q (one double-backward per step, ~7 fwd-eq vs 3; output-dimension independent).

Field problem (batch 100, WD 0.01, linear decay, best lr), train/test RMSE at 3000 steps:

| target | beta = 0 (AdamW) | beta = 0.3 | beta = 1.0 | AdamW at equal FLOPs (7000 steps) |
|---|---|---|---|---|
| p = 0 | 0.032 / 0.052 | 0.056 / 0.073 | 0.134 / 0.150 | 0.019 / 0.044 |
| p = 1 | 0.258 / 0.571 | 0.379 / 0.606 | 0.616 / 0.637 | 0.181 / 0.674 |

The supply term hurts monotonically in beta at equal steps and loses further at equal FLOPs. Raising q directly is
not the same as what Adam/Muon do: they raise it as a by-product of displacement that also lowers the loss, while an
explicit alignment term spends displacement on rotating the kernel toward a residual that is itself moving. One-step-map
results in runs/segflow/supply_step.log.

One-step map (H = 160, full batch, linear decay, best lr), final E; same objective:

| depth | Delta | steps | Adam (beta=0) | beta=0.1 | beta=0.3 | beta=1.0 | Adam at equal FLOPs (7/3 x steps) |
|---|---|---|---|---|---|---|---|
| 2 | 0.5 | 1000 | 0.050 | 0.043 | 0.049 | 0.089 | 0.017 |
| 2 | 0.5 | 3000 | 0.012 | 0.011 | 0.012 | 0.027 | 0.0036 |
| 2 | 1.0 | 1000 | 0.279 | 0.315 | 0.307 | 0.741 | 0.062 |
| 2 | 1.0 | 3000 | 0.035 | 0.028 | 0.056 | 0.516 | 0.0069 |
| 3 | 0.5 | 1000 | 0.023 | 0.030 | 0.030 | 0.176 | 0.0069 |
| 3 | 0.5 | 3000 | 0.0047 | 0.0063 | 0.0066 | 0.092 | 0.0020 |
| 3 | 1.0 | 1000 | 0.115 | 0.192 | 0.306 | 0.774 | 0.012 |
| 3 | 1.0 | 3000 | 0.0068 | 0.0136 | 0.0339 | 0.359 | 0.0012 |

At best (beta = 0.1, depth 2) a 15-20% gain at equal steps that disappears at equal FLOPs (Adam given the same
compute is 3-10x better everywhere). Verdict: the explicit supply objective does not raise the rough-mode
eigenvalues usefully; alignment emerges from descent (Shan & Bordelon 2021) and is not improved by being asked for.

## 49. What is and is not established about "nudging the weights toward the rough spectrum" (closing note, 2026-10-11)

Proved: inside the linear model and a Taylor radius no step beats Levenberg-Marquardt (Section 30); the paper's drive
bound Theta_R <= sigma_R lambda_1^{1/2} H2 ||E|| is a ceiling on roughness supply that gradient flow realises to ~1e-5
(Section 38) -- orders of magnitude of headroom, not an impossibility. Literature: feature learning at all widths
requires spectral-norm scaling of updates (Yang, Simon, Bernstein 2023), and Muon is steepest descent under that norm
(Bernstein & Newhouse 2024); kernel alignment emerges from descent and accelerates it, more so in deeper nets (Shan &
Bordelon 2021); explicit NTK training beats the network when the objective is a risk estimator (NTK-KARE, 2025).
Established here: spectral flattening beats Adam by 10-100x on the full-batch one-step map with a roughness-growing
margin; every route tried loses to Adam on smooth-input field regression. Not established: that nothing beats Adam
there. The open direction is a per-layer spectral geometry combined with a kernel objective that estimates risk
rather than alignment, on field data.

## 50. Per-layer roughness-supply sensitivity along Adam (code/layer_supply.py) -- no allocation signal

Measured every 100-150 Adam steps: the first-order change of q = <r,Kr>/||r||^2 (r fixed) produced by Adam's actual step,
split by layer (share_l = (grad q)_l . dtheta_l / sum |.|), and per unit displacement.  L63 (H 160, depth 3, Delta 1.0,
full batch) and the field problem (p = 1, minibatch 100).

* The step's first-order effect on q is NEGATIVE at most checkpoints on both problems (dq/step < 0): Adam's step moves
  the kernel away from the current residual at first order, while q itself swings by 1e3 between checkpoints because
  the step re-excites fast modes (q is transient-dominated on an oscillating path).
* Per-layer shares are sign-indefinite and flip between checkpoints on the field problem (input layer share
  +0.9 -> -0.9 within 150 steps). Late on L63 the pattern is stable (input -0.48, hidden-2 +0.40) but dq/step is 1e-5 and
  q ~ 0.4 -- the residual sits in the tail and the quantity is irrelevant to the loss decrease.
* Conclusion: a cross-layer step allocation driven by the kernel's residual Rayleigh quotient would be chasing a noisy,
  transient-dominated signal. The coherent-spike structure (Part V) is a property of the gradient-flow path's weight
  change, not an instantaneous per-step sensitivity that an optimizer can read off cheaply. Not pursued further.

## 51. Truncated spectral flattening on the field problem (code/trunc_spectral_field.py) -- negative

Rank-k truncated update (top-k singular directions of the momentum gradient set to equal weight, rest dropped),
batch 100, WD 0.01, linear decay, 3000 steps, best lr; train/test RMSE [FLOP-eq]:

| target | AdamW | Muon (full) | trunc k=8 | k=32 | k=128 | AdamW equal FLOPs |
|---|---|---|---|---|---|---|
| p=0 | 0.032/0.052 [450] | 0.132/0.149 [8590] | 0.205/0.211 [525] | 0.159/0.170 [647] | 0.140/0.157 [1263] | 0.019/0.044 [1050] |
| p=1 | 0.258/0.571 [450] | 0.400/0.466 [8590] | 0.616/0.630 [525] | 0.497/0.523 [647] | 0.422/0.472 [1263] | 0.181/0.674 [1050] |

Truncation makes the flattened update monotonically worse (k = 8 is 1.6x worse than full Muon). The hypothesis that
Muon fails on smooth fields because it flattens data-empty directions is refuted: the small singular directions help,
and removing them hurts. The field-problem failure of spectral flattening is not explained by the spike-rank picture;
on this architecture and data per-coordinate normalisation (Adam) is simply better per FLOP.

## 52. Band-wise output-roughness preconditioning on the field problem (code/band_precond_field.py) -- negative

Batch residual split into k wavenumber shells of the 16x16 output; one backward per band gives g_j; band rate
q_j = ||g_j||^2/||r_j||^2 (EMA 0.9); direction sum_j w_j g_j with w_j proportional to 1/q_j (mean 1, cap 10) fed to AdamW.
Batch 100, WD 0.01, linear decay, 3000 steps, best lr of 1e-3/3e-3/1e-2; train/test RMSE [FLOP-eq]:

| target | k=1 (= AdamW) | k=3 | k=5 | AdamW at equal FLOPs to k=5 |
|---|---|---|---|---|
| p=0 | 0.032/0.052 [450] | 0.043/0.061 [1050] | 0.035/0.056 [1650] | 0.0135/0.044 [1650] |
| p=1 | 0.258/0.571 [450] | 0.259/0.675 [1050] | 0.269/0.620 [1650] | 0.150/0.732 [1650] |

Final band weights rise with wavenumber as designed (p=1, k=5: 0.29 ... 1.96), but reweighting never beats plain AdamW
even at equal steps, and at equal FLOPs AdamW wins on train error by about 2x. On the rough target the band variants'
test error lies between AdamW at equal steps and AdamW at equal FLOPs, which is an overfitting difference, not a
training gain. Reweighting residual roughness classes by an instantaneous per-band Rayleigh quotient does not add
anything Adam's per-coordinate normalisation lacks on this problem.

## 53. Heavy-ball momentum is gradient flow in rescaled time, plus an edge-of-stability gain (code/momentum_modes.py)

Linearised prediction (frozen K, buf <- beta buf + g, theta -= eta buf): a mode with eta*lam < (1-sqrt beta)^2 is
overdamped and follows gradient flow with time step eta/(1-beta); stiffer modes oscillate and decay at ~(1-sqrt beta) per
step whatever lam; stability needs eta*lam1 < 2(1+beta). At fixed beta there is NO sqrt(lam) speedup of the slow modes
(that needs beta tuned per mode); the gain is the time step, (1+beta)/(1-beta) = 19x over GD at matched stability margin
for beta = 0.9. Test: H48 d2 Delta 0.5 s1 (n = 200), against the saved gradient-flow run (GD at eta*lam1 <= 1, 327210
steps to T = 3000), comparing at rescaled time t = sum eta/(1-beta); eta = C/lam1(t). Runs: runs/momentum/*.log.

| beta, C | steps to t=3000 | E/E_GF over the run | E(T)/E_GF(T) | lam_1 / lam_300 vs GF at T | residual energy in ranks >100 (GF: 1.00) |
|---|---|---|---|---|---|
| 0.5, 2.0 | 68147 | 0.99-1.06 | 1.06 | 0.99 / 1.07 | 1.00 |
| 0.9, 1.0 | 25254 | 1.00-1.20 | 1.005 | 0.94 / 0.73 | 1.00 |
| 0.9, 2.0 | 15674 | 1.00-1.31 | 1.002 | 1.10 / 0.76 | 1.00 |
| 0.9, 3.5 (edge: 2(1+beta) = 3.8) | 1612 | 0.90-2.9 | 0.90 | 0.26 / 2.2 (11.7 at t=155) | 1.00 from t=660 |
| 0.99, 2.0 | 694 | 1.1-5.4 | 1.11 | 0.54 / 0.88 | 0.43-0.86 |
| 0.0 (GD), 1.8 (edge: 2) | 28487 | 1.0-1.8 | 1.58 | 0.24 / 0.96 | 0.38-0.68 |

Findings.
* Momentum away from its stability edge IS gradient flow in rescaled time: beta = 0.5 tracks E(t) to 6 % and the kernel
  spectrum to 5-20 %; beta = 0.9 at C = 1-2 tracks E(t) to 30 % and ends within 0.5 %. The chain theory therefore applies
  to momentum verbatim with t = eta k/(1-beta) (residual in the overdamped modes, which is where it lives after t ~ 300).
  One unexplained residue: at beta = 0.9 the deep-tail lift (rank 300) is 25 % below GF's at equal E.
* Momentum near its own stability edge does BETTER than gradient flow: at C = 3.5 the sharpness is pinned at
  eta*lam1 ~ 3.5-3.8 (lam1 at 22-26 % of GF's), the deep tail is lifted 2-12x more than under GF, and E(T) is 10 % below
  E_GF(T) (GF's final error reached at t = 2640, 1417 steps). GD at its own edge also pins lam1 (24 %) but ends 1.58x
  WORSE than GF: 30-60 % of its residual stays in the top 100 modes, which at eta*lam ~ 2 oscillate undamped. Momentum's
  underdamped modes decay at (1-sqrt beta) per step even at the edge, so its residual leaves the top modes (rank>100
  energy 1.00).
* beta = 0.99 at C = 2 puts ranks up to 241 in the underdamped regime (threshold (1-sqrt beta)^2 = 2.5e-5 eta^-1) and
  lags GF in rescaled time (E up to 5.4x), as predicted; per step it is still the cheapest run (694 steps, E(T) 1.11x).
* Decomposition of the measured 31x of momentum over GD (Section 40 ablation: 1650 vs 51500 steps): ~19x is the time
  step ((1+beta)/(1-beta) at matched stability margin), ~1.1x is momentum's edge-of-stability gain over GF, and ~1.6x is
  GD's edge-of-stability loss to GF (undamped top modes). 19 x 1.14 x 1.58 = 34x predicted from these factors vs 31x measured.
* What this means for the theory: momentum adds nothing to WHICH modes are learned, only how fast gradient flow is
  integrated, EXCEPT at the stability edge, where capping the sharpness while damping the top modes lifts the tail more
  than gradient flow does. That edge effect is outside the chain (which is gradient flow) and is the one place where a
  first-order method beat gradient flow at equal rescaled time here. One design, one seed, n = 200.
* Cross-check with the existing KAdam sweeps (runs/kadam/sweep1, sweep_layer, sweep_fix; n = 2000, H160 d2, 3000 steps,
  all with one shared beta1 = 0.9, so per-mode damping was never varied). Per step, equal push (q = 0) beats q = 0.5 by
  2-4x at every lr on both targets; per-mode sign (signmode 1) reaches E = 0.0038 at Delta 0.5 vs Adam 0.024, Muon 0.011,
  KronSign 0.005. In the global-basis runs a larger lr gives a smaller top eigenvalue (157 -> 92) and a less stiff
  spectrum (lam1/lam101 4080 -> 280, i.e. lam101 8.5x larger), the same signature as the momentum edge run; but the
  larger-lr runs also trained further, there is no gradient-flow reference at n = 2000, and no per-rank lift was logged,
  so the edge effect is suggested, not shown, there. KAdam's exact Jacobian costs ~290x Adam's FLOPs per run.

## 54. Front-locked push window and front-damped momentum in KAdam (code/kadam_front.py, runs/front; job 59717857)

Copy of kadam.py (layer blocks, n = 2000, H160 d2, 3000 steps) with (i) a push window: modes with lam_k < lam_front/rho get
push (lam_k/lam_lo)^0.5 instead of the equal (normalised or sign) push; (ii) mode-space momentum with beta fixed (0.9),
one beta per block critically damping the front band ("front", beta = (1 - sqrt(a_front))^2 with a_k = lr lam_k/sqrt(v_k)),
or per-mode critical damping floored at the window edge ("mode"). Front = the block's residual Rayleigh quotient
n|g_b|^2/|r|^2. 96 runs (8 variants x 2 targets x 2 bases x 3 lrs), 1 GPU node, 26 min. Best lr per variant, train/val:

| target, base | equal push, beta 0.9 (= KAdam) | best other variant |
|---|---|---|
| Delta 0.5, sign | 0.0037 / 0.0123 | front beta, no window, lr 3e-3: 0.0034 / 0.0121 (tie) |
| Delta 0.5, RMS | 0.0222 / 0.0467 | window rho 30, lr 0.09: 0.0197 / 0.0501 (train -11 %, val +7 %) |
| Delta 1.0, sign | 0.447 / 0.623 (lr 1e-3); 0.041 / 0.843 (lr 3e-3) | window rho 30, lr 3e-3: 0.432 / **0.582** (best val, -7 %) |
| Delta 1.0, RMS | 0.291 / 0.616 | windows lr 0.09: 0.17 / 0.68-0.85 (fit more, overfit more) |

* No variant beats equal push on training error where it matters; windows cost up to 40x on Delta 0.5 sign.
* The window as built cut the push on the residual itself: at the end 60-100 % of the residual energy lies more than 10x
  below the Rayleigh front. The Rayleigh quotient (residual-energy-weighted mean eigenvalue) is held up by the residual
  that sign/normalised steps leave oscillating in the fitted top modes; it does not locate the fitting wave.
* On the rough target the only gain is validation (-7 %) from cutting the deep push: deep pushing fits sample-specific
  modes (sign at lr 3e-3: train 0.041, val 0.843), as the validation theorem predicts.
* Per-mode critical damping lifts the deep tail most (rank-1000 eigenvalue up to 100x) and has the worst errors: lift
  without fit.
* The wave, measured correctly (saved gradient-flow runs, n = 200): weight modes by their share of the loss removal,
  lam_k e_k (= n x gradient energy per mode; g_k u_k for any optimizer). Its median eigenvalue tracks 1/t late
  (t lam_wave = 0.4-2.8 from t = 20 on, band lam t in [0.3, 3] carries 50-90 % of the removal), and runs 5-20x below
  1/t during the early transition. Next test: window and damping keyed to this removal-weighted wave (not run).

## 55. Window and damping keyed to the removal-weighted wave (kadam_front.py --front wave; job 59719077, 13.5 min)

Wave = per-mode share of the first-order loss removal, g_k x (step direction)_k, EMA 0.9, negatives zeroed; window lower
edge = removal-weighted lower quartile q25 / rho (equal push above, (lam/lam_lo)^0.5 below), optional boost 3 on
[q25/rho, q25); "front" beta keyed to the wave median. 40 runs; baselines = equal push at the same lr (Section 54).
Best per target/base vs the best-lr baseline, train / val:

| target, base | best baseline | best wave variant |
|---|---|---|
| Delta 0.5, sign | 0.0037 / 0.0123 | front beta (lr 3e-3): 0.0032 / 0.0135 (tie); windows 5-20x worse |
| Delta 0.5, RMS | 0.0222 / 0.0467 | rho 10 boost 3 (lr 0.09): 0.0174 / 0.0571 (train -22 %, val +22 %) |
| Delta 1.0, sign | 0.447 / 0.623 | rho 10 boost 3 (lr 1e-3): 0.454 / 0.592 (val -5 %) |
| Delta 1.0, RMS | 0.291 / 0.616 | rho 3 (lr 0.03): 0.294 / **0.543** (val -12 %, same train) |

* The wave estimate behaves as intended under RMS normalisation (median falls 1e-3 -> 1e-4 -> 6e-5 over the run). Under
  sign steps it is contaminated mid-run by the chattering top modes (median jumps to 10-40, the top of the spectrum).
* Keyed to the wave, limiting the push far below it does not speed up training at the best settings; equal push remains
  best on training error. It does reduce overfitting on the rough target (best val 0.543 vs 0.616, also better than the
  Rayleigh-keyed 0.582), i.e. it acts as a spectral regulariser, not an accelerator.
* Boosting the band just below the wave (x3) gives the lowest training error at the high RMS lr on both targets, but
  pays in validation on the smooth target.
* Verdict for the per-mode-push line at n = 2000: no per-step training gain over equal (sign) push; a 5-12 % validation
  gain on the rough target. All per step; per FLOP the exact-basis method is ~290x Adam and cannot compete.
* Correction (user, 2026-10-12): at n = 2000 the validation error is 2-3x the training error on both targets (e.g.
  0.29 vs 0.62 RMS on Delta 1.0), so validation differences measure regularisation of a data-starved sample, not
  anything that carries to large-data training. On the metric that matters at climate scale (training loss per FLOP,
  with train ~ test), the wave-keyed window is a NEGATIVE result: equal push is best or tied in every case.

## 56. Exponent q < 0, Gauss-Newton cap, and the Taylor-remainder detector (kadam_front.py; job 59719667, 7 min)

Sign per mode with weight (s_k/s_1)^q (per-step residual decrease ~ lam_k^((1+q)/2)); optional per-mode cap at the
Gauss-Newton step |g_k|/(lam_k lr L) (L = number of layer blocks); optional trust-region lr controller holding the
measured Taylor remainder nu = |actual dL - GN-predicted dL| / |first-order dL| (one jvp per step) at kappa. Also logged:
gamma(t), the slope of residual energy vs eigenvalue in the hidden layer's kernel basis (free: e_k = n g_k^2/lam_k).
lr 3e-3 with linear decay unless stated; n = 2000, H160 d2, 3000 steps; training error (the metric that matters):

| q, cap | rough (Delta 1.0) | smooth (Delta 0.5) | remainder nu at step 1000 (rough / smooth) |
|---|---|---|---|
| 0, no cap (= KAdam sign) | 0.0409 | 0.0038 | 0.010 / 0.026 |
| -0.25, no cap | 0.0147 | 0.0021 | 0.39 / 0.087 |
| **-0.25, GN cap** | **0.0032** | **0.0004** | 0.081 / 0.074 |
| -0.5, no cap | 0.472 | 0.193 | 6.2 / 1.13 |
| -0.5, GN cap | 0.381 | 0.227 | 1.7 / 1.56 |
| TR controller (kappa 0.1/0.3, no decay), q 0 to -0.5 | 0.054-0.46 | 0.0004-0.016 | (controller throttled lr to 1e-5..1e-4) |

* q = -0.25 with the Gauss-Newton cap is 13x (rough) and 10x (smooth) better in training error than equal sign push at
  the same lr and cost; the cap alone is worth 4-5x at q = -0.25 (it stops the top modes from overshooting).
* The Taylor remainder identifies when q is too negative: nu ~ 0.01-0.1 for q = 0 and -0.25, 1-6 for q = -0.5, where
  the run fails. The radius, not the residual tilt, sets the usable q: gamma(t) runs -0.9 -> -0.4 (the residual tilts
  toward the deep modes as the top ones are fitted), so the unconstrained "equal finishing time" q* = gamma - 1 ~ -1.5
  is far beyond the radius.
* The lr trust-region controller as built is worse than the decay schedule (it cut lr by 100x early and never
  recovered); the remainder is better used to choose q than lr.
* Still exact-basis (per step ~290x Adam's FLOPs); per step it is now ~100x Adam (0.39) and ~20x Muon (0.068) on the
  rough target at 3000 steps.

## 57. What sets the per-mode step: measured loss curvature along each kernel mode (code/mode_curvature.py, mode_curvature_path.py)

Audit of the KAdam-front rule: theory supplies the basis (exact per-layer GN modes), the leverage s_k = sqrt(n lam_k)
(singular value of the layer Jacobian; "square root" is only the parametrisation), the GN cap and the Taylor detector.
Sign push, a single power law (s_k/s_1)^q, normalisation by s_1 (= an lr rescale, matters only at fixed lr), beta 0.9,
the 1e-12 mode cutoff and the cap split across layers are design choices. To replace q by a measured quantity, the loss
curvature along each mode was measured directly (n = 2000, H160 d2, Delta 1.0, seed 1; 16-24 log-spaced modes per hidden
layer; forward-over-forward jvp):
* Function-space curvature |f''[V_k,V_k]| is nearly flat in s_k (c ~ s^0.14-0.19): the per-mode radius from the size of
  the remainder alone scales as s^0.8, predicting q ~ +0.8 (deep modes should move LESS). Contradicted by Section 56.
* Residual curvature r.f''[V_k,V_k]/n (the non-GN part of the loss Hessian) AT INITIALISATION: negative on 100 % of
  sampled modes in both hidden layers, ~constant in size (1e-2 layer 0, 1e-1 layer 1) over 12 decades of lam; the full
  curvature h_k = lam_k + r.f''/n is negative from rank 21-31 down: every deep mode is a negative-curvature direction.
* ALONG TRAINING (Adam lr 1e-2): at steps 300, 1000, 3000 the residual curvature is POSITIVE on 93-100 % of modes,
  median 0.006-0.02, so h_k ~ lam_k + const with the constant dominating every mode below rank ~20-30.
* The initial negative sign is the "network unlearning its own initial function" of THEORY.md 12.1 / Lemma 15 (raw,
  uncentred residual); it is gone by step 300. The theory has the identity (residual-weighted second derivative = the
  roughness drive) but not the sign, which is measured, not derived.
Consequence (theory-derived, not yet tested): along training the per-mode Newton step is g_k/(lam_k + h_res) with h_res
the measured residual curvature, i.e. Levenberg-Marquardt whose damping is not a tuning knob but the measured non-GN
curvature; at initialisation, negative-curvature modes are moved to the Taylor radius. No exponent q appears.

## 58. Large-n test (n = 20000, H96 d2; job 59720680, 22 min): q vs lr, derived trust-region Newton, lazy model, centring

Train / val RMSE at 3000 steps (FLOP-eq: exact-basis methods ~2.8e6, Adam 9e3, low-rank TR 2.4e4):

| method | rough (Delta 1.0) | smooth (Delta 0.5) |
|---|---|---|
| sign + GN cap, q 0, lr 1e-3 / 3e-3 / 1e-2 | 0.497 / 0.196 / **0.102** | 0.0028 / **0.0006** / 0.0007 |
| sign + GN cap, q -0.25, lr 1e-3 / 3e-3 / 1e-2 | 0.150 / 0.126 / 0.221 | **0.0006** / 0.0010 / 0.0030 |
| sign + GN cap, q -0.25, lr 3e-3, centred init | **0.0905** (val 0.200) | 0.0011 |
| trust-region Newton, exact global basis, measured h_r / pure LM / centred | 0.289 / 0.299 / 0.288 | 0.0027 / 0.0026 / 0.0029 |
| trust-region Newton, low-rank (32 modes) | 0.602 | 0.049 |
| Adam (best of 3e-3, 1e-2, 3e-2) | 0.410 | 0.029 |

Lazy (frozen-kernel) simulation of the same sign rule (n = 2000): final error falls monotonically as q decreases, to
q = -1 on both targets, raw and centred (e.g. rough: 1.05 at q 0, 0.90 at -0.25, 0.73 at -0.75).

* q = -0.25 is not special. The best q shifts with lr exactly as the Taylor-radius explanation predicts: q -0.25 wins at
  lr <= 3e-3 (3.3x at 1e-3 on rough), q 0 wins at lr 1e-2. Tuned (q, lr) pairs reach similar error (rough 0.09-0.13,
  smooth 0.0006): q and lr are substitutes for one quantity, how far the deep modes move per step, which is limited by
  the network's Taylor radius (the frozen kernel alone would push to q = -1). Section 56's "q = -0.25 everywhere" was
  the fixed lr 3e-3.
* The derived local trust-region Newton step (exact GN + measured residual curvature) loses 3x (rough) and 4x (smooth)
  per step to sign push: the local quadratic model penalises deep motion that pays off over the run.
* Centring the initialisation helps sign push on the rough target (0.0905 vs 0.126) and does nothing elsewhere.
* Overfitting is NOT gone on the rough target: strong optimizers reach train 0.09-0.13 with val 0.20-0.22 (Adam:
  0.41 / 0.44). P/N = 10k/60k does not prevent it; the rough map needs more data. Rankings by train and by val agree.

## 59. Per-mode push weights learned online by hypergradient (kronq3h.py; job 59721698) and the comparison landscape

KronQ2 (Kronecker sign push + GN cap) with per-mode log-weights u_k updated by the one-step (horizon 1) or traced
(horizon 100) hypergradient dL_t/du_k = -lr eta (Gr_t)_k z_k, normalised (IDBD / Autostep style); n = 20000, H96 d2,
3000 steps, 2 seeds. Reference: the 42-cell KronQ2 q x lr grid (best cell rough 0.2184, smooth 0.0030).

| | rough (Delta 1.0) | smooth (Delta 0.5) |
|---|---|---|
| hypergradient, scale + shape, horizon 1, base lr 1e-3 / 3e-3 / 1e-2 | 0.242 / 0.237 / 0.217 | 0.0024 / 0.0022 / 0.0024 |
| hypergradient, shape only (mean removed), horizon 1, lr 3e-3 / 1e-2 | 0.243 / 0.234 | 0.0028 / 0.0033 |
| q 0 at the same lr 3e-3 / 1e-2 | 0.266 / 0.260 | 0.0051 / 0.0061 |
| best of the whole q x lr grid | 0.218 | 0.0030 |

* Learned online, the per-mode weights match the exhaustively tuned grid optimum on the rough target and beat it by
  27 % on the smooth one, and the smooth result is insensitive to the base lr (0.0022-0.0024 over 10x): the
  hypergradient removes both q and lr tuning here.
* The per-mode SHAPE is worth 10 % (rough) to 45 % (smooth) over q = 0 at the same lr; the small 300-step test (4 %)
  understated it.
* Horizon 1 is as good as or better than horizon 100.
Comparison with the other instance's tuned baselines (runs/kadam/tune, field20k; seed means): L63 rough n = 20000,
10k steps (validation RMSE): Kronecker-basis method 0.209, tuned Adam (lr, beta2, warmup) 0.292, Muon 0.335. Field
problem (n = 20k, widths 512, batch 100, equal steps): Kronecker-basis Adam rule 2.2-3x better than tuned Adam, the
sign rule 1.4-2.2x. SOAP not yet compared at matched tuning.

## 60. Final q x lr grid and horizon analysis (job 59721274, 2 nodes, 33 min; code/q_grid_summary.py, horizon_q.py)

Exact-basis sign + GN cap, n = 20000, H96 d2, 3000 steps (training error; mid-run hidden-layer Taylor remainder at the
best q for each lr in brackets):
* rough: best q per lr runs -0.375 (lr 5e-4) -> 0 (lr 2e-2); remainder at the optimum 0.09-0.15 at every lr; best cell
  0.0649 at the grid edge (q 0, lr 2e-2).
* smooth: a broad flat optimum 0.0006-0.0007 along the whole ridge, remainder at the optimum 0.088-0.13 at every lr.
* KronQ2 (Kronecker basis, 2 seeds): rough best 0.218 (q -0.5, lr 5e-4, grid edge), smooth 0.0030; remainder along the
  ridge 0.11-0.21 (rough) and 0.012-0.035 (smooth).
So in the exact basis the optimal ridge is a constant-remainder contour at ~0.1 for BOTH targets; the Kronecker basis
shifts the level (smooth ~0.02).

Horizon test (best q at fixed lr judged at 300 / 1000 / 3000 steps): the best q does NOT become more negative at longer
horizons; where it moves, it moves toward q = 0 later (exact rough lr 3e-3: -0.25 -> -0.125; lr 1e-2: -0.125 -> 0;
smooth lr 1e-3: -0.25 -> -0.125). Together with Section 59 (one-step hypergradient as good as a 100-step trace), the
benefit of pushing deep modes harder is realised within a few steps, not through a delayed lift.
CORRECTION to Sections 58 and OPTIMIZER_SUMMARY.md: "the gain comes from multi-step feature learning that no greedy
criterion sees" is not supported. The local trust-region Newton step lost because its MODEL (GN + interpolated residual
curvature, GN-on-stiff / GD-on-deep shape) mispredicts the one-step gain of deep pushes; a greedy criterion that
MEASURES the one-step gain per mode (the hypergradient) finds it. Measure, do not model.

## 61. Optimizer scaling study: results (job 59722234, 4 GPU nodes, 36 min; optimizer/scale_study.py, scale_summary.py; 2026-10-12)

Field problem (16x16 GRF input, target tanh(2 N[(-Lap)^p u]), p = 0 smooth / p = 2 rough), fresh minibatch of 256 every
step, MLP 256 -> w x 2 -> 256 GELU fp32, widths 256-4096, 5000 steps, one seed, no warmup / schedule / weight decay,
3 runs per GPU (wall times inflated and only roughly comparable). 220 runs, all finished, none NaN. Tables (per run CSV
and the printed tables) in runs/scale/summary/. Validation MSE at step 5000, best of the lr grid (AdamW 3e-4/1e-3/3e-3,
Muon 5e-3/2e-2/5e-2 with lr_adam 1e-3, SOAP 1e-3/3e-3, KronQ 1e-3/3e-3); state memory per parameter in brackets.

| variant (state/P at w=4096) | rough w=256 | 512 | 1024 | 2048 | 4096 | smooth w=256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|
| AdamW (2.0) | 0.527 | 0.486 | 0.443 | 0.457 | 0.390 | 4.0e-3 | 2.2e-3 | 1.2e-3 | 8.2e-4 | 7.2e-4 |
| Muon (1.1) | 0.435 | 0.396 | 0.343 | 0.304 | 0.264 | 2.5e-3 | 1.2e-3 | 7.1e-4 | 3.8e-4 | 2.2e-4 |
| SOAP full (9.1) | 0.171 | 0.124 | 0.119 | 0.116 | 0.107 | 1.3e-3 | 4.4e-4 | 1.4e-4 | 9.4e-5 | **4.5e-5** |
| SOAP block 256 (2.5) | 0.174 | 0.126 | 0.120 | 0.112 | **0.083** | 1.3e-3 | 5.2e-4 | 2.5e-4 | 1.5e-4 | 1.1e-4 |
| KronQ full (11.1) | **0.128** | **0.120** | **0.116** | **0.110** | 0.108 | 1.2e-3 | 4.4e-4 | 1.6e-4 | 1.1e-4 | 1.2e-4 |
| KronQ block 1024 (5.8) | 0.128 | 0.120 | 0.116 | 0.109 | 0.106 | 1.2e-3 | 4.4e-4 | 1.8e-4 | 1.2e-4 | 2.4e-3 |
| KronQ block 256 (4.5) | 0.127 | 0.122 | 0.119 | 0.109 | FAIL | 1.4e-3 | 6.4e-4 | 3.3e-4 | 2.7e-4 | 3.6e-3 |
| KronQ block 64 (4.1) | 0.139 | 0.125 | 0.120 | 0.104 | FAIL | 2.0e-3 | 1.1e-3 | 5.8e-4 | 8.6e-4 | 6.1e-3 |
| KronQ b256 + compressed state, bf16 momentum (2.0) | 0.128 | 0.123 | 0.119 | 0.109 | FAIL | 1.2e-3 | 6.3e-4 | 3.4e-4 | 3.4e-4 | 3.8e-3 |
| KronQ diagonal sides > 1024 (4.0) | 0.128 | 0.120 | 0.120 | FAIL | FAIL | 1.2e-3 | 4.4e-4 | 1.2e-3 | 9.7e-4 | 3.6e-3 |

FAIL = the run sits at the constant predictor (val 0.636 = target variance) at both lrs: the network collapses to the
mean, no NaN. Per-step wall at w = 4096 (shared GPU): AdamW 8-14 ms, KronQ diag 26, KronQ b1024 58, KronQ b256/b64 150-200,
KronQ full 191, SOAP b256 195, Muon 180-230, SOAP full 290. KronQ full's cost is the two 4096^2 eighs per layer every 20
steps (380 s of 955 s) and the rotations (545 s).

Findings.
1. Against the user's criterion (keep the advantage at Adam's memory): KronQ as built does NOT scale. The one variant at
   Adam's memory (b256 + compress, 2.0 state/P) collapses on the rough target at w = 4096 and is 32x worse than full
   KronQ on the smooth one; every reduction's cost grows with width on the smooth target (b256: 1.1x at w = 256, 2.1x at
   1024, 2.4x at 2048, 31x at 4096; b1024: 1.0x up to 2048, 21x at 4096) and on the rough target the reductions hold
   to w = 2048 (0.95-1.03x) and then fail outright.
2. At w = 4096 the best method is SOAP with blocks of 256 (2.5 state/P): 0.083 rough vs KronQ full 0.108 (and
   KronQ b1024 0.106 at 5.8 state/P); on the smooth target SOAP full 4.5e-5 vs KronQ full 1.2e-4. KronQ's lead over SOAP
   (1.3x at w = 256 rough, equal at 512-2048) reverses at 4096. Muon is second on smooth (2.2e-4) and far behind on rough
   (0.26); AdamW is last on both (0.39 / 7.2e-4).
3. The usable learning rate of KronQ shrinks with width: lr 3e-3 fails (collapse or 10-100x worse) at every width >= 1024
   for every variant including full, on both targets; at 4096 only lr 1e-3 works and only for full and b1024. The claim
   "insensitive to the base lr over 10x" (REPORT 59, H96, n = 20000) does not hold at width; the 2-point grid has its
   optimum at the edge, so the reductions may be recoverable at lr 3e-4 or 1e-4 (not run).
4. Mechanism, MEASURED in two diagnostic jobs (59723886 and 59724517, 1 GPU node each, 13 + 9 min, 2026-10-12;
   optimizer/scale_diag.sh, scale_fix.sh, scale_diag_summary.py; runs/scale_diag/, runs/scale_fix/). All at w = 4096, p = 2,
   lr 1e-3, 5000 steps, with per-layer diagnostics every 250 steps.
   What the failure is: the second hidden layer's preactivations z_1 = W_1 h_0 grow from rms ~0.3 to 50-70 within the first
   250 steps (and to 100-230 by step 1000) in every failed run, 100 % of its GELU units sit beyond |z| = 3 from step 250,
   the gradient through it then vanishes, the GN cap binds on every mode (capfrac 1.0 by step 1500), and the output layer
   settles on the mean of the targets (val 0.63569 = target variance). Full factors keep zrms(z_1) at 0.3-0.4 throughout.
   What it is NOT (each tested, each failing identically with zrms(z_1) 57-68 at step 250): the hypergradient (off,
   clamped at |u| <= 1, or 10x slower: no change; u stays within +-0.25 when on); a gradient-energy mode mask at 1e-3 or
   1e-5 of rms(G) (no change; by step 250 the mask then freezes the run because the gradient is gone); the GN cap and
   live mask computed from the CURRENT batch's Kronecker factors instead of the EMA (worst of all, zrms 68 at step 250).
   What it IS: the Kronecker curvature model misprices the sign step by a large factor, and the per-mode cap is exact
   inside the model and blind outside it. For one sample the gradient is delta a^T, so in the eigenbases its sign pattern
   is rank one, sign(delta~_i) sign(a~_j), and the sign push moves every mode coherently with that sample: the true GN
   curvature of the step on that sample is (||delta~||_1 ||a~||_1)^2 while the Kronecker model (which pairs delta(x) with
   a(x') for all x', i.e. discards the sample pairing that Theorem 2's Hadamard product A (.) D keeps) prices it at
   ||delta~||_2^2 ||a~||_2^2. The ratio is d_eff(delta~) d_eff(a~), the product of the effective dimensions of the two
   vectors in the working basis (d_eff(v) = ||v||_1^2/||v||_2^2): small in the full eigenbasis of a steep spectrum,
   ~ (number of blocks) x small for blocks, ~ width^2 in raw coordinates (the diagonal variant, which saturates within 20
   steps). Measured at width 64 with blocks of 16 (smoke run of --gnls): q_true / q_Kronecker = 12-18 in the hidden layers,
   0.65 in the output layer. This predicts the failure order (diagonal first, then b64, b256, compressed; b1024 and full
   survive) and the fall of the usable lr with width at fixed block size.
   The fix, MEASURED: a budget on the step's actual effect on the output. --rem 0.1 (one extra forward per step; scale the
   whole step so that ||f(theta + d) - f(theta)|| <= 0.1 ||E|| on the batch) with blocks of 256 at w = 4096 on the rough target:
   val 0.1065, equal to full KronQ (0.1077) and blocks of 1024 (0.106), at 4.46 state/P; zrms(z_1) 0.26-0.67, no saturation;
   the budget binds early (unscaled ratio 0.28-0.39 at steps 250-750) and not late (0.05 at 3000). A gradient mask on top of
   full factors is harmless (0.108 vs 0.1077: control). The per-layer form of the same idea, --gnls 1 (line search of the
   sign step against the exact per-sample GN curvature on the batch, one matmul per layer, logs q_true / q_Kronecker), is
   implemented and smoke-tested, not yet run at width.
   Consequence for the scaling criterion: the failure of the memory reductions at width 4096 is a step-length failure of
   the harness, not a property of the reductions; whether b256 + compressed state (2.0 state/P, Adam's level) reaches
   full-factor accuracy on both targets with the budget is the next run (8 runs, 1 node, ~45 min).
5. FIX RETEST at w = 4096, both targets (job 59724760, 1 node, 9 min; optimizer/scale_fix2.sh; runs/scale_fix2/), 5000 steps,
   lr 1e-3, equal steps:
   | variant (state/P) | rough | smooth |   references: full KronQ (11.1) 0.108 / 1.2e-4; SOAP b256 (2.5) 0.083 / 1.1e-4;
   | b256 + compress, --rem 0.1 (1.96) | 0.107 | 3.0e-4 |   AdamW (2.0) 0.390 / 7.2e-4; Muon (1.1) 0.264 / 2.2e-4.
   | b256 + compress, --gnls 1 (1.96) | 0.081 | 7.3e-4 |
   | b64, --rem 0.1 (4.1)             | 0.107 | 4.7e-4 |
   | diagonal sides, --rem 0.1 (4.0)  | FAIL  | 7.0e-4 |
   Rough: the Adam-memory variant reaches full-factor accuracy with the budget and the best KronQ value of the study with the
   line search (0.081, = SOAP b256 at 20 % less memory): the scaling criterion is met at equal steps. The line search measured
   q_true / q_Kronecker at this width: 30-380 (square layer), 5-53 (first layer), 6-9 (output layer). Smooth: --rem 0.1 is far
   too tight (unbudgeted output change 4-5x the residual -> the step is cut 40-50x all run); --gnls lets z_1 drift to rms 28
   (alpha 0.4-0.8 early, 1 later: the per-sample probe prices the loss-visible part exactly and is blind to directions the
   output layer does not read, which the global budget suppresses as a side effect). Neither constant is right for both
   targets; the diagonal variant fails on rough whatever is done (z_1 rms 23 at step 250 with the output change at 0.1 % of
   the residual: a loss-invisible drift) and is dropped.
6. Equal-FLOP reading of the existing curves (analytic per-step costs in units of F0 = 6 P m: AdamW 1.0; Muon ~1 + 60 w^3/F0
   (~140 at w = 4096); SOAP/KronQ full ~1 + 13 (2w^3)/F0 (~57); blocks of b: 1 + 24 w^2 b/F0 (b1024 ~14, b256 ~4.6, b64 ~1.9);
   diagonal ~1.2; --rem +0.33; --gnls +1.3. NOTE the harness's blocked rotations are a Python loop, so wall time does not
   follow these counts.) At the budget of AdamW's 5000 steps, read at the last eval within budget (every 250 steps; coarse):
   rough w = 4096: AdamW 0.390 (5000 steps); SOAP b256 0.197 (1000); KronQ b256 + compress + rem / gnls 0.137 / 0.134 (1000);
   rough w = 1024: AdamW 0.443; KronQ b64 0.129 (2750); KronQ b256 0.137 (1250); SOAP b256 0.173 (1250).
   smooth w = 4096: AdamW 7.2e-4; SOAP b256 2.4e-3 (1000); KronQ b256 + compress + rem 2.9e-3 (1000): AdamW wins 3-4x.
   smooth w = 1024: AdamW 1.2e-3; KronQ b64 9.0e-4 (2750); SOAP b256 1.8e-3 (1250): about a tie.
   So at equal FLOPs the Kronecker methods beat AdamW 2-3.5x on the rough target at both widths and lose to it on the smooth
   target at w = 4096. A proper equal-FLOP study needs step counts set per method from the cost model, evals every 50
   steps, a larger budget (AdamW is still improving at 5000 steps), and a budget constant that is not hand-set per target.
7. STEP-LENGTH RULE AND PER-MODE RULE at w = 4096 (job 59725179, 1 node, 16 min; optimizer/scale_alloc.sh; runs/scale_alloc/),
   Adam-memory KronQ (b256 + compressed state) throughout, lr 1e-3, 5000 steps, equal steps:
   | per-mode rule | step-length rule | rough | smooth | state/P |
   | hypergradient | one-sided budget --rem 0.1 (ran at 0.2, see item 8) + linear lr decay | 0.106 | 1.55e-4 | 1.96 |
   | hypergradient | two-sided controller holding the output change at 0.1, no decay | 0.128 | 3.1e-4 | 1.96 |
   | hypergradient | controller at 0.03, no decay | 0.114 | 9.5e-4 | 1.96 |
   | derived allocation (--alloc) | controller 0.1 | 0.663 (z_1 rms 2.9e4: diverged) | 9.2e-4 | 1.46 |
   | derived allocation | controller 0.03 | 0.125 | 2.1e-3 | 1.46 |
   | derived allocation | --rem 0.1 (at 0.2) + decay | 0.115 | 1.26e-3 | 1.46 |
   Findings. (a) The decay is not waste: holding the output change at a constant fraction of the residual to the end is
   worse than the decaying step on both targets (fresh batches: a constant relative step sets a noise floor; the decay
   is the annealing). The claim in KRONQ_STEP_NOTE that the late slack "throws away the radius" is withdrawn.
   (b) The budget constant 0.2 (not 0.1) on the output change brings the Adam-memory variant to 1.55e-4 on the smooth target,
   within 30 % of full KronQ (1.2e-4) and SOAP b256 (1.1e-4), and 0.106 on rough (= full). So the smooth gap of item 5 was
   the budget constant, not the compression.
   (c) The derived allocation as implemented did NOT test the derivation: its cutoff gamma came out at 1e-12 lam_max all
   run (the weight-space budget (T_rem R)^2 with R = lr sqrt(N) was never binding), so the rule degenerated to the Newton
   direction g / (lam T_rem) scaled by the output controller: trust-region Newton again (REPORT 58), and it loses in every
   pairing (10-20 % rough, 2-7x smooth) and diverges with the controller at 0.1. To test the derivation the budget R must be
   the weight-space Taylor radius (isotropic form of Thm 16), which is what the derivation assumes and what no run has
   measured at this width. Without the hypergradient state the variant is 1.46 state/P.
   (d) Weight profiles (u binned by lambda decile, square layer, step 5000): the learned u rises from the stiffest modes
   (-1.2 to -1.8 in log) toward the slow ones in every run; with the tighter controller it peaks in the 2nd-4th slowest
   deciles and falls slightly at the slowest (rough: +0.99, +1.01, +0.87, ...; smooth: +1.24, +1.40, +1.46, +1.48, +1.46,
   +1.42, +1.32, +1.15, +0.86, +0.15), the interior peak the ridge solution predicts; with the looser budget it is monotone
   to the slowest decile. The learned rule is "more push to slower modes up to a cutoff".
8. CODE NOTE (budget semantics). From 2026-10-12 20:27 to 20:42 (commits a36fd54, 6e29d3a) the one-sided --rem undid a step
   only down to 2 x rem when the raw step exceeded 2 x rem, i.e. ran at an effective 0.2; restored to exact semantics in
   d1a6b96. Affected: the scale_alloc rem rows (reported above as "ran at 0.2") and the scale_flops rem rows started in that
   window (flagged by git hash in scale_flops_summary.py); the gnls rows are unaffected.
9. EQUAL-FLOP STUDY (job 59725019, 1 node, 30 min; optimizer/scale_jobs_flops.py, scale_flops_summary.py; runs/scale_flops/,
   tables in summary/tables.txt). 108 runs, all finished. Budget = AdamW x 20000 steps at each width; each method's step
   count = budget / its analytic FLOPs per step (item 6); evals every 1 % of the run; best of its lr grid; one seed.
   Validation MSE at the budget, with optimizer-state bytes per parameter in brackets (w = 4096 values):
   | method (state/P)            | rough 1024 | 2048 | 4096 | smooth 1024 | 2048 | 4096 |
   | AdamW (2.0)                  | 0.317 | 0.337 | 0.2455 | 5.1e-4 | 2.8e-4 | 1.56e-4 |
   | Muon (1.1; fp32 NS count)    | 0.479 | 0.534 | 0.576 | 5.1e-3 | 1.1e-2 | 2.3e-2 |
   | KronQ b256 + compress, gnls (1.96) | 0.120 | 0.114 | 0.104 | 4.1e-4 | 3.4e-4 | 7.3e-4 |
   | KronQ b256 + compress, rem (1.96)  | 0.119 | 0.111* | 0.1065 | 3.9e-4 | 1.85e-4* | 4.1e-4 |
   | KronQ b256 + compress, gnls + gmask (1.96) | 0.120 | 0.115 | 0.110 | 4.0e-4 | 3.3e-4 | 6.1e-4 |
   | KronQ b256, gnls (4.46)      | 0.119 | 0.113 | 0.098 | 3.8e-4 | 3.4e-4 | 8.5e-4 |
   | SOAP b256 (2.46)             | 0.120 | 0.114 | 0.0945 | 2.3e-4 | 1.7e-4 | 1.37e-4 |
   | KronQ b1024 gnls (5.8)       | 0.123 | 0.121 | 0.119 | 6.2e-4 | 6.4e-4 | 6.4e-4 |
   | SOAP full (9.1)              | 0.136 | 0.202 | 0.539 | 6.7e-4 | 1.8e-3 | 2.9e-2 |
   | KronQ full (11.1)            | 0.122 | 0.180 | 0.414 | 6.0e-4 | 1.2e-3 | 3.4e-3 |
   (* rows that ran under the undo-to-0.2 budget code, item 8.) Steps at w = 4096: AdamW 20000, SOAP b256 4344, KronQ b256
   rem 4050 / gnls 3387, b1024 1207, full 319, Muon 139.
   Findings at equal FLOPs.
   (a) Rough target: every blocked Kronecker method beats AdamW by 2.3-3.0x at every width; the Adam-memory KronQ (1.96
       state/P) is at 0.104-0.107 at w = 4096 vs AdamW 0.2455 (2.35x) and is within 10 % of SOAP b256 (0.0945, 2.46 state/P);
       at widths 1024-2048 they tie. Full factors are out of the race per FLOP from w = 2048 (too few steps).
   (b) Smooth target: SOAP b256 is best at every width (2.2x, 1.6x, 1.14x over AdamW). The Adam-memory KronQ beats AdamW by
       1.25x at 1024, ties at 2048, and LOSES at 4096 (gnls 4.6x worse, rem 2.6x worse); AdamW at 20000 steps reaches 1.56e-4
       there. The uncompressed b256 (4.46) is no better than the compressed one: the loss is not the compression.
   (c) Against the criterion (advantage at Adam's memory, now at equal FLOPs): met on the rough target (2.35x over AdamW,
       ~SOAP), NOT met on the smooth target at w = 4096. The sign push's edge over SOAP's rotated-Adam rule at equal steps
       (item 1: 1.3x at w = 256) is gone at equal FLOPs; SOAP b256 is the better method per FLOP here at 25 % more memory.
   (d) Stability: KronQ at lr 3e-3 with the output budget still collapses on rough at w >= 2048 (budget 0.2 does not
       protect at that lr); with the line search it trains (2048: 0.114 at 3e-3). The gradient mask adds nothing.
   (e) Muon's analytic fp32 Newton-Schulz count (60 w^3) leaves it 139 steps at w = 4096; with a bf16 count it would get
       ~1000 and still trail (its 5000-step value, 0.264, is above KronQ's 1000-step value). AdamW's best lr is the grid
       edge (3e-4) at w >= 2048: 1e-4 might improve it on the smooth target.
   Caveats: one seed; analytic FLOPs (the harness's blocked rotations are a Python loop: wall times are 3-5x the FLOP ratio);
   rem rows mixed semantics; no weight decay / warmup / schedule tuning for the baselines.
10. CONTINUITY CHECK (job 59725665, 1 node, 6 min; optimizer/scale_continuity.sh, scale_continuity_summary.py; runs/scale_continuity/).
   (A) The other instance's train_ks.py on its field20k_p{0,2}.npz at the saved tuned settings (width 512, batch 100, 6000
   steps, seed 1) reproduces the saved values to 4 digits: p0 adam 0.3572 / ks-sign 0.1776 / ks-adam 0.1607; p2 adam 11.48 /
   ks-sign 5.246 / ks-adam 5.225 (2.0-2.2x over tuned Adam at equal steps; the Adam rule slightly ahead on smooth).
   (B) Today's harness (scale_study.py, online field problem, tanh-squashed targets) at the same width 512 / batch 100 /
   6000 steps, best of the lr grid, equal steps: smooth: AdamW 2.55e-3, SOAP full 5.3e-4 (4.8x), KronQ full 5.0e-4 (5.1x),
   KronQ full + rem 0.2 4.5e-4 (5.7x); rough: AdamW 0.508, SOAP full 0.127 (4.0x), KronQ full 0.121 (4.2x), + rem 0.121 (4.2x).
   So the harness is continuous with the earlier line: at width 512 the Kronecker methods beat AdamW by 4-5x at equal steps
   (more than the saved 2x: different target normalisation and online data; AdamW's best lr is again the low grid edge),
   KronQ full is at or slightly ahead of SOAP there, and the budget costs nothing. Nothing regressed; what the scaling and
   equal-FLOP studies added is that this per-step margin shrinks per FLOP as the width grows (item 9).
11. FAST KRONQ (reparametrised, all options; optimizer/kronq_fast.py, --opt kronqf; job 59726146, 1 node, 7 min;
   runs/scale_fast/). Width 4096, Adam-memory blocks of 256 + compressed state (1.96 state/P), lr 1e-3, 5000 steps, TF32:
   | variant | rough | smooth | wall (3 per GPU) |   old harness, same memory: rough 0.106 / smooth 1.55e-4 (rem 0.2)
   | rem 0.2 (stats every step) | 0.0733 | 1.29e-4 | 308 s |   references: full KronQ 0.108 / 1.2e-4; SOAP b256 0.083-0.0945 / 1.1-1.37e-4
   | ekfac + gnls | 0.0703 | 7.5e-4 | 322 s |
   | rule adam (ks_adam) | 0.1247 | 2.2e-4 | 284 s |   (lr 1e-3 is too small for the Adam rule: 0.63 at step 250)
   | stats every step + eig_iters 2 + adapt_refresh 0.1 + rem 0.2 | 0.0639 | 1.35e-4 | 151 s |
   | lr_kappa 0.1 | diverged (eta unbounded -> eigh failure) | - | - |   (eta now clamped to [1e-2, 30]; eigh with float64 fallback)
   At equal steps the Adam-memory variant is now BETTER than SOAP b256 on the rough target (0.064 vs 0.083-0.095) and tied on
   the smooth one (1.3e-4 vs 1.1-1.4e-4), at 20 % less memory. The gain over the old harness (0.106 -> 0.064-0.073) comes
   from statistics accumulated every step (ablation --stats_every 20 pending) and the adaptive refresh.
   Isolated timing (one run per GPU, 600 steps): old branch 35 ms/step (eig 23, step 9, fwd/bwd 2); fast path 24 ms/step
   (eig 14, step 4.5, fwd/bwd 2.5-3); TF32 changes little here because eigh dominates; batched QR subspace iteration is
   SLOWER than eigh on the GPU (eig 35 ms/step) and is dropped in favour of the adaptive interval (which cut t_eig 4x over
   5000 steps: 58 s vs 220 s). NOTE: every wall time in items 1-9 was measured with 3 runs per GPU and is ~5x inflated;
   the isolated old blocked KronQ is 35 ms/step, AdamW ~8.
   Per-step FLOPs of the fast path with blocks of 256 (cost model in scale_jobs_flops.py): 1.12 F0 + 0.67/stats_every
   (MC probe) + 0.33 (rem) = 2.1 F0 at stats every step, 1.6 F0 at every 4: the equal-FLOP study (item 9) must be redone;
   at AdamW's 20000-step budget the fast path gets 10000-13000 steps instead of 4050.
12. Equal wall-clock comparison is not meaningful at w >= 2048 here: AdamW finishes 5000 steps in 40-70 s, before any
   other method reaches its first evaluation, and the shared GPUs make the wall times unreliable. At w <= 512, at AdamW's
   wall budget, KronQ b1024 / SOAP reach 1.5-3x lower error than AdamW on the rough target.
Caveats: one seed; baselines untuned (3 fixed lrs, no schedule, beta2 = 0.99); 5000 steps; see THEORY.md 21(d).


13. ABLATION of the fast-path configuration (job 59727605, 1 node, 1 h; optimizer/scale_ablate.sh; runs/scale_ablate/), width
   4096, both targets, Adam memory, 5000 steps, float32 eigh at refresh 20 + adaptive 0.1, stats every step. Validation MSE:
   | variant | rough | smooth |
   | reference: hyper sign push + cap + budget 0.2 | 0.0758 | 1.49e-4 |
   | no budget | 0.0800 | 1.10e-3 |   | no cap | 0.0798 | 1.88e-4 |   | no hypergradient | (cut off) | 2.82e-4 |
   | plain sign push (none of the three) | 0.1176 | 1.55e-3 |
   | rotated Adam rule (SOAP's rule in our basis), lr 1e-3 / 3e-3 | 0.1112 / 0.1105 | 2.14e-4 / 2.00e-4 |
   | full factors | (cut off by the limit) | (cut off) |
   Smooth is deterministic (the reference reproduced 1.4949e-4 in two later launches); the rough reference is 0.0758 with
   float32 eigh and 0.064-0.066 with subspace iteration / float64 eigh (items 11, presets): the eigendecomposition variant
   is worth 15 % on rough, deterministically, so the three single-rule removals on rough (0.080) are not resolved.
   Readings: the budget carries 7x on smooth, the hypergradient 1.9x, the cap 1.3x, all three 10x; the rotated Adam rule
   in our basis (0.111) is WORSE than SOAP b256 in its own basis (0.083-0.095, item 1), so the gain over SOAP on rough is
   the sign push + budget, not the Gauss-Newton factors. The three rules that work are the three that measure (the real
   output change, the realised gain of the previous step, the real gradient) rather than model.
14. PER-MODE LOOKAHEAD RULE (LOOKAHEAD_THEORY.md Thm 7; job 59728694, 1 node, 45 min; optimizer/scale_lookahead.sh;
   runs/scale_lookahead/). The residual curvature per mode c_ij = <E, f_uu> from factor statistics (Prop. 10; exact
   autograd check passes at width 64 and 4096 on the last hidden layer; the first layer's estimate is 3-8x low late in
   training from the dropped downstream term). Measured at width 4096 along the whole run, both targets: fraction of
   concave modes (lam + c <= 0) 1e-5 to 1e-3 per layer; median |c| / lam ~ 0.01, positive. The learned weight of the
   reference is 0.3 on the (few) concave modes and decays 0.29 -> 0.05 on the convex ones (Cor. 8(c) in direction).
   Results (same reference 0.0758 / 1.49e-4): two-regime rule with decay 0.1199 / 8.01e-4; with the remainder controller
   0.1370 / 0.1576; with sphere sharing (box 3) 0.1210 / 1.12e-2; c := 0 control 0.1190 / 8.17e-4. The rule is the frozen
   ridge (c changes nothing) and loses in every form; the 8e-4 is what removing the budget does (item 13). Verdict: no
   per-mode lift regime at this width; the per-mode quadratic model (assumption A1, no interaction) is wrong, not its
   constants. The theory fork's exact-model test of the FIRST-order lookahead (LIFT_GRADIENT.md, commit 5e070b5: leverage
   of the slow residual raised 20x, error worse, lift alone diverges) closes the same mechanism at first order.
   Code notes: a retained .grad on a cached intermediate accumulates every backward through it, the random probe's
   included (~4000x the loss gradient); --S_mode emp had used such a gradient for its one-time rescale (no reported run
   used it); fixed by snapshotting before the probe. The squared-overlap transport is wrong for c (cross terms of
   (p_j . a)^2); c restarts in the new basis at each refresh. The same transport is used for the EKFAC diagonal (--ekfac)
   and is therefore suspect on slow modes between refreshes.
15. TAYLOR-RADIUS SCAN (job 59729529, 1 node, 6 min; --radius_scan; optimizer/scale_radius_summary.py; runs/scale_radius/;
   COLLECTIVE_RULE.md section 2). Reference configuration, width 4096, scans at steps 100 / 500 / 1500 on both hidden
   layers, TF32 off, seven directions (step, sign, gradient, two Kronecker ridges, random, single mode): leverage and the
   length s* at which the relative remainder reaches 0.1, in parameter and output units. ROUGH, layer 0, step 500
   (||E|| = 106 on the batch): leverages span 250x, parameter radii 200x (single mode / gradient 0.01, sign / step 2.2),
   OUTPUT radii 18-23 for six of the seven directions (random 13.5): the Taylor radius is a sphere in output space of
   radius 0.19 ||E|| (0.13 at step 1500) = the budget constant. SMOOTH, layer 1, step 500 (||E|| = 13): output radii 26
   (step), 59 (sign), 532 (gradient), 44-66 (ridges), 3-5 (random, mode): not a sphere, and 2-40 ||E||: the step is
   completion-limited and the budget acts as a damping (a fifth of the way per step), not a trust region. The sign
   push's natural length sits at the radius on rough (s*/s0 ~ 1-1.4). Consequence: with the output change per step fixed,
   directions are ranked by the alignment cos(E, J u) of their output response with the residual (recorded by the scan
   from commit dd4cf48 on; not yet run); the gradient direction moves the outputs along K E (the fast part of E), the
   ridge (K + gamma)^{-1} E maximises the alignment; the matrix-free Krylov ridge with the output-space radius is the
   test that follows (COLLECTIVE_RULE.md section 4).
16. Surprises of 2026-10-12/13 and where the holes are (chat summary, 00:25): (1) the width-4096 collapse was mispricing of
   the coherent step (30-380x), (2) the fix is an output-space budget whose constant is the measured output-space radius
   (item 15), (3) every derived per-mode rule lost (frozen ridge, T-step allocation, LM, lift-aware), (4) the remainder
   controller loses even measured correctly, (5) decay anneals minibatch noise, (6) stats every step 1.75x, (7) eigh
   variant 15 %, (8) SOAP's rule in our basis is worse than in SOAP's basis, (9) the rules that work are the ones that
   measure, (10-11) no concave modes and a near-unit weight on convex slow modes still wins, (12) spread directions go
   200x further than concentrated ones in parameter units, (13) late-run rises (plain sign 0.134 -> 0.204 over the last
   2000 steps; frozen-ridge smooth runs double), (14) the optimizer has never computed a Krylov step. Holes: the geometry
   of the trust region (now measured: output-space sphere on rough), interactions between modes (the Kronecker basis is
   not the kernel's), and time (minibatch noise, decay, kernel motion rate, the end phase).
17. KRYLOV STEP TEST (job 59729771, 1 node, 8 min; optimizer/krylov_step.py, --opt krylov; scale_krylov.sh; runs/scale_krylov/).
   Matrix-free ridge on the batch: CG on (K + gamma) x = E with K = J J^T applied by vjp + jvp, step -J^T x scaled to the
   output-space radius 0.2 ||E|| (decaying), optional heavy-ball momentum with the trust region on the accumulated update.
   Width 4096, 5000 steps, both targets (reference 0.0758 / 1.49e-4; AdamW 0.39 / 7.2e-4 at 5000 steps):
   | M, gamma / Rayleigh, momentum | rough | smooth |  wall (3 per GPU) |
   | 1, 1, 0 | 0.555 | 0.0414 | 56 s |   | 3, 0.1, 0 | 0.547 | 0.0223 | 96 s |   | 3, 1, 0 | 0.534 | 0.0204 | 75 s |
   | 8, 0.1, 0 | 0.509 | 0.0100 | 150 s |   | 3, 0.1, 0.9 | 0.480 | 0.0060 | 77 s |
   Diagnostics along the run: alignment cos(E, K x) 0.06-0.17 rough, 0.07-0.3 smooth; CG residual 0.3-7 ||E|| (the solve
   barely reduces the batch residual at these M); the output-space budget scales the step by 0.03-0.6. Reading: the
   exact-K Krylov step at affordable M is far from the ridge (Thm stepkrylov's slow convergence for a residual on the
   slow modes) and, even so, is the one-step optimum of the wrong trust region (TRUST_REGION.md): worse than AdamW on
   rough and 40x worse than the reference on smooth. The prediction "loses on rough" held; "competitive on smooth" did
   not, for the truncation reason. The Kronecker diagonal solve is a better ridge than 8 exact Krylov terms.
   ALIGNMENT SCAN (same job, reference configuration, res_j5 / res_j11): at equal output change, rough, layer 1, step 500,
   cos(E, J u) = sign 0.43 > ridge2 0.39 > ridge4 0.37 > step 0.34 > grad 0.20 > random / single mode ~0; smooth: ridge4
   0.58 > sign 0.34 > ridge2 0.31 > step 0.27 > grad 0.08. The sign direction is the best-aligned direction on the rough
   target: its output response carries the slow-mode part of E that the gradient's (K E) does not.
18. TRUST-REGION DERIVATION (TRUST_REGION.md). With the two measured facts of item 15 (output-space sphere of radius
   r ||E||; per-mode box |d_k| <= delta_k), the first-order optimal step is d_k = clip(g_k / (2 mu lambda_k), +-delta_k):
   the Gauss-Newton step on modes where the box is slack, a fixed push with the gradient's sign where it binds (the
   held-rough modes: large residual coordinate, small leverage), the whole scaled to the sphere. That is cap + sign push +
   budget, with the learned per-mode weight as the estimate of delta_k. It explains items 4-15 and 17 without lookahead,
   lift or concavity. To measure: delta_k by lambda decile (scan extended with one mode per decile), the learned weight
   against it; open: the smooth-target length (a damping), the late-run degradation, momentum and decay.
19. DECILE SCAN (job 59729964, 1 node, 4 min; runs/scale_decile/; reference configuration, 2000 steps, scans at 100 / 500 /
   1500, one single-mode direction per lambda decile = the mode with the largest |g| in the decile, plus the learned weight
   and lambda medians per decile). Single-mode radius s* (parameter units) and s* x leverage (output units):
   ROUGH, layer 1 (hidden -> hidden), step 500: s* = 0.60, 0.95, 3.9, 3.9, 1.8, 6.6, 3.2, 4.5, 5.9, 5.0 for deciles 0-9
   while the leverage falls 5.85 -> 0.065 (90x) and lambda 2.4e-6 -> 1e-8: the box is FLAT in parameter units (3-6)
   from decile 2 on, smaller (0.6-1) on the top deciles; the learned weight exp(u) is 0.965 ... 0.995, flat and lowest on
   decile 0. Same at 1500 (0.8-14, flat in the middle). The uniform push is the right box step on this layer.
   ROUGH, layer 0 (input layer), step 500: s* = 0.022, 0.19, 0.28, 0.91, 1.55, 1.3, 2.2, 2.7, 5.4, 18 (800x, rising toward
   slow modes) while s* x lev = 16, 0.71, 0.31, 0.41, 0.45, 0.28, 0.26, 0.22, 0.30, 0.59: constant in OUTPUT units (0.2-0.7)
   below the top decile, i.e. the per-mode radius goes as 1 / leverage, delta_k ~ 1 / s_k (the 1/sqrt(alpha_j) of
   TRUST_REGION section 5: the input modes' variance spans the deciles). The learned weight rises 0.90, 1.20, 1.55, 1.78,
   2.04, 2.83, 2.94, 2.72, 2.55, 2.36: the hypergradient tracks the direction of the box (larger pushes on slower input
   modes) but by 3x where the radius grows 100x. Same at 1500 (weights 0.68 -> 3.6).
   SMOOTH: layer 1 radii erratic across deciles (0.04-20, several at the noise floor), weights flat 1.29-1.35; layer 0
   weights 1.06-1.2. Single-mode values with s* at the smallest grid length (lev ~ 0.01-0.05: the linear response is at
   the fp32 noise floor) are not measurements and are left out above.
   Alignment: single modes ~0 (|cosE| < 0.05) on both targets; sign 0.43 / step 0.34 (rough l1, step 500) as in item 17.
   Reading. (a) The box branch of the derivation (item 18) is confirmed on the hidden layer with a flat delta_k, and the
   learned weights are flat there: the uniform push is derived for that layer. (b) On the input layer the box is in
   output units per mode: the box step is d_k ~ sign(g_k) / s_k, equal output change per mode, the q = -1 profile of the
   old per-mode rule (KRONQ_STEP_NOTE: "q = -1 above the cutoff"), and the hypergradient reproduces its direction but not
   its size. Prediction: --q_in -1 (weight (lam / lam_max)^(-1/2) on the input layer, cap and budget as now) improves the
   rough result; the sign push today is 10-100x inside the radius on the slow input modes. To run: reference vs --q_in -1
   vs --q_in -0.5, both targets (6 runs, one node, 15 min).
20. INPUT-SIDE BOX TEST (job 59745633, 1 node, 20 min; optimizer/scale_alpha.sh; runs/scale_alpha/). Weighting every hidden
   layer's modes by (alpha_j / alpha_max)^(qa/2), floored, cap and budget as in the reference; and the radius scan with
   one single mode per ALPHA decile (alpha_j, sigma_i logged). Results (reference 0.0758 / 1.49e-4):
   | qa, floor | rough | smooth |   | -1, 1e-4 | 0.957 | 3.75e-4 |   | -1, 1e-6 | 1.239 | 9.74e-4 |   | -0.5, 1e-4 | 0.638 | 1.10e-4 |
   The -1 weighting destroys the rough fit (worse than the initial 0.668) and hurts smooth 2.5-6x; -0.5 is the best smooth
   number at Adam memory so far (1.10e-4, = SOAP b256's 1.1-1.37e-4) and loses the rough target entirely.
   THE LAW HOLDS. Single-mode radius s* against the mode's input eigenvalue alpha_j, rough target: s* sqrt(alpha_j) =
   0.74, 0.91, 0.88, 0.82, 1.01, 1.08, 0.93, 0.86, 0.78, 0.95 on layer 0 at step 500 across alpha from 1.1e3 to 2.5e-5
   (eight decades; s* from 0.022 to 191), and 0.13-0.26 on layer 1 (a1-a9; a0 is the bias column) with alpha over 30x;
   at step 1500 layer 1: 0.044-0.079. So the per-mode radius is delta_j = c_l / sqrt(alpha_j) on BOTH hidden layers, the
   pre-activation sphere of TRUST_REGION section 5, to +-20 %; the lambda-decile scan (item 19) saw it as flat on layer 1
   only because a lambda decile mixes alpha_j. Smooth: layer 1 within 3x (noisier), layer 0 at the noise floor beyond a2.
   WHAT THE TEST FALSIFIED. The identification of the sign push with the "box branch" (item 18) is wrong in magnitude: the
   operating push per mode is lr eta w ~ 1e-3 while the box is 3-200, i.e. the uniform push is ~1000x INSIDE every box
   and the box never binds at the operating point; the sphere (the budget, scale 0.03-0.6 per step) is the active
   constraint. Pushing modes to their box (qa = -1) spends the output radius on thousands of slow modes with negligible
   residual coordinate and starves the fast ones. Under sphere x box the correct optimum is SPARSE: a mode is on its box
   only if its residual coordinate exceeds a threshold set by the output-side eigenvalue and the sphere multiplier
   (e_k > c sqrt(sigma_i) / r); the rest take the scaled Gauss-Newton step. That rule (box pushes on the few modes with
   the largest residual coordinate, GN elsewhere) has not been tested; the learned per-mode weight is a stochastic
   version of the selection. The uniform sign push is therefore NOT derived by the trust-region argument; what is
   derived and now measured is the trust region itself (output sphere, per-layer pre-activation sphere), and what is
   measured without derivation is that the sign push's output response is the best-aligned with the residual among the
   affordable directions (item 17). The theory has the geometry right and the step still open.
   Timing: eigendecompositions were 78-96 % of wall in every run of items 13-20 (adaptive refresh at period 1 with the
   batched eigh); the subspace-iteration variant (item 11, 0.064) spends 5x less there. Isolated per-step wall against
   AdamW has not been re-measured since item 11 (24 vs 8 ms).
21. MECHANISM MEASUREMENT (job 59746635, 1 node, 23 min; optimizer/mechanism.py, --mech; scale_mech.sh, scale_mech_summary.py;
   runs/scale_mech/). Four rules x two targets, width 4096, 5000 steps, diagnostics every 250 steps on the training batch
   in original weight coordinates: loss decomposition of the step (actual = first-order + Gauss-Newton + remainder),
   residual split by the top-32 Ritz pairs of the current kernel (Lanczos, 48 matvecs), leverage of the residual
   direction before/after with Cross and Q, alignment cos(E, J d), ||J d|| / ||E||, and the minibatch SNR of the gradient
   from two half-batches (overall and per lambda decile). Fast path in its fast configuration (subspace iteration, refresh
   floor 5): rough 0.0612 (484 s per 5000 steps at 2 per GPU with the diagnostics), adam rule 0.1114, frozen ridge 0.1162,
   AdamW 0.3903 (33 s); smooth 1.41e-4 / 2.16e-4 / 8.70e-4 / 8.61e-4.
   Findings.
   (a) FIRST ORDER. For every rule the per-step loss change on the batch is first-order plus Gauss-Newton: the remainder is
       0.2-6 % of the first-order term (sign push 6 %, the largest). No second-order mechanism carries the decrease.
   (b) NOISE-DOMINATED. The gradient's half-batch SNR ||g1 + g2|| / ||g1 - g2|| is 1.0-1.3 per lambda decile in the
       Kronecker basis on both hidden layers (top decile 1.5-2.3 on smooth), i.e. per-mode signal ~ 0.17 x the per-mode
       noise at batch 256; the overall SNR 1-2.5. The per-step batch-loss decrease (1e-2 at step 1000, rough sign push) is
       ~1000x the per-step validation decrease (1.6e-5): each step mostly fits its own batch. The comparison between
       rules is therefore about their expected drift under noise, which batch-side quantities cannot resolve; the
       population-side first-order gain (jvp on validation data with the step) is the measurement that can.
   (c) PER-STEP GAIN ON ROUGH. Batch decrease per step at steps 250 / 1000: sign 1.85e-2 / 1.11e-2, adam rule 8.0e-3 /
       7.4e-3, ridge 6.9e-3 / 4.4e-3, AdamW 2.1e-3 / 3.2e-3. The decrease is 2 cosE ||E|| ||J d|| / N - ||J d||^2 / N; the sign
       push has the largest product cosE x ||J d|| / ||E|| (0.046-0.054 vs adam 0.013-0.042, ridge 0.011-0.017, AdamW
       0.003-0.012). The frozen ridge had the BEST alignment at the start (0.54 vs sign 0.23) but an output step of
       0.02-0.07 ||E|| because its T-step length rule (completion / T_eff) never reached the budget: its loss is a length
       effect, and the claim that the ridge direction is worse is NOT supported by this run. AdamW: alignment 0.07-0.11.
   (d) RESIDUAL SPLIT. From step 250 on, 0.1-1 % of the residual energy sits on the top-32 kernel modes for every rule and
       90-100 % of the first-order decrease is on the rest: the top-32 split is too coarse to separate mechanisms.
   (e) KERNEL MOTION. On rough the leverage of the residual direction grows over the run under every rule (sign 137 ->
       319, adam 85 -> 220, ridge 109 -> 233, AdamW 10 -> 124) with Cross negative in most diagnostics and Q positive;
       per-step relative change 2-3 % (0.5 % on smooth): the statistics horizon of ~10-20 steps is the right order. On
       smooth the leverage falls (67 -> 36) under sign / adam / AdamW.
   Consequence. The open question is a stochastic one: with per-mode SNR ~ 0.2, the sign of the momentum on a mode is
   right with probability ~0.6-0.7, the expected push is ~ delta x SNR per mode (a noise-normalised step, as Adam's
   m / sqrt(v) is), and the differences between sign, Adam rule and ridge are differences in the variance and bias of the
   drift, not in curvature. Theory to write: the expected step and its population first-order gain under per-mode noise
   for the three rules; measurement to make: population-side alignment and gain per step (validation jvp), with the
   ridge given the budget as its length.
22. MECHANISM MEASUREMENT 2 (job 59747389, 1 node, 27 min; scale_mech2.sh; runs/scale_mech2/): validation-side first-order
   gain / Gauss-Newton term / actual change per step (jvp on 1024 held-out samples with the actual step), the sign push + cap
   with uniform weights (alloc2 oneshot, which is sign on box-limited modes + completion on fast ones: NOT a distinct ridge
   direction; a true Kronecker ridge g/(lam + gamma) scaled to the budget is still untested), and batch 1024 on rough.
   Results: rough sign 0.0612, adam rule 0.1114, sign+cap w=1 0.0889, AdamW 0.3903; smooth 1.41e-4 / 2.16e-4 / 2.55e-4 /
   8.61e-4. Batch 1024, 2500 steps: sign 0.0815 (vs 0.101 at step 2500 with batch 256: 1.24x per step, 2x the samples:
   worse per FLOP), AdamW 0.449 (no gain from the batch).
   (a) ON HELD-OUT DATA EVERY STEP IS AT ITS OWN LINE-SEARCH POINT: gn_v = |first_v| within 5-10 % for every rule at every
       diagnostic (sign at 250: first -1.31e-2, gn +1.3e-2, actual -8e-4). The actual per-step validation change is
       ~1 % of either term and of either sign (AdamW's sampled sum is positive while its loss falls). Training here is a
       zero-mean oscillation of amplitude 1-5 % of the loss per step around a drift of 0.1-0.5 % per step. The output
       budget 0.2 ||E|| on the batch is the length at which the population-side quadratic cost equals the population-side
       linear gain: the largest step that does not raise the held-out loss, which is set by the noise (batch 1024: the
       held-out / batch gain ratio rises from 0.3-0.4 to 0.5-0.6, so the useful budget rises with the batch). This is
       the explanation of the budget constant.
   (b) NOISE FITTING BY RULE: held-out first-order gain / batch first-order gain = sign 0.15-0.40, adam 0.31-0.43, sign+cap
       w=1 0.41-0.51, AdamW 0.8-0.9. The sign push fits its batch the most and wins anyway: its held-out alignment early
       (0.10-0.12) is above AdamW's (0.01-0.07) and its held-out output step is at the budget; AdamW's steps are "real"
       but point along the fast part of the residual.
   (c) The learned per-mode weights are worth 1.45x rough / 1.8x smooth over uniform weights with the same sign + cap +
       budget (0.0612 vs 0.0889; 1.41e-4 vs 2.55e-4), as in the ablation.
   (d) Per-mode SNR is lowest on the slow modes and the per-step batch gain is 60-85 % noise fitting; the per-mode
       momentum window from the online SNR (--mom_adapt, implemented 2026-10-13 12:40: T_k = clip((v - m^2) / m^2, T_min,
       T_max), beta_k = 1 - 1/T_k, v the EMA of g^2 as in Adam, transported with the other per-mode state) is the
       derived answer to (a)-(c): longer averaging exactly where the sign is unreliable, re-set every step so a lifted
       mode shortens its own window. To run: scale_momadapt.sh (reference vs T_max 50 / 200, with and without the
       learned weight; 8 runs).
23. ONLINE PER-MODE MOMENTUM WINDOW (job 59748427, 1 node, 9 min; --mom_adapt; scale_momadapt.sh; runs/scale_momadapt/).
   Window per mode = min(noise limit, staleness limit): noise limit (v - m^2) / m^2 from the mode's smoothed de-biased
   signal estimate (an EMA with window T carries sigma^2 / (2T - 1) of noise; a raw single-mode estimate pins every window
   at the ceiling); staleness limit = 1 / (|d_k| lam_k / |g_k|) (step over completion length, the per-mode model's
   gradient-change rate) capped by 2x the layer's adaptive refresh period; floor 2, ceiling 1000 (never binding). Width
   4096, both targets, fast configuration, 5000 steps:
   | rule | rough | smooth |   | reference (beta 0.9) | 0.0612 | 1.41e-4 |   | online window + learned weights | 0.0835 | 1.79e-4 |
   | online window, uniform weights | 0.0988 | 3.26e-4 |
   Windows: 10-40 steps early (runs track the reference: rough 0.144 vs 0.134 at 1000; smooth 5.6e-4 vs 5.7e-4 at 2000),
   then 300-700 steps on the slow deciles (rough layer 0: 412 at 2000; smooth layer 1: 344-704), after which the runs
   fall behind and stall late (smooth 4.7e-4 at 3000-4000 vs 3.7e-4 / 2.8e-4). A fixed 200-step window stalled at the
   mean-output level in the login-node check. Reading: both online limits are large on slow modes; the per-mode
   staleness limit is wrong because a slow mode's gradient changes through the motion of every other mode (the 2-3 %
   per step kernel motion of item 21), not through its own completion, and the refresh-period cap measures basis
   staleness, not gradient staleness: assumption A1 (no interaction) again, in the time domain. A correct online limit
   needs a MEASURED per-mode gradient-change rate (e.g. two EMAs of different windows per mode: disagreement beyond the
   noise model = the signal moved); not built. Momentum 0.9 is near the staleness limit on this problem; the headroom
   from averaging is bounded by the batch-1024 result (1.24x per step, item 22).
23. ONLINE PER-MODE MOMENTUM WINDOW (job 59748427, 1 node, 14 min; --mom_adapt; scale_momadapt.sh; runs/scale_momadapt/).
   Window per mode = min(noise limit, staleness limit): noise limit = (v - m^2) / m^2 with the smoothed de-biased signal
   estimate; staleness limit = min(completion length / step length per mode, 2 x the layer's adaptive refresh period).
   No tuned momentum constant. Results (reference beta 0.9: rough 0.0612, smooth 1.41e-4): online window with the learned
   weight 0.0835 / 1.80e-4; with uniform weights 0.0988 / 3.26e-4. LOSES on both targets, 1.3-2.3x.
   Why: the windows ended at 50-700 steps late in the run (per decile: 118-176 rough, 236-694 smooth w=1), uniform across
   deciles, i.e. set by the layer cap, because both online staleness measures stop binding once the steps are small
   (completion / step -> large under the lr decay; the adaptive refresh period grows to 100-350 since the g.D criterion
   sees little change per small step), and the noise limit asks for long windows (the gradient IS noise-dominated).
   A 10-step window beats all of them. So either the gradient's true decorrelation time per mode is ~10-30 steps for
   reasons the per-mode model does not contain (cross-mode kernel motion at 2-3 %/step is the candidate: item 21(e)),
   or the averaging premise itself is weak (batch 1024, which halves the noise, gave only 1.24x per step: item 22). The
   measurement that decides: the per-mode gradient autocorrelation vs lag along the run (a sample of modes, lags to 64),
   which gives the window with no model; not run.
24. GRADIENT AUTOCORRELATION PER MODE (job 59749119, 1 node, 9 min; --acf; scale_acf.sh; runs/scale_acf/). Reference run,
   200 modes per lambda decile, basis held for 64 steps at steps 250 / 1000 / 2500 / 4000: R(tau) = <g_t g_t+tau> / <g^2>
   and sig = (time mean)^2 / mean square (the persistent fraction; its noise floor is ~1/64).
   SMOOTH: R(1) = 0.01-0.10 on every decile of both layers, R(tau >= 2) = 0 +- 0.03, sig 0.01-0.03 at every stage: each
   mode's gradient is white noise with a persistent part at or below the floor; no per-mode lifetime is resolvable. Top
   decile: R(2..4) = -0.1 to -0.3 (oscillation: the fast modes sit at the stability edge the cap leaves them at).
   ROUGH, layer 0 at step 250: the SLOW input modes carry a persistent signal: d9 R(tau) = 0.45 flat to lag 63, sig 0.41;
   d7 0.31 -> 0.10, sig 0.26; d5 0.13 -> 0.02, sig 0.08; d0-d3 at the floor. By step 1000 every decile is at the floor
   (sig 0.01-0.02), and layer 1 is white at every stage (R(2) = -0.1 to -0.2 on all deciles at 250: oscillation).
   Reading: (a) the same momentum window for every mode is right everywhere except the first few hundred steps of the
   rough target on the input layer, where the slow modes' gradients persist for > 64 steps and a long window would raise
   their SNR (the online-window run of item 23 had its windows capped at 10-20 there by the refresh-period proxy, i.e.
   short exactly where long was safe, and long late where everything is white); (b) the per-mode momentum has headroom
   only in that early window; (c) the regime is 97-99 % noise per mode for most of the run, and the sign push's fixed
   per-mode step bounds the random walk at +-delta per step while the drift accumulates linearly (signSGD-type
   robustness) - the first candidate explanation of its margin consistent with items 21-24; testable from recorded
   gradients and steps (drift vs random-walk amplitude per mode for sign vs the Adam rule).
24. GRADIENT AUTOCORRELATION PER MODE (job 59749119, 1 node, 9 min; --acf; scale_acf.sh; runs/scale_acf/). Reference run,
   200 modes per lambda decile, basis held for 64 steps at steps 250 / 1000 / 2500 / 4000; R(tau) = <g_t g_t+tau> / <g^2>
   and the persistent fraction sig = (time mean)^2 / mean square. SMOOTH: R(1) = 0.01-0.10 on every decile and layer,
   R(tau >= 2) = 0 within noise, sig 0.01-0.03 (the estimate's own floor); the top decile shows R(2..4) = -0.1 to -0.3
   (oscillation at the stability edge). ROUGH: the same everywhere EXCEPT layer 0's slow deciles at step 250: R(tau) = 0.3-0.45
   flat out to lag 63 and sig 0.26-0.41 on deciles 7-9 (0.08 on decile 5), gone by step 1000 (sig 0.01-0.02). Reading: each
   mode's gradient is white noise with a persistent part of 1-3 % at every stage, for slow and fast modes alike; the one
   exception is the slow input modes early on the rough target, where the signal is strong and persistent (the held-rough
   part being learned through the input layer) and needs no averaging. There is no per-mode lifetime to set a window by:
   the same window for every mode is right, the 10-step momentum is at the noise/lag trade-off, and the per-mode window
   (item 23) had nothing to gain. What the measurement adds: the optimizer works on gradients that are 97-99 % noise per
   mode and extracts the rest by accumulation; a fixed-size per-mode step (sign) bounds the noise's random walk at +-delta
   per step while the drift accumulates linearly, which a magnitude-scaled step does not: the candidate explanation of the
   sign push's margin that is consistent with items 13-23, to be tested on the recorded gradients (drift vs random-walk
   amplitude per mode for sign against the Adam rule), not with a sweep.
24. GRADIENT AUTOCORRELATION PER MODE (job 59749119, 1 node, 9 min; --acf; scale_acf.sh; runs/scale_acf/). Reference run,
   both targets; at steps 250 / 1000 / 2500 / 4000 the rotated gradient on 200 modes per lambda decile recorded for 64 steps
   with the basis held; R(tau) = <g_t g_t+tau> / <g^2> per decile, and the persistent fraction sig = time-mean^2 / mean square.
   (a) WHITE NOISE ALMOST EVERYWHERE. Smooth target, all deciles, both layers, all four stages: R(1) = 0.01-0.10, R(tau >= 2)
       = 0 within noise, sig = 0.01-0.03 (the estimator's own floor for 64 samples): each mode's gradient is white step to
       step with a persistent part of at most ~1 % of its mean square. Rough target from step 1000 on: the same.
   (b) THE EXCEPTION, AND IT IS THE ROUGH TARGET'S SIGNAL. Rough, step 250, input layer: the slow deciles carry a persistent
       gradient: d9 R(tau) = 0.44 flat to tau = 63, sig = 0.41; d7 0.30 / 0.26; d5 0.10 / 0.08; d0-d3 white. Per-batch
       SNR ~ 0.8 on the slowest input modes early in the run, so their sign is right ~80 % of the time without averaging;
       by step 1000 it is consumed (sig 0.01-0.02). This is the held-rough part being learned through the input layer's
       low-variance modes, and it is the one place a per-mode treatment has something to act on (a larger push, not a
       longer window: the sign is already reliable there).
   (c) OSCILLATION ON THE FAST MODES (edge of stability). Top decile, both targets, both layers: R(2), R(4) = -0.1 to -0.5
       (rough l0 step 250: R(2) = -0.51), i.e. the gradient reverses every 2-4 steps: the push on the fastest modes
       overshoots their completion point. At step 250 on rough, layer 1 shows R(2) = -0.1 to -0.2 on EVERY decile (the
       budget binds and the whole step is at the stability edge); later only the top decile. The cap (completion / L with
       the Kronecker lambda) does not prevent it because the Kronecker curvature underprices the coherent step; the learned
       weight lowers the top decile only slowly (0.9 vs 1.0).
   Consequences. The per-mode momentum window (item 23) had nothing to act on: with white gradients and sig ~ 1 % no
   window is better than another, and the same window for every mode is right. The two per-mode structures that DO exist
   are (b) and (c), and both call for a per-mode STEP SIZE driven by sign consistency: grow the push on modes whose gradient
   keeps its sign (the persistent slow input modes early), cut it on modes whose gradient flips (the oscillating fast
   modes). Implemented as --hyper_rule rprop (x1.2 / x0.5 on the log weight, reusing the hypergradient state); the one-step
   hypergradient is the same signal weighted by magnitude at rate 0.01. Not yet run.
26. SIGN-CORRELATION STEP RULE (job 59749946, 1 node, ~8 min; --hyper_rule signcorr; scale_signcorr.sh; runs/scale_signcorr/).
   u_k += 0.05 x EMA_0.98[sign(g_t)(sign(g_t-1) + sign(g_t-2))] in place of the one-step hypergradient. Results: rough 0.1305
   (reference 0.0612), smooth 2.49e-4 (reference 1.41e-4). LOSES 2.1x / 1.8x, and the numbers equal the UNIFORM-weight sign
   push (0.118-0.13 rough, 2.55e-4 smooth; items 13, 22): the rule degenerated to it. Why: the smoothed sign correlations are
   small but positive on every mode (+0.01 to +0.05 at lag 1, the ~1 % persistent part and the momentum's own correlation),
   so an integrating rule with no restoring term drives every weight to the clamp (exp(4.6) = 99 by step 1000-2500 on
   both layers); the budget then rescales the whole step and the per-mode shape is gone. The per-mode structure of item
   24 is in the trace (slow input modes +0.14-0.16 vs +0.05 on fast ones at step 250; top decile lag-2 -0.07 on layer 1)
   but it is a 2-3x contrast on a 0.05 baseline, which the one-step hypergradient already turns into 2-3x weights on the
   slow input modes and 0.9 on the top decile (item 19). Verdict: the one-step hypergradient stays; no rule built on the
   gradient's sign statistics has beaten it, and the per-mode time structure it could use is small. Closed.
27. OPEN MEASUREMENTS OF MECHANISM.md (job 59750990, 1 node, 10 min; scale_open.sh; runs/scale_open/). Reference: sign push +
   cap + learned weights 0.0612 rough / 1.41e-4 smooth (batch 256, 5000 steps).
   (1) Adam rule WITH the cap and the learned weights (--adam_fair): rough 0.1103, smooth 1.62e-4. The sign push is 1.8x
       better on rough with everything else equal; on smooth 1.15x. The 1.45x of item 22 was not the missing cap/weights.
       The sign is essential on the rough target. MECHANISM Prop. 2's parity between bounded-magnitude rules is REFUTED.
   (2) Derived per-mode weight (--wrule derived; ran in its small-SNR limit, w ~ C^2 / lam^1.5, normalised, clipped
       [0.1, 10], no learning): rough DIVERGES (0.164 at 500 -> 1.49 at 5000), smooth 4.5e-4 (3.2x worse than learned, worse
       than uniform 2.55e-4). The model's per-mode optimum e_k erf(SNR_k) / s_k is NOT what the hypergradient finds: the
       learned profile is far flatter (2-3x on the slow input modes, 0.9 on the top decile) and far better.
   (3) Batch 1024 with the budget raised: rem 0.3 -> 0.0443, rem 0.4 -> 0.0428 at 2500 steps (rem 0.2: 0.0815; batch 256 at
       2500 steps: 0.101, at 5000 steps: 0.0612). With the budget re-set to the batch's break-even, 4x the batch gives 2.35x
       per step (the model's ~2x) and beats the batch-256 run at twice the samples. The budget constant scales with the
       batch, as item 22 predicted; open item 2 of MECHANISM section 7 is resolved and the practical lever is real.
   (4) Per-sample spread tau_k = std_x(z_x,k) / s_k per lambda decile (100 modes per decile, true residual): flat within
       1.0-1.4x on the input layer and 1.0-1.7x on the hidden layer (slowly rising toward slow modes), both targets, both
       stages: assumption (S) holds to a factor < 2 over the whole spectrum. Per-batch SNR (mean / (std / sqrt m)) per
       decile: 0.6-0.9 on every decile, both targets (1.1-1.4 at batch 1024): Lemma 1 (SNR flat in lambda) holds
       directly. The Kronecker estimate sqrt(m) |g| / sqrt(lam) is off by 4e3-1e5 in scale (normalisation of the probe vs
       the loss gradient) and is not usable as is. NOTE the per-batch SNR of 0.6-0.9 is much higher than the ~0.2 inferred
       from the 64-step persistent fraction (item 24): the signal changes within 64 steps, so the time-mean underestimates
       it. With momentum over ~19 batches the momentum SNR is ~3: the sign of the momentum is right ~99 % of the time,
       i.e. the sign push's DIRECTION is nearly deterministic at this batch size. Noise handling is therefore NOT where
       the sign beats the Adam rule in the same basis: both have nearly deterministic directions and bounded magnitudes;
       the difference is the magnitude PROFILE across modes (sign: 1 on every mode; Adam: |m| / sqrt v ~ 0.5-0.7 varying
       with the per-mode SNR), and the reason a flat profile wins on rough is the open question, now sharply posed.
   Status of MECHANISM.md after this: Lemma 1 and (S) confirmed; the budget-as-break-even and its batch scaling confirmed;
   Prop. 2's ranking GD < ridge-type < sign confirmed, its sign/Adam parity refuted; section 3's per-mode optimum refuted.
28. PER-DECILE STEP PROFILE, sign push vs the fair Adam rule (job 59752816, 1 node, 9 min; --diag prof_*; scale_profile.sh;
   runs/scale_profile/). Reproduces item 27 (rough 0.0612 vs 0.1103; smooth 1.41e-4 vs 1.62e-4). Per lambda decile at
   steps 250 / 1000 / 2500: learned weight w, |D| / w (= |m| / sqrt(v) for the Adam rule, = 1 for the sign), and the share
   of the step's Kronecker output energy sum lam D^2.
   (a) SAME DIRECTION. The output-energy shares of the two rules agree within ~10 % on every decile, layer and stage
       (e.g. rough layer 1 at 2500: sign 0.674 / 0.141 / 0.075 / 0.045 ..., Adam 0.739 / 0.132 / 0.060 / 0.031 ...), and
       |D| / w for the Adam rule is FLAT across deciles (0.15 everywhere on rough at 2500; 0.10-0.28 earlier): the Adam
       rule's magnitude profile is not SNR-shaped, it is a constant ~0.15 times the sign's. (The 0.15: v is a 100-step
       average of g^2 on a shrinking residual, so sqrt(v) remembers larger past gradients.) The learned weights rise toward
       the slow input modes under both rules (sign 0.4 -> 2.6, Adam 0.45 -> 1.6 on layer 0 at 2500).
   (b) DIFFERENT LENGTH. The output change before the budget (rem_rho): rough, sign 0.21 at step 250 then 0.17, 0.11, 0.09,
       0.07 ... 0.01 (the budget binds only at the start; afterwards the lr decay sets the length, BELOW the budget);
       Adam 0.07, 0.08, 0.05, 0.04 ... 0.01: 2-3x shorter than the sign at every stage. Smooth: sign 2.6-3.8 x the budget
       until step ~3000 (the budget binds hard), Adam 0.2-0.9 (binds mildly): the budget equalises the two lengths there
       and the rules nearly tie (1.15x).
   CONCLUSION (closes the open question of items 22, 27): the sign push has NO directional advantage over the bounded-
   magnitude (Adam) rule in the same basis with the same cap and weights; the 1.8x on rough is a step-length effect
   (the Adam rule's |m| / sqrt(v) ~ 0.15 shortens its step 2-3x in output units where the budget does not bind), and
   where the budget binds (smooth) the two are equivalent. The mechanism of the method is therefore: (i) a bounded,
   flat-across-modes step in the kernel's eigenbasis (output response ~ sqrt(lambda_k), between GD's lambda and the
   ridge's flat profile: MECHANISM Prop. 2, which explains the ranking against GD / AdamW / exact ridge via Lemma 1),
   (ii) the learned per-mode weight (2-6x toward the slow input modes on rough), (iii) the cap on the top modes, (iv) the
   length: the budget where it binds, the lr schedule where it does not. The sign is the simplest rule that delivers
   (i) at full length. COROLLARY and prediction: on the rough target the reference runs BELOW the budget for 95 % of the
   run (rem_rho 0.01-0.17 < 0.2), i.e. below the break-even length; a larger base lr or slower decay should improve the
   rough result until the budget binds throughout (the smooth target is already budget-bound). Not yet run.
29. UNDER-STEPPING TEST (job 59753295, 1 node, 9 min; scale_understep.sh; runs/scale_understep/). Rough target, reference
   lr 1e-3 / budget 0.2 (0.0612) against lr 2e-3 / 0.2, lr 3e-3 / 0.2, lr 2e-3 / 0.3. ALL THREE COLLAPSE to the mean-output
   level (0.636-0.650 at step 250, flat to 4000, then 1.3-1574 at 5000). The prediction of item 28 is REFUTED: the
   reference's natural step being below the output budget for 95 % of the run is not slack. The output budget does not
   prevent the collapse at lr >= 2e-3 (natural step / residual 0.04-0.27 in those runs: the budget barely binds while
   the network sits at the mean), so the constraint that binds on the rough target is NOT the output change. It is the
   per-layer one: the width-4096 collapse was diagnosed (item 4) as saturation of the first hidden layer's
   pre-activations, and the per-layer trust region is a sphere in pre-activation space (item 20: delta_j = c_l /
   sqrt(alpha_j), confirmed to +-20 %), which the output budget cannot see when the output layer compensates. On rough
   the lr decay is what keeps the hidden layers inside their pre-activation radius; on smooth the output budget binds
   first. Consequence: the length rule the mechanism needs on rough is the per-layer pre-activation budget ||dz_l|| <= r_l
   (the derived trust region, measured on the batch at one matmul per layer), not the output budget alone; its
   constant r_l is the pre-activation change at the measured radius, which the radius scan can log (not yet logged).
   Status of the mechanism statement (item 28): parts (i)-(iii) unchanged; part (iv), the length, is "the output budget
   where it binds (smooth), the per-layer pre-activation radius where it binds (rough), with the lr decay as its
   current proxy".
30. PER-LAYER PRE-ACTIVATION BUDGET (job 59753730, 1 node, 12 min; --zbud; scale_zbud.sh; runs/scale_zbud/). The radius scan
   now logs the relative pre-activation change ||dz|| / ||z|| at the rf = 0.1 radius along each direction: along the actual
   step, rough 0.08-0.15 (layer 0) and 0.022-0.024 (layer 1); smooth 0.04-0.05 and 0.07-0.12. Runs with the step scaled per
   layer to those radii: rough lr 1e-3: 0.111 (reference 0.0612: 1.8x WORSE); rough lr 2e-3: diverges (val 35 at step 250,
   3e5 at 1000); rough lr 2e-3 at half the radii: diverges (1e5); smooth lr 1e-3: 9.0e-4 (reference 1.41e-4: 6x worse).
   Two facts. (a) The reference operates FAR beyond the rf = 0.1 radius on the hidden layer: its relative pre-activation
   change at step 250 is 0.51 on layer 1 (20x the measured radius) and it trains well; the budget at the radius throttles
   layer 1 twentyfold early and the run is slower. The first-order Taylor radius in pre-activation units is NOT the
   constraint the method obeys on rough. (b) In the lr 2e-3 runs both batch-measured quantities were within budget (dz / z
   0.02-0.16 on the hidden layers after scaling; the output budget scaling the batch output change to 0.2 ||E_B|| every
   step) and the VALIDATION outputs still diverged (val 35 -> 3e5 while the batch output change was held). The motion that
   destroys the network is invisible to the training batch: it lies in the null space of the batch Jacobian (the layer has
   1.7e7 parameters, the batch 6.5e4 output values; the sign push moves every live mode, including the ~1.6e7 batch-null
   ones, at full length, and the learned weights are 2-6x on the slowest deciles). The collapse of items 4 and 29 is
   therefore a GENERALISATION failure, not a Taylor-remainder one: the constraint that binds on rough is the accumulated
   motion in directions the batch does not see, which the lr decay anneals.
   This is Theorem 24 / Corollary 24a's content exactly: the full-set residual changes by the CROSS kernel K_{full,B} g_B,
   and a step's null-space component (zero batch effect) has a nonzero full-set effect. The derived length rule for rough
   is therefore a budget on the HELD-OUT output change (one forward on a fresh batch per step, 0.33 F0), not on the batch
   output change and not on the pre-activation change. Implemented as --rem_val (budget measured on a fresh batch; the
   training-batch budget kept as well). Not yet run.
31. HELD-OUT BUDGET (job 59756224, 1 node, 10 min; --rem_val 0.2 on a fresh batch each step, batch budget 0.2 kept;
   scale_remval.sh; runs/scale_remval/). Rough lr 1e-3: 0.0603 (reference 0.0612; the held-out budget barely binds:
   held-out change / residual 0.19 at 250 then 0.15 ... 0.01); smooth lr 1e-3: 1.37e-4 (reference 1.41e-4). Rough lr 2e-3
   and 3e-3: STILL COLLAPSE to the mean (0.65 / 0.64 at step 250) with BOTH budgets holding the per-step output change at
   <= 0.2 ||E|| on the batch and on fresh data. So the collapse is not a step-LENGTH-measurement problem either: it is the
   VALUE of the budget. Reading (item 22's held-out decomposition): the held-out loss change of a step of relative output
   size rho and alignment c is -2 c rho + rho^2 (times ||E||^2); it is negative only for rho < 2c, and the held-out
   alignment early in a rough run is c ~ 0.10-0.12 (item 22), so the break-even is rho* ~ 0.2-0.24 and the budget 0.2 sits
   AT it. With lr 1e-3 the natural step early is 0.15-0.20 (just inside: slow progress, alignment improves, the run goes);
   with lr >= 2e-3 the natural step is above and is cut to exactly 0.2: zero or negative held-out progress per step, the
   function degrades, the alignment falls, and the run slides to the mean. This explains 0.2 > 0.1 (0.1 is inside the
   break-even but half the useful length), the batch-1024 result (higher alignment, higher break-even, 0.3-0.4 better),
   and the lr sensitivity. The derived length is therefore not a constant at all: the step that minimises the held-out
   quadratic along the direction is rho = c, i.e. the exact LINE SEARCH on the fresh batch, alpha = <E_v, -df_v> / ||df_v||^2
   with df_v = f(theta + d) - f(theta) on the fresh batch (the same two forwards as the held-out budget; no constant).
   Implemented as --ls_val (alpha clipped to [0, 1]). Not yet run. Prediction: lr-insensitive, >= the reference on both
   targets, and the batch budget becomes unnecessary.

25 (this fork). APPROXIMATION CONTROLS AND BASIS TRACKING (job 59749741, 1 node, 12 min; optimizer/tracking.py, --track;
   scale_exact.sh; runs/scale_exact/). Width 4096, both targets: reference fast configuration (blocks 256, bf16 state, two
   subspace iterations, refresh floor 5) against "exact Kronecker" (full factors, fp32 state, float64 eigh, refresh floor
   20) and the latter with the statistics horizon halved (beta_stats 0.9). All with the tracking diagnostic every 500 steps.
   | | rough | smooth |   reference 0.0612 / 1.41e-4;  exact-Kronecker 0.1168 / 4.77e-5;  exact + beta_stats 0.9 0.1142 / 4.74e-5.
   THE APPROXIMATIONS MATTER, IN OPPOSITE DIRECTIONS: removing them is 3.0x BETTER on smooth (4.8e-5, the best smooth number
   of the whole line, below SOAP full's 4.5e-5 within noise and 2.5x below the old full KronQ) and 1.9x WORSE on rough
   (0.117, = the old full-factor KronQ's 0.108: blocks have been better on rough all along). The statistics horizon (0.95
   vs 0.9) changes nothing.
   Tracking, what each approximation does (true kernel per layer by Lanczos, top 32; Kronecker modes sampled per decile):
   (a) FAST-MODE FIDELITY: the energy of the true kernel's top-32 directions on the top-256 Kronecker modes is 0.94-1.00
       with full factors and 0.05-0.49 with blocks of 256 (reaching 0.97-1.0 only at 65536 blocked modes): the block
       structure scatters the true top eigenvectors over tens of thousands of modes. The smooth residual lives on those
       directions; that is the 3x.
   (b) CURVATURE ACCURACY: the true diagonal curvature of a Kronecker mode over its Kronecker value (sigma_i alpha_j) is
       0.84-2.5 on every decile with blocks, but 1.6 -> 390 (decile 0 -> 9, layer 1, step 500; 2-16x late) with full
       factors: the product of two accurate factor eigenvalues underestimates the slow modes' curvature by two orders of
       magnitude (the EKFAC discrepancy at its extreme), so the cap and the live mask act on wrong numbers exactly on the
       modes that carry the rough residual. Candidate cause of the 1.9x; --ekfac (the exact diagonal in the Kronecker
       basis) is the direct fix and is untested with full factors.
   (c) COUPLING: the fraction of a mode's response shared with other sampled modes rises from ~0.01 (fast) to 0.7-8
       (slow, layer 1) and to 1e4-1e6 on the slow input-layer modes, for blocks and full factors alike: the Kronecker
       basis does not decouple the slow modes at all; their responses are dominated by what other modes do. This is the
       interaction the per-mode model missed, measured per decile.
   (d) STEP MISPRICING ||J D||^2 / sum lam D^2: layer 1 reference 235 -> 163 along the run, full factors 116 -> 4.2;
       layer 0: 8.8 -> 2.5 and 3.1 -> 0.8.
   Wall at 2 per GPU: reference 523 s (eig 395), exact 679 s (eig 439, float64 full eigh every >= 20 steps).
   Consequence: the "clean" configuration is not one configuration: full factors for the fast modes (smooth) and a correct
   slow-mode curvature (blocks, or EKFAC) for the rough target. The attribution ladder to run: full factors + EKFAC; full
   factors with bf16 state; blocks with float64 eigh; both targets (6 runs). Memory of full factors: 11 state/P (not
   Adam's); the user has set that aside for the controls.

25 (this fork). APPROXIMATION CONTROLS AND BASIS TRACKING (job 59749741, 1 node, 12 min; optimizer/tracking.py, --track;
   scale_exact.sh; runs/scale_exact/). Width 4096, both targets, 5000 steps. Reference = fast configuration (blocks of 256,
   bf16 state, two subspace iterations, adaptive refresh floored at 5). "Exact within the Kronecker model" = full factors,
   fp32 state, float64 eigh, adaptive refresh floored at 20, statistics every step; and the same with beta_stats 0.9.
   | | rough | smooth |   | reference | 0.0612 | 1.41e-4 |   | exact | 0.1168 | 4.77e-5 |   | exact, beta_stats 0.9 | 0.1142 | 4.74e-5 |
   The approximations are NOT innocuous: removing them makes the smooth result 3x BETTER (4.7e-5, the best at this width
   and memory class so far; SOAP full 4.5e-5) and the rough result 1.9x WORSE. The statistics horizon (0.95 vs 0.9) does
   not matter.
   Tracking diagnostics (every 500 steps, both runs; 64 sampled modes per lambda decile, true per-layer kernel top-32 by
   Lanczos):
   (a) COUPLING. Sum over other sampled modes of G_kk'^2 / G_kk^2 (response shared with others): top decile 0.01-0.3, then
       growing 10-1000x per decile to 10^3-10^6 on the slow deciles of layer 0 and 1-600 on layer 1, in BOTH bases. The
       Kronecker basis (blocked or full) diagonalises the kernel on the top ~2 deciles only; on the slow modes, where the
       rough residual lives, a mode's own response is a small part of what it shares with others: these are not
       eigen-directions of anything.
   (b) CURVATURE. True per-mode curvature / Kronecker lambda, median per decile: blocked reference 0.85-1.3 on the fast
       deciles, up to 2.5 on the slow; FULL factors 1.0-1.6 fast and 16-390 on the slow deciles of layer 1 (the product of
       two tiny factor eigenvalues underestimates the curvature by 10-400x). With full factors the cap g / lam is therefore
       10-400x too loose on the slow modes: the candidate reason the full-factor sign push is worse on rough (its slow
       pushes overshoot), to be confirmed by full factors + the subspace iteration (basis continuity) and by the cap alone.
   (c) TRACKING OF THE TRUE TOP MODES. Energy of the true kernel's top-32 directions on the top-K Kronecker modes: blocked
       reference 0.25-0.49 at K = 256, 0.3-1.0 at 4096, ~1 at 65536; full factors 0.85-1.00 at K = 256. The blocks lose the
       fast modes' directions (they are spread over thousands of blocked modes), which is where the smooth target lives:
       the 3x smooth gain of the full basis.
   (d) MISPRICING OF THE STEP, ||J D||^2 / sum lam D^2: blocked 8-9 (layer 0) and 160-235 (layer 1) throughout; full 3 -> 0.8
       and 116 -> 4 over the run. The blocked basis misprices the step on layer 1 by two orders of magnitude at every stage.
   Answer to "are we capturing the changing modes": no. Either Kronecker basis captures the top ~20 % of modes and nothing
   below; the two forms trade a 3x on smooth (fast-mode directions) against a 2x on rough (slow-mode curvature, through the
   cap). The rough result's dependence on an approximation (blocks) that is wrong on the modes that matter is not a sound
   footing. The control that would settle the Kronecker factorisation itself is the exact per-layer kernel basis with
   minibatches at a width where its top ~2000 modes can be kept (1024), against the Kronecker sign push at equal steps: the
   width-160 full-batch version of this gave 3.4x (REPORT 56-60). Not built.
