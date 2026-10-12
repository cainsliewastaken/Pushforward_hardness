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
