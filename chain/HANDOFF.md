# Held rough error on Lorenz-63: handoff

Context: final part of "Roughness and the error of a trained network". New section
(held_rough_error.tex, PDF preview alongside) proves that the part of the error along a
roughness functional loses energy only through sigma_R(t) (Prop 1), gives an a priori
floor from sigma0, lambda1, H2 (Thm 2, Cor 3), the moving cutoff (Prop 5), the closure
"efficient roughness supply" (Assumption 6, Prop 7) and a reduced ignition model (Prop 8).

This test uses the Krylov functionals phi_gamma = E_gamma/||E_gamma|| at t=0 (Theorem 1(c),
computed by CG = the Krylov series of Theorem 2) as rank-one roughness functions, for which
C_R = 1, sigma_R = ||phi||_K and Assumption 6 holds with c = 1 exactly. So the two-sided band
is exact up to the leak; what is being tested is (i) the leak size, (ii) the a priori floor,
(iii) ignition times vs the reduced model, (iv) depth.

## Files
- l63test.py   : data (direct stepper, Delta=0.5, states standardized), tanh MLP, full-batch GD
                 as gradient-flow proxy (eta kept below 1/lambda1), per-sample Jacobians, kernel,
                 Krylov functionals, logging. numpy only.
                 usage: python3 l63test.py N DELTA T_TOTAL DEPTHS   e.g. python3 l63test.py 200 0.5 2000 1,3
                 writes res_depth{d}.npy
- analyze.py   : bands, a priori floor, ignition table, figures res_depth*_bands.png
                 usage: python3 analyze.py res_depth1.npy res_depth3.npy
- sandbox_run.log, res_depth1.npy : the one-hidden-layer run that finished in the sandbox.

## What the sandbox run (depth 1, H=96, n=200) already shows
- Prop 15(a) exact: logged drive -2<q_phi,E> equals finite-difference d/dt sigma^2 to 4 digits.
- Rough target: ||E_gamma|| ~ 1.1-1.6 of ||E0||=1.86 on slow Krylov directions.
- sigma^2 on the slowest Krylov direction (gamma=1e-4 lambda1) grew 2.5e-4 -> 0.24 (x1000):
  kernel grows on the rough residual. That direction halved at t=165 vs t_lazy=1406.
- Realized drive is ~1e3 below the bound 2 sigma lambda1^{1/2} H2 ||E||, so the a priori
  floor (Thm 2) is informative only for t < ~0.3 here. The measured-sigma band is what works.

## What to run on the cluster
1. depths 1,2,3,4 at H=96 and H=256; T_TOTAL >= 3000; n=200 (kernel is 600x600; n=400 fine).
2. Delta in {0.1, 0.5, 1.0} (smooth -> rough targets) to see the cutoff move with roughness.
3. Record per depth: sigma0 (per gamma), H2 along g_E (logged every 5th log), lambda1,
   t_half per gamma. Compare t_half vs reduced-model t_red with b replaced by the REALIZED
   median drive ratio (drive/bound column) times lambda1^{1/2} H2 -- that is the honest version.
4. For the paper: report (a) band figure, (b) sigma(t) vs Sigma(t), (c) ignition table,
   (d) depth table. The a priori floor should be reported as it is (short-lived) and the
   section's claims rested on the measured-sigma band and the ignition scaling.

## Known limitations of the script
- H2 is estimated along g_E (||Psi_u||_op/sqrt(n) <= H2), a lower estimate of H2bar.
- GD with eta ~ 0.4/lambda1 approximates gradient flow; eta is halved when eta*lambda1 > 1.
- The Krylov functionals are frozen at t=0; the section's propositions assume a fixed P.
