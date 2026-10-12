
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
