# KronQ: the per-mode step in one notation (2026-10-12)

Written after the width-4096 failure and its diagnosis (REPORT 61). Companion to THEORY.md section 21 and
OPTIMIZER_HANDOFF.md. Every symbol is defined here once; the code is optimizer/scale_study.py.

## Objects

Network: input x in R^256, two hidden layers of width w, output yhat in R^256. For one layer with weight W (o x i):
* a(x) in R^i: the layer's INPUT. First layer: x. Other layers: the previous layer's hidden state after the GELU. The code
  appends a constant 1 so the bias is a column of W (dropped below).
* z(x) = W a(x): the pre-activation; the next hidden state is GELU(z(x)).
* delta(x) = d ell(x) / d z(x) in R^o: the backpropagated signal into the layer for sample x, ell(x) = ||yhat(x) - y(x)||^2 / 256.
  (For the curvature the code uses a random +-1 combination of the outputs in place of ell; same expectation.)
* L = (1/m) sum_x ell(x), m = 256 the batch.
* G = dL/dW = (1/m) sum_x delta(x) a(x)^T (o x i): the gradient.
* A = (1/m) sum_x a(x) a(x)^T (i x i), S = (1/m) sum_x delta(x) delta(x)^T (o x o): exponential averages over batches
  (factor 0.95), re-diagonalised every 20 steps. A = sum_j alpha_j p_j p_j^T, S = sum_i sigma_i q_i q_i^T.

## Modes and the three numbers per mode

A mode (i, j) is the o x i matrix M_ij = q_i p_j^T. The M_ij are orthonormal (Frobenius) and span weight space:
any Delta W = sum_ij c_ij M_ij with c_ij = q_i^T Delta W p_j.
* gradient on the mode:  g_ij = q_i^T G p_j           (the array U_S^T G U_A in the code)
* curvature of the mode: lam_ij = kappa sigma_i alpha_j, kappa = 2/256. This is the Kronecker model's second derivative of
  L along M_ij; the exact Gauss-Newton value is (2/(256 m)) sum_x (q_i . delta(x))^2 (p_j . a(x))^2 and the model replaces
  the average of the product by the product of the averages.
* step on the mode: d_ij, with Delta W = - sum_ij d_ij M_ij.
In the model a step d_ij changes L by -g_ij d_ij + (1/2) lam_ij d_ij^2 and the gradient on the mode to g_ij - lam_ij d_ij.
The length g_ij / lam_ij zeroes the gradient on the mode: the completion length.

## The rules (eta = learning rate; the code uses the momentum, EMA of g with factor 0.9, in place of g)

* gradient descent:        d_ij = eta g_ij.  Gradient on the mode shrinks by (1 - eta lam_ij): spectral bias.
* KronQ sign push (all scaling-study runs): d_ij = eta w_ij sign(g_ij), truncated at |g_ij| / lam_ij (the GN cap), zero if
  lam_ij < 1e-10 lam_max. w_ij = exp(u_ij) learned by the hypergradient (w = 1 with it off). Older code: w = (lam/lam_max)^(q/2).
* derived allocation (--alloc): d_ij = g_ij / ((lam_ij + gamma) T_rem), T_rem = steps left, gamma > 0 chosen each step so
  that sum_ij d_ij^2 = R^2. lam >> gamma: d = g/(lam T_rem), one T_rem-th of the completion length, gradient shrinks by
  (1 - 1/T_rem) per step for every such mode (they finish together). lam << gamma: d = g/(gamma T_rem), gradient descent.
* the budget R: per-step length sum d^2 <= R^2; initially eta sqrt(o i) (a sign push with w = 1). --rem_ctrl multiplies the
  whole step by a running factor so that the measured output change ||yhat_after - yhat_before|| on the batch stays at a
  fixed fraction (0.1 or 0.03) of the residual norm ||yhat - y||.

## Where the derived rule comes from

Assume the lam_ij do not change for the rest of the run and sum_ij d_ij(t)^2 <= R^2 at every step. Choose the lengths for
the remaining T_rem steps to minimise the model loss sum_ij (g_ij - lam_ij x_ij)^2 / lam_ij, x_ij = sum_t d_ij(t) the total
length on the mode. Even spreading uses the least budget (sum_t d^2 = x^2 / T_rem), so the constraint is
sum_ij x_ij^2 <= (T_rem R)^2: a ridge regression, x_ij = g_ij / (lam_ij + gamma). Divide by T_rem. Recomputed every step.

## How realistic the assumptions are

* Frozen curvatures: false over a run (the lift: 1e2-1e4 at n = 200; the factors are re-estimated every 20 steps because
  they move). The rule is applied receding-horizon, so the error is conservative: a mode that will be lifted later looks
  more expensive now, the cutoff gamma sits too high, deep modes get less than optimal. Size of the cost: unknown; it is
  what the alloc-vs-hypergradient comparison (runs/scale_alloc) measures. This is where the chain's stage bookkeeping could
  improve the rule.
* Isotropic budget (one unit of weight motion costs the same remainder on every mode): measured once (REPORT 57, width 160,
  L63, exact per-layer basis): the second derivative along mode k goes as s_k^0.14-0.19 over 12 decades of lam, flat to a
  factor of a few. Not measured at width 4096 or on the field problem. Neglects cross-terms between modes (the coherence
  effect that broke the sign push, REPORT 61 item 4; the ridge direction is a smooth function of g, not a rank-one sign
  pattern, so the cross-terms should be much smaller; not measured: the line-search runs log q_true / q_Kronecker for the
  sign step only). And sum d^2 is a surrogate for the true constraint (the Taylor radius on the function), which is why the
  controller measures the output change directly.

## What was established before this note (do not lose)

* q and lr are two coordinates of one quantity, the per-step nonlinearity budget; the optimum is the constant-remainder
  ridge (~0.1 exact basis, both targets; REPORT 60). The hypergradient finds a point on the ridge online (REPORT 59).
* The sign push with the cap is not the one-step optimum (that is the ridge / LM step, Thm stepsup; its trust-region form
  lost 3-4x per step, REPORT 58); its case is multi-step, which the derived rule makes explicit: q = -1 above a budget-set
  cutoff, q = +1 below. Prediction: the learned u_ij, binned by lam_ij, rises toward slow modes, peaks at the cutoff, falls.
  (Logged in --diag as u_bin / ld_bin per lambda decile from 2026-10-12.)
* The width-4096 failure was the step LENGTH (Kronecker model misprices the coherent sign step by d_eff(delta) d_eff(a)),
  not the per-mode shape; fixed by pricing the step on the batch (--rem, --gnls, --rem_ctrl).
