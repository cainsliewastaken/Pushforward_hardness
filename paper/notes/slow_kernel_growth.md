# Fitting a rough target forces the slow kernel to grow

Note from the edge-of-stability work (2026-10-06). Not in main.tex; kept for the later work on integrating the error
dynamics over training. Check: `numerics/eos/eos_slow_forced.py`, output `$SCRATCH/pushforward_eos/gd_v1/slow_forced.csv`.

## Statement

Notation of main.tex: sampled functions with `<a,b> = a.b/n`, features `Psi`, kernel `K = (1/n) Psi Psi^T`, so that the
operator norm of `P Psi` (parameters to sampled functions) satisfies `||P Psi||^2 = lambda_max(P K P)`.

Let `P` be the projection on the eigenvectors of the kernel at the start, `K_0`, with eigenvalues below `gamma`, so that
`lambda_max(P K_0 P) < gamma`. Let gradient descent take steps `Delta_s`, with path length `L_t = sum_{s<t} |Delta_s|`.
The output change is exact, step by step (only `C^1` in the parameters is needed):

    Yhat_t - Yhat_0 = sum_{s<t} int_0^1 Psi(theta_s + tau Delta_s) Delta_s dtau .

Project with `P` and bound each term by `||P Psi|| |Delta_s|`:

    max over the path (iterates and the segments between them) of lambda_max(P K P)  >=  ( ||P (Yhat_t - Yhat_0)|| / L_t )^2 .

This is Theorem `dataset` with the seminorm `R(u) = ||P u||`, read the other way: it bounds the feature roughness that
the path must reach, rather than the motion a given feature roughness requires.

## Reading

To build output on the directions that are slow for the starting kernel, gradient descent must either

* keep the kernel and travel a path of length at least `||P DeltaYhat|| / sqrt(gamma)` (huge for a rough target; the
  only option of a model linear in its parameters), or
* grow the slow block of the kernel by at least the factor `( ||P DeltaYhat|| / (L sqrt(gamma)) )^2`.

No sign assumption and no hypothesis on the architecture. It forces growth of the slow block only, not of lambda_1:
`lambda_1 >= lambda_max(P K P)`, but the forced value is far below `lambda_1` in the runs.

## Numerical check (seed 0, gd_v1, snapshots 4000, 10000, 20000)

Slow subspace from the 5th, 10th or 20th eigenvector of `K_0` (`lambda_5 = 0.24`, `lambda_10 = 1.8e-3`,
`lambda_20 = 2.8e-7`; `lambda_1 = 43.5`).

| target | forced growth of the slow block | actual growth |
|---|---|---|
| sin1x, sin2x | none (factor < 1) | x1.0 to 1.4 |
| sin5x | none | x2 to 840 |
| sin10x, from v_20 | up to x10 | x1e5 to 1e6 |
| sin20x, from v_10 | x2.8 to 7.8 | x60 to 5000 |
| sin20x, from v_20 | x10 to 1100 | x1e4 to 1e7 |
| sin40x, from v_20 | up to x9 | up to x2e5 |

* Never violated; forced only for the rough targets and most in the slowest blocks.
* 3 to 7 orders of magnitude below the actual growth: the path length counts all motion, most of which does not build
  slow output, and at the edge the oscillation inflates it (sin10x: path 142 to 169 against a displacement of 3.3).
  A sharper version would count only the motion that builds slow output.

## Relation to the edge of stability

At the edge one step learns mode `k` at the rate `2 lambda_k / lambda_1` (Prop. `eos:prop:best`). The edge holds
`lambda_1` at `2/eta`; this result says the slow `lambda_k` must grow for a rough target. Together they explain why the
rough part is still learned at the edge, with rates improving as the slow block grows. It does not bear on why
`lambda_1` rises or why it caps, which Section 9 already covers, so it is not in the paper.

## Use for integrating the error dynamics

The error dynamics with a fixed kernel give `e^{-tK} E`; the obstacle to integrating is that the slow eigenvalues
change. This gives a lower bound on how far they must have grown by the time a given amount of slow output has been
built, in terms of the path length; combined with an upper bound on growth per unit path (`|sqrt(lambda_k') -
sqrt(lambda_k)| <= rho |Delta|`, the one-step bound from the same day, measured 0.0003 to 0.14 tight), it brackets the
slow spectrum along training.
