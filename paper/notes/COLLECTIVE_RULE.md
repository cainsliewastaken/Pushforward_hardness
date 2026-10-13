# The collective step: what the chain gives as a lookahead rule, and what the radius scan says (2026-10-13)

SUPERSEDED in part by TRUST_REGION.md (01:00): the scan's two facts (output sphere, parameter box) give the sign push +
cap + budget as the FIRST-order optimum of the right trust region, with no lookahead; the Krylov test of section 4 was
run (REPORT 61 item 17) and lost on both targets. Sections 2 (the scan) and 3(a)-(b) stand; section 1 is withdrawn.

Follows LOOKAHEAD_THEORY.md (per-mode model, Theorem 7) and its test (REPORT 61 items 13-15): the per-mode rule fails in
every form at width 4096, and the measured residual curvature says why (no concave modes, |c| ~ 1 % of lambda). This note
states the collective form of the same lookahead, which needs no per-mode decomposition, and records the Taylor-radius
scan that fixes the trust region. Notation as in LOOKAHEAD_THEORY section 0.

## 1. The collective lookahead objective (derived from Theorem 2, not an optimality theorem)

WITHDRAWN as a step rule (2026-10-13 00:30), on the result of the theory fork (LIFT_GRADIENT.md, commit 5e070b5): the
first-order form of this objective (steepest descent on grad L - kappa S, S the sharpening gradient of the slow
residual's leverage, kappa = e Lambda_rem / (N s)) was tested in an exact parameter-space model (width 128, no Kronecker
basis); it raised the slow residual's leverage 20-25x over gd / momentum / sign and the error at T was WORSE at every
kappa, and lift alone diverged. Reasons found there: the lift direction is not orthogonal to the descent direction
(cos(g, S) = +0.1 to +0.3, the lift costs error now), and the radius shrinks with the sharpness the lift creates. With
the second-order form (the hill criterion, Theorem 7) also empty at width 4096 (REPORT 61 item 14), investing in
leverage is closed as a mechanism in both orders. The text below is kept as the derivation; sections 2-3 (the trust
region and the alignment) are what remains live.

Theorem 2(b) of LOOKAHEAD_THEORY, applied at the current stage with the residual's own direction E_hat = E / ||E||: over
the remaining path Lambda_rem, the most the error can decrease along E_hat is
    ell(E_hat) Lambda_rem + (1/2) L_E Lambda_rem^2,
with ell(E_hat) = ||J^T E_hat|| the leverage of the residual direction. A step d changes two things in this bound: the
residual (first order: ||E|| decreases by <E_hat, J d> ... sign as a descent step) and the leverage of E_hat. The leverage
change under d is exact from the stage identity restricted to E_hat:
    ell(E_hat)^2 -> || J^T E_hat + M_E d ||^2 = ell^2 + 2 <J^T E_hat, M_E d> + ||M_E d||^2,
where M_E = grad^2_theta <E_hat, f(theta)> is the Hessian of the residual-projected output (the collective lift matrix;
the per-mode curvature c_k of LOOKAHEAD_THEORY is its diagonal u_k^T M_E u_k, measured small; its action ||M_E d||^2 on
a spread step need not be). M_E d is one Hessian-vector product (double backward, ~2 forward-backward).

Greedy-against-the-bound policy (the rule): choose the step d inside the trust region to maximise
    V(d) = <E, J d>_descent  +  Lambda_rem * ( sqrt(ell^2 + 2 <J^T E_hat, M_E d> + ||M_E d||^2) - ell ),
i.e. the first-order decrease now plus the remaining path times the leverage gained. Status: a policy that maximises an
upper bound on removable error; not a proof that it is optimal (the bound is loose late in training, LOOKAHEAD Remark
(ii)); it is derived, collective, and computable. It reduces to the per-mode two-regime rule when the modes do not
interact (M_E diagonal in the mode basis), and it differs exactly where the per-mode rule failed.

Two of its parts are already classical: with Lambda_rem = 0 it is the first-order problem of Thm stepsup, whose
solution in the trust region is the ridge (K + gamma)^{-1} E in output space, computable matrix-free by the Krylov
series (CG on the batch, one jvp + one vjp per term). The lift term is the new part, and its weight Lambda_rem is what
makes the rule change shape over the run.

## 2. The trust region: the Taylor-radius scan (runs/scale_radius, 2026-10-13 00:12; optimizer/scale_radius_summary.py)

Reference configuration (hyper sign push + cap + budget 0.2, Adam memory) at width 4096, scans at steps 100 / 500 / 1500
on both hidden layers, TF32 off, seven directions: the actual step D, sign(g), g, the Kronecker ridges g/(lam + 1e-2
lam_max) and g/(lam + 1e-4 lam_max), a random direction, a single mode. For each: the leverage lev = ||J u|| per unit
parameter length, and the radius s* at which the relative remainder reaches 0.1, in parameter units and in output units
(s* lev = the linear output change at the radius).

ROUGH target, layer 0, step 500 (residual norm on the batch ||E|| = 106):
  | direction | lev | s* (param) | s* lev (output) |
  | step | 8.7 | 2.2 | 19.4 |   | sign | 90 | 0.21 | 19.3 |   | grad | 2140 | 0.011 | 22.7 |
  | ridge2 | 564 | 0.032 | 18.2 |   | ridge4 | 97 | 0.33 | 31.8 |   | rand | 33 | 0.41 | 13.5 |   | mode | 858 | 0.023 | 19.5 |
  Leverages span 250x, parameter radii 200x, output radii 1.5x (2.4x with the random direction). Step 1500: output radii
  7-19 (||E|| = 91); step 100: 11-29 (rand 3.3). Layer 1: output radii within ~4x (single mode apart).
  So on the rough target the Taylor radius is, to a factor ~1.5, a SPHERE IN OUTPUT SPACE of radius ~0.19 ||E|| at step
  500 (0.13 at 1500): the output-change budget rem = 0.2 is this radius. That is why the budget is the dominant rule
  (ablation: 7x on smooth, and the one rule that fixed the width-4096 collapse) and why 0.2 beat 0.1.
SMOOTH target, layer 1, step 500 (||E|| = 13.3):
  output radii: step 26, sign 59, grad 532, ridge2 66, ridge4 44, rand 2.8, mode 4.8. Not a sphere in either space, and
  every useful direction can move the outputs by 2-40 ||E|| before the first-order model breaks: the step is
  completion-limited, not radius-limited, and the budget acts there as a damping (a fifth of the way to the target per
  step in output space), not as a trust region. The gradient direction is the most linear direction of the network.
Both targets: the actual step direction D sits at s*/s0 ~ 1-1.4 before the budget's scaling (the sign push's natural
length is at the radius on rough); in parameter units spread directions (sign, step, ridge4, rand) share a radius of
2-5 while concentrated ones (grad, ridge2, single mode) have 0.01-0.3: the parameter-space geometry is box-like, the
output-space geometry (rough) spherical.

## 3. Consequences

(a) The trust region for the rough target is ||J d|| <= r ||E|| with r ~ 0.1-0.2, in output space; the parameter-space
    sphere of the theorems is the wrong picture for this network, and the sign push's merit in parameter units ("spread
    steps go further") is a statement about parameter geometry that the output-space radius makes irrelevant: at equal
    OUTPUT change every direction is allowed the same step.
(b) With the output change per step fixed, the first-order rule ranks directions by the alignment of their output
    response with the residual, cos(E, J u). The gradient direction moves the outputs along K E (the fast part of E),
    not along E; the ridge (K + gamma)^{-1} E with small gamma maximises the alignment; the sign direction's alignment is
    the number the scan now records (rscan_*_cosE, added after the first scan). Prediction: the matrix-free Krylov ridge
    with the output-space radius beats the sign push per step on the rough target iff its alignment is higher at equal
    output change AND the lift term of section 1 does not favour the sign direction. That is the next test.
(c) On the smooth target the question is different: the step is completion-limited, so the rule is a damped Newton
    step in output space, and the Krylov ridge with gamma set by the damping is its exact form.
(d) [superseded by the withdrawal above; the measurement is still a diagnostic of what the sign step does to the
    leverage, not a rule] The lookahead term: measure ||M_E d||^2 and <J^T E_hat, M_E d> for the sign step against the ridge step at equal
    output change (two Hessian-vector products per direction). If the sign step's leverage gain is larger, that is the
    collective lift the per-mode model could not see and the explanation of its multi-step edge; if not, the edge is
    minibatch averaging (momentum + sign as a denoiser), which none of this covers.

## 4. Experiment (one node, ~30 min)
  1. Scan with the alignment column on both targets (the run above, re-done; 5 min).
  2. Krylov step on the batch (M = 1, 2, 3 CG terms, gamma from the output-space radius; momentum on the step),
     width 4096, both targets, 5000 steps, against the reference at equal steps and at equal FLOPs (the Krylov step
     costs ~M F0, the reference ~2.1 F0).
  3. The lift measurement of 3(d) at steps 100 / 500 / 1500 on the scan run.
