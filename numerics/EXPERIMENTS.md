# Experiment plan

Paper: `old_tex/hardness_framework_v3.tex` in the Overleaf project (no .tex files in this repo). Tests T1--T6 are in its Section 8; the tracked
quantities Q1--Q5 are in Section 3.4; the protocol P1--P5 is in Section 2.2.
Goal: for vision transformers of increasing size, find the time step and network size at which a
Taylor method of order p with finite-difference lower slots has lower one-step error than order
p-1. Fit scaling laws for each method's unlearned fraction and tracked quantities, predict the
crossover step `h*_p(P)` and crossover size `P*_p(h)`, and check the prediction on held-out
larger networks.

## 1. Systems and data

| ID | System | Data | Notes |
|----|--------|------|-------|
| KS | 1D Kuramoto--Sivashinsky, periodic, `u_t = -u u_x - u_xx - u_xxxx` | `training_data/1d_KS/KS_1024_full.h5`: `u` (1024, 2.6M) float64, L = 100, stored dt = 0.001 | solver in `training_data/1d_KS/canonicalPDEs.py` |
| TURB | 2D forced turbulence, vorticity form, Re = 1000, forcing (4,4), drag r = 0.1 | `training_data/2d_turbulence/full_index.h5`: `omega` (579983, 256, 256) float32 | py2d solver in `training_data/py2d_turb_sims/`; confirm beta and the save interval |

The networks see only states from the data. Time derivatives `x^(k)` are needed only for the
analysis (the norms `m_k` and the remainder check of Section 6), and are estimated by finite
differences of the stored fine-dt trajectories. No method uses the vector field `f`.

## 2. Methods

One network per method. All methods use the same architecture, so they cost the same FLOPs per
step; a finite-difference slot adds only a few vector additions. The network input is the
current state only (P2), the target is standardized by its training RMS (P3), and the final
layer is zero-initialized (P4). History states `x_{j-1}, x_{j-2}` enter only the known part;
in training they are true states from the data, in rollout they are predicted states.

| ID | Known part `Q_h` | Target `V = Phi_h - Q_h` (leading term) |
|----|------------------|------------------------------------------|
| `direct` | 0 | `Phi_h` |
| `euler` | `x_j` | `h f + O(h^2)` |
| `t2_fd` | `2 x_j - x_{j-1}` (two-point slot) | `h^2 x^(2) + O(h^4)` |
| `t3_fd` | `3 x_j - 3 x_{j-1} + x_{j-2}` (two-point slots for `f` and `x^(2)`) | `h^3 x^(3) + O(h^4)` |
| `t2_fd3` (optional) | `x_j + h F_1`, three-point `F_1 = (3/2 x_j - 2 x_{j-1} + 1/2 x_{j-2})/h` | `h^2 x^(2)/2 + O(h^3)` |

With minimal-order finite-difference slots the known part of order p is the polynomial
extrapolation through `x_j, ..., x_{j-p+1}`, and the target is `h^p x^(p)` to leading order
(checked symbolically; the paper's Prop. 3 is the case p = 2). The slot error does not vanish
relative to the target as `h -> 0`: it multiplies the target by `p!` and leaves its shape.
So `delta_p = h^p m_p eta_p` and the crossover with order p-1 is
`h*_p = (m_{p-1}/m_p)(eta_{p-1}/eta_p)`, a factor `p` below the exact-slot value.
`t2_fd3` uses one more history state to make the slot error subleading, and recovers the
exact-slot crossover up to `O(h)`.

Rollout: with a state-only network, the lifted step map of `t2_fd` and `t3_fd` has Jacobian
determinant of modulus 1 (the last extrapolation weight is +-1), so it cannot contract volume.
On a linear decaying mode it has the parasitic root `e^{-z}` (Prop. 4). Test this early (T6).

## 3. Architecture and sizes

- Vision transformer (patch embedding, pre-norm blocks, linear unpatchify head with zero init).
  Reuse and adapt `~/Flop_Constrained_Temporal_Scaling/{1d_KS,2d_turbulence}/networks/nn_ViT.py`.
- Size axis: non-embedding parameter count `P`, with the aspect ratio `d_model / n_layers`
  fixed in the 40--100 range from the literature (value and citation to be fixed with the
  user). Heads = `d_model / 64`, MLP ratio 4.
- Sizes: 5--6 values spanning about two decades of `P` for fitting, plus 1--2 larger held-out
  sizes for the prediction test.
- Patch size: KS 16 (64 tokens), TURB 8 or 16 (1024 or 256 tokens); fixed across sizes.
- Optimizer: Adam, no weight decay; same schedule and step budget for every method at a given
  size (P5). Training time `t` is in optimizer steps, and `Pi = diag(lr_t / (sqrt(v_hat) + eps))`
  from the Adam state (momentum neglected).
- Seeds: 3 at the fit sizes, 1--3 at the held-out sizes.

## 4. Sweeps and scaling fits

- Grid: methods {`direct`, `euler`, `t2_fd`} first, x sizes, x 3--4 values of `h`, x seeds.
  `t3_fd` is added once the order-2 fits show where order 2 wins.
- For each method, fit `eta(P) = eta_inf + a P^(-gamma)` for the training and validation
  unlearned fractions, with bootstrap over seeds. Theory says `eta` of `euler`, `t2_fd`, `t3_fd`
  does not depend on `h` (J2); test this across the `h` values before pooling them. `direct` is
  fit at each `h` separately, since its `O(h)` correction is not small next to `eta(id)`.
- Fit the same form to the tracked summaries (Section 5): slow power, final decay rate `rho(T)`,
  and on the probe kernel the slow count and tail exponent. Check whether `gamma` of each method
  follows from how its slow power moves with `P`.
- From the fits and `m_k`, compute `h*_p(P)` and `P*_p(h)` with bootstrap bands.
- Prediction test (T4): train the held-out sizes at `h` on both sides of the predicted crossover.

## 5. Tracked quantities per run

The full kernel has size `(n n_x)^2` with `n n_x ~ 10^8`, so Q1--Q4 are measured on a fixed
probe set.

- Every logged step (cheap): training and validation loss (hence `eta`, `eta_va`), and the decay
  rate `rho = g^T Pi g / ||e||_tr^2` from a full-batch gradient `g` on a fixed probe batch.
- About 20 checkpoints (moderate): probe kernel `J_s Pi J_s^T` on `n_s` probe states and `k`
  random output projections per state (`n_s k` about 2000--8000, one VJP per row): eigenvalues
  (Q1), target alignment `C_V` (Q2), residual spectrum and trapped error (Q3), drift between
  checkpoints (Q4), and `kappa_va` (Q5).
- From the loss curve: under a nearly frozen kernel `||r(t)||^2 = sum_i e^{-2 t lambda_i} V_i^2`,
  so the training curve estimates `C_V`. Compare with the probe kernel.
- Rank regime (J4): the training problem has `P < n n_x`; the probe kernel is full rank.

Log every metric as a tidy table row: run id, system, method, `h`, `P`, `d_model`, `n_layers`,
seed, step, metric, value. Log git hash, config, and wall time for every run.

## 6. Measurements that need no training

1. `m_k = ||x^(k)||_mu` for `k = 0..4`, per wavenumber as well, from finite differences of the
   stored fine-dt trajectories (check convergence in the stencil spacing).
2. The remainder ratio `h m_{p+1}/m_p` for each `h`; the theory applies where it is small. On KS
   at 1024 points the stiff modes may make it large at any practical `h`.
3. Per-mode `z_k = h lambda_k` of the linear part, to see which modes are unresolved.
4. Time-step grid chosen from items 1--3 and reported before training.

## 7. Tests (paper Section 8)

| Test | What | Where |
|------|------|-------|
| T1 trapping | `rho(t)` decreasing, residual moving to `lambda <= 1/t`, Eq. (lower) with small drift | one mid-size run per system and method |
| T2 alignment | `C_V` of each method's target on one reference probe kernel; slow power increasing with order | one mid size |
| T3 `h`-independence | `delta_p ∝ h^p` with one `eta_p` across `h` | from the sweep |
| T4 size scaling | Section 4 | both systems |
| T5 reference-run prediction | frozen-form error of every method from one reference run vs trained values | probe set, one size |
| T6 slots and rollout | `t2_fd` vs `t2_fd3` one-step error ratio 2; rollout growth along the parasitic root | smallest size, first |

## 8. Validation before experiments

- Finite-difference `x^(k)`: convergence in stencil spacing on the stored trajectories.
- Kernel code: on a small MLP and a low-dimensional system, compare the probe kernel with the
  full kernel, and check Eq. (residual-dynamics) against the measured loss change over a few steps.
- `rho` from `g^T Pi g` against the finite-difference loss decrease per step.
- Each method with the network output replaced by its exact target reproduces `Phi_h` to data
  precision.

## 9. Milestones

1. No-training measurements (Section 6) for KS and TURB; report `m_k`, the remainder ratios,
   and the proposed `h` grid.
2. ViT, methods, and kernel code; validation (Section 8); timing runs per size.
3. T6 rollout check at the smallest size.
4. Cost estimate for the full sweep, shown to the user before launching.
5. KS sweep, T1, T2, fits; then TURB.
6. Held-out prediction test (T4), T5, final figures.
Report back to the user at the end of each milestone with numbers and plots before continuing.
