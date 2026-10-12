# Project: hardness theory for learned time steppers

## What this project is
Experiments for a paper that extends Ainslie et al. (arXiv:2608.16084, "Eigenanalysis framework
for autoregressive neural emulators of multi-scale chaotic dynamics"). That paper explains error
*amplification* (Jacobian spectrum of the learned step map). This paper explains the one-step
error through the training kernel of the network: which method (direct step, Euler step, Taylor
method of order p) a given network should use, and how the answer changes with network size.

The theory is in `paper/hardness_framework_v3.tex` (compiled: `paper/hardness_framework_v3.pdf`).
Read all of it before writing experiment code. The experiment plan is in `EXPERIMENTS.md`.
`paper/archive/` holds an out-of-date earlier version. Do not consult it for anything.

## Core idea (one paragraph)
A method is a step map `G(x) = Q_h(x) + sigma * N_theta(x)` with known part `Q_h`, scale
`sigma`, and target `V = Phi_h - Q_h`. Under Adam modeled as preconditioned gradient flow, the
residual obeys `dr/dt = -A_tr(t) r` with training kernel `A_tr = (1/n) J Pi J^T`. It decays at
rate `rho`, the Rayleigh quotient of the residual on the kernel, so the error left after training
is the residual mass on the slow eigenspectrum (eigenvalues <= 1/T). That trapped error is the
hardness. The unlearned fraction `eta(V) = ||r(T)|| / ||V||` depends on the kernel spectrum
(network and states) and the target alignment (target). For a Taylor method of order p with exact
lower slots, the standardized target is `x^(p)/m_p + O(h)`, so the trained one-step error is
`delta_p = (h^p/p!) m_p eta_p` with `eta_p` independent of `h`. Order p beats order p-1 for
`h < h*_p = p (m_{p-1}/m_p)(eta_{p-1}/eta_p)`. With the size ansatz `eta_p ~ a_p N^{-gamma_p}`
this gives the size `N*_p(h)` above which order p should be used.
In the experiments the lower slots are finite differences of past states, not exact
derivatives, so every method is one network of the same cost. With minimal-order
finite-difference slots the target is `h^p x^(p)` to leading order (`p!` times the exact-slot
target, same shape), and `h*_p = (m_{p-1}/m_p)(eta_{p-1}/eta_p)`.

## Terminology (use these exact names in code, configs, plots, and messages)
Use the defined terms of the paper and do not substitute synonyms:
method, step map, known part, scale, target, residual, protocol (P1--P5), training kernel,
validation kernel, mean kernel, preconditioner, decay rate, trapped error, hardness,
unlearned fraction, slow eigenspectrum, slow count, tail exponent, target alignment, slow power,
residual spectrum, drift, drift certificate, filtered target, frozen form, trained form,
size ansatz, plateau, rank regime, time derivative `x^(k)`, lower slot, exact slot,
state-based slot, history-based slot, finite-difference slot, parasitic root, direct step,
Euler step, Taylor method of order p, crossover step `h*_p`, crossover size `N*_p`,
tracked quantities Q1--Q5, tests T1--T6.
Method and target IDs are fixed in `EXPERIMENTS.md` Section 2. Use them as config keys and
legend labels.

## Working rules
- Compute: the global rule applies (ask before any Slurm job or heavy run, with account, QOS,
  GPUs, and time). Report the estimated cost of each sweep before running it.
- Hardware: NERSC Perlmutter (A100 GPU nodes). Data and checkpoints are on `$SCRATCH` behind the
  `training_data` and `model_checkpoints` symlinks.
- Reusable code: ViT and KS/turbulence data loading in `~/Flop_Constrained_Temporal_Scaling`;
  kernel and spectral-bias diagnostics in `~/spectral_bias`; the eigenanalysis code at
  https://github.com/cainsliewastaken/Spectral_Stability . Check these before writing new versions.
- Every experiment is driven by a config file; results go to a tidy table (one row per
  run x metric) so figures and scaling fits can be regenerated without retraining.
- Fix seeds. Log git hash, config, and wall time for every run.
- Validate each component against an analytic or numerical reference before using it
  (see EXPERIMENTS.md Section 8).
- When a result disagrees with a prediction in the paper, report it plainly with the numbers.
  Do not tune until it agrees. Disagreements are results.
