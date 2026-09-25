# Project: hardness theory for learned time steppers

## What this project is
Experiments for a paper that extends Ainslie et al. (arXiv:2608.16084, "Eigenanalysis framework
for autoregressive neural emulators of multi-scale chaotic dynamics"). That paper explains error
*amplification* (Jacobian spectrum of the learned step map). This paper explains error *injection*:
why some parameterizations of the step map (direct step, Euler step, higher-order Taylor methods,
Runge--Kutta wrappers) inject less one-step error than others.

The theory is in `paper/learned_steppers.tex` (compiled: `paper/learned_steppers.pdf`).
Read Sections 3--10 before writing experiment code. The experiment plan is in `EXPERIMENTS.md`.

## Core idea (one paragraph)
A method is a step map `x_{j+1} = Q_h(x_j) + c_p * N_theta(x_j)`, where `Q_h` is the known part
and `N_theta` is the network in the network slot. The hardness of a method is
`Lip(Phi_h - Q_h^ex)`, the Lipschitz constant of the flow map minus the known part.
Under a capacity model, injected error <= rho * hardness^alpha + floors.
Direct step: hardness ~ 1. Euler step: hardness ~ h L. Taylor order p with exact lower slots:
hardness ~ h^p L_p / p!. For a linear test mode with z = h*lambda, per-mode hardness of an affine
method is |e^z - known multiplier| (Table 3 of the paper).

## Terminology (use these exact names in code, configs, plots, and messages)
Use the defined terms from Table 2 of the paper and do not substitute synonyms:
method, step map, known part, network slot, lower slot, exact slot, state-based slot,
history-based slot, slot error (systematic part, random part), slot weight, target, hardness,
per-mode hardness, injected error, rollout error, amplification factor, test mode,
resolved mode (|z| <= 1), unresolved mode (|z| > 1), parasitic root, exponential known part,
Runge--Kutta wrapper, implicit Euler wrapper, direct step, Euler step, Taylor method of order p.
Method IDs are fixed in `EXPERIMENTS.md` Section 2. Use them as config keys and legend labels.

## Working rules
- Ask the user before launching any job expected to take more than ~1 GPU-hour, and report the
  estimated cost of each sweep before running it.
- Ask the user about available hardware (GPU type/count, cluster vs. local) at the start.
- The user's previous code for the eigenanalysis paper is at
  https://github.com/cainsliewastaken/Spectral_Stability . Check it for reusable KS data
  generation, FNO/MLP models, integrator layers, and the Jacobian eigenvalue diagnostic before
  writing new versions.
- Every experiment is driven by a config file; results go to a tidy table (one row per
  run x metric) so figures can be regenerated without retraining.
- Fix seeds. Log git hash, config, and wall time for every run.
- Validate each component against an analytic or numerical reference before using it
  (see EXPERIMENTS.md Section 7).
- When a result disagrees with a prediction in the paper, report it plainly with the numbers.
  Do not tune until it agrees. Disagreements are results.
