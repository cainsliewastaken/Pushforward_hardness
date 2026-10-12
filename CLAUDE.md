# Project: Roughness and the error of a trained network

## What this repository is
The single repository for the paper *Roughness and the error of a trained network* (Overleaf
project `6ab704f85ea5c8567e1108c9`) and every experimental line behind it. It extends Ainslie
et al. (arXiv:2608.16084, "Eigenanalysis framework for autoregressive neural emulators of
multi-scale chaotic dynamics"), which explains error *amplification*; this paper explains the
error of training itself through the training kernel and target roughness.

**The current theory is the chained expansion.** Read, in this order, before touching theory
or code: `paper/overleaf/README.md`, `paper/overleaf/PAPER_OUTLINE.md`,
`paper/notes/THEORY.md` (sections 8-20 are the chain; 1-7 are kept for their exact
statements), then `paper/overleaf/chain_section.tex`. The earlier init-only / fixed-projector
drafts are in `paper/old/` and are superseded; do not take statements from them without
checking `THEORY.md` first.

## Core idea (one paragraph)
Training is gradient flow (or Adam as preconditioned flow) on the residual E = Yhat - Y with
kernel K = Psi Psi^T / n. At any time t the dynamics can be expanded about the current state;
the expansion is valid inside its Taylor radius, and the next expansion starts where the
previous one ends. The chain of such expansions is an exact, infinite-series description of
training from 0 to T. Statements are proved term by term in that series and then resummed,
so claims about the error at T (Euler vs direct order by order, spike formation whenever the
network learns, sharpening and the edge of stability, the cutoff / front bound with its
rotation remainder, why the late error is not predictable from initialization) follow
without infinite-width, activation-specific or MLP-specific assumptions. Roughness enters
through the roughness functions R and the peak feature roughness sigma_R: the part of the
error along a rough direction loses energy only through sigma_R.

## Terminology (use the paper's names in code, configs, plots and messages)
roughness function, projector seminorm, peak feature roughness sigma_R, H_2, lambda_1,
residual, training kernel, held rough error, cutoff, front, moving frame, transported
projector, rotation remainder, chain, stage, stage identity, Taylor radius, richness
expansion, resummation, spike, lift, lazy / rich regime, Euler step, direct step, sharpening,
edge of stability. For the one-step numerics also: method, step map, known part, scale,
target, unlearned fraction, slow eigenspectrum, trapped error, hardness, Taylor method of
order p, crossover step. Do not invent synonyms; if a quantity has no name in `THEORY.md`,
name it there first.

## Layout
`paper/` (overleaf = current sections; notes; old), `chain/` (L63 held-rough and chain
experiments), `optimizer/` (SegFlow, KAdam, KronSign / Kronecker-Adam, KronRoot, baselines),
`multistep/` (unrolled training; own history), `numerics/` (one-step hardness checks, `eos/`),
`validate/` (theorem validation). See `README.md` for the table and where each line's runs live.

## Working rules
- Compute: the global rules apply (ask before any Slurm job or heavy run, with account, QOS,
  GPUs and time; state the estimated cost of a sweep before running it). Slurm account used
  so far for this project: m5326, confirmed per run.
- Every Python run goes through Shifter `nersc/pytorch:26.01.01`.
- Outputs, checkpoints and logs go to `$SCRATCH`, never into the repo: `chain/runs` and
  `optimizer/runs` -> `$SCRATCH/l63_held_rough/runs`, `multistep/results` ->
  `$SCRATCH/multistep_roughness`, `validate/runs` -> `$SCRATCH/validate_training_theory`,
  `numerics/eos/results` -> `$SCRATCH/pushforward_eos`. The L63 scripts read the run tree
  from `L63_ROOT` (default `$SCRATCH/l63_held_rough`).
- `optimizer/` scripts import `l63_design`, `l63test_torch`, `l63_late` from `chain/` (the
  path is added at the top of each script); keep the shared modules in `chain/`.
- **n = 200 is a toy of a toy.** Every chain / optimizer verification, exponent, sign, fit and
  ranker so far is an n = 200 statement (user decision 2026-10-11). Use n in the thousands
  for anything new, and do not quote old numbers in the paper until redone.
- Do not propose forecasting E(t) from initialization or early training again: init-only and
  early-extrapolation predictors, Gaussian virtual ensembles and member-averaged kernels were
  tested and rejected (WRITEUP.md 4, 11; REPORT.md 19, 22, 40). The chain is a theoretical
  object for proofs; its truncations are an appendix matter, not a predictor.
- No infinite-width, tanh-only or MLP-only assumptions in new theory: a basic network whose
  activation has a derivative (ReLU included) is the setting. Adam must be covered.
- Keep the matter-of-fact register of the existing tex and notes: claims with a pointer or a
  number, no intensifiers. When a result disagrees with a prediction, report it with the
  numbers; disagreements are results.
- Fix seeds; log git hash, config and wall time for every run; keep results regenerable from
  saved tables without retraining.
- Git: commit with clear messages; ask before pushing; never force-push.
