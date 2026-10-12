# Roughness and the error of a trained network

Everything for the paper *Roughness and the error of a trained network* (Overleaf project
`6ab704f85ea5c8567e1108c9`): the theory notes, the LaTeX sections, and the code of every
experimental line that feeds it. One repository, one history; run outputs stay on `$SCRATCH`
behind git-ignored symlinks.

**The current theory is the chained expansion** ("the chain"): expansions of the training
dynamics about the state at an arbitrary time, chained within the Taylor radius, with every
claim proved term by term in the resulting infinite series and resummed. It replaces the
init-only / fixed-projector statements of the earlier drafts. The files to read first are in
`paper/overleaf/` (the set uploaded to Overleaf on 2026-10-12) and `paper/notes/THEORY.md`
(sections 8-20).

## Layout

| Path | What it is | Runs live in |
|---|---|---|
| `paper/overleaf/` | **Current paper sections**: `chain_section.tex` (training through time: the chained expansion and the limits of prediction), `chain_appendix.tex` (truncation and error propagation, the n = 200 verifications), `held_rough_error_lean.tex` (the held-rough section with the closure assumption, two-sided band and reduced ignition model cut). `PAPER_OUTLINE.md` says what is kept, replaced and cut; `README.md` says how to splice them into the Overleaf main file. | |
| `paper/notes/` | `THEORY.md` (every statement, proof and status label; sections 8-20 are the chain), `WRITEUP.md` (consolidated experimental record), `REPORT.md` (the step-by-step log, sections 0-46), `optimizer_ideas.md`, `part6_segflow.md`, `part7_theory.md`, `slow_kernel_growth.md` | |
| `paper/eos_*.tex` | The sharpening / edge-of-stability application section and its bibliography | |
| `paper/old/` | Superseded drafts, kept for reference: `training_theory_2026-10-07.tex` (full paper build before the chain), `hardness_framework_v3.tex`, `rollout_history_note.tex`, `multistep_training_theory*.tex`, `archive/learned_steppers.tex` | |
| `chain/` | Lorenz-63 testbed for the held-rough and chain results: GD ground-truth harness (`l63_late.py`, `opt_stepper.py`), kernel / spike / specQ analyses, the exact staged model (`ensemble_model.py`), reduced models, init-only ranker, perturbation (chaos) tests, design sweeps. `HANDOFF.md` is the original brief; `original_upload/` is the experiment exactly as first uploaded on 2026-10-09 (first `l63test.py` / `analyze.py`, sandbox result). | `$SCRATCH/l63_held_rough/runs` (symlink `chain/runs`) |
| `optimizer/` | The optimizer line built on the theory: SegFlow (Krylov segments within the Taylor radius), KAdam (Adam in the kernel eigenbasis), KronSign / Kronecker-Adam / SOAP-style and KronRoot (Newton-Schulz whitening) per-layer optimizers, Muon and Adam baselines, the field and KS problems, equal-budget tuning and speed sweeps. Imports `l63_design`, `l63test_torch`, `l63_late` from `chain/`. | `$SCRATCH/l63_held_rough/runs` (symlink `optimizer/runs`) |
| `multistep/` | Multistep (unrolled) training of learned steppers on Lorenz-63: contraction along the rollout error, injected vs retained error, rollout kernel, curvature-scaled and line step rules, nudged training and the online kappa rule; `ks/` and `turb2d/` prototypes. Imported with its own git history. | `$SCRATCH/multistep_roughness` (symlink `multistep/results`) |
| `numerics/` | One-step hardness numerics (sine and Lorenz targets, Krylov series, seminorm floors, soft bounds, wide / deep steppers) and `eos/` (edge-of-stability measurements). `report.md` and `EXPERIMENTS.md` describe them. | `$SCRATCH/pushforward_eos`, `$SCRATCH/pf_logs`, `numerics/results*` (untracked) |
| `validate/` | `validate_tt2.py`: numerical validation of the theorems of the 2026-10-07 paper build (tests 1-8, GPU, float64) | `$SCRATCH/validate_training_theory` (symlink `validate/runs`) |

Symlinks at the root (`training_data`, `model_checkpoints`, `py2d`) point at the shared data
and checkpoint trees on scratch and are git-ignored.

## Status (2026-10-12)

- The chain theory is written up (`paper/overleaf/chain_section.tex`, `THEORY.md` 8-20) and
  the held-rough section has been cut down to the statements the chain supports.
- **Every experiment in `chain/` and `optimizer/` so far used n = 200 training points.** By
  decision (2026-10-11) all of them must be redone at n in the thousands before any number is
  quoted; the theorems carry n as a symbol and are unaffected.
- Optimizer line: SegFlow loses to Adam per FLOP (REPORT.md 36, 40); the Kronecker-eigenbasis
  line (KronSign / Kronecker-Adam, KronRoot) is the open direction, with the reparametrised
  rotations and the basis-refresh schedule under test (REPORT.md 41-46, `optimizer_ideas.md`).
- Multistep: 13 follow-ups recorded in `multistep/results.md`; the KS and 2D-turbulence
  prototypes and the online kappa rule are the newest, untested at scale.

## Syncing with the pre-repository working copy

The L63 sessions that predate this repository edit `$SCRATCH/l63_held_rough/code` and `report/`
directly. Until the last of them has ended, that folder is the live copy of `chain/`, `optimizer/`
and `paper/notes`: run `scripts/sync_l63_code.sh` (dry run) and `scripts/sync_l63_code.sh --apply`
to pull newer or new files into the repo, then commit. Delete the scratch `code/` and `report/`
folders only after a sync that reports nothing to copy. The run tree `runs/` stays on scratch.

## Running

All Python runs use Shifter with `nersc/pytorch:26.01.01`; see `CLAUDE.md` for the compute
rules. The L63 scripts locate their run tree through `L63_ROOT` (default
`$SCRATCH/l63_held_rough`); shell launchers locate their own directory, so they can be run
from anywhere. Scripts that read `runs/...` relative paths expect to be run from `chain/` or
`optimizer/`, where `runs` is the symlink.
