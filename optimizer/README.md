# optimizer/

## THE MAIN OPTIMIZER: `kronq_main.py` (class `KronQ`)

This is the method. It is self-contained (it does not import the research code) and contains only what was kept:
the per-layer Kronecker eigenbasis (blocks of 256), the sign of the momentum times a learned per-mode weight (one-step hypergradient),
weights in ordinary coordinates with the gradient rotated in and the update rotated out, the basis work in fp32, the QR guard, and
optional decoupled decay per step for finite data. What was dropped, and why, is listed in its docstring and in
`paper/notes/OPTIMIZER_PLAN.md` (section 2b and the results tables).

Run it through the harness with the preset (the preset sets every option; explicit options override; dropped options are refused):

    python scale_study.py --preset current --lr 1e-3 ...                    # fresh data (the field problem)
    python scale_study.py --preset current --lr 2e-4 --wd_abs 1e-3 ...     # finite / replayed data (turbulence)

`--preset current` selects `--opt kronq_main`. Equivalence: on CPU it reproduces the research configuration
`--opt kronqf --orig 1 --orig_hp rot,stats,refresh --qr_guard 1 --cap 0 --rem 0` to every printed digit (200 steps, with and without decay).

## Research code (kept to reproduce the record; do not use for new runs)

- `kronq_fast.py` (`--opt kronqf`): the research harness with every component tried (budget, cap, slow adaptation, cross-fitting,
  shrinks, gating, diagnostics), weights in the rotating basis.
- `kronq_orig.py` (`--opt kronqf --orig 1`): the refactor as developed, with its precision switches.
- `slow_adapt.py`, `mechanism.py`, `tracking.py`, `preflight.py`: diagnostics and dropped controllers.
- `scale_*.sh`: one launcher per experiment, named in `paper/notes/REPORT.md`; their outputs live under `$SCRATCH/l63_held_rough/runs/`.
