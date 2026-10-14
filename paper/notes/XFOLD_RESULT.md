# Cross-fitting in the optimizer (branch xfold, 2026-10-13; job 59766128, 1 node, 25 min, released by the chain)

Kept on its own branch and file to avoid clashing with REPORT items written on main by a parallel session (its items 53-54).
Code: optimizer/kronq_fast.py and scale_study.py (--xfold, --xfold_rule), scale_xfold.sh, xfold_summary.py; runs/scale_xfold/.

Question. On replayed data the learned weight's hypergradient correlates the next batch's gradient with the last push; both batches come
from one finite set, so the correlation estimates (signal + frozen sampling error)^2, not signal^2. Test: two contiguous folds, batches
alternating between them, a momentum per fold; diagnostic every 10 steps per lambda decile: g . sign(C_same fold), g . sign(C_other fold),
and a held-out gradient's agreement (measurement only). Rule: hypergradient against the push rebuilt from the other fold's momentum.
Predictions: turbulence slow deciles from epoch 2: same > cross ~ held-out; field: all equal; the rule leaves the field unchanged and
narrows the turbulence gap.

Results (seed 1; turbulence 20 epochs, lr 1e-3, budget 0.15 capped, wd 0; field 5000 steps):
| run | diag only (folds, rule off) | cross-fold rule | rule + shrink 3e-4 |
| turbulence increment | 0.1941 | 0.1946 | 0.1904 |
| turbulence direct, final (best) | 0.0996 (0.0812) | 0.0999 (0.0817) | 0.0964 (0.0805) |
| field rough | 0.1019 final, 0.0608 at step 4750 (last-interval spike) | 0.0618 | 0.0953 |
| field smooth | 1.41e-4 | 2.20e-4 | 2.49e-4 |
1. The rule changes nothing on turbulence (within 0.5 % at every checkpoint) and costs 1.55x on the smooth field: the field
   prediction FAILED on smooth. With the shrink it is worse on the field on both targets (the parallel session's item 51 finding).
2. The diagnostic confirms the bias on the slow modes: fraction of windows with positive agreement, training stream vs held-out:
   output layer middle / slow deciles 75-100 % vs 1-5 %; hidden layer slowest decile 54-83 % vs 14-19 %; fast deciles 0-1 % vs 0-2 %.
3. The cross-fold statistic is not a fix: it is biased the other way (aggregate -2.4 vs held-out -1.9 x1e-6 on the increment), since
   both folds are trained on.
4. No per-decile agreement statistic marks the validation minimum (direct, step 2250): even the held-out (oracle) aggregate is
   negative from step 500 and rises smoothly through the minimum, so an lr cut triggered by gradient-momentum agreement cannot time it.
5. Identifiability: every statistic of the training stream depends on the population signal and the frozen sampling error only through
   their sum; separating them needs data not fitted (a validation split) or an assumed effective sample size.
6. Effective sample size of the turbulence data (login-node measurement, 24 start times, 64 x 64): input-field correlation 0.92 at
   5 snapshots, 0.42 at 20, ~0.28 floor (forcing pattern); increment target 0.70 at 5, 0.22 at 10, ~0 beyond 20. Integrated times
   ~30 (input) and ~14 (target) snapshots: 400k pairs ~ 14k-28k independent samples. The validation set (4096 consecutive pairs) is
   ~140-290 independent states.
Verdict: cross-fitting closed. The explicit regulariser (weight decay, items 51-53 on main) is what separates the slow-mode gain from
memorisation; the detector line (training-only) is closed by point 5.
