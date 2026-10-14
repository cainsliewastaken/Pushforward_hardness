# Late-run spikes on the field: what the instrumented runs show (branch xfold, 2026-10-14)

Jobs 59767397 (scale_spikefield, crashed at the realised-change logger, onsets only) and 59767814 (scale_spikefix, 12 runs, frozen code
8927d51). Spike = an evaluation > 3x the running best after 10 % of training.

| run | final | best | spikes | mean log-weight at 30 / 60 / 90 % | weights > e^4 at end | budget kept at 30 / 90 % |
| smooth B512 T20000, reference | 81.9 | 6.8e-5 | 38 | -1.00 / -1.53 / -0.66 | 0.1 % | 0.49 / 0.51 |
| smooth B512 T20000, hyper_budget | 7.2e-5 | 7.1e-5 | 0 | -0.68 / -3.39 / -3.17 | 10 % | 0.14 / 0.79 |
| smooth B1024 T10000, reference | 5.5e-3 | 4.9e-5 | 11 | -0.46 / -1.49 / -1.01 | 0.5 % | 0.20 / 1.00 |
| smooth B1024 T10000, hyper_budget, seed 1 / 2 | 0.0227 / 1.6e-3 | 5.5e-5 / 6.0e-5 | 6 / 5 | -0.22 / -0.72 / -2.03 (seed 1) | <0.01 % | 0.05 / 0.24 |
| rough B1024 T10000, reference | 0.0125 | 0.0077 | 5 | -0.32 / -0.27 / 0.14 | 0.7 % | 1.00 / 1.00 |
| rough B1024 T10000, hyper_budget | 0.0074 | 0.0074 | 0 | -0.47 / -0.45 / 0.16 | 0.3 % | 1.00 / 1.00 |
| rough B1024 T10000, decay 3e-4 / 1e-3 / gated 1e-3 | 0.0080 / 0.0103 / 0.0075 | | 0 / 3 / 0 | | | 1.00 |
| presets B256 T5000 with hyper_budget: rough / smooth | 0.0717 / 1.63e-4 (refs 0.0622 / 1.42e-4) | 0.0628 / 1.63e-4 | 0 / 0 | | | |

1. The weight-inflation account is refuted: the mean log-weight FALLS in every reference run; only 0.1-0.7 % of weights sit near the clamp,
   in spiking and non-spiking runs alike.
2. On the rough target the budget never binds (kept = 1.00 throughout), so --hyper_budget computes nothing there: the reference and the
   "fix" are the same computation, yet one spiked 5 times and the other not at all. Spike occurrence varies between identical runs (GPU
   non-determinism), so single-run comparisons of remedies are not evidence; that also covers the decay runs (0 / 3 / 0 spikes).
3. --hyper_budget did not stop the spikes on smooth B1024 (both seeds) and costs ~15 % at the presets. Not adopted.
4. Earlier rejections stand: not Cholesky-QR (exact QR spiked), not the budget rescale (realised change 1.0-1.3x the budget at onsets).
5. What remains consistent with every observation: spikes need a low loss and many steps, are transient with recovery, appear in AdamW too
   (21 of 149 runs at lr 1e-3 / 3e-3), and occur with the budget slack. That is the profile of the chain's sharpening / edge-of-stability
   statement (THEORY.md); untested. The test: the top kernel eigenvalue times the effective step along spiking runs, several seeds each.
