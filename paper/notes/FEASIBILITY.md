# Optimizer components: measured benefit vs online cost (ledger, started 2026-10-12)

Rule (user, 2026-10-12): keep a component only if its online cost is modest; if costs grow too large, fall back to
something basic. Costs are per step in forward-pass equivalents (Adam = 3) and extra memory in units of the parameter
count P (Adam = 2P). "w" = layer width. Benefits are per step unless marked per FLOP; L63 testbed unless noted.

| component | measured benefit | compute / step | memory | scales? | status |
|---|---|---|---|---|---|
| exact per-layer GN eigenbasis (KAdam) | sign push in it: 4x Adam (rough), 50x (smooth) at n = 20000 | ~900 (2.8e6 / 3000) | J: N x P | no | research instrument only |
| Kronecker (K-FAC) eigenbasis (KronQ) | q effect carries over at n = 500 (smoke); large-n not yet run | ~1.4 + amortised eigh | 1P momentum + 2(in^2 + out^2) per layer (~4P for square layers) | only with blocking / one-sided / sketches | to test per FLOP |
| sign push per mode (vs RMS / magnitude) | best rule in every test | free given basis | none | yes | keep |
| Gauss-Newton cap per mode | 4-5x at n = 2000 (q -0.25); not isolated at n = 20000 | free given basis | none | yes | keep, isolate at large n |
| leverage exponent q | substitutes for lr: tuned (q, lr) pairs tie at n = 20000 | free given basis | none | yes | optional (one knob either way) |
| online q from Taylor remainder | holds remainder at target; benefit at large n unproven | +0.1 to +0.3 (round-robin check every 10 steps on 10 % of data) | none | yes | keep only if it replaces lr tuning |
| derived trust-region Newton (GN + measured residual curvature) | loses 3-4x to sign push | +1 (trial forward) + probes | same as basis | basis-limited | dropped |
| low-rank trust-region Newton (32 modes) | loses badly (0.60 vs 0.10 rough) | ~+5 | 32P | memory-limited | dropped |
| front / wave windows on the push | negative on training error | free | none | yes | dropped |
| wave-keyed momentum damping | tie | free | none | yes | dropped |
| adaptive (error-controlled) cache refresh | predicted refresh cost ~ log T | measurement reuses refresh work | none | yes | to implement and test |
| centred initialisation | 1.4x on rough with sign push, nothing elsewhere | free | 1x outputs | yes | optional |

Open cost items: KronQ per-FLOP comparison vs AdamW at n = 20000 and on the field problem (minibatch); memory
reduction for the Kronecker factors (see Section on SOAP / Muon memory in the conversation of 2026-10-12).
