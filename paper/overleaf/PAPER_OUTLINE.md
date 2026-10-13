# Proposed lean structure (2026-10-12)

Keep from the existing paper
- Part I definitions: roughness function R, projector seminorms P, peak feature roughness sigma_R, H_2, lambda_1; Theorem 1 / Corollary 1 (kernel on rough directions bounded by sigma_R^2); Proposition 15 (rate of sigma_R) = our Prop. 5 in their notation.
- The static floor (held rough error, exact part): Prop. rough-energy (a)-(c), Theorem a-priori floor, Definition cutoff, Prop. cutoff-floor. These are fixed-projector statements and remain true; they are the A = 0 (no rotation) reading of the transported-cutoff bound.
- Euler-vs-direct and edge-of-stability sections, rewritten to cite the chain results (Thm Euler order-by-order; sharpening via Cor. 38/47 and the sign flip of the cross term).

Replace
- Assumption "efficient roughness supply" + Prop. two-sided band + Prop. reduced model of ignition  ->  the chain's targeting theorem (Thm 11/13: the lift is along the smoothed residual, with a computable lower bound on the Rayleigh quotient), the front/cutoff bound (Thm 48, transported projector, leakage = cross-front rotation), and the richness expansion (Thm 26; its radius is Prop. 27, truncation accuracy only, the lower bound on R unproved) for ignition/onset. The constant-c closure is replaced by c(t) = lambda_eff(t)/sigma_R(t)^2, which decays as the rough subspace's top is lifted (measured below) — the residual sits at the front, not at the top of the rough subspace.
- Proposition 20 (radius of expansions about theta_0)  ->  Prop. 27 (truncation accuracy at any expansion point) + Thm 28 (the chain as resummation, needs only R > 0). Prop. 20 becomes a corollary.

Add (the new section, ~12 pages main + appendix)
1. Identities and the chain (Prop. chain-identities, stage identity, Thms bracket/chain) — 2 pp.
2. Spike theorem (init + correlation-free trajectory form) — 1.5 pp.
3. Richness expansion, radius, resummation; term-wise corollaries (Euler order by order; extreme eigenvalues all orders; rotation vs density; seed spread bound) — 2.5 pp.
4. Moving frame and the cutoff bound (Lemma 45, Thm 46, Thm 48 with continuity transport, Cor. 49 crossing bracket) — 2 pp.
5. Applications: Euler/direct, sharpening & EoS, Adam, validation, delayed feature learning — 2 pp.
6. Why the late stage is not predictable: no chaos, hidden state, visibility, amplification; conditioning corollary — 1.5 pp.
7. What a statistical description can promise; the expected-late-error diagnostic; what is measured and what is not — 1 p.
Appendix: truncation/error propagation (Props 31-32, Thm 33, Cor. 34), numerical verifications (all n = 200, to be redone), negative results (ensemble closures, transfer sweep, universal grokking exponent).

Cut
- The per-model numerics of the early sweeps (sections 0-30 of REPORT.md); the init-only ranker (keep one sentence as the expected accuracy of a one-Jacobian statistic); the Gaussian/real-init ensemble variants (one paragraph in the appendix as negative results); the v2-v5 reduced models entirely.

Open before submission
- Review of 2026-10-12 (THEORY.md, items marked REVISED / CORRECTION): the leakage bound uses the block spectral norm; the cutoff bound's decay input and its leakage integral are run quantities not yet computed; the sign condition is a per-stage hypothesis (compressed-operator check), not discharged; Thm 36 is first order only; Prop. 27's radius bound is unproved; Adam's stage identity carries a dP term; the rank-to-roughness law (F2) is the unproved bridge to the paper's title claim; the a priori floor has never been evaluated and may be vacuous. The experiments these require are in ../notes/EXPERIMENTS.md.
- Redo at n in the thousands (every number). The sign of the init-feature part of the cross term (Lemma 15). The continuously transported cutoff input of Thm 48 along a run. The front law from the chain (the one derivation that makes the error bound closed-form).
