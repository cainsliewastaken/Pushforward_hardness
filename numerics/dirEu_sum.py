"""Check of the sharpened direct-vs-Euler theorem on Lorenz-63 at initialization (same data and
kernel as fid_lorenz.py / Figure 3).

Both networks share their initial parameters, so they have the same gradient features Psi and
kernel K. With c_D = 1 and Euler output scale c_E, the smallest errors after a step of length s are
    F_D(s)  = min_{|g|<=s} ||Phi_h - Psi g||            (direct)
    F_E(s)  = min_{|g|<=s} ||(Phi_h - id) - c_E Psi g||  (Euler)
    F_id(s) = min_{|g|<=s} ||id - Psi g||               (unbuilt identity)
The new bound is  F_D(s) >= F_id((1 + c_E) s) - F_E(s), so F_D(s) > F_E(s) whenever
F_id((1 + c_E) s) > 2 F_E(s). The old condition is F_id(s) > 2 ||Phi_h - id||.
Conventions: c_E = h (network predicts (Phi_h - id)/h) and c_E = 1 (network predicts Phi_h - id).
No training. Output: results/dirEu_sum.csv, results/fig_dirEu_sum.{png,pdf}."""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C
import lorenz as Lz

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
cfg = dict(Lz.CFG)
sol = Lz.trajectory(cfg)
n, H = cfg["n"], cfg["width"]
S = np.logspace(-1, 3, 81)


def F(lam, u, cs):
    return C.F_closed(lam, u, cs)[0]


def last_true(mask):
    """Largest s up to which mask holds on the whole prefix of the grid (0 if it fails at once)."""
    if not mask[0]:
        return 0.0
    bad = np.where(~mask)[0]
    return S[-1] if bad.size == 0 else S[bad[0] - 1]


rows, summary = [], []
for h in cfg["hs"]:
    for seed in cfg["seeds"]:
        X, Y = Lz.dataset(cfg, sol, h, seed)
        mu, sd = X.mean(0), X.std(0)
        xt = torch.tensor((X - mu) / sd, dtype=torch.float64)
        ker = C.Kernel(Lz.jac(Lz.init_params(seed, H, "cpu"), xt, H).numpy(), n)
        lam, rid = ker.reduced(X.ravel())
        _, rphi = ker.reduced(Y.ravel())
        _, rinc = ker.reduced((Y - X).ravel())
        nd = C.nrm((Y - X).ravel(), n)
        for conv, cE in (("cE=h", h), ("cE=1", 1.0)):
            FD = np.array([F(lam, rphi, s) for s in S])
            FE = np.array([F(lam, rinc, cE * s) for s in S])
            Fid_new = np.array([F(lam, rid, (1 + cE) * s) for s in S])
            Fid_old = np.array([F(lam, rid, s) for s in S])
            bound = Fid_new - FE
            viol = float(np.max(bound - FD))  # must be <= 0 (rounding) if the bound is right
            truth, new, old = FD > FE, Fid_new > 2 * FE, Fid_old > 2 * nd
            summary.append(dict(h=h, seed=seed, conv=conv, max_bound_minus_FD=viol,
                                n_new_but_false=int(np.sum(new & ~truth)),
                                s_truth=last_true(truth), s_new=last_true(new), s_old=last_true(old),
                                truth_all_s=bool(truth.all()), n_truth=int(truth.sum()),
                                n_new=int(new.sum()), n_old=int(old.sum())))
            for i, s in enumerate(S):
                rows.append(dict(h=h, seed=seed, conv=conv, s=s, F_D=FD[i], F_E=FE[i],
                                 F_id_new=Fid_new[i], F_id_old=Fid_old[i], bound=bound[i],
                                 two_dphi=2 * nd))
            q = summary[-1]
            print(f"h={h:<5} seed={seed} {conv:5}  bound-F_D max={viol:+.1e}  "
                  f"F_D>F_E up to s={q['s_truth']:8.2f} (all s: {q['truth_all_s']})  "
                  f"new cond up to s={q['s_new']:8.2f}  old cond up to s={q['s_old']:8.2f}", flush=True)

C.write_rows(os.path.join(OUT, "dirEu_sum.csv"), rows)
C.write_rows(os.path.join(OUT, "dirEu_sum_summary.csv"), summary)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
cols = {0.01: "tab:blue", 0.1: "tab:red"}
for a, conv in zip(ax, ("cE=h", "cE=1")):
    for h, c in cols.items():
        r = [q for q in rows if q["h"] == h and q["seed"] == 0 and q["conv"] == conv]
        s = [q["s"] for q in r]
        a.loglog(s, [q["F_D"] for q in r], color=c, lw=2, label=f"$F_{{\\mathrm{{D}}}}(s)$, h={h}")
        a.loglog(s, [q["F_E"] for q in r], color=c, ls="--", label=f"$F_{{\\mathrm{{E}}}}(s)$, h={h}")
        a.loglog(s, [max(q["bound"], 1e-6) for q in r], color=c, ls=":", label=f"bound, h={h}")
    a.set_ylim(1e-3, 60)
    a.set_xlabel("step length $s$")
    a.set_title("(a) $c_{\\mathrm{E}} = h$" if conv == "cE=h" else "(b) $c_{\\mathrm{E}} = 1$", fontsize=10)
    a.legend(fontsize=7)
ax[0].set_ylabel("state units")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_dirEu_sum.{ext}"), dpi=140)
print("ok")
