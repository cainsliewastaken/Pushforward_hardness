"""Relative residual curves, Lorenz-63 at initialization (same data and kernel as fid_lorenz.py).

Both networks are trained on targets normalized to unit root-mean-square size, so both see a target
of size 1 and the same set of reachable output changes {Psi g : |g| <= s}. For a target u,
    r_u(s) = min_{|g|<=s} || u/||u|| - Psi g ||        (relative residual, between 0 and 1)
is the fraction of the target that the best step of length s leaves unbuilt. In state units
    F_D(s) = ||Phi_h|| r_D(s),   F_E(s) = ||Phi_h - id|| r_E(s),
so  F_D/F_E = (size ratio) / (difficulty ratio),  size ratio = ||Phi_h||/||Phi_h - id||,
difficulty ratio = r_E(s)/r_D(s). The Euler step has the smaller error where the difficulty ratio is
below the size ratio. Targets: raw, and centred (mean over the training inputs removed), as in the
trained runs. Output: results/relresid_lorenz.csv, results/fig_relresid_lorenz.{png,pdf}."""
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
S = np.logspace(-2, 3, 101)


def rel(ker, u):
    nu = C.nrm(u.ravel(), n)
    lam, e = ker.reduced(u.ravel() / nu)
    return nu, np.array([C.F_closed(lam, e, s)[0] for s in S])


rows = []
print("h     seed targets  size_ratio | difficulty ratio r_E/r_D at s = 0.1    1     10    100   1000 | Euler better up to s | r_D(1000) r_E(1000)")
for h in cfg["hs"]:
    for seed in cfg["seeds"]:
        X, Y = Lz.dataset(cfg, sol, h, seed)
        xs = (X - X.mean(0)) / X.std(0)
        ker = C.Kernel(Lz.jac(Lz.init_params(seed, H, "cpu"), torch.tensor(xs, dtype=torch.float64), H).numpy(), n)
        for kind in ("raw", "centred"):
            D, E, I = Y, Y - X, X
            if kind == "centred":
                D, E, I = D - D.mean(0), E - E.mean(0), I - I.mean(0)
            nD, rD = rel(ker, D)
            nE, rE = rel(ker, E)
            nI, rI = rel(ker, I)
            size = nD / nE
            diff = rE / np.maximum(rD, 1e-300)
            better = nD * rD > nE * rE                      # Euler has the smaller state error
            bad = np.where(~better)[0]
            s_upto = 0.0 if not better[0] else (S[-1] if bad.size == 0 else S[bad[0] - 1])
            for i, s in enumerate(S):
                rows.append(dict(h=h, seed=seed, targets=kind, s=s, r_D=rD[i], r_E=rE[i], r_id=rI[i],
                                 norm_D=nD, norm_E=nE, norm_id=nI, size_ratio=size, difficulty_ratio=diff[i],
                                 F_D=nD * rD[i], F_E=nE * rE[i]))
            pick = lambda v: [v[np.argmin(np.abs(S - t))] for t in (0.1, 1, 10, 100, 1000)]
            print(f"{h:<5} {seed}    {kind:8} {size:9.2f} | " + " ".join(f"{d:6.2f}" for d in pick(diff)) +
                  f" | {s_upto:10.2f} (frac {better.mean():.2f}) | {rD[-1]:.4f}   {rE[-1]:.4f}", flush=True)
C.write_rows(os.path.join(OUT, "relresid_lorenz.csv"), rows)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
cols = dict(zip(cfg["hs"], ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]))
sel = lambda h, kind: [q for q in rows if q["h"] == h and q["seed"] == 0 and q["targets"] == kind]
for a, kind in zip(ax[:2], ("raw", "centred")):
    for h in (0.01, 0.1):
        r = sel(h, kind)
        a.loglog(S, [q["r_D"] for q in r], color=cols[h], lw=2, label=f"direct target, h={h}")
        a.loglog(S, [q["r_E"] for q in r], color=cols[h], ls="--", label=f"Euler target, h={h}")
    a.set_ylim(1e-4, 1.5)
    a.set_xlabel("step length $s$ (normalized targets)")
    a.set_title(f"({'a' if kind == 'raw' else 'b'}) relative residual, {kind} targets", fontsize=10)
    a.legend(fontsize=7)
ax[0].set_ylabel("fraction of the target left unbuilt")
for h in cfg["hs"]:
    r = sel(h, "centred")
    ax[2].loglog(S, [q["difficulty_ratio"] for q in r], color=cols[h], label=f"h={h}")
    ax[2].axhline(r[0]["size_ratio"], color=cols[h], ls=":", lw=1)
ax[2].set_xlabel("step length $s$ (normalized targets)")
ax[2].set_title("(c) difficulty ratio $r_E/r_D$ (solid) and size ratio (dotted), centred", fontsize=10)
ax[2].legend(fontsize=7)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_relresid_lorenz.{ext}"), dpi=140)
print("ok")
