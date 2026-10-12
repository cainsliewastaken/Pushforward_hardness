"""Relative difficulty of the direct and Euler targets on Lorenz-63 at initialization (same data
and kernel as fid_lorenz.py). Both networks share the kernel K at initialization.

Each target u is normalized to unit size, u_hat = u / ||u||, so there is no head start and no
output-scale effect (c = 1 for every target).
  A. relative best-step residual  r_u(s) = min_{|g|<=s} ||u_hat - Psi g||  (= F_u(s ||u||)/||u||),
     and the step length needed to reach relative residual 0.5, 0.1, 0.01;
     the part of u_hat outside the range of K (relative floor).
  B. relative removal rate under gradient flow, rho(u_hat) = <u_hat, K u_hat> (= -d/dt log||E|| at
     the start), reported as rho / lambda_1; equivalently the s -> 0 slope of r_u(s) is sqrt(rho).
Targets: direct Phi_h, Euler increment Phi_h - id, and the identity, raw and centred (the network
has output biases, so a constant is built for free).
Output: results/ab_lorenz.csv, results/ab_lorenz_curves.csv, results/fig_ab_lorenz.{png,pdf}."""
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
S = np.logspace(-3, 2, 61)
LEVELS = [0.5, 0.1, 0.01]
rows, curves = [], []
for h in cfg["hs"]:
    for seed in cfg["seeds"]:
        X, Y = Lz.dataset(cfg, sol, h, seed)
        mu, sd = X.mean(0), X.std(0)
        xt = torch.tensor((X - mu) / sd, dtype=torch.float64)
        ker = C.Kernel(Lz.jac(Lz.init_params(seed, H, "cpu"), xt, H).numpy(), n)
        for centred in (False, True):
            Xc, Yc = (X - X.mean(0), Y - Y.mean(0)) if centred else (X, Y)
            targets = dict(direct=Yc, euler=Yc - Xc, identity=Xc)
            for name, U in targets.items():
                u = U.ravel()
                nu = C.nrm(u, n)
                lam, ur = ker.reduced(u / nu)
                rho = ker.knorm(u / nu) ** 2
                row = dict(h=h, seed=seed, centred=centred, target=name, norm=nu,
                           rho_over_lam1=rho / ker.lam1, slope=np.sqrt(rho), rel_floor=abs(ur[-1]),
                           kinv=float(np.sqrt(np.sum(ur[:-1] ** 2 / lam[:-1]))), rank=ker.rank)
                for eps in LEVELS:
                    row[f"s_to_{eps}"] = C.s_needed(lam, ur, 1.0, eps)
                rows.append(row)
                for s in S:
                    curves.append(dict(h=h, seed=seed, centred=centred, target=name, s=s,
                                       r=C.F_closed(lam, ur, s)[0]))
        for centred in (False, True):
            q = {r["target"]: r for r in rows if r["h"] == h and r["seed"] == seed and r["centred"] == centred}
            print(f"h={h:<5} seed={seed} centred={int(centred)} | "
                  + " | ".join(f"{t[:3]}: rho/l1={q[t]['rho_over_lam1']:.2e} floor={q[t]['rel_floor']:.1e} "
                               f"s(.5)={q[t]['s_to_0.5']:.3g} s(.1)={q[t]['s_to_0.1']:.3g} "
                               f"s(.01)={q[t]['s_to_0.01']:.3g}" for t in ("direct", "euler", "identity")),
                  flush=True)
C.write_rows(os.path.join(OUT, "ab_lorenz.csv"), rows)
C.write_rows(os.path.join(OUT, "ab_lorenz_curves.csv"), curves)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
sty = dict(direct=("tab:red", "-"), euler=("tab:blue", "-"), identity=("gray", ":"))
for a, h in zip(ax, (0.01, 0.1)):
    for name, (c, ls) in sty.items():
        r = [q for q in curves if q["h"] == h and q["seed"] == 0 and q["centred"] and q["target"] == name]
        a.loglog([q["s"] for q in r], [max(q["r"], 1e-5) for q in r], color=c, ls=ls, lw=2, label=name)
    a.set_xlabel("step length $s$ (target normalized to unit size)")
    a.set_title(f"h = {h}, centred targets, seed 0", fontsize=10)
    a.set_ylim(1e-4, 1.5)
    a.legend(fontsize=8)
ax[0].set_ylabel("relative best-step residual $r_u(s)$")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_ab_lorenz.{ext}"), dpi=140)
print("ok")
