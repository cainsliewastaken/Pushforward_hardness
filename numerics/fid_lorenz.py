"""F_id(s) = min_{|g|<=s} ||id - c Psi g|| for the direct network at initialization on Lorenz-63
(the best roughness-function bound on the unbuilt identity after a step of length s), its Krylov
values F_{id,M}(s), and the comparison F_id(s) > 2||Phi_h - id|| of the direct-vs-Euler theorem.
Only Jacobians at initialization; no training. Output: results/fid_lorenz.csv, fig_fid_lorenz.{png,pdf}."""
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
MS = [1, 2, 5, 10, 20]
rows = []
for h in cfg["hs"]:
    for seed in cfg["seeds"]:
        X, Y = Lz.dataset(cfg, sol, h, seed)
        mu, sd = X.mean(0), X.std(0)
        xt = torch.tensor((X - mu) / sd, dtype=torch.float64)
        idv, dphi = X.ravel(), (Y - X).ravel()
        nd, nid, nPhi = C.nrm(dphi, n), C.nrm(idv, n), C.nrm(Y.ravel(), n)
        ker = C.Kernel(Lz.jac(Lz.init_params(seed, H, "cpu"), xt, H).numpy(), n)
        lam, idr = ker.reduced(idv)
        for s in S:
            F = C.F_closed(lam, idr, s)[0]
            FM, _ = C.krylov_F(lam, idr, s, max(MS))
            rows.append(dict(h=h, seed=seed, s=s, F_id=F, dphi=nd, norm_id=nid, norm_Phi=nPhi, d0=idr[-1],
                             **{f"F_id_M{M}": FM[M - 1] for M in MS}))
        print(f"h={h} seed={seed} ||id||={nid:.3f} 2||Phi-id||={2 * nd:.3f} d0={idr[-1]:.4f} "
              f"F_id(1)={C.F_closed(lam, idr, 1.0)[0]:.3f} F_id(10)={C.F_closed(lam, idr, 10.0)[0]:.3f} "
              f"F_id(100)={C.F_closed(lam, idr, 100.0)[0]:.3f}", flush=True)
C.write_rows(os.path.join(OUT, "fid_lorenz.csv"), rows)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
cols = dict(zip(cfg["hs"], ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for h in cfg["hs"]:
    r = [q for q in rows if q["h"] == h and q["seed"] == 0]
    ax[0].loglog([q["s"] for q in r], [q["F_id"] for q in r], color=cols[h], label=f"$F_{{\\mathrm{{id}}}}(s)$, h={h}")
    ax[0].axhline(2 * r[0]["dphi"], color=cols[h], ls=":", lw=1)
ax[0].set_xlabel("step length $s$ (parameter units, $c_{\\mathrm{D}} = 1$)")
ax[0].set_ylabel("state units")
ax[0].set_title("(a) $F_{\\mathrm{id}}(s)$ (solid) and $2\\|\\Phi_h - \\mathrm{id}\\|$ (dotted)", fontsize=10)
ax[0].legend(fontsize=7)
r = [q for q in rows if q["h"] == cfg["hs"][0] and q["seed"] == 0]
ax[1].loglog([q["s"] for q in r], [q["F_id"] for q in r], "k-", lw=2, label="$F_{\\mathrm{id}}(s)$")
for M, c in zip(MS, ["#fdd0a2", "#fdae6b", "#fd8d3c", "#e6550d", "#a63603"]):
    ax[1].loglog([q["s"] for q in r], [max(q[f"F_id_M{M}"], 1e-6) for q in r], color=c, ls="--", label=f"$F_{{\\mathrm{{id}},{M}}}(s)$")
ax[1].set_ylim(1e-3, 50)
ax[1].set_xlabel("step length $s$")
ax[1].set_title("(b) Krylov values $F_{\\mathrm{id},M}(s)$, seed 0", fontsize=10)
ax[1].legend(fontsize=7)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_fid_lorenz.{ext}"), dpi=140)
print("ok")
