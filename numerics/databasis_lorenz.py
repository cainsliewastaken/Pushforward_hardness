"""Data-only series for the one-step bound, Lorenz-63 at initialization (same data/kernel as
fid_lorenz.py).

Data basis: eigenvectors v_1, v_2, ... of the graph Laplacian on the standardized training inputs
(kNN graph, k = 10), lifted to the three output components. V_M = span of the first M graph modes
(dimension 3M). It is fixed by the training inputs alone.

For a target u and output scale c,
    F_{u,M}(s) = max_{phi in V_M, ||phi||<=1} ( <phi,u> - c s ||phi||_K )
               = min_{|g|<=s} || P_M (u - c Psi g) ||,
the kernel-ridge closed form with the kernel written in the data basis, G_M = P_M K P_M, and the
target replaced by its first M graph coefficients. F_{u,M} is nondecreasing in M and equals the
exact F_u(s) at M = n.

Outputs (new files only): results/databasis_lorenz.csv, results/databasis_lorenz_cert.csv,
results/fig_databasis_lorenz.{png,pdf}."""
import os
import sys
import time
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
HS, SEED, KNN = [0.01, 0.1], 0, 10
MS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, n]
S_FEW = [1.0, 3.0, 10.0, 30.0, 100.0]
S_GRID = np.logspace(-1, 3, 81)


def graph_modes(x, k):
    """Euclidean-orthonormal eigenvectors (columns) of the kNN graph Laplacian, smooth to rough."""
    D = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    rg = np.median(D[np.arange(len(x)), idx[:, -1]])
    W = np.zeros_like(D)
    rows = np.repeat(np.arange(len(x)), k)
    W[rows, idx.ravel()] = np.exp(-(D[rows, idx.ravel()] / rg) ** 2)
    W = np.maximum(W, W.T)
    mu, V = np.linalg.eigh(np.diag(W.sum(1)) - W)
    return mu, V


def upto(mask):
    if not mask[0]:
        return 0.0
    bad = np.where(~mask)[0]
    return float(S_GRID[-1] if bad.size == 0 else S_GRID[bad[0] - 1])


rows, cert, curves = [], [], {}
for h in HS:
    t0 = time.time()
    X, Y = Lz.dataset(cfg, sol, h, SEED)
    mu_x, sd = X.mean(0), X.std(0)
    xs = (X - mu_x) / sd
    J = Lz.jac(Lz.init_params(SEED, H, "cpu"), torch.tensor(xs, dtype=torch.float64), H).numpy()
    d = J.shape[1]
    _, V = graph_modes(xs, KNN)
    # coefficients in the data basis: mode j, output component c
    Jc = np.einsum("ij,icd->jcd", V, J.reshape(n, 3, d))
    targets = {"id": X, "inc": Y - X, "Phi": Y}
    uc = {k: V.T @ u for k, u in targets.items()}              # (n modes, 3)
    nd = C.nrm((Y - X).ravel(), n)
    ker = C.Kernel(J, n)
    exact = {k: ker.reduced(u.ravel()) for k, u in targets.items()}
    scales = {"id": 1.0, "Phi": 1.0, "inc": 1.0, "inc_cE=h": h}
    tkey = {"id": "id", "Phi": "Phi", "inc": "inc", "inc_cE=h": "inc"}
    kerM = {M: C.Kernel(Jc[:M].reshape(3 * M, d), n) for M in MS}

    def F_exact(name, s):
        lam, e = exact[tkey[name]]
        return C.F_closed(lam, e, scales[name] * s)[0]

    def F_M(name, M, s):
        lam, e = kerM[M].reduced(uc[tkey[name]][:M].ravel())
        return C.F_closed(lam, e, scales[name] * s)[0]

    for name in scales:
        nu = C.nrm(targets[tkey[name]].ravel(), n)
        for M in MS:
            energy = float(np.sum(uc[tkey[name]][:M] ** 2) / n) / nu ** 2
            for s in S_FEW:
                rows.append(dict(h=h, target=name, M=M, s=s, F_M=F_M(name, M, s), F=F_exact(name, s),
                                 norm=nu, energy_frac=energy))
    # certificates on the fine s grid
    Fid = np.array([F_exact("id", s) for s in S_GRID])
    FD = np.array([F_exact("Phi", s) for s in S_GRID])
    FE1 = np.array([F_exact("inc", s) for s in S_GRID])
    FEh = np.array([F_exact("inc_cE=h", s) for s in S_GRID])
    curves[h] = dict(Fid=Fid, two_nd=2 * nd, FidM={})
    for M in MS:
        FidM = np.array([F_M("id", M, s) for s in S_GRID])
        FDM = np.array([F_M("Phi", M, s) for s in S_GRID])
        curves[h]["FidM"][M] = FidM
        assert np.all(FidM <= Fid + 1e-8 * Fid.max()) and np.all(FDM <= FD + 1e-8 * FD.max())
        cert.append(dict(h=h, M=M, dim=3 * M,
                         s_cert_id=upto(FidM > 2 * nd),            # data-basis analogue of Eq. dirEucond
                         s_cert_id_exact=upto(Fid > 2 * nd),
                         s_cert_D_vs_E_cE1=upto(FDM > FE1), s_truth_cE1=upto(FD > FE1),
                         s_cert_D_vs_E_cEh=upto(FDM > FEh), s_truth_cEh=upto(FD > FEh)))
    print(f"h={h}: n={n} d={d} rank={ker.rank} ||id||={C.nrm(X.ravel(), n):.2f} ||inc||={nd:.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    print("  M   energy(id) energy(inc) | F_M/F at s=10: id    inc    Phi | cert s: id>2|inc|  D>E(cE=1)  D>E(cE=h)")
    for M, q in zip(MS, [c for c in cert if c["h"] == h]):
        g = lambda t, key: [r for r in rows if r["h"] == h and r["target"] == t and r["M"] == M and r["s"] == 10.0][0][key]
        print(f"{M:5d} {g('id', 'energy_frac'):9.4f} {g('inc', 'energy_frac'):10.4f} |"
              f" {g('id', 'F_M') / g('id', 'F'):14.3f} {g('inc', 'F_M') / g('inc', 'F'):6.3f} {g('Phi', 'F_M') / g('Phi', 'F'):6.3f} |"
              f" {q['s_cert_id']:12.2f} {q['s_cert_D_vs_E_cE1']:10.2f} {q['s_cert_D_vs_E_cEh']:10.2f}", flush=True)
    q = [c for c in cert if c["h"] == h][-1]
    print(f"  exact: id>2|inc| up to s={q['s_cert_id_exact']:.2f}; F_D>F_E up to s={q['s_truth_cE1']:.2f} (cE=1), "
          f"{q['s_truth_cEh']:.2f} (cE=h)", flush=True)

C.write_rows(os.path.join(OUT, "databasis_lorenz.csv"), rows)
C.write_rows(os.path.join(OUT, "databasis_lorenz_cert.csv"), cert)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
h0 = HS[0]
ax[0].loglog(S_GRID, curves[h0]["Fid"], "k-", lw=2, label="$F_{\\mathrm{id}}(s)$")
for M, c in zip([1, 3, 10, 100, 1000], ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]):
    ax[0].loglog(S_GRID, np.maximum(curves[h0]["FidM"][M], 1e-6), color=c, ls="--", label=f"first {M} graph modes")
for h, c in zip(HS, ["tab:orange", "tab:red"]):
    ax[0].axhline(curves[h]["two_nd"], color=c, ls=":", lw=1, label=f"$2\\|\\Phi_h - \\mathrm{{id}}\\|$, h={h}")
ax[0].set_ylim(1e-3, 50)
ax[0].set_xlabel("step length $s$")
ax[0].set_ylabel("state units")
ax[0].set_title(f"(a) unbuilt identity from the data basis, h={h0}", fontsize=10)
ax[0].legend(fontsize=7)
for name, c, lab in (("id", "k", "identity"), ("inc", "tab:blue", "$\\Phi_h - \\mathrm{id}$"), ("Phi", "tab:green", "$\\Phi_h$")):
    r = [q for q in rows if q["h"] == h0 and q["target"] == name and q["s"] == 10.0]
    ax[1].semilogx([q["M"] for q in r], [q["F_M"] / q["F"] for q in r], "-o", ms=3, color=c, label=lab)
ax[1].set_xlabel("number of graph modes $M$")
ax[1].set_ylabel("$F_{u,M}(s) / F_u(s)$")
ax[1].set_title(f"(b) convergence of the data-only series, s=10, h={h0}", fontsize=10)
ax[1].legend(fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_databasis_lorenz.{ext}"), dpi=140)
print("ok")
