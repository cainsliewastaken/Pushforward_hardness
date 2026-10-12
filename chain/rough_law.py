"""Spectral roughness law: lam_k vs roughness index rho_k of the k-th eigenvector, at t=0 and t=T, per run."""
import numpy as np, sys, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
outfig = sys.argv[1]; os.makedirs(outfig, exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for fn in sys.argv[2:]:
    r = np.load(fn, allow_pickle=True).item(); n = r['n']; mu = r['lap_mu']; U = r['lap_U']
    j = {0.1: 0, 0.5: 1, 1.0: 2}.get(r['Delta'])
    if j is None: continue
    for i, ls in [(0, ':'), (-1, '-')]:
        K = r['snap_K'][i].astype(np.float64); w, V = np.linalg.eigh(K); w = w[::-1]; V = V[:, ::-1]
        rho = np.array([((U.T @ V[:, k].reshape(n, 3))**2).sum(1) @ mu for k in range(400)])
        # bin by roughness
        edges = np.linspace(0, 1.4, 15); mids = 0.5*(edges[1:]+edges[:-1]); lam_b = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (rho >= a) & (rho < b); lam_b.append(np.exp(np.mean(np.log(np.maximum(w[:400][m], 1e-16)))) if m.sum() else np.nan)
        ax[j].semilogy(mids, lam_b, ls, color={32: 'tab:blue', 96: 'tab:orange', 256: 'tab:green'}.get(r['H'], 'k'), alpha=0.4+0.15*r['depth'], label=f"H={r['H']} d={r['depth']} {'t=0' if i == 0 else 't=T'}")
    ax[j].set_title(f"lam vs eigenvector roughness, Delta={r['Delta']} (dotted t=0, solid t=T)"); ax[j].set_xlabel('graph-Laplacian roughness index of eigenvector'); ax[j].set_ylim(1e-12, 1e3)
ax[0].legend(fontsize=5, ncol=3)
fig.tight_layout(); fig.savefig(f'{outfig}/roughness_law.png', dpi=100); print('saved')
