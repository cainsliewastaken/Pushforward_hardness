"""Check: roughness index of the k-th eigenvector vs k, for K0 and K(T), across runs (widths, depths, scales, Delta).
Reports spread across runs per rank bin and saves an overlay figure."""
import numpy as np, sys, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
outfig = sys.argv[1]; os.makedirs(outfig, exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8)); curves0 = []; curvesT = []; tags = []
for fn in sys.argv[2:]:
    r = np.load(fn, allow_pickle=True).item(); n = r['n']
    if n != 200: continue
    mu = r['lap_mu']; U = r['lap_U']
    def rough_curve(K):
        w, V = np.linalg.eigh(K.astype(np.float64)); V = V[:, ::-1]
        return np.array([((U.T @ V[:, k].reshape(n, 3))**2).sum(1) @ mu for k in range(600)])
    c0 = rough_curve(r['snap_K'][0]); cT = rough_curve(r['snap_K'][-1])
    curves0.append(c0); curvesT.append(cT); tags.append(r['tag'])
    k = np.arange(1, 601); sm = lambda c: np.convolve(c, np.ones(15)/15, mode='same')
    ax[0].semilogx(k, sm(c0), lw=0.8, alpha=0.6); ax[1].semilogx(k, sm(cT), lw=0.8, alpha=0.6, label=r['tag'] if len(tags) <= 12 else None)
ax[0].set_title(f'roughness of k-th eigenvector of K0 ({len(tags)} runs)'); ax[1].set_title('same for K(T)'); ax[1].legend(fontsize=5)
for a in ax: a.set_xlabel('rank k'); a.set_ylabel('graph-Laplacian roughness index (15-pt smoothed)'); a.set_ylim(0, 1.4)
fig.tight_layout(); fig.savefig(f'{outfig}/rank_roughness.png', dpi=100)
C0 = np.array(curves0); CT = np.array(curvesT)
print(f"{len(tags)} runs. Per rank bin: mean roughness (K0 / K(T)), spread across runs = std of bin means (K0 / K(T)), and mean within-run bin std")
for a, b in [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, 600)]:
    m0 = C0[:, a:b].mean(1); mT = CT[:, a:b].mean(1)
    print(f"  ranks {a+1:3d}-{b:3d}: {m0.mean():.3f} / {mT.mean():.3f}   spread {m0.std():.3f} / {mT.std():.3f}   within-run std {C0[:, a:b].std(1).mean():.3f} / {CT[:, a:b].std(1).mean():.3f}")
