"""Architecture summary figures from late_*.npy: (1) scale factor lam_k(T)/lam_k(0) vs k overlay, (2) alpha, E_T,
tail lift at k=100, trace ratio vs depth for each H and Delta; (3) CKR late ratio."""
import numpy as np, sys, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
outfig = sys.argv[1]; os.makedirs(outfig, exist_ok=True)
R = {}
for fn in sys.argv[2:]:
    r = np.load(fn, allow_pickle=True).item()
    if r['seed'] != 0 or r['n'] != 200: continue
    R[(r['H'], r['depth'], r['Delta'])] = r
Hs = sorted({k[0] for k in R}); ds = sorted({k[1] for k in R}); Ds = sorted({k[2] for k in R})
cols = {32: 'tab:blue', 96: 'tab:orange', 256: 'tab:green', 512: 'tab:red'}
fig, ax = plt.subplots(1, len(Ds), figsize=(5*len(Ds), 4.2))
for j, D in enumerate(Ds):
    for (H, d, DD), r in sorted(R.items()):
        if DD != D: continue
        lam = r['lam'][:, ::-1]; ok = lam[0] > 1e-9*lam[0][0]
        ax[j].loglog(np.arange(1, 601)[ok], (lam[-1]/np.where(ok, lam[0], 1))[ok], color=cols.get(H, 'k'), ls=['-', '--', ':', '-.'][d-1], lw=1, label=f'H={H} d={d}')
    ax[j].set_title(f'lam_k(T)/lam_k(0), Delta={D}'); ax[j].set_xlabel('k'); ax[j].legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(f'{outfig}/scale_vs_k.png', dpi=100); plt.close(fig)
def late_alpha(r):
    t = r['t']; E = r['E']; m = t > t[-1]/10; return -np.polyfit(np.log(t[m]), np.log(E[m]), 1)[0]
fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
for D, mk in zip(Ds, ['o', 's', '^', 'D']):
    for H in Hs:
        xs = [d for d in ds if (H, d, D) in R]
        if not xs: continue
        rs = [R[(H, d, D)] for d in xs]
        ax[0].plot(xs, [late_alpha(r) for r in rs], marker=mk, color=cols.get(H, 'k'), label=f'H={H} D={D}')
        ax[1].semilogy(xs, [r['E'][-1] for r in rs], marker=mk, color=cols.get(H, 'k'))
        ax[2].semilogy(xs, [r['lam'][-1][::-1][99]/r['lam'][0][::-1][99] for r in rs], marker=mk, color=cols.get(H, 'k'))
        ax[3].semilogy(xs, [r['lam'][-1][::-1][99] for r in rs], marker=mk, color=cols.get(H, 'k'))
ax[0].set_title('late exponent alpha (||E|| ~ t^-alpha)'); ax[1].set_title('||E(T)||'); ax[2].set_title('tail lift lam_100(T)/lam_100(0)'); ax[3].set_title('lam_100(T)')
for a in ax: a.set_xlabel('depth'); a.set_xticks(ds)
ax[0].legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(f'{outfig}/arch_summary.png', dpi=100); plt.close(fig)
print('saved')
