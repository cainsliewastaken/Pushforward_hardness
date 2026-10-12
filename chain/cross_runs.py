"""Cross-run figures from late_*.npy: (1) kernel tail at init vs width (RMT), (2) tail eigenvalue trajectories,
(3) current-kernel-regression ratio sqrt(S_inst)/||E|| over time for all runs, (4) late alpha and E_T tables."""
import numpy as np, sys, os, glob
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
outfig = sys.argv[1]; files = sorted(sys.argv[2:]); os.makedirs(outfig, exist_ok=True)
R = {}
for fn in files:
    r = np.load(fn, allow_pickle=True).item(); R[(r['H'], r['depth'], r['Delta'])] = r
Hs = sorted({k[0] for k in R}); ds = sorted({k[1] for k in R}); Ds = sorted({k[2] for k in R})
ks = [0, 9, 49, 99, 199, 399]
# (1) init spectrum vs width, per depth (Delta=0.5 data)
fig, ax = plt.subplots(1, len(ds), figsize=(4.5*len(ds), 4))
ax = np.atleast_1d(ax)
for j, d in enumerate(ds):
    for H in Hs:
        k = (H, d, 0.5) if (H, d, 0.5) in R else next((kk for kk in R if kk[0] == H and kk[1] == d), None)
        if k is None: continue
        lam0 = R[k]['lam'][0][::-1]; ax[j].loglog(np.arange(1, 601), lam0, label=f'H={H} (D={k[2]})')
    ax[j].set_title(f'K(0) spectrum, depth {d}'); ax[j].set_xlabel('k'); ax[j].legend(fontsize=7); ax[j].set_ylim(1e-14, 1e3)
fig.tight_layout(); fig.savefig(f'{outfig}/init_spectrum_vs_width.png', dpi=100); plt.close(fig)
print('init tail eigenvalues lam_k(0), k=1,10,50,100,200,400:')
for key in sorted(R):
    lam0 = R[key]['lam'][0][::-1]; print(f'  H={key[0]:3d} d={key[1]} D={key[2]}: ' + ' '.join(f'{lam0[k]:9.2e}' for k in ks) + f'   | T: ' + ' '.join(f'{R[key]["lam"][-1][::-1][k]:9.2e}' for k in ks))
# (2) tail trajectories
fig, ax = plt.subplots(len(Ds), len(ds), figsize=(4.5*len(ds), 3.6*len(Ds)), squeeze=False)
for i, D in enumerate(Ds):
    for j, d in enumerate(ds):
        a = ax[i, j]
        for H, ls in zip(Hs, ['-', '--', ':']):
            key = (H, d, D)
            if key not in R: continue
            r = R[key]; t = r['t'][1:]; lam = r['lam'][1:, ::-1]
            for k, c in zip([0, 49, 199], ['k', 'b', 'r']):
                a.loglog(t, lam[:, k], c, ls=ls, label=f'H={H} k={k+1}' if True else None)
            a.loglog(t, r['lam_eff'][1:], 'g', ls=ls, alpha=0.6, label=f'H={H} lam_eff')
        a.set_title(f'depth {d}, Delta {D}'); a.set_xlabel('t')
        if i == 0 and j == 0: a.legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(f'{outfig}/tail_trajectories.png', dpi=100); plt.close(fig)
# (3) ratio over time
fig, ax = plt.subplots(len(Ds), len(ds), figsize=(4.5*len(ds), 3.6*len(Ds)), squeeze=False)
for i, D in enumerate(Ds):
    for j, d in enumerate(ds):
        a = ax[i, j]
        for H in Hs:
            key = (H, d, D)
            if key not in R: continue
            r = R[key]; t = r['t']; lam = r['lam'][:, ::-1]; src = (r['e0V2'] if 'e0V2' in r else r['yV2'])[:, ::-1]
            S = np.array([(src[m]*np.exp(-2*lam[m]*t[m])).sum() for m in range(len(t))])
            a.semilogx(t[1:], np.sqrt(S[1:])/r['E'][1:], label=f'H={H}')
        a.axhline(1, color='gray', ls=':'); a.set_ylim(0, 2.5); a.set_title(f'sqrt(S_inst)/||E||, depth {d}, Delta {D}'); a.set_xlabel('t')
        if i == 0 and j == 0: a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{outfig}/ckr_ratio.png', dpi=100); plt.close(fig)
print('saved', outfig)
