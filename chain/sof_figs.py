"""Figures for the second-order-feature runs: (1) q_k/H2^2 (energy of Psi_g on current mode k) vs k at several
times, with the residual spectrum eE2_k and lam_k; (2) alignment cosine vs k; (3) reduced law by rank:
sqrt(lam_k(t)) - sqrt(lam_k(0)) vs int |g| sqrt(q_k) |cos_k| ds (ignores eigenvector rotation)."""
import numpy as np, sys, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
outfig = sys.argv[1]; os.makedirs(outfig, exist_ok=True)
for fn in sys.argv[2:]:
    r = np.load(fn, allow_pickle=True).item()
    if 'sof_q' not in r or len(r['sof_q']) == 0: continue
    t = r['t']; lam = r['lam'][:, ::-1]; q = r['sof_q'][:, ::-1]; cs = r['sof_cos'][:, ::-1]; eE2 = r['eE2'][:, ::-1]; H2 = r['H2_g']; g = r['gnorm']; E = r['E']
    k = np.arange(1, lam.shape[1]+1)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    idx = [5, len(t)//6, len(t)//3, len(t)//2, 2*len(t)//3, len(t)-1]
    for i in idx:
        ax[0].loglog(k, q[i]/H2[i]**2, label=f't={t[i]:.0f}')
        ax[1].loglog(k, eE2[i]/eE2[i].sum(), label=f't={t[i]:.0f}')
        ax[2].semilogx(k, -cs[i], '.', ms=2, label=f't={t[i]:.0f}')
    ax[0].set_title(f"{r['tag']}: q_k/H2^2 (second-order features on mode k)"); ax[0].set_xlabel('rank k'); ax[0].legend(fontsize=7); ax[0].set_ylim(1e-12, 2)
    ax[1].set_title('residual energy fraction on mode k'); ax[1].set_xlabel('rank k'); ax[1].set_ylim(1e-10, 1)
    ax[2].set_title('-cos_k (alignment of second-order feature with mode feature)'); ax[2].set_xlabel('rank k'); ax[2].set_ylim(-1, 1)
    # reduced law by rank
    I = np.zeros(lam.shape[1]); pred = [np.sqrt(lam[0])]
    for i in range(1, len(t)):
        inc = 0.5*((g[i]*np.sqrt(q[i])*(-cs[i])) + (g[i-1]*np.sqrt(q[i-1])*(-cs[i-1])))*(t[i]-t[i-1])
        I += np.where(np.isfinite(inc), inc, 0); pred.append(np.sqrt(lam[0]) + I)
    pred = np.array(pred)
    for kk, c in zip([9, 49, 99, 199, 399], ['k', 'b', 'g', 'r', 'm']):
        ax[3].loglog(t[1:], lam[1:, kk], c, label=f'lam_{kk+1}'); ax[3].loglog(t[1:], np.maximum(pred[1:, kk], 1e-300)**2, c, ls=':')
    ax[3].set_title('lam_k(t) (solid) vs by-rank integrated HF rate (dotted)'); ax[3].set_xlabel('t'); ax[3].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{outfig}/sof_{r['tag']}.png", dpi=100); plt.close(fig)
    # summary numbers: where does q_k sit relative to the residual? energy-weighted mean rank of q vs of eE2
    for i in idx:
        mq = (q[i]*k).sum()/q[i].sum(); me = (eE2[i]*k).sum()/eE2[i].sum()
        tail_q = q[i][99:].sum()/q[i].sum()
        print(f"{r['tag']:28s} t={t[i]:8.1f} mean-rank(q)={mq:6.1f} mean-rank(E)={me:6.1f} q tail(k>100) frac={tail_q:8.2e} H2={H2[i]:.2f} |g|={g[i]:.2e} ||E||={E[i]:.4f} median -cos(k<=100)={np.nanmedian(-cs[i][:100]):.3f} (k>100)={np.nanmedian(-cs[i][100:400]):.3f}")
print('saved')
