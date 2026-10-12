"""Per-layer analysis of lay_*.npy. At each snapshot (kernel time tau):
 (1) tail attribution: share of sum_{k>100} lam_k(K) carried by each layer block (v_k^T K_l v_k);
 (2) dW_l spikes: top singular value, eff. rank of dW_l, roughness of the spike's input feature and output feature;
 (3) W_l ESD: top singular value / MP bulk edge, Hill tail exponent alpha of the top 25% eigenvalues of W^T W."""
import numpy as np, sys, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
outfig = sys.argv[1]; os.makedirs(outfig, exist_ok=True)
def eff_rank(sv):
    p = sv**2/(sv**2).sum(); p = p[p > 0]; return float(np.exp(-(p*np.log(p)).sum()))
def hill_alpha(ev, frac=0.25):
    ev = np.sort(ev)[::-1]; k = max(3, int(len(ev)*frac)); top = ev[:k]
    if top[-1] <= 0: return np.nan
    return 1 + k/np.sum(np.log(top/top[-1]))
for fn in sys.argv[2:]:
    r = np.load(fn, allow_pickle=True).item(); L = len(r['snap_svW'][0]); taus = r['snap_t']*r['lam1_0']
    print(f"== {r['tag']}  (layers 0..{L-1}; last = output layer; fan_in: 3, H.., H)")
    print(f"   {'tau':>8s} {'t':>8s} {'E_tr':>7s} {'E_val':>7s} | tail share per layer (k>100) | lam100(K) | top sv dW per layer | effrank dW | rough(spike in, out) layer0, layer1 | top sv W / MP edge | Hill alpha per layer")
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    shares = []; amps = []; alphas = []; roughs = []
    for j in range(len(taus)):
        K = r['snap_K'][j].astype(np.float64); w, V = np.linalg.eigh(K); order = np.argsort(-w); w = w[order]; V = V[:, order]
        Vt = V[:, 100:]; tail = w[100:].sum()
        sh = [float(np.einsum('ij,ji->i', Vt.T @ r['snap_Kl'][j][l].astype(np.float64), Vt).sum()/tail) for l in range(L)]
        svW = r['snap_svW'][j]; svd = r['snap_svdW'][j]
        fan_in = [3] + [r['H']]*(L-1); fan_out = [r['H']]*(L-1) + [3]
        edge = [np.sqrt(1/fan_in[l])*(np.sqrt(fan_in[l]) + np.sqrt(fan_out[l])) for l in range(L)]
        al = [hill_alpha(svW[l]**2) if len(svW[l]) >= 12 else np.nan for l in range(L)]
        rg = r['snap_spike_rough'][j]
        i = max(0, np.searchsorted(r['t'], r['snap_t'][j], side='right')-1)
        shares.append(sh); amps.append([s[0] for s in svd]); alphas.append(al); roughs.append(rg[:, 0])
        print(f"   {taus[j]:8.0f} {r['snap_t'][j]:8.2f} {r['E'][i]:7.4f} {r['Ev'][i]:7.4f} | " + ' '.join(f'{x:5.2f}' for x in sh) + f" | {w[99]:8.2e} | " + ' '.join(f'{s[0]:6.3f}' for s in svd) +
              ' | ' + ' '.join(f'{eff_rank(s):4.1f}' for s in svd) + ' | ' + ' '.join(f'{rg[l,0]:.2f},{rg[l,1]:.2f}' for l in range(min(2, L))) + ' | ' + ' '.join(f'{svW[l][0]/edge[l]:5.2f}' for l in range(L)) + ' | ' + ' '.join(f'{a:5.2f}' for a in al))
    shares = np.array(shares); amps = np.array(amps); alphas = np.array(alphas); roughs = np.array(roughs)
    for l in range(L):
        ax[0].semilogx(taus, shares[:, l], 'o-', label=f'layer {l}'); ax[1].loglog(taus, amps[:, l], 'o-', label=f'layer {l}'); ax[2].semilogx(taus, roughs[:, l], 'o-', label=f'layer {l}'); ax[3].semilogx(taus, alphas[:, l], 'o-', label=f'layer {l}')
    ax[0].set_title(f"{r['tag']}: share of kernel tail (k>100) per layer block"); ax[1].set_title('top singular value of dW_l'); ax[2].set_title('roughness of top spike input feature'); ax[3].set_title('Hill alpha of W_l spectrum (top 25%)')
    for a in ax: a.set_xlabel('kernel time tau'); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{outfig}/layers_{r['tag']}.png", dpi=100); plt.close(fig)
