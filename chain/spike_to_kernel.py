"""Link weight-side spikes to the kernel tail. For each run and snapshot: per-layer block tail lam_100(K_l),
whole-kernel lam_100(K), hidden-layer spike amplitude a_l = top sv of dW_l, Hill alpha_l, and per-layer tail mass.
Fits: log lam100(K_l) vs log a_l (slope, per layer, over snapshots with a_l > 0.3); log [lam100(K)-lam100(K0)] vs
log a_hidden across ALL runs and snapshots (slope, scatter); same vs Hill alpha."""
import numpy as np, sys, glob
from scipy.stats import spearmanr
rows = []
for fn in sys.argv[1:]:
    r = np.load(fn, allow_pickle=True).item(); L = len(r['snap_svW'][0]); taus = r['snap_t']*r['lam1_0']
    K0 = r['snap_K'][0].astype(np.float64); w0 = np.sort(np.linalg.eigvalsh(K0))[::-1]
    Kl0 = [np.sort(np.linalg.eigvalsh(r['snap_Kl'][0][l].astype(np.float64)))[::-1] for l in range(L)]
    print(f"== {r['tag']}")
    print(f"   {'tau':>7s} | lam100(K)/lam100(K0) | per-layer lam100(K_l)/lam100(K_l,0) | spike amp a_l | Hill alpha (hidden)")
    for j in range(len(taus)):
        w = np.sort(np.linalg.eigvalsh(r['snap_K'][j].astype(np.float64)))[::-1]
        wl = [np.sort(np.linalg.eigvalsh(r['snap_Kl'][j][l].astype(np.float64)))[::-1] for l in range(L)]
        amp = r['snap_spike_amp'][j]; svW = r['snap_svW'][j]
        def hill(ev, frac=0.25):
            ev = np.sort(ev)[::-1]; k = max(3, int(len(ev)*frac)); top = ev[:k]; return 1 + k/np.sum(np.log(top/top[-1]))
        al = [hill(svW[l]**2) if len(svW[l]) >= 12 else np.nan for l in range(L)]
        print(f"   {taus[j]:7.0f} | {w[99]/w0[99]:8.1f} | " + ' '.join(f'{wl[l][99]/Kl0[l][99]:8.1f}' for l in range(L)) + ' | ' + ' '.join(f'{a:6.2f}' for a in amp) + ' | ' + ' '.join(f'{a:5.2f}' for a in al if np.isfinite(a)))
        for l in range(L):
            rows.append(dict(tag=r['tag'], l=l, hidden=(0 < l < L-1), tau=taus[j], lift=w[99]/w0[99], lift_l=wl[l][99]/Kl0[l][99], amp=amp[l], alpha=al[l], D=r['Delta'], s=r['outscale']))
import pandas as pd
df = pd.DataFrame(rows)
print("\nPer-layer fits of log(lift_l - 1) vs log amp_l over snapshots (slope), hidden layers, amp > 0.3:")
for (tag, l), g in df[(df.hidden) & (df.amp > 0.3) & (df.lift_l > 1.5)].groupby(['tag', 'l']):
    if len(g) >= 3: print(f"   {tag:28s} layer {l}: slope {np.polyfit(np.log(g.amp), np.log(g.lift_l-1), 1)[0]:5.2f}  (n={len(g)})")
h = df[(df.hidden) & (df.amp > 0.3) & (df.lift > 1.5)]
print(f"\nAcross all runs/snapshots (hidden layers, {len(h)} points): slope of log(lift-1) vs log amp = {np.polyfit(np.log(h.amp), np.log(h.lift-1), 1)[0]:.2f}, Spearman {spearmanr(h.amp, h.lift)[0]:.2f}; vs Hill alpha: Spearman {spearmanr(h.alpha, h.lift)[0]:.2f}")
for D in sorted(h.D.unique()):
    hh = h[h.D == D]; print(f"   Delta={D}: slope {np.polyfit(np.log(hh.amp), np.log(hh.lift-1), 1)[0]:.2f}, Spearman(amp) {spearmanr(hh.amp, hh.lift)[0]:.2f}, Spearman(alpha) {spearmanr(hh.alpha, hh.lift)[0]:.2f}, n={len(hh)}")
