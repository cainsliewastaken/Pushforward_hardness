"""Per-layer share of the training-loss decrease, <E,K_l E>/<E,K E> at each snapshot, and the validation descent
rate <Ev, K_vt E> (sign = whether the current kernel is reducing validation error), from lay_*.npy."""
import numpy as np, sys, os
files = {}
for fn in sys.argv[1:]:
    tag = os.path.basename(fn)[4:-4]
    if tag not in files or os.path.getmtime(fn) > os.path.getmtime(files[tag]): files[tag] = fn
for fn in sorted(files.values()):
    r = np.load(fn, allow_pickle=True).item(); L = r['snap_Kl'].shape[1]; taus = r['snap_t']*r['lam1_0']
    print(f"== {r['tag']}")
    print(f"   {'tau':>7s} {'E_tr':>7s} {'E_val':>7s} | share of <E,K E> per layer | tail share (k>100) of <E,KE> | val descent <Ev,Kvt E>/(|Ev||E|)")
    for j in range(len(taus)):
        E = r['snap_E'][j]; Ev = r['snap_Ev'][j]; K = r['snap_K'][j].astype(np.float64); tot = E @ K @ E
        sh = [float(E @ r['snap_Kl'][j][l].astype(np.float64) @ E)/tot for l in range(L)]
        w, V = np.linalg.eigh(K); order = np.argsort(-w); V = V[:, order]; w = w[order]
        c = V.T @ E; tail = float((w[100:]*c[100:]**2).sum()/tot)
        vd = float(Ev @ r['snap_Kvt'][j].astype(np.float64) @ E)/(np.linalg.norm(Ev)*np.linalg.norm(E)) if len(r['snap_Kvt']) > j else np.nan
        i = max(0, np.searchsorted(r['t'], r['snap_t'][j], side='right')-1)
        print(f"   {taus[j]:7.0f} {r['E'][i]:7.4f} {r['Ev'][i]:7.4f} | " + ' '.join(f'{x:5.2f}' for x in sh) + f" | {tail:5.2f} | {vd:+.3f}")
