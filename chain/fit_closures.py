"""Fit the two closure curves from saved spike-logging runs (lay_*.npy), rank-16 amplitudes.
 Curve 1: rough(delta_0) = f(P), P = prod_{hidden l} ||sv_1..16(dW_l)||  (only runs with factor roughness)
 Curve 2: per-rank lift  lam_k(t)/lam_k(0) = 1 + exp(c0 + c1*log P + c2*log(k/100) + c3*(depth-2))  [log-linear fit]
 plus the per-layer descent share model: share_l = <E,K_l E>/<E,K E> vs tau (mean per depth, early vs late).
 Saves closures.npz."""
import numpy as np, sys, os, glob
files = {}
for fn in sys.argv[1:]:
    tag = os.path.basename(fn)[4:-4]
    if tag not in files or os.path.getmtime(fn) > os.path.getmtime(files[tag]): files[tag] = fn
X2 = []; y2 = []; X1 = []; y1 = []; shares = {}
ks = np.array([10, 20, 50, 100, 150, 200, 300, 400, 500])
for fn in files.values():
    r = np.load(fn, allow_pickle=True).item()
    if 'snap_svdW' not in r or len(r['snap_svdW']) == 0 or r['outscale'] != 1.0: continue   # fit on standard parametrisation first
    L = len(r['snap_svdW'][0]); hid = list(range(1, L-1)) if L > 2 else [1]
    w0 = np.sort(np.linalg.eigvalsh(r['snap_K'][0].astype(np.float64)))[::-1]
    e0 = r['e0V2_0'][::-1]; frough = float(e0[100:].sum()/e0.sum())
    for j in range(len(r['snap_t'])):
        P = np.prod([np.linalg.norm(r['snap_svdW'][j][l][:16]) for l in hid])
        if P < 0.3: continue
        w = np.sort(np.linalg.eigvalsh(r['snap_K'][j].astype(np.float64)))[::-1]
        for k in ks:
            lift = w[k-1]/w0[k-1]
            if lift > 1.2 and w0[k-1] > 1e-12*w0[0]:
                X2.append([1., np.log(P), np.log(k/100.), r['depth']-2, np.log(frough), np.log(k/100.)*np.log(frough), np.log(k/100.)*np.log(P)]); y2.append(np.log(lift-1))
        if 'snap_factor_rough' in r and len(r['snap_factor_rough']) > j:
            X1.append([1., np.log(P)]); y1.append(np.log(r['snap_factor_rough'][j][0][2]))
        if 'snap_Kl' in r and len(r['snap_Kl']) > j:
            E = r['snap_E'][j]; K = r['snap_K'][j].astype(np.float64); tot = E @ K @ E
            sh = [float(E @ r['snap_Kl'][j][l].astype(np.float64) @ E)/tot for l in range(L)]
            shares.setdefault(r['depth'], []).append(sh)
X2 = np.array(X2); y2 = np.array(y2); c2, *_ = np.linalg.lstsq(X2, y2, rcond=None); res2 = y2 - X2 @ c2
X1 = np.array(X1); y1 = np.array(y1); c1, *_ = np.linalg.lstsq(X1, y1, rcond=None); res1 = y1 - X1 @ c1
print(f"curve 2 (lift-1 ~ P^a (k/100)^b e^(c depth) e^(d log frough)): coeffs {c2.round(3)}, n={len(y2)}, log-residual std {res2.std():.2f} (factor {np.exp(res2.std()):.2f})")
print(f"curve 1 (rough(delta0) ~ P^e): coeffs {c1.round(3)}, n={len(y1)}, log-residual std {res1.std():.2f}")
for d, s in shares.items(): s = np.array(s); print(f"depth {d}: mean per-layer descent share {s.mean(0).round(2)} (n={len(s)})")
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'closures.npz'), c2=c2, c1=c1, shares={d: np.array(s).mean(0) for d, s in shares.items()})
print('saved closures.npz')
