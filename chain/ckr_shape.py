"""Reduced model test: Ktilde(t) = K0 + q(t) Qhat, Qhat = Q(T)/trQ(T), q(t) = trQ(t). At each snapshot time compare
||E(t)|| with CKR using K(t) (exact kernel) and with Ktilde(t) (shape-frozen, scalar amplitude), and with
K0 + Q(T) (fully frozen late kernel), source = E0."""
import numpy as np, sys
def ckr(K, E0, t, n=200):
    w, V = np.linalg.eigh(K); c2 = (V.T @ E0)**2; return np.sqrt((c2*np.exp(-2*np.maximum(w, 0)*t)).sum()/n)
print(f"{'run':22s}   t      ||E||   CKR(K(t))/E  CKR(K0+qQhat)/E  CKR(K0+Q(T))/E  q(t)/q(T)")
for fn in sys.argv[1:]:
    r = np.load(fn, allow_pickle=True).item()
    if 'snap_Q' not in r or len(r['snap_Q']) == 0: continue
    E0 = r['Evec'][0]; t = r['t']; st = r['snap_t']; K0 = r['snap_K'][0].astype(np.float64)
    QT = r['snap_Q'][-1].astype(np.float64); qT = np.trace(QT); Qhat = QT/qT
    for i in range(len(st)//3, len(st), max(1, len(st)//8)):
        j = min(np.searchsorted(t, st[i]), len(t)-1); E = r['E'][j]
        K = r['snap_K'][i].astype(np.float64); q = np.trace(r['snap_Q'][i].astype(np.float64))
        print(f"{r['tag']:22s} {st[i]:7.1f} {E:8.4f} {ckr(K, E0, st[i], r["n"])/E:10.2f} {ckr(K0+q*Qhat, E0, st[i], r["n"])/E:14.2f} {ckr(K0+QT, E0, st[i], r["n"])/E:14.2f} {q/qT:10.3f}")
