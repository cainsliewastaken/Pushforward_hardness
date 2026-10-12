"""Mechanism test on runs with L63_QUAD=1: dK = X + Q, X = (J0 dJ^T + dJ J0^T)/n (cross), Q = dJ dJ^T/n (quadratic).
In the tail eigenbasis of K0 (ranks >= k0), compare the spectrum of K(T) with: K0 + Q only, K0 + X only, and
Q alone; also report ||P_tail X P_tail|| vs ||P_tail Q P_tail|| and the effective rank of Q."""
import numpy as np, sys
ks = [9, 49, 99, 199, 399]
def eff_rank(ev):
    ev = np.abs(ev); p = ev/ev.sum(); p = p[p > 0]; return float(np.exp(-(p*np.log(p)).sum()))
print(f"{'run':22s} snap_t | lam_k: K(T) / K0+Q / K0+X / Q  (k=10,50,100,200,400) | tail(k>=100) norms: X, Q | rank Q | trQ/trX")
for fn in sys.argv[1:]:
    r = np.load(fn, allow_pickle=True).item()
    if 'snap_Q' not in r or len(r['snap_Q']) == 0: continue
    K0 = r['snap_K'][0].astype(np.float64); w0, V0 = np.linalg.eigh(K0); V0 = V0[:, ::-1]
    for i in [len(r['snap_t'])//2, len(r['snap_t'])-1]:
        KT = r['snap_K'][i].astype(np.float64); Q = r['snap_Q'][i].astype(np.float64); Xc = r['snap_X'][i].astype(np.float64)
        wT = np.linalg.eigvalsh(KT)[::-1]; wQ = np.linalg.eigvalsh(K0+Q)[::-1]; wX = np.linalg.eigvalsh(K0+Xc)[::-1]; wq = np.linalg.eigvalsh(Q)[::-1]
        Pt = V0[:, 99:]
        nX = np.linalg.norm(Pt.T @ Xc @ Pt, 2); nQ = np.linalg.norm(Pt.T @ Q @ Pt, 2)
        print(f"{r['tag']:22s} {r['snap_t'][i]:7.1f} | " + ' '.join(f"{wT[k]:.1e}/{wQ[k]:.1e}/{wX[k]:.1e}/{wq[k]:.1e}" for k in ks) +
              f" | {nX:.2e} {nQ:.2e} | {eff_rank(wq):5.1f} | {np.trace(Q)/max(abs(np.trace(Xc)),1e-300):6.2f}")
