"""RMT test of the tail lift: is spec(K(T)) reproduced by spec(K(0) + Q dK Q^T) with Q a random rotation
(free-convolution proxy), or by the K(0)-diagonal part of dK only? Also dK's trace, effective rank, roughness."""
import numpy as np, sys
rng = np.random.default_rng(0)
ks = [0, 9, 49, 99, 199, 399]
def eff_rank(ev):
    ev = np.abs(ev); p = ev/ev.sum(); p = p[p > 0]; return float(np.exp(-(p*np.log(p)).sum()))
print(f"{'run':22s} {'tr dK/tr K0':>11s} {'rank dK':>7s} {'rough dK top':>12s} | lam_k(T) actual / random-rotation / diag-only, k=1,10,50,100,200,400")
for fn in sys.argv[1:]:
    r = np.load(fn, allow_pickle=True).item()
    K0 = r['snap_K'][0].astype(np.float64); KT = r['snap_K'][-1].astype(np.float64); dK = KT-K0
    n = r['n']; ny = 3; mu = r['lap_mu']; U = r['lap_U']
    w0, V0 = np.linalg.eigh(K0); wT = np.linalg.eigh(KT)[0][::-1]
    wd, Vd = np.linalg.eigh(dK); order = np.argsort(-np.abs(wd)); wd = wd[order]; Vd = Vd[:, order]
    rough = [float(((U.T @ Vd[:, j].reshape(n, ny))**2).sum(1) @ mu) for j in range(3)]
    rot = []
    for _ in range(3):
        Q, _ = np.linalg.qr(rng.normal(size=(K0.shape[0], K0.shape[0])))
        rot.append(np.linalg.eigh(K0 + Q @ dK @ Q.T)[0][::-1])
    rot = np.median(rot, 0)
    diag = np.linalg.eigh(K0 + V0 @ np.diag(np.diag(V0.T @ dK @ V0)) @ V0.T)[0][::-1]
    print(f"{r['tag']:22s} {np.trace(dK)/np.trace(K0):11.2f} {eff_rank(wd):7.1f} {np.mean(rough):12.2f} | " +
          ' '.join(f"{wT[k]:.1e}/{rot[k]:.1e}/{diag[k]:.1e}" for k in ks))
