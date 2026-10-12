"""Tail mass M_k(t) = sum_{j>=k} lam_j(t) at snapshot times vs path length ell(t) = int |g_E| and
curvature-weighted ellH = int H2 |g_E|: exponents and prefactors. Tests lift ~ (H2 ell)^2 (quadratic feature term)."""
import numpy as np, sys
print(f"{'run':22s} | exps M_k-M_k(0) vs ell (k=50,100,200) | vs ellH | lift100/ell^2 | lift100/ellH^2 | lift100/(H2bar ell)^2 | ell(T) ellH(T) H2bar")
for fn in sys.argv[1:]:
    r = np.load(fn, allow_pickle=True).item(); t = r['t']; lam = r['lam'][:, ::-1]; g = r['gnorm']; H2 = r['H2_g']
    H2f = np.where(np.isfinite(H2), H2, np.nanmedian(H2))
    ell = np.concatenate([[0], np.cumsum(0.5*(g[1:]+g[:-1])*np.diff(t))])
    ellH = np.concatenate([[0], np.cumsum(0.5*(g[1:]*H2f[1:]+g[:-1]*H2f[:-1])*np.diff(t))])
    out = []
    for q, name in [(ell, 'ell'), (ellH, 'ellH')]:
        ex = []
        for k in (49, 99, 199):
            M = lam[:, k:].sum(1); d = M-M[0]; mm = (d > 0.5*M[0]) & (q > 0)
            ex.append(np.polyfit(np.log(q[mm]), np.log(d[mm]), 1)[0] if mm.sum() > 5 else np.nan)
        out.append(ex)
    M100 = lam[:, 99:].sum(1); lift = M100[-1]-M100[0]; H2bar = np.nanmax(H2)
    print(f"{r['tag']:22s} | " + ' '.join(f'{e:5.2f}' for e in out[0]) + ' | ' + ' '.join(f'{e:5.2f}' for e in out[1]) +
          f" | {lift/ell[-1]**2:9.2e} | {lift/ellH[-1]**2:9.2e} | {lift/(H2bar*ell[-1])**2:9.2e} | {ell[-1]:7.2f} {ellH[-1]:8.2f} {H2bar:5.2f}")
