"""Tail-lift exponents: lam_k(t) ~ t^gamma_k on the last decade; ||E|| ~ t^-alpha; and the theory link:
lam_k(t) - lam_k(0) vs C(t) = int_0^t H2(s) ||E(s)|| ds (the integrated drive bound of Prop. 15)  -> exponent p_k."""
import numpy as np, sys
ks = [9, 49, 99, 199, 399]
print(f"{'run':22s} {'alpha':>5s} | gamma_k (k=10,50,100,200,400) | p_k: dlam_k ~ C^p | C(T) | H2 late | lift_k(T)/C(T)^2 (k=100,200)")
for fn in sys.argv[1:]:
    r = np.load(fn, allow_pickle=True).item(); t = r['t']; lam = r['lam'][:, ::-1]; E = r['E']; H2 = r['H2_g']
    m = t > t[-1]/10
    alpha = -np.polyfit(np.log(t[m]), np.log(E[m]), 1)[0]
    H2f = np.where(np.isfinite(H2), H2, np.nanmedian(H2))
    C = np.concatenate([[0], np.cumsum(0.5*(H2f[1:]*E[1:]+H2f[:-1]*E[:-1])*np.diff(t))])
    gam = []; pk = []; ratio = []
    for k in ks:
        gam.append(-np.polyfit(np.log(t[m]), -np.log(np.maximum(lam[m, k], 1e-300)), 1)[0])
        d = lam[:, k]-lam[0, k]; mm = (d > 0.5*lam[0, k]) & (C > 0)
        pk.append(np.polyfit(np.log(C[mm]), np.log(d[mm]), 1)[0] if mm.sum() > 5 else np.nan)
    print(f"{r['tag']:22s} {alpha:5.2f} | " + ' '.join(f'{g:5.2f}' for g in gam) + ' | ' + ' '.join(f'{p:5.2f}' for p in pk) +
          f" | {C[-1]:8.1f} | {np.nanmedian(H2[m]):6.2f} | {(lam[-1,99]-lam[0,99])/C[-1]**2:8.1e} {(lam[-1,199]-lam[0,199])/C[-1]**2:8.1e}")
