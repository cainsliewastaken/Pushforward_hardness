"""Analyse the roughness-theory measurements along SegFlow paths (runs/rough/*.npy):
  c_m(t) efficiency, realized drive fraction phi = (d sigma^2/dt) / (2 sigma b ||E||) against the a-priori bound,
  and the two-constant reduced model r' = -c sigma^2 r, (sigma^2)' = 2 phi sigma b r with per-run constants."""
import glob, numpy as np, sys
M = int(sys.argv[1]) if len(sys.argv) > 1 else 10
print(f'high-pass above Laplacian mode m={M}; ||E|| in the Euclidean norm on the N=3n sampled outputs')
print('design                     | c: start   mid    end  | phi (drive/bound): median  [10%,90%]   | sigma2: start -> end | lam1: start -> end | H2 range')
for f in sorted(glob.glob('../runs/rough/*.npy')):
    d = np.load(f, allow_pickle=True).item(); R = d['recs']; n = 200
    t = np.array([r['t'] for r in R]); E = np.array([r['E'] for r in R])*np.sqrt(n); Er = np.array([r[f'Er_{M}'] for r in R])*np.sqrt(n)
    c = np.array([r[f'c_{M}'] for r in R]); s2 = np.array([r[f'sig2_{M}'] for r in R]); b = np.array([r['b'] for r in R]); l1 = np.array([r['lam1'] for r in R]); H2 = np.array([r['H2'] for r in R])
    ds2 = np.gradient(s2, t); phi = ds2/(2*np.sqrt(s2)*b*E)
    k = len(t); i1, i2 = k//3, 2*k//3
    print('%-26s | %.4f %.4f %.4f | %.2e [%.1e,%.1e] | %.2f -> %.2f | %.1f -> %.1f | %.2f-%.2f' % (d['tag'], c[1], c[i2], c[-1], np.median(phi[1:]), np.quantile(phi[1:], .1), np.quantile(phi[1:], .9), s2[0], s2[-1], l1[0], l1[-1], H2.min(), H2.max()))
    # scaling of c with the residual: fit log c = p log E + q over the path (excluding the first point)
    p = np.polyfit(np.log(E[1:]), np.log(c[1:]), 1)
    # reduced two-constant model with this run's own median c and phi, integrated from the first checkpoint
    cm, pm = np.median(c[1:]), np.median(phi[1:]); bm = np.median(b)
    r, s = Er[1], s2[1]; pred = [r]
    for j in range(1, k-1):
        dt = t[j+1]-t[j]; sub = max(1, int(dt*cm*s/0.05)+1); h = dt/sub
        for _ in range(sub):
            r = r*np.exp(-cm*s*h); s = s + 2*pm*np.sqrt(s)*bm*r*h
        pred.append(r)
    pred = np.array(pred); ratio = pred/Er[1:]
    print('      c ~ E^%.2f ;  two-constant reduced model (c, phi fixed at run medians): pred/meas E_r at mid %.2f, end %.2f' % (p[0], ratio[len(ratio)//2], ratio[-1]))

print()
print('Rate law on the path: q(t) = <E,KE>/||E||^2 (effective eigenvalue the residual sits at; -d ln||E||^2/dt = 2q).  Fit log q = alpha log E + log q1 for E < 0.8 E_0')
print('design                     | alpha  q1 (q at E=1 rms) | q(E) prefactor vs init: lam1(0)  sig2_10(0)  H2(0) | E-fold time check: predicted t to reach E_end from E_1 (law) / measured')
for f in sorted(glob.glob('../runs/rough/*.npy')):
    d = np.load(f, allow_pickle=True).item(); R = d['recs']; n = 200
    t = np.array([r['t'] for r in R]); E = np.array([r['E'] for r in R]); q = np.array([r['rate_id'] for r in R])/2
    sel = (E < 0.8*E[0]) & (np.arange(len(E)) > 0)
    a, lq1 = np.polyfit(np.log(E[sel]), np.log(q[sel]), 1); q1 = np.exp(lq1)
    # integrate dE/dt = -q(E) E with the fitted law from the first selected checkpoint to the final E, compare times
    j0 = np.argmax(sel); Ei, Ef = E[j0], E[-1]
    # dt = -dE/(q1 E^(a+1)) -> t = (E^-a) difference / (a q1)  (a != 0)
    t_law = (Ef**(-a) - Ei**(-a))/(a*q1) if abs(a) > 1e-6 else np.log(Ei/Ef)/q1
    print('%-26s | %.2f  %.3e | %.1f  %.2f  %.2f | %.3g / %.3g' % (d['tag'], a, q1, R[0]['lam1'], R[0]['sig2_10'], R[0]['H2'], t_law, t[-1]-t[j0]))

print()
print('Gain at the residual position: measured q(E) on the path vs the frozen-kernel q_f(E) computed from the init spectrum alone')
print('(q_f(E): along E(t) = sum_k c_k e^{-lam_k t} v_k with lam_k, c_k from theta_0; ratio > 1 is the eigenvalue gain where the residual sits)')
levels = [0.5, 0.3, 0.2, 0.1, 0.05, 0.03]
print('design                     | ' + '  '.join('E/E0=%.2f' % l for l in levels) + '   (q/q_f at matched E; "-" = level not reached)')
for f in sorted(glob.glob('../runs/rough/*.npy')):
    d = np.load(f, allow_pickle=True).item(); R = d['recs']; n = 200; tag = d['tag']
    des = np.load(f'../runs/heldout/des_{tag}.npy', allow_pickle=True).item()
    lam = np.clip(np.asarray(des['lam0'], float), 0, None); ck2 = np.asarray(des['e0V2_0'], float)*n     # c_k^2 (Euclidean)
    tt = np.geomspace(1e-4/lam.max(), 1e9/lam.max(), 4000)
    w = ck2[None, :]*np.exp(-2*lam[None, :]*tt[:, None]); Ef = np.sqrt(w.sum(1)/n); qf = (w*lam[None, :]).sum(1)/w.sum(1)
    E = np.array([r['E'] for r in R]); q = np.array([r['rate_id'] for r in R])/2; E0 = E[0]
    out = []
    for l in levels:
        Et = l*E0
        if Et < E[1:].min() or Et > E[1:].max(): out.append('    -    '); continue
        qm = np.interp(-Et, -E[1:], q[1:])          # E decreasing along the path
        qfz = np.interp(-Et, -Ef, qf) if Et > Ef.min() else np.nan
        out.append('%9.1f' % (qm/qfz) if np.isfinite(qfz) else '  (frozen never reaches)')
    print('%-26s | %s' % (tag, '  '.join(out)))
