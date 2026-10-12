"""Fit the lift exponent p: lam_k(T)/lam_k(0) ~ k^p over k in [20, 300], per run; also the init spectral slope a0
(lam_k(0) ~ k^-a0) and final slope aT over the same range. Prints a table sorted by Delta."""
import numpy as np, sys
rows = []
for fn in sys.argv[1:]:
    r = np.load(fn, allow_pickle=True).item(); lam0 = r['lam'][0][::-1]; lamT = r['lam'][-1][::-1]
    k = np.arange(20, 301); x = np.log(k)
    ok = lam0[19:300] > 0
    p = np.polyfit(x[ok], np.log(lamT[19:300][ok]/lam0[19:300][ok]), 1)[0]
    a0 = -np.polyfit(x[ok], np.log(lam0[19:300][ok]), 1)[0]; aT = -np.polyfit(x[ok], np.log(np.maximum(lamT[19:300][ok], 1e-300)), 1)[0]
    t = r['t']; E = r['E']; m = t > t[-1]/10; alpha = -np.polyfit(np.log(t[m]), np.log(E[m]), 1)[0]
    # target coefficient decay in the kernel basis: smoothed (log-binned mean) yV2_k ~ k^-b, over k in [20,300]
    def bfit(y):
        y = y[::-1][19:300]; kk = np.arange(20, 301)
        edges = np.unique(np.geomspace(20, 301, 14).astype(int)); xs = []; ys = []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (kk >= a) & (kk < b)
            if sel.sum(): xs.append(np.log(kk[sel]).mean()); ys.append(np.log(y[sel].mean()))
        return -np.polyfit(xs, ys, 1)[0]
    bY0 = bfit(r['yV2'][0]); bYT = bfit(r['yV2'][-1]); bET = bfit(r['eE2'][-1])
    rows.append((r['Delta'], r['H'], r['depth'], r['n'], r['seed'], r['tag'], p, a0, aT, alpha, E[-1], t[-1], bY0, bYT, bET))
rows.sort()
print(f"{'tag':28s} {'Delta':>5s} {'H':>4s} {'d':>2s} {'n':>4s} | {'p':>5s} {'a0':>5s} {'aT':>5s} | {'bY0':>5s} {'bYT':>5s} {'bET':>5s} | {'alpha':>5s} {'E_T':>7s} {'T':>6s}")
for D, H, d, n, s, tag, p, a0, aT, al, ET, T, bY0, bYT, bET in rows:
    print(f"{tag:28s} {D:5.2f} {H:4d} {d:2d} {n:4d} | {p:5.2f} {a0:5.2f} {aT:5.2f} | {bY0:5.2f} {bYT:5.2f} {bET:5.2f} | {al:5.2f} {ET:7.4f} {T:6.0f}")
