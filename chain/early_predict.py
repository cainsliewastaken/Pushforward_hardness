"""Predict E(T) from training up to t0 using (a) the frozen kernel K(t0), (b) rank-wise power-law extrapolation of the spectrum
with exponents measured between t0/3 and t0. Inputs: des_/lay_ runs with t, E, lam (ascending), eE2 (same order).
usage: python early_predict.py OUT.csv glob1 glob2 ..."""
import numpy as np, glob, sys, os, itertools
outcsv = sys.argv[1]; fracs = [0.003, 0.01, 0.03, 0.1, 0.3]; rows = []
for pat in sys.argv[2:]:
    for fn in sorted(glob.glob(pat)):
        r = np.load(fn, allow_pickle=True).item()
        if r.get('opt', 'gd') != 'gd' or r.get('loss', 'sq') != 'sq': continue
        t = np.asarray(r['t']); E = np.asarray(r['E']); lam = np.asarray(r['lam'])[:, ::-1]; e2 = np.asarray(r['eE2'])[:, ::-1]; T = t[-1]
        rec = dict(tag=r['tag'], data=r.get('data', 'l63'), D=float(r['Delta']), ET=float(E[-1]))
        for f in fracs:
            j = int(np.searchsorted(t, f*T)); j = min(max(j, 2), len(t)-2); t0 = t[j]; j1 = int(np.searchsorted(t, t0/3)); j1 = min(max(j1, 1), j-1)
            lam0 = np.clip(lam[j], 1e-300, None); frozen = np.sqrt((e2[j]*np.exp(-2*lam0*(T-t0))).sum())
            beta = np.log(np.clip(lam[j], 1e-300, None)/np.clip(lam[j1], 1e-300, None))/np.log(t0/t[j1]); beta = np.clip(beta, 0.0, 3.0)
            integ = lam0*t0*((T/t0)**(beta+1)-1)/(beta+1); extrap = np.sqrt((e2[j]*np.exp(-2*integ)).sum())
            rec[f'frozen_{f}'] = float(frozen); rec[f'extrap_{f}'] = float(extrap); rec[f'E_{f}'] = float(E[j]); rec[f't0_{f}'] = float(t0)
        rows.append(rec)
import csv
with open(outcsv, 'w') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(x) for x in rows]
ET = np.array([x['ET'] for x in rows]); print(f'{len(rows)} GD/squared-loss runs')
def pairacc(pred):
    pairs = list(itertools.combinations(range(len(rows)), 2)); gaps = np.array([max(ET[i], ET[j])/min(ET[i], ET[j]) for i, j in pairs])
    ok = np.array([np.sign(ET[i]-ET[j]) == np.sign(pred[i]-pred[j]) for i, j in pairs]); out = []
    for lo, hi in [(1.0, 1.1), (1.1, 1.3), (1.3, 2.0), (2.0, 1e9)]:
        m = (gaps > lo) & (gaps <= hi); out.append(f'{lo:.1f}-{hi if hi < 1e9 else "inf"}:{100*ok[m].mean():.0f}%({m.sum()})' if m.sum() else '')
    return ' '.join(out)
for f in fracs:
    for name in ('frozen', 'extrap'):
        p = np.array([x[f'{name}_{f}'] for x in rows]); rat = p/ET
        print(f't0={100*f:5.1f}% of T  {name:7s}: median pred/meas {np.median(rat):.2f}  10-90% [{np.percentile(rat,10):.2f} {np.percentile(rat,90):.2f}]  pairs by true gap {pairacc(p)}')
    e0 = np.array([x[f'E_{f}'] for x in rows]); print(f'                (E(t0)/E(T) median {np.median(e0/ET):.1f}; naive "E(t0)" ordering: {pairacc(e0)})')
