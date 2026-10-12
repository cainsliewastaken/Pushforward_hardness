"""Residual 'front' k*(t): rank (descending current spectrum) where the cumulative residual energy reaches 50 %.
(1) Is d log k*/d log t constant? (2) Predict E(T) = E(t0) * sqrt(tailY(k*(T))/tailY(k*(t0))) with k*(T) extrapolated as a
power law from (t0/3, t0) and tailY from the target spectrum in the INIT basis (yV2 at t=0). usage: python front_predict.py glob..."""
import numpy as np, glob, sys, itertools
fracs = [0.01, 0.03, 0.1, 0.3]; rows = []; slopes_all = []
for pat in sys.argv[1:]:
    for fn in sorted(glob.glob(pat)):
        r = np.load(fn, allow_pickle=True).item()
        if r.get('opt', 'gd') != 'gd' or r.get('loss', 'sq') != 'sq': continue
        t = np.asarray(r['t']); E = np.asarray(r['E']); e2 = np.asarray(r['eE2'])[:, ::-1]; y2 = np.asarray(r['yV2'])[0, ::-1]; T = t[-1]
        cum = np.cumsum(e2, 1)/np.clip(e2.sum(1, keepdims=True), 1e-300, None); ks = np.array([np.searchsorted(c, 0.5)+1 for c in cum]).astype(float)
        tailY = np.cumsum(y2[::-1])[::-1]   # tailY[k] = sum_{j>=k} y_j^2 (0-based)
        def kstar(tt): j = min(max(int(np.searchsorted(t, tt)), 1), len(t)-1); return ks[j], j
        # slope per decade
        sl = []
        for lo, hi in [(1, 10), (10, 100), (100, 1000), (1000, T)]:
            (k1, _), (k2, _) = kstar(lo), kstar(hi); sl.append(np.log(k2/k1)/np.log(hi/lo))
        slopes_all.append(sl); rec = dict(tag=r['tag'], ET=float(E[-1]), kT=ks[-1], slopes=sl)
        for f in fracs:
            k0, j0 = kstar(f*T); k1, j1 = kstar(f*T/3); b = np.log(k0/k1)/np.log(t[j0]/t[j1]); kT = min(k0*(T/t[j0])**b, len(y2)-1)
            pred = E[j0]*np.sqrt(tailY[int(kT)]/tailY[int(k0)]); rec[f'pred_{f}'] = float(pred); rec[f'kT_{f}'] = float(kT); rec[f'b_{f}'] = float(b)
        rows.append(rec)
S = np.array(slopes_all); print(f'{len(rows)} runs. front slope d log k*/d log t per decade [1-10, 10-100, 100-1000, 1000-T]: median {np.round(np.median(S,0),2)}, 10-90% {np.round(np.percentile(S,10,0),2)} .. {np.round(np.percentile(S,90,0),2)}')
for r in rows[:6]: print(f"   {r['tag']:26s} slopes {np.round(r['slopes'],2)}  k*(T)={r['kT']:.0f}")
ET = np.array([x['ET'] for x in rows])
def pairacc(pred):
    pairs = list(itertools.combinations(range(len(rows)), 2)); gaps = np.array([max(ET[i], ET[j])/min(ET[i], ET[j]) for i, j in pairs])
    ok = np.array([np.sign(ET[i]-ET[j]) == np.sign(pred[i]-pred[j]) for i, j in pairs]); out = []
    for lo, hi in [(1.0, 1.1), (1.1, 1.3), (1.3, 2.0), (2.0, 1e9)]:
        m = (gaps > lo) & (gaps <= hi); out.append(f'{lo:.1f}-{hi if hi < 1e9 else "inf"}:{100*ok[m].mean():.0f}%' if m.sum() else '')
    return ' '.join(out)
for f in fracs:
    p = np.array([x[f'pred_{f}'] for x in rows]); rat = p/ET; kT = np.array([x[f'kT_{f}'] for x in rows]); kTm = np.array([x['kT'] for x in rows])
    print(f't0={100*f:4.0f}%: E(T) pred/meas median {np.median(rat):.2f} 10-90% [{np.percentile(rat,10):.2f} {np.percentile(rat,90):.2f}]; k*(T) pred/meas median {np.median(kT/kTm):.2f}; pairs by gap {pairacc(p)}')
