"""Which facts survive across optimizer / loss / data / scale variants. For each des_/lay_ file:
 F1  CKR late ratio: sqrt(sum_k e0V2_k(t) e^{-2 lam_k(t) t}) / ||E|| (median over last quarter)
 F3  lift exponent p (lam_k(T)/lam_k(0) ~ k^p over ranks 20-300) and lift at rank 100
 F2  rank-roughness (lay_ files with lap_U, snapshot K at T vs K0): mean roughness of ranks 101-200 and 201-400
 F5  spikes (lay_ files): eff. rank of dW_l at T, hidden amplitude, top sv / MP edge
 E_tr(T), E_val(T), lam1 growth.  Output: one line per run, grouped by variant."""
import numpy as np, sys, glob, os
def eff_rank(sv):
    p = sv**2/(sv**2).sum(); p = p[p > 0]; return float(np.exp(-(p*np.log(p)).sum()))
rows = []
for pat in sys.argv[1:]:
    for fn in sorted(glob.glob(pat)):
        r = np.load(fn, allow_pickle=True).item(); t = r['t']; E = r['E']; lam = r['lam'][:, ::-1]; e0 = r['e0V2'][:, ::-1]
        late = slice(len(t)*3//4, None)
        S = np.array([(e0[i]*np.exp(-2*lam[i]*t[i])).sum() for i in range(len(t))]); ckr = float(np.median(np.sqrt(S[late])/E[late]))
        lam0 = lam[0]; lamT = lam[-1]; k = np.arange(20, 301); ok = lam0[19:300] > 0
        p = float(np.polyfit(np.log(k[ok]), np.log(lamT[19:300][ok]/lam0[19:300][ok]), 1)[0]); l100 = float(lamT[99]/lam0[99])
        rec = dict(tag=r['tag'], opt=r.get('opt', 'gd'), loss=r.get('loss', 'sq'), data=r.get('data', 'l63'), n=r['n'], H=r['H'], depth=r['depth'], D=r['Delta'],
                   Etr=float(E[-1]), Ev=float(r['Ev'][-1]), ckr=ckr, p=p, lift100=l100, lam1_ratio=float(lamT[0]/lam0[0]))
        if 'lap_U' in r and 'snap_K' in r and len(r['snap_K']) > 0:
            n = r['n']; U = r['lap_U']; mu = r['lap_mu']
            def rough(K):
                w, V = np.linalg.eigh(K.astype(np.float64)); V = V[:, ::-1]
                return np.array([((U.T @ V[:, j].reshape(n, 3))**2).sum(1) @ mu for j in range(400)])
            r0 = rough(r['snap_K'][0]); rT = rough(r['snap_K'][-1])
            rec.update(rough0_100_200=float(r0[100:200].mean()), roughT_100_200=float(rT[100:200].mean()), rough0_200_400=float(r0[200:400].mean()), roughT_200_400=float(rT[200:400].mean()))
        if 'snap_svdW' in r and len(r['snap_svdW']) > 0:
            sv = r['snap_svdW'][-1]; L = len(sv); hid = list(range(1, L-1)) if L > 2 else [1]
            rec.update(erank_hidden=float(np.mean([eff_rank(sv[l]) for l in hid])), amp_hidden=float(np.max([sv[l][0] for l in hid])), erank_all=[round(eff_rank(s), 1) for s in sv])
        rows.append(rec)
import pandas as pd
df = pd.DataFrame(rows)
cols = ['tag', 'Etr', 'Ev', 'ckr', 'p', 'lift100', 'lam1_ratio'] + [c for c in ['roughT_100_200', 'rough0_100_200', 'roughT_200_400', 'rough0_200_400', 'erank_hidden', 'amp_hidden'] if c in df]
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
for (o, l, d), g in df.groupby(['opt', 'loss', 'data']):
    print(f"\n=== opt={o} loss={l} data={d}: {len(g)} runs"); print(g[cols].sort_values('tag').to_string(index=False, float_format=lambda x: f'{x:.3f}'))
df.to_csv('survival_table.csv', index=False)
