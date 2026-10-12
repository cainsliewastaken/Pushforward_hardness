"""Design ranking analysis on des_*.npy. Budgets in kernel time tau_b = lam1(0) t. Ground truth: E_tr, E_val at tau_b.
Init scores: lazy train S0, lazy VAL prediction via cross kernel (NTK regression), lam_100(0)/lam1(0), chi, H2, q0 tail.
Few-step: S_f (train) from the kernel snapshot at tau_f. Spearman vs ground truth over designs per (Delta, budget)."""
import numpy as np, sys, os
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
outfig = sys.argv[1]; os.makedirs(outfig, exist_ok=True)
R = []
for fn in sys.argv[2:]:
    r = np.load(fn, allow_pickle=True).item()
    if r['H'] == 32 and r['depth'] == 1: continue
    R.append(r)
print(len(R), 'runs')
TAUB = [3e3, 1e4, 3e4, 1e5]
def at_tau(r, tau):
    t = tau/r['lam1_0']; i = max(0, np.searchsorted(r['t'], t, side='right')-1); return i
def lazy_val(r, t):
    K0 = r['K0'].astype(np.float64); w, V = np.linalg.eigh(K0); w = np.maximum(w, 0)
    f = np.where(w > 1e-300, (1-np.exp(-w*t))/np.where(w > 1e-300, w, 1), t)
    Ev = r['Ev0'] - r['K_vt'].astype(np.float64) @ (V @ (f*(V.T @ r['E0'])))
    return np.sqrt(Ev@Ev/r['nval'])
rows = []
for r in R:
    lam0 = r['lam0'][::-1]; e0 = r['e0V2_0'][::-1]; n = r['n']
    d = dict(tag=r['tag'], D=r['Delta'], H=r['H'], depth=r['depth'], s=r['outscale'], n=n, lam1_0=r['lam1_0'],
             tail0=-lam0[min(99, len(lam0)-1)]/r['lam1_0'], chi=-r['chi'], H2=-r['H2_0'], q0tail=-r['q0'][::-1][min(99, len(lam0)-1):].sum()/r['q0'].sum(),
             rate0tail=-r['rate0'][::-1][min(99, len(lam0)-1):].sum()/r['lam1_0']**2)
    for tb in TAUB:
        i = at_tau(r, tb); t = r['t'][i]
        d[f'Etr@{tb:g}'] = r['E'][i]; d[f'Ev@{tb:g}'] = r['Ev'][i]
        d[f'S0@{tb:g}'] = np.sqrt((e0*np.exp(-2*lam0*t)).sum()); d[f'LV@{tb:g}'] = lazy_val(r, t)
        for j, ts in enumerate(r['snap_t']):
            NOM = np.array([1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 60000, 100000]); tauf = NOM[np.argmin(np.abs(np.log(NOM) - np.log(max(ts*r['lam1_0'], 1e-6))))]; K = r['snap_K'][j].astype(np.float64); w, V = np.linalg.eigh(K); w = np.maximum(w, 0)
            ii = max(0, np.searchsorted(r['t'], ts, side='right')-1); Ei = None
            c2 = r['eE2'][ii][::-1]  # energies in the basis at the snapshot's log (same time)
            d[f'Sf{tauf:.0f}@{tb:g}'] = np.sqrt((c2*np.exp(-2*np.sort(w)[::-1]*max(t-ts, 0))).sum()) if t > ts else np.nan
            if 'snap_Kvt' in r and len(r['snap_Kvt']) > j and t > ts:
                f = np.where(w > 1e-300, (1-np.exp(-w*(t-ts)))/np.where(w > 1e-300, w, 1), t-ts)
                Evp = r['snap_Ev'][j] - r['snap_Kvt'][j].astype(np.float64) @ (V @ (f*(V.T @ r['snap_E'][j])))
                d[f'FV{tauf:.0f}@{tb:g}'] = np.sqrt(Evp@Evp/r['nval'])
    rows.append(d)
import pandas as pd
df = pd.DataFrame(rows); df.to_csv(f'{outfig}/design_table.csv', index=False)
scores = ['tail0', 'chi', 'H2', 'q0tail', 'rate0tail'] + [f'S0@', f'LV@'] + [f'Sf{int(tf)}@' for tf in [1, 3, 10, 30, 100, 300]] + [f'FV{int(tf)}@' for tf in [10, 30, 100, 300]]
for D in sorted(df.D.unique()):
    for subset, name in [(df[(df.D == D) & (df.n == 200)], 'all scales'), (df[(df.D == D) & (df.n == 200) & (df.s == 1)], 's=1 only')]:
        if len(subset) < 4: continue
        print(f"\n== Delta={D}, {name}: {len(subset)} designs")
        print(f"   {'budget':>7s} {'target':>6s} | " + ' '.join(f'{sc.replace("@", ""):>9s}' for sc in scores))
        for tb in TAUB:
            for tgt in ['Etr', 'Ev']:
                y = subset[f'{tgt}@{tb:g}'].values
                vals = []
                for sc in scores:
                    col = sc if '@' not in sc else f'{sc}{tb:g}'
                    x = subset[col].values if col in subset else np.full(len(y), np.nan)
                    m = np.isfinite(x) & np.isfinite(y)
                    vals.append(spearmanr(x[m], y[m])[0] if m.sum() >= 4 else np.nan)
                print(f"   {tb:7.0e} {tgt:>6s} | " + ' '.join(f'{v:9.2f}' for v in vals) + f"   [range {y.min():.3f}-{y.max():.3f}]")
# n effect
sub = df[(df.H == 96) & (df.s == 1)]
print("\nn effect (H=96, s=1): tag, E_tr, E_val at tau=1e5:")
for _, q in sub.sort_values(['D', 'depth', 'n']).iterrows(): print(f"   {q.tag:32s} Etr={q['Etr@1e+05']:.4f} Ev={q['Ev@1e+05']:.4f} LV={q['LV@1e+05']:.4f} S0={q['S0@1e+05']:.4f}")
# figure: Ev at largest budget vs few-step score and vs lazy val prediction
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for D, mk in zip(sorted(df.D.unique()), ['o', 's', '^']):
    sub = df[(df.D == D) & (df.n == 200)]
    for s, c in zip([0.25, 1, 4], ['tab:red', 'tab:orange', 'tab:blue']):
        ss = sub[sub.s == s]
        ax[0].loglog(ss['LV@1e+05'], ss['Ev@1e+05'], mk, color=c, label=f'D={D} s={s}', alpha=0.7)
        ax[1].loglog(ss['Sf100@1e+05'], ss['Ev@1e+05'], mk, color=c, alpha=0.7)
        ax[2].loglog(ss['Sf100@1e+05'], ss['Etr@1e+05'], mk, color=c, alpha=0.7)
for a, xl, yl in zip(ax, ['lazy validation prediction (init only)', 'few-step score S_f(tau_f=100)', 'few-step score S_f(tau_f=100)'], ['E_val at tau=1e5', 'E_val at tau=1e5', 'E_train at tau=1e5']):
    lim = [1e-3, 3]; a.plot(lim, lim, 'k:'); a.set_xlabel(xl); a.set_ylabel(yl)
ax[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(f'{outfig}/design_scatter.png', dpi=100); print('saved')
