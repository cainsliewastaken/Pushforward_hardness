"""Spike theory as a design predictor. Inputs: des_/lay_ files (lay_ have spikes). Ground truth: E_tr, E_val at
kernel-time budgets. Predictors: (A) measured-kernel few-step score S_f(tau_f) [reference]; (B) spike amplitude at
tau_f, a_f; (C) spike-theory score: CKR with K0 spectrum lifted by lam_k -> lam_k (1 + c_d a_f^2 (k/100)^p0)
where c_d = depth factor (fit once per depth from s=1 runs: c_d = (lift100(T)-1)/a(T)^2) and p0 = 2;
(D) init-only chi. Spearman per (Delta, budget), all scales pooled and s=1 only. Dedupe by tag (latest file)."""
import numpy as np, sys, glob, os
from scipy.stats import spearmanr
files = {}
for fn in sys.argv[1:]:
    tag = os.path.basename(fn).split('_', 1)[1][:-4]
    if tag not in files or os.path.getmtime(fn) > os.path.getmtime(files[tag]): files[tag] = fn
R = [np.load(fn, allow_pickle=True).item() for fn in files.values()]
R = [r for r in R if not (r['H'] == 32 and r['depth'] == 1) and r['n'] == 200]
print(len(R), 'unique runs;', sum('snap_spike_amp' in r and len(r['snap_spike_amp']) > 0 for r in R), 'with spikes')
TAUB = [1e4, 3e4, 1e5]; NOM = np.array([1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000])
def nom(tau): return NOM[np.argmin(np.abs(np.log(NOM)-np.log(max(tau, 1e-6))))]
rows = []
for r in R:
    has = 'snap_spike_amp' in r and len(r['snap_spike_amp']) > 0
    L = len(r['snap_svW'][0]) if has else None
    taus = r['snap_t']*r['lam1_0']; lam0 = r['lam0'][::-1]; e0 = r['e0V2_0'][::-1]; k = np.arange(1, len(lam0)+1)
    d = dict(tag=r['tag'], D=r['Delta'], depth=r['depth'], H=r['H'], s=r['outscale'], chi=-r['chi'])
    amp = {}
    for j, tau in enumerate(taus):
        if has: amp[nom(tau)] = max(r['snap_spike_amp'][j][1:L-1]) if L > 2 else r['snap_spike_amp'][j][1]
    for tb in TAUB:
        t = tb/r['lam1_0']; i = max(0, np.searchsorted(r['t'], t, side='right')-1)
        d[f'Etr@{tb:g}'] = r['E'][i]; d[f'Ev@{tb:g}'] = r['Ev'][i]
        for j, tau in enumerate(taus):
            tf = nom(tau); ts = r['snap_t'][j]
            if t <= ts: continue
            w, V = np.linalg.eigh(r['snap_K'][j].astype(np.float64)); w = np.sort(np.maximum(w, 0))[::-1]
            ii = max(0, np.searchsorted(r['t'], ts, side='right')-1); c2 = r['eE2'][ii][::-1]
            d[f'Sf{tf}@{tb:g}'] = np.sqrt((c2*np.exp(-2*w*(t-ts))).sum())
            if has:
                a = amp[tf]; d[f'a{tf}'] = -a
                for cd_name, cd in [('c1', 1.0), ('cdepth', {2: 4., 3: 16., 4: 25.}.get(r['depth'], 4.))]:
                    lift = 1 + cd*a**2*(k/100.)**2
                    d[f'T{cd_name}{tf}@{tb:g}'] = np.sqrt((e0*np.exp(-2*lam0*lift*t)).sum())
    rows.append(d)
import pandas as pd
df = pd.DataFrame(rows); df.to_csv('spike_design_table.csv', index=False)
preds = ['chi', 'Sf30@', 'Sf100@', 'Sf300@', 'a30', 'a100', 'a300', 'Tc130@', 'Tc1100@', 'Tc1300@', 'Tcdepth30@', 'Tcdepth100@', 'Tcdepth300@']
for D in sorted(df.D.unique()):
    for name, sub in [('all scales, runs with spikes', df[(df.D == D) & df['a100'].notna()]), ('s=1 only (spikes)', df[(df.D == D) & (df.s == 1) & df['a100'].notna()]), ('all runs (kernel scores)', df[df.D == D])]:
        if len(sub) < 5: continue
        print(f"\n== Delta={D}: {name}: {len(sub)} designs")
        print(f"   {'budget':>7s} {'tgt':>4s} | " + ' '.join(f'{p.replace("@", ""):>10s}' for p in preds))
        for tb in TAUB:
            for tgt in ['Etr', 'Ev']:
                y = sub[f'{tgt}@{tb:g}'].values; out = []
                for p in preds:
                    col = p if '@' not in p else f'{p}{tb:g}'
                    x = sub[col].values if col in sub else np.full(len(y), np.nan); m = np.isfinite(x) & np.isfinite(y)
                    out.append(spearmanr(x[m], y[m])[0] if m.sum() >= 5 else np.nan)
                print(f"   {tb:7.0e} {tgt:>4s} | " + ' '.join(f'{v:10.2f}' for v in out) + f"  [{y.min():.3f}-{y.max():.3f}]")
