"""Closure curves with rank-16 spike amplitudes, from lay_*.npy (snapshots with per-layer dW singular values, kernel,
factor roughness where available). Measures per snapshot: per-layer A16_l = ||top-16 singular values||_2 of dW_l,
P = prod over hidden layers of A16_l, S = sum over hidden; lift_k = lam_k(t)/lam_k(0) at k = 50, 100, 200, 400;
rough(delta_0) where logged. Reports Spearman of lift_k vs P, S, and max single amplitude; and of roughness vs P."""
import numpy as np, sys, os, glob
from scipy.stats import spearmanr
files = {}
for fn in sys.argv[1:]:
    tag = os.path.basename(fn)[4:-4]
    if tag not in files or os.path.getmtime(fn) > os.path.getmtime(files[tag]): files[tag] = fn
rows = []
for fn in files.values():
    r = np.load(fn, allow_pickle=True).item()
    if 'snap_svdW' not in r or len(r['snap_svdW']) == 0: continue
    L = len(r['snap_svdW'][0]); taus = r['snap_t']*r['lam1_0']
    w0 = np.sort(np.linalg.eigvalsh(r['snap_K'][0].astype(np.float64)))[::-1]
    for j in range(len(taus)):
        w = np.sort(np.linalg.eigvalsh(r['snap_K'][j].astype(np.float64)))[::-1]
        A16 = [np.linalg.norm(r['snap_svdW'][j][l][:16]) for l in range(L)]; A1 = [r['snap_svdW'][j][l][0] for l in range(L)]
        hid = list(range(1, L-1)) if L > 2 else [1]
        d = dict(tag=r['tag'], D=r['Delta'], depth=r['depth'], H=r['H'], s=r['outscale'], tau=taus[j],
                 P16=np.prod([A16[l] for l in hid]), S16=np.sum([A16[l] for l in hid]), P1=np.prod([A1[l] for l in hid]), A1max=max(A1[l] for l in hid),
                 Pall16=np.prod(A16), lift50=w[49]/w0[49], lift100=w[99]/w0[99], lift200=w[199]/w0[199], lift400=w[399]/max(w0[399], 1e-300))
        if 'snap_factor_rough' in r and len(r['snap_factor_rough']) > j: d['rough_d0'] = r['snap_factor_rough'][j][0][2]
        rows.append(d)
import pandas as pd
df = pd.DataFrame(rows); print(len(df), 'snapshots from', df.tag.nunique(), 'runs; depths', sorted(df.depth.unique()), 'scales', sorted(df.s.unique()))
m = df.lift100 > 1.5
print("\nSpearman of lift_k with amplitude measures (snapshots with lift100 > 1.5):")
for k in ['lift50', 'lift100', 'lift200', 'lift400']:
    print(f"  {k:8s}: P16 {spearmanr(df.P16[m], df[k][m])[0]:.2f}  S16 {spearmanr(df.S16[m], df[k][m])[0]:.2f}  P1 {spearmanr(df.P1[m], df[k][m])[0]:.2f}  A1max {spearmanr(df.A1max[m], df[k][m])[0]:.2f}  Pall16 {spearmanr(df.Pall16[m], df[k][m])[0]:.2f}")
for D in sorted(df.D.unique()):
    mm = m & (df.D == D); print(f"  Delta={D} (n={mm.sum()}): lift100 vs P16 {spearmanr(df.P16[mm], df.lift100[mm])[0]:.2f}, vs P1 {spearmanr(df.P1[mm], df.lift100[mm])[0]:.2f}, vs A1max {spearmanr(df.A1max[mm], df.lift100[mm])[0]:.2f}")
# log-log slopes by depth: lift100 vs P16
print("\nlog-log slope of lift100-1 vs P16 by (Delta, depth), and residual scatter (std of log residual):")
for (D, d), g in df[m].groupby(['D', 'depth']):
    if len(g) >= 4:
        c = np.polyfit(np.log(g.P16), np.log(g.lift100-1), 1); res = np.log(g.lift100-1) - np.polyval(c, np.log(g.P16))
        print(f"  Delta={D} depth={d}: slope {c[0]:.2f}, intercept {c[1]:.2f}, scatter {res.std():.2f}, n={len(g)}")
if 'rough_d0' in df:
    mr = m & df.rough_d0.notna(); print(f"\nrough(delta_0) vs P16: Spearman {spearmanr(df.P16[mr], df.rough_d0[mr])[0]:.2f} (n={mr.sum()}); vs P1 {spearmanr(df.P1[mr], df.rough_d0[mr])[0]:.2f}; vs A1max {spearmanr(df.A1max[mr], df.rough_d0[mr])[0]:.2f}")
df.to_csv('closure16_table.csv', index=False)
