"""Final cross-sweep figures: (1) p, aT, bYT vs Delta (all runs); (2) CKR late ratio histogram; (3) alpha vs depth."""
import numpy as np, sys, os, glob
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
outfig = sys.argv[1]; os.makedirs(outfig, exist_ok=True)
rows = []
for fn in sys.argv[2:]:
    r = np.load(fn, allow_pickle=True).item(); lam0 = r['lam'][0][::-1]; lamT = r['lam'][-1][::-1]; t = r['t']; E = r['E']
    if r['H'] == 32 and r['depth'] == 1: continue
    x = np.log(np.arange(20, 301)); ok = lam0[19:300] > 0
    p = np.polyfit(x[ok], np.log(lamT[19:300][ok]/lam0[19:300][ok]), 1)[0]
    aT = -np.polyfit(x[ok], np.log(np.maximum(lamT[19:300][ok], 1e-300)), 1)[0]; a0 = -np.polyfit(x[ok], np.log(lam0[19:300][ok]), 1)[0]
    m = t > t[-1]/10; alpha = -np.polyfit(np.log(t[m]), np.log(E[m]), 1)[0]
    lam = r['lam'][:, ::-1]; src = (r['e0V2'] if 'e0V2' in r else r['yV2'])[:, ::-1]
    late = slice(len(t)*3//4, None)
    S = np.array([(src[i]*np.exp(-2*lam[i]*t[i])).sum() for i in range(len(t))])
    ratio = np.sqrt(S[late])/E[late]
    rows.append(dict(D=r['Delta'], H=r['H'], d=r['depth'], n=r['n'], p=p, aT=aT, a0=a0, alpha=alpha, ckr_med=np.median(ratio), ckr_min=ratio.min(), ckr_max=ratio.max(), ckr_all=np.sqrt(S[1:])/E[1:], T=t[-1], tag=r['tag']))
cols = {32: 'tab:blue', 96: 'tab:orange', 256: 'tab:green', 512: 'tab:red'}
fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
for q in rows:
    jit = (q['d']-2.5)*0.02
    ax[0].plot(q['D']+jit, q['p'], 'o', color=cols[q['H']], ms=3+q['d'], alpha=0.7)
    ax[1].plot(q['D']+jit, q['aT'], 'o', color=cols[q['H']], ms=3+q['d'], alpha=0.7); ax[1].plot(q['D']+jit, q['a0'], 'x', color=cols[q['H']], ms=4, alpha=0.4)
    ax[2].plot(q['d'], q['alpha'], {0.1: 'o', 0.25: 'v', 0.5: 's', 0.75: 'P', 1.0: '^', 1.5: 'X', 2.0: 'D'}.get(q['D'], 'o'), color=cols[q['H']], alpha=0.7)
ax[0].set_title('lift exponent p: lam_k(T)/lam_k(0) ~ k^p  (marker size = depth, colour = width)'); ax[0].set_xlabel('Delta (target roughness)')
ax[1].set_title('kernel spectral slope: x = a0 (init), o = aT (final)'); ax[1].set_xlabel('Delta')
ax[2].set_title('late exponent alpha vs depth (marker = Delta)'); ax[2].set_xlabel('depth')
allr = np.concatenate([q['ckr_all'] for q in rows])
ax[3].hist(np.clip(allr, 0, 3), bins=60, color='gray'); ax[3].set_title(f'CKR ratio sqrt(S)/||E||, all times, all {len(rows)} runs: median {np.median(allr):.2f}, 5-95% {np.percentile(allr, 5):.2f}-{np.percentile(allr, 95):.2f}'); ax[3].set_xlabel('ratio')
fig.tight_layout(); fig.savefig(f'{outfig}/final_summary.png', dpi=100)
print(f"{'tag':30s} {'D':>4s} {'H':>4s} {'d':>2s} {'p':>5s} {'aT':>5s} {'alpha':>5s} {'ckr med':>7s} {'min':>5s} {'max':>5s}")
for q in sorted(rows, key=lambda q: (q['D'], q['H'], q['d'])):
    print(f"{q['tag']:30s} {q['D']:4.2f} {q['H']:4d} {q['d']:2d} {q['p']:5.2f} {q['aT']:5.2f} {q['alpha']:5.2f} {q['ckr_med']:7.2f} {q['ckr_min']:5.2f} {q['ckr_max']:5.2f}")
print('CKR ratio over all runs and times: median', np.median(allr), '5-95%', np.percentile(allr, [5, 95]))
