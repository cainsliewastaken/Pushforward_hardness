"""How few steps? For each saved run, freeze the kernel spectrum at kernel time tau_f = lam1(0) t_f and score
   S_f(T) = sum_k eE2_k(t_f) exp(-2 lam_k(t_f) (T - t_f))   (current-basis energies at t_f, frozen spectrum)
and also lam_100(t_f) and the relative kernel change |lam(t_f)-lam(0)|/lam(0) at rank 100.
Spearman between each score and the measured ||E(T)|| across ALL designs (H, depth, scale) at fixed Delta."""
import numpy as np, sys
from scipy.stats import spearmanr
R = []
for fn in sys.argv[1:]:
    r = np.load(fn, allow_pickle=True).item(); tag = r['tag']
    if r['n'] != 200 or r['seed'] != 0 or (r['H'] == 32 and r['depth'] == 1) or '_ds' in tag: continue
    sc = 0.25 if '_os0.25' in tag else (4.0 if '_os4' in tag else (16.0 if '_os16' in tag else 1.0))
    R.append(dict(D=r['Delta'], H=r['H'], d=r['depth'], s=sc, lam=r['lam'][:, ::-1], eE2=r['eE2'][:, ::-1], t=r['t'], E=r['E'], tag=tag, lam1_0=r['lam'][0][-1]))
for D in sorted({q['D'] for q in R}):
    grp = [q for q in R if q['D'] == D]
    # budget in kernel time: tau_T = 3000*lam1(0) at s=1; scaled runs have T=3000/s^2 with lam1 ~ s^2 -> same tau
    print(f"== Delta={D}: {len(grp)} designs incl. scales {sorted({q['s'] for q in grp})}")
    print(f"   {'tau_f':>7s} {'t_f(s=1)':>8s} | rho(S_f, E_T) rho(-lam100(t_f), E_T) rho(-rel.change100, E_T) | same, s=1 designs only")
    for tau_f in [0.1, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]:
        rows = []
        for q in grp:
            tf = tau_f/q['lam1_0']; i = max(0, np.searchsorted(q['t'], tf, side='right')-1)
            T = q['t'][-1]; S = np.sqrt((q['eE2'][i]*np.exp(-2*q['lam'][i]*(T-q['t'][i]))).sum())
            rows.append((S, -q['lam'][i][99]/q['lam1_0'], -(q['lam'][i][99]-q['lam'][0][99])/q['lam'][0][99], q['E'][-1], q['s']))
        rows = np.array(rows)
        allr = [spearmanr(rows[:, j], rows[:, 3])[0] for j in range(3)]
        m = rows[:, 4] == 1.0; s1 = [spearmanr(rows[m, j], rows[m, 3])[0] for j in range(3)]
        print(f"   {tau_f:7.1f} {tau_f/grp[0]['lam1_0']:8.3f} | {allr[0]:6.2f} {allr[1]:8.2f} {allr[2]:10.2f} | {s1[0]:6.2f} {s1[1]:8.2f} {s1[2]:10.2f}")
