"""Offline ranking test on saved runs: can an INIT-TIME score rank designs (H, depth, scale) by final training
error at budget T, at fixed target? Scores:
  lazy(T)   = sum_k y_k^2 exp(-2 lam_k(0) T)                       (K0 basis, no lift)
  lift(T,p) = sum_k y_k^2 exp(-2 lam_k(0) k^p T)                   (K0 basis, lifted spectrum, p per target class)
  tail0     = lam_100(0)                                           (initial tail level)
Reports Spearman rho between score and measured ||E(T)|| across designs, per (Delta, budget)."""
import numpy as np, sys
from scipy.stats import spearmanr
R = []
for fn in sys.argv[1:]:
    r = np.load(fn, allow_pickle=True).item()
    if r['n'] != 200 or r['seed'] != 0 or (r['H'] == 32 and r['depth'] == 1): continue
    tag = r['tag']; sc = 0.25 if '_os0.25' in tag else (4.0 if '_os4' in tag else (16.0 if '_os16' in tag else 1.0))
    if '_ds' in tag: continue
    lam0 = r['lam'][0][::-1]; y = r['yV2'][0][::-1]; t = r['t']; E = r['E']
    R.append(dict(D=r['Delta'], H=r['H'], d=r['depth'], s=sc, lam0=lam0, y=y, t=t, E=E, T=t[-1], tag=tag))
pcal = {0.1: 0.0, 0.25: 0.55, 0.5: 2.0, 0.75: 2.9, 1.0: 3.2, 1.5: 3.35, 2.0: 3.4}   # from lift_exponent (s=1)
k = np.arange(1, 601)
def score(lam0, y, T, p): return np.sqrt((y*np.exp(-2*lam0*np.minimum(k**p, 1e12)*T)).sum())
print("per target: designs ranked; Spearman rho(score, ||E(T_b)||) for budgets T_b; higher rho = better ranking")
for D in sorted({q['D'] for q in R}):
    grp = [q for q in R if q['D'] == D and q['s'] == 1.0 and q['T'] >= 3000]
    if len(grp) < 4: continue
    print(f"== Delta={D}: {len(grp)} designs (s=1): " + ', '.join(f"H{q['H']}d{q['d']}" for q in grp))
    for Tb in [100, 300, 1000, 3000]:
        ET = np.array([q['E'][np.searchsorted(q['t'], Tb, side='right')-1] for q in grp])
        lazy = np.array([score(q['lam0'], q['y'], Tb, 0.) for q in grp]); lift = np.array([score(q['lam0'], q['y'], Tb, pcal[D]) for q in grp])
        tail = np.array([-q['lam0'][99] for q in grp]); top = np.array([-q['lam0'][0] for q in grp])
        print(f"   T={Tb:5d}: rho lazy={spearmanr(lazy, ET)[0]:5.2f}  lift(p={pcal[D]})={spearmanr(lift, ET)[0]:5.2f}  -lam100(0)={spearmanr(tail, ET)[0]:5.2f}  -lam1(0)={spearmanr(top, ET)[0]:5.2f} | E(T) range {ET.min():.4f}-{ET.max():.4f}")
    # across scales at H96 d2
    grp2 = [q for q in R if q['D'] == D and q['H'] == 96 and q['d'] == 2]
    if len(grp2) >= 3:
        print(f"   scales at H96 d2: " + ', '.join(f"s={q['s']}: E(T)={q['E'][-1]:.4f} lazy={score(q['lam0'], q['y'], q['T'], 0.):.3f}" for q in sorted(grp2, key=lambda q: q['s'])))
