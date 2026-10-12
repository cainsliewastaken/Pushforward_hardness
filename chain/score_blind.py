"""Score model predictions against ground truth. usage: python score_blind.py PRED_DIR PREFIX [GT_DIR=../runs/heldout]  (npz files PREFIX_<tag>.npz with hist/cols; GT des_<tag>.npy)"""
import numpy as np, glob, sys, os, itertools
from scipy.stats import spearmanr
pdir, pre = sys.argv[1], sys.argv[2]; gdir = sys.argv[3] if len(sys.argv) > 3 else '../runs/heldout'; rows = []
for f in sorted(glob.glob(f'{pdir}/{pre}_*.npz')):
    tag = os.path.basename(f)[len(pre)+1:-4]; g = f'{gdir}/des_{tag}.npy'
    if not os.path.exists(g): continue
    r = np.load(g, allow_pickle=True).item(); h = np.load(f)['hist']; lam = np.asarray(r['lam'])[:, ::-1]
    rows.append(dict(tag=tag, D=float(r['Delta']), Em=float(r['E'][-1]), Ep=float(h[-1, 1]), g100m=float(lam[-1, 99]/lam[0, 99]), g100p=float(h[-1, 2]), g1m=float(lam[-1, 0]/lam[0, 0]), g1p=float(h[-1, 4]) if h.shape[1] > 4 else np.nan))
if not rows: print('no finished predictions'); sys.exit()
Em = np.array([r['Em'] for r in rows]); Ep = np.array([r['Ep'] for r in rows]); rat = Ep/Em
print(f'{pre}: {len(rows)} designs scored')
print(f'  E(T): Spearman {spearmanr(Em, Ep).correlation:.2f}; median pred/meas {np.median(rat):.2f}; 10-90% [{np.percentile(rat,10):.2f} {np.percentile(rat,90):.2f}]; max |log ratio| {np.max(np.abs(np.log(rat))):.2f}')
g = np.array([r['g100p']/r['g100m'] for r in rows]); print(f'  gain100: median pred/meas {np.median(g):.2f}; 10-90% [{np.percentile(g,10):.2f} {np.percentile(g,90):.2f}]')
for D in sorted(set(r['D'] for r in rows)):
    idx = [i for i, r in enumerate(rows) if r['D'] == D]
    if len(idx) > 2: print(f'  Delta={D}: n={len(idx)} Spearman {spearmanr(Em[idx], Ep[idx]).correlation:.2f}; median ratio {np.median(rat[idx]):.2f}; range {rat[idx].min():.2f}-{rat[idx].max():.2f}')
pairs = list(itertools.combinations(range(len(rows)), 2)); ok = [np.sign(Em[i]-Em[j]) == np.sign(Ep[i]-Ep[j]) for i, j in pairs]
big = [(i, j) for i, j in pairs if max(Em[i], Em[j])/min(Em[i], Em[j]) > 1.3]; okb = [np.sign(Em[i]-Em[j]) == np.sign(Ep[i]-Ep[j]) for i, j in big]
print(f'  pairwise: {100*np.mean(ok):.0f}% of {len(pairs)}; >1.3x apart: {100*np.mean(okb):.0f}% of {len(big)}')
gaps = np.array([max(Em[i], Em[j])/min(Em[i], Em[j]) for i, j in pairs]); okа = np.array(ok)
for lo, hi in [(1.0, 1.05), (1.05, 1.1), (1.1, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 1e9)]:
    m = (gaps > lo) & (gaps <= hi)
    if m.sum(): print(f'    true gap {lo:.2f}-{hi if hi < 1e9 else "inf"}: {100*okа[m].mean():.0f}% correct of {m.sum()} pairs')
for r in rows: print(f"    {r['tag']:28s} meas {r['Em']:.4f} pred {r['Ep']:.4f} ratio {r['Ep']/r['Em']:.2f} | gain100 {r['g100m']:.0f}/{r['g100p']:.0f} gain1 {r['g1m']:.1f}/{r['g1p']:.1f}")
