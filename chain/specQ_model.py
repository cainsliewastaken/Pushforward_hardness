"""Statistical model of spec(Q): mass from init statistic, roughness distribution from the second-order-feature spectra.
(1) rho_k(P): normalised spectrum of the second-order features over current-kernel ranks (sweep S: sof_q) as a
    function of the hidden amplitude product P (layers2 amplitudes, matched by time). Fit log rho_k = a + b log k + c log P + d log k log P.
(2) tr Q = kappa * sum_l A_l^2 m_l, kappa fitted once on sweep C (snap_Q) with layers2 amplitudes and initm m_l.
(3) spec(K(T)) = spec(K0) (+) spec(Q(T)) by random rotation (3 draws); gains vs measured on held-out designs,
    with A_l(T) from the reduced model (DRIVE=0) or, where available, measured.
"""
import numpy as np, sys, os, glob
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RUNS = L63_ROOT + '/runs'
def amp_interp(lay_tag):
    r = np.load(f'{RUNS}/layers2/lay_{lay_tag}.npy', allow_pickle=True).item(); L = len(r['snap_svdW'][0]); hid = list(range(1, L-1)) if L > 2 else [1]
    ts = r['snap_t']; A = np.array([[np.linalg.norm(r['snap_svdW'][j][l][:16]) for l in range(L)] for j in range(len(ts))])
    return lambda t: np.array([np.exp(np.interp(np.log(max(t, ts[0])), np.log(ts), np.log(np.maximum(A[:, l], 1e-6)))) for l in range(L)]), hid
# ---- (1) roughness distribution of second-order features vs P
X = []; y = []
for fn in glob.glob(f'{RUNS}/sweepS/late_*.npy'):
    r = np.load(fn, allow_pickle=True).item(); tag = r['tag']
    if not os.path.exists(f'{RUNS}/layers2/lay_{tag}.npy'): continue
    af, hid = amp_interp(tag); k = np.arange(1, 601)
    for i in range(0, len(r['t']), 10):
        q = r['sof_q'][i][::-1]; q = q/q.sum(); A = af(r['t'][i]); P = np.prod(A[hid])
        if P < 0.3: continue
        sel = (k >= 5) & (k <= 500) & (q > 0)
        X += [[1., np.log(kk/100.), np.log(P), np.log(kk/100.)*np.log(P)] for kk in k[sel]]; y += list(np.log(q[sel]))
X = np.array(X); y = np.array(y); c1, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ c1
print(f"(1) rho_k(P): coeffs {c1.round(3)}, n={len(y)}, log-residual std {res.std():.2f}")
def rho(P):
    k = np.arange(1, 601); lk = np.log(k/100.); r = np.exp(c1[0] + c1[1]*lk + c1[2]*np.log(P) + c1[3]*lk*np.log(P)); return r/r.sum()
# ---- (2) kappa: tr Q vs sum_l A_l^2 m_l on sweep C
xs = []; ys = []
for fn in glob.glob(f'{RUNS}/sweepC/late_*.npy'):
    r = np.load(fn, allow_pickle=True).item(); tag = r['tag']
    if not os.path.exists(f'{RUNS}/layers2/lay_{tag}.npy') or not os.path.exists(f'{RUNS}/initm/m_{tag}.npz'): continue
    af, hid = amp_interp(tag); m = np.load(f'{RUNS}/initm/m_{tag}.npz')['m']
    for j, ts in enumerate(r['snap_t']):
        trQ = np.trace(r['snap_Q'][j].astype(np.float64)); A = af(ts)
        if trQ > 1e-3: xs.append(np.log((A**2*m).sum())); ys.append(np.log(trQ))
xs = np.array(xs); ys = np.array(ys); slope, icpt = np.polyfit(xs, ys, 1); kappa = np.exp(np.mean(ys - xs))
print(f"(2) tr Q vs sum A^2 m: log-log slope {slope:.2f} (1 = quadratic), kappa (median ratio) {kappa:.2f}, scatter {np.std(ys-xs-np.log(kappa)):.2f}, n={len(xs)}")
# ---- (3) held-out test
import reduced_model2 as M
rng = np.random.default_rng(0)
def free_conv(lam0, specQ, ndraw=3):
    n = len(lam0); out = []
    for _ in range(ndraw):
        R, _ = np.linalg.qr(rng.normal(size=(n, n))); K = np.diag(lam0) + R @ np.diag(specQ) @ R.T; out.append(np.sort(np.linalg.eigvalsh(K))[::-1])
    return np.median(out, 0)
rows = []
for fn in sorted(glob.glob(f'{RUNS}/heldout/des_*.npy')):
    r = np.load(fn, allow_pickle=True).item(); tag = r['tag']
    if not os.path.exists(f'{RUNS}/initm/m_{tag}.npz'): continue
    z = np.load(f'{RUNS}/initq/q_{tag}.npz'); m = np.load(f'{RUNS}/initm/m_{tag}.npz')['m']
    mod = M.integrate(z['lam0'], z['c0'], z['q'], r['depth'], r['t'][-1], H2=float(r['H2_0'])); A = mod['A'][-1]; L = len(A); hid = list(range(1, L-1)) if L > 2 else [1]
    P = np.prod(A[hid]); trQ = kappa*(A**2*m).sum(); specQ = trQ*rho(P)
    lamT = free_conv(z['lam0'], specQ); gm = (r['lam'][-1]/r['lam'][0])[::-1]; gp = lamT/z['lam0']
    rows.append(dict(tag=tag, g100=gp[99]/gm[99], g200=gp[199]/gm[199], g400=gp[399]/gm[399], trQ_pred=trQ, lift_meas100=gm[99], lift_pred100=gp[99], P=P))
    print(f"{tag:26s} P={P:6.2f} trQ_pred={trQ:8.1f} | gain meas/pred at k=100: {gm[99]:8.1f}/{gp[99]:8.1f}  k=200: {gm[199]:8.1f}/{gp[199]:8.1f}  k=400: {gm[399]:9.1f}/{gp[399]:9.1f}")
import pandas as pd
df = pd.DataFrame(rows); print(f"\nheld-out ({len(df)}): median pred/meas gain at k=100 {df.g100.median():.2f}, k=200 {df.g200.median():.2f}, k=400 {df.g400.median():.2f}; 10-90% at k=200 {np.percentile(df.g200, [10, 90]).round(2)}; Spearman(pred, meas) at k=200 {spearmanr(df.lift_pred100, df.lift_meas100)[0]:.2f}")
