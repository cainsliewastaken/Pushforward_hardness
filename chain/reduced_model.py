"""Init-only reduced model (GD, squared loss). Inputs per design: lam0 (K0 spectrum, desc), c0 (E0 energy per K0 mode),
depth, closures.npz. State: c_k, A_l (rank-16 amplitude per layer). Dynamics:
  lam_k = lam0_k * (1 + exp(b0 + b1 log P + b2 log(k/100) + b3 (depth-2) + b4 log frough)),  P = prod_hidden A_l
  dc_k/dt = -2 lam_k c_k ;  dA_l/dt = kappa * sqrt(share_l * sum_k lam_k c_k)
Returns E(t) = sqrt(sum c_k) on a time grid. Also provides a loader for saved runs and an evaluation entry point."""
import numpy as np, sys, os, glob
from scipy.stats import spearmanr
CL = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'closures.npz'), allow_pickle=True)
B = CL['c2']; SH = CL['shares'].item()
def shares_for(depth):
    if depth in SH: return SH[depth]
    if depth == 1: return np.array([0.4, 0.6])
    return SH[max(SH)]
def integrate(lam0, c0, depth, T, kappa=1.0, nsteps=3000, frough=None):
    lam0 = np.asarray(lam0, float); c = np.asarray(c0, float).copy(); k = np.arange(1, len(lam0)+1)
    if frough is None: frough = c[100:].sum()/c.sum()
    L = depth+1; sh = shares_for(depth); hid = list(range(1, L-1)) if L > 2 else [1]
    A = np.full(L, 1e-3); ts = np.concatenate([[0.], np.geomspace(1e-4/max(lam0[0], 1e-12)*lam0[0]*1e-2, T, nsteps)])
    out_t = []; out_E = []; out_lift = []
    for i in range(1, len(ts)):
        dt = ts[i]-ts[i-1]
        P = np.prod(A[hid]); lift = 1 + np.exp(B[0] + B[1]*np.log(P) + B[2]*np.log(k/100.) + B[3]*(depth-2) + B[4]*np.log(max(frough, 1e-6)))
        lam = lam0*lift
        drive = (lam*c).sum()
        c = c*np.exp(-2*lam*dt)
        A = A + kappa*np.sqrt(np.maximum(sh*drive, 0))*dt
        out_t.append(ts[i]); out_E.append(np.sqrt(c.sum())); out_lift.append(lift[99])
    return np.array(out_t), np.array(out_E), np.array(out_lift), A
def load_runs(patterns):
    R = []
    for pat in patterns:
        for fn in glob.glob(pat):
            r = np.load(fn, allow_pickle=True).item(); tag = r['tag']
            if r['n'] != 200 or (r['H'] == 32 and r['depth'] == 1): continue
            if 'lam0' in r: lam0 = r['lam0'][::-1]; c0 = r['e0V2_0'][::-1]; s = r['outscale']
            else:
                lam0 = r['lam'][0][::-1]; c0 = (r['e0V2'][0] if 'e0V2' in r else r['yV2'][0])[::-1]
                s = 0.25 if '_os0.25' in tag else (4.0 if '_os4' in tag else (16.0 if '_os16' in tag else 1.0))
            R.append(dict(tag=tag, D=r['Delta'], H=r['H'], depth=r['depth'], s=s, lam0=lam0, c0=c0, t=r['t'], E=r['E'], T=r['t'][-1], fn=fn))
    return R
if __name__ == '__main__':
    kappa = float(os.environ.get('KAPPA', '1.0'))
    R = load_runs(sys.argv[1:]); print(len(R), 'designs; kappa', kappa)
    rows = []
    for q in R:
        t, E, lift, A = integrate(q['lam0'], q['c0'], q['depth'], q['T'], kappa=kappa)
        # measured at a few budgets
        rec = dict(tag=q['tag'], D=q['D'], H=q['H'], depth=q['depth'], s=q['s'], T=q['T'], E_meas=q['E'][-1], E_pred=E[-1], lift_pred=lift[-1], A=A.round(2))
        for frac in [0.1, 0.3]:
            i = max(0, np.searchsorted(q['t'], frac*q['T'], side='right')-1); j = max(0, np.searchsorted(t, frac*q['T'], side='right')-1)
            rec[f'E_meas@{frac}'] = q['E'][i]; rec[f'E_pred@{frac}'] = E[j]
        rows.append(rec)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv('reduced_model_eval.csv', index=False)
    print(f"{'tag':30s} {'D':>4s} {'d':>2s} {'s':>5s} | {'E_meas':>7s} {'E_pred':>7s} {'ratio':>6s} | {'lift100 pred':>12s} | amplitudes")
    for _, q in df.sort_values(['D', 's', 'H', 'depth']).iterrows():
        print(f"{q.tag:30s} {q.D:4.2f} {q.depth:2d} {q.s:5.2f} | {q.E_meas:7.4f} {q.E_pred:7.4f} {q.E_pred/q.E_meas:6.2f} | {q.lift_pred:12.1f} | {q.A}")
    for name, sub in [('all', df), ('s=1', df[df.s == 1])]:
        print(f"\n== {name}: {len(sub)} designs: Spearman(pred, meas) at T = {spearmanr(sub.E_pred, sub.E_meas)[0]:.2f}; median ratio {np.median(sub.E_pred/sub.E_meas):.2f}, 10-90% {np.percentile(sub.E_pred/sub.E_meas, [10, 90]).round(2)}")
        for D in sorted(sub.D.unique()):
            g = sub[sub.D == D]
            if len(g) >= 4: print(f"   Delta={D}: n={len(g)} Spearman {spearmanr(g.E_pred, g.E_meas)[0]:.2f} (at 0.3T: {spearmanr(g['E_pred@0.3'], g['E_meas@0.3'])[0]:.2f}); median ratio {np.median(g.E_pred/g.E_meas):.2f}")
