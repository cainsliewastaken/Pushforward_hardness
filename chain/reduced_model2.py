"""Init-only reduced model v2 (GD, squared loss), amplitude law DERIVED from E-dot:
  modes k of K0 (lam0_k, c0_k = E0 energy), per-layer per-mode energies q_lk = v_k^T K_l v_k (init).
  lift_k(t) = 1 + exp(B.[1, log P, log(k/100), depth-2, log frough]),  P = prod_hidden A_l   (the one fitted curve)
  lam_k = lam0_k lift_k ;  dc_k/dt = -2 lam_k c_k ;  dJ_k/dt = sqrt(c_k) sqrt(lift_k)   (integrated residual per mode,
  with the feature norm lifted);  A_l^2 = sum over top-16 k of q_lk J_k^2  (rank-16 amplitude of dW_l).
Outputs E(t), A_l(t), eff. rank of dW_l. Evaluates against saved runs (ground truth E_meas(T), measured amplitudes)."""
import numpy as np, sys, os, glob
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
from scipy.stats import spearmanr
DBM = float(os.environ.get('DBM', '0'))
DRIVE = os.environ.get('DRIVE', '0') == '1'; COS = float(os.environ.get('COS', '0.25'))
CL = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'closures.npz'), allow_pickle=True); B = CL['c2']
def integrate(lam0, c0, q, depth, T, nsteps=1500, H2=1.0):
    lam0 = np.asarray(lam0, float); c = np.asarray(c0, float).copy(); q = np.asarray(q, float); k = np.arange(1, len(lam0)+1)
    frough = max(c[100:].sum()/c.sum(), 1e-6); L = q.shape[0]; hid = list(range(1, L-1)) if L > 2 else [1]
    lift_prev = np.ones_like(c); lam_x = np.zeros_like(c)
    J = np.zeros_like(c); ts = np.concatenate([[0.], np.geomspace(1e-3/lam0[0], T, nsteps)]); A = np.full(L, 1e-3)
    out = dict(t=[], E=[], lift100=[], A=[])
    for i in range(1, len(ts)):
        dt = ts[i]-ts[i-1]; P = max(np.prod(A[hid]), 1e-6)
        lift = 1 + np.exp(B[0] + B[1]*np.log(P) + B[2]*np.log(k/100.) + B[3]*(depth-2) + B[4]*np.log(frough) + B[5]*np.log(k/100.)*np.log(frough) + B[6]*np.log(k/100.)*np.log(P))
        lam = lam0*lift
        J = J + np.sqrt(np.maximum(c, 0))*np.sqrt(lift)*dt
        if DRIVE:
            # rank-one kernel growth along the current residual direction e (drive identity), distributed over modes
            # in proportion to their share of the residual; accumulates as an extra eigenvalue per mode
            tot = max(c.sum(), 1e-300); lam_eff = ((lam + lam_x)*c).sum()/tot
            gnorm = np.sqrt(max(((lam + lam_x)*c).sum(), 0))
            rq = min(0.5, 4e-4*(max(P, 1e-3)/0.1)**1.2)
            rate = 2*gnorm*np.sqrt(lam_eff)*np.sqrt(rq)*H2*COS
            lam_x = lam_x + rate*(c/tot)*dt
            lam_dec = lam + lam_x
        else: lam_dec = lam
        c = c*np.exp(-2*lam_dec*dt)
        if DBM > 0:   # finite-perturbation eigenvector flow: mode k exchanges a fraction DBM*dlog(lam_k) of its energy
            # with neighbours, Lorentzian in eigenvalue with width lam_k (the perturbation size at that level)
            dlog = np.log(lift) - np.log(lift_prev); lift_prev = lift.copy(); f = np.clip(DBM*dlog, 0, 0.5)
            if f.max() > 0:
                dl = lam[:, None]-lam[None, :]; W = 1.0/(1.0 + (dl/lam[:, None])**2); np.fill_diagonal(W, 0); W = W/W.sum(1, keepdims=True)
                c = c - f*c + W.T @ (f*c)
        contrib = q*J[None, :]**2                                  # L x 600
        A = np.sqrt(np.sort(contrib, axis=1)[:, -16:].sum(1)) + 1e-6
        out['t'].append(ts[i]); out['E'].append(np.sqrt(c.sum())); out['lift100'].append(lift[99]); out['A'].append(A.copy())
    p = contrib/contrib.sum(1, keepdims=True); erank = np.exp(-(p*np.log(np.maximum(p, 1e-300))).sum(1))
    return {k_: np.array(v) for k_, v in out.items()} | dict(erank=erank)
def inputs_for(r, fn):
    tag = r['tag']
    if 'snap_Kl' in r and len(r['snap_Kl']) > 0:
        K0 = r['snap_K'][0].astype(np.float64); w, V = np.linalg.eigh(K0); V = V[:, ::-1]
        q = np.array([np.einsum('ij,ji->i', V.T @ r['snap_Kl'][0][l].astype(np.float64), V) for l in range(r['snap_Kl'].shape[1])])
        return r['lam0'][::-1], r['e0V2_0'][::-1], q
    qf = f"{L63_ROOT}/runs/initq/q_{tag}.npz"
    if os.path.exists(qf):
        z = np.load(qf); return z['lam0'], z['c0'], z['q']
    return None
if __name__ == '__main__':
    rows = []
    for pat in sys.argv[1:]:
        for fn in glob.glob(pat):
            r = np.load(fn, allow_pickle=True).item()
            if r['n'] != 200 or (r['H'] == 32 and r['depth'] == 1): continue
            inp = inputs_for(r, fn)
            if inp is None: continue
            lam0, c0, q = inp; T = r['t'][-1]; m = integrate(lam0, c0, q, r['depth'], T, H2=float(r.get('H2_0', 1.0)))
            rec = dict(tag=r['tag'], D=r['Delta'], H=r['H'], depth=r['depth'], s=r['outscale'], T=T, E_meas=r['E'][-1], E_pred=m['E'][-1], lift_pred=m['lift100'][-1], A_pred=m['A'][-1].round(2), erank_pred=m['erank'].round(1))
            if 'snap_svdW' in r and len(r['snap_svdW']) > 0:
                rec['A_meas'] = np.array([np.linalg.norm(sv[:16]) for sv in r['snap_svdW'][-1]]).round(2)
            i = max(0, np.searchsorted(r['t'], 0.3*T, side='right')-1); j = max(0, np.searchsorted(m['t'], 0.3*T, side='right')-1)
            rec['E_meas@0.3'] = r['E'][i]; rec['E_pred@0.3'] = m['E'][j]
            rows.append(rec)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv('reduced_model2_eval.csv', index=False)
    print(f"{'tag':30s} {'D':>4s} {'d':>2s} {'s':>5s} | {'E_meas':>7s} {'E_pred':>7s} {'ratio':>6s} | {'lift100':>8s} | A_pred ; A_meas ; erank_pred")
    for _, qq in df.sort_values(['D', 's', 'H', 'depth']).iterrows():
        print(f"{qq.tag:30s} {qq.D:4.2f} {qq.depth:2d} {qq.s:5.2f} | {qq.E_meas:7.4f} {qq.E_pred:7.4f} {qq.E_pred/qq.E_meas:6.2f} | {qq.lift_pred:8.1f} | {qq.A_pred} ; {qq.get('A_meas', '-')} ; {qq.erank_pred}")
    for name, sub in [('all', df), ('s=1', df[df.s == 1])]:
        print(f"\n== {name}: {len(sub)} designs: Spearman(pred, meas) at T = {spearmanr(sub.E_pred, sub.E_meas)[0]:.2f}; at 0.3T {spearmanr(sub['E_pred@0.3'], sub['E_meas@0.3'])[0]:.2f}; median ratio {np.median(sub.E_pred/sub.E_meas):.2f}, 10-90% {np.percentile(sub.E_pred/sub.E_meas, [10, 90]).round(2)}")
        for D in sorted(sub.D.unique()):
            g = sub[sub.D == D]
            if len(g) >= 4: print(f"   Delta={D}: n={len(g)} Spearman {spearmanr(g.E_pred, g.E_meas)[0]:.2f} (0.3T: {spearmanr(g['E_pred@0.3'], g['E_meas@0.3'])[0]:.2f}); median ratio {np.median(g.E_pred/g.E_meas):.2f}")
