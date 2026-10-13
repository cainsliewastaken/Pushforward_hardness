"""THEORY.md section 23: Theorem 52 (lifts along the residual beat the lazy bound) tested against the saved snapshots.

Per run (K(t) at 30 geometric snapshot times, E(t) at 400 log times; n = 200):
  (a) pure-targeted surrogate: K~(t) = K_0 + sum_{s: t_s <= t} q_s e_s e_s^T with e_s = E(t_s)/||E(t_s)|| and
      q_s = <e_s, (K(t_s) - K(t_{s-1})) e_s> (the along-residual part of the measured stage increment, clipped at 0);
      integrate E~' = -K~ E~ exactly (piecewise constant K~, one eigh per stage) and report ||E~(T)||/||E_lazy(T)|| against
      the real ||E(T)||/||E_lazy(T)||.  Theorem 52: the surrogate ratio is <= 1.
  (b) transverse fraction at creation: ||Delta_s e_s - q_s e_s|| / q_s per stage (median over stages with q_s > 0).
  (c) Proposition 53's sign condition for the principal eigenvector u_s of Delta_s: min_c <u_s, e^{-c K(t_s)} e_s> / <u_s, e_s>
      over c in [0, T - t_s] (fraction of stages where it stays >= 0, and the median of the minimum).
"""
import os, sys, glob, csv, numpy as np
ROOT = os.environ.get('L63_ROOT', os.path.expandvars('$SCRATCH/l63_held_rough'))
SWEEP = sys.argv[1] if len(sys.argv) > 1 else 'sweepA'
OUT = f'{ROOT}/runs/targeted_lift'; os.makedirs(OUT, exist_ok=True)
T_REPORT = [30., 100., 300., 1000., 3000.]

def run_one(f):
    d = np.load(f, allow_pickle=True).item(); n = d['n']; tag = d['tag']
    t = d['t']; E = d['Evec']; st = d['snap_t']; SK = d['snap_K'].astype(np.float64)
    ip = lambda u, v: u @ v / n; nrm = lambda u: np.sqrt(ip(u, u))
    js = [int(np.argmin(np.abs(t - s))) for s in st]
    K0 = SK[0]; E0 = E[js[0]]
    lam0, U0 = np.linalg.eigh(K0); lam0 = np.clip(lam0, 0, None); c0 = U0.T @ E0
    lazy = lambda tt: U0 @ (np.exp(-lam0 * tt) * c0)
    # stage quantities
    S = len(st); q = np.zeros(S); tf = np.full(S, np.nan); sgn = np.full(S, np.nan); es = []
    Kt = K0.copy(); Esur = E0.copy(); t_prev = st[0]
    sur = {}; real = {}
    for s in range(1, S):
        e = E[js[s]] / nrm(E[js[s]]); es.append(e)
        D = SK[s] - SK[s - 1]; De = D @ e
        q[s] = ip(e, De); c = De - q[s] * e
        if q[s] > 0: tf[s] = nrm(c) / q[s]
        # Prop 53 sign condition for the principal direction of the increment
        w, V = np.linalg.eigh(D); u = V[:, -1] * np.sqrt(n)            # unit in <.,.>
        if ip(u, e) < 0: u = -u
        ls, Us = np.linalg.eigh(SK[s]); ls = np.clip(ls, 0, None); ce = Us.T @ e; cu = Us.T @ u
        cs = np.geomspace(1e-3, max(t[-1] - st[s], 1e-3), 60)
        g = np.array([np.sum(cu * np.exp(-ls * cc) * ce) for cc in cs]) / max(ip(u, e), 1e-300)
        sgn[s] = g.min()
        # surrogate: advance E~ from t_prev to st[s] under K~ (= K0 + lifts up to s-1), then add the lift q_s e e^T
        lw, LV = np.linalg.eigh(Kt); lw = np.clip(lw, 0, None)
        Esur = LV @ (np.exp(-lw * (st[s] - t_prev)) * (LV.T @ Esur)); t_prev = st[s]
        Kt = Kt + max(q[s], 0.) * np.outer(e, e) / n                    # e e^T in the <.,.> metric: (e e^T/n) v = e <e, v>
        for tr in T_REPORT:
            if abs(st[s] - tr) / tr < 0.35 and tr not in sur:
                sur[tr] = nrm(Esur) / nrm(lazy(st[s])); real[tr] = nrm(E[js[s]]) / nrm(lazy(st[s]))
    # final
    lw, LV = np.linalg.eigh(Kt); lw = np.clip(lw, 0, None)
    EsurT = LV @ (np.exp(-lw * (t[-1] - t_prev)) * (LV.T @ Esur))
    sur[3000.] = nrm(EsurT) / nrm(lazy(t[-1])); real[3000.] = nrm(E[-1]) / nrm(lazy(t[-1]))
    ok = np.isfinite(sgn[1:])
    return dict(tag=tag, H=d['H'], depth=d['depth'], Delta=d['Delta'], q_neg_frac=np.mean(q[1:] < 0), tf_med=np.nanmedian(tf),
                tf_q90=np.nanquantile(tf, .9), sign_ok_frac=np.mean(sgn[1:][ok] >= -1e-3), sign_min_med=np.nanmedian(sgn[1:]),
                **{f'sur_{int(k)}': v for k, v in sur.items()}, **{f'real_{int(k)}': v for k, v in real.items()})

if __name__ == '__main__':
    rows = []
    for f in sorted(glob.glob(f'{ROOT}/runs/{SWEEP}/late_*.npy')):
        try: r = run_one(f)
        except Exception as ex: print('FAIL', f, repr(ex), flush=True); continue
        rows.append(r)
        print(f"{r['tag']:28s} surrogate/lazy at T {r['sur_3000']:.3f} vs real {r['real_3000']:.3f} | transverse at creation med {r['tf_med']:.2f} | "
              f"sign cond ok {r['sign_ok_frac']:.2f} (min med {r['sign_min_med']:+.2f}) | q<0 stages {r['q_neg_frac']:.2f}", flush=True)
    with open(f'{OUT}/{SWEEP}_table.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    def q(k):
        v = np.array([r.get(k, np.nan) for r in rows], float); v = v[np.isfinite(v)]
        return f'{np.median(v):6.3f} [{np.quantile(v,.1):6.3f},{np.quantile(v,.9):6.3f}]' if len(v) else '   -   '
    print('\nmedian [10 %, 90 %] over runs:  ||E~(t)||/||E_lazy(t)|| (pure-targeted surrogate, Thm 52 says <= 1) vs real ||E(t)||/||E_lazy(t)||')
    for tr in T_REPORT: print(f'  t={tr:5.0f}: surrogate {q(f"sur_{int(tr)}")}   real {q(f"real_{int(tr)}")}')
    print('transverse fraction at creation (median over stages):', q('tf_med'), ' 90% over stages:', q('tf_q90'))
    print('Prop 53 sign condition: fraction of stages satisfied', q('sign_ok_frac'), ' median of the minimum overlap ratio', q('sign_min_med'))
    print('fraction of stages with q_s < 0 (increment negative on the residual):', q('q_neg_frac'))
    print('written:', OUT)
