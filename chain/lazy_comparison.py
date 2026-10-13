"""Theorem 51 (THEORY.md section 22): the lazy comparison through the lift's action on the residual, on saved snapshots.

E' = -K(t) E, Delta(t) = K(t) - K_0. Split Delta E = a E + w with a = <E, Delta E>/||E||^2 (lift along the residual) and
<w, E> = 0 (transverse action). Exactly, with A(t) = int_0^t a:
    E(T) = e^{-A(T)} [ E_lazy(T) - int_0^T e^{-K_0 (T-u)} e^{A(u)} w(u) du ],   E_lazy(T) = e^{-K_0 T} E_0.
Hence ||E(T)|| <= e^{-A(T)} ||E_lazy(T)|| + e^{-A(T)} int_0^T e^{A(u)} ||w(u)|| du, and feature learning beats the lazy
bound at T whenever  int_0^T e^{A(u)} ||w(u)|| du  <=  (e^{A(T)} - 1) ||E_lazy(T)||.
Per run (runs/sweepA, n = 200; K(t) at 30 snapshots, E(t) at 400 log times): a(u), tau(u) = ||w||/(a ||E||) at the
snapshots, A(T) by log-interpolation, the exact-identity residual (the bracket evaluated with the snapshot integral vs
the measured E(T)), the bound (b), the sufficient condition (c), and the measured ratio ||E(T)|| / ||E_lazy(T)||.
Output: $L63_ROOT/runs/lazy_comparison/<sweep>_table.csv and a printed summary.
"""
import os, sys, glob, csv, numpy as np
ROOT = os.environ.get('L63_ROOT', os.path.expandvars('$SCRATCH/l63_held_rough'))
SWEEP = sys.argv[1] if len(sys.argv) > 1 else 'sweepA'
OUT = f'{ROOT}/runs/lazy_comparison'; os.makedirs(OUT, exist_ok=True)
T_REPORT = [10., 30., 100., 300., 1000., 3000.]

def run_one(f):
    d = np.load(f, allow_pickle=True).item(); n = d['n']; tag = d['tag']
    t = d['t']; E = d['Evec']; st = d['snap_t']; SK = d['snap_K'].astype(np.float64)
    ip = lambda u, v: u @ v / n; nrm = lambda u: np.sqrt(ip(u, u))
    j0 = int(np.argmin(np.abs(t - st[0]))); E0 = E[j0]; K0 = SK[0]
    lam, U = np.linalg.eigh(K0); lam = np.clip(lam, 0, None)
    c0 = U.T @ E0
    def lazy(tt): return U @ (np.exp(-lam * tt) * c0)
    # a, w at the snapshots
    js = [int(np.argmin(np.abs(t - s))) for s in st]
    a = np.zeros(len(st)); wn = np.zeros(len(st)); En = np.zeros(len(st)); tau = np.full(len(st), np.nan)
    for s, K in enumerate(SK):
        e = E[js[s]]; De = (K - K0) @ e; En[s] = nrm(e)
        a[s] = ip(e, De) / En[s] ** 2; w = De - a[s] * e; wn[s] = nrm(w)
        if a[s] > 0: tau[s] = wn[s] / (a[s] * En[s])
    # A(t) on the log grid: interpolate a (linear in log t, a may be negative early)
    lt = np.log(np.maximum(t, 1e-12)); lst = np.log(np.maximum(st, 1e-12))
    a_t = np.interp(lt, lst, a); w_t = np.exp(np.interp(lt, lst, np.log(np.maximum(wn, 1e-300))))
    dt = np.diff(t); A = np.concatenate([[0.], np.cumsum(0.5 * (a_t[1:] + a_t[:-1]) * dt)])
    rows = []
    for tr in T_REPORT:
        j = int(np.argmin(np.abs(t - tr))); Tj = t[j]
        EL = nrm(lazy(Tj)); Ej = nrm(E[j])
        # int_0^T e^{A(u)} ||w(u)|| du (triangle bound) and the exact vector integral int e^{-K0 (T-u)} e^{A(u)} w(u) du at snapshots
        g = np.exp(A[:j + 1]) * w_t[:j + 1]
        I_tri = np.sum(0.5 * (g[1:] + g[:-1]) * dt[:j]) if j > 0 else 0.
        # exact identity check: vector integral with the snapshot w (piecewise, log-interpolated in time is not available for vectors;
        # use the snapshots up to T as quadrature nodes in log t)
        sel = [s for s in range(len(st)) if st[s] <= Tj + 1e-9]
        vec = np.zeros(E.shape[1])
        for k in range(len(sel) - 1):
            s1, s2 = sel[k], sel[k + 1]; u1, u2 = st[s1], st[s2]
            for s_, u_ in ((s1, u1), (s2, u2)):
                e = E[js[s_]]; De = (SK[s_] - K0) @ e; w = De - a[s_] * e
                Au = np.interp(np.log(max(u_, 1e-12)), lt, A)
                prop = U @ (np.exp(-lam * (Tj - u_)) * (U.T @ w))
                vec += 0.5 * (u2 - u1) * np.exp(Au) * prop
        E_pred = np.exp(-A[j]) * (lazy(Tj) - vec)
        # propagated transverse term: split w at the K0 cutoff lam_c = 10/T; the fast part is damped by e^{-lam_c (T-u)}
        lam_c = 10. / Tj; slow = lam <= lam_c
        I_slow = 0.; I_fast = 0.
        for k in range(len(sel) - 1):
            s1, s2 = sel[k], sel[k + 1]; u1, u2 = st[s1], st[s2]
            for s_, u_ in ((s1, u1), (s2, u2)):
                e = E[js[s_]]; De = (SK[s_] - K0) @ e; w = De - a[s_] * e; cw = U.T @ w / np.sqrt(n)
                Au = np.interp(np.log(max(u_, 1e-12)), lt, A)
                I_slow += 0.5 * (u2 - u1) * np.exp(Au) * np.sqrt(np.sum(cw[slow] ** 2))
                I_fast += 0.5 * (u2 - u1) * np.exp(Au) * np.exp(-lam_c * (Tj - u_)) * np.sqrt(np.sum(cw[~slow] ** 2))
        cond_prop = (I_slow + I_fast) / max((np.exp(A[j]) - 1) * EL, 1e-300)
        bound_prop = np.exp(-A[j]) * (EL + I_slow + I_fast)
        rows.append(dict(tag=tag, H=d['H'], depth=d['depth'], Delta=d['Delta'], t=Tj, E=Ej, E_lazy=EL, ratio=Ej / EL,
                         A=A[j], gain=np.exp(-A[j]), bound_b=np.exp(-A[j]) * (EL + I_tri), cond_c=I_tri / max((np.exp(A[j]) - 1) * EL, 1e-300),
                         bound_prop=bound_prop, cond_prop=cond_prop, w_slow_frac=I_slow / max(I_slow + I_fast, 1e-300),
                         identity_err=nrm(E_pred - E[j]) / Ej,
                         a_now=np.interp(np.log(max(Tj, 1e-12)), lst, a), tau_now=np.interp(np.log(max(Tj, 1e-12)), lst, np.nan_to_num(tau, nan=np.nan))))
    return rows, dict(tag=tag, a_min=a.min(), t_a_pos=st[np.argmax(a > 0)] if np.any(a > 0) else np.inf,
                      tau_med=np.nanmedian(tau), tau_max=np.nanmax(tau) if np.any(np.isfinite(tau)) else np.nan)

if __name__ == '__main__':
    allrows, metas = [], []
    for f in sorted(glob.glob(f'{ROOT}/runs/{SWEEP}/late_*.npy')):
        try: rows, m = run_one(f)
        except Exception as ex: print('FAIL', f, repr(ex), flush=True); continue
        allrows += rows; metas.append(m)
        print(f"{m['tag']:28s} a<0 until t={m['t_a_pos']:.3g}  a_min={m['a_min']:+.3e}  tau median {m['tau_med']:.2f} max {m['tau_max']:.2f}", flush=True)
    with open(f'{OUT}/{SWEEP}_table.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    print('\nmedian [10 %, 90 %] over runs')
    print('   t    ||E||/||E_lazy||   gain e^-A     bound(b)/||E||   cond(c)        bound_prop/||E||   cond_prop (<=1 beats lazy)   w slow frac   identity err   a(t)        tau(t)')
    for tr in T_REPORT:
        sel = [r for r in allrows if abs(r['t'] - tr) / tr < 0.5]
        def q(k, div=None):
            v = np.array([r[k] / (r[div] if div else 1.) for r in sel], float); v = v[np.isfinite(v)]
            return f'{np.median(v):8.3g} [{np.quantile(v,.1):7.3g},{np.quantile(v,.9):7.3g}]' if len(v) else '     -     '
        print(f'{tr:6.0f} ' + ' '.join([q('ratio'), q('gain'), q('bound_b', 'E'), q('cond_c'), q('bound_prop', 'E'), q('cond_prop'), q('w_slow_frac'), q('identity_err'), q('a_now'), q('tau_now')]))
    nb = sum(1 for r in allrows if abs(r['t'] - 3000) / 3000 < 0.5 and r['ratio'] <= 1); nt = sum(1 for r in allrows if abs(r['t'] - 3000) / 3000 < 0.5)
    print(f'runs with ||E(T)|| <= ||E_lazy(T)||: {nb}/{nt};  with cond_prop <= 1 at T: {sum(1 for r in allrows if abs(r["t"]-3000)/3000 < 0.5 and r["cond_prop"] <= 1)}/{nt}')
    print('written:', OUT)
