"""C1 in the Krylov form (EXPERIMENTS.md C1, 2026-10-12): the floors of the roughness theory evaluated with the best
roughness function, the Krylov functional, on the saved snapshots of the ground-truth GD runs (runs/sweepA, n = 200).

Conventions of the main text: <u, v> = u.v / n on stacked sampled functions, ||u||^2 = u.u / n, K = Psi Psi^T / n,
eigenpairs (lam_k, v_k) with v_k orthonormal in <.,.>, e_k = <v_k, E>, ||E||_K^2 = sum lam_k e_k^2 = |g_E|^2 = |theta_dot|^2.
Slow part E_gamma = gamma (K + gamma)^-1 E, Upsilon(gamma) = sum lam e^2 / (lam + gamma)^2, Krylov functional phi_gamma = E_gamma / ||E_gamma||.

Per run, along the logged times t_j (400 points) with K(t) at 30 geometric snapshot times:
  ell(t)        path length  int |theta_dot| ds  (gnorm = |g_E|)
  F0(ell)       the floor at time 0 (Theorem time0 / stepsup): max_{||phi||<=1} <phi, E0> - ell ||phi||_{K0}, ridge closed form
  floor_t0      F0(ell(t)) - 1/2 H2bar(ell) ell^2,  H2bar = running max of the logged H2 along the gradient (a proxy for the ball max)
  floor_path    best over a gamma grid of the path floor with the FIXED functional phi = phi_gamma(K0, E0):
                <phi, E0> - cost_t(phi),  cost_t(phi) = int ||phi||_{K(s)} |theta_dot(s)| ds   (Theorem path; no leak)
  floor_energy  the energy-form floor of the held-rough draft for the same phi (Prop rough-energy (b) with eps = 1):
                [<phi,E0>^2 exp(-3 int sigma^2) - int lam1 ||E_s||^2 exp(-3 int_s^t sigma^2) ds]_+^(1/2),  sigma = ||phi||_{K(s)}
  slow_now      the one-time floor at the current state, ||E_gamma(t)|| with gamma = 1/t, as a fraction of ||E(t)||
Checks: the exact inequality <phi, E(t)> >= <phi, E0> - cost_t(phi) (tests the snapshot interpolation of the cost);
        sum lam e^2 = gnorm^2 at t = 0.
Output: one CSV per run and a summary table, in $L63_ROOT/runs/krylov_floor/.
"""
import os, sys, glob, numpy as np

ROOT = os.environ.get('L63_ROOT', os.path.expandvars('$SCRATCH/l63_held_rough'))
OUT = f'{ROOT}/runs/krylov_floor'; os.makedirs(OUT, exist_ok=True)
SWEEP = sys.argv[1] if len(sys.argv) > 1 else 'sweepA'
T_REPORT = [10., 30., 100., 300., 1000., 3000.]
GAMMAS = np.geomspace(1e-5, 1e2, 36)

def ridge_floor(lam, e, ell, tol=1e-10):
    """F0(ell) = min_{|g|<=ell} ||E0 + Psi g|| in closed form (Theorem stepsup (c))."""
    pos = lam > tol * lam.max()
    ups0 = np.sum(e[pos] ** 2 / lam[pos])              # Upsilon(0+)
    null = np.sqrt(np.sum(e[~pos] ** 2))
    if ell <= 0: return np.sqrt(np.sum(e ** 2))
    if ell ** 2 >= ups0: return null
    # solve Upsilon(gamma) = ell^2, Upsilon decreasing in gamma
    lo, hi = 1e-16, 1e16
    for _ in range(200):
        g = np.sqrt(lo * hi)
        u = np.sum(lam[pos] * e[pos] ** 2 / (lam[pos] + g) ** 2)
        if u > ell ** 2: lo = g
        else: hi = g
    g = np.sqrt(lo * hi)
    return np.sqrt(np.sum(g ** 2 * e[pos] ** 2 / (lam[pos] + g) ** 2) + null ** 2)

def run_one(f):
    d = np.load(f, allow_pickle=True).item(); n = d['n']; tag = d['tag']
    t = d['t']; E = d['Evec']; gn = d['gnorm']; H2 = d['H2_g']; st = d['snap_t']; SK = d['snap_K'].astype(np.float64)
    N = E.shape[1]
    ip = lambda u, v: u @ v / n
    nrm = lambda u: np.sqrt(ip(u, u))
    # time 0
    j0 = int(np.argmin(np.abs(t - st[0]))); E0 = E[j0]; K0 = SK[0]
    lam, U = np.linalg.eigh(K0); lam = np.clip(lam, 0, None)[::-1]; U = U[:, ::-1]      # U Euclidean-orthonormal
    V = U * np.sqrt(n)                                                                   # orthonormal in <.,.>
    e0 = V.T @ E0 / n
    chk_g = np.sqrt(np.sum(lam * e0 ** 2)) / gn[j0]
    # path length
    ell = np.concatenate([[0.], np.cumsum(0.5 * (gn[1:] + gn[:-1]) * np.diff(t))])
    H2bar = np.maximum.accumulate(np.nan_to_num(H2, nan=0.0))
    # time-0 floor
    F0 = np.array([ridge_floor(lam, e0, l) for l in ell])
    floor_t0 = F0 - 0.5 * H2bar * ell ** 2
    # fixed Krylov functionals of (K0, E0) on a gamma grid; their kernel size along the snapshots
    phis = []
    for g in GAMMAS:
        w = g / (lam + g); Eg = V @ (w * e0); phis.append(Eg / nrm(Eg))
    phis = np.array(phis)                                                                 # (G, N)
    sig_snap = np.array([[np.sqrt(max(ip(p, K @ p), 0.)) for p in phis] for K in SK])    # (S, G) ||phi||_{K(s)}
    # interpolate log sigma in log t onto the log grid (clamped at the ends)
    lt, lst = np.log(np.maximum(t, 1e-12)), np.log(np.maximum(st, 1e-12))
    sig = np.exp(np.array([np.interp(lt, lst, np.log(np.maximum(sig_snap[:, k], 1e-300))) for k in range(len(GAMMAS))])).T  # (T, G)
    cost = np.concatenate([np.zeros((1, len(GAMMAS))), np.cumsum(0.5 * (sig[1:] + sig[:-1]) * (gn[1:, None] + gn[:-1, None]) / 2 * np.diff(t)[:, None], axis=0)])
    phiE0 = phis @ E0 / n                                                                # (G,)
    phiE = E @ phis.T / n                                                                 # (T, G)
    floor_path_g = phiE0[None, :] - cost                                                  # (T, G)
    viol = np.max((floor_path_g - phiE) / np.maximum(np.abs(phiE0)[None, :], 1e-12))     # should be <= ~0 up to interpolation
    floor_path = floor_path_g.max(1)
    # energy-form floor (held-rough draft, Prop rough-energy (b), eps = 1) for the same functionals
    lam1 = np.array([np.linalg.eigvalsh(K)[-1] for K in SK]); lam1_t = np.exp(np.interp(lt, lst, np.log(lam1)))
    En2 = np.array([ip(x, x) for x in E])
    floor_energy_g = np.zeros_like(floor_path_g); leak_ratio_g = np.zeros_like(floor_path_g)
    dt = np.diff(t)
    for k in range(len(GAMMAS)):
        s2 = sig[:, k] ** 2
        I = np.concatenate([[0.], np.cumsum(0.5 * (s2[1:] + s2[:-1]) * dt)])
        Es2 = En2 - phiE[:, k] ** 2
        f = lam1_t * Es2
        for j in range(len(t)):                       # J(t_j) = int_0^{t_j} lam1 ||E_s||^2 exp(-3 (I(t_j) - I(s))) ds, no overflow
            wgt = np.exp(-3 * (I[j] - I[:j + 1])); fi = f[:j + 1] * wgt
            J = np.sum(0.5 * (fi[1:] + fi[:-1]) * dt[:j]) if j > 0 else 0.
            main = phiE0[k] ** 2 * np.exp(-3 * I[j])
            floor_energy_g[j, k] = np.sqrt(max(main - J, 0.)); leak_ratio_g[j, k] = J / max(main, 1e-300)
    kbest_e = floor_energy_g.argmax(1); floor_energy = floor_energy_g.max(1)
    leak_ratio = leak_ratio_g[np.arange(len(t)), kbest_e]
    # cost split for the best path functional: first-order part ell ||phi||_{K0} vs the actual cost (the excess is the lift)
    kbest = floor_path_g.argmax(1); sig0 = sig_snap[0]
    cost_first = ell * sig0[kbest]; cost_best = cost[np.arange(len(t)), kbest]
    lift_cost = cost_best / np.maximum(cost_first, 1e-300)
    # one-time floor at the current state: ||E_gamma(t)|| with gamma = 1/t, at snapshot times (held on the log grid by nearest snapshot)
    slow_snap = np.full(len(st), np.nan)
    for s, K in enumerate(SK):
        j = int(np.argmin(np.abs(t - st[s])))
        if t[j] <= 0: continue
        ls, Us = np.linalg.eigh(K); ls = np.clip(ls, 0, None); es = (Us * np.sqrt(n)).T @ E[j] / n
        g = 1.0 / t[j]; slow_snap[s] = np.sqrt(np.sum((g / (ls + g)) ** 2 * es ** 2)) / nrm(E[j])
    slow_now = np.array([slow_snap[int(np.argmin(np.abs(st - tj)))] for tj in t])
    Enorm = np.sqrt(En2)
    # report rows
    rows = []
    for tr in T_REPORT:
        j = int(np.argmin(np.abs(t - tr)))
        rows.append(dict(tag=tag, H=d['H'], depth=d['depth'], Delta=d['Delta'], t=t[j], ell=ell[j], E=Enorm[j],
                         F0=F0[j], floor_t0=floor_t0[j], floor_path=floor_path[j], floor_energy=floor_energy[j],
                         slow_now=slow_now[j] if np.isfinite(slow_now[j]) else np.nan, H2bar=H2bar[j],
                         lift_cost=lift_cost[j], leak_ratio=leak_ratio[j], gamma_best=GAMMAS[kbest[j]]))
    # where the time-0 floor becomes uninformative: first t with floor_t0 < 0.5 F0
    k_un = np.argmax(floor_t0 < 0.5 * F0) if np.any(floor_t0 < 0.5 * F0) else len(t) - 1
    k_f0 = np.argmax(F0 < 0.5 * Enorm[j0]) if np.any(F0 < 0.5 * Enorm[j0]) else len(t) - 1
    meta = dict(tag=tag, chk_gnorm=chk_g, viol_path=viol, t_H2_uninformative=t[k_un], ell_H2=ell[k_un], t_F0_half=t[k_f0],
                ell_F0_half=ell[k_f0], ell_end=ell[-1], ell_bound_end=np.sqrt(t[-1] / 2) * Enorm[j0], E0=Enorm[j0], E_end=Enorm[-1])
    np.save(f'{OUT}/{SWEEP}_{tag}.npy', dict(t=t, ell=ell, E=Enorm, F0=F0, floor_t0=floor_t0, floor_path=floor_path, floor_energy=floor_energy,
                                             slow_now=slow_now, gammas=GAMMAS, floor_path_g=floor_path_g, meta=meta), allow_pickle=True)
    return rows, meta

if __name__ == '__main__':
    files = sorted(glob.glob(f'{ROOT}/runs/{SWEEP}/late_*.npy'))
    allrows, metas = [], []
    for f in files:
        try:
            rows, meta = run_one(f)
        except Exception as ex:
            print('FAIL', f, repr(ex), flush=True); continue
        allrows += rows; metas.append(meta)
        m = meta
        print(f"{m['tag']:28s} chk_g={m['chk_gnorm']:.4f} viol={m['viol_path']:+.3f} E0={m['E0']:.3f} E(T)={m['E_end']:.4f} "
              f"ell(T)={m['ell_end']:.3f} (bound {m['ell_bound_end']:.1f})  F0 half at t={m['t_F0_half']:.3g} (ell {m['ell_F0_half']:.3f});"
              f"  H2 term halves the floor at t={m['t_H2_uninformative']:.3g} (ell {m['ell_H2']:.3f})", flush=True)
    import csv
    with open(f'{OUT}/{SWEEP}_table.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    with open(f'{OUT}/{SWEEP}_meta.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(metas[0].keys())); w.writeheader(); w.writerows(metas)
    # summary: ratios floor / ||E|| at the report times, medians and 10-90 % over runs
    print('\nratio of each floor to the measured ||E(t)||, median [10 %, 90 %] over runs')
    print('   t     F0(ell)/E        floor_t0/E       floor_path/E     floor_energy/E   slow_now=||E_{1/t}||/||E||  lift_cost=cost/(ell sig0)  leak/main (energy form)')
    for tr in T_REPORT:
        sel = [r for r in allrows if abs(r['t'] - tr) / tr < 0.5]
        if not sel: continue
        def q(key):
            v = np.array([r[key] / r['E'] if key in ('F0', 'floor_t0', 'floor_path', 'floor_energy') else r[key] for r in sel], float); v = v[np.isfinite(v)]
            return f'{np.median(v):8.3g} [{np.quantile(v, .1):7.3g},{np.quantile(v, .9):7.3g}]' if len(v) else '      -      '
        print(f'{tr:6.0f} ' + ' '.join(q(k) for k in ['F0', 'floor_t0', 'floor_path', 'floor_energy', 'slow_now', 'lift_cost', 'leak_ratio']))
    print('\nwritten:', OUT)
