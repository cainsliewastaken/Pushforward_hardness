"""Trust-region Newton in the exact global Gauss-Newton eigenbasis, with measured residual curvature (derived design).

Local model (exact to 2nd order):  L(th + d) ~ L + g.d + 1/2 d^T (G + H_r) d,  G = J^T J / n,  H_r = sum_i r_i f_i'' / n.
Basis: global eigenbasis of G (all inter-layer terms), refreshed when the gradient's out-of-span fraction exceeds
--refresh_tol or every --refresh_max steps. Curvature per mode: h_k = lam_k + hr_k, hr_k = r.f''[V_k,V_k]/n measured on
--n_probe log-spaced modes at each refresh and interpolated in log(lam); h_k clipped at 0 (negative-curvature directions
are left to the trust region). Step: d = -sum_k g_k/(h_k + mu) V_k - g_rest/(hr_med + mu). Trust region (Nocedal & Wright
4.1): rho = actual/predicted reduction; rho < 0.25 -> mu *= 4; rho > 0.75 -> mu /= 2; step rejected if rho < 0.1.
--hr 0 gives pure Levenberg-Marquardt (GN model only). --beta: optional heavy-ball on accepted steps. Full batch.
"""
import os, sys, json, math, time, argparse, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
from l63_design import make_data_stats
from l63test_torch import MLP
p = argparse.ArgumentParser()
p.add_argument('--n', type=int, default=20000); p.add_argument('--nval', type=int, default=5000)
p.add_argument('--H', type=int, default=96); p.add_argument('--depth', type=int, default=2); p.add_argument('--Delta', type=float, default=1.0)
p.add_argument('--steps', type=int, default=3000); p.add_argument('--seed', type=int, default=1); p.add_argument('--dataseed', type=int, default=1)
p.add_argument('--hr', type=int, default=1); p.add_argument('--n_probe', type=int, default=16)
p.add_argument('--refresh_tol', type=float, default=0.05); p.add_argument('--refresh_max', type=int, default=100)
p.add_argument('--beta', type=float, default=0.0); p.add_argument('--centred', type=int, default=0)
p.add_argument('--reltol', type=float, default=1e-14); p.add_argument('--data_cache', default=''); p.add_argument('--out', required=True)
p.add_argument('--log_every', type=int, default=10)
p.add_argument('--basis', default='exact', choices=['exact', 'lowrank'], help='exact global GN eigenbasis, or top-k modes by matrix-free subspace iteration (scalable)')
p.add_argument('--rank', type=int, default=32); p.add_argument('--sub_iters0', type=int, default=6)
a = p.parse_args()
torch.set_default_dtype(torch.float64); DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; t_wall0 = time.time()
if a.data_cache and os.path.exists(a.data_cache):
    zc = np.load(a.data_cache); Xr, Yr, Xvr, Yvr = zc['X'], zc['Y'], zc['Xv'], zc['Yv']
else:
    Xr, Yr = make_data_stats(a.n, a.Delta, a.dataseed); Xvr, Yvr = make_data_stats(a.nval, a.Delta, a.dataseed + 100)
    if a.data_cache: np.savez(a.data_cache, X=Xr, Y=Yr, Xv=Xvr, Yv=Yvr, Delta=a.Delta, dataseed=a.dataseed)
mu_x, sd_x = Xr.mean(0), Xr.std(0)
X = torch.tensor((Xr - mu_x) / sd_x, device=DEV); Y = torch.tensor((Yr - mu_x) / sd_x, device=DEV).reshape(-1)
Xv = torch.tensor((Xvr - mu_x) / sd_x, device=DEV); Yv = torch.tensor((Yvr - mu_x) / sd_x, device=DEV).reshape(-1)
net = MLP(3, a.H, a.depth, 3, seed=a.seed)
Ws = [w.to(DEV).clone() for w in net.W]; bs = [b.to(DEV).clone() for b in net.b]
shapes = [w.shape for w in Ws] + [b.shape for b in bs]; theta = torch.cat([q.reshape(-1) for q in Ws + bs]).clone()
P = theta.numel(); N = Y.numel(); FWD = 2 * a.n * P; L1 = a.depth + 1
def f_at(th, Xin):
    out = []; i = 0
    for s in shapes: k = int(np.prod(s)); out.append(th[i:i + k].reshape(s)); i += k
    h = Xin
    for l in range(a.depth): h = torch.tanh(h @ out[l].T + out[L1 + l])
    return (h @ out[a.depth].T + out[L1 + a.depth]).reshape(-1)
if a.centred:
    with torch.no_grad(): Y = Y + f_at(theta, X); Yv = Yv + f_at(theta, Xv)
def loss_grad(th):
    th = th.detach().requires_grad_(True); r = f_at(th, X) - Y; L = 0.5 * torch.dot(r, r) / a.n
    g, = torch.autograd.grad(L, th); return r.detach(), float(L), g
def loss_only(th):
    with torch.no_grad(): r = f_at(th, X) - Y; return 0.5 * float(torch.dot(r, r)) / a.n
def jacobian(th):
    f1 = lambda t, x: f_at(t, x[None, :])
    chunks = [torch.func.vmap(torch.func.jacrev(f1), in_dims=(None, 0))(th, X[i:i + 2000]).reshape(-1, P) for i in range(0, a.n, 2000)]
    return torch.cat(chunks, 0)
def Gv(th, Vm):                                  # G Vm = J^T J Vm / n, matrix-free (k jvps + k vjps)
    f = lambda t: f_at(t, X)
    with torch.enable_grad():
        _, vjp_fn = torch.func.vjp(f, th)
        JV = torch.func.vmap(lambda v: torch.func.jvp(f, (th,), (v,))[1], in_dims=1, out_dims=1)(Vm)
        return torch.func.vmap(lambda u: vjp_fn(u)[0], in_dims=1, out_dims=1)(JV) / a.n
def fpp(th, v):
    gfun = lambda t: torch.func.jvp(lambda u: f_at(u, X), (t,), (v,))[1]
    return torch.func.jvp(gfun, (th,), (v,))[1]
rows = []
def emit(metric, value, step=None):
    rows.append(dict(opt='trgn', args=' '.join(sys.argv[1:]), n=a.n, H=a.H, Delta=a.Delta, hr=a.hr, beta=a.beta, centred=a.centred, step=step, metric=metric, value=value))
mu = None; V = lam = hk = None; hr_med = 0.0; flop = 0.0; n_ref = 0; last_ref = -10 ** 9; vel = torch.zeros_like(theta); n_rej = 0
r, L, g = loss_grad(theta); flop += 3
for step in range(1, a.steps + 1):
    need = V is None or step - last_ref >= a.refresh_max
    if V is not None and not need:
        if a.basis == 'exact':
            gk = V.T @ g; frac = float((g - V @ gk).norm() / g.norm().clamp(min=1e-300)); need = frac > a.refresh_tol
    if need:
        if a.basis == 'exact':
            with torch.enable_grad(): J = jacobian(theta)
            G = J.T @ J / a.n; del J
            lam, V = torch.linalg.eigh(G); lam, V = lam.flip(0), V.flip(1); keep = lam > a.reltol * lam[0]; lam, V = lam[keep], V[:, keep]; del G
            flop += 9 + (2 * N * P * P + 10 * P ** 3) / FWD
        else:
            if V is None:
                gen = torch.Generator(device=DEV); gen.manual_seed(1234 + a.seed)
                Vm = torch.linalg.qr(torch.randn(P, a.rank, device=DEV, generator=gen))[0]; iters = a.sub_iters0
            else:
                Vm = V; iters = 1
            for _ in range(iters): Vm = torch.linalg.qr(Gv(theta, Vm))[0]
            Z = Gv(theta, Vm); ev, Wr = torch.linalg.eigh(Vm.T @ Z); V = Vm @ Wr.flip(1); lam = ev.flip(0).clamp(min=0.0)
            flop += 5 * a.rank * (iters + 1) + 4 * P * a.rank ** 2 / FWD
        hr_vals = torch.zeros_like(lam)
        if a.hr:
            ks = np.unique(np.geomspace(1, lam.numel(), a.n_probe if a.basis == 'exact' else 6).astype(int)) - 1; hs = []
            for k in ks: hs.append(float(r @ fpp(theta, V[:, k])) / a.n)
            flop += 4 * len(ks); hs = np.array(hs); ll = np.log(lam[ks].clamp(min=1e-300).cpu().numpy())
            hr_vals = torch.tensor(np.interp(np.log(lam.clamp(min=1e-300).cpu().numpy())[::-1], ll[::-1], hs[::-1])[::-1].copy(), device=DEV)
            if a.basis == 'exact':
                hr_med = float(np.median(hs))
            else:                                       # deep bulk: random unit probes orthogonal to the top subspace
                gen2 = torch.Generator(device=DEV); gen2.manual_seed(99 + step); hd = []
                for _ in range(4):
                    v = torch.randn(P, device=DEV, generator=gen2); v = v - V @ (V.T @ v); v = v / v.norm(); hd.append(float(r @ fpp(theta, v)) / a.n)
                flop += 16; hr_med = float(np.median(hd))
            if n_ref % 5 == 0: emit('hr_median', hr_med, step); emit('hr_neg_frac', float((hs < 0).mean()), step)
        hk = (lam + hr_vals).clamp(min=0.0)
        if mu is None: mu = float(lam[0])
        n_ref += 1; last_ref = step
        if n_ref % 5 == 1: emit('lam1', float(lam[0]), step); emit('k_modes', int(lam.numel()), step)
    gk = V.T @ g; g_rest = g - V @ gk
    while True:
        ak = -gk / (hk + mu); d = V @ ak - g_rest / (max(hr_med, 0.0) + mu)
        if a.beta > 0: d = d + a.beta * vel
        dV = V.T @ d; d_out = d - V @ dV
        pred = -(float(g @ d) + 0.5 * float((hk * dV ** 2).sum()) + 0.5 * max(hr_med, 0.0) * float(d_out @ d_out))   # model reduction
        Lnew = loss_only(theta + d); flop += 1 + 4 * P * lam.numel() / FWD
        rho = (L - Lnew) / pred if (pred > 0 and math.isfinite(Lnew)) else -1.0
        if rho < 0.25: mu *= 4.0
        elif rho > 0.75: mu = max(mu / 2.0, 1e-12)
        if math.isfinite(Lnew) and rho >= 0.1: break
        n_rej += 1
        if a.beta > 0: vel.zero_()
        if mu > 1e12: break
    theta = theta + d; vel = d
    r, L, g = loss_grad(theta); flop += 3
    if step % a.log_every == 0 or step == a.steps:
        emit('E_train', math.sqrt(2 * L), step); emit('flop_eq', flop, step); emit('mu', mu, step); emit('rho', rho, step)
with torch.no_grad():
    Ef = float(torch.sqrt(torch.dot(f_at(theta, X) - Y, f_at(theta, X) - Y) / a.n)); rv = f_at(theta, Xv) - Yv; Evf = float(torch.sqrt(torch.dot(rv, rv) / a.nval))
for k_, v_ in (('E_train_final', Ef), ('E_val_final', Evf), ('flop_eq_final', flop), ('n_refresh', n_ref), ('n_reject', n_rej), ('wall_s', time.time() - t_wall0), ('P', P)):
    emit(k_, v_, a.steps)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, 'a') as fh:
    for rw in rows: fh.write(json.dumps(rw) + '\n')
print(f'trgn basis={a.basis} hr={a.hr} beta={a.beta} centred={a.centred} H={a.H} n={a.n} Delta={a.Delta}: E_train={Ef:.4g} E_val={Evf:.4g} flop_eq={flop:.4g} '
      f'refreshes={n_ref} rejects={n_rej} mu_end={mu:.3g} wall={time.time() - t_wall0:.0f}s', flush=True)
