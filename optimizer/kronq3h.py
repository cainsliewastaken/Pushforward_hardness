"""KronQ3h = KronQ2 + per-mode push weights learned online by hypergradient with eligibility traces (IDBD /
Autostep style, per Kronecker eigenmode; --hyper 1). Mode k of layer l has log-weight u_k (push w_k = exp(u_k), start 0).
dL_t/du_k = -lr eta (Gr_t)_k z_k with trace z_k <- gamma z_k + D_{t-1,k} (gamma = 1 - 1/--horizon; horizon 1 = one-step
hypergradient). Normalised update u_k <- u_k - beta_u hg_k / sqrt(EMA(hg_k^2)), u in [-4.6, 4.6]. u, z and the EMA are
carried into a new basis by squared overlaps of the factor eigenvectors (as the second moment). No extra passes.

KronQ2: scalable sign push in a per-layer Kronecker Gauss-Newton eigenbasis, with the memory and cost reductions:
  --block b      block-diagonal factors per side (blocks of b units; memory 2 b^2 per block per side instead of 2 dim^2)
  --max_dim m    a side wider than m is treated diagonally (no rotation; eigenvalues = factor diagonal), as SOAP one-sided
  --lr_kappa k   per-layer step scale eta_l set online to hold the layer's function-space Taylor remainder
                 rf_l = ||f(th + d_l) - f(th) - J_l d_l|| / ||J_l d_l|| at k (round-robin check, one layer per check)
  --adapt_refresh refresh interval R set by error control: at each refresh e = |g.D_new - g.D_old| / |g.D_new| (relative
                 change of the first-order decrease from refreshing factors/bases); R <- R * clip(k/e, 1/2, 2)
Step (rotated coordinates, per layer): D_ij = (lam_ij / lam_max)^(q/2) sign(C_ij), capped at the Gauss-Newton step
|Gr_ij| / (lam_ij lr eta_l L); null modes (lam_ij < reltol lam_max) get 0. W -= lr eta_l U_S D U_A^T.
"""
import os, sys, json, math, time, argparse, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
from l63_design import make_data_stats
from l63test_torch import MLP
p = argparse.ArgumentParser()
p.add_argument('--n', type=int, default=20000); p.add_argument('--nval', type=int, default=5000)
p.add_argument('--H', type=int, default=96); p.add_argument('--depth', type=int, default=2); p.add_argument('--Delta', type=float, default=1.0)
p.add_argument('--steps', type=int, default=3000); p.add_argument('--seed', type=int, default=1); p.add_argument('--dataseed', type=int, default=1)
p.add_argument('--lr', type=float, default=3e-3); p.add_argument('--b1', type=float, default=0.9); p.add_argument('--decay', type=int, default=1)
p.add_argument('--q', type=float, default=0.0); p.add_argument('--cap', type=int, default=1)
p.add_argument('--block', type=int, default=0); p.add_argument('--max_dim', type=int, default=10 ** 9)
p.add_argument('--lr_kappa', type=float, default=0.0); p.add_argument('--check_every', type=int, default=10)
p.add_argument('--refresh', type=int, default=10, help='initial / fixed refresh interval (stats + eigenbases)')
p.add_argument('--adapt_refresh', type=int, default=0); p.add_argument('--refresh_kappa', type=float, default=0.1)
p.add_argument('--stats_n', type=int, default=2000); p.add_argument('--beta_stats', type=float, default=0.95); p.add_argument('--reltol', type=float, default=1e-12)
p.add_argument('--hyper_shape', type=int, default=0); p.add_argument('--hyper', type=int, default=0); p.add_argument('--horizon', type=float, default=100.0); p.add_argument('--beta_u', type=float, default=0.01)
p.add_argument('--centred', type=int, default=0); p.add_argument('--batch', type=int, default=0)
p.add_argument('--data_cache', default=''); p.add_argument('--out', required=True); p.add_argument('--log_every', type=int, default=10)
a = p.parse_args()
torch.set_default_dtype(torch.float64); DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; t0w = time.time()
if a.data_cache and os.path.exists(a.data_cache):
    zc = np.load(a.data_cache); Xr, Yr, Xvr, Yvr = zc['X'], zc['Y'], zc['Xv'], zc['Yv']
else:
    Xr, Yr = make_data_stats(a.n, a.Delta, a.dataseed); Xvr, Yvr = make_data_stats(a.nval, a.Delta, a.dataseed + 100)
mx, sx = Xr.mean(0), Xr.std(0)
X = torch.tensor((Xr - mx) / sx, device=DEV); Y = torch.tensor((Yr - mx) / sx, device=DEV)
Xv = torch.tensor((Xvr - mx) / sx, device=DEV); Yv = torch.tensor((Yvr - mx) / sx, device=DEV)
net = MLP(3, a.H, a.depth, 3, seed=a.seed); L1 = a.depth + 1
Wt = [torch.cat([w.to(DEV), b.to(DEV)[:, None]], 1).clone() for w, b in zip(net.W, net.b)]
P = sum(w.numel() for w in Wt); FWD = 2 * a.n * P
def forward(Ws, Xin):
    h = Xin
    for l in range(L1):
        z = torch.cat([h, torch.ones(h.shape[0], 1, device=DEV)], 1) @ Ws[l].T; h = torch.tanh(z) if l < L1 - 1 else z
    return h
if a.centred:
    with torch.no_grad(): Y = Y + forward(Wt, X); Yv = Yv + forward(Wt, Xv)
def grads(Ws, Xb, Yb):
    Wr = [w.detach().requires_grad_(True) for w in Ws]; r = forward(Wr, Xb) - Yb
    L = 0.5 * (r * r).sum() / Xb.shape[0]; gs = torch.autograd.grad(L, Wr); return [g.detach() for g in gs]
def blocks_of(dim):
    if dim > a.max_dim: return None                                   # diagonal side
    b = a.block if 0 < a.block < dim else dim
    return [torch.arange(i, min(i + b, dim), device=DEV) for i in range(0, dim, b)]
class Side:
    def __init__(self, dim):
        self.dim = dim; self.bl = blocks_of(dim); self.F = None; self.U = None; self.e = None
    def update(self, Z, beta):                                        # Z: samples x dim (activations or signals)
        m = Z.shape[0]
        new = (Z * Z).mean(0) if self.bl is None else [Z[:, ix].T @ Z[:, ix] / m for ix in self.bl]
        if self.F is None: self.F = new
        elif self.bl is None: self.F.mul_(beta).add_(new, alpha=1 - beta)
        else:
            for F_, N_ in zip(self.F, new): F_.mul_(beta).add_(N_, alpha=1 - beta)
    def eig(self):
        if self.bl is None: return None, self.F.clamp(min=0).clone()
        U = []; e = torch.zeros(self.dim, device=DEV)
        for ix, F_ in zip(self.bl, self.F):
            ev, V = torch.linalg.eigh(F_); U.append(V); e[ix] = ev.clamp(min=0)
        return U, e
    def mem(self):
        return self.dim if self.bl is None else 2 * sum(len(ix) ** 2 for ix in self.bl)
def rot(M, U, bl, side, inv):                                         # side 'L' acts on rows (out), 'R' on columns (in)
    if U is None: return M
    out = M.clone()
    for ix, V in zip(bl, U):
        if side == 'L': out[ix, :] = (V @ M[ix, :]) if inv else (V.T @ M[ix, :])
        else: out[:, ix] = (M[:, ix] @ V.T) if inv else (M[:, ix] @ V)
    return out
st = [dict(S=Side(w.shape[0]), A=Side(w.shape[1]), US=None, eS=None, UA=None, eA=None, eta=1.0, rf=None,
           u=torch.zeros_like(w), z=torch.zeros_like(w), hv=torch.zeros_like(w), Dprev=None) for w in Wt]
def factor_stats(Ws, Xs):
    Wr = [w.detach().requires_grad_(True) for w in Ws]; h = Xs; zs = []; acts = []
    for l in range(L1):
        at = torch.cat([h, torch.ones(h.shape[0], 1, device=DEV)], 1); acts.append(at.detach())
        z = at @ Wr[l].T; zs.append(z); h = torch.tanh(z) if l < L1 - 1 else z
    sig = [[] for _ in range(L1)]
    for c in range(h.shape[1]):
        gz = torch.autograd.grad(h[:, c].sum(), zs, retain_graph=True)
        for l in range(L1): sig[l].append(gz[l])
    return acts, [torch.cat(s, 0) * math.sqrt(1.0) for s in sig]      # S = E[sum_c d_c d_c^T]: stack the n_y signal sets
def step_dir(l, m, g, lr_l):
    s_ = st[l]; C = rot(rot(m, s_['US'], s_['S'].bl, 'L', False), s_['UA'], s_['A'].bl, 'R', False)
    Gr = rot(rot(g, s_['US'], s_['S'].bl, 'L', False), s_['UA'], s_['A'].bl, 'R', False)
    lam = s_['eS'][:, None] * s_['eA'][None, :] * (s_['nS'] if 'nS' in s_ else 1.0); lmax = float(lam.max())
    live = lam > a.reltol * lmax
    w = torch.where(live, (lam.clamp(min=1e-300) / lmax) ** (0.5 * a.q), torch.zeros_like(lam))
    if a.hyper: w = w * torch.exp(s_['u'])
    Dr = w * torch.sign(C)
    if a.cap: Dr = torch.sign(Dr) * torch.minimum(Dr.abs(), Gr.abs() / (lam.clamp(min=1e-300) * max(lr_l, 1e-300) * L1))
    s_['Dr_last'] = Dr
    return rot(rot(Dr, s_['US'], s_['S'].bl, 'L', True), s_['UA'], s_['A'].bl, 'R', True)
M = [torch.zeros_like(w) for w in Wt]; flop = 0.0; rows = []; R = a.refresh; next_ref = 1; n_ref = 0
gen = torch.Generator(device=DEV); gen.manual_seed(7 + a.seed)
def emit(metric, value, step=None):
    rows.append(dict(opt='kronq3h', args=' '.join(sys.argv[1:]), n=a.n, H=a.H, Delta=a.Delta, step=step, metric=metric, value=value))
mem_extra = sum(s_['S'].mem() + s_['A'].mem() for s_ in st) + P     # factors + bases + momentum
emit('mem_extra_over_P', mem_extra / P, 0)
for step in range(1, a.steps + 1):
    lr = a.lr * ((1 - (step - 1) / a.steps) if a.decay else 1.0)
    if a.batch: idx = torch.randint(0, a.n, (a.batch,), device=DEV, generator=gen); Xb, Yb = X[idx], Y[idx]
    else: Xb, Yb = X, Y
    G = grads(Wt, Xb, Yb); flop += 3 * Xb.shape[0] / a.n
    with torch.no_grad():
        for m, g in zip(M, G): m.mul_(a.b1).add_(g, alpha=1 - a.b1)
    if step >= next_ref:
        D_old = [step_dir(l, M[l], G[l], lr * st[l]['eta']) for l in range(L1)] if st[0]['US'] is not None or st[0]['eS'] is not None else None
        sidx = torch.randint(0, a.n, (min(a.stats_n, a.n),), device=DEV, generator=gen)
        acts, sigs = factor_stats(Wt, X[sidx]); flop += 7 * len(sidx) / a.n
        for l, s_ in enumerate(st):
            s_['A'].update(acts[l], a.beta_stats); s_['S'].update(sigs[l] * math.sqrt(sigs[l].shape[0] / len(sidx)), a.beta_stats)
            UA_old, US_old = s_['UA'], s_['US']
            s_['UA'], s_['eA'] = s_['A'].eig(); s_['US'], s_['eS'] = s_['S'].eig()
            if a.hyper and UA_old is not None:
                def ov(Unew, Uold, bl, dim):
                    if Unew is None: return None
                    O = torch.zeros(dim, dim, device=DEV)
                    for ix, Vn, Vo in zip(bl, Unew, Uold): O[ix[:, None], ix[None, :]] = (Vn.T @ Vo) ** 2
                    return O
                OS = ov(s_['US'], US_old, s_['S'].bl, s_['S'].dim); OA = ov(s_['UA'], UA_old, s_['A'].bl, s_['A'].dim)
                for key in ('u', 'z', 'hv'):
                    T = s_[key]
                    if OS is not None: T = OS @ T
                    if OA is not None: T = T @ OA.T
                    s_[key] = T
                s_['Dprev'] = None
            flop += 10 * sum(len(ix) ** 3 for side in (s_['A'], s_['S']) if side.bl is not None for ix in side.bl) / FWD
        n_ref += 1
        if D_old is not None:
            D_new = [step_dir(l, M[l], G[l], lr * st[l]['eta']) for l in range(L1)]
            # staleness error = relative change of the first-order decrease g.D (sign flips on near-zero components cost nothing)
            pn = sum(float((g_ * dn).sum()) for g_, dn in zip(G, D_new)); po = sum(float((g_ * do).sum()) for g_, do in zip(G, D_old))
            err = abs(pn - po) / max(abs(pn), 1e-300)
            if a.adapt_refresh: R = int(min(max(R * min(max(a.refresh_kappa / max(err, 1e-12), 0.5), 2.0), 1), 1000))
            if n_ref % 5 == 0: emit('refresh_err', err, step); emit('refresh_R', R, step)
        next_ref = step + R
    with torch.no_grad():
        if a.hyper and st[0]['eS'] is not None:
            gam = 1.0 - 1.0 / max(a.horizon, 1.0)
            for l, s_ in enumerate(st):
                if s_['Dprev'] is None: continue
                Gr = rot(rot(G[l], s_['US'], s_['S'].bl, 'L', False), s_['UA'], s_['A'].bl, 'R', False)
                s_['z'].mul_(gam).add_(s_['Dprev'])
                hg = -lr * s_['eta'] * Gr * s_['z']                     # dL_t / du_k
                s_['hv'].mul_(0.99).add_(hg * hg, alpha=0.01)
                s_['u'].sub_(a.beta_u * hg / (s_['hv'].sqrt() + 1e-30)).clamp_(-4.6, 4.6)
                if a.hyper_shape: s_['u'].sub_(s_['u'].mean())            # learn only the shape across modes, not the scale
        D_all = [step_dir(l, M[l], G[l], lr * st[l]['eta']) for l in range(L1)]
        if a.hyper:
            for s_ in st: s_['Dprev'] = s_['Dr_last'].clone()
        flop += sum(4 * w.shape[0] * w.shape[1] * (min(a.block or w.shape[0], w.shape[0]) + min(a.block or w.shape[1], w.shape[1])) for w in Wt) / FWD
        if step % a.check_every == 0:              # round-robin per-layer Taylor remainder; drives eta_l if lr_kappa > 0
            l = (step // a.check_every) % L1
            cidx = torch.randint(0, a.n, (min(2000, a.n),), device=DEV, generator=gen); Xc = X[cidx]
            tang = tuple((-lr * st[k]['eta'] * D_all[k]) if k == l else torch.zeros_like(Wt[k]) for k in range(L1))
            with torch.enable_grad():
                f_old, Jd = torch.func.jvp(lambda *ws: forward(list(ws), Xc), tuple(Wt), tang)
            f_l = forward([w + t for w, t in zip(Wt, tang)], Xc); lin = float(Jd.norm()); flop += 3 * len(cidx) / a.n
            if lin > 0:
                rf = float((f_l - f_old - Jd).norm()) / lin; st[l]['rf'] = rf if st[l]['rf'] is None else 0.7 * st[l]['rf'] + 0.3 * rf
                if a.lr_kappa > 0 and lr > 0.05 * a.lr and l < L1 - 1:   # output layer is linear: radius never binds
                    if st[l]['rf'] > 1.5 * a.lr_kappa: st[l]['eta'] *= 0.7
                    elif st[l]['rf'] < a.lr_kappa / 1.5: st[l]['eta'] *= 1.25
                    st[l]['eta'] = min(max(st[l]['eta'], 1e-3), 1e3)
                if a.lr_kappa > 0 and l == L1 - 2: st[L1 - 1]['eta'] = st[L1 - 2]['eta']   # output layer follows the last hidden layer
        for w, D, s_ in zip(Wt, D_all, st): w.sub_(lr * s_['eta'] * D)
    if step % a.log_every == 0 or step == a.steps:
        with torch.no_grad(): rr = forward(Wt, X) - Y
        emit('E_train', float(torch.sqrt((rr * rr).sum() / a.n)), step); emit('flop_eq', flop, step)
        if step % 100 == 0:
            for l, s_ in enumerate(st):
                emit(f'eta_l{l}', s_['eta'], step)
                if a.hyper: emit(f'u_stats_l{l}', [float(s_['u'].mean()), float(s_['u'].std()), float(s_['u'].min()), float(s_['u'].max())], step)
                if s_['rf'] is not None: emit(f'rf_l{l}', s_['rf'], step)
with torch.no_grad():
    rr = forward(Wt, X) - Y; Ef = float(torch.sqrt((rr * rr).sum() / a.n)); rv = forward(Wt, Xv) - Yv; Evf = float(torch.sqrt((rv * rv).sum() / a.nval))
for k_, v_ in (('E_train_final', Ef), ('E_val_final', Evf), ('flop_eq_final', flop), ('n_refresh', n_ref), ('wall_s', time.time() - t0w), ('P', P)): emit(k_, v_, a.steps)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, 'a') as fh:
    for rw in rows: fh.write(json.dumps(rw) + '\n')
print(f'kronq3h hyper={a.hyper} horizon={a.horizon} beta_u={a.beta_u} q={a.q} block={a.block} max_dim={a.max_dim} lr={a.lr} lr_kappa={a.lr_kappa} adapt={a.adapt_refresh} H={a.H} n={a.n} Delta={a.Delta}: '
      f'E_train={Ef:.4g} E_val={Evf:.4g} flop_eq={flop:.4g} refreshes={n_ref} mem_extra/P={mem_extra / P:.2f} eta={[round(s_["eta"], 3) for s_ in st]} wall={time.time() - t0w:.0f}s', flush=True)
