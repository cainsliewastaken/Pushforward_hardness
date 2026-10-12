"""Per-mode Taylor radius at initialisation. For each layer block's kernel eigenmode V_k (unit vector in that block's
parameters, J V_k = s_k Q_k, s_k = sqrt(n lam_k)), the function-space curvature c_k = ||f''[V_k, V_k]|| (forward-over-forward
jvp on the training data). Holding the remainder (1/2) a^2 c_k <= kappa a s_k gives the per-mode radius a_k <= 2 kappa s_k / c_k,
so if c_k ~ s_k^m the radius scales as s_k^(1-m): the theory's per-mode exponent is q = 1 - m (local slope, not assumed constant)."""
import sys, os, numpy as np, torch
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
torch.set_default_dtype(torch.float64); torch.set_num_threads(16)
from l63test_torch import MLP
n, H, depth = 2000, 160, 2; D = sys.argv[1] if len(sys.argv) > 1 else '1.0'
z = np.load(f'{L63_ROOT}/runs/front/data_n2000_D{D}.npz'); Xr = z['X']
mu, sd = Xr.mean(0), Xr.std(0); X = torch.tensor((Xr - mu) / sd); Y = torch.tensor((z['Y'] - mu) / sd).reshape(-1)
net = MLP(3, H, depth, 3, seed=1); Ws = [w.cpu().clone() for w in net.W]; bs = [b.cpu().clone() for b in net.b]
shapes = [w.shape for w in Ws] + [b.shape for b in bs]; theta = torch.cat([q.reshape(-1) for q in Ws + bs]); P = theta.numel()
offs = np.cumsum([0] + [int(np.prod(s)) for s in shapes]); L1 = depth + 1
def f_flat(th):
    out = []; i = 0
    for s in shapes: k = int(np.prod(s)); out.append(th[i:i + k].reshape(s)); i += k
    Wl, bl = out[:L1], out[L1:]; h = X
    for l in range(depth): h = torch.tanh(h @ Wl[l].T + bl[l])
    return (h @ Wl[-1].T + bl[-1]).reshape(-1)
def f_one(th, x):
    out = []; i = 0
    for s in shapes: k = int(np.prod(s)); out.append(th[i:i + k].reshape(s)); i += k
    Wl, bl = out[:L1], out[L1:]; h = x[None, :]
    for l in range(depth): h = torch.tanh(h @ Wl[l].T + bl[l])
    return (h @ Wl[-1].T + bl[-1]).reshape(-1)
J = torch.func.vmap(torch.func.jacrev(f_one), in_dims=(None, 0))(theta, X).reshape(3 * n, P)
r0 = None
def curv(v):  # f''[v, v] via jvp of jvp: returns ||f''|| and the residual-projected curvature r.f''/n
    g = lambda th: torch.func.jvp(f_flat, (th,), (v,))[1]
    fpp = torch.func.jvp(g, (theta,), (v,))[1]
    return float(fpp.norm()), float(r0 @ fpp) / n
r0 = (f_flat(theta) - Y).detach()
print(f'Delta {D}, init, seed 1. Per block: modes sampled log-spaced in rank; s_k, c_k, radius s_k/c_k; local q = 1 - dlog c / dlog s')
for l in range(L1 - 1):     # output layer: f is linear in its parameters, f'' = 0 exactly
    ix = np.r_[offs[l]:offs[l + 1], offs[L1 + l]:offs[L1 + l + 1]]; Jb = J[:, ix]
    G = Jb.T @ Jb / n if len(ix) <= 3 * n else None
    if G is not None:
        lam, V = torch.linalg.eigh(G); lam, V = lam.flip(0), V.flip(1)
    else:
        lam, Qm = torch.linalg.eigh(Jb @ Jb.T / n); lam, Qm = lam.flip(0), Qm.flip(1)
    keep = int((lam > 1e-12 * lam[0]).sum())
    ranks = np.unique(np.geomspace(1, keep, 24).astype(int)) - 1
    rows = []
    for k in ranks:
        lk = float(lam[k]); sk = (n * lk) ** 0.5
        vb = V[:, k] if G is not None else (Jb.T @ Qm[:, k]) / sk
        v = torch.zeros(P); v[torch.tensor(ix)] = vb / vb.norm()
        ck, hk = curv(v); rows.append((k + 1, lk, sk, ck, hk))
    R = np.array(rows); ls, lc = np.log10(R[:, 2]), np.log10(R[:, 3])
    neg = (R[:, 4] < 0).mean(); big = (np.abs(R[:, 4]) > R[:, 1]).mean()
    m_all = np.polyfit(ls, lc, 1)[0]
    print(f'\nlayer {l}: {keep} modes; c ~ s^{m_all:.2f} (q from |f\'\'| radius = {1 - m_all:+.2f}); residual curvature r.f\'\'/n negative on {neg:.0%} of sampled modes, larger than lambda_k on {big:.0%}')
    print('   rank   lambda      s_k       c_k      r.f\'\'/n     full Hessian h_k = lambda + r.f\'\'/n   ratio h/lambda')
    for (k, lk, sk, ck, hk) in rows:
        print(f'   {int(k):5d}  {lk:9.2e}  {sk:9.2e}  {ck:9.2e}  {hk:+10.2e}   {lk + hk:+10.2e}   {(lk + hk) / lk:+10.2e}')
