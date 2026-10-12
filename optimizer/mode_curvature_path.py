"""Per-mode loss curvature along training. At Adam checkpoints (lr 1e-2, linear decay over 3000 steps; n = 2000, H160 d2,
seed 1), for log-spaced kernel eigenmodes V_k of each hidden layer block: lam_k (Gauss-Newton curvature), the residual
curvature r.f''[V_k,V_k]/n (the non-GN part of the loss Hessian; forward-over-forward jvp), and the full curvature
h_k = lam_k + r.f''/n. Output layer omitted (f is linear in it: f'' = 0)."""
import sys, os, numpy as np, torch
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
torch.set_default_dtype(torch.float64); torch.set_num_threads(16)
from l63test_torch import MLP
n, H, depth = 2000, 160, 2; D = sys.argv[1]; CK = [int(c) for c in sys.argv[2].split(',')]
z = np.load(f'{L63_ROOT}/runs/front/data_n2000_D{D}.npz'); Xr = z['X']
mu, sd = Xr.mean(0), Xr.std(0); X = torch.tensor((Xr - mu) / sd); Y = torch.tensor((z['Y'] - mu) / sd).reshape(-1)
net = MLP(3, H, depth, 3, seed=1); Ws = [w.cpu().clone() for w in net.W]; bs = [b.cpu().clone() for b in net.b]
shapes = [w.shape for w in Ws] + [b.shape for b in bs]; th0 = torch.cat([q.reshape(-1) for q in Ws + bs]); P = th0.numel()
offs = np.cumsum([0] + [int(np.prod(s)) for s in shapes]); L1 = depth + 1
def unpack(th):
    out = []; i = 0
    for s in shapes: k = int(np.prod(s)); out.append(th[i:i + k].reshape(s)); i += k
    return out[:L1], out[L1:]
def f_at(th, Xin):
    Wl, bl = unpack(th); h = Xin
    for l in range(depth): h = torch.tanh(h @ Wl[l].T + bl[l])
    return (h @ Wl[-1].T + bl[-1]).reshape(-1)
f_flat = lambda th: f_at(th, X)
f_one = lambda th, x: f_at(th, x[None, :])
states = {0: th0.clone()}; th = th0.clone().requires_grad_(True); opt = torch.optim.Adam([th], lr=1e-2)
for s in range(1, max(CK) + 1):
    for gp in opt.param_groups: gp['lr'] = 1e-2 * (1 - (s - 1) / 3000)
    opt.zero_grad(); rr = f_flat(th) - Y; (0.5 * (rr @ rr) / n).backward(); opt.step()
    if s in CK: states[s] = th.detach().clone()
print(f'Delta {D}: per-mode curvature along an Adam run (columns: rank, lambda, r.f\'\'/n, h = lambda + r.f\'\'/n)')
for c in CK:
    theta = states[c]; r = (f_flat(theta) - Y).detach()
    J = torch.func.vmap(torch.func.jacrev(f_one), in_dims=(None, 0))(theta, X).reshape(3 * n, P)
    print(f'\n=== step {c}: E_rms {float(r.norm()) / n ** 0.5:.4f}')
    for l in range(L1 - 1):
        ix = np.r_[offs[l]:offs[l + 1], offs[L1 + l]:offs[L1 + l + 1]]; Jb = J[:, ix]
        if len(ix) <= 3 * n:
            lam, V = torch.linalg.eigh(Jb.T @ Jb / n); lam, V = lam.flip(0), V.flip(1); vec = lambda k: V[:, k]
        else:
            lam, Qm = torch.linalg.eigh(Jb @ Jb.T / n); lam, Qm = lam.flip(0), Qm.flip(1); vec = lambda k: Jb.T @ Qm[:, k]
        keep = int((lam > 1e-12 * lam[0]).sum()); ranks = np.unique(np.geomspace(1, keep, 16).astype(int)) - 1
        rows = []
        for k in ranks:
            vb = vec(k); v = torch.zeros(P); v[torch.tensor(ix)] = vb / vb.norm()
            g = lambda t: torch.func.jvp(f_flat, (t,), (v,))[1]
            fpp = torch.func.jvp(g, (theta,), (v,))[1]; hr = float(r @ fpp) / n; rows.append((k + 1, float(lam[k]), hr))
        R = np.array(rows); cross = next((int(k) for k, lk, hr in rows if lk + hr < 0), None)
        print(f'  layer {l}: residual curvature negative on {(R[:, 2] < 0).mean():.0%} of modes; full curvature h < 0 from rank {cross}; '
              f'|r.f\'\'/n| median {np.median(np.abs(R[:, 2])):.2g}')
        print('    ' + '  '.join(f'{int(k)}:{lk:.1e}/{hr:+.1e}/{lk + hr:+.1e}' for k, lk, hr in rows))
