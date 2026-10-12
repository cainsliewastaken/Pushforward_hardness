"""Linearised (frozen-kernel) KAdam-sign dynamics, simulated exactly in output space. Per layer block b, the kept GN modes
give J_b V_b = Q_b diag(s_b). Each step: g_k = s_k (Q_k^T r)/n per mode; momentum m_k (beta 0.9, fixed basis); displacement
a_k = lr_t (s_k/s_1b)^q sign(m_k), optionally capped at |g_k|/(lam_k L); residual update r -= sum_b Q_b (s_b a_b).
Same lr (3e-3, linear decay), steps (3000), init and data as the real n = 2000 runs. No feature learning: if q* here
matches the real optimum, q = -0.25 is a property of the initial kernel and the target's projection on it."""
import sys, os, numpy as np, torch
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
torch.set_default_dtype(torch.float64); torch.set_num_threads(16)
from l63test_torch import MLP
n, H, depth = 2000, 160, 2
def setup(D, centred=False):
    z = np.load(f'{L63_ROOT}/runs/front/data_n2000_D{D}.npz'); Xr = z['X']
    mu, sd = Xr.mean(0), Xr.std(0); X = torch.tensor((Xr - mu) / sd); Y = torch.tensor((z['Y'] - mu) / sd).reshape(-1)
    net = MLP(3, H, depth, 3, seed=1); Ws = [w.cpu().clone() for w in net.W]; bs = [b.cpu().clone() for b in net.b]
    shapes = [w.shape for w in Ws] + [b.shape for b in bs]; th = torch.cat([q.reshape(-1) for q in Ws + bs]); P = th.numel()
    offs = np.cumsum([0] + [int(np.prod(s)) for s in shapes]); L1 = depth + 1
    def f_one(t, x):
        out = []; i = 0
        for s in shapes: k = int(np.prod(s)); out.append(t[i:i + k].reshape(s)); i += k
        h = x[None, :]
        for l in range(depth): h = torch.tanh(h @ out[l].T + out[L1 + l])
        return (h @ out[depth].T + out[L1 + depth]).reshape(-1)
    J = torch.func.vmap(torch.func.jacrev(f_one), in_dims=(None, 0))(th, X).reshape(3 * n, P)
    f0 = torch.func.vmap(f_one, in_dims=(None, 0))(th, X).reshape(-1)
    r0 = (-Y) if centred else (f0 - Y)
    blocks = []
    for l in range(L1):
        ix = np.r_[offs[l]:offs[l + 1], offs[L1 + l]:offs[L1 + l + 1]]; Jb = J[:, ix]
        if len(ix) <= 3 * n:
            lam, V = torch.linalg.eigh(Jb.T @ Jb / n); lam, V = lam.flip(0), V.flip(1); keep = lam > 1e-12 * lam[0]
            lam = lam[keep]; QS = Jb @ V[:, keep]                      # = Q diag(s)
        else:
            lam, Q = torch.linalg.eigh(Jb @ Jb.T / n); lam, Q = lam.flip(0), Q.flip(1); keep = lam > 1e-12 * lam[0]
            lam = lam[keep]; QS = Q[:, keep] * (n * lam).sqrt()
        s = (n * lam).sqrt(); blocks.append(dict(QS=QS, s=s, lam=lam))
    return r0, blocks
def run(r0, blocks, q, cap, lr0=3e-3, steps=3000, beta=0.9):
    r = r0.clone(); L = len(blocks); ms = [torch.zeros_like(B['s']) for B in blocks]; curve = {}
    for t in range(1, steps + 1):
        lr = lr0 * (1 - (t - 1) / steps); dr = torch.zeros_like(r)
        for B, m in zip(blocks, ms):
            gk = (B['QS'].T @ r) / n                                   # = s_k (Q_k^T r)/n
            m.mul_(beta).add_(gk, alpha=1 - beta)
            a = lr * (B['s'] / B['s'][0]) ** q * torch.sign(m)
            if cap: a = torch.sign(a) * torch.minimum(a.abs(), gk.abs() / (B['lam'] * L))
            dr += B['QS'] @ a
        r -= dr
        if t in (300, 1000, 3000): curve[t] = float(r.norm()) / n ** 0.5
    return curve
qs = [0.5, 0.25, 0.0, -0.125, -0.25, -0.375, -0.5, -0.75, -1.0]
for D in ('1.0', '0.5'):
    for centred in (False, True):
        r0, blocks = setup(D, centred)
        print(f'\nDelta {D}, {"centred" if centred else "raw"} init: lazy (frozen-kernel) E_rms at steps 300 / 1000 / 3000; E0 = {float(r0.norm()) / n ** 0.5:.3f}')
        for cap in (0, 1):
            res = {q: run(r0, blocks, q, cap) for q in qs}
            best = min(qs, key=lambda q: res[q][3000])
            print(f'  cap={cap}: ' + '  '.join(f'q{q:+.3g}: {res[q][300]:.3f}/{res[q][1000]:.3f}/{res[q][3000]:.4f}' for q in qs) + f'   -> best q {best:+.3g}', flush=True)
