"""Residual spectrum slope gamma (e_k ~ lam_k^gamma) in the kernel eigenbasis at initialisation, per layer block and global,
for the KAdam setup (n = 2000, H160 d2, seed 1); predicted q* = gamma - 1. Also the displacement needed per mode, |r_k|/s_k."""
import sys, os, numpy as np, torch
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
torch.set_default_dtype(torch.float64); torch.set_num_threads(16)
from l63test_torch import MLP
n, H, depth = 2000, 160, 2
for D in ('0.5', '1.0'):
    z = np.load(f'{L63_ROOT}/runs/front/data_n2000_D{D}.npz'); Xr, Yr = z['X'], z['Y']
    mu, sd = Xr.mean(0), Xr.std(0); X = torch.tensor((Xr - mu) / sd); Y = torch.tensor((Yr - mu) / sd).reshape(-1)
    net = MLP(3, H, depth, 3, seed=1); Ws = [w.cpu().clone() for w in net.W]; bs = [b.cpu().clone() for b in net.b]
    shapes = [w.shape for w in Ws] + [b.shape for b in bs]; theta = torch.cat([q.reshape(-1) for q in Ws + bs])
    offs = np.cumsum([0] + [int(np.prod(s)) for s in shapes]); L1 = depth + 1
    def f_flat(th, Xin):
        out = []; i = 0
        for s in shapes: k = int(np.prod(s)); out.append(th[i:i + k].reshape(s)); i += k
        Wl, bl = out[:L1], out[L1:]; h = Xin
        for l in range(depth): h = torch.tanh(h @ Wl[l].T + bl[l])
        return (h @ Wl[-1].T + bl[-1]).reshape(-1)
    J = torch.func.vmap(torch.func.jacrev(lambda t, x: f_flat(t, x[None, :])), in_dims=(None, 0))(theta, X).reshape(3 * n, -1)
    r = (f_flat(theta, X) - Y)
    blocks = {f'layer{l}': np.r_[offs[l]:offs[l + 1], offs[L1 + l]:offs[L1 + l + 1]] for l in range(L1)}; blocks['global'] = np.arange(theta.numel())
    print(f'\nDelta {D}: |r0|^2/n = {float(r @ r) / n:.3f}')
    for name, ix in blocks.items():
        Jb = J[:, ix]; K = Jb @ Jb.T / n
        lam, Q = torch.linalg.eigh(K); lam, Q = lam.flip(0).numpy(), Q.flip(1)
        e = ((Q.T @ r) ** 2).numpy(); keep = lam > 1e-12 * lam[0]; lam, e = lam[keep], e[keep]
        # decade bins: mean residual energy per mode vs eigenvalue
        lb = np.floor(np.log10(lam)); xs, ys = [], []
        for b in np.unique(lb):
            m = lb == b
            if m.sum() >= 5: xs.append(np.log10(lam[m]).mean()); ys.append(np.log10(e[m].mean()))
        xs, ys = np.array(xs), np.array(ys)
        g_all = np.polyfit(xs, ys, 1)[0]
        top = xs > np.log10(lam[0]) - 6; g_top = np.polyfit(xs[top], ys[top], 1)[0] if top.sum() >= 2 else np.nan
        cum = np.cumsum(e) / e.sum(); kmed = int(np.searchsorted(cum, 0.5))
        disp = np.sqrt(e) / np.sqrt(n * lam)            # weight displacement needed to zero each mode alone
        print(f'  {name:7s} modes {keep.sum():5d}  lam1 {lam[0]:.3g}  energy median at rank {kmed} (lam {lam[kmed]:.2g})  '
              f'gamma all {g_all:+.2f} (q* {g_all - 1:+.2f})  gamma top-6-decades {g_top:+.2f} (q* {g_top - 1:+.2f})  '
              f'needed displacement median/90%: {np.median(disp):.2g}/{np.quantile(disp, 0.9):.2g} vs |theta_b| {float(theta[ix].norm()):.3g}')
        print('     decade profile log10(lam): ' + ' '.join(f'{x:.0f}:{y:+.1f}' for x, y in zip(xs, ys)))
