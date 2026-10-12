"""Partial spectral flattening: per-layer update U diag(sigma^q) V^T of the Nesterov-momentum gradient (exact SVD; H=160 is small).
q = 1 momentum GD (scaled), q = 0 Muon, q = -0.5 Gauss-Newton-like. Theory prediction: the best q decreases with target roughness.
Fixed budget, linear lr decay, lr swept; final training error."""
import sys, os, math, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
torch.set_default_dtype(torch.float64); torch.set_num_threads(int(os.environ.get('SEGFLOW_THREADS', '16'))); torch.set_num_interop_threads(1)
from l63_design import make_data_stats
from l63test_torch import MLP
H = 160; n = 200; steps_max = int(os.environ.get('SP_STEPS', '1000')); seed = 1
qs = [1.0, 0.5, 0.25, 0.0, -0.25, -0.5]; lrs = (3e-3, 1e-2, 3e-2, 1e-1, 3e-1)
def run(q, lr, Delta, depth):
    Xr, Yr = make_data_stats(n, Delta, 1); mu, sd = Xr.mean(0), Xr.std(0); X = torch.tensor((Xr-mu)/sd); Y = torch.tensor((Yr-mu)/sd).reshape(-1)
    net = MLP(3, H, depth, 3, seed=seed); Ws = [torch.nn.Parameter(w.clone()) for w in net.W]; bs = [torch.nn.Parameter(b.clone()) for b in net.b]
    opt = torch.optim.Adam(bs, lr=3e-3); bufs = [torch.zeros_like(w) for w in Ws]
    def fwd():
        a = X
        for l in range(depth): a = torch.tanh(a @ Ws[l].T + bs[l])
        return (a @ Ws[-1].T + bs[-1]).reshape(-1)
    E = None
    for step in range(1, steps_max+1):
        fac = 1 - (step-1)/steps_max
        for grp in opt.param_groups: grp['lr'] = 3e-3*fac
        for p in Ws+bs: p.grad = None
        r = fwd() - Y; L = 0.5*torch.dot(r, r)/n; L.backward(); opt.step()
        with torch.no_grad():
            for w, buf in zip(Ws, bufs):
                g = w.grad; buf.mul_(0.95).add_(g); gn = g + 0.95*buf
                U, S, Vh = torch.linalg.svd(gn, full_matrices=False); Sq = (S/S.max())**q          # normalised so the top singular value maps to 1 for every q
                upd = (U*Sq) @ Vh
                w.add_(upd, alpha=-lr*fac*max(1.0, w.shape[0]/w.shape[1])**0.5)
        if step % 20 == 0:
            E = float(torch.sqrt(2*L.detach()))
            if not math.isfinite(E) or E > 1e3: return float('inf')
    return float(torch.sqrt(2*L.detach()))
print(f'H={H} n={n} full batch, budget {steps_max} steps, linear lr decay; final E (best lr) per spectral power q: ' + ' '.join(f'q={q:g}' for q in qs) + '   (q=0 is Muon, q=1 momentum GD)')
for depth in (2, 3):
    for Delta in (0.25, 0.5, 0.75, 1.0, 1.25):
        row = []
        for q in qs:
            best = min((run(q, lr, Delta, depth), lr) for lr in lrs); row.append(best)
        bq = qs[int(np.argmin([b[0] for b in row]))]
        print(f'depth {depth} Delta {Delta:4.2f} | ' + '  '.join(f'{b[0]:.4f}(lr {b[1]:g})' for b in row) + f'  | best q = {bq:g}', flush=True)
