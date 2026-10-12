"""Efficient-roughness-supply step: Adam on F = L - beta * L.detach() * log q(theta), q = <r,K r>/||r||^2 = ||J^T r||^2/(n||r||^2)
with r held fixed inside q (so grad q rotates/grows the kernel toward the current residual). beta = 0 is plain Adam.
Cost per step: fwd+bwd (3) + double-backward for grad q (~4) = 7 fwd-eq vs Adam's 3; reported at equal steps and equal FLOPs."""
import sys, os, math, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
torch.set_default_dtype(torch.float64); torch.set_num_threads(int(os.environ.get('SEGFLOW_THREADS', '16'))); torch.set_num_interop_threads(1)
from l63_design import make_data_stats
from l63test_torch import MLP
H = 160; n = 200; seed = 1
def run(beta, lr, Delta, depth, steps_max):
    Xr, Yr = make_data_stats(n, Delta, 1); mu, sd = Xr.mean(0), Xr.std(0); X = torch.tensor((Xr-mu)/sd); Y = torch.tensor((Yr-mu)/sd).reshape(-1)
    net = MLP(3, H, depth, 3, seed=seed); Ws = [torch.nn.Parameter(w.clone()) for w in net.W]; bs = [torch.nn.Parameter(b.clone()) for b in net.b]; ps = Ws+bs
    opt = torch.optim.Adam(ps, lr=lr)
    def fwd():
        a = X
        for l in range(depth): a = torch.tanh(a @ Ws[l].T + bs[l])
        return (a @ Ws[-1].T + bs[-1]).reshape(-1)
    E = None
    for step in range(1, steps_max+1):
        for grp in opt.param_groups: grp['lr'] = lr*(1 - (step-1)/steps_max)
        opt.zero_grad(); out = fwd(); r = out - Y; L = 0.5*torch.dot(r, r)/n
        if beta > 0:
            g = torch.autograd.grad(torch.dot(out, r.detach()), ps, create_graph=True)   # J^T r with r fixed
            q = sum((gi**2).sum() for gi in g)/(n*float(torch.dot(r, r).detach()))
            F = L - beta*L.detach()*torch.log(q)
        else: F = L
        F.backward(); opt.step()
        if step % 20 == 0:
            E = float(torch.sqrt(2*L.detach()))
            if not math.isfinite(E) or E > 1e3: return float('inf')
    return float(torch.sqrt(2*L.detach()))
print(f'H={H} n={n} full batch, linear lr decay; final E (best lr of 3e-3,1e-2,3e-2) ; beta=0 is Adam. Equal-steps columns, then Adam at equal FLOPs (7/3 x steps).')
for depth in (2, 3):
    for Delta in (0.5, 1.0):
        for steps in (1000, 3000):
            cells = []
            for beta in (0.0, 0.1, 0.3, 1.0):
                best = min((run(beta, lr, Delta, depth, steps), lr) for lr in (3e-3, 1e-2, 3e-2)); cells.append(best)
            adam_eq = min((run(0.0, lr, Delta, depth, int(steps*7/3)), lr) for lr in (3e-3, 1e-2, 3e-2))
            print(f'depth {depth} Delta {Delta} steps {steps} | ' + '  '.join(f'b={b}: {c[0]:.4f}(lr {c[1]:g})' for b, c in zip((0.0, 0.1, 0.3, 1.0), cells)) + f' | Adam at equal FLOPs ({int(steps*7/3)} steps): {adam_eq[0]:.4f}', flush=True)
