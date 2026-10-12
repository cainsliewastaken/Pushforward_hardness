"""Which ingredient of Adam gives the speed? Steps (and FLOP-eq = 3*steps) to reach GD's final error for:
Adam, RMSprop (no momentum), heavy-ball momentum GD (no normalisation), signGD (normalisation without memory),
layer-normalised GD (one scale per layer = layer balance only), plain GD at the largest stable step. Also the per-layer
displacement ||W_l(T)-W_l(0)|| of each method, and of the saved GD run."""
import sys, os, re, time, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
torch.set_default_dtype(torch.float64); torch.set_num_threads(8); torch.set_num_interop_threads(1)
from l63_design import make_data_stats
from l63test_torch import MLP
tag = sys.argv[1]; maxsteps = int(sys.argv[2])
mm = re.match(r'H(\d+)_D([\d.]+)_depth(\d+)_s(\d+)', tag); H, Delta, depth, seed = int(mm[1]), float(mm[2]), int(mm[3]), int(mm[4])
d = np.load(f'../runs/heldout/des_{tag}.npy', allow_pickle=True).item(); n = d['n']; E_T = float(d['E'][-1]); lam1 = float(d['lam1_0'])
Xr, Yr = make_data_stats(n, Delta, 1); mu, sd = Xr.mean(0), Xr.std(0); X = torch.tensor((Xr-mu)/sd); Y = torch.tensor((Yr-mu)/sd).reshape(-1)
net = MLP(3, H, depth, 3, seed=seed); W0 = [w.clone() for w in net.W]; b0 = [b.clone() for b in net.b]
def run(method, lr, **kw):
    Ws = [torch.nn.Parameter(w.clone()) for w in W0]; bs = [torch.nn.Parameter(b.clone()) for b in b0]; ps = Ws + bs
    def fwd():
        a = X
        for l in range(depth): a = torch.tanh(a @ Ws[l].T + bs[l])
        return (a @ Ws[-1].T + bs[-1]).reshape(-1)
    if method == 'adam': opt = torch.optim.Adam(ps, lr=lr)
    elif method == 'rmsprop': opt = torch.optim.Adam(ps, lr=lr, betas=(0.0, 0.999))
    elif method == 'signgd': opt = torch.optim.Adam(ps, lr=lr, betas=(0.0, 0.0), eps=1e-12)
    elif method == 'momentum': opt = torch.optim.SGD(ps, lr=lr, momentum=0.9)
    elif method == 'gd': opt = torch.optim.SGD(ps, lr=lr)
    elif method == 'layernorm': opt = None
    hit = None; E = None
    for step in range(1, maxsteps+1):
        for p in ps: p.grad = None
        r = fwd() - Y; L = 0.5*torch.dot(r, r)/n; L.backward()
        if opt is None:
            with torch.no_grad():
                for l in range(depth+1):          # one scale per layer: unit-norm layer gradient (weights+bias together)
                    g = torch.sqrt(torch.sum(Ws[l].grad**2) + torch.sum(bs[l].grad**2)) + 1e-12
                    Ws[l] -= lr*Ws[l].grad/g; bs[l] -= lr*bs[l].grad/g
        else: opt.step()
        if step % 50 == 0:
            E = float(torch.sqrt(2*L.detach()))
            if not np.isfinite(E) or E > 1e3: return None, None
            if E <= E_T: hit = step; break
    disp = [float(torch.linalg.norm(Ws[l].detach()-W0[l])) for l in range(depth+1)]
    return hit, disp
print(f'{tag}: target E_GD(T)={E_T:.4f}; lam1(0)={lam1:.1f}; GD run took {d["steps"]} steps')
best = {}
for method, lrs in [('adam', (3e-3, 1e-2, 3e-2)), ('rmsprop', (3e-3, 1e-2, 3e-2)), ('signgd', (3e-4, 1e-3, 3e-3)), ('momentum', tuple(x/lam1 for x in (0.5, 1.0, 2.0, 3.0))), ('gd', tuple(x/lam1 for x in (0.4, 1.0, 1.8))), ('layernorm', (0.01, 0.03, 0.1))]:
    for lr in lrs:
        t0 = time.time(); hit, disp = run(method, lr)
        tagl = f'{method:9s} lr={lr:.3g}'
        print(f'   {tagl:22s}: ' + (f'{hit:6d} steps' if hit else f'  not reached in {maxsteps}') + (f'  |dW_l| = {np.round(disp, 2)}' if disp else '') + f'  ({time.time()-t0:.0f}s)', flush=True)
        if hit and (method not in best or hit < best[method][0]): best[method] = (hit, lr, disp)
print('   best per method:', {k: v[0] for k, v in best.items()})
