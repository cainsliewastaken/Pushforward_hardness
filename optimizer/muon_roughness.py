"""Muon vs Adam as a function of target roughness (L63 one-step map, Delta = time step), full batch.
Muon: Nesterov momentum (0.95) on each weight matrix, Newton-Schulz (5 it.) orthogonalisation, Jordan's shape scaling;
biases by Adam. Metric: steps (and FLOP-eq incl. Newton-Schulz flops) to reach error levels; best lr per optimizer."""
import sys, os, time, math, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
torch.set_default_dtype(torch.float64); torch.set_num_threads(int(os.environ.get('SEGFLOW_THREADS', '16'))); torch.set_num_interop_threads(1)
from l63_design import make_data_stats
from l63test_torch import MLP
H = int(os.environ.get('MR_H', '160')); n = 200; maxsteps = int(os.environ.get('MR_STEPS', '4000')); seed = 1; DECAY = os.environ.get('MR_DECAY', '0') == '1'   # linear lr decay to zero over maxsteps; report final E at the fixed budget
levels = [0.2, 0.1, 0.05]
def ns5(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315); X = G/(G.norm()+eps); tr = G.shape[0] > G.shape[1]
    if tr: X = X.T
    m, k = X.shape; fl = 0
    for _ in range(steps):
        A = X @ X.T; B = b*A + c*(A @ A); X = a*X + B @ X; fl += 2*m*m*k + 2*m*m*m + 2*m*m*k
    return (X.T if tr else X), fl
def run(opt_name, lr, Delta, depth):
    Xr, Yr = make_data_stats(n, Delta, 1); mu, sd = Xr.mean(0), Xr.std(0); X = torch.tensor((Xr-mu)/sd); Y = torch.tensor((Yr-mu)/sd).reshape(-1)
    net = MLP(3, H, depth, 3, seed=seed); Ws = [torch.nn.Parameter(w.clone()) for w in net.W]; bs = [torch.nn.Parameter(b.clone()) for b in net.b]
    P = sum(p.numel() for p in Ws+bs); FWD = 2*n*P
    def fwd():
        a = X
        for l in range(depth): a = torch.tanh(a @ Ws[l].T + bs[l])
        return (a @ Ws[-1].T + bs[-1]).reshape(-1)
    TR = opt_name == 'muon_tr'; kappa = lr   # for muon_tr the 'lr' slot carries kappa
    if opt_name == 'adam': opt = torch.optim.Adam(Ws+bs, lr=lr); bufs = None
    elif TR: opt = None; bufs = [torch.zeros_like(w) for w in Ws]; bbufs = [torch.zeros_like(b) for b in bs]
    else: opt = torch.optim.Adam(bs, lr=3e-3); bufs = [torch.zeros_like(w) for w in Ws]
    shapes = [w.shape for w in Ws] + [b.shape for b in bs]
    def unpack(th):
        out = []; i = 0
        for sh in shapes: k = int(np.prod(sh)); out.append(th[i:i+k].reshape(sh)); i += k
        return out[:depth+1], out[depth+1:]
    def f_flat(th):
        Wl, bl = unpack(th); a = X
        for l in range(depth): a = torch.tanh(a @ Wl[l].T + bl[l])
        return (a @ Wl[-1].T + bl[-1]).reshape(-1)
    flops = 0.0; hit = [None]*len(levels); E = None
    for step in range(1, maxsteps+1):
        fac = (1 - (step-1)/maxsteps) if DECAY else 1.0
        if opt is not None:
            for grp in opt.param_groups: grp['lr'] = (lr if opt_name == 'adam' else 3e-3)*fac
        for p in Ws+bs: p.grad = None
        r = fwd() - Y; L = 0.5*torch.dot(r, r)/n; L.backward(); flops += 3*FWD
        if bufs is None: opt.step()
        elif TR:
            with torch.no_grad():
                dirs = []
                for w, buf in zip(Ws, bufs):
                    g = w.grad; buf.mul_(0.95).add_(g); gn = g + 0.95*buf; O, fl = ns5(gn); flops += fl
                    dirs.append(O*max(1.0, w.shape[0]/w.shape[1])**0.5)
                bdirs = []
                for b_, bb in zip(bs, bbufs):
                    g = b_.grad; bb.mul_(0.95).add_(g); gn = g + 0.95*bb; bdirs.append(gn/(gn.norm()+1e-12))
                D = torch.cat([q.reshape(-1) for q in dirs + bdirs]); theta = torch.cat([p.reshape(-1) for p in Ws+bs])
                JD = torch.func.jvp(f_flat, (theta,), (D,))[1]; flops += 2*FWD/3                      # one forward-mode product
                eta_lin = float(torch.dot(r.detach(), JD)/torch.dot(JD, JD))                           # linear-model line search
                h2 = torch.func.jvp(lambda th: torch.func.jvp(f_flat, (th,), (D,))[1], (theta,), (D,))[1]; flops += 4*FWD/3   # f''[D,D]
                eta_rad = 2*kappa*float(torch.linalg.norm(JD))/max(float(torch.linalg.norm(h2)), 1e-300)     # Taylor radius
                eta = min(max(eta_lin, 0.0), eta_rad)
                theta = theta - eta*D; Wl, bl = unpack(theta)
                for w, wn in zip(Ws, Wl): w.copy_(wn)
                for b_, bn in zip(bs, bl): b_.copy_(bn)
        else:
            opt.step()                                           # biases
            with torch.no_grad():
                for w, buf in zip(Ws, bufs):
                    g = w.grad; buf.mul_(0.95).add_(g); gn = g + 0.95*buf
                    O, fl = ns5(gn); flops += fl
                    w.add_(O, alpha=-lr*fac*max(1.0, w.shape[0]/w.shape[1])**0.5)
        if step % 10 == 0:
            E = float(torch.sqrt(2*L.detach()))
            if not math.isfinite(E) or E > 1e3: break
            for i, lv in enumerate(levels):
                if hit[i] is None and E <= lv: hit[i] = (step, flops/FWD)
            if hit[-1] and not DECAY: break
    return hit, E
print(f'H={H}, n={n}, full batch, max {maxsteps} steps, linear lr decay={DECAY} (with decay: best lr by final E at the fixed budget). Entries: steps (FLOP-eq) to reach E=0.2 / 0.1 / 0.05, best lr per optimizer. Muon FLOP-eq includes Newton-Schulz.')
for depth in (2, 3):
    for Delta in (0.25, 0.5, 0.75, 1.0, 1.25):
        res = {}
        OPTS = (('adam', (3e-3, 1e-2, 3e-2)), ('muon', (3e-3, 1e-2, 3e-2, 1e-1))) if os.environ.get('MR_TRONLY', '0') != '1' else (('muon_tr', (0.1, 0.3)),)
        for opt_name, lrs in OPTS:
            best = None
            for lr in lrs:
                hit, E = run(opt_name, lr, Delta, depth)
                score = (E if DECAY else 0,) + tuple((h[0] if h else maxsteps+1) for h in hit)
                if best is None or score < best[0]: best = (score, lr, hit, E)
            res[opt_name] = best
        def fmt(b):
            return ' / '.join((f'{h[0]}({h[1]:.0f})' if h else '-') for h in b[2]) + f'  lr={b[1]:g} finalE={b[3]:.3f}'
        if 'adam' not in res:
            rt = res['muon_tr']; print(f'depth {depth} Delta {Delta:4.2f} | Muon-TR (no lr, no schedule): {fmt(rt)}', flush=True); continue
        ra = res['adam']; rm = res['muon']
        ratio = [ (ra[2][i][0]/rm[2][i][0]) if (ra[2][i] and rm[2][i]) else float('nan') for i in range(len(levels))]
        print(f'depth {depth} Delta {Delta:4.2f} | Adam: {fmt(ra)} | Muon: {fmt(rm)} | steps ratio Adam/Muon: ' + ' '.join(f'{x:.2f}' for x in ratio), flush=True)
