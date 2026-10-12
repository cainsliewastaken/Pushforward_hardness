"""FLOPs (forward-equivalents) needed by SegFlow, GD and Adam (best lr per threshold) to reach error thresholds E0/4, E0/16, E0/64
on the two non-L63 problems. fwd 1, bwd 2, jvp 2, vjp 3."""
import sys, os, time, math, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
torch.set_default_dtype(torch.float64); torch.set_num_threads(int(os.environ.get('SEGFLOW_THREADS', '8'))); torch.set_num_interop_threads(1); torch.manual_seed(0)
from segflow import SegFlow
def run_problem(name, model, X, Y, n, max_steps):
    params = dict(model.named_parameters()); names = list(params); shapes = [params[k].shape for k in names]
    theta0 = torch.cat([params[k].detach().reshape(-1) for k in names]).clone()
    def f(theta):
        i = 0; d = {}
        for k, s in zip(names, shapes): m = int(np.prod(s)); d[k] = theta[i:i+m].reshape(s); i += m
        return torch.func.functional_call(model, d, (X,)).reshape(-1)
    y = Y.reshape(-1); rms = lambda th: float(torch.sqrt(torch.sum((f(th)-y)**2)/n)); E0 = rms(theta0); thr = [E0/4, E0/16, E0/64]
    def loss(th): return 0.5*torch.sum((f(th)-y)**2)/n
    # SegFlow
    opt = SegFlow(f, theta0, y, n=n, delta0=0.1, m_max=30); fl = lambda: 2*opt.counts['jvp'] + 3*opt.counts['vjp'] + opt.counts['forward']
    hit_sf = [None]*3
    while fl() < 3*max_steps and opt.rms(opt.r) > thr[-1]*0.999:
        opt.step(); e = opt.rms(opt.r)
        for i, t in enumerate(thr):
            if hit_sf[i] is None and e <= t: hit_sf[i] = fl()
    # GD
    v = torch.randn_like(y)
    for _ in range(30):
        _, pull = torch.func.vjp(f, theta0); u = pull(v)[0]; Kv = torch.func.jvp(f, (theta0,), (u,))[1]/n; lam1 = float(torch.linalg.norm(Kv)); v = Kv/lam1
    th = theta0.clone(); hit_gd = [None]*3
    for s in range(1, max_steps+1):
        th = th - (0.4/lam1)*torch.func.grad(loss)(th)
        if s % 50 == 0:
            e = rms(th)
            for i, t in enumerate(thr):
                if hit_gd[i] is None and e <= t: hit_gd[i] = 3*s
            if hit_gd[-1]: break
    # Adam, best lr per threshold
    hit_ad = [None]*3
    for lr in (1e-3, 3e-3, 1e-2, 3e-2):
        th = theta0.clone().requires_grad_(True); ad = torch.optim.Adam([th], lr=lr); h = [None]*3
        for s in range(1, max_steps+1):
            ad.zero_grad(); L = loss(th); L.backward(); ad.step()
            if s % 50 == 0:
                e = float(torch.sqrt(2*L.detach()))
                for i, t in enumerate(thr):
                    if h[i] is None and e <= t: h[i] = 3*s
                if h[-1]: break
        for i in range(3):
            if h[i] is not None and (hit_ad[i] is None or h[i] < hit_ad[i]): hit_ad[i] = h[i]
    fmt = lambda hs: '  '.join('%8s' % (str(x) if x else '>%d' % (3*max_steps)) for x in hs)
    print(f'{name} (P={theta0.numel()}, E0={E0:.3f}); fwd-eq FLOPs to reach E0/4, E0/16, E0/64:'); print('   SegFlow ', fmt(hit_sf)); print('   GD      ', fmt(hit_gd)); print('   Adam*   ', fmt(hit_ad), ' (* best lr per threshold of 1e-3..3e-2)', flush=True)
n = 256; X = torch.linspace(-1, 1, n).reshape(-1, 1); Y = torch.sin(12*X) + 0.5*torch.sign(torch.sin(5*X)) + 0.3*torch.abs(X)
mlp = torch.nn.Sequential(torch.nn.Linear(1, 64), torch.nn.GELU(), torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
run_problem('GELU-MLP 1D rough', mlp, X, Y, n, max_steps=20000)
n = 200; Xi = torch.randn(n, 1, 8, 8); Yi = (Xi[:, 0, 2:6, 2:6].mean((1, 2))*torch.cos(Xi[:, 0, 0, 0])).reshape(-1, 1)
cnn = torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.Tanh(), torch.nn.Conv2d(8, 8, 3, padding=1), torch.nn.Tanh(), torch.nn.Flatten(), torch.nn.Linear(8*64, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
run_problem('tanh-CNN 8x8 regression', cnn, Xi, Yi, n, max_steps=5000)
