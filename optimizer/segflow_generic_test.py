"""Generality test of SegFlow on problems unrelated to L63: (a) GELU MLP fitting a rough 1-D function,
(b) small ReLU conv net regressing a smooth 2-D field.  Compared with full-batch GD (eta = 0.4/lam1) and Adam at a
matched budget of forward+backward passes (one SegFlow Lanczos iteration = 1 jvp + 1 vjp is counted as one pass)."""
import sys, os, time, math, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
torch.set_default_dtype(torch.float64); torch.set_num_threads(int(os.environ.get('SEGFLOW_THREADS', '8'))); torch.set_num_interop_threads(1); torch.manual_seed(0)
from segflow import SegFlow

def run_problem(name, model, X, Y, n, budget_passes, adam_lr=1e-3):
    # FLOP-matched: Adam/GD get budget_passes steps (3 fwd-eq each); SegFlow stops when its fwd-eq count reaches 3*budget_passes
    params = dict(model.named_parameters()); names = list(params); shapes = [params[k].shape for k in names]
    theta0 = torch.cat([params[k].detach().reshape(-1) for k in names]).clone()
    def f(theta):
        i = 0; d = {}
        for k, s in zip(names, shapes): m = int(np.prod(s)); d[k] = theta[i:i+m].reshape(s); i += m
        return torch.func.functional_call(model, d, (X,)).reshape(-1)
    y = Y.reshape(-1); rms = lambda th: float(torch.sqrt(torch.sum((f(th)-y)**2)/n))
    # SegFlow until the pass budget is spent
    opt = SegFlow(f, theta0, y, n=n, delta0=0.1, m_max=30); t0 = time.time()
    flops = lambda: 2*opt.counts['jvp'] + 3*opt.counts['vjp'] + opt.counts['forward']      # forward-equivalents: fwd 1, bwd 2, jvp 2, vjp 3
    while flops() < 3*budget_passes and opt.rms(opt.r) > 1e-6: opt.step()
    E_sf = opt.rms(opt.r); w_sf = time.time()-t0; seg = opt.counts['segments']; fl_sf = flops()
    # GD with eta = 0.4/lam1 (lam1 by power iteration on K) and Adam, same number of passes
    def loss(th): return 0.5*torch.sum((f(th)-y)**2)/n
    v = torch.randn_like(y); 
    for _ in range(30):
        _, pull = torch.func.vjp(f, theta0); u = pull(v)[0]; Kv = torch.func.jvp(f, (theta0,), (u,))[1]/n; lam1 = float(torch.linalg.norm(Kv)); v = Kv/lam1
    eta = 0.4/lam1
    th = theta0.clone(); t0 = time.time()
    for _ in range(budget_passes):
        g = torch.func.grad(loss)(th); th = th - eta*g
    E_gd = rms(th); w_gd = time.time()-t0
    E_ad = {}; t0 = time.time()
    for lr in (3e-4, 1e-3, 3e-3, 1e-2):
        th = theta0.clone().requires_grad_(True); ad = torch.optim.Adam([th], lr=lr)
        for _ in range(budget_passes):
            ad.zero_grad(); L = loss(th); L.backward(); ad.step()
        E_ad[lr] = rms(th.detach())
    adam_lr = min(E_ad, key=E_ad.get); w_ad = time.time()-t0; E_ad_all = E_ad; E_ad = E_ad[adam_lr]
    print(f'{name}: P={theta0.numel()} N={y.numel()} E0={rms(theta0):.4f} | FLOP budget = {budget_passes} Adam/GD steps (= {3*budget_passes} fwd-eq; SegFlow used {fl_sf}) | SegFlow {E_sf:.5f} ({seg} segments, {opt.counts['jvp']} Lanczos its, {w_sf:.0f}s) | GD {E_gd:.5f} ({w_gd:.0f}s) | Adam best lr={adam_lr:g}: {E_ad:.5f} (all: ' + ' '.join(f'{k:g}:{v:.4f}' for k, v in E_ad_all.items()) + f') ({w_ad:.0f}s)', flush=True)
    return opt

# (a) GELU MLP, rough 1-D target
n = 256; X = torch.linspace(-1, 1, n).reshape(-1, 1); Y = torch.sin(12*X) + 0.5*torch.sign(torch.sin(5*X)) + 0.3*torch.abs(X)
mlp = torch.nn.Sequential(torch.nn.Linear(1, 64), torch.nn.GELU(), torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
o = run_problem('GELU-MLP 1D rough', mlp, X, Y, n, budget_passes=3000)
# (b) small conv net, 2-D field regression (8x8 images -> scalar)
n = 200; Xi = torch.randn(n, 1, 8, 8); Yi = (Xi[:, 0, 2:6, 2:6].mean((1, 2))*torch.cos(Xi[:, 0, 0, 0])).reshape(-1, 1)
cnn = torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.Tanh(), torch.nn.Conv2d(8, 8, 3, padding=1), torch.nn.Tanh(), torch.nn.Flatten(), torch.nn.Linear(8*64, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
run_problem('tanh-CNN 8x8 regression', cnn, Xi, Yi, n, budget_passes=2000, adam_lr=1e-3)
