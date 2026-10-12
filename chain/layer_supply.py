"""Per-layer roughness-supply sensitivity along Adam runs: q = <r,Kr>/||r||^2 (r fixed); for Adam's actual step dtheta,
share_l = (grad_theta q)_l . dtheta_l / sum_l (...) = fraction of the kernel-gain-on-the-residual bought by layer l's move,
and sens_l = (grad q)_l . dtheta_l / ||dtheta_l|| (supply per unit displacement). Reported every 'every' steps.
Problems: L63 one-step map (H=160, depth 3, Delta 1.0, full batch) and the field problem (p=1, minibatch 100)."""
import sys, os, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_default_dtype(torch.float64); torch.set_num_threads(int(os.environ.get('SEGFLOW_THREADS', '16'))); torch.set_num_interop_threads(1); torch.manual_seed(0)
def measure(params, out, r, names):
    g = torch.autograd.grad((out*r.detach()).sum(), params, create_graph=True)
    q = sum((gi**2).sum() for gi in g)/float((r**2).sum().detach())
    gq = torch.autograd.grad(q, params, retain_graph=True)
    return float(q), gq
def report(tag, step, E, q, gq, dth, names):
    contrib = np.array([float((a*b).sum()) for a, b in zip(gq, dth)]); nd = np.array([float(b.norm()) for b in dth])
    share = contrib/np.abs(contrib).sum(); sens = contrib/np.maximum(nd, 1e-300)
    print(f'{tag} step {step:5d} E={E:.4f} q={q:.3e} dq/step={contrib.sum():+.2e} | share by layer: ' + ' '.join(f'{n}:{s:+.2f}' for n, s in zip(names, share)) + ' | per unit displacement: ' + ' '.join(f'{s:+.1e}' for s in sens), flush=True)
# ---- L63
from l63_design import make_data_stats
from l63test_torch import MLP
H, depth, Delta, n = 160, 3, 1.0, 200
Xr, Yr = make_data_stats(n, Delta, 1); mu, sd = Xr.mean(0), Xr.std(0); X = torch.tensor((Xr-mu)/sd); Y = torch.tensor((Yr-mu)/sd).reshape(-1)
net = MLP(3, H, depth, 3, seed=1); Ws = [torch.nn.Parameter(w.clone()) for w in net.W]; bs = [torch.nn.Parameter(b.clone()) for b in net.b]
names = [f'W{l}' for l in range(depth+1)]; opt = torch.optim.Adam(Ws+bs, lr=1e-2); steps = 2000
def fwd():
    a = X
    for l in range(depth): a = torch.tanh(a @ Ws[l].T + bs[l])
    return (a @ Ws[-1].T + bs[-1]).reshape(-1)
print('L63 H=160 depth 3 Delta 1.0, Adam lr 1e-2 (decayed); layers W0=input, W1,W2 hidden, W3=output (weights only)')
for step in range(1, steps+1):
    for grp in opt.param_groups: grp['lr'] = 1e-2*(1 - (step-1)/steps)
    opt.zero_grad(); out = fwd(); r = out - Y; L = 0.5*torch.dot(r, r)/n
    if step % 100 == 1:
        q, gq = measure(Ws, out, r, names); old = [w.detach().clone() for w in Ws]
    L.backward(); opt.step()
    if step % 100 == 1:
        dth = [w.detach() - o for w, o in zip(Ws, old)]; report('L63  ', step, float(torch.sqrt(2*L.detach())), q, gq, dth, names)
# ---- field
G = 16; d = G*G; ntr, b = 2000, 100; Hf = 512
kx = torch.fft.fftfreq(G)*G; KX, KY = torch.meshgrid(kx, kx, indexing='ij'); K2 = KX**2 + KY**2
z = torch.randn(ntr, G, G, dtype=torch.complex128); U = torch.fft.ifft2(z*(1 + K2)**-2.0).real; U = U/U.std()
v = torch.fft.ifft2(torch.fft.fft2(U)*K2).real; v = v/v.std(); Yf = torch.tanh(2*v).reshape(ntr, -1); Xf = U.reshape(ntr, -1)
torch.manual_seed(1); netf = torch.nn.Sequential(torch.nn.Linear(d, Hf), torch.nn.GELU(), torch.nn.Linear(Hf, Hf), torch.nn.GELU(), torch.nn.Linear(Hf, d))
mats = [p for p in netf.parameters() if p.ndim == 2]; namesf = ['W0', 'W1', 'W2']; optf = torch.optim.AdamW(netf.parameters(), lr=3e-3, weight_decay=0.01); stepsf = 3000
g_ = torch.Generator().manual_seed(0); perm = torch.randperm(ntr, generator=g_); pos = 0
print('field p=1, MLP 256-512-512-256, AdamW lr 3e-3 (decayed), minibatch 100')
for step in range(1, stepsf+1):
    for grp in optf.param_groups: grp['lr'] = 3e-3*(1 - (step-1)/stepsf)
    if pos + b > ntr: perm = torch.randperm(ntr, generator=g_); pos = 0
    idx = perm[pos:pos+b]; pos += b
    optf.zero_grad(); out = netf(Xf[idx]); r = out - Yf[idx]; L = 0.5*torch.mean(r**2)*d
    if step % 150 == 1:
        q, gq = measure(mats, out, r, namesf); old = [w.detach().clone() for w in mats]
    L.backward(); optf.step()
    if step % 150 == 1:
        dth = [w.detach() - o for w, o in zip(mats, old)]
        with torch.no_grad(): E = float(torch.sqrt(torch.mean((netf(Xf) - Yf)**2)))
        report('field', step, E, q, gq, dth, namesf)
