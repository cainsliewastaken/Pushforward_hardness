"""Adam vs SegFlow to error targets below GD's final error (E_GD(T)/3, /10, /30): FLOP-eq to reach each."""
import sys, os, re, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_default_dtype(torch.float64); torch.set_num_threads(8); torch.set_num_interop_threads(1)
from l63_design import make_data_stats
from l63test_torch import MLP
from segflow import SegFlow
tag = sys.argv[1]; maxsteps = int(sys.argv[2])
mm = re.match(r'H(\d+)_D([\d.]+)_depth(\d+)_s(\d+)', tag); H, Delta, depth, seed = int(mm[1]), float(mm[2]), int(mm[3]), int(mm[4])
d = np.load(f'../runs/heldout/des_{tag}.npy', allow_pickle=True).item(); n = d['n']; E_T = float(d['E'][-1]); targets = [E_T/30, E_T/100, E_T/300]
Xr, Yr = make_data_stats(n, Delta, 1); mu, sd = Xr.mean(0), Xr.std(0); X = torch.tensor((Xr-mu)/sd); Y = torch.tensor((Yr-mu)/sd).reshape(-1)
net = MLP(3, H, depth, 3, seed=seed); shapes = [w.shape for w in net.W] + [b.shape for b in net.b]
theta0 = torch.cat([p.reshape(-1) for p in net.W + net.b]).clone()
def unpack(theta):
    out = []; i = 0
    for s in shapes: k = int(np.prod(s)); out.append(theta[i:i+k].reshape(s)); i += k
    return out[:depth+1], out[depth+1:]
def f(theta):
    Ws, bs = unpack(theta); a = X
    for l in range(depth): a = torch.tanh(a @ Ws[l].T + bs[l])
    return (a @ Ws[-1].T + bs[-1]).reshape(-1)
fmt = lambda hs: '  '.join('%9s' % (str(int(x)) if x else '   -') for x in hs)
# SegFlow
opt = SegFlow(f, theta0, Y, n=n, delta0=0.25, m_max=30); fl = lambda: 2*opt.counts['jvp'] + 3*opt.counts['vjp'] + opt.counts['forward']
hit = [None]*3; t0 = time.time()
while fl() < 3*maxsteps and hit[-1] is None and opt.counts['segments'] < 5000:
    opt.step(); e = opt.rms(opt.r)
    for i, t in enumerate(targets):
        if hit[i] is None and e <= t: hit[i] = fl()
print(f'{tag}: E_GD(T)={E_T:.4f}; targets E/30, E/100, E/300 = {targets[0]:.4f} {targets[1]:.4f} {targets[2]:.5f}; FLOP-eq to reach:')
print('   SegFlow      ', fmt(hit), f'  (final E {opt.rms(opt.r):.5f}, {opt.counts["segments"]} segments, {time.time()-t0:.0f}s)', flush=True)
# Adam, several lrs, also with a cosine-free simple decay variant (lr halves when no progress for 2000 steps)
for lr in (3e-3,):
    for decay in (False, True):
        th = theta0.clone().requires_grad_(True); ad = torch.optim.Adam([th], lr=lr); h = [None]*3; best = 1e9; since = 0; t0 = time.time(); cur = lr
        for s in range(1, maxsteps+1):
            ad.zero_grad(); r = f(th) - Y; L = 0.5*torch.dot(r, r)/n; L.backward(); ad.step()
            if s % 100 == 0:
                e = float(torch.sqrt(2*L.detach()))
                for i, t in enumerate(targets):
                    if h[i] is None and e <= t: h[i] = 3*s
                if e < best*0.999: best = e; since = 0
                else: since += 100
                if decay and since >= 3000: cur *= 0.5; since = 0; [g.__setitem__('lr', cur) for g in ad.param_groups]
                if h[-1]: break
        print(f'   Adam lr={lr:g}{" +decay" if decay else "       "}', fmt(h), f'  (best E {best:.5f}, {time.time()-t0:.0f}s)', flush=True)
