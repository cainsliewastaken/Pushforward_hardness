"""Width scan at fixed data (n=200, Delta=0.5, depth 2): FLOP-eq to reach a common target error for SegFlow (explicit
layer-structured kernel + momentum 0.6) and full-batch Adam (lr sweep). Also segment count, mean segment displacement."""
import sys, os, time, math, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
torch.set_default_dtype(torch.float64); torch.set_num_threads(int(os.environ.get('SEGFLOW_THREADS', '16'))); torch.set_num_interop_threads(1)
from l63_design import make_data_stats
from l63test_torch import MLP
from segflow import SegFlow
Delta, depth, seed, n = float(os.environ.get('WS_DELTA', '0.5')), int(os.environ.get('WS_DEPTH', '2')), 1, 200
E_target = float(os.environ.get('WS_TARGET', '0.05')); widths = [int(x) for x in os.environ.get('WS_WIDTHS', '48,160,512,1536').split(',')]
Xr, Yr = make_data_stats(n, Delta, 1); mu, sd = Xr.mean(0), Xr.std(0); X = torch.tensor((Xr-mu)/sd); Y = torch.tensor((Yr-mu)/sd).reshape(-1)
print(f'n={n} Delta={Delta} depth={depth} target E={E_target}; FLOP-eq in forward passes (fwd 1, bwd 2, jvp 2, vjp 3; explicit kernel and dense Lanczos counted in flops / (2nP))')
print('  H       P    | SegFlow: FLOP-eq  segments  kernel fwd-eq/seg  mean|dtheta|/seg  path | Adam best: FLOP-eq  steps  lr   | ratio Adam/SegFlow')
for H in widths:
    net = MLP(3, H, depth, 3, seed=seed); shapes = [w.shape for w in net.W] + [b.shape for b in net.b]
    theta0 = torch.cat([p.reshape(-1) for p in net.W + net.b]).clone(); P = theta0.numel()
    def unpack(theta):
        out = []; i = 0
        for s in shapes: k = int(np.prod(s)); out.append(theta[i:i+k].reshape(s)); i += k
        return out[:depth+1], out[depth+1:]
    def f(theta):
        Ws, bs = unpack(theta); a = X
        for l in range(depth): a = torch.tanh(a @ Ws[l].T + bs[l])
        return (a @ Ws[-1].T + bs[-1]).reshape(-1)
    def kernel_explicit(theta):
        Ws, bs = unpack(theta); A = [X]; a = X
        for l in range(depth): a = torch.tanh(a @ Ws[l].T + bs[l]); A.append(a)
        N = n*3; flops = 2*n*P; K = torch.zeros(N, N); ones3 = torch.ones(3, 3)
        Ab = torch.cat([A[-1], torch.ones(n, 1)], 1); GA = Ab @ Ab.T; flops += 2*n*n*Ab.shape[1]; K += torch.kron(GA, torch.eye(3)); flops += N*N
        D = torch.stack([Ws[-1][c].expand(n, -1) for c in range(3)], 1)
        for l in range(depth-1, -1, -1):
            D = D*(1 - A[l+1]**2)[:, None, :]; Dm = D.reshape(N, -1); GD = Dm @ Dm.T; flops += 2*N*N*Dm.shape[1]
            Ab = torch.cat([A[l], torch.ones(n, 1)], 1); GA = Ab @ Ab.T; flops += 2*n*n*Ab.shape[1]; K += GD*torch.kron(GA, ones3); flops += 2*N*N
            if l > 0: D = D @ Ws[l]; flops += 2*n*3*Ws[l].numel()
        return K/n, flops
    opt = SegFlow(f, theta0, Y, n=n, delta0=float(os.environ.get('WS_DELTA0', '0.25')), m_max=int(os.environ.get('WS_M', '30')), outer_momentum=0.6, kernel_fn=kernel_explicit); t0 = time.time()
    while opt.rms(opt.r) > E_target and opt.counts['segments'] < 3000: opt.step()
    fl_sf = 2*opt.counts['jvp'] + 3*opt.counts['vjp'] + opt.counts['forward'] + opt.flops_extra/(2*n*P)
    disp = [h['disp'] for h in opt.hist if h['accepted']]; kseg = opt.flops_extra/(2*n*P)/max(1, opt.counts['segments'])
    w_sf = time.time()-t0
    best = None
    for lr in ((3e-3, 1e-2, 3e-2) if os.environ.get('WS_SKIPADAM', '0') != '1' else ()):
        th = theta0.clone().requires_grad_(True); ad = torch.optim.Adam([th], lr=lr); hit = None
        for s in range(1, 20001):
            ad.zero_grad(); r = f(th) - Y; L = 0.5*torch.dot(r, r)/n; L.backward(); ad.step()
            if s % 25 == 0 and float(torch.sqrt(2*L.detach())) <= E_target: hit = s; break
        if hit and (best is None or hit < best[0]): best = (hit, lr)
    ad_fl = 3*best[0] if best else float('inf')
    print(f'  {H:5d} {P:8d} | {fl_sf:9.0f}  {opt.counts["segments"]:5d}      {kseg:6.1f}          {np.mean(disp):6.3f}       {np.sum(disp):6.1f} | {ad_fl:9.0f}  {best[0] if best else "-":>5}  {best[1] if best else "-":<5} | {ad_fl/fl_sf:5.2f}x   ({w_sf:.0f}s SegFlow)', flush=True)
