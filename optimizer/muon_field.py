"""Synthetic field-to-field problem with a target-roughness knob: Muon vs Adam (minibatch), full-train and test RMSE
at equal step counts; FLOPs incl. Newton-Schulz also reported.  Input: Gaussian random fields on a 16x16 periodic
grid with spectrum (1+k^2)^-2.  Target: y = tanh(2 * N[(-Lap)^p u]) with N = unit-std normalisation; p controls the
output roughness (p=0 smooth pointwise map; p=1, 2 progressively rougher output fields)."""
import sys, os, time, math, numpy as np, torch
torch.set_default_dtype(torch.float64); torch.set_num_threads(int(os.environ.get('SEGFLOW_THREADS', '16'))); torch.set_num_interop_threads(1); torch.manual_seed(0)
G = 16; d = G*G; ntr, nte, b = 2000, 500, int(os.environ.get('MF_B', '100')); steps_max = int(os.environ.get('MF_STEPS', '6000')); H = int(os.environ.get('MF_H', '512')); WD = float(os.environ.get('MF_WD', '0.0')); DECAY = os.environ.get('MF_DECAY', '0') == '1'
kx = torch.fft.fftfreq(G)*G; KX, KY = torch.meshgrid(kx, kx, indexing='ij'); K2 = KX**2 + KY**2
def grf(m):
    z = torch.randn(m, G, G, dtype=torch.complex128); amp = (1 + K2)**-2.0; u = torch.fft.ifft2(z*amp).real
    return u/u.std()
def target(u, p):
    uh = torch.fft.fft2(u); v = torch.fft.ifft2(uh*(K2**p)).real if p > 0 else u.clone(); v = v/v.std()
    return torch.tanh(2*v)
def rough_frac(y):   # energy fraction at |k| > G/4 (upper half of the wavenumber range), averaged over samples
    yh = torch.fft.fft2(y); e = (yh.abs()**2); return float(e[:, torch.sqrt(K2) > G/4].sum()/e.sum())
U = grf(ntr+nte); checkpoints = [250, 500, 1000, 2000, 4000, 6000]; checkpoints = [c for c in checkpoints if c <= steps_max]
def ns5(X0, steps=5, eps=1e-7):
    a, bb, c = (3.4445, -4.7750, 2.0315); X = X0/(X0.norm()+eps); tr = X.shape[0] > X.shape[1]
    if tr: X = X.T
    m, k = X.shape; fl = 0
    for _ in range(steps):
        A = X @ X.T; B = bb*A + c*(A @ A); X = a*X + B @ X; fl += 4*m*m*k + 2*m**3
    return (X.T if tr else X), fl
def make_net():
    torch.manual_seed(1)
    return torch.nn.Sequential(torch.nn.Linear(d, H), torch.nn.GELU(), torch.nn.Linear(H, H), torch.nn.GELU(), torch.nn.Linear(H, d))
def run(opt_name, lr, Xtr, Ytr, Xte, Yte):
    net = make_net(); mats = [p for p in net.parameters() if p.ndim == 2]; vecs = [p for p in net.parameters() if p.ndim == 1]
    P = sum(p.numel() for p in net.parameters()); FWD = 2*ntr*P
    if opt_name == 'adam': opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WD); bufs = None
    else: opt = torch.optim.Adam(vecs, lr=3e-3); bufs = [torch.zeros_like(w) for w in mats]
    g = torch.Generator().manual_seed(0); perm = torch.randperm(ntr, generator=g); pos = 0; flops = 0.0; out = {}
    rmse = lambda Xs, Ys: float(torch.sqrt(torch.mean((net(Xs) - Ys)**2)))
    for step in range(1, steps_max+1):
        if pos + b > ntr: perm = torch.randperm(ntr, generator=g); pos = 0
        idx = perm[pos:pos+b]; pos += b
        fac = (1 - (step-1)/steps_max) if DECAY else 1.0
        for grp in opt.param_groups: grp['lr'] = (lr if bufs is None else 3e-3)*fac
        opt.zero_grad(); loss = 0.5*torch.mean((net(Xtr[idx]) - Ytr[idx])**2)*d; loss.backward(); flops += 3*2*b*P
        if bufs is None: opt.step()
        else:
            opt.step()
            with torch.no_grad():
                for w, buf in zip(mats, bufs):
                    gr = w.grad; buf.mul_(0.95).add_(gr); gn = gr + 0.95*buf; O, fl = ns5(gn); flops += fl
                    w.mul_(1 - lr*fac*WD); w.add_(O, alpha=-lr*fac*max(1.0, w.shape[0]/w.shape[1])**0.5)   # decoupled weight decay as in Muon-with-WD
        if step in checkpoints:
            with torch.no_grad(): out[step] = (rmse(Xtr, Ytr), rmse(Xte, Yte), flops/FWD)
        if not math.isfinite(float(loss)): break
    return out
print(f'weight decay {WD}, linear lr decay {DECAY}'); print(f'field problem: grid {G}x{G} (dout={d}), n_train={ntr}, batch={b}, MLP {d}-{H}-{H}-{d} GELU; entries: train RMSE / test RMSE at step checkpoints {checkpoints}; FLOP-eq (full-data forwards, incl. Newton-Schulz) in brackets at the last checkpoint')
for p in [float(x) for x in os.environ.get('MF_PS', '0,1,2').split(',')]:
    Yall = target(U, p); Xtr, Xte = U[:ntr].reshape(ntr, -1), U[ntr:].reshape(nte, -1); Ytr, Yte = Yall[:ntr].reshape(ntr, -1), Yall[ntr:].reshape(nte, -1)
    print(f'--- p={p}: target rough-energy fraction {rough_frac(Yall):.3f} (input {rough_frac(U):.3f}); target std {float(Yall.std()):.3f}', flush=True)
    for opt_name, lrs in (('adam', (1e-3, 3e-3)), ('muon', (1e-3, 3e-3, 1e-2))):
        best = None
        for lr in lrs:
            t0 = time.time(); out = run(opt_name, lr, Xtr, Ytr, Xte, Yte)
            if not out: continue
            last = out[max(out)]
            line = f'   {opt_name:4s} lr={lr:<6g} ' + '  '.join(f'{s}:{v[0]:.4f}/{v[1]:.4f}' for s, v in sorted(out.items())) + f'  [{last[2]:.0f}]  ({time.time()-t0:.0f}s)'
            print(line, flush=True)
