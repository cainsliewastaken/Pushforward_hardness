"""Roughness-supply objective on the field problem (minibatch): AdamW on F = L - beta L.detach() log q vs plain AdamW,
at equal steps and at equal FLOPs (plain AdamW gets 7/3 x the steps). Same data/model as muon_field.py."""
import sys, os, math, numpy as np, torch
torch.set_default_dtype(torch.float64); torch.set_num_threads(int(os.environ.get('SEGFLOW_THREADS', '16'))); torch.set_num_interop_threads(1); torch.manual_seed(0)
G = 16; d = G*G; ntr, nte, b = 2000, 500, 100; steps_max = int(os.environ.get('SF_STEPS', '3000')); H = 512; WD = 0.01
kx = torch.fft.fftfreq(G)*G; KX, KY = torch.meshgrid(kx, kx, indexing='ij'); K2 = KX**2 + KY**2
def grf(m):
    z = torch.randn(m, G, G, dtype=torch.complex128); u = torch.fft.ifft2(z*(1 + K2)**-2.0).real; return u/u.std()
def target(u, p):
    uh = torch.fft.fft2(u); v = torch.fft.ifft2(uh*(K2**p)).real if p > 0 else u.clone(); v = v/v.std(); return torch.tanh(2*v)
U = grf(ntr+nte)
def make_net():
    torch.manual_seed(1); return torch.nn.Sequential(torch.nn.Linear(d, H), torch.nn.GELU(), torch.nn.Linear(H, H), torch.nn.GELU(), torch.nn.Linear(H, d))
def run(beta, lr, steps, Xtr, Ytr, Xte, Yte):
    net = make_net(); ps = list(net.parameters()); opt = torch.optim.AdamW(ps, lr=lr, weight_decay=WD)
    g_ = torch.Generator().manual_seed(0); perm = torch.randperm(ntr, generator=g_); pos = 0
    for step in range(1, steps+1):
        for grp in opt.param_groups: grp['lr'] = lr*(1 - (step-1)/steps)
        if pos + b > ntr: perm = torch.randperm(ntr, generator=g_); pos = 0
        idx = perm[pos:pos+b]; pos += b
        opt.zero_grad(); out = net(Xtr[idx]); r = out - Ytr[idx]; L = 0.5*torch.mean(r**2)*d
        if beta > 0:
            g = torch.autograd.grad((out*r.detach()).sum(), ps, create_graph=True)
            q = sum((gi**2).sum() for gi in g)/float((r**2).sum().detach())
            F = L - beta*L.detach()*torch.log(q)
        else: F = L
        F.backward(); opt.step()
        if not math.isfinite(float(L)): return (float('inf'), float('inf'))
    with torch.no_grad(): return (float(torch.sqrt(torch.mean((net(Xtr)-Ytr)**2))), float(torch.sqrt(torch.mean((net(Xte)-Yte)**2))))
print(f'field problem, batch {b}, WD {WD}, linear decay; train/test RMSE, best lr of (1e-3, 3e-3). beta=0 is AdamW at equal steps; last column AdamW at equal FLOPs ({int(steps_max*7/3)} steps).')
for p in (0.0, 1.0):
    Yall = target(U, p); Xtr, Xte = U[:ntr].reshape(ntr, -1), U[ntr:].reshape(nte, -1); Ytr, Yte = Yall[:ntr].reshape(ntr, -1), Yall[ntr:].reshape(nte, -1)
    cells = []
    for beta in (0.0, 0.3, 1.0):
        best = min((run(beta, lr, steps_max, Xtr, Ytr, Xte, Yte), lr) for lr in (1e-3, 3e-3)); cells.append(best)
    eq = min((run(0.0, lr, int(steps_max*7/3), Xtr, Ytr, Xte, Yte), lr) for lr in (1e-3, 3e-3))
    print(f'p={p} | ' + '  '.join(f'b={bt}: {c[0][0]:.4f}/{c[0][1]:.4f}(lr {c[1]:g})' for bt, c in zip((0.0, 0.3, 1.0), cells)) + f' | AdamW equal FLOPs: {eq[0][0]:.4f}/{eq[0][1]:.4f}', flush=True)
