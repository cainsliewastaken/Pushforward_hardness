"""Band-wise (output-roughness) preconditioned Adam on the field problem. The batch residual is split into k wavenumber
shells of the output grid; one backward pass per band gives g_j = J^T r_j; the band's effective eigenvalue is
q_j = ||g_j||^2/(||r_j||^2) (EMA-smoothed); the combined direction sum_j w_j g_j with w_j ∝ 1/q_j (normalised to mean 1,
capped at CAP) is fed to AdamW as the gradient. k=1 is plain AdamW. Cost per step: 1 fwd + k bwd = 1 + 2k fwd-eq vs 3."""
import sys, os, math, numpy as np, torch
torch.set_default_dtype(torch.float64); torch.set_num_threads(int(os.environ.get('SEGFLOW_THREADS', '16'))); torch.set_num_interop_threads(1); torch.manual_seed(0)
G = 16; d = G*G; ntr, nte, b = 2000, 500, 100; steps_max = int(os.environ.get('BP_STEPS', '3000')); H = 512; WD = 0.01
CAP = float(os.environ.get('BP_CAP', '10')); EMA = 0.9
kx = torch.fft.fftfreq(G)*G; KX, KY = torch.meshgrid(kx, kx, indexing='ij'); K2 = KX**2 + KY**2; KR = torch.sqrt(K2)
def grf(m):
    z = torch.randn(m, G, G, dtype=torch.complex128); u = torch.fft.ifft2(z*(1 + K2)**-2.0).real; return u/u.std()
def target(u, p):
    uh = torch.fft.fft2(u); v = torch.fft.ifft2(uh*(K2**p)).real if p > 0 else u.clone(); v = v/v.std(); return torch.tanh(2*v)
U = grf(ntr+nte)
def band_masks(k):
    edges = np.linspace(0, G/2*math.sqrt(2)+1e-9, k+1); edges[0] = -1
    return [((KR > edges[j]) & (KR <= edges[j+1])) for j in range(k)]
def split_bands(r, masks):   # r: (b, d) -> list of band components (b, d), exact decomposition
    rh = torch.fft.fft2(r.reshape(-1, G, G)); return [torch.fft.ifft2(rh*m).real.reshape(r.shape[0], -1) for m in masks]
def make_net():
    torch.manual_seed(1); return torch.nn.Sequential(torch.nn.Linear(d, H), torch.nn.GELU(), torch.nn.Linear(H, H), torch.nn.GELU(), torch.nn.Linear(H, d))
def run(k, lr, steps, Xtr, Ytr, Xte, Yte):
    net = make_net(); ps = list(net.parameters()); P = sum(p.numel() for p in ps); opt = torch.optim.AdamW(ps, lr=lr, weight_decay=WD)
    masks = band_masks(k) if k > 1 else None; qema = None
    g_ = torch.Generator().manual_seed(0); perm = torch.randperm(ntr, generator=g_); pos = 0; flops = 0.0
    for step in range(1, steps+1):
        for grp in opt.param_groups: grp['lr'] = lr*(1 - (step-1)/steps)
        if pos + b > ntr: perm = torch.randperm(ntr, generator=g_); pos = 0
        idx = perm[pos:pos+b]; pos += b
        opt.zero_grad(); out = net(Xtr[idx]); r = out - Ytr[idx]; flops += 2*b*P
        if k == 1:
            L = 0.5*torch.sum(r**2)/b; L.backward(); flops += 2*2*b*P
        else:
            bands = split_bands(r.detach(), masks); gs = []; qs = []
            for rj in bands:
                gj = torch.autograd.grad((out*rj).sum()/b, ps, retain_graph=True); flops += 2*2*b*P
                gs.append(gj); n2 = float((rj**2).sum())/b; qs.append(float(sum((x**2).sum() for x in gj))/max(n2, 1e-300) if n2 > 1e-12 else float('nan'))
            qs = np.array(qs); qema = qs if qema is None else np.where(np.isfinite(qs), EMA*qema + (1-EMA)*qs, qema)
            w = 1.0/np.maximum(qema, 1e-300); w = np.where(np.isfinite(w), w, 0.0); w = w/np.mean(w[w > 0]); w = np.minimum(w, CAP)
            for p_i, p in enumerate(ps): p.grad = sum(float(w[j])*gs[j][p_i] for j in range(k))
        opt.step()
        if not math.isfinite(float((r**2).sum())): return (float('inf'), float('inf'), flops/(2*ntr*P), None)
    with torch.no_grad(): return (float(torch.sqrt(torch.mean((net(Xtr)-Ytr)**2))), float(torch.sqrt(torch.mean((net(Xte)-Yte)**2))), flops/(2*ntr*P), (np.round(w, 2) if k > 1 else None))
print(f'field problem, batch {b}, WD {WD}, linear decay, {steps_max} steps; train/test RMSE [FLOP-eq] (best lr of 1e-3,3e-3,1e-2); band weights at the end in braces; cap {CAP}')
for p in (0.0, 1.0):
    Yall = target(U, p); Xtr, Xte = U[:ntr].reshape(ntr, -1), U[ntr:].reshape(nte, -1); Ytr, Yte = Yall[:ntr].reshape(ntr, -1), Yall[ntr:].reshape(nte, -1)
    rows = []
    for k in (1, 3, 5):
        best = min((run(k, lr, steps_max, Xtr, Ytr, Xte, Yte), lr) for lr in (1e-3, 3e-3, 1e-2))
        rows.append(f'k={k}: {best[0][0]:.4f}/{best[0][1]:.4f} [{best[0][2]:.0f}] lr {best[1]:g}' + (f' w={best[0][3]}' if best[0][3] is not None else ''))
    eq = min((run(1, lr, int(steps_max*11/3), Xtr, Ytr, Xte, Yte), lr) for lr in (1e-3, 3e-3, 1e-2))
    print(f'p={p} | ' + ' | '.join(rows) + f' | AdamW at equal FLOPs to k=5 ({int(steps_max*11/3)} steps): {eq[0][0]:.4f}/{eq[0][1]:.4f} [{eq[0][2]:.0f}]', flush=True)
