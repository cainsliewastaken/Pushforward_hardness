"""Truncated spectral update on the field problem: per layer, momentum gradient M -> U_k V_k^T over its top-k singular
directions only (rank-k randomized SVD, O(H^2 k)); nothing is moved along the remaining directions. k = full is Muon.
Biases by Adam. Compared with AdamW at equal steps and equal FLOPs (FLOPs counted: fwd 1, bwd 2, SVD/NS flops/(2 b P))."""
import sys, os, math, numpy as np, torch
torch.set_default_dtype(torch.float64); torch.set_num_threads(int(os.environ.get('SEGFLOW_THREADS', '16'))); torch.set_num_interop_threads(1); torch.manual_seed(0)
G = 16; d = G*G; ntr, nte, b = 2000, 500, 100; steps_max = int(os.environ.get('TS_STEPS', '3000')); H = 512; WD = 0.01
kx = torch.fft.fftfreq(G)*G; KX, KY = torch.meshgrid(kx, kx, indexing='ij'); K2 = KX**2 + KY**2
def grf(m):
    z = torch.randn(m, G, G, dtype=torch.complex128); u = torch.fft.ifft2(z*(1 + K2)**-2.0).real; return u/u.std()
def target(u, p):
    uh = torch.fft.fft2(u); v = torch.fft.ifft2(uh*(K2**p)).real if p > 0 else u.clone(); v = v/v.std(); return torch.tanh(2*v)
U = grf(ntr+nte)
def make_net():
    torch.manual_seed(1); return torch.nn.Sequential(torch.nn.Linear(d, H), torch.nn.GELU(), torch.nn.Linear(H, H), torch.nn.GELU(), torch.nn.Linear(H, d))
def ns5(X0, steps=5, eps=1e-7):
    a, bb, c = (3.4445, -4.7750, 2.0315); X = X0/(X0.norm()+eps); tr = X.shape[0] > X.shape[1]
    if tr: X = X.T
    m, k = X.shape; fl = 0
    for _ in range(steps): A = X @ X.T; B = bb*A + c*(A @ A); X = a*X + B @ X; fl += 4*m*m*k + 2*m**3
    return (X.T if tr else X), fl
def trunc_orth(M, k, gen):
    """top-k singular directions of M via randomized subspace iteration (2 passes); returns U_k V_k^T and flops"""
    m, n_ = M.shape; Om = torch.randn(n_, k+8, generator=gen); Y = M @ Om; Q, _ = torch.linalg.qr(Y); Y = M @ (M.T @ Q); Q, _ = torch.linalg.qr(Y)
    B = Q.T @ M; Ub, S, Vh = torch.linalg.svd(B, full_matrices=False); Uk = Q @ Ub[:, :k]; fl = 2*m*n_*(k+8)*3 + 2*m*(k+8)**2*2 + 2*(k+8)*n_*(k+8)
    return Uk @ Vh[:k], fl
def run(method, k, lr, steps, Xtr, Ytr, Xte, Yte):
    net = make_net(); mats = [p for p in net.parameters() if p.ndim == 2]; vecs = [p for p in net.parameters() if p.ndim == 1]; P = sum(p.numel() for p in net.parameters())
    if method == 'adam': opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WD); bufs = None
    else: opt = torch.optim.Adam(vecs, lr=3e-3); bufs = [torch.zeros_like(w) for w in mats]
    g_ = torch.Generator().manual_seed(0); gen = torch.Generator().manual_seed(1); perm = torch.randperm(ntr, generator=g_); pos = 0; flops = 0.0
    for step in range(1, steps+1):
        fac = 1 - (step-1)/steps
        for grp in opt.param_groups: grp['lr'] = (lr if bufs is None else 3e-3)*fac
        if pos + b > ntr: perm = torch.randperm(ntr, generator=g_); pos = 0
        idx = perm[pos:pos+b]; pos += b
        opt.zero_grad(); loss = 0.5*torch.mean((net(Xtr[idx]) - Ytr[idx])**2)*d; loss.backward(); flops += 3*2*b*P
        if bufs is None: opt.step()
        else:
            opt.step()
            with torch.no_grad():
                for w, buf in zip(mats, bufs):
                    gr = w.grad; buf.mul_(0.95).add_(gr); gn = gr + 0.95*buf
                    if method == 'muon': O, fl = ns5(gn)
                    else: O, fl = trunc_orth(gn, k, gen)
                    flops += fl; w.mul_(1 - lr*fac*WD); w.add_(O, alpha=-lr*fac*max(1.0, w.shape[0]/w.shape[1])**0.5)
        if not math.isfinite(float(loss)): return (float('inf'), float('inf'), flops/(2*ntr*P))
    with torch.no_grad(): return (float(torch.sqrt(torch.mean((net(Xtr)-Ytr)**2))), float(torch.sqrt(torch.mean((net(Xte)-Yte)**2))), flops/(2*ntr*P))
print(f'field problem, batch {b}, WD {WD}, linear decay, {steps_max} steps; train/test RMSE [FLOP-eq] at best lr. trunc-k = rank-k truncated spectral update; muon = full.')
for p in (0.0, 1.0):
    Yall = target(U, p); Xtr, Xte = U[:ntr].reshape(ntr, -1), U[ntr:].reshape(nte, -1); Ytr, Yte = Yall[:ntr].reshape(ntr, -1), Yall[ntr:].reshape(nte, -1)
    rows = []
    for method, k, lrs in (('adam', 0, (1e-3, 3e-3)), ('muon', 0, (1e-3, 3e-3, 1e-2)), ('trunc', 8, (3e-3, 1e-2, 3e-2)), ('trunc', 32, (3e-3, 1e-2, 3e-2)), ('trunc', 128, (1e-3, 3e-3, 1e-2))):
        best = min((run(method, k, lr, steps_max, Xtr, Ytr, Xte, Yte), lr) for lr in lrs)
        rows.append(f'{method}{k if k else ""}: {best[0][0]:.4f}/{best[0][1]:.4f} [{best[0][2]:.0f}] lr {best[1]:g}')
    eq = min((run('adam', 0, lr, int(steps_max*7/3), Xtr, Ytr, Xte, Yte), lr) for lr in (1e-3, 3e-3))
    print(f'p={p} | ' + ' | '.join(rows) + f' | adam equal-FLOPs(7000): {eq[0][0]:.4f}/{eq[0][1]:.4f} [{eq[0][2]:.0f}]', flush=True)
