"""Minibatch segments (explicit batch kernel, trust radius, optional momentum) vs minibatch Adam, on a larger L63 dataset.
Both report full-training-set error vs FLOPs, in full-data forward-equivalents (flops / (2 n P)); evaluation not counted."""
import sys, os, time, math, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
torch.set_default_dtype(torch.float64); torch.set_num_threads(int(os.environ.get('SEGFLOW_THREADS', '16'))); torch.set_num_interop_threads(1)
from l63_design import make_data_stats
from l63test_torch import MLP
from segflow import lanczos
n = int(os.environ.get('MB_N', '4000')); b = int(os.environ.get('MB_B', '200')); H = int(os.environ.get('MB_H', '256')); depth = int(os.environ.get('MB_DEPTH', '2'))
Delta = float(os.environ.get('MB_DELTA', '0.5')); budget = float(os.environ.get('MB_BUDGET', '3000')); OM = float(os.environ.get('MB_OM', '0.0')); seed = 1; FRESH = os.environ.get('MB_FRESH', '0') == '1'   # trust test on an independent batch
Xr, Yr = make_data_stats(n, Delta, 1); mu, sd = Xr.mean(0), Xr.std(0); X = torch.tensor((Xr-mu)/sd); Y = torch.tensor((Yr-mu)/sd)
net = MLP(3, H, depth, 3, seed=seed); shapes = [w.shape for w in net.W] + [bb.shape for bb in net.b]
theta0 = torch.cat([p.reshape(-1) for p in net.W + net.b]).clone(); P = theta0.numel(); FWD = 2*n*P
def unpack(theta):
    out = []; i = 0
    for s in shapes: k = int(np.prod(s)); out.append(theta[i:i+k].reshape(s)); i += k
    return out[:depth+1], out[depth+1:]
def fwd(theta, Xb):
    Ws, bs = unpack(theta); a = Xb
    for l in range(depth): a = torch.tanh(a @ Ws[l].T + bs[l])
    return (a @ Ws[-1].T + bs[-1]).reshape(-1)
def kernel_explicit(theta, Xb):
    Ws, bs = unpack(theta); A = [Xb]; a = Xb; m = Xb.shape[0]
    for l in range(depth): a = torch.tanh(a @ Ws[l].T + bs[l]); A.append(a)
    N = m*3; flops = 2*m*P; K = torch.zeros(N, N); ones3 = torch.ones(3, 3)
    Ab = torch.cat([A[-1], torch.ones(m, 1)], 1); GA = Ab @ Ab.T; flops += 2*m*m*Ab.shape[1]; K += torch.kron(GA, torch.eye(3)); flops += N*N
    D = torch.stack([Ws[-1][c].expand(m, -1) for c in range(3)], 1)
    for l in range(depth-1, -1, -1):
        D = D*(1 - A[l+1]**2)[:, None, :]; Dm = D.reshape(N, -1); GD = Dm @ Dm.T; flops += 2*N*N*Dm.shape[1]
        Ab = torch.cat([A[l], torch.ones(m, 1)], 1); GA = Ab @ Ab.T; flops += 2*m*m*Ab.shape[1]; K += GD*torch.kron(GA, ones3); flops += 2*N*N
        if l > 0: D = D @ Ws[l]; flops += 2*m*3*Ws[l].numel()
    return K/m, flops
def full_err(theta):
    with torch.no_grad(): r = fwd(theta, X) - Y.reshape(-1); return float(torch.sqrt(r@r/n))
marks = [250, 500, 1000, 2000, 3000, 4000, 6000]; marks = [mk for mk in marks if mk <= budget]
def report(name, curve):
    print(f'  {name:28s}: ' + '  '.join(f'{mk}:{e:.4f}' for mk, e in curve), flush=True)
print(f'n={n} b={b} H={H} depth={depth} Delta={Delta} P={P}; full-train error at FLOP budgets (full-data forward-equivalents); init E={full_err(theta0):.4f}')
# ---- minibatch segments
theta = theta0.clone(); delta = 0.25; flops = 0.0; curve = []; mi = 0; prev = None; seg = 0; rej = 0; g = torch.Generator().manual_seed(0); perm = torch.randperm(n, generator=g); pos = 0
while flops/FWD < budget:
    if pos + b > n: perm = torch.randperm(n, generator=g); pos = 0
    idx = perm[pos:pos+b]; pos += b; Xb, yb = X[idx], Y[idx].reshape(-1)
    fb = lambda th: fwd(th, Xb)
    r = fb(theta) - yb; flops += 2*b*P
    K, fl = kernel_explicit(theta, Xb); flops += fl; N = 3*b
    def matvec(u):
        global flops; flops += 2*N*N; return K @ u
    Q, T = lanczos(matvec, r, 30); th, S = torch.linalg.eigh(T); th = th.clamp(min=0); c = S.T @ (Q.T @ r)
    def coeffs(t):
        z = th*t; return torch.where(z > 1e-12, (1-torch.exp(-z))/torch.where(z > 1e-12, z, 1.0), 1-z/2)*t
    disp = lambda t: float(torch.sqrt(torch.dot(coeffs(t)*c, th*(coeffs(t)*c))/b))
    lo, hi = 1e-12, 1.0
    while disp(hi) < delta and hi < 1e30: hi *= 4
    for _ in range(80): t = math.sqrt(lo*hi); lo, hi = ((t, hi) if disp(t) < delta else (lo, t))
    t = lo; u = Q @ (S @ (coeffs(t)*c)); _, pull = torch.func.vjp(fb, theta); dtheta = -pull(u)[0]/b; flops += 3*2*b*P
    if OM and prev is not None: dtheta = dtheta + OM*prev
    if FRESH:   # predicted (linear) and actual decrease on an independent batch: unbiased for the full loss, rejects batch-overfitting steps
        if pos + b > n: perm = torch.randperm(n, generator=g); pos = 0
        idx2 = perm[pos:pos+b]; pos += b; Xf, yf = X[idx2], Y[idx2].reshape(-1); ff = lambda th: fwd(th, Xf)
        rf, jd = torch.func.jvp(ff, (theta,), (dtheta,)); rf = rf - yf; flops += 2*2*b*P
        rf_new = ff(theta + dtheta) - yf; flops += 2*b*P
        dec_pred = float(rf@rf - (rf+jd)@(rf+jd)); dec_act = float(rf@rf - rf_new@rf_new)
    else:
        r_pred = Q @ (S @ ((1 - th*coeffs(t))*c)) + (r - Q @ (Q.T @ r))
        r_new = fb(theta + dtheta) - yb; flops += 2*b*P
        dec_pred = float(r@r - r_pred@r_pred); dec_act = float(r@r - r_new@r_new)
    rho = dec_act/dec_pred if dec_pred > 0 else (1.0 if dec_act >= 0 else -1.0)
    if rho > 0: theta = theta + dtheta; prev = dtheta
    else: rej += 1; prev = None
    if rho > 0.75: delta = min(delta*1.6, 1e3)
    elif rho < 0.25: delta *= 0.5
    seg += 1
    while mi < len(marks) and flops/FWD >= marks[mi]: curve.append((marks[mi], full_err(theta))); mi += 1
report(f'segments b={b} om={OM} fresh={FRESH} ({seg} seg, {rej} rej)', curve)
# ---- minibatch Adam
for lr in ((1e-3, 3e-3, 1e-2) if os.environ.get('MB_SKIPADAM', '0') != '1' else ()):
    th = theta0.clone().requires_grad_(True); ad = torch.optim.Adam([th], lr=lr); flops = 0.0; curve = []; mi = 0; g = torch.Generator().manual_seed(0); perm = torch.randperm(n, generator=g); pos = 0; step = 0
    while flops/FWD < budget:
        if pos + b > n: perm = torch.randperm(n, generator=g); pos = 0
        idx = perm[pos:pos+b]; pos += b
        ad.zero_grad(); r = fwd(th, X[idx]) - Y[idx].reshape(-1); L = 0.5*torch.dot(r, r)/b; L.backward(); ad.step(); flops += 3*2*b*P; step += 1
        while mi < len(marks) and flops/FWD >= marks[mi]: curve.append((marks[mi], full_err(th.detach()))); mi += 1
    report(f'Adam lr={lr:g} b={b} ({step} steps)', curve)
