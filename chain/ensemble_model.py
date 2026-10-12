"""Statistical model with persistent virtual Gaussian units (no weight matrices; layers couple through Grams and u).
Init: per layer, NV virtual units (z_q, m_q) sampled from the measured init covariances (Zc_l, D_{l+1}/fan_out).
Stage: tables from the ensemble -> kernel (measured init spectrum x ensemble lift ratio by rank, in the ensemble's
eigenbasis) -> residual decays by kernel regression over the stage -> du = int E dt / n -> coherent maps applied to
every unit with du and the CURRENT Grams -> next stage.  usage: python ensemble_model.py H DEPTH SEED DELTA OUTSCALE T NSTAGE NV OUT.npz"""
import numpy as np, sys, os, torch, time
os.environ['L63_OUTSCALE'] = sys.argv[5]
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
torch.set_default_dtype(torch.float64)
H, depth, seed, Delta, sc, T, NST, NV, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), sys.argv[9]
EPS = float(os.environ.get('EPS', '0'))   # >0: adaptive stage budget on the RMS integrated-residual increment
n = int(os.environ.get('N', '200')); ny = 3; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV); Yv = Y.reshape(-1)
net = MLP(3, H, depth, 3, seed=seed)
if sc != 1.0: net.forward(X)
L = len(net.W); fan_out = [W.shape[0] for W in net.W]
out0, Acts = net.forward(X); Zs = [Acts[l] @ net.W[l].T + net.b[l] for l in range(L)]
deltas = [torch.zeros(n, fan_out[l], ny, device=DEV) for l in range(L)]
for c in range(ny):
    dl = torch.zeros(n, ny, device=DEV); dl[:, c] = 1.; deltas[L-1][:, :, c] = dl; dl = dl @ net.W[-1]
    for l in range(L-2, -1, -1):
        dl = dl*(1-Acts[l+1]**2); deltas[l][:, :, c] = dl
        if l > 0: dl = dl @ net.W[l]
A0 = [Acts[l] @ Acts[l].T + 1. for l in range(L)]; D0 = [torch.einsum('ipc,jpd->icjd', deltas[l], deltas[l]) for l in range(L)]; Zc0 = [Zs[l] @ Zs[l].T/fan_out[l] for l in range(L)]
E = out0.reshape(-1) - Yv
def assemble(A, D):
    Kf = torch.zeros(n, ny, n, ny, device=DEV)
    for l in range(L): Kf += A[l][:, None, :, None]*D[l]
    return Kf.reshape(n*ny, n*ny)/n
K0 = assemble(A0, D0); w0, V0 = torch.linalg.eigh(K0); w0 = w0.flip(0); V0 = V0.flip(1)
gen = torch.Generator(device=DEV).manual_seed(seed)
def gp(C, N):
    w, V = torch.linalg.eigh(C); return (torch.randn(N, C.shape[0], generator=gen, device=DEV)*torch.sqrt(torch.clamp(w, min=0))) @ V.T
REAL = int(os.environ.get('REAL', '0'))      # >0: build the ensemble from REAL random initialisations of the architecture
Zu = []; Mu = []
if REAL > 0:
    zs = [[] for _ in range(L)]; ms = [[] for _ in range(L)]; Wmem = []; bmem = []
    OWN = int(os.environ.get('OWN', '0'))     # 1: the first member is the run's OWN initialization (seed); self-test of maps + assembly without sampling error
    for k in range(REAL):
        netk = MLP(3, H, depth, 3, seed=(seed if (OWN and k == 0) else 1000+seed*100+k))
        if sc != 1.0: netk.forward(X)
        ok_, Ak = netk.forward(X); Zk = [Ak[l] @ netk.W[l].T + netk.b[l] for l in range(L)]; Wmem.append([W.clone() for W in netk.W]); bmem.append([b.clone() for b in netk.b])
        for l in range(L-1):
            # incoming backward signal m_q^c(x) = (W_{l+1}^T delta_{l+1}^c(x))_q, computed from this init's backward pass
            mk = torch.zeros(n, fan_out[l], ny, device=DEV)
            for c in range(ny):
                dl = torch.zeros(n, ny, device=DEV); dl[:, c] = 1.; dl = dl @ netk.W[-1]
                for l2 in range(L-2, l, -1):
                    dl = dl*(1-Ak[l2+1]**2); dl = dl @ netk.W[l2]
                mk[:, :, c] = dl
            zs[l].append(Zk[l].T); ms[l].append(mk.permute(1, 0, 2))
    for l in range(L-1): Zu.append(torch.cat(zs[l], 0)); Mu.append(torch.cat(ms[l], 0))
    Zu.append(None); Mu.append(None); NV = Zu[0].shape[0]
    PROP = int(os.environ.get('PROP', '0'))   # 1: each member keeps its own hidden-hidden init weights (block-diagonal over members) so that W*da (forward) and W^T*d(delta) (backward) propagate
    if PROP:
        Wt = [None] + [torch.block_diag(*[Wk[l] for Wk in Wmem]) for l in range(1, L-1)]; bt = [None] + [torch.cat([bk[l] for bk in bmem]) for l in range(1, L-1)]
        bmask = torch.block_diag(*[torch.ones(H, H, device=DEV) for _ in range(REAL)])   # coherent weight increments stay within a member
        print(f'PROP=1: explicit hidden-hidden weights for interfaces {list(range(1, L-1))}, {REAL} member blocks', flush=True)
    print(f'real-init ensemble: {REAL} inits x width {H} = {NV} units per layer', flush=True)
else:
    PROP = 0
    for l in range(L):
        Zu.append(gp(Zc0[l], NV))
        Mu.append(gp((D0[l+1]/fan_out[l]).reshape(n*ny, n*ny), NV).reshape(NV, n, ny) if l < L-1 else None)
def tables():
    A = [A0[0]]; D = [None]*L
    for l in range(L-1):
        a = torch.tanh(Zu[l]); sp = 1-a**2; d = sp[:, :, None]*Mu[l]
        A.append(fan_out[l]*(a.T @ a)/NV + 1.); D[l] = fan_out[l]*torch.einsum('qxc,qyd->xcyd', d, d)/NV
    D[L-1] = D0[L-1]
    return A, D
A_v0, D_v0 = tables(); wv0 = torch.linalg.eigvalsh(assemble(A_v0, D_v0)).flip(0)
ADD = int(os.environ.get('ADD', '0'))        # 1: additive transfer K = K0_meas + (K_ens(t) - K_ens(0)) by rank (default 0: multiplicative ratio)
print('init spectrum by rank (measured w0 / ensemble wv0): ' + ' '.join(f'k{i+1}:{float(w0[i]):.3g}/{float(wv0[i]):.3g}' for i in (0, 9, 49, 99, 199, 399) if i < len(w0)), flush=True)
def kernel_now(A, D):
    Kv = assemble(A, D); wv, Vv = torch.linalg.eigh(Kv); wv = wv.flip(0); Vv = Vv.flip(1)
    if ADD:
        lam = w0 + torch.clamp(wv - wv0, min=0.0); ratio = lam/w0
    else:
        ratio = torch.clamp(wv/torch.clamp(wv0, min=1e-300), min=1.0); lam = w0*ratio
    return (Vv*lam) @ Vv.T, ratio       # measured init spectrum by rank, lifted by the ensemble (ratio or increment), in the ensemble's eigenbasis
OPEN = os.environ.get('OPENLOOP', '')       # path to an l63_late run (.npy with t, Evec, lam_eff): drive the maps with the MEASURED residual history
if OPEN:
    _r = np.load(OPEN, allow_pickle=True).item(); t_r = torch.tensor(np.asarray(_r['t']), device=DEV); E_r = torch.tensor(np.asarray(_r['Evec']), device=DEV); le_r = np.asarray(_r['lam_eff'])
    def E_real(tt):
        tt = min(max(tt, float(t_r[0])), float(t_r[-1])); j = int(torch.searchsorted(t_r, torch.tensor(tt, device=DEV)).clamp(1, len(t_r)-1)); f = (tt-float(t_r[j-1]))/max(float(t_r[j]-t_r[j-1]), 1e-300)
        return E_r[j-1]*(1-f) + E_r[j]*f
    print(f'open-loop drive from {OPEN}: {len(t_r)} points, E(T) meas {float(torch.sqrt(E_r[-1]@E_r[-1]/n)):.4f}', flush=True)
hist = []; t0 = time.time(); A_c, D_c = A_v0, D_v0; t = 0.; dt = 1e-3/float(w0[0]); nstage = 0; next_print = 1e-2/float(w0[0])
while t < T:
    K, ratio = kernel_now(A_c, D_c); w, V = torch.linalg.eigh(K); w = torch.clamp(w, min=0); a = V.T @ E
    Edrv = E_real(t) if OPEN else E
    if EPS > 0:   # choose dt so that RMS(int E dt)/n ~ EPS, using the current decay (closed form), capped at 20% of t and at T-t
        Erms = float(torch.sqrt(Edrv@Edrv/n)); dt = min(max(EPS/max(Erms, 1e-12), 1e-6), max(0.2*t, 1e-3/float(w0[0])), T-t)
    else:
        dt = min(dt*(T/(1e-3/float(w0[0])))**(1.0/NST), T-t)
    dec = torch.exp(-w*dt); intE = V @ (a*torch.where(w > 1e-300, (1-dec)/torch.clamp(w, min=1e-300), torch.full_like(w, dt))); E = V @ (a*dec)
    if OPEN:  # trapezoid of the measured residual over the stage
        intE = 0.5*dt*(Edrv + E_real(t+dt))
    du = (intE/n).reshape(n, ny); t += dt; nstage += 1
    if PROP:
        a_old = [torch.tanh(Zu[l]) for l in range(L-1)]; d_old = [(1-a_old[l]**2)[:, :, None]*Mu[l] for l in range(L-1)]
        # coherent weight increments (first order, stage-start states): input layer via s on z; hidden-hidden explicitly; output layer via g on m
        Zu[0] = Zu[0] - torch.einsum('qic,ic,ix->qx', d_old[0], du, A_c[0])
        for l in range(1, L-1):
            Wt[l] = Wt[l] - bmask*torch.einsum('qic,ic,pi->qp', d_old[l], du, a_old[l-1]); bt[l] = bt[l] - torch.einsum('qic,ic->q', d_old[l], du)
        for l in range(1, L-1):   # forward recompute with the UPDATED previous layer: includes W*da and dW*da
            Zu[l] = Wt[l] @ torch.tanh(Zu[l-1]) + bt[l][:, None]
        g = torch.einsum('idxc,id->ixc', D_c[L-1], du); Mu[L-2] = Mu[L-2] - torch.einsum('qi,ixc->qxc', a_old[L-2], g)
        for l in range(L-3, -1, -1):   # backward recompute through the explicit weights: includes W^T*d(delta)
            Mu[l] = torch.einsum('pq,pxc->qxc', Wt[l+1], (1-torch.tanh(Zu[l+1])**2)[:, :, None]*Mu[l+1])
    else:
        for l in range(L-1):
            a_l = torch.tanh(Zu[l]); d_l = (1-a_l**2)[:, :, None]*Mu[l]
            s_ = -torch.einsum('qic,ic,ix->qx', d_l, du, A_c[l]); g = torch.einsum('idxc,id->ixc', D_c[l+1], du)
            Mu[l] = Mu[l] - torch.einsum('qi,ixc->qxc', a_l, g); Zu[l] = Zu[l] + s_
    A_c, D_c = tables()
    if t >= next_print or t >= T:
        Ed = E_real(t) if OPEN else E; lam_eff = float(Ed @ (K @ Ed)/(Ed @ Ed)); lam_eff_meas = float(np.interp(t, np.asarray(_r['t']), le_r)) if OPEN else float('nan')
        hist.append((t, float(torch.sqrt(E@E/n)), float(ratio[99]), float(ratio[199]), float(ratio[0]), float(ratio[9]), float(ratio[49]), lam_eff, lam_eff_meas)); next_print *= 3.
        print(f"   t={t:9.3f} ||E||={hist[-1][1]:.4f} gain1={hist[-1][4]:6.2f} gain10={hist[-1][5]:6.2f} gain50={hist[-1][6]:7.1f} gain100={hist[-1][2]:8.1f} gain200={hist[-1][3]:9.1f} lam_eff={lam_eff:.3e}" + (f" meas={lam_eff_meas:.3e}" if OPEN else "") + f" stages={nstage} ({time.time()-t0:.0f}s)", flush=True)
if os.environ.get('DUMP', ''):
    torch.save(dict(t=t, Zu=[z.cpu() if z is not None else None for z in Zu], Mu=[m.cpu() if m is not None else None for m in Mu], K=kernel_now(A_c, D_c)[0].cpu(), E=E.cpu()), os.environ['DUMP'])
np.savez(out, hist=np.array(hist), cols=np.array(['t', 'E', 'gain100', 'gain200', 'gain1', 'gain10', 'gain50', 'lam_eff', 'lam_eff_meas']))
