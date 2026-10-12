"""Heavy-ball momentum vs the gradient-flow reference on the L63 testbed (n = 200), in rescaled time.

Linearised prediction: with buf <- beta buf + g, theta <- theta - eta buf, a kernel mode with eta*lam < (1-sqrt beta)^2 is
overdamped and follows gradient flow with time step eta/(1-beta); stiffer modes oscillate and decay at ~(1-sqrt beta) per
step whatever lam; stability needs eta*lam1 < 2(1+beta). So heavy ball = gradient flow at t = eta*k/(1-beta) on the slow
modes. This script tests that on the nonlinear network: E(t), the kernel spectrum lam_k(t) (the lift), the residual's
energy per K(t) eigenmode, and the low-rank structure of the hidden-layer change, against the saved GF run.
usage: python momentum_modes.py TAG BETA C   (eta = C/lam1(t), re-set whenever eta*lam1 leaves [0.8C, 1.25C])
"""
import sys, os, time, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
torch.set_default_dtype(torch.float64); torch.set_num_threads(8); torch.set_num_interop_threads(1)
from l63_design import make_data_stats
from l63test_torch import MLP, kernel
import re
tag, beta, C = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
mm = re.match(r'H(\d+)_D([\d.]+)_depth(\d+)_s(\d+)', tag); H, Delta, depth, seed = int(mm[1]), float(mm[2]), int(mm[3]), int(mm[4])
d = np.load(f'../runs/heldout/des_{tag}.npy', allow_pickle=True).item(); n = d['n']
Xr, Yr = make_data_stats(n, Delta, d['dseed']); mu, sd = Xr.mean(0), Xr.std(0)
X = torch.tensor((Xr-mu)/sd); Y = torch.tensor((Yr-mu)/sd)
assert np.allclose(Y.numpy(), d['Y']), 'data mismatch with saved GF run'
net = MLP(3, H, depth, 3, seed=seed); th = net.pack().clone(); W0 = [w.clone() for w in net.W]
def measure(th):
    net.unpack(th); J, out = net.jacobian(X); E = (out-Y).reshape(-1); K = kernel(J, n)
    lam, V = torch.linalg.eigh(K); lam, V = lam.flip(0), V.flip(1)
    return float(torch.sqrt(E@E/n)), lam.numpy(), ((V.T@E)**2).numpy()
E0, lam0, _ = measure(th)
assert abs(E0 - d['E'][0]) < 1e-6*max(1, E0) or d['t'][0] > 0, (E0, d['E'][0])
grid = d['t'][d['t'] > 0]
buf = torch.zeros_like(th); t = 0.; k = 0; eta = C/lam0[0]; rows = []
t0w = time.time()
for target in grid:
    while t < target:
        net.unpack(th); out, A = net.forward(X); g = net.grad(X, (out-Y)/n, A)
        buf.mul_(beta).add_(g); th.sub_(eta*buf); t += eta/(1-beta); k += 1
        if not torch.isfinite(th).all(): print('diverged at t', t); sys.exit(1)
    E, lam, e2 = measure(th)
    if not (0.8*C <= eta*lam[0] <= 1.25*C): eta = C/lam[0]
    net.unpack(th); dW = net.W[1]-W0[1]; s = torch.linalg.svdvals(dW); top3 = float((s[:3]**2).sum()/(s**2).sum())
    rows.append(dict(t=t, k=k, E=E, lam=lam, e2=e2, eta=eta, top3=top3))
np.save(f'../runs/momentum/{tag}_b{beta}_c{C}.npy', dict(rows=rows, beta=beta, C=C), allow_pickle=True)
# ---- summary against the GF reference
tg, Eg, lamg, e2g = d['t'], d['E'], d['lam'][:, ::-1], d['eE2'][:, ::-1]   # saved run is ascending
def at(tq, arr):  # GF quantity at time tq (log-interp)
    i = np.searchsorted(tg, tq); i = min(max(i, 1), len(tg)-1); w = (np.log(tq)-np.log(tg[i-1]))/(np.log(tg[i])-np.log(tg[i-1]))
    return (1-w)*arr[i-1] + w*arr[i]
print(f'{tag} beta={beta} C={C}: {k} steps to t={t:.0f} ({time.time()-t0w:.0f}s); GF run used {d["steps"]} GD steps')
print('      t    steps  E_HB/E_GF  lam_k(HB)/lam_k(GF) k=1,10,100,300   resid.energy rank>100 HB|GF   eta*lam1  under-damped below rank   top3(dW1)')
for r in rows[::max(1, len(rows)//14)] + [rows[-1]]:
    lg = at(r['t'], lamg); eg = at(r['t'], e2g)
    lr = [r['lam'][j]/lg[j] for j in (0, 9, 99, 299)]
    fr = r['e2'][100:].sum()/r['e2'].sum(); fg = eg[100:].sum()/eg.sum()
    ud = int(np.sum(r['eta']*r['lam'] > (1-np.sqrt(beta))**2))
    print(f'{r["t"]:9.2f} {r["k"]:7d}  {r["E"]/at(r["t"], Eg):8.3f}   ' + ' '.join(f'{x:6.3f}' for x in lr) +
          f'     {fr:.3f} | {fg:.3f}      {r["eta"]*r["lam"][0]:5.2f}   {ud:5d}        {r["top3"]:.3f}', flush=True)
Ef = rows[-1]['E']; EgT = Eg[-1]
hit = next((r for r in rows if r['E'] <= EgT), None)
print(f'   final E_HB={Ef:.4f} vs E_GF(T)={EgT:.4f}; steps to reach E_GF(T): ' + (f'{hit["k"]} (t={hit["t"]:.0f})' if hit else 'not reached'))
