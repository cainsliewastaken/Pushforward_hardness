"""Response-law universality: for a design, build the coherent chain direction from the reduced model's J_k
(model v2, DRIVE=0), probe K at W0 + s*dW_chain for scales s (chain rescaled so its hidden-layer rank-16 amplitude
equals the model's predicted amplitude at s=1), fit n_k = alpha*log(lam1/lam_k0)+beta (exponent between s=0.5 and 1)
and log(lift_k(1)-1) = gamma*log(lam1/lam_k0)+delta. usage: python hermite_scan.py H DEPTH SEED DELTA OUTSCALE"""
import numpy as np, sys, os, torch
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
os.environ['L63_OUTSCALE'] = sys.argv[5]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
import reduced_model2 as M
torch.set_default_dtype(torch.float64)
H, depth, seed, Delta, sc = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
n = 200; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV); Yv = Y.reshape(-1)
net = MLP(3, H, depth, 3, seed=seed)
if sc != 1.0: net.forward(X)
W0 = [W.clone() for W in net.W]; L = len(W0)
J, o = net.jacobian(X); E0 = o.reshape(-1)-Yv; K0 = kernel(J, n); w0, V = torch.linalg.eigh(K0); w0 = w0.flip(0); V = V.flip(1)
tag = f'H{H}_D{Delta}_depth{depth}_s{seed}' + (f'_os{sc:g}' if sc != 1 else '')
z = np.load(f'{L63_ROOT}/runs/initq/q_{tag}.npz')
# J_k from model v2 (fitted gain law only sets the shape of J_k; amplitude rescaled below)
lam0 = z['lam0']; c = z['c0'].copy(); q = z['q']; k = np.arange(1, 601); frough = c[100:].sum()/c.sum(); B = M.B; T = 3000 if sc == 1 else 3000/sc**2
Jk = np.zeros(600); ts = np.concatenate([[0.], np.geomspace(1e-3/lam0[0], T, 1500)]); A = np.full(L, 1e-3); hid = list(range(1, L-1)) if L > 2 else [1]
for i in range(1, len(ts)):
    dt = ts[i]-ts[i-1]; P = max(np.prod(A[hid]), 1e-6); lk = np.log(k/100.)
    lift = 1 + np.exp(B[0]+B[1]*np.log(P)+B[2]*lk+B[3]*(depth-2)+B[4]*np.log(frough)+B[5]*lk*np.log(frough)+B[6]*lk*np.log(P)); lam = lam0*lift
    Jk += np.sqrt(np.maximum(c, 0))*np.sqrt(lift)*dt; c = c*np.exp(-2*lam*dt); contrib = q*Jk[None, :]**2; A = np.sqrt(np.sort(contrib, axis=1)[:, -16:].sum(1))+1e-6
G = J.T @ V / np.sqrt(n); sizesW = [W.numel() for W in net.W]; offs = np.cumsum([0]+sizesW)
dW = [-(G[offs[l]:offs[l+1], :] @ torch.tensor(Jk, device=DEV)).reshape(W0[l].shape) for l in range(L)]
# normalise so that the hidden-layer Frobenius norm at s=1 equals the model's amplitude / 1.7 (measured overshoot)
Ahid = float(dW[hid[0]].norm()); target = float(A[hid[0]])/1.7; dW = [x*target/Ahid for x in dW]
def spec(s):
    for l in range(L): net.W[l].copy_(W0[l]+s*dW[l])
    Jp, _ = net.jacobian(X)
    for l in range(L): net.W[l].copy_(W0[l])
    return torch.linalg.eigvalsh(kernel(Jp, n)).flip(0).cpu().numpy()
ss = [0.5, 1.0]; lam = np.array([spec(s) for s in ss]); l0 = w0.cpu().numpy(); logr = np.log(l0[0]/np.maximum(l0, 1e-300))
sel = np.arange(19, 400); g = lam[:, sel]/l0[sel]-1; nk = np.log(g[1]/g[0])/np.log(2.); ok = np.isfinite(nk) & (g[0] > 0.05)
ca = np.polyfit(logr[sel][ok], nk[ok], 1); cb = np.polyfit(logr[sel][ok], np.log(g[1][ok]), 1)
print(f"{tag:26s} |dW_hid|={target:5.2f} | n_k = {ca[0]:.3f} logr + {ca[1]:5.2f} (res {np.std(nk[ok]-np.polyval(ca, logr[sel][ok])):.2f}) | log(lift_k(1)-1) = {cb[0]:.3f} logr + {cb[1]:5.2f} | lift at k=100,200,400 (s=1): {lam[1][99]/l0[99]:7.1f} {lam[1][199]/l0[199]:8.1f} {lam[1][399]/l0[399]:9.1f}")
