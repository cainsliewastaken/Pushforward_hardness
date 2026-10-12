"""First-order coherent model of the learned kernel, everything at theta_0:
  delta_Psi = sum_{k in top-M} J_k Psi_{G_k},  G_k = Psi^T v_k / sqrt(n),  Psi_{G_k} = directional derivative of Psi at theta_0
  K(J) = (Psi + dPsi)(Psi + dPsi)^T / n ;  J_k from the reduced model (DRIVE=0), optionally rescaled per layer to the
  trained |dW_l|_F (to separate direction/structure from amplitude). Compare lam_k/lam_k(0) at k=50,100,200,400 with trained."""
import numpy as np, sys, os, torch
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l63test import make_data
from l63test_torch import MLP, kernel, DEV, along_dir_second
import reduced_model2 as M
torch.set_default_dtype(torch.float64)
fn, Delta, depth, H, MM = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
d = np.load(fn, allow_pickle=True).item(); W0 = [torch.tensor(w, device=DEV) for w in d['W0']]; WT = [torch.tensor(w, device=DEV) for w in d['WT']]
n = 200; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV); Yv = Y.reshape(-1)
net = MLP(3, H, depth, 3, seed=0); theta0 = net.pack().clone()
J0, out = net.jacobian(X); K0 = kernel(J0, n); w0, V = torch.linalg.eigh(K0); w0 = w0.flip(0); V = V.flip(1); E0 = out.reshape(-1)-Yv
sizesW = [W.numel() for W in net.W]; offs = np.cumsum([0]+sizesW); L = len(W0); ks = [49, 99, 199, 399]
# model J_k (same integration as coherence_test)
z = np.load(f'{L63_ROOT}/runs/initq/q_H{H}_D{Delta}_depth{depth}_s0.npz'); lam0 = z['lam0']; c = z['c0'].copy(); q = z['q']; k = np.arange(1, 601); fr = c[100:].sum()/c.sum(); B = M.B
Jk = np.zeros(600); ts = np.concatenate([[0.], np.geomspace(1e-3/lam0[0], 3000, 1500)]); A = np.full(L, 1e-3); hid = list(range(1, L-1)) if L > 2 else [1]
for i in range(1, len(ts)):
    dt = ts[i]-ts[i-1]; P = max(np.prod(A[hid]), 1e-6); lk = np.log(k/100.)
    lift = 1 + np.exp(B[0]+B[1]*np.log(P)+B[2]*lk+B[3]*(depth-2)+B[4]*np.log(fr)+B[5]*lk*np.log(fr)+B[6]*lk*np.log(P)); lam = lam0*lift
    Jk += np.sqrt(np.maximum(c, 0))*np.sqrt(lift)*dt; c = c*np.exp(-2*lam*dt); contrib = q*Jk[None, :]**2; A = np.sqrt(np.sort(contrib, axis=1)[:, -16:].sum(1))+1e-6
G = J0.T @ V/np.sqrt(n)                                   # d x 600
top = np.argsort(-(q.sum(0)*Jk**2))[:MM]
# second-order features along each selected mode direction (at theta_0), 2 Jacobians each
PG = {}
for kk in top:
    g = G[:, kk]; u = g/g.norm(); PG[kk] = along_dir_second(net, X, theta0, u)*float(g.norm())     # d/dh Psi(theta0 + h G_k)
def lifts_from(Jvec, label):
    dPsi = sum(float(Jvec[kk])*(-1.0)*PG[kk] for kk in top)          # dW = -sum J_k G_k  ->  dPsi = -sum J_k Psi_{G_k}
    Kf = kernel(J0 + dPsi, n); w = torch.linalg.eigvalsh(Kf).flip(0)
    print(f"   {label:52s}: " + ' '.join(f'{float(w[j]/w0[j]):9.1f}' for j in ks))
# trained reference and the exact coherent chain (network evaluated), for comparison
for l in range(L): net.W[l].copy_(WT[l])
bT = [torch.tensor(b, device=DEV) for b in d['bT']]
for l in range(L): net.b[l].copy_(bT[l])
Jt, _ = net.jacobian(X); wt = torch.linalg.eigvalsh(kernel(Jt, n)).flip(0); print(f"{os.path.basename(fn)}  lifts at k=50,100,200,400")
print(f"   {'trained':52s}: " + ' '.join(f'{float(wt[j]/w0[j]):9.1f}' for j in ks))
net.unpack(theta0)
fro = [float((WT[l]-W0[l]).norm()) for l in range(L)]
dWm = [-(G[offs[l]:offs[l+1], :][:, top] @ torch.tensor(Jk[top], device=DEV)).reshape(W0[l].shape) for l in range(L)]
scale = np.mean([fro[l]/max(float(dWm[l].norm()), 1e-12) for l in range(L)])
print(f"   model |dW| per layer {[round(float(x.norm()),2) for x in dWm]} vs trained {np.round(fro,2)}; mean rescale {scale:.2f}")
lifts_from(Jk, f'first-order coherent, model J ({MM} modes), unscaled')
lifts_from(Jk*scale, f'first-order coherent, rescaled to trained amplitude')
lifts_from(Jk*scale*0.5, f'   same at half amplitude')
lifts_from(Jk*scale*0.25, f'   same at quarter amplitude')
