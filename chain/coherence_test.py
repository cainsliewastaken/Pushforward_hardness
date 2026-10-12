"""Coherence test on a run with saved W0, WT. (a) trained dW; (b) per-layer randomly rotated dW (same singular values,
cross-layer coherence destroyed); (c) coherent gradient-generated chain dW_model_l = -sum_k J_k Psi_l^T v_k/sqrt(n)
with J_k from the reduced model (DRIVE=0), rescaled per layer to the trained amplitude (to isolate direction from
amplitude) and also unscaled. Report lam_k(K)/lam_k(K0) at k=50,100,200,400 for each."""
import numpy as np, sys, os, torch
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
import reduced_model2 as M
torch.set_default_dtype(torch.float64)
fn, Delta, depth, H = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
d = np.load(fn, allow_pickle=True).item(); W0 = [torch.tensor(w, device=DEV) for w in d['W0']]; WT = [torch.tensor(w, device=DEV) for w in d['WT']]; b0 = [torch.tensor(b, device=DEV) for b in d['b0']]; bT = [torch.tensor(b, device=DEV) for b in d['bT']]
n = 200; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV); Yv = Y.reshape(-1)
net = MLP(3, H, depth, 3, seed=0); J, out = net.jacobian(X); E0 = out.reshape(-1)-Yv; K0 = kernel(J, n); w0, V = torch.linalg.eigh(K0); w0 = w0.flip(0); V = V.flip(1)
sizesW = [W.numel() for W in net.W]; offs = np.cumsum([0]+sizesW); L = len(W0); ks = [49, 99, 199, 399]
def probe(Ws, bs):
    for l in range(L): net.W[l].copy_(Ws[l]); net.b[l].copy_(bs[l])
    Jp, _ = net.jacobian(X); w = torch.linalg.eigvalsh(kernel(Jp, n)).flip(0)
    for l in range(L): net.W[l].copy_(W0[l]); net.b[l].copy_(b0[l])
    return [float(w[k]/w0[k]) for k in ks]
dW = [WT[l]-W0[l] for l in range(L)]; amps = [float(torch.linalg.svdvals(x)[0]) for x in dW]; fro = [float(x.norm()) for x in dW]
print(f"{os.path.basename(fn)}: trained lift at k=50,100,200,400:", np.round(probe(WT, bT), 1), " |dW|_F per layer", np.round(fro, 2))
g = torch.Generator(device=DEV).manual_seed(1); res = []
for _ in range(3):
    Ws = []
    for l in range(L):
        U, S, Vh = torch.linalg.svd(dW[l], full_matrices=False); Ql, _ = torch.linalg.qr(torch.randn(U.shape[0], U.shape[0], generator=g, device=DEV)); Qr, _ = torch.linalg.qr(torch.randn(Vh.shape[1], Vh.shape[1], generator=g, device=DEV))
        Ws.append(W0[l] + (Ql @ U) @ torch.diag(S) @ (Vh @ Qr))
    res.append(probe(Ws, bT))
print("  incoherent (per-layer random rotation, same spectra, trained biases):", np.round(np.median(res, 0), 1))
# coherent gradient-generated chain from the reduced model's J_k
z = np.load(f'{L63_ROOT}/runs/initq/q_H{H}_D{Delta}_depth{depth}_s0.npz')
# integrate model to T=3000 and recover J_k
lam0 = z['lam0']; c = z['c0'].copy(); q = z['q']; k = np.arange(1, 601); frough = c[100:].sum()/c.sum(); B = M.B
Jk = np.zeros(600); ts = np.concatenate([[0.], np.geomspace(1e-3/lam0[0], 3000, 1500)]); A = np.full(L, 1e-3); hid = list(range(1, L-1)) if L > 2 else [1]
for i in range(1, len(ts)):
    dt = ts[i]-ts[i-1]; P = max(np.prod(A[hid]), 1e-6); lk = np.log(k/100.)
    lift = 1 + np.exp(B[0]+B[1]*np.log(P)+B[2]*lk+B[3]*(depth-2)+B[4]*np.log(frough)+B[5]*lk*np.log(frough)+B[6]*lk*np.log(P)); lam = lam0*lift
    Jk += np.sqrt(np.maximum(c, 0))*np.sqrt(lift)*dt; c = c*np.exp(-2*lam*dt); contrib = q*Jk[None, :]**2; A = np.sqrt(np.sort(contrib, axis=1)[:, -16:].sum(1))+1e-6
G = J.T @ V / np.sqrt(n)                     # d x 600 : Psi^T v_k / sqrt n
dWm = []
for l in range(L):
    Gl = G[offs[l]:offs[l+1], :]; dWm.append(-(Gl @ torch.tensor(Jk, device=DEV)).reshape(W0[l].shape))
print("  model chain |dW_model|_F per layer", np.round([float(x.norm()) for x in dWm], 2))
print("  coherent chain, unscaled:", np.round(probe([W0[l]+dWm[l] for l in range(L)], bT), 1))
print("  coherent chain, rescaled to trained |dW|_F per layer:", np.round(probe([W0[l]+dWm[l]*fro[l]/float(dWm[l].norm()) for l in range(L)], bT), 1))
# same chain but with only the top-16 modes' contributions
top = np.argsort(-(q.sum(0)*Jk**2))[:16]; Jt = np.zeros(600); Jt[top] = Jk[top]
dWt = [-(G[offs[l]:offs[l+1], :] @ torch.tensor(Jt, device=DEV)).reshape(W0[l].shape) for l in range(L)]
print("  coherent chain, top-16 modes only, rescaled:", np.round(probe([W0[l]+dWt[l]*fro[l]/max(float(dWt[l].norm()), 1e-12) for l in range(L)], bT), 1))
