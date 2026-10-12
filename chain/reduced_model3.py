"""Reduced model v3 (init-only, no fitted gain curve). Spectrum from the coherent gradient-generated chain:
at checkpoint times, dW_l = -sum_k J_k Psi_l^T v_k / sqrt(n), kernel probed at W0 + dW (one Jacobian), its spectrum
used until the next checkpoint. J_k integrated from the residual: dJ_k/dt = sqrt(c_k); dc_k/dt = -2 lam_k c_k with
lam_k the probed spectrum by rank. usage: python reduced_model3.py H DEPTH SEED DELTA OUTSCALE T NCHECK OUT.npz"""
import numpy as np, sys, os, torch, time
os.environ['L63_OUTSCALE'] = sys.argv[5]
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
torch.set_default_dtype(torch.float64)
H, depth, seed, Delta, sc, T, NCK, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]), sys.argv[8]
AMP = float(os.environ.get('AMPFIX', '1.0'))     # optional global correction of the amplitude overshoot (1.0 = none)
n = 200; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV); Yv = Y.reshape(-1)
net = MLP(3, H, depth, 3, seed=seed)
if sc != 1.0: net.forward(X)
W0 = [W.clone() for W in net.W]; b0 = [b.clone() for b in net.b]; L = len(W0)
J, o = net.jacobian(X); E0 = o.reshape(-1)-Yv; K0 = kernel(J, n); w0, V = torch.linalg.eigh(K0); w0 = w0.flip(0); V = V.flip(1)
G = (J.T @ V / np.sqrt(n)); sizesW = [W.numel() for W in net.W]; offs = np.cumsum([0]+sizesW)
c = ((V.T @ E0)**2/n).cpu().numpy(); lam = w0.cpu().numpy().copy(); lam0np = lam.copy(); Jk = np.zeros(600)
t0 = time.time(); checks = np.geomspace(1e-2/lam[0]*lam[0]*10, T, NCK); ts = np.concatenate([[0.], np.geomspace(1e-3/lam[0], T, 2000)]); ic = 0
hist_t = []; hist_E = []; hist_lam100 = []
for i in range(1, len(ts)):
    dt = ts[i]-ts[i-1]
    Jk += np.sqrt(np.maximum(c, 0))*np.sqrt(lam/lam0np)*dt; c = c*np.exp(-2*lam*dt)
    if ic < NCK and ts[i] >= checks[ic]:
        Jt = torch.tensor(AMP*Jk, device=DEV)
        for l in range(L): net.W[l].copy_(W0[l] - (G[offs[l]:offs[l+1], :] @ Jt).reshape(W0[l].shape))
        Jp, _ = net.jacobian(X); lam = torch.linalg.eigvalsh(kernel(Jp, n)).flip(0).cpu().numpy(); ic += 1
    hist_t.append(ts[i]); hist_E.append(np.sqrt(c.sum())); hist_lam100.append(lam[99])
for l in range(L): net.W[l].copy_(W0[l])
np.savez(out, t=np.array(hist_t), E=np.array(hist_E), lam100=np.array(hist_lam100), lamT=lam, lam0=w0.cpu().numpy(), H=H, depth=depth, seed=seed, Delta=Delta, s=sc, wall=time.time()-t0)
print(f"H{H} d{depth} s{seed} D{Delta} os{sc:g}: E(T)={hist_E[-1]:.4f} lift100(T)={lam[99]/float(w0[99]):.1f} wall {time.time()-t0:.0f}s", flush=True)
