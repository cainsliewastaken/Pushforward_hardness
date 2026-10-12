"""Init-time per-layer, per-mode kernel energies q_lk = v_k^T K_l v_k (L x 600), plus lam0, c0 (E0 energy per mode),
for a design. usage: python init_blocks.py H DEPTH SEED DELTA OUTSCALE OUT.npz  (n=200, data seed 1)"""
import numpy as np, sys, os, torch
os.environ['L63_OUTSCALE'] = sys.argv[5]
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
torch.set_default_dtype(torch.float64)
H, depth, seed, Delta, sc, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
n = 200; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV)
net = MLP(3, H, depth, 3, seed=seed)
if sc != 1.0: net.forward(X)
J, o = net.jacobian(X); E0 = o.reshape(-1) - Y.reshape(-1); K = kernel(J, n); w, V = torch.linalg.eigh(K); w = w.flip(0); V = V.flip(1)
sizesW = [W.numel() for W in net.W]; sizesB = [b.numel() for b in net.b]; offB = sum(sizesW); L = len(net.W)
q = []; o_ = 0; ob = offB
for l in range(L):
    idx = torch.cat([torch.arange(o_, o_+sizesW[l]), torch.arange(ob, ob+sizesB[l])]).to(DEV); o_ += sizesW[l]; ob += sizesB[l]
    Jl = J[:, idx]; Kl = Jl @ Jl.T / n; q.append(torch.einsum('ij,ji->i', V.T @ Kl, V).cpu().numpy())
np.savez(out, lam0=w.cpu().numpy(), c0=((V.T @ E0)**2/n).cpu().numpy(), q=np.array(q), H=H, depth=depth, seed=seed, Delta=Delta, s=sc)
print(out, 'lam1', float(w[0]), 'sum q vs lam check', float(np.abs(np.array(q).sum(0)-w.cpu().numpy()).max()))
