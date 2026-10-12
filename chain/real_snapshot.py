"""Integrate the real network by full-batch GD (float64, eta=0.4/lam1) to time T8 and dump per-unit z_q, m_q, K, E.
usage: python real_snapshot.py H DEPTH SEED DELTA T8 OUT.pt"""
import numpy as np, sys, os, torch
os.environ['L63_OUTSCALE'] = '1.0'
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
torch.set_default_dtype(torch.float64)
H, depth, seed, Delta, T8, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
n = 200; ny = 3; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV); Yv = Y.reshape(-1)
net = MLP(3, H, depth, 3, seed=seed); L = len(net.W)
J, o = net.jacobian(X); E = o.reshape(-1) - Yv; K = kernel(J, n); lam1 = float(torch.linalg.eigvalsh(K)[-1]); eta = 0.4/lam1
th = net.pack(); t = 0.; steps = 0
while t < T8 - 1e-12:
    h = min(eta, T8 - t); J, o = net.jacobian(X); E = o.reshape(-1) - Yv
    th = th - h*(J.T @ E)/n; net.unpack(th); t += h; steps += 1
J, o = net.jacobian(X); E = o.reshape(-1) - Yv; K = kernel(J, n)
o, Acts = net.forward(X); Zs = [Acts[l] @ net.W[l].T + net.b[l] for l in range(L)]
ms = []
for l in range(L-1):
    mk = torch.zeros(n, net.W[l].shape[0], ny, device=DEV)
    for c in range(ny):
        dl = torch.zeros(n, ny, device=DEV); dl[:, c] = 1.; dl = dl @ net.W[-1]
        for l2 in range(L-2, l, -1): dl = dl*(1-Acts[l2+1]**2); dl = dl @ net.W[l2]
        mk[:, :, c] = dl
    ms.append(mk.permute(1, 0, 2).cpu())
print(f'real GD to t={t:.3f} in {steps} steps (eta={eta:.4e}): ||E||={float(torch.sqrt(E@E/n)):.4f}')
torch.save(dict(t=t, Z=[z.T.cpu() for z in Zs], M=ms, K=K.cpu(), E=E.cpu()), out)
