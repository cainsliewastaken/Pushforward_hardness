"""Init statistic m_l = E_u ||Psi_u^(l)||_F^2 / n over random unit directions u in layer l (3 draws), per design.
usage: python init_m.py H DEPTH SEED DELTA OUTSCALE OUT.npz"""
import numpy as np, sys, os, torch
os.environ['L63_OUTSCALE'] = sys.argv[5]
from l63test import make_data
from l63test_torch import MLP, DEV, along_dir_second
torch.set_default_dtype(torch.float64)
H, depth, seed, Delta, sc, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
n = 200; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV)
net = MLP(3, H, depth, 3, seed=seed)
if sc != 1.0: net.forward(X)
theta = net.pack(); sizesW = [W.numel() for W in net.W]; offs = np.cumsum([0]+sizesW); g = torch.Generator(device=DEV).manual_seed(0)
m = []
for l in range(len(net.W)):
    vals = []
    for _ in range(3):
        u = torch.zeros_like(theta); u[offs[l]:offs[l+1]] = torch.randn(sizesW[l], generator=g, device=DEV); u = u/u.norm()
        Pu = along_dir_second(net, X, theta, u); vals.append(float((Pu**2).sum())/n)
    m.append(np.mean(vals))
np.savez(out, m=np.array(m), H=H, depth=depth, seed=seed, Delta=Delta, s=sc); print(out, np.array(m).round(4))
