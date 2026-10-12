"""Seconds-long CPU checks of sine_deep: the autograd kernel equals the hand-written one (depth 1),
and the explicit Jacobian matches the matrix-free products (depth 2, with a preconditioner).
Also prints lr * lambda_1 at initialization for the large_sine setup (width 512, lr 0.0025)."""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import krylov_mf as KM
import sine_deep as SD

dt = torch.float64
cfg = dict(depth=1, width=24, n=50, omegas=[1, 20], random_labels=True, seeds=[0, 1])
X, Y, P, runs, _ = SD.setup(cfg, "cpu")
torch.manual_seed(0)
P = [p + 0.3 * torch.randn_like(p) for p in P]          # nonzero output layer
net = SD.Net(P, X)
ref = KM.TanhNets(P[0], P[1], P[2], P[3], X)
v = torch.randn(len(runs), cfg["n"], dtype=dt)
w = {}
w["out depth1"] = float((net.out() - ref.out()).abs().max())
w["K depth1"] = float((net.K(v) - ref.K(v)).abs().max())
for depth in (2, 3):
    cfg2 = dict(cfg, depth=depth)
    X2, Y2, P2, runs2, _ = SD.setup(cfg2, "cpu")
    P2 = [p + 0.3 * torch.randn_like(p) for p in P2]
    pis = [torch.rand_like(p) + 0.1 for p in P2]
    net2 = SD.Net(P2, X2, pis)
    v2 = torch.randn(len(runs2), cfg["n"], dtype=dt)
    Kv = net2.K(v2)
    for b in range(len(runs2)):
        A = net2.psi_scaled(b)
        w[f"K vs explicit depth{depth}"] = max(w.get(f"K vs explicit depth{depth}", 0),
                                               float((Kv[b] - A @ (A.T @ v2[b])).abs().max()))
for k, x in w.items():
    print(f"{k:24s} {x:.2e}")
print("PASS" if max(w.values()) < 1e-12 else "FAIL")
# lr * lambda_1 at initialization of the large_sine runs (seed 0)
cfg3 = dict(depth=1, width=512, n=1000, omegas=[1], random_labels=False, seeds=[0])
X3, Y3, P3, _, _ = SD.setup(cfg3, "cpu")
l1 = KM.lam1_krylov(SD.Net(P3, X3).K, 1, 1000, dt, "cpu")[0]
print(f"large_sine: lambda_1 at init {l1:.4g}, lr * lambda_1 = {0.0025 * l1:.4f}")
cfg4 = dict(cfg3, depth=2)
X4, Y4, P4, _, _ = SD.setup(cfg4, "cpu")
print(f"depth 2 width 512: lambda_1 at init {KM.lam1_krylov(SD.Net(P4, X4).K, 1, 1000, dt, 'cpu')[0]:.4g}")
