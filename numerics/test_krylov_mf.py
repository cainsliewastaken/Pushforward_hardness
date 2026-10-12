"""Seconds-long CPU checks of krylov_mf against the explicit kernel and core.krylov_F."""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C
import krylov_mf as KM
import replicate_rsk as RK

torch.manual_seed(0)
dt = torch.float64
B, n, H, din, dout = 3, 40, 12, 3, 3
W1, b1 = torch.randn(B, H, din, dtype=dt), torch.randn(B, H, dtype=dt)
W2, b2 = torch.randn(B, dout, H, dtype=dt), torch.randn(B, dout, dtype=dt)
x = torch.randn(n, din, dtype=dt)
d = H * din + H + dout * H + dout
pi = torch.rand(B, d, dtype=dt) + 0.1
net = KM.TanhNets(W1, b1, W2, b2, x, pi=pi)
worst = {}
for b in range(B):
    A = net.psi_scaled(b)
    J = RK.net_jac((W1[b], b1[b], W2[b], b2[b]), x)                         # independent Jacobian
    worst["jac"] = max(worst.get("jac", 0), float((A - J * pi[b].sqrt() / np.sqrt(n)).abs().max()))
    v = torch.randn(B, n * dout, dtype=dt)
    worst["K"] = max(worst.get("K", 0), float((net.K(v)[b] - A @ (A.T @ v[b])).abs().max()))
    out = (torch.tanh(x @ W1[b].T + b1[b]) @ W2[b].T + b2[b]).reshape(-1)
    worst["out"] = max(worst.get("out", 0), float((net.out()[b] - out).abs().max()))
u = torch.randn(B, n * dout, dtype=dt)
M = 30
G, a, orth, nre = KM.lanczos(net.K, u, M)
lam1 = KM.lam1_krylov(net.K, B, n * dout, dt, "cpu")
for b in range(B):
    A = net.psi_scaled(b).numpy()
    ker = C.Kernel(A * np.sqrt(n), n)                                          # SVD route of core
    lam_ext, e_ext = ker.reduced(u[b].numpy() * np.sqrt(n))
    worst["lam1"] = max(worst.get("lam1", 0), abs(lam1[b] / ker.lam1 - 1))
    for g in (1e-4, 1e-2, 0.3):
        cs = np.sqrt(C.upsilon(g * ker.lam1, lam_ext, e_ext))
        Fref, _ = C.krylov_F(lam_ext, e_ext, cs, M)
        Fmf = KM.krylov_F(G[b], a[b], cs, list(range(1, M + 1)))
        worst["F_M vs core"] = max(worst.get("F_M vs core", 0), float(np.max(np.abs(np.array(Fmf) - Fref))))
        Fex = C.F_closed(lam_ext, e_ext, cs)[0]
        worst["F_M - F (must be <= 0)"] = max(worst.get("F_M - F (must be <= 0)", -1), max(Fmf) - Fex)
        ex = KM.Exact(net.psi_scaled(b))
        e2 = ex.reduced(u[b])
        worst["Exact vs core F"] = max(worst.get("Exact vs core F", 0), abs(C.F_closed(ex.lam_ext, e2, cs)[0] - Fex))
        cs_r = np.sqrt(KM.ritz_upsilon(G[b], a[b], g * ker.lam1))
        worst["ritz ups rel"] = max(worst.get("ritz ups rel", 0), abs(cs_r / cs - 1))
for g in (1e-4, 1e-2, 0.3):
    gam = torch.tensor(g * lam1, dtype=dt)
    slow, ups, cres, cits = KM.cg_slow(net.K, u, gam)
    for b in range(B):
        ex = KM.Exact(net.psi_scaled(b))
        e2 = ex.reduced(u[b])
        slow_ex = np.linalg.norm(g * lam1[b] * e2 / (ex.lam_ext + g * lam1[b]))
        ups_ex = C.upsilon(g * lam1[b], ex.lam_ext, e2)
        worst["cg slow rel"] = max(worst.get("cg slow rel", 0), abs(slow[b] / slow_ex - 1))
        worst["cg ups rel"] = max(worst.get("cg ups rel", 0), abs(ups[b] / ups_ex - 1))
        # Proposition 4: F(s_gamma) = ||u_gamma||
        worst["F(s_g) vs slow"] = max(worst.get("F(s_g) vs slow", 0),
                                      abs(C.F_closed(ex.lam_ext, e2, np.sqrt(ups[b]))[0] / slow[b] - 1))
worst["orth"] = float(orth.max())
worst["restarts"] = int(nre.max())
for k, v in worst.items():
    print(f"{k:28s} {v:.3e}")
ok = (worst["jac"] < 1e-12 and worst["K"] < 1e-12 and worst["out"] < 1e-12 and worst["F_M vs core"] < 1e-8
      and worst["F_M - F (must be <= 0)"] < 1e-10 and worst["Exact vs core F"] < 1e-8 and worst["lam1"] < 1e-8 and worst["cg slow rel"] < 1e-7 and worst["cg ups rel"] < 1e-7
      and worst["F(s_g) vs slow"] < 1e-7)
print("PASS" if ok else "FAIL")
