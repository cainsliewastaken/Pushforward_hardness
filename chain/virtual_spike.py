"""Init-only probe of the lift law: add a virtual spike of amplitude a to layer l along the direction the ROUGH part
of the initial error drives (gradient of P_tail E0 w.r.t. W_l, rank reduced to its top singular pair), evaluate the
kernel tail lift lam_100(K)/lam_100(K0) and the per-layer block lift. No training.
usage: python virtual_spike.py N DELTA DEPTH H [amps] [ktail]   (env L63_OUTSCALE for scale)"""
import numpy as np, sys, os, math, torch
from l63test import make_data
from l63test_torch import MLP, kernel, DEV, OUTSCALE
torch.set_default_dtype(torch.float64)
n = int(sys.argv[1]); Delta = float(sys.argv[2]); depth = int(sys.argv[3]); H = int(sys.argv[4])
amps = [float(x) for x in (sys.argv[5] if len(sys.argv) > 5 else '0.5,1,2,4,8').split(',')]; ktail = int(sys.argv[6]) if len(sys.argv) > 6 else 100
Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV); Yv = Y.reshape(-1)
net = MLP(3, H, depth, 3, seed=0)
if OUTSCALE != 1.0: net.forward(X)
J, out = net.jacobian(X); E0 = out.reshape(-1) - Yv; K0 = kernel(J, n); w0, V0 = torch.linalg.eigh(K0); w0 = w0.flip(0); V0 = V0.flip(1)
lam1 = float(w0[0]); tail0 = float(w0[ktail-1])
# rough part of the initial error and its gradient per layer
Er = V0[:, ktail:] @ (V0[:, ktail:].T @ E0); Es = E0 - Er
sizesW = [W.numel() for W in net.W]; sizesB = [b.numel() for b in net.b]; offB = sum(sizesW)
def layer_grad(Evec):
    g = J.T @ Evec / n; out = []; o = 0
    for l, W in enumerate(net.W): out.append(g[o:o+sizesW[l]].reshape(W.shape)); o += sizesW[l]
    return out
gr = layer_grad(Er); gs = layer_grad(Es)
W0 = [W.clone() for W in net.W]
print(f"[H{H} d{depth} D{Delta} s{OUTSCALE:g}] lam1={lam1:.3e} lam_{ktail}(0)={tail0:.3e} ||E0||={float(torch.sqrt(E0@E0/n)):.3f} rough frac of E0^2={float(Er@Er/(E0@E0)):.3f}")
print("   per-layer |grad| from rough / smooth residual: " + ' '.join(f'L{l}: {float(torch.linalg.norm(gr[l])):.2e}/{float(torch.linalg.norm(gs[l])):.2e}' for l in range(len(net.W))))
def lift_with(dirs, a):
    for l, W in enumerate(net.W): W.copy_(W0[l] + a*dirs[l])
    Jp, _ = net.jacobian(X); K = kernel(Jp, n); w = torch.linalg.eigvalsh(K).flip(0)
    for l, W in enumerate(net.W): W.copy_(W0[l])
    return float(w[ktail-1]/tail0), float(w[0]/lam1)
# direction A: rank-1 (top singular pair) of the rough gradient in ONE hidden layer; direction B: full rough gradient, all layers
print(f"   amp | rank-1 spike in each layer (lift_{ktail}, lam1 ratio) | full rough-gradient direction all layers | smooth-gradient direction all layers")
for a in amps:
    row = []
    for l in range(len(net.W)):
        U, S, Vh = torch.linalg.svd(gr[l], full_matrices=False); d1 = [torch.zeros_like(W) for W in net.W]; d1[l] = torch.outer(U[:, 0], Vh[0])
        row.append(lift_with(d1, a))
    nrm = math.sqrt(sum(float((g**2).sum()) for g in gr)); fullr = lift_with([g/nrm for g in gr], a)
    nrs = math.sqrt(sum(float((g**2).sum()) for g in gs)); fulls = lift_with([g/nrs for g in gs], a)
    print(f"   {a:5.2f} | " + ' '.join(f'L{l}: {x[0]:7.1f} ({x[1]:4.2f})' for l, x in enumerate(row)) + f" | {fullr[0]:7.1f} ({fullr[1]:4.2f}) | {fulls[0]:7.1f} ({fulls[1]:4.2f})")
