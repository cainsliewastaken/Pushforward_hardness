"""Train one case (GD, T_total), save W0 and W(T); then probe the kernel tail with (a) W0 + alpha*dW for alpha in [0..1]
(static scaling of the trained change), (b) rank-r truncations of each layer's dW at alpha=1, (c) dW restricted to
hidden layers only / input layer only. Lift = lam_100(K)/lam_100(K0). usage: python dw_probe.py DELTA DEPTH H T OUT"""
import numpy as np, sys, os, math, torch
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
from l63_late import GraphStepper
torch.set_default_dtype(torch.float64)
Delta = float(sys.argv[1]); depth = int(sys.argv[2]); H = int(sys.argv[3]); T_total = float(sys.argv[4]); out = sys.argv[5]
n = 200; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV); Yv = Y.reshape(-1)
net = MLP(3, H, depth, 3, seed=0); W0 = [W.clone() for W in net.W]; b0 = [b.clone() for b in net.b]
J, o = net.jacobian(X); K0 = kernel(J, n); w0 = torch.linalg.eigvalsh(K0).flip(0); lam1 = float(w0[0]); tail0 = float(w0[99])
st = GraphStepper(net, X, Y, n, 0.4/lam1); t = 0.; eta = 0.4/lam1; step = 0
while t < T_total:
    k = 2000; st.run(k); t += k*eta; step += k
    if step % 100000 == 0:
        Jc, oc = net.jacobian(X); l1 = float(torch.linalg.eigvalsh(kernel(Jc, n))[-1])
        if eta*l1 > 1: eta = 0.5/l1; st.eta.fill_(eta)
J, o = net.jacobian(X); E = o.reshape(-1) - Yv; KT = kernel(J, n); wT = torch.linalg.eigvalsh(KT).flip(0)
print(f'trained H{H} d{depth} D{Delta}: T={t:.0f} steps={step} ||E||={float(torch.sqrt(E@E/n)):.4f} lift_100={float(wT[99])/tail0:.1f} lam1 {lam1:.1f}->{float(wT[0]):.1f}', flush=True)
WT = [W.clone() for W in net.W]; bT = [b.clone() for b in net.b]
dW = [WT[l]-W0[l] for l in range(len(WT))]; db = [bT[l]-b0[l] for l in range(len(bT))]
def probe(Ws, bs):
    for l in range(len(Ws)): net.W[l].copy_(Ws[l]); net.b[l].copy_(bs[l])
    Jp, _ = net.jacobian(X); w = torch.linalg.eigvalsh(kernel(Jp, n)).flip(0); return float(w[99])/tail0, float(w[0])/lam1
print('amps of dW per layer (top sv):', [round(float(torch.linalg.svdvals(d)[0]), 2) for d in dW])
print('(a) static scaling alpha*dW (all layers, biases too): alpha -> lift_100, lam1 ratio')
for al in [0.1, 0.25, 0.5, 0.75, 1.0]:
    print(f'   alpha={al:4.2f}: {probe([W0[l]+al*dW[l] for l in range(len(W0))], [b0[l]+al*db[l] for l in range(len(b0))])}')
print('(b) rank-r truncation of each layer dW at alpha=1 (biases full):')
for r in [1, 2, 4, 8, 16, 1000]:
    Ws = []
    for l in range(len(W0)):
        U, S, Vh = torch.linalg.svd(dW[l], full_matrices=False); rr = min(r, S.numel()); Ws.append(W0[l] + (U[:, :rr]*S[:rr]) @ Vh[:rr])
    print(f'   r={r:4d}: {probe(Ws, bT)}')
print('(c) which layers: only hidden dW / only input dW / only output dW (others at init, biases trained):')
L = len(W0)
for name, sel in [('hidden', range(1, L-1)), ('input', [0]), ('output', [L-1]), ('input+hidden', range(0, L-1))]:
    Ws = [W0[l] + (dW[l] if l in sel else 0) for l in range(L)]; print(f'   {name:13s}: {probe(Ws, bT)}')
print('(d) biases only (W at init):', probe(W0, bT))
np.save(out, dict(W0=[w.cpu().numpy() for w in W0], WT=[w.cpu().numpy() for w in WT], b0=[b.cpu().numpy() for b in b0], bT=[b.cpu().numpy() for b in bT]), allow_pickle=True)
