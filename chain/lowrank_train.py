"""Dynamic spike test: GD where each layer's weight gradient is truncated to its top-r singular pairs at every step
(r = 0 -> full gradient). Logs ||E||, lam1, lam_100 lift, per-layer dW top sv and effective rank at ~25 times.
usage: python lowrank_train.py DELTA DEPTH H T_TOTAL R OUT"""
import numpy as np, sys, math, torch, time
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
torch.set_default_dtype(torch.float64)
Delta = float(sys.argv[1]); depth = int(sys.argv[2]); H = int(sys.argv[3]); T_total = float(sys.argv[4]); R = int(sys.argv[5]); out = sys.argv[6]
n = 200; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV); Yv = Y.reshape(-1)
net = MLP(3, H, depth, 3, seed=0); W0 = [W.clone() for W in net.W]
J, o = net.jacobian(X); K0 = kernel(J, n); w0 = torch.linalg.eigvalsh(K0).flip(0); lam1 = float(w0[0]); tail0 = float(w0[99])
eta = 0.4/lam1; t = 0.; step = 0; t0 = time.time()
tgrid = np.unique(np.concatenate([np.geomspace(0.1, T_total, 20), [T_total]])); ig = 0; logs = []
def eff_rank(sv):
    p = sv**2/(sv**2).sum(); p = p[p > 0]; return float(torch.exp(-(p*torch.log(p)).sum()))
while True:
    if t >= tgrid[ig] or step == 0:
        Jc, oc = net.jacobian(X); E = oc.reshape(-1) - Yv; w = torch.linalg.eigvalsh(kernel(Jc, n)).flip(0)
        svs = [torch.linalg.svdvals(net.W[l]-W0[l]) for l in range(len(W0))]
        rec = dict(t=t, step=step, E=float(torch.sqrt(E@E/n)), lam1=float(w[0]), lift100=float(w[99])/tail0, amp=[float(s[0]) for s in svs], erank=[eff_rank(s) for s in svs])
        logs.append(rec); print(f"   r={R} t={t:8.2f} ||E||={rec['E']:.4f} lift100={rec['lift100']:7.1f} amps={[round(a,2) for a in rec['amp']]} eranks={[round(e,1) for e in rec['erank']]} ({time.time()-t0:.0f}s)", flush=True)
        if eta*float(w[0]) > 1: eta = 0.5/float(w[0])
        while ig < len(tgrid) and t >= tgrid[ig]: ig += 1
        if ig >= len(tgrid): break
    o, A = net.forward(X); delta = (o - Y)/n
    gW = [None]*len(net.W); gb = [None]*len(net.b)
    gW[-1] = delta.T @ A[-1]; gb[-1] = delta.sum(0); dl = delta @ net.W[-1]
    for l in range(net.L-1, -1, -1):
        dl = dl*(1-A[l+1]**2); gW[l] = dl.T @ A[l]; gb[l] = dl.sum(0)
        if l > 0: dl = dl @ net.W[l]
    for l in range(len(net.W)):
        g = gW[l]
        if R > 0 and min(g.shape) > R:
            U, S, V = torch.svd_lowrank(g, q=min(R+4, min(g.shape)), niter=2); g = (U[:, :R]*S[:R]) @ V[:, :R].T
        net.W[l].sub_(eta*g); net.b[l].sub_(eta*gb[l])
    t += eta; step += 1
np.save(out, dict(logs=logs, R=R, Delta=Delta, depth=depth, H=H), allow_pickle=True); print('done', flush=True)
