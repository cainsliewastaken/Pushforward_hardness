"""Test of the Hermite/RMT response law at initialization: probe K at W0 + s*dW_dir for scales s, dW_dir = trained
coherent change (direction only). Local exponent n_k = dlog(lam_k(s)-lam_k(0))/dlog s per rank, vs log(lam1/lam_k(0)).
usage: python hermite_test.py W.npy DELTA DEPTH H"""
import numpy as np, sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
torch.set_default_dtype(torch.float64)
fn, Delta, depth, H = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
d = np.load(fn, allow_pickle=True).item(); W0 = [torch.tensor(w, device=DEV) for w in d['W0']]; WT = [torch.tensor(w, device=DEV) for w in d['WT']]; b0 = [torch.tensor(b, device=DEV) for b in d['b0']]; bT = [torch.tensor(b, device=DEV) for b in d['bT']]
n = 200; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV)
net = MLP(3, H, depth, 3, seed=0); L = len(W0); dW = [WT[l]-W0[l] for l in range(L)]; db = [bT[l]-b0[l] for l in range(L)]
def spec(s):
    for l in range(L): net.W[l].copy_(W0[l]+s*dW[l]); net.b[l].copy_(b0[l]+s*db[l])
    J, _ = net.jacobian(X); return torch.linalg.eigvalsh(kernel(J, n)).flip(0).cpu().numpy()
ss = np.array([0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25]); lam = np.array([spec(s) for s in ss]); lam0 = lam[0]
k = np.arange(1, 601); logr = np.log(lam0[0]/np.maximum(lam0, 1e-300))
print(f"{os.path.basename(fn)}  (depth {depth})")
print("  rank k | log(lam1/lam_k(0)) | lift at s=0.25,0.5,1.0 | local exponent n_k between s=0.25-0.5, 0.5-1.0")
for kk in [10, 20, 50, 100, 150, 200, 300, 400]:
    g = lam[:, kk-1]/lam0[kk-1]-1
    n1 = np.log(g[3]/g[2])/np.log(2.); n2 = np.log(g[5]/g[3])/np.log(2.)
    print(f"  {kk:6d} | {logr[kk-1]:6.2f} | {g[2]+1:8.1f} {g[3]+1:8.1f} {g[5]+1:9.1f} | {n1:5.2f} {n2:5.2f}")
# linear fit of n_k (0.5-1.0) vs log(lam1/lam_k0) over ranks 20-400
sel = np.arange(19, 400); g = lam[:, sel]/lam0[sel]-1; nk = np.log(g[5]/g[3])/np.log(2.); ok = np.isfinite(nk) & (g[3] > 0.05)
c = np.polyfit(logr[sel][ok], nk[ok], 1); res = nk[ok]-np.polyval(c, logr[sel][ok])
print(f"  fit n_k = {c[0]:.3f} * log(lam1/lam_k0) + {c[1]:.2f}  over ranks 20-400 (n={ok.sum()}), residual std {res.std():.2f}")
# also: is the exponent a function of log(lam1/lam_k0) alone when compared across the two layers depths? printed per run; compare slopes across runs
np.savez(fn.replace('.npy', '_hermite.npz'), ss=ss, lam=lam, logr=logr, fit=c)
