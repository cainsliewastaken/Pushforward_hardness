"""Reduced model v4: piecewise-linearised gradient flow. The residual is carried as a vector; between checkpoints the
features (Jacobian) are frozen at the last probe, so the residual decays exactly by kernel regression in the probed
eigenbasis and the weights accumulate the gradient -Psi^T E/n; at each checkpoint the Jacobian is re-evaluated at the
accumulated weights (one Jacobian per checkpoint; no training, no fitted curve). Directions are the gradient chain.
usage: python reduced_model4.py H DEPTH SEED DELTA OUTSCALE T NCHECK OUT.npz"""
import numpy as np, sys, os, torch, time
os.environ['L63_OUTSCALE'] = sys.argv[5]
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
torch.set_default_dtype(torch.float64)
H, depth, seed, Delta, sc, T, DELTA_R, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), sys.argv[8]
n = 200; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV); Yv = Y.reshape(-1)
net = MLP(3, H, depth, 3, seed=seed)
if sc != 1.0: net.forward(X)
theta = net.pack().clone()
def eigh_safe(K):
    try: w, V = torch.linalg.eigh(K)
    except Exception: w, V = torch.linalg.eigh(K.cpu() + 1e-10*torch.eye(K.shape[0])); w, V = w.to(DEV), V.to(DEV)
    return w.flip(0), V.flip(1)
t0 = time.time(); J, o = net.jacobian(X); E = o.reshape(-1)-Yv; K = kernel(J, n); w, V = eigh_safe(K); w0 = w.clone()
hist_t = []; hist_E = []; hist_l100 = []; ic = 0; last = torch.zeros_like(theta)
# time grid: fine geometric
ts = np.concatenate([[0.], np.geomspace(1e-3/float(w0[0]), T, 3000)])
a = (V.T @ E)/np.sqrt(n); dtheta = torch.zeros_like(theta)
for i in range(1, len(ts)):
    dt = ts[i]-ts[i-1]
    # exact kernel-regression step in the frozen basis: a_k <- a_k e^{-lam_k dt}; weight increment -Psi^T E/n dt
    # integral of E over the step: E(s) = sqrt(n) V a e^{-lam s}  ->  int E ds = sqrt(n) V (a (1-e^{-lam dt})/lam)
    dec = torch.exp(-torch.clamp(w, min=0)*dt); intE = np.sqrt(n)*(V @ (a*torch.where(w > 1e-300, (1-dec)/torch.clamp(w, min=1e-300), torch.full_like(w, dt))))
    dtheta -= (J.T @ intE)/n; a = a*dec
    if float((dtheta-last).norm()) > DELTA_R or i == len(ts)-1:
        E = np.sqrt(n)*(V @ a); net.unpack(theta + dtheta); J, o = net.jacobian(X); E = o.reshape(-1)-Yv     # re-evaluate features AND the true output error at the new weights
        K = kernel(J, n); w, V = eigh_safe(K); a = (V.T @ E)/np.sqrt(n); ic += 1; last = dtheta.clone()
    hist_t.append(ts[i]); hist_E.append(float(torch.sqrt((a*a).sum()))); hist_l100.append(float(w[99]))
np.savez(out, t=np.array(hist_t), E=np.array(hist_E), lam100=np.array(hist_l100), lamT=w.cpu().numpy(), lam0=w0.cpu().numpy(), H=H, depth=depth, seed=seed, Delta=Delta, s=sc, wall=time.time()-t0)
print(f"H{H} d{depth} s{seed} D{Delta} os{sc:g}: E(T)={hist_E[-1]:.4f} lift100(T)={float(w[99]/w0[99]):.1f} ({ic} Jacobians, delta={DELTA_R}, {time.time()-t0:.0f}s)", flush=True)
