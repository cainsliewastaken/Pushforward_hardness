"""Small model v5: no weights, no features along the path. State c_k, J_k, A_l. Response law (init-time property of the
architecture class, coefficients = class averages from hermite_scan): with r_k = log(lam1/lam_k(0)),
  n_k = ALPHA r_k + BETA,  L_k = GAMMA r_k + DELTA  (log(lift_k - 1) at hidden amplitude A_ref),
  lam_k(A) = lam_k(0) [1 + exp(L_k + n_k log(A/A_ref))],  A = hidden-layer rank-16 amplitude / AMPFIX (1.7, measured
  overshoot of the orthogonal-mode amplitude law). dc_k/dt = -2 lam_k c_k ; dJ_k/dt = sqrt(c_k lam_k/lam_k(0)).
usage: python reduced_model5.py TAG T   (reads initq/q_TAG.npz)"""
import numpy as np, sys, os
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
ALPHA, BETA, GAMMA, DELTA, A_REF, AMPFIX = 0.158, 1.19, 0.267, 1.17, 6.2, 1.7
tag, T = sys.argv[1], float(sys.argv[2])
z = np.load(f'{L63_ROOT}/runs/initq/q_{tag}.npz'); lam0 = z['lam0']; c = z['c0'].copy(); q = z['q']; L = q.shape[0]; depth = L-1
hid = list(range(1, L-1)) if L > 2 else [1]; r = np.log(lam0[0]/np.maximum(lam0, 1e-300)); nk = ALPHA*r + BETA; Lk = GAMMA*r + DELTA
Jk = np.zeros(600); ts = np.concatenate([[0.], np.geomspace(1e-3/lam0[0], T, 3000)]); hist = []
for i in range(1, len(ts)):
    dt = ts[i]-ts[i-1]
    contrib = q*Jk[None, :]**2; A = np.sqrt(np.sort(contrib, axis=1)[:, -16:].sum(1))/AMPFIX + 1e-9; Ah = np.mean(A[hid])
    lift = 1 + np.exp(np.minimum(Lk + nk*np.log(Ah/A_REF), 30.)); lam = lam0*lift
    Jk += np.sqrt(np.maximum(c, 0))*np.sqrt(lift)*dt; c = c*np.exp(-2*lam*dt)
    hist.append((ts[i], np.sqrt(c.sum()), lift[99], Ah))
hist = np.array(hist); print(f"{tag:26s} E(T)={hist[-1,1]:.4f} lift100(T)={hist[-1,2]:.1f} A_hid(T)={hist[-1,3]:.2f}")
np.savez(f'{L63_ROOT}/runs/v5/v5_{tag}.npz', t=hist[:,0], E=hist[:,1], lift100=hist[:,2], A=hist[:,3])
