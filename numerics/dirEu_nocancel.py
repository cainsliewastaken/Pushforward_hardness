"""Direct vs Euler in the kernel eigenbasis (Lorenz-63 at initialization, shared kernel, c_E = c_D = 1).

u_i = <v_i, Phi_h>, w_i = <v_i, Phi_h - id> along the eigenvectors v_i of K (plus one null-space
coordinate each). Claim to check:
  (1) F_D(s) >= F_E(s) - eps for every s, with eps = ( sum_i (|w_i| - |u_i|)_+^2 )^(1/2)
      (eps = 0 if the identity cancels the increment along no eigendirection);
  (2) if eps = 0:  F_D(s)^2 - F_E(s)^2 >= sum_i (u_i^2 - w_i^2) gamma^2 / (lam_i + gamma)^2,
      gamma the multiplier of the direct problem at s.
Null space: compared through the norms of the null-space parts (the set of reachable outputs is
rotation-invariant there). Output: results/dirEu_nocancel.csv (new file)."""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C
import lorenz as Lz

HERE = os.path.dirname(os.path.abspath(__file__))
cfg = dict(Lz.CFG)
sol = Lz.trajectory(cfg)
n, H = cfg["n"], cfg["width"]
S = np.logspace(-1, 3, 81)
rows = []
print("h     seed  bad/all  eps      ||inc||  eps/||inc||  min(F_D-F_E)  min(F_D-F_E+eps)  (1) ok  F_D>F_E all s")
for h in cfg["hs"]:
    for seed in cfg["seeds"]:
        X, Y = Lz.dataset(cfg, sol, h, seed)
        xs = (X - X.mean(0)) / X.std(0)
        ker = C.Kernel(Lz.jac(Lz.init_params(seed, H, "cpu"), torch.tensor(xs, dtype=torch.float64), H).numpy(), n)
        lam, u = ker.reduced(Y.ravel())          # last coordinate: norm of the null-space part
        _, w = ker.reduced((Y - X).ravel())
        _, a = ker.reduced(X.ravel())
        au, aw = np.abs(u), np.abs(w)
        bad = aw > au
        eps = float(np.sqrt(np.sum((aw - au)[bad] ** 2)))
        nd = C.nrm((Y - X).ravel(), n)
        FD = np.array([C.F_closed(lam, u, s)[0] for s in S])
        FE = np.array([C.F_closed(lam, w, s)[0] for s in S])
        gap = FD - FE
        ok1 = bool(np.all(gap + eps >= -1e-9 * FD.max()))
        # margin formula on the good coordinates only (valid lower bound when eps = 0)
        marg = []
        for s in S:
            _, gam, Es = C.F_closed(lam, u, s)
            r2 = Es ** 2                                   # squared residual coordinates of the direct problem
            d2 = np.where(au > 0, np.minimum(aw / np.maximum(au, 1e-300), 1.0) ** 2, 0.0)
            marg.append(float(np.sum((1 - d2) * r2)))
        marg = np.array(marg)
        rows.append(dict(h=h, seed=seed, n_bad=int(bad.sum()), n_all=len(u), eps=eps, norm_inc=nd,
                         min_gap=float(gap.min()), ok1=ok1, all_s=bool(np.all(gap > 0)),
                         null_u=float(au[-1]), null_w=float(aw[-1]), null_id=float(abs(a[-1])),
                         min_FD2_minus_FE2_minus_margin=float(np.min(FD ** 2 - FE ** 2 - marg))))
        print(f"{h:<5} {seed}     {int(bad.sum()):3d}/{len(u):3d}  {eps:7.4f}  {nd:7.3f}  {eps / nd:9.4f}  "
              f"{gap.min():12.4f}  {(gap + eps).min():14.4f}   {ok1}   {bool(np.all(gap > 0))}", flush=True)
C.write_rows(os.path.join(HERE, "results", "dirEu_nocancel.csv"), rows)
print("ok")
