"""Forced growth of the slow block of the kernel: python eos/eos_slow_forced.py <run dir> [--snaps 0 4000 10000 20000].

Theorem dataset along the gradient-descent path, with the seminorm R(u) = ||P u||, P the projection on the eigenvectors
v_j (j >= k) of the kernel at initialization K_0 (eigenvalues below gamma_k = lambda_k(K_0)): the output change
Yhat_t - Yhat_0 = sum_s int_0^1 Psi(theta_s + tau Delta_s) Delta_s dtau, so
    max over the path of lambda_max(P K P)  >=  ( ||P (Yhat_t - Yhat_0)|| / L_t )^2  =: demand,
with L_t the path length, while lambda_max(P K_0 P) = lambda_k(K_0) =: initial.
At each listed snapshot this script reports initial, demand, and the actual lambda_max(P K_t P).
Output: <run dir>/slow_forced.csv."""
import argparse
import csv
import json
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import core as C
import eos_sine as ES
from eos_decompose import jac_scaled

torch.set_default_dtype(torch.float64)
KS = [5, 10, 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--snaps", type=int, nargs="+", default=[0, 4000, 10000, 20000])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_default_device(a.device)
    d = a.run
    meta = json.load(open(os.path.join(d, "eos_meta.json")))
    ES.set_arch(meta.get("arch", "tanh"))
    runs = [tuple(r) for r in meta["runs"]]
    spec = np.load(os.path.join(d, "eos_spectra.npz"))
    X = torch.tensor(spec["x"])[:, :, None]
    B, n = spec["y"].shape
    path = {}
    for r in csv.DictReader(open(os.path.join(d, "eos_log.csv"))):
        path[(r["target"], float(r["lr_rel"]), int(float(r["step"])))] = float(r["path"])
    Z = {k: np.load(os.path.join(d, "snaps", f"params_{k:07d}.npz")) for k in a.snaps}
    rows = []
    for b, (sd, tg, lr) in enumerate(runs):
        X1 = X[b:b + 1]
        K, Yh = {}, {}
        for k in a.snaps:
            ps = [torch.tensor(Z[k][f"p{i}"][b:b + 1]) for i in range(len(Z[k].files))]
            J = jac_scaled(ps, X1).cpu().numpy()
            K[k] = J @ J.T / n
            Yh[k] = ES.forward(ps, X1)[0].detach().cpu().numpy()
        lam0, U0 = np.linalg.eigh(K[a.snaps[0]])
        lam0, U0 = lam0[::-1], U0[:, ::-1]
        for kk in KS:
            Uq = U0[:, kk - 1:]
            P = Uq @ Uq.T
            for k in a.snaps[1:]:
                L = path.get((tg, float(lr), k), np.nan)
                Pd = P @ (Yh[k] - Yh[a.snaps[0]])
                demand = (np.linalg.norm(Pd) / np.sqrt(n) / L) ** 2
                actual = float(np.linalg.eigvalsh(P @ K[k] @ P)[-1])
                rows.append(dict(seed=sd, target=tg, lr_rel=lr, k=kk, step=k, path=L, initial=lam0[kk - 1],
                                 slow_out=np.linalg.norm(Pd) / np.sqrt(n), demand=demand, actual=actual,
                                 forced_growth=demand / lam0[kk - 1], actual_growth=actual / lam0[kk - 1],
                                 tight=demand / actual))
                r = rows[-1]
                print(f"{tg:6s} lr {lr} k {kk:2d} step {k:6d} | initial {r['initial']:.2e} demand {demand:.2e} actual {actual:.2e} "
                      f"| forced x{r['forced_growth']:.1e} actual x{r['actual_growth']:.1e} demand/actual {r['tight']:.2e}", flush=True)
    C.write_rows(os.path.join(d, "slow_forced.csv"), rows)
    os._exit(0)


if __name__ == "__main__":
    main()
