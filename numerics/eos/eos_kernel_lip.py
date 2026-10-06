"""How tight are the roughness bounds on how the kernel changes? python eos/eos_kernel_lip.py <run dir> --device cuda.

Bounds (one step theta -> theta' = theta + D, Pi = I, paper norms):
  |sqrt(lambda_k(theta')) - sqrt(lambda_k(theta))| <= rho |D|,        rho   = sup_{|a|=|b|=1} ||D^2 N[a, b]||
  |sigma_P(theta') - sigma_P(theta)| <= rho_P |D|,                    rho_P = sup_{|a|=|b|=1} ||P D^2 N[a, b]||
with sigma_P = sqrt(lambda_max(P K P)), P the projection off the Fourier modes below omega/2 (as in eos_sine.py), and the
suprema over the segment. rho = max_{||phi||=1} ||S(phi)||_op is estimated by alternating power iteration
(phi <- D^2N[a,a] normalized, a <- top eigenvector of S(phi)); the estimate is a lower bound of the supremum at a point,
taken at both ends of the segment. For pairs of consecutive saved states (2000 steps apart) of the chosen networks it
reports the actual changes and the ratio actual / (rho_hat |D|). Output: <run dir>/kernel_lip.csv."""
import argparse
import glob
import json
import os
import sys
import numpy as np
import torch
from torch.func import jvp
from scipy.sparse.linalg import LinearOperator, eigsh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import core as C
import eos_sine as ES
from eos_decompose import jac_scaled, S_times, unflat

torch.set_default_dtype(torch.float64)


def d2N(ps, X1, a):
    """D^2 N[a, a] at the training inputs (vector of n outputs)."""
    aa = unflat(a, ps)
    f = lambda *q: ES.forward(list(q), X1)[0]
    g = lambda *q: jvp(f, tuple(q), aa)[1]
    return jvp(g, tuple(ps), aa)[1]


def rho_est(ps, X1, n, Q=None, iters=6, seed=0):
    d = sum(p.numel() for p in ps)
    a = torch.tensor(np.random.default_rng(seed).standard_normal(d)).to(ps[0].device)
    a = a / a.norm()
    val = 0.0
    for _ in range(iters):
        u = d2N(ps, X1, a).cpu().numpy()
        if Q is not None:
            u = u - Q @ (Q.T @ u)
        nu = np.linalg.norm(u) / np.sqrt(n)
        if nu == 0:
            return 0.0
        phi = u / nu
        mv = lambda v: S_times(ps, X1, torch.tensor(phi), torch.tensor(np.asarray(v, np.float64).ravel())).cpu().numpy()
        op = LinearOperator((d, d), matvec=mv, dtype=np.float64)
        vals, vecs = eigsh(op, k=1, which="LM", tol=1e-6, ncv=20)
        val = abs(float(vals[0]))
        a = torch.tensor(vecs[:, 0]).to(ps[0].device)
    return val                                   # ||S(phi)||_op at the final phi = estimate of rho (or rho_P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--targets", nargs="+", default=["sin1", "sin10", "sin20"])
    ap.add_argument("--lr-rel", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_default_device(a.device)
    d = a.run
    meta = json.load(open(os.path.join(d, "eos_meta.json")))
    ES.set_arch(meta.get("arch", "tanh"))
    runs = [tuple(r) for r in meta["runs"]]
    spec = np.load(os.path.join(d, "eos_spectra.npz"))
    X = torch.tensor(spec["x"])[:, :, None]
    xs = spec["x"]
    B, n = spec["y"].shape
    snaps = sorted(glob.glob(os.path.join(d, "snaps", "params_*.npz")))
    rows = []
    for b, (sd, tg, lr) in enumerate(runs):
        if tg not in a.targets or float(lr) != a.lr_rel or sd != a.seed:
            continue
        Q = ES.band_basis(xs[b], float(tg[3:]) / 2)
        X1 = X[b:b + 1]
        prev = None
        for f in snaps:
            k = int(os.path.basename(f)[7:14])
            z = np.load(f)
            ps = [torch.tensor(z[f"p{i}"][b:b + 1]) for i in range(len(z.files))]
            J = jac_scaled(ps, X1).cpu().numpy()
            Km = J @ J.T / n
            lam = np.linalg.eigvalsh(Km)[::-1]
            Pm = np.eye(n) - Q @ Q.T
            sigP = float(np.sqrt(max(np.linalg.eigvalsh(Pm @ Km @ Pm)[-1], 0.0)))
            th = np.concatenate([p.reshape(-1).cpu().numpy() for p in ps])
            r_all = rho_est(ps, X1, n)
            r_P = rho_est(ps, X1, n, Q=Q)
            cur = dict(step=k, th=th, sq=np.sqrt(np.maximum(lam[[0, 9, 19]], 0)), sigP=sigP, rho=r_all, rhoP=r_P)
            if prev is not None:
                D = float(np.linalg.norm(cur["th"] - prev["th"]))
                rho_hat, rhoP_hat = max(prev["rho"], cur["rho"]), max(prev["rhoP"], cur["rhoP"])
                dsq = np.abs(cur["sq"] - prev["sq"])
                row = dict(target=tg, lr_rel=lr, seed=sd, step0=prev["step"], step1=k, D=D, rho=rho_hat, rhoP=rhoP_hat,
                           bound=rho_hat * D, boundP=rhoP_hat * D,
                           dsqrt_l1=dsq[0], dsqrt_l10=dsq[1], dsqrt_l20=dsq[2], dsigP=abs(cur["sigP"] - prev["sigP"]),
                           ratio_l1=dsq[0] / (rho_hat * D), ratio_l10=dsq[1] / (rho_hat * D),
                           ratio_l20=dsq[2] / (rho_hat * D), ratio_P=abs(cur["sigP"] - prev["sigP"]) / (rhoP_hat * D),
                           sqrt_l1=cur["sq"][0], sqrt_l10=cur["sq"][1], sigP=cur["sigP"])
                rows.append(row)
                print(f"{tg} {prev['step']}->{k}: |D| {D:.3f} rho {rho_hat:.2f} rho_P {rhoP_hat:.3f} | d sqrt(l1) {dsq[0]:.3f} "
                      f"(ratio {row['ratio_l1']:.3f}) d sqrt(l10) {dsq[1]:.3f} ({row['ratio_l10']:.3f}) "
                      f"d sigma_P {row['dsigP']:.3f} ({row['ratio_P']:.3f})", flush=True)
            prev = cur
    C.write_rows(os.path.join(d, "kernel_lip.csv"), rows)
    os._exit(0)


if __name__ == "__main__":
    main()
