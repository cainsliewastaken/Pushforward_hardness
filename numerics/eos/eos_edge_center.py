"""Sharpness at the centre of the edge-of-stability oscillation, against self-stabilization and central flows:
python eos/eos_edge_center.py <out dir of eos_sine.py> --target sin10 --lr-rel 1.0 --start 10000 --steps 200

Replays gradient descent from saved parameters. At every step k, with theta_k -> theta_{k+1} the GD step,
the centre is the midpoint c_k = (theta_k + theta_{k+1}) / 2. Records:
  * lambda_1 at the iterate and at the centre, x = <v_1, E> at the iterate, and the half-step along the top
    direction, delta = eta sqrt(lambda_1) |x| / 2 (the distance of each iterate from the centre along w_1);
  * at the centre: grad S = 2 S(v_1) g_{v_1} (S = lambda_1 of K), alpha = -grad S . grad L (progressive sharpening
    rate, Damian et al.), beta = |grad S|^2 (their stabilizing coefficient), and the central-flow variance
    sigma^2 = 2 alpha / beta (Cohen et al., central flows, eq. 13), to compare with delta^2;
  * Damian's predicted change of the sharpness at the centre per step, eta (alpha - beta delta^2 / 2).
Output: <out>/center_<target>_lr<lr>_<start>.csv."""
import argparse
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
from eos_decompose import jac_scaled, S_times

torch.set_default_dtype(torch.float64)


def top(ps, X1, n):
    J = jac_scaled(ps, X1).numpy()
    lam, U = np.linalg.eigh(J @ J.T / n)
    return J, lam[-1], U[:, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--target", default="sin10")
    ap.add_argument("--lr-rel", type=float, default=1.0)
    ap.add_argument("--start", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    meta = json.load(open(os.path.join(a.out, "eos_meta.json")))
    ES.set_arch(meta.get("arch", "tanh"))
    runs = [tuple(r) for r in meta["runs"]]
    b = next(i for i, r in enumerate(runs) if r[1] == a.target and float(r[2]) == a.lr_rel)
    eta = float(meta["eta"][b])
    spec = np.load(os.path.join(a.out, "eos_spectra.npz"))
    X1 = torch.tensor(spec["x"][b:b + 1])[:, :, None]
    y = torch.tensor(spec["y"][b])
    n = y.shape[0]
    z = np.load(os.path.join(a.out, "snaps", f"params_{a.start:07d}.npz"))
    ps = [torch.tensor(z[f"p{i}"][b:b + 1]) for i in range(len(z.files))]
    print(f"{a.target} lr_rel {a.lr_rel} eta {eta:.5g} from step {a.start}", flush=True)
    rows = []
    for k in range(a.steps):
        J, lam1, u1 = top(ps, X1, n)
        Yh = ES.forward(ps, X1)[0].detach().numpy()
        E = Yh - y.numpy()
        x = float(np.sqrt(n) * u1 @ E / n)
        q_ps = [p.clone().requires_grad_(True) for p in ps]
        loss = 0.5 * (ES.forward(q_ps, X1)[0] - y).pow(2).mean()
        gr = torch.autograd.grad(loss, q_ps)
        nxt = [(p - eta * gg).detach() for p, gg in zip(q_ps, gr)]
        cen = [(p + q) / 2 for p, q in zip(ps, nxt)]
        Jc, lamc, uc = top(cen, X1, n)
        vc = np.sqrt(n) * uc
        Ec = ES.forward(cen, X1)[0].detach().numpy() - y.numpy()
        g_vc = Jc.T @ vc / n
        gradS = 2 * S_times(cen, X1, torch.tensor(vc), torch.tensor(g_vc)).numpy()
        gradL = Jc.T @ Ec / n
        alpha = float(-gradS @ gradL)
        beta = float(gradS @ gradS)
        delta = eta * np.sqrt(lam1) * abs(x) / 2
        rows.append(dict(step=a.start + k, lam1_iter=lam1, lam1_center=lamc, eta_lam1_center=eta * lamc, x=x,
                         delta=delta, delta2=delta ** 2, alpha=alpha, beta=beta, sigma2_cf=2 * alpha / beta,
                         damian_dS_step=eta * (alpha - beta * delta ** 2 / 2)))
        ps = nxt
        if k % 20 == 0:
            r = rows[-1]
            print(f"step {r['step']}: eta*lam1 iterate {eta*lam1:.4f} centre {eta*lamc:.4f} | delta^2 {r['delta2']:.3e} "
                  f"sigma^2_CF {r['sigma2_cf']:.3e} | alpha {alpha:+.3e} beta {beta:.3e}", flush=True)
    fn = os.path.join(a.out, f"center_{a.target}_lr{a.lr_rel}_{a.start}.csv")
    C.write_rows(fn, rows)
    lc = np.array([r["lam1_center"] for r in rows])
    d2c = lc[2:] - lc[:-2]                      # two-step change of the centre sharpness (same parity)
    dam = np.array([r["damian_dS_step"] for r in rows])
    damian2 = dam[1:-1] + dam[:-2]
    dl2 = np.array([r["delta2"] for r in rows])
    s2 = np.array([r["sigma2_cf"] for r in rows])
    print(f"centre sharpness two-step change: mean {d2c.mean():+.3e}; Damian prediction mean {damian2.mean():+.3e}; "
          f"corr {np.corrcoef(d2c, damian2)[0, 1]:+.2f}")
    print(f"observed delta^2 median {np.median(dl2):.3e} vs central-flow sigma^2 median {np.median(s2):.3e}")
    print("wrote", fn)


if __name__ == "__main__":
    main()
