"""Certificate for sharpening at the centre of the edge-of-stability oscillation:
python eos/eos_edge_cert.py <run dir> [...] --device cuda [--steps 20 --every 4].

Damian et al. need progressive sharpening (alpha = -grad L . grad S > 0) on the trajectory projected onto the stable
set, not at the oscillating iterates. As a proxy for the projected point this script uses the centre of the
oscillation, the midpoint c = (theta_j + theta_{j+1}) / 2 of two consecutive gradient-descent iterates.
For every network whose eta*lambda_1 reaches 1.9, and every saved checkpoint where eta*lambda_1 >= 1.9, it replays
--steps GD steps and, every --every steps, evaluates at the centre:
  main  = 2 (lambda_1 - k_L)/|theta_L|^2 <Yhat, Y - Yhat>            (output-layer term of Theorem 9.1)
  T_L   = -2 <Psi h_perp, E>                                         (remainder)
  bound = 2 min( min_g ||(K+g)^(-1/2) Psi h_perp|| (||E||_K^2 + g ||E||^2)^(1/2), |h_perp| ||E||_K )
  alpha = main + T_L                                                 (progressive-sharpening coefficient, kernel)
and whether the certificate main > bound holds. Output: <run dir>/edge_certificate.csv."""
import argparse
import csv
import glob
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


def certificate(ps, X1, y, n):
    J = jac_scaled(ps, X1).cpu().numpy()
    lam, U = np.linalg.eigh(J @ J.T / n)
    lam1, v1 = lam[-1], np.sqrt(n) * U[:, -1]
    Yh = ES.forward(ps, X1)[0].detach().cpu().numpy()
    Y = y.cpu().numpy()
    E = Yh - Y
    g_v1 = J.T @ v1 / n
    h = S_times(ps, X1, torch.tensor(v1), torch.tensor(g_v1)).cpu().numpy()
    sizes = [p.numel() for p in ps]
    off = np.concatenate([[0], np.cumsum(sizes)])
    Lidx = np.arange(off[-3], off[-1])
    th = np.concatenate([p.reshape(-1).cpu().numpy() for p in ps])
    thL = th[Lidx]
    kL = float(np.sum((J[:, Lidx].T @ v1 / n) ** 2))
    cL = float(thL @ h[Lidx]) / float(thL @ thL)
    hp = h.copy()
    hp[Lidx] -= cL * thL
    Jhp = J @ hp
    nrm = lambda u: np.linalg.norm(u) / np.sqrt(n)
    main_t = 2 * (lam1 - kL) / float(thL @ thL) * float(Yh @ (Y - Yh)) / n
    T_L = -2 * float(Jhp @ E) / n
    gE = J.T @ E / n
    ak, ek = U.T @ Jhp / np.sqrt(n), U.T @ E / np.sqrt(n)
    lamc = np.maximum(lam, 0.0)
    gams = lam1 * 10.0 ** np.linspace(-10, 2, 121)
    fg = [np.sqrt(np.sum(ak ** 2 / (lamc + g)) * np.sum((lamc + g) * ek ** 2)) for g in gams]
    bound = 2 * min(min(fg), float(np.linalg.norm(hp) * np.linalg.norm(gE)))
    return dict(lam1=lam1, main=main_t, T_L=T_L, bound=bound, alpha=-2 * float(h @ gE),
                certified=bool(main_t > bound), ratio=main_t / bound if bound > 0 else np.nan,
                err=nrm(E), x_top=float(v1 @ E / n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--every", type=int, default=4)
    a = ap.parse_args()
    torch.set_default_device(a.device)
    for d in a.runs:
        meta = json.load(open(os.path.join(d, "eos_meta.json")))
        ES.set_arch(meta.get("arch", "tanh"))
        runs, eta = [tuple(r) for r in meta["runs"]], np.array(meta["eta"])
        spec = np.load(os.path.join(d, "eos_spectra.npz"))
        X = torch.tensor(spec["x"])[:, :, None]
        Yall = torch.tensor(spec["y"])
        B, n = Yall.shape
        log = list(csv.DictReader(open(os.path.join(d, "eos_log.csv"))))
        el = {}
        for r in log:
            el[(int(float(r.get("seed", meta["seeds"][0]))), r["target"], float(r["lr_rel"]), int(float(r["step"])))] = float(r["eta_lam1"])
        rows = []
        for f in sorted(glob.glob(os.path.join(d, "snaps", "params_*.npz"))):
            k = int(os.path.basename(f)[7:14])
            z = np.load(f)
            for b, (sd, tg, lr) in enumerate(runs):
                if el.get((sd, tg, float(lr), k), 0.0) < 1.9:
                    continue
                ps = [torch.tensor(z[f"p{i}"][b:b + 1]) for i in range(len(z.files))]
                X1, y = X[b:b + 1], Yall[b]
                for j in range(a.steps):
                    q = [p.clone().requires_grad_(True) for p in ps]
                    loss = 0.5 * (ES.forward(q, X1)[0] - y).pow(2).mean()
                    gr = torch.autograd.grad(loss, q)
                    nxt = [(p - eta[b] * g).detach() for p, g in zip(q, gr)]
                    if j % a.every == 0:
                        cen = [(p + r) / 2 for p, r in zip(ps, nxt)]
                        c = certificate(cen, X1, y, n)
                        c.update(seed=sd, target=tg, lr_rel=lr, snapshot=k, step=k + j, eta_lam1=eta[b] * c["lam1"])
                        rows.append(c)
                    ps = nxt
            print(f"{d} snapshot {k} done ({len(rows)} rows)", flush=True)
        C.write_rows(os.path.join(d, "edge_certificate.csv"), rows)


if __name__ == "__main__":
    main()
