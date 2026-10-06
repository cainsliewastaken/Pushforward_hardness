"""Error parked in the top mode at the edge of stability: python eos/eos_edge_park.py <run dir> [...] --device cuda.

For every network and saved checkpoint with eta*lambda_1 >= 1.9, replay --steps gradient-descent steps from the saved
parameters. At step j, with centre c_j = (theta_j + theta_{j+1}) / 2 and lambda_1, v_1 of K(c_j):
  osc_j    = (e1(theta_j) - e1(theta_{j+1})) / 2,  e1(.) = <v_1(c_j), E(.)>      measured top-mode oscillation, = sqrt(lambda_1) x
  share_j  = e1(theta_j)^2 / ||E(theta_j)||^2                                     share of the squared error in the top mode
  pred_j   = lambda_1 alpha / (2 |h|^2),  alpha = -2 h . g_E,  h = S(v_1) g_{v_1}  (all at c_j)
pred is the balance of the self-stabilization of Damian et al. in kernel terms: the drift changes lambda_1 by eta*alpha per
step and the oscillation by -(eta/2) x^2 |grad lambda_1|^2 with grad lambda_1 = 2h, so x^2 = 2 alpha / (4|h|^2) and
osc^2 = lambda_1 x^2. Both sides are averaged over the replay (the balance is an average over the oscillation).
Output: <run dir>/edge_park.csv (one row per checkpoint) and edge_park_steps.csv (one row per step)."""
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


def gd_step(ps, X1, y, eta):
    q = [p.clone().requires_grad_(True) for p in ps]
    loss = 0.5 * (ES.forward(q, X1)[0] - y).pow(2).mean()
    gr = torch.autograd.grad(loss, q)
    return [(p - eta * g).detach() for p, g in zip(q, gr)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=40)
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
        el = {}
        for r in csv.DictReader(open(os.path.join(d, "eos_log.csv"))):
            el[(int(float(r.get("seed", meta["seeds"][0]))), r["target"], float(r["lr_rel"]), int(float(r["step"])))] = float(r["eta_lam1"])
        rows, srows = [], []
        for f in sorted(glob.glob(os.path.join(d, "snaps", "params_*.npz"))):
            k = int(os.path.basename(f)[7:14])
            z = np.load(f)
            for b, (sd, tg, lr) in enumerate(runs):
                if el.get((sd, tg, float(lr), k), 0.0) < 1.9:
                    continue
                ps = [torch.tensor(z[f"p{i}"][b:b + 1]) for i in range(len(z.files))]
                X1, y = X[b:b + 1], Yall[b]
                Ynp = y.cpu().numpy()
                Ecur = ES.forward(ps, X1)[0].detach().cpu().numpy() - Ynp
                osc2, pred, share, el1 = [], [], [], []
                for j in range(a.steps):
                    nxt = gd_step(ps, X1, y, eta[b])
                    Enext = ES.forward(nxt, X1)[0].detach().cpu().numpy() - Ynp
                    cen = [(p + r) / 2 for p, r in zip(ps, nxt)]
                    J = jac_scaled(cen, X1).cpu().numpy()
                    lam, U = np.linalg.eigh(J @ J.T / n)
                    lam1, v1 = lam[-1], np.sqrt(n) * U[:, -1]
                    Ec = ES.forward(cen, X1)[0].detach().cpu().numpy() - Ynp
                    g_v1, gE = J.T @ v1 / n, J.T @ Ec / n
                    h = S_times(cen, X1, torch.tensor(v1), torch.tensor(g_v1)).cpu().numpy()
                    alpha = -2 * float(h @ gE)
                    p_j = lam1 * alpha / (2 * float(h @ h))
                    e1a, e1b = float(v1 @ Ecur) / n, float(v1 @ Enext) / n
                    o_j = (e1a - e1b) / 2
                    s_j = e1a ** 2 / (float(Ecur @ Ecur) / n)
                    osc2.append(o_j ** 2); pred.append(p_j); share.append(s_j); el1.append(eta[b] * lam1)
                    srows.append(dict(seed=sd, target=tg, lr_rel=lr, snapshot=k, j=j, eta_lam1=eta[b] * lam1, osc=o_j,
                                      osc2=o_j ** 2, pred=p_j, alpha=alpha, h2=float(h @ h), share_top=s_j,
                                      err=np.sqrt(float(Ecur @ Ecur) / n), e1_centre=float(v1 @ Ec) / n))
                    ps, Ecur = nxt, Enext
                mo, mp = float(np.mean(osc2)), float(np.mean(pred))
                rows.append(dict(seed=sd, target=tg, lr_rel=lr, snapshot=k, eta_lam1=float(np.mean(el1)),
                                 osc2_mean=mo, pred_mean=mp, ratio=mo / mp if mp > 0 else np.nan,
                                 frac_alpha_neg=float(np.mean(np.array(pred) < 0)), share_top_mean=float(np.mean(share)),
                                 err=srows[-1]["err"]))
                r = rows[-1]
                print(f"{tg:6s} lr {lr} seed {sd} snap {k:6d} eta*lam1 {r['eta_lam1']:.3f} | osc^2 {mo:.3e} pred {mp:.3e} "
                      f"ratio {r['ratio']:.3f} | top share of error^2 {r['share_top_mean']:.3f} alpha<0 {r['frac_alpha_neg']:.2f}",
                      flush=True)
        C.write_rows(os.path.join(d, "edge_park.csv"), rows)
        C.write_rows(os.path.join(d, "edge_park_steps.csv"), srows)
    os._exit(0)


if __name__ == "__main__":
    main()
