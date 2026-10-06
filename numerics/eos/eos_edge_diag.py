"""Where the remainder T_L comes from at the edge: python eos/eos_edge_diag.py <run dir> [...] --device cuda.

At the centre of the oscillation (midpoint of two consecutive GD iterates; centres after 0 and 10 replayed steps
from each checkpoint with eta*lambda_1 >= 1.9) it splits, exactly,
  alpha = main + T_L,  main = 2 (lambda_1 - k_L)/|theta_L|^2 <Yhat, Y - Yhat>,  T_L = -2 h_perp . g_E,
  * T_L by parameter block (h_perp_l . g_l summed within each parameter tensor; for the output layer this is the part
    of h that does not rescale the output layer, i.e. its rotation),
  * T_L and main by eigenband of K (sum over k in a band of -2 a_k e_k, resp. -2 (lambda_1-k_L)/|theta_L|^2 yhat_k e_k),
  * T_L from the slow part E_gamma and from E - E_gamma (gamma = 1e-3 lambda_1),
and records x = <v_1, E>, ||E||, ||E||_K. Output: <run dir>/edge_diag.csv."""
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
BANDS = [(0, 1), (1, 5), (5, 20), (20, 100), (100, None)]
NAMES = ["W1", "b1", "W2", "b2", "Wout", "bout"]


def diag(ps, X1, y, n):
    J = jac_scaled(ps, X1).cpu().numpy()
    lam, U = np.linalg.eigh(J @ J.T / n)
    lam, U = lam[::-1].copy(), U[:, ::-1].copy()
    lam1, v1 = lam[0], np.sqrt(n) * U[:, 0]
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
    aL = (lam1 - kL) / float(thL @ thL)
    cL = float(thL @ h[Lidx]) / float(thL @ thL)
    hp = h.copy()
    hp[Lidx] -= cL * thL
    gE = J.T @ E / n
    main_t = 2 * aL * float(Yh @ (Y - Yh)) / n
    T_L = -2 * float(hp @ gE)
    r = dict(lam1=lam1, main=main_t, T_L=T_L, alpha=main_t + T_L, x=float(v1 @ E / n),
             err=np.linalg.norm(E) / np.sqrt(n), errK=float(np.linalg.norm(gE)))
    for i, nm in enumerate(NAMES[:len(ps)]):
        s = slice(off[i], off[i + 1])
        r[f"T_{nm}"] = -2 * float(hp[s] @ gE[s])
    a = U.T @ (J @ hp) / np.sqrt(n)
    e = U.T @ E / np.sqrt(n)
    yk = U.T @ Yh / np.sqrt(n)
    for lo, hi in BANDS:
        tag = f"{lo + 1}-{hi if hi else 'end'}"
        r[f"T_band_{tag}"] = float(-2 * np.sum(a[lo:hi] * e[lo:hi]))
        r[f"main_band_{tag}"] = float(-2 * aL * np.sum(yk[lo:hi] * e[lo:hi]))
    gam = 1e-3 * lam1
    w = gam / (lam + gam)
    r["T_slow"] = float(-2 * np.sum(a * w * e))
    r["T_fast"] = T_L - r["T_slow"]
    r["main_slow"] = float(-2 * aL * np.sum(yk * w * e))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cpu")
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
        rows = []
        for f in sorted(glob.glob(os.path.join(d, "snaps", "params_*.npz"))):
            k = int(os.path.basename(f)[7:14])
            z = np.load(f)
            for b, (sd, tg, lr) in enumerate(runs):
                if el.get((sd, tg, float(lr), k), 0.0) < 1.9:
                    continue
                ps = [torch.tensor(z[f"p{i}"][b:b + 1]) for i in range(len(z.files))]
                X1, y = X[b:b + 1], Yall[b]
                for j in range(11):
                    q = [p.clone().requires_grad_(True) for p in ps]
                    loss = 0.5 * (ES.forward(q, X1)[0] - y).pow(2).mean()
                    gr = torch.autograd.grad(loss, q)
                    nxt = [(p - eta[b] * g).detach() for p, g in zip(q, gr)]
                    if j in (0, 10):
                        r = diag([(p + s) / 2 for p, s in zip(ps, nxt)], X1, y, n)
                        r.update(seed=sd, target=tg, lr_rel=lr, step=k + j, eta_lam1=eta[b] * r["lam1"])
                        rows.append(r)
                    ps = nxt
            print(f"{d} snapshot {k} done ({len(rows)} rows)", flush=True)
        C.write_rows(os.path.join(d, "edge_diag.csv"), rows)
    os._exit(0)                                          # avoid the hang seen at interpreter shutdown


if __name__ == "__main__":
    main()
