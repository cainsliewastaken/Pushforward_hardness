"""What moves lambda_1: python eos/eos_decompose.py <out dir of eos_sine.py> [--gamma-delta 1000].

At every saved parameter snapshot of an eos_sine.py run, for each network:
  * lambda_1, v_1 of K (explicit), the error E, the gradient g_E = (1/n) J^T E;
  * the rate of change of lambda_1 under gradient flow, d lambda_1/dt = -2 g_{v1} . S(v1) g_E, where
    g_{v1} = (1/n) J^T v_1 and S(v1) is the Hessian (in the parameters) of <v_1, N>; reported per GD
    step (times eta). It is linear in E, so it splits exactly into the part driven by the slow part
    E_gamma = gamma (K + gamma I)^{-1} E, gamma = 1 / (eta * gamma_delta), and the part driven by E - E_gamma;
  * the measured change of lambda_1 per step from eos_log.csv (next log minus this one, divided by the
    number of steps) for comparison;
  * the two terms of d ln(rho)/dt, rho = <E, K E> / ||E||^2: the fixed-kernel term -2 Var_E(lambda) / rho and
    the kernel-change term <E, dK/dt E> / <E, K E> = -2 g_E . S(E) g_E / |g_E|^2 (both per GD step).
Output: <out>/decompose.csv."""
import argparse
import csv
import glob
import json
import os
import sys
import numpy as np
import torch
from torch.func import grad, jvp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import core as C
import sine_deep as SD
import eos_sine as ES

torch.set_default_dtype(torch.float64)


def jac_scaled(ps, X1):
    from torch.func import jacrev
    n = X1.shape[1]
    f1 = lambda *q: ES.forward(list(q), X1)[0]
    Js = jacrev(f1, argnums=tuple(range(len(ps))))(*ps)
    return torch.cat([j.reshape(n, -1) for j in Js], 1)  # J, (n, d)


def unflat(v, ps):
    out, i = [], 0
    for p in ps:
        out.append(v[i:i + p.numel()].reshape(p.shape))
        i += p.numel()
    return tuple(out)


def S_times(ps, X1, phi, vec):
    """S(phi) vec, S(phi) = Hessian of <phi, N> = (1/n) phi . N in the parameters."""
    n = X1.shape[1]
    f = lambda *q: (ES.forward(list(q), X1)[0] * phi).sum() / n
    gf = grad(f, argnums=tuple(range(len(ps))))
    hv = jvp(lambda *q: gf(*q), tuple(ps), unflat(vec, ps))[1]
    return torch.cat([h.reshape(-1) for h in hv])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--gamma-delta", type=float, default=1000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_default_device(a.device)
    meta = json.load(open(os.path.join(a.out, "eos_meta.json")))
    ES.set_arch(meta.get("arch", "tanh"))
    runs, eta = [tuple(r) for r in meta["runs"]], np.array(meta["eta"])
    spec = np.load(os.path.join(a.out, "eos_spectra.npz"))
    X = torch.tensor(spec["x"])[:, :, None]
    Y = torch.tensor(spec["y"])
    B, n = Y.shape
    logrows = list(csv.DictReader(open(os.path.join(a.out, "eos_log.csv"))))
    lam_log = {}
    for r in logrows:
        lam_log[(r["target"], float(r["lr_rel"]), int(r["step"]))] = float(r["lam1"])
    steps_log = sorted({int(r["step"]) for r in logrows})
    rows = []
    for f in sorted(glob.glob(os.path.join(a.out, "snaps", "params_*.npz"))):
        k = int(os.path.basename(f)[7:14])
        z = np.load(f)
        P = [torch.tensor(z[f"p{i}"]) for i in range(len(z.files))]
        for b in range(B):
            ps = [p[b:b + 1] for p in P]
            X1 = X[b:b + 1]
            J = jac_scaled(ps, X1)
            Km = (J @ J.T / n).cpu().numpy()
            lam, U = np.linalg.eigh(Km)
            lam, U = lam[::-1].copy(), U[:, ::-1].copy()
            out = ES.forward(ps, X1)[0].detach()
            E = (out - Y[b]).cpu().numpy()
            gam = 1.0 / (eta[b] * a.gamma_delta)
            c = U.T @ E
            Eg = U @ (gam / (lam + gam) * c)
            v1 = np.sqrt(n) * U[:, 0]                       # ||v1|| = 1 in <a,b> = a.b/n
            Jt = J.cpu().numpy().T
            g_v1 = Jt @ v1 / n
            gE, gEg = Jt @ E / n, Jt @ Eg / n
            Sv = S_times(ps, X1, torch.tensor(v1), torch.tensor(g_v1)).cpu().numpy()
            rate = lambda gg: -2 * float(Sv @ gg) * eta[b]   # per GD step
            r_tot, r_slow = rate(gE), rate(gEg)
            w = c ** 2 / np.sum(c ** 2)
            rho = float(np.sum(w * lam))
            var = float(np.sum(w * lam ** 2) - rho ** 2)
            SE = S_times(ps, X1, torch.tensor(E), torch.tensor(gE)).cpu().numpy()
            kdot = -2 * float(gE @ SE) / float(gE @ gE)
            # exact one-step change: take the actual GD step theta' = theta - eta g_E and recompute lambda_1
            step = unflat(torch.tensor(gE), ps)
            ps1 = [p - eta[b] * s for p, s in zip(ps, step)]
            J1 = jac_scaled(ps1, X1)
            lam1_next = float(np.linalg.eigvalsh((J1 @ J1.T / n).cpu().numpy())[-1])
            meas = lam1_next - lam[0]
            rows.append(dict(seed=runs[b][0], target=runs[b][1], omega=float(runs[b][1][3:]), lr_rel=runs[b][2], step=k,
                             err=np.linalg.norm(E) / np.sqrt(n), slow=np.linalg.norm(Eg) / np.sqrt(n),
                             lam1=lam[0], eta_lam1=eta[b] * lam[0],
                             dlam1_pred=r_tot, dlam1_from_slow=r_slow, dlam1_from_rest=r_tot - r_slow,
                             dlam1_exact_step=meas, remainder=meas - r_tot,
                             remainder_rel=(meas - r_tot) / abs(meas) if meas != 0 else np.nan, rho_step=eta[b] * rho,
                             slowdown_fixedK_step=-2 * eta[b] * var / rho if rho > 0 else np.nan,
                             kernel_change_step=eta[b] * kdot))
            rr = rows[-1]
            print(f"{rr['target']:6s} lr {rr['lr_rel']} step {k:6d} err {rr['err']:.3g} eta*lam1 {rr['eta_lam1']:.2f} | "
                  f"dlam1/step 1st-order {r_tot:+.3e} (slow {r_slow:+.3e}) exact {meas:+.3e} | "
                  f"d ln rho/step: fixedK {rr['slowdown_fixedK_step']:+.2e} kernel {rr['kernel_change_step']:+.2e}",
                  flush=True)
    C.write_rows(os.path.join(a.out, "decompose.csv"), rows)


if __name__ == "__main__":
    main()
