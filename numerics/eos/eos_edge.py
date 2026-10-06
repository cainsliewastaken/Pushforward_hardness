"""Replay gradient descent at the edge of stability, one step at a time:
python eos/eos_edge.py <out dir of eos_sine.py> --target sin10 --lr-rel 1.0 --start 10000 --steps 200

From the saved parameters at --start, takes --steps GD steps with the run's eta and records at every step:
  * lambda_1 of K, eta * lambda_1, and x = <v_1, E>, the error along the top eigenvector (sign of v_1 kept
    consistent from step to step);
  * the step length s = eta |g| and the error after the step, against F(s), the smallest error any step of
    length s can reach with the features held at the current point (main.tex, Theorem "Supremum over roughness
    functions"), and against the linearized GD error ||(I - eta K) E||; the share of the gap
    ||(I - eta K)E||^2 - F(s)^2 that sits on v_1;
  * the first-order change of lambda_1, -2 eta <q, E>, its part from the error off v_1 (the drive),
    the exact change, and the remainder R = exact - first order;
  * every --curv-every steps, the second derivative of lambda_1 along the unit top parameter direction
    w_1 = J^T v_1 / |J^T v_1|, by central differences at two spacings.
Output: <out>/edge_<target>_lr<lr>_<start>.csv."""
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


def lam1_at(ps, X1, n):
    J = jac_scaled(ps, X1).numpy()
    return float(np.linalg.eigvalsh(J @ J.T / n)[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--target", default="sin10")
    ap.add_argument("--lr-rel", type=float, default=1.0)
    ap.add_argument("--start", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--curv-every", type=int, default=10)
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

    rows, u1_prev, prev = [], None, None
    for k in range(a.steps + 1):
        J = jac_scaled(ps, X1).numpy()
        lam, U = np.linalg.eigh(J @ J.T / n)
        lam, U = lam[::-1].copy(), U[:, ::-1].copy()
        u1 = U[:, 0] * (np.sign(U[:, 0] @ u1_prev) if u1_prev is not None and U[:, 0] @ u1_prev != 0 else 1.0)
        u1_prev = u1
        v1 = np.sqrt(n) * u1
        Yh = ES.forward(ps, X1)[0].detach().numpy()
        E = Yh - y.numpy()
        c = U.T @ E / np.sqrt(n)                 # components <v_k, E>
        x = float(v1 @ E / n)
        err = np.linalg.norm(E) / np.sqrt(n)
        if prev is not None:                     # close the previous row with the exact outcome of its step
            prev["err_after"] = err
            prev["dlam1_exact"] = lam[0] - prev["lam1"]
            prev["remainder"] = prev["dlam1_exact"] - prev["dlam1_first"]
            prev["F_ratio"] = (prev["err_after"] - prev["F"]) / max(prev["err"] - prev["F"], 1e-300)
            rows.append(prev)
        if k == a.steps:
            break
        g = J.T @ E / n
        s = eta * float(np.linalg.norm(g))
        F = C.F_closed(np.append(lam, 0.0), np.append(c, 0.0), s)[0]
        lin = np.sqrt(np.sum((1 - eta * lam) ** 2 * c ** 2))
        gam = C.solve_gamma(np.append(lam, 0.0), np.append(c, 0.0), s)
        keep_best = gam / (lam[0] + gam) if gam not in (None, np.inf) else np.nan
        gap = lin ** 2 - F ** 2
        gap_top = ((1 - eta * lam[0]) ** 2 - keep_best ** 2) * c[0] ** 2
        g_v1 = J.T @ v1 / n
        h = S_times(ps, X1, torch.tensor(v1), torch.tensor(g_v1)).numpy()
        q = J @ h
        first = float(-2 * eta * (q @ E) / n)
        drive = float(-2 * eta * (q @ (E - x * v1)) / n)
        r = dict(step=a.start + k, lam1=lam[0], eta_lam1=eta * lam[0], x=x, x2=x * x, err=err,
                 s=s, F=F, lin=lin, gamma=gam if gam is not None else np.nan, keep_gd=1 - eta * lam[0],
                 keep_best=keep_best, gap_share_top=gap_top / gap if gap > 0 else np.nan,
                 dlam1_first=first, dlam1_drive=drive, dlam1_osc=first - drive)
        if k % a.curv_every == 0:
            w1 = g_v1 / np.linalg.norm(g_v1)
            st = max(eta * np.sqrt(lam[0]) * abs(x), 1e-6)   # size of the step's move along w_1
            r["step_along_w1"] = st
            for tag, mult in (("a", 1.0), ("b", 3.0)):
                ep = mult * st
                sh = []
                i0 = 0
                for p in ps:
                    m = p.numel()
                    sh.append(torch.tensor(w1[i0:i0 + m]).reshape(p.shape))
                    i0 += m
                lp = lam1_at([p + ep * d for p, d in zip(ps, sh)], X1, n)
                lm = lam1_at([p - ep * d for p, d in zip(ps, sh)], X1, n)
                r[f"d2lam1_w1_{tag}"] = (lp + lm - 2 * lam[0]) / ep ** 2
        prev = r
        # the GD step itself
        q_ps = [p.clone().requires_grad_(True) for p in ps]
        loss = 0.5 * (ES.forward(q_ps, X1)[0] - y).pow(2).mean()
        gr = torch.autograd.grad(loss, q_ps)
        ps = [(p - eta * gg).detach() for p, gg in zip(q_ps, gr)]
        if k % 20 == 0:
            print(f"step {r['step']}: eta*lam1 {r['eta_lam1']:.3f} x {x:+.3e} err {err:.4g} | GD keeps {r['keep_gd']:+.2f} "
                  f"best keeps {keep_best:.2f} on v1, gap share on v1 {r['gap_share_top']:.2f} | dlam1 first "
                  f"{first:+.2e} (drive {drive:+.2e})" + (f" | d2lam1/dw1^2 {r['d2lam1_w1_a']:+.3e} / {r['d2lam1_w1_b']:+.3e}"
                                                            if "d2lam1_w1_a" in r else ""), flush=True)
    fn = os.path.join(a.out, f"edge_{a.target}_lr{a.lr_rel}_{a.start}.csv")
    C.write_rows(fn, rows)
    print("wrote", fn)


if __name__ == "__main__":
    main()
