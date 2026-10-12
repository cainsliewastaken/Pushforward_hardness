"""Where the one-step error field lies, relative to the true tangent flow and to the one-step kernel.

For each run (final.pt of train_kappa.py), on held-out attractor points, CPU, float64:

 (A) Decompose eps(x) = G(x) - Gt(x) along the right singular vectors v1 >= v2 >= v3 of the true propagator
     grad S_T(x) over horizon T (most amplified, flow, contracting). Report the rms of each component, the singular
     values, and the amplification of eps by the true flow, rms|grad S_T eps| / rms|eps|, against the isotropic
     reference sqrt(mean s_i^2).

 (B) First-order objective with TRUE propagators along the true trajectory, e1_{q+1} = J^S_q e1_q + eps_q, against the
     measured free error, at the horizons --Ts: rms over starts of |e1(T)|, |e(T)|, and the 0..T averages.

 (C) Kernel Rayleigh quotient of the one-step kernel on the amplified part of eps, phi_1 = <v1, eps> v1, on the rest,
     and on eps itself: r(phi) = |g_phi|^2 / (n |phi|^2) with g_phi = (1/n) sum_i grad_theta Gt(x_i)^T phi_i
     (one VJP per phi). Under gradient flow on the one-step loss this is the rate at which that part of the error
     field is removed per unit of its size; a small r on phi_1 means the amplified part is a slow direction.

Writes <out_root>/kappa/checks/epsdir_<name>.pt per run. New files only.
"""
import argparse
import os
import time

import torch
from torch.func import jacfwd, vmap

from msr.config import load_cfg
from msr.core import Arch, Model, make_data, nsteps


def rms(a):
    return torch.sqrt((a * a).sum(-1).mean(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--Ts", default="0.32,0.64,1.28")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--nsvd", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = "cpu"
    cfg = load_cfg("config.yaml")
    dt = a.dt
    Ts = [float(t) for t in a.Ts.split(",")]

    cks = {r: torch.load(os.path.join(r, "final.pt"), map_location=dev) for r in a.runs}
    stats0 = next(iter(cks.values()))["stats"]
    t0 = time.time()
    data = make_data(cfg, dt, dev, train=False, stats=stats0, horizon=max(Ts))
    tr = data["truth"]
    uT = data["uT_test"][:, :a.n]
    print(f"data: {uT.shape} ({time.time() - t0:.0f}s)", flush=True)
    jacG = vmap(jacfwd(lambda x: tr.G(x, dt)))
    JS = [jacG(uT[q]) for q in range(uT.shape[0] - 1)]            # true per-step Jacobians along the true trajectory

    # true propagators over each T at the first nsvd starts, by products of the per-step Jacobians
    X = uT[0, :a.nsvd]
    frames = {}
    for T in Ts:
        p = nsteps(T, dt)
        P = torch.eye(3, dtype=uT.dtype).expand(a.nsvd, 3, 3).clone()
        for q in range(p):
            P = JS[q][:a.nsvd] @ P
        U, S, Vh = torch.linalg.svd(P)
        frames[T] = dict(P=P, S=S, V=Vh.transpose(-1, -2))         # V[:, :, j] = right singular vector j
        print(f"T={T}: singular values of grad S_T, median {S.median(0).values.tolist()}, "
              f"isotropic amplification {float((S ** 2).mean().sqrt()):.3g}", flush=True)

    out_dir = os.path.join(cfg["out_root"], "kappa", "checks")
    os.makedirs(out_dir, exist_ok=True)
    for r, ck in cks.items():
        name = os.path.basename(os.path.normpath(r))
        arch = Arch(cfg["hidden"], cfg["act"])
        model = Model(arch, "euler", dt, ck["stats"])
        th = ck["theta"].to(dev)
        print(f"\n===== {name} =====", flush=True)
        res = dict(run=r, A={}, B={}, C={})
        with torch.no_grad():
            eps_all = [uT[q + 1] - model.Gt(th, uT[q]) for q in range(uT.shape[0] - 1)]   # (n, 3) per step
            eps0 = eps_all[0][:a.nsvd]
            print(f"rms|eps| on data = {float(rms(eps0)):.3g}")
            # ---- (A)
            print("(A) eps along the singular directions of the true propagator over T (rms over points)")
            print(f"{'T':>5} {'|eps.v1|':>9} {'|eps.v2|':>9} {'|eps.v3|':>9} {'s1 med':>8} {'s2 med':>8} {'s3 med':>8} {'amp(eps)':>9} {'amp(iso)':>9} {'amp/iso':>8}")
            for T in Ts:
                fr = frames[T]
                comp = torch.einsum("nij,ni->nj", fr["V"], eps0)        # (n, 3) components along v1, v2, v3
                amp = rms(torch.einsum("nij,nj->ni", fr["P"], eps0)) / rms(eps0)
                iso = (fr["S"] ** 2).mean().sqrt()
                row = dict(T=T, comp_rms=comp.pow(2).mean(0).sqrt().tolist(), s_med=fr["S"].median(0).values.tolist(),
                           amp=float(amp), amp_iso=float(iso))
                res["A"][T] = row
                c = row["comp_rms"]; s = row["s_med"]
                print(f"{T:5.2f} {c[0]:9.3g} {c[1]:9.3g} {c[2]:9.3g} {s[0]:8.3g} {s[1]:8.3g} {s[2]:8.3g} {row['amp']:9.3g} {row['amp_iso']:9.3g} {row['amp'] / row['amp_iso']:8.3f}", flush=True)
            # ---- (B)
            print("(B) first-order objective with true propagators vs measured free error (rms over starts)")
            print(f"{'T':>5} {'|e1(T)|':>9} {'|e(T)|':>9} {'avg|e1|':>9} {'avg|e|':>9} {'med|e1(T)|':>10} {'med|e(T)|':>10}")
            for T in Ts:
                p = nsteps(T, dt)
                e1, u = torch.zeros_like(uT[0]), uT[0]
                s1, s2, m1, m2 = 0.0, 0.0, None, None
                for q in range(p):
                    e1 = torch.einsum("nab,nb->na", JS[q], e1) + eps_all[q]
                    u = model.Gt(th, u)
                    e = uT[q + 1] - u
                    s1 += float((e1 * e1).sum(-1).mean()); s2 += float((e * e).sum(-1).mean())
                row = dict(T=T, rms_e1=float(rms(e1)), rms_e=float(rms(e)), avg_e1=(s1 / p) ** 0.5, avg_e=(s2 / p) ** 0.5,
                           med_e1=float((e1 * e1).sum(-1).sqrt().median()), med_e=float((e * e).sum(-1).sqrt().median()))
                res["B"][T] = row
                print(f"{T:5.2f} {row['rms_e1']:9.3g} {row['rms_e']:9.3g} {row['avg_e1']:9.3g} {row['avg_e']:9.3g} {row['med_e1']:10.3g} {row['med_e']:10.3g}", flush=True)
        # ---- (C)
        print("(C) one-step kernel Rayleigh quotient r(phi) = |g_phi|^2 / (n |phi|^2) on parts of eps (Pi = I)")
        print(f"{'T':>5} {'r(eps)':>10} {'r(amp part)':>12} {'r(rest)':>10} {'|amp|/|eps|':>12}")
        n = a.nsvd

        def kernel_size2(phi):
            t = th.clone().requires_grad_(True)
            val = (model.Gt(t, X) * phi).sum() / n
            g = torch.autograd.grad(val, t)[0]
            return float((g * g).sum())
        for T in Ts:
            V1 = frames[T]["V"][:, :, 0]
            phi1 = (eps0 * V1).sum(-1, keepdim=True) * V1
            phir = eps0 - phi1
            r = {}
            for lab, phi in (("eps", eps0), ("amp", phi1), ("rest", phir)):
                r[lab] = kernel_size2(phi) / (float((phi * phi).sum()) / n)
            row = dict(T=T, r=r, frac_amp=float(rms(phi1) / rms(eps0)))
            res["C"][T] = row
            print(f"{T:5.2f} {r['eps']:10.3g} {r['amp']:12.3g} {r['rest']:10.3g} {row['frac_amp']:12.3f}", flush=True)
        torch.save(res, os.path.join(out_dir, f"epsdir_{name}.pt"))
        print(f"saved epsdir_{name}.pt ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
