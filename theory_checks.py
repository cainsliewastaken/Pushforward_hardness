"""Checks (v) and (i) of multistep_rollout_kernel.tex (notation of that draft; Pi = I).

(v) One training step on the rollout loss L_w = (1/2) sum_q w_q ||e_q||_n^2, w_q = 1/p (window tau = p dt), on
    n held-out starts. With b(theta) = W^(1/2) e / sqrt(n) (stacked errors, so |b| = ||e||_w) and A = -db/dtheta
    (= W^(1/2) Psi^roll / sqrt(n)), M = A A^T is the weighted rollout kernel, formed explicitly (one JVP per VJP).
    First-order value F(s) = min_{|g|<=s} |b - A g| via the ridge path g(gamma) = A^T (M + gamma)^(-1) b, from the
    eigendecomposition of M (Theorem "best rollout loss after one training step", first-order version).
    Realized: |b(theta + g)| for actual steps g: gradient descent (g ~ A^T b) and Adam's first step from a fresh state
    (g = lr sign(A^T b)) at several lr, and the first-order optimal step g(gamma) of the same size.
(i) Euler models: ell = max over visited states of mu(grad H) + (dt/2)||grad H||^2, h = max ||grad H||, and
    s_i^2 = lambda_max(K_{H,i}) along the rollouts of n held-out starts; the bounds of Theorem "rate",
    (dt sum_{i<q} e^{ell (q-i-1) dt} s_i)^2 and e^{2|ell| dt} phi(ell, tau_q)^2 max_i s_i^2, against the measured
    lambda_max(K^(q)) (meas.pt, same first n held-out starts).

  python theory_checks.py --part v --runs dt0.08_onestep ... [--windows 0.32 0.64 1.28] [--seeds 0]
  python theory_checks.py --part i --runs dt0.01_onestep ...
Writes <out_root>/theory_checks_<part>.pt.
"""
import argparse
import math
import os

import numpy as np
import torch
from torch.func import jvp, vjp, vmap

from msr.config import load_cfg
from msr.core import Arch, Model, make_data, nsteps


def load_run(cfg, run, dev, horizon):
    ck = torch.load(os.path.join(cfg["out_root"], "runs", run, "ckpts.pt"), map_location=dev, weights_only=False)
    data = make_data(cfg, ck["dt"], dev, train=False, stats={k: v.to(dev) for k, v in ck["stats"].items()},
                     horizon=horizon)
    return ck, data, Arch(ck["cfg"]["hidden"], ck["cfg"]["act"])


def check_v(m, th, uT, p, chunk, lrs):
    n = uT.shape[1]
    th = th.detach()

    def bf(t):
        return ((uT[1:p + 1] - m.rollout(t, uT[0], p)[1:]) / math.sqrt(p * n)).reshape(-1)

    b0, vjp_fn = vjp(bf, th)
    N = b0.numel()
    M = torch.empty(N, N, device=th.device)
    eye = torch.eye(N, device=th.device)
    jv = vmap(lambda v: jvp(bf, (th,), (v,))[1])
    for s in range(0, N, chunk):
        V = vmap(vjp_fn)(eye[s:s + chunk])[0]               # rows (db/dtheta)^T e_j
        M[:, s:s + chunk] = jv(V).T                         # (db/dtheta)(db/dtheta)^T e_j = M e_j
    M = 0.5 * (M + M.T)
    lam, U = torch.linalg.eigh(M)
    lam = lam.clamp_min(0)
    beta = U.T @ b0
    E0 = float(b0.norm())

    def path(gam):
        F = torch.sqrt((gam ** 2 * beta ** 2 / (lam + gam) ** 2).sum())
        s = torch.sqrt((lam * beta ** 2 / (lam + gam) ** 2).sum())
        return float(s), float(F)

    lmax = float(lam[-1])
    gams = np.logspace(math.log10(lmax) + 3, math.log10(lmax) - 16, 240)
    curve = np.array([path(torch.tensor(g, device=th.device)) for g in gams])   # columns s, F

    def F_of_s(s):                     # gammas decrease along the grid, so s increases and F decreases
        return float(np.interp(s, curve[:, 0], curve[:, 1]))     # clamps to F(0) = |b| and to min F

    def g_opt(s):                                           # first-order optimal step of size s
        i = int(np.argmin(np.abs(np.log(curve[:, 0] + 1e-300) - math.log(s))))
        gam = torch.tensor(gams[i], device=th.device)
        return -vjp_fn(U @ (beta / (lam + gam)))[0]         # A^T (M + gam)^(-1) b

    AtB = -vjp_fn(b0)[0]                                    # A^T b = -grad L: the gradient-descent direction
    real = lambda g: float(bf(th + g).norm())
    lin = lambda g: float((b0 + jvp(bf, (th,), (g,))[1]).norm())
    rows = []
    for lr in lrs:
        g_adam = lr * torch.sign(AtB)
        s = float(g_adam.norm())
        g_gd = AtB * (s / float(AtB.norm()))
        go = g_opt(s)
        rows.append(dict(lr=lr, s=s, F=F_of_s(s) / E0,
                         adam_real=real(g_adam) / E0, adam_lin=lin(g_adam) / E0,
                         gd_real=real(g_gd) / E0, gd_lin=lin(g_gd) / E0,
                         opt_real=real(go) / E0, opt_lin=lin(go) / E0, opt_s=float(go.norm())))
    return dict(E0=E0, N=N, lam_max=lmax, tr_over_lam=float(lam.sum()) / lmax, lam=lam.cpu().numpy(),
                curve=curve, steps=rows)


def check_i(m, th, uT, qs, dev):
    n, dt = uT.shape[1], m.dt
    qmax = max(qs)
    th = th.detach()
    with torch.no_grad():
        up = m.rollout(th, uT[0], qmax)
    I3 = torch.eye(3, device=dev)
    ell_i, h_i, s2_i = [], [], []
    eye = torch.eye(3 * n, device=dev)
    for i in range(qmax):
        with torch.no_grad():
            A = (m.Jstate(th, up[i]) - I3) / dt               # grad H at the visited states (Euler)
            mu = torch.linalg.eigvalsh(0.5 * (A + A.transpose(-1, -2)))[:, -1]
            nrm = torch.linalg.matrix_norm(A, ord=2)
        ell_i.append(float((mu + 0.5 * dt * nrm ** 2).max()))
        h_i.append(float(nrm.max()))
        _, vf = vjp(lambda t: m.H(t, up[i]).reshape(-1), th)
        Th = vmap(vf)(eye)[0]                               # (3n, P): rows of the network features Theta_i
        s2_i.append(float(torch.linalg.eigvalsh(Th @ Th.T / n)[-1]))
    ell_i, h_i, s_i = np.array(ell_i), np.array(h_i), np.sqrt(np.array(s2_i))
    out = []
    for q in qs:
        ell = float(ell_i[:q].max())
        tau = q * dt
        phi = (math.exp(ell * tau) - 1) / ell if ell != 0 else tau
        mid = (dt * sum(math.exp(ell * (q - i - 1) * dt) * s_i[i] for i in range(q))) ** 2
        top = math.exp(2 * abs(ell) * dt) * phi ** 2 * float((s_i[:q] ** 2).max())
        out.append(dict(q=q, tau=tau, ell=ell, h=float(h_i[:q].max()), s2_max=float((s_i[:q] ** 2).max()),
                        bound_mid=mid, bound_top=top))
    return dict(rows=out, ell_i=ell_i, h_i=h_i, s_i=s_i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--part", choices=["v", "i"], required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--windows", nargs="*", type=float, default=[0.32, 0.64, 1.28])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--classes", nargs="*", default=["euler", "direct"])
    ap.add_argument("--lrs", nargs="*", type=float, default=[1e-6, 1e-5, 1e-4])
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.set)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    outf = os.path.join(cfg["out_root"], f"theory_checks_{a.part}.pt")
    res = torch.load(outf, weights_only=False) if os.path.exists(outf) else {}
    for run in a.runs:
        hz = max(a.windows) if a.part == "v" else max(cfg["horizons"])
        ck, data, arch = load_run(cfg, run, dev, hz)
        uT = data["uT_test"][:, :a.n]
        final = max(ck["theta"])
        meas = None
        if a.part == "i":
            mf = os.path.join(cfg["out_root"], "runs", run, "meas.pt")
            meas = torch.load(mf, weights_only=False) if os.path.exists(mf) else None
        for j, (cls, seed) in enumerate(ck["names"]):
            if seed not in a.seeds or cls not in a.classes or (a.part == "i" and cls != "euler"):
                continue
            m = Model(arch, cls, ck["dt"], data["stats"])
            th = ck["theta"][final][j]
            if a.part == "v":
                for w in a.windows:
                    r = check_v(m, th, uT, nsteps(w, ck["dt"]), a.chunk, a.lrs)
                    res[(run, cls, seed, w)] = r
                    st = " | ".join(f"lr {x['lr']:.0e}: F {x['F']:.4f} adam {x['adam_real']:.4f} gd {x['gd_real']:.4f} "
                                    f"opt {x['opt_real']:.4f}" for x in r["steps"])
                    print(f"{run} {cls} s{seed} window {w}: N={r['N']} lam_max {r['lam_max']:.3g} "
                          f"tr/lam {r['tr_over_lam']:.3g} || {st}", flush=True)
            else:
                qs = [nsteps(T, ck["dt"]) for T in cfg["horizons"]]
                r = check_i(m, th, uT, qs, dev)
                if meas is not None and (cls, seed, final) in meas:
                    for row, lm in zip(r["rows"], meas[(cls, seed, final)]["K_lam_max"]):
                        row["lam_meas"] = float(lm)
                res[(run, cls, seed)] = r
                print(f"{run} {cls} s{seed}: " + " | ".join(
                    f"tau {x['tau']:.2f}: ell {x['ell']:.3g} meas {x.get('lam_meas', float('nan')):.3g} "
                    f"mid {x['bound_mid']:.3g} top {x['bound_top']:.3g}" for x in r["rows"]), flush=True)
            torch.save(res, outf)
    print(f"wrote {outf}")


if __name__ == "__main__":
    main()
