"""Matrix-free diagnostics for saved 2D-turbulence steppers (the KS set of 2026-10-08, without explicit Jacobians).

  python turb2d/diagnostics.py --runs <run dirs> --T 1.0 --n 64

Propagator actions are central finite differences of the truth map G (float64) and of the model (cast to float64).
Per run, on held-out starts, along the true trajectory to T (p steps) and the model's free rollout:
 (1)  |Em(u_pred)| / |eps| along the free rollout (model error at the predicted state against the on-data error).
 (B)  e1 (true propagators), e1m (model propagators) and the measured e; the gap e1m - e1 is the Jacobian-error term.
 (A)  amplification of eps by the true propagator over T, |P eps|/|eps|, against the isotropic sqrt(mean s^2)
      estimated from random unit fields; the same for the model's propagator.
 (M2) coherence of the injections under the true propagators: |sum_i M_i eps_i|^2 / sum_i |M_i eps_i|^2 and the
      lag-1 correlation of consecutive contributions M_i eps_i (forward propagation of each injection, O(p^2)).
 (2a) Jacobian error (J_model - J_true) applied to: the rollout's own deviation direction, a random low-k field
      (|k| < 8) and a random high-k field (|k| > N/4), relative to |(J_true - I) v| / dt.
 (2b) |Em(x + delta v)| for v the deviation direction and the high-k field, against |Em(x)|.
 (M1) one-step kernel Rayleigh quotients at the visited states: r(e), r(sep), r(Em(x)), and r(eps) at true states.
 (D)  top Hessian eigenvalue of the free window loss over that of the one-step loss, on a small batch.
Errors are rms per grid point (vorticity units). Writes <out_root>/turb2d/checks/diag_<name>.pt.
"""
import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ks.config import load_cfg
from turb2d.core import FNO, Model, loss_kappa
from turb2d.solver import Turb2D


def rms(a):  # (n, N, N) -> rms per point over starts
    return torch.sqrt((a * a).flatten(-2).mean(-1).mean(-1))


def pnorm(a):  # per-start rms per point, (n,)
    return torch.sqrt((a * a).flatten(-2).mean(-1))


def fd_jvp(f, u, v, eps=1e-5):
    s = eps * (pnorm(u)[:, None, None] + 1e-30) / (pnorm(v)[:, None, None] + 1e-30)
    return (f(u + s * v) - f(u - s * v)) / (2 * s)


def band_field(N, lo, hi, n, gen, dev):
    """Random real fields with support |k| in [lo, hi), unit rms per point, (n, N, N)."""
    k = torch.fft.fftfreq(N, d=1.0 / N, device=dev)
    K = torch.sqrt(k[:, None] ** 2 + k[None, : N // 2 + 1] ** 2)
    mask = ((K >= lo) & (K < hi)).double()
    z = torch.randn(n, N, N, generator=gen, device=dev)
    zf = torch.fft.rfft2(z) * mask
    x = torch.fft.irfft2(zf, s=(N, N))
    return x / (pnorm(x)[:, None, None] + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cfg", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--nD", type=int, default=8)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--deltas", default="0.03,0.1,0.3,1,3")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.set)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(cfg["threads"])
    cks = {r: torch.load(os.path.join(r, "final.pt"), map_location=dev) for r in a.runs}
    ck0 = next(iter(cks.values()))
    cfgd = ck0["cfg"]
    for k in ("N", "Re", "fkx", "fky", "alpha", "beta", "h", "dt"):
        cfg[k] = cfgd[k]
    ks = Turb2D(cfg["N"], cfg["Re"], cfg["fkx"], cfg["fky"], cfg["alpha"], cfg["beta"], cfg["h"], dev)
    dt, N = cfg["dt"], cfg["N"]
    # held-out trajectories from the cached data of the training run
    import glob
    cache = sorted(glob.glob(os.path.join(cfg["out_root"], "turb2d", "data", "*.pt")), key=os.path.getmtime)
    d = None
    for cpath in reversed(cache):
        dd = torch.load(cpath)
        if dd["uT_test"].shape[-1] == N and dd["uT_test"].shape[0] - 1 >= int(round(a.T / dt)):
            d = dd
            print(f"held-out data from {cpath}", flush=True)
            break
    assert d is not None, "no cached data with this N and horizon; run train.py first"
    p = int(round(a.T / dt))
    uT = d["uT_test"][: p + 1, : a.n].double().to(dev)                  # (p+1, n, N, N)
    n = uT.shape[1]
    t0 = time.time()
    G = lambda x: ks.G(x, dt)
    gen = torch.Generator(device=dev).manual_seed(0)
    deltas = [float(x) for x in a.deltas.split(",")]
    out_dir = os.path.join(cfg["out_root"], "turb2d", "checks")
    os.makedirs(out_dir, exist_ok=True)

    # isotropic amplification of the true propagator over T, from random unit fields (Hutchinson)
    with torch.no_grad():
        xi = torch.randn(n, N, N, generator=gen, device=dev)
        xi = xi / pnorm(xi)[:, None, None]
        v = xi.clone()
        for q in range(p):
            v = fd_jvp(G, uT[q], v)
        amp_iso = float(pnorm(v).pow(2).mean().sqrt())
    print(f"data {tuple(uT.shape)} p={p}; isotropic amplification of the true propagator over T: {amp_iso:.3g} ({time.time() - t0:.0f}s)", flush=True)

    for r, ck in cks.items():
        nm = os.path.basename(os.path.normpath(r))
        cfgr = ck["cfg"]
        arch = FNO(N, cfgr["modes"], cfgr["width"], cfgr["layers"], dev)
        arch.coords = arch.coords.double()
        stats = {k: v.to(dev).double() for k, v in ck["stats"].items()}
        m = Model(arch, dt, stats)
        th = ck["theta"].to(dev).double()
        Gt = lambda x: m.Gt(th, x)
        Em = lambda x: G(x) - Gt(x)
        res = dict(run=nm)
        print(f"\n===== {nm} =====", flush=True)
        with torch.no_grad():
            eps = [uT[q + 1] - Gt(uT[q]) for q in range(p)]
            # ---- free rollout, (1), (B): e, e1 (true prop.), e1m (model prop.)
            u, e1, e1m = uT[0].clone(), torch.zeros_like(uT[0]), torch.zeros_like(uT[0])
            xs, rows = [u.clone()], []
            for q in range(p):
                Emp = Em(u)
                e1 = fd_jvp(G, uT[q], e1) + eps[q]
                e1m = fd_jvp(Gt, uT[q], e1m) + eps[q]
                u = Gt(u)
                xs.append(u.clone())
                e = uT[q + 1] - u
                ratio = pnorm(Emp) / pnorm(eps[q])
                rows.append(dict(t=(q + 1) * dt, rms_eps=float(rms(eps[q])), rms_Em_pred=float(rms(Emp)), med_ratio=float(ratio.median()),
                                 rms_e1=float(rms(e1)), rms_e1m=float(rms(e1m)), rms_e=float(rms(e)), rms_gap=float(rms(e1m - e1)),
                                 med_e1=float(pnorm(e1).median()), med_e1m=float(pnorm(e1m).median()), med_e=float(pnorm(e).median())))
            res["rollout"] = rows
            print("(1)+(B) along the free rollout (rms per point over starts)")
            print(f"{'t':>5} {'|eps|':>8} {'|Em(up)|':>8} {'ratio':>6} {'med rat':>7} {'|e1|':>8} {'|e1m|':>8} {'|e|':>8} {'|e1m-e1|':>8} {'med e1':>8} {'med e':>8}")
            for w in rows:
                if abs(w["t"] * 4 - round(w["t"] * 4)) < 1e-9 or w is rows[-1]:
                    print(f"{w['t']:5.2f} {w['rms_eps']:8.3g} {w['rms_Em_pred']:8.3g} {w['rms_Em_pred'] / w['rms_eps']:6.2f} {w['med_ratio']:7.2f} {w['rms_e1']:8.3g} {w['rms_e1m']:8.3g} {w['rms_e']:8.3g} {w['rms_gap']:8.3g} {w['med_e1']:8.3g} {w['med_e']:8.3g}", flush=True)
            # ---- (A) amplification of eps by true and model propagators over T
            vT, vM = eps[0].clone(), eps[0].clone()
            for q in range(p):
                vT = fd_jvp(G, uT[q], vT)
                vM = fd_jvp(Gt, uT[q], vM)
            ampT, ampM = float(pnorm(vT).pow(2).mean().sqrt() / rms(eps[0])), float(pnorm(vM).pow(2).mean().sqrt() / rms(eps[0]))
            res["A"] = dict(amp_true=ampT, amp_model=ampM, amp_iso=amp_iso)
            print(f"(A) amplification of eps over T: true propagator {ampT:.3g}, model propagator {ampM:.3g}, isotropic {amp_iso:.3g}; amp/iso {ampT / amp_iso:.3f}", flush=True)
            # ---- (M2) coherence of injections under the true propagators (forward propagation of each injection)
            contrib = []
            for i in range(p):
                v = eps[i].clone()
                for q in range(i + 1, p):
                    v = fd_jvp(G, uT[q], v)
                contrib.append(v)
            C = torch.stack(contrib)                                              # (p, n, N, N)
            coh = float((C.sum(0) ** 2).flatten(-2).mean(-1).mean())
            inc = float((C ** 2).flatten(-2).mean(-1).sum(0).mean())
            dots = [(C[i] * C[i + 1]).flatten(-2).mean(-1) for i in range(p - 1)]
            lag1 = float(torch.stack(dots).mean() / (C[:-1] ** 2).flatten(-2).mean(-1).mean().sqrt() / (C[1:] ** 2).flatten(-2).mean(-1).mean().sqrt())
            res["M2"] = dict(coh=coh, inc=inc, ratio=coh / inc, lag1=lag1, rms_e1_check=float(rms(C.sum(0))))
            print(f"(M2) coherence of injections: coherent {coh:.3g} incoherent {inc:.3g} ratio {coh / inc:.3f}; lag-1 corr of contributions {lag1:+.3f} (sum = e1: {float(rms(C.sum(0))):.3g})", flush=True)
            # ---- (2a), (2b)
            X = uT[0]
            dirs = dict(dev=xs[p // 2] - uT[p // 2], lowk=band_field(N, 0.5, 8, n, gen, dev), highk=band_field(N, N / 4, N / 2, n, gen, dev))
            dirs["dev"] = dirs["dev"] / (pnorm(dirs["dev"])[:, None, None] + 1e-30)
            print("(2a) Jacobian error (J_model - J_true) v, relative to |(J_true - I) v|, at the data (rms over starts)")
            res["J"] = {}
            for lab, v in dirs.items():
                jt, jm = fd_jvp(G, X, v), fd_jvp(Gt, X, v)
                num, den = pnorm(jm - jt), pnorm(jt - v)
                res["J"][lab] = dict(rms_err=float(num.pow(2).mean().sqrt()), rms_ref=float(den.pow(2).mean().sqrt()), med_rel=float((num / den).median()))
                print(f"  {lab:>6}: |D1 v| {res['J'][lab]['rms_err']:.3g}  |(J-I) v| {res['J'][lab]['rms_ref']:.3g}  median rel {res['J'][lab]['med_rel']:.3g}", flush=True)
            E0 = float(rms(Em(X)))
            print(f"(2b) model error off the data (on data {E0:.3g}; field rms {float(rms(X)):.3g}): displaced by delta along dev / highk / lowk")
            res["grow"] = {}
            for dl in deltas:
                g = {lab: float(0.5 * (rms(Em(X + dl * v)) + rms(Em(X - dl * v)))) for lab, v in dirs.items()}
                res["grow"][dl] = g
                print(f"  {dl:6.2f}: {g['dev']:8.3g} {g['highk']:8.3g} {g['lowk']:8.3g}   ratios {g['dev'] / E0:6.2f} {g['highk'] / E0:6.2f} {g['lowk'] / E0:6.2f}", flush=True)
        # ---- (M1) kernel rates at the visited states
        def ksize2(phi, Xq):
            t = th.clone().requires_grad_(True)
            val = (m.Gt(t, Xq) * phi).sum() / (n * N * N)
            g = torch.autograd.grad(val, t)[0]
            return float((g * g).sum()) / (float((phi * phi).sum()) / (n * N * N))
        print("(M1) one-step kernel Rayleigh quotients at the visited states x_q")
        print(f"{'t':>5} {'|e|':>8} {'|sep|':>8} {'|Em(x)|':>8} {'r(e)':>9} {'r(sep)':>9} {'r(Em(x))':>9} {'r(eps@uT)':>9}")
        res["M1"] = {}
        for q in sorted(set([max(p // 2, 1), p - 1])):
            Xq = xs[q]
            with torch.no_grad():
                Emx = Em(Xq)
                sep = G(uT[q]) - G(Xq)
                e = uT[q + 1] - xs[q + 1]
            row = dict(q=q, rms_e=float(rms(e)), rms_sep=float(rms(sep)), rms_Em=float(rms(Emx)), r_e=ksize2(e, Xq), r_sep=ksize2(sep, Xq), r_Em=ksize2(Emx, Xq), r_eps=ksize2(eps[q], uT[q]))
            res["M1"][q] = row
            print(f"{(q + 1) * dt:5.2f} {row['rms_e']:8.3g} {row['rms_sep']:8.3g} {row['rms_Em']:8.3g} {row['r_e']:9.3g} {row['r_sep']:9.3g} {row['r_Em']:9.3g} {row['r_eps']:9.3g}", flush=True)
        # ---- (D) curvature ratio (float32 model, as trained)
        th32 = ck["theta"].to(dev)
        arch32 = FNO(N, cfgr["modes"], cfgr["width"], cfgr["layers"], dev)
        m32 = Model(arch32, dt, {k: v.to(dev) for k, v in ck["stats"].items()})
        U = uT[:, : a.nD].float()
        lam = {}
        for lab, kap in (("onestep", 1.0), ("free", None)):
            f_ = lambda x: loss_kappa(m32, x, U, p, kap)

            def hvp(v):
                x = th32.clone().requires_grad_(True)
                gr = torch.autograd.grad(f_(x), x, create_graph=True)[0]
                return torch.autograd.grad(gr @ v, x)[0]
            v = torch.randn(arch32.P, device=dev, dtype=torch.float32)
            v /= v.norm()
            val = 0.0
            for _ in range(a.iters):
                w = hvp(v)
                val = float(w @ v)
                v = w / (w.norm() + 1e-30)
            lam[lab] = val
        res["D"] = dict(T=a.T, lam=lam, ratio=lam["free"] / lam["onestep"])
        print(f"(D) top Hessian eigenvalue over the window ({p} steps, {a.nD} starts): one-step {lam['onestep']:.3g}, free {lam['free']:.3g}, ratio {lam['free'] / lam['onestep']:.3g}", flush=True)
        torch.save(res, os.path.join(out_dir, f"diag_{nm}.pt"))
        print(f"saved diag_{nm}.pt ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
