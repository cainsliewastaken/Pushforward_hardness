"""The three diagnostics (plus the curvature ratio) on saved KS steppers, as in the Lorenz-63 checks of 2026-10-08
(offdata_checks.py, eps_directions.py), with the attractor frame taken from the singular directions of the true
propagator grad S_T (amplified s > 1, neutral, contracting) instead of local PCA.

  python ks/diagnostics.py --runs <run dir> [...] --Ts 5,10,20,40

Per run and horizon T (time units):
 (1) |Em(u_pred)| / |eps| along the free rollout: Em(x) = G(x) - Gt(x) at the predicted state against the one-step
     error at the true state. Ratio 1: the rollout stays where the model's error is its on-attractor error.
 (A) eps in the singular frame of grad S_T at the start: fraction of |eps|^2 on the amplified directions (s > 1) and
     on the top one; amplification rms|grad S_T eps| / rms|eps| against the isotropic sqrt(mean s^2).
 (B) first-order objective with true propagators, e1_{q+1} = J^S_q e1_q + eps_q, against the measured free error.
 (C) one-step kernel Rayleigh quotient |g_phi|^2 / (n |phi|^2) on the amplified part of eps and on the rest.
 (2a) Jacobian error (grad Gt - grad G)/dt applied to the most amplified, a neutral and the most contracting unit
     direction of grad S_T at the start, against |(grad G - I)/dt v|.
 (2b) |Em(x + delta v)| for v the most amplified and the most contracting direction, against |Em(x)|.
 (D) top Hessian eigenvalue of the free window loss (p = T_D/dt) over that of the one-step loss, on a batch.
Writes <out_root>/ks/checks/diag_<name>.pt. New files only.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ks.config import load_cfg
from ks.core import Arch, Model, loss_kappa, make_data


def rms(a):
    return torch.sqrt((a * a).sum(-1).mean(-1))


def top_eig(hvp, P, dev, iters, gen):
    v = torch.randn(P, device=dev, generator=gen)
    v /= v.norm()
    lam = 0.0
    for _ in range(iters):
        w = hvp(v)
        lam = float(w @ v)
        v = w / (w.norm() + 1e-300)
    return lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cfg", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    ap.add_argument("--Ts", default="5,10,20,40")
    ap.add_argument("--n", type=int, default=None, help="held-out starts (default cfg n_eval)")
    ap.add_argument("--deltas", default="0.01,0.03,0.1,0.3,1.0")
    ap.add_argument("--TD", type=float, default=None, help="window for the curvature ratio (default cfg T_eval)")
    ap.add_argument("--nD", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.set)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(cfg["threads"])
    Ts = [float(t) for t in a.Ts.split(",")]
    deltas = [float(x) for x in a.deltas.split(",")]
    n = a.n or cfg["n_eval"]
    TD = a.TD or cfg["T_eval"]

    cks = {r: torch.load(os.path.join(r, "final.pt"), map_location=dev) for r in a.runs}
    stats0 = next(iter(cks.values()))["stats"]
    t0 = time.time()
    data = make_data(cfg, dev, horizon=max(Ts + [TD]), train=False, stats=stats0)
    ks, dt = data["ks"], data["dt"]
    uT = data["uT_test"][:, :n]
    N = ks.N
    qT = {T: int(round(T / dt)) for T in Ts}
    pmax = max(qT.values())
    print(f"data {tuple(uT.shape)} ({time.time() - t0:.0f}s); horizons {Ts} = steps {list(qT.values())}", flush=True)

    arch = Arch(N, cfg["hidden"], cfg["act"])
    models = {}
    for r, ck in cks.items():
        name = os.path.basename(os.path.normpath(r))
        models[name] = (Model(arch, dt, ck["stats"]), ck["theta"].to(dev))

    # ---- one pass along the true trajectories: true Jacobians, propagators, first-order errors, free rollouts
    with torch.no_grad():
        eps = {nm: [uT[q + 1] - m.Gt(th, uT[q]) for q in range(pmax)] for nm, (m, th) in models.items()}
        e1 = {nm: torch.zeros_like(uT[0]) for nm in models}
        u = {nm: uT[0].clone() for nm in models}
        s1 = {nm: 0.0 for nm in models}
        s2 = {nm: 0.0 for nm in models}
        P = torch.eye(N, device=dev).expand(n, N, N).clone()
        JS0 = None
        rec = {nm: {} for nm in models}
        frames = {}
        for q in range(pmax):
            JS = ks.jac_fd(uT[q], dt)
            if q == 0:
                JS0 = JS.clone()
            P = JS @ P
            for nm, (m, th) in models.items():
                e1[nm] = torch.einsum("nab,nb->na", JS, e1[nm]) + eps[nm][q]
                Em_pred = ks.G(u[nm], dt) - m.Gt(th, u[nm])
                u[nm] = m.Gt(th, u[nm])
                e = uT[q + 1] - u[nm]
                s1[nm] += float((e1[nm] ** 2).sum(-1).mean())
                s2[nm] += float((e ** 2).sum(-1).mean())
                if (q + 1) in qT.values():
                    T = [T for T, qq in qT.items() if qq == q + 1][0]
                    ratio = (Em_pred ** 2).sum(-1).sqrt() / (eps[nm][q] ** 2).sum(-1).sqrt()
                    rec[nm][T] = dict(rms_eps=float(rms(eps[nm][q])), rms_Em_pred=float(rms(Em_pred)),
                                      med_ratio=float(ratio.median()), q90_ratio=float(ratio.quantile(0.9)),
                                      rms_e=float(rms(e)), med_e=float((e ** 2).sum(-1).sqrt().median()),
                                      rms_e1=float(rms(e1[nm])), med_e1=float((e1[nm] ** 2).sum(-1).sqrt().median()),
                                      avg_e=(s2[nm] / (q + 1)) ** 0.5, avg_e1=(s1[nm] / (q + 1)) ** 0.5)
            if (q + 1) in qT.values():
                T = [T for T, qq in qT.items() if qq == q + 1][0]
                U_, S_, Vh_ = torch.linalg.svd(P)
                frames[T] = dict(P=P.clone(), S=S_, V=Vh_.transpose(-1, -2))
                namp = int((S_ > 1).sum(-1).double().median())
                print(f"T={T}: singular values of grad S_T (median over starts): top3 {S_.median(0).values[:3].tolist()}, "
                      f"#(s>1) median {namp}, s_min median {float(S_.median(0).values[-1]):.3g}, iso amp {float((S_ ** 2).mean().sqrt()):.3g} ({time.time() - t0:.0f}s)", flush=True)
        print(f"pass done ({time.time() - t0:.0f}s)", flush=True)

    out_dir = os.path.join(cfg["out_root"], "ks", "checks")
    os.makedirs(out_dir, exist_ok=True)
    gen = torch.Generator(device=dev).manual_seed(0)
    X = uT[0]
    for nm, (m, th) in models.items():
        print(f"\n===== {nm} =====  rms|eps| on data = {float(rms(eps[nm][0])):.3g}", flush=True)
        res = dict(run=nm, rec=rec[nm], A={}, C={}, J={}, grow={}, D={})
        e0 = eps[nm][0]
        print("(1)+(B) along free rollouts (rms over starts unless 'med')")
        print(f"{'T':>5} {'|eps|':>9} {'|Em(up)|':>9} {'ratio':>6} {'med ratio':>9} {'q90 ratio':>9} {'|e1(T)|':>9} {'|e(T)|':>9} {'avg|e1|':>9} {'avg|e|':>9} {'med|e1|':>9} {'med|e|':>9}")
        for T in Ts:
            w = rec[nm][T]
            print(f"{T:5.1f} {w['rms_eps']:9.3g} {w['rms_Em_pred']:9.3g} {w['rms_Em_pred'] / w['rms_eps']:6.2f} {w['med_ratio']:9.2f} {w['q90_ratio']:9.2f} "
                  f"{w['rms_e1']:9.3g} {w['rms_e']:9.3g} {w['avg_e1']:9.3g} {w['avg_e']:9.3g} {w['med_e1']:9.3g} {w['med_e']:9.3g}", flush=True)
        print("(A) eps in the singular frame of grad S_T at the start; (C) one-step kernel Rayleigh quotients")
        print(f"{'T':>5} {'frac amp':>9} {'frac top1':>9} {'#amp':>5} {'amp(eps)':>9} {'amp(iso)':>9} {'amp/iso':>8} {'r(eps)':>9} {'r(amp)':>9} {'r(rest)':>9}")
        for T in Ts:
            fr = frames[T]
            comp = torch.einsum("nij,ni->nj", fr["V"], e0)                     # components along v_j
            mask = (fr["S"] > 1).double()
            frac_amp = float(((comp ** 2) * mask).sum() / (comp ** 2).sum())
            frac_top = float((comp[:, 0] ** 2).sum() / (comp ** 2).sum())
            amp = float(rms(torch.einsum("nij,nj->ni", fr["P"], e0)) / rms(e0))
            iso = float((fr["S"] ** 2).mean().sqrt())
            phi_amp = torch.einsum("nij,nj->ni", fr["V"], comp * mask)
            phi_rest = e0 - phi_amp

            def ksize2(phi):
                t = th.clone().requires_grad_(True)
                val = (m.Gt(t, X) * phi).sum() / n
                g = torch.autograd.grad(val, t)[0]
                return float((g * g).sum())
            r = {lab: ksize2(phi) / (float((phi * phi).sum()) / n) for lab, phi in (("eps", e0), ("amp", phi_amp), ("rest", phi_rest))}
            res["A"][T] = dict(frac_amp=frac_amp, frac_top=frac_top, namp=float(mask.sum(-1).median()), amp=amp, amp_iso=iso)
            res["C"][T] = r
            print(f"{T:5.1f} {frac_amp:9.3f} {frac_top:9.3f} {int(mask.sum(-1).median()):5d} {amp:9.3g} {iso:9.3g} {amp / iso:8.3f} {r['eps']:9.3g} {r['amp']:9.3g} {r['rest']:9.3g}", flush=True)
        # ---- (2a), (2b) in the frame of the shortest horizon
        T = Ts[0]
        fr = frames[T]
        with torch.no_grad():
            J = m.Jstate(th, X)
            D1 = (J - JS0) / dt
            F0 = (JS0 - torch.eye(N, device=dev)) / dt
            nmid = int((fr["S"] > 1).sum(-1).double().median())
            dirs = dict(top=fr["V"][:, :, 0], neutral=fr["V"][:, :, min(nmid, N - 1)], bottom=fr["V"][:, :, -1])
            print(f"(2a) Jacobian error (grad H - grad f) on unit directions of grad S_{T} (rms over starts); (2b) error off the data")
            print(f"{'dir':>8} {'rms|D1 v|':>10} {'med|D1 v|':>10} {'rms|gradf v|':>12} {'med rel':>8} {'s med':>8}")
            for lab, v in dirs.items():
                dv = torch.einsum("nab,nb->na", D1, v).norm(dim=-1)
                fv = torch.einsum("nab,nb->na", F0, v).norm(dim=-1)
                sj = {"top": fr["S"][:, 0], "neutral": fr["S"][:, min(nmid, N - 1)], "bottom": fr["S"][:, -1]}[lab]
                res["J"][lab] = dict(rms_D1v=float(dv.pow(2).mean().sqrt()), med_D1v=float(dv.median()), rms_fv=float(fv.pow(2).mean().sqrt()), med_rel=float((dv / fv).median()))
                print(f"{lab:>8} {res['J'][lab]['rms_D1v']:10.3g} {res['J'][lab]['med_D1v']:10.3g} {res['J'][lab]['rms_fv']:12.3g} {res['J'][lab]['med_rel']:8.3g} {float(sj.median()):8.3g}", flush=True)
            Em = lambda x: ks.G(x, dt) - m.Gt(th, x)
            E0 = float(rms(Em(X)))
            print(f"{'delta':>7} {'top':>9} {'bottom':>9} {'top/on':>8} {'bottom/on':>9}   (on data {E0:.3g})")
            for dl in deltas:
                et = 0.5 * (rms(Em(X + dl * dirs["top"])) + rms(Em(X - dl * dirs["top"])))
                eb = 0.5 * (rms(Em(X + dl * dirs["bottom"])) + rms(Em(X - dl * dirs["bottom"])))
                res["grow"][dl] = dict(top=float(et), bottom=float(eb))
                print(f"{dl:7.3f} {float(et):9.3g} {float(eb):9.3g} {float(et) / E0:8.2f} {float(eb) / E0:9.2f}", flush=True)
        # ---- (D) curvature ratio
        pD = int(round(TD / dt))
        U = uT[:pD + 1, :a.nD]
        lam = {}
        for lab, kap, pp in (("onestep", 1.0, pD), ("free", None, pD)):
            f_ = lambda x: loss_kappa(m, x, U, pp, kap)

            def hvp(v):
                x = th.clone().requires_grad_(True)
                gr = torch.autograd.grad(f_(x), x, create_graph=True)[0]
                return torch.autograd.grad(gr @ v, x)[0]
            lam[lab] = top_eig(hvp, arch.P, dev, a.iters, gen)
        res["D"] = dict(TD=TD, lam=lam, ratio=lam["free"] / lam["onestep"])
        print(f"(D) top Hessian eigenvalue, window {TD} ({pD} steps), {a.nD} starts: one-step {lam['onestep']:.3g}, free {lam['free']:.3g}, ratio {lam['free'] / lam['onestep']:.3g}", flush=True)
        torch.save(res, os.path.join(out_dir, f"diag_{nm}.pt"))
        print(f"saved diag_{nm}.pt ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
