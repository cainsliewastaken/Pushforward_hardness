"""Three mechanism measurements on saved KS steppers (theory_reevaluation_2026-10-08.md, "Next measurements").

  python ks/mechanism.py --runs <run dirs> --T 10 --n 512

Per run, on held-out starts, along the true trajectory to T (p steps) and the model's free rollout:
 (M1) Kernel rates at the VISITED states. At the predicted states x_q = u_pred(t_q) the free loss's residual is
      e_{q+1} = sep_q + Em(x_q), with sep_q = G(uT_q) - G(x_q) the true separation (depends on the start, not on x_q)
      and Em(x_q) = G(x_q) - Gt(x_q) the model's own error there. Rayleigh quotient r(phi) = |g_phi|^2/(n|phi|^2) of the
      one-step kernel with features at x_q, for phi = e, sep, Em(x), and for eps at the true states as reference.
      A small r(sep) against r(Em) says the separation is the slow (rough) part the loss cannot remove.
 (M2) Temporal coherence of the amplified injections. With M_i = P^S(T, t_i) (true propagator from step i to T),
      e1(T) = sum_i M_i eps_i. Top-direction scalar a_i = s1_i <v1_i, eps_i> and left vector u1_i: coherent energy
      |sum_i a_i u1_i|^2 against incoherent sum_i a_i^2; ratio < 1 means the injections cancel under the propagators.
      Also the lag-1 and lag-2 correlation of a_i along the trajectory, and the full e1 against its top-direction part.
 (M3) First-order error with the MODEL's propagators, e1m_{q+1} = J^model_q e1m_q + eps_q (model Jacobians at the true
      states), against e1 (true propagators) and the measured e: e1m - e1 is the model's Jacobian error acting on the
      deviation, e - e1m the rest.
Writes <out_root>/ks/checks/mech_<name>.pt. New files only.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ks.config import load_cfg
from ks.core import Arch, Model, make_data


def rms(a):
    return torch.sqrt((a * a).sum(-1).mean(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cfg", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    ap.add_argument("--T", type=float, default=10.0)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.set)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(cfg["threads"])
    cks = {r: torch.load(os.path.join(r, "final.pt"), map_location=dev) for r in a.runs}
    stats0 = next(iter(cks.values()))["stats"]
    t0 = time.time()
    data = make_data(cfg, dev, horizon=a.T, train=False, stats=stats0)
    ks, dt = data["ks"], data["dt"]
    uT = data["uT_test"][:, :a.n]
    p = int(round(a.T / dt))
    n, N = a.n, ks.N
    JS = [ks.jac_fd(uT[q], dt) for q in range(p)]                     # true Jacobians along the true trajectory
    # M_i = J_{p-1} ... J_i, by backward accumulation
    M = [None] * p
    acc = torch.eye(N, device=dev).expand(n, N, N).clone()
    for i in range(p - 1, -1, -1):
        acc = acc @ JS[i]
        M[i] = acc.clone()
    svd = [torch.linalg.svd(M[i]) for i in range(p)]                   # U, S, Vh per injection time
    print(f"data {tuple(uT.shape)}, p={p}, propagators done ({time.time() - t0:.0f}s); "
          f"s1 of M_0 median {float(svd[0][1][:, 0].median()):.3g}, of M_{p - 1} {float(svd[p - 1][1][:, 0].median()):.3g}", flush=True)
    arch = Arch(N, cfg["hidden"], cfg["act"])
    out_dir = os.path.join(cfg["out_root"], "ks", "checks")
    os.makedirs(out_dir, exist_ok=True)

    for r, ck in cks.items():
        nm = os.path.basename(os.path.normpath(r))
        m, th = Model(arch, dt, ck["stats"]), ck["theta"].to(dev)
        res = dict(run=nm, M1={}, M2={}, M3={})
        print(f"\n===== {nm} =====", flush=True)
        with torch.no_grad():
            eps = [uT[q + 1] - m.Gt(th, uT[q]) for q in range(p)]
            # free rollout states
            xs, u = [], uT[0]
            for q in range(p):
                xs.append(u)
                u = m.Gt(th, u)
            xs.append(u)

        def ksize2(phi, X):
            t = th.clone().requires_grad_(True)
            val = (m.Gt(t, X) * phi).sum() / n
            g = torch.autograd.grad(val, t)[0]
            return float((g * g).sum()) / (float((phi * phi).sum()) / n)

        # ---- (M1)
        print("(M1) one-step kernel Rayleigh quotients at the visited states x_q (features at x_q); eps at true states for reference")
        print(f"{'t_q':>5} {'|e|':>8} {'|sep|':>8} {'|Em(x)|':>8} {'|eps|':>8} {'r(e)':>9} {'r(sep)':>9} {'r(Em(x))':>9} {'r(eps@uT)':>9}")
        for q in (p // 4 - 1, p // 2 - 1, p - 1):
            X = xs[q]
            with torch.no_grad():
                Emx = ks.G(X, dt) - m.Gt(th, X)
                sep = ks.G(uT[q], dt) - ks.G(X, dt)
                e = uT[q + 1] - xs[q + 1]
            row = dict(q=q, rms_e=float(rms(e)), rms_sep=float(rms(sep)), rms_Em=float(rms(Emx)), rms_eps=float(rms(eps[q])),
                       r_e=ksize2(e, X), r_sep=ksize2(sep, X), r_Em=ksize2(Emx, X), r_eps=ksize2(eps[q], uT[q]))
            res["M1"][q] = row
            print(f"{(q + 1) * dt:5.1f} {row['rms_e']:8.3g} {row['rms_sep']:8.3g} {row['rms_Em']:8.3g} {row['rms_eps']:8.3g} "
                  f"{row['r_e']:9.3g} {row['r_sep']:9.3g} {row['r_Em']:9.3g} {row['r_eps']:9.3g}", flush=True)

        # ---- (M2)
        with torch.no_grad():
            e1 = sum(torch.einsum("nab,nb->na", M[i], eps[i]) for i in range(p))
            avec = torch.stack([svd[i][1][:, 0] * (svd[i][2][:, 0, :] * eps[i]).sum(-1) for i in range(p)], 1)   # (n, p)
            uvec = torch.stack([svd[i][0][:, :, 0] for i in range(p)], 1)                                           # (n, p, N)
            coh = (avec[:, :, None] * uvec).sum(1)                                                                   # (n, N)
            coh_e = (coh ** 2).sum(-1).mean()
            inc_e = (avec ** 2).sum(1).mean()
            # sign-align u1 across i to the last one so that lag correlations of a_i are meaningful
            sgn = torch.sign((uvec * uvec[:, -1:, :]).sum(-1))
            aa = avec * sgn
            def lagcorr(k):
                x, y = aa[:, :-k], aa[:, k:]
                return float(((x * y).mean()) / (x.pow(2).mean().sqrt() * y.pow(2).mean().sqrt()))
            top_part = coh
            row = dict(rms_e1=float(rms(e1)), rms_top=float(rms(top_part)), coh=float(coh_e), inc=float(inc_e),
                       ratio=float(coh_e / inc_e), lag1=lagcorr(1), lag2=lagcorr(2), lag4=lagcorr(4),
                       rms_a=float(aa.pow(2).mean().sqrt()))
            res["M2"] = row
        print("(M2) amplified injections along the trajectory: coherent |sum a_i u1_i|^2 vs incoherent sum a_i^2; lag correlations of a_i")
        print(f"  rms e1(T) {row['rms_e1']:.3g}, top-direction part {row['rms_top']:.3g}; coherent {row['coh']:.3g} incoherent {row['inc']:.3g} "
              f"ratio {row['ratio']:.3f}; corr lag1 {row['lag1']:+.3f} lag2 {row['lag2']:+.3f} lag4 {row['lag4']:+.3f}; rms a_i {row['rms_a']:.3g}", flush=True)

        # ---- (M3)
        with torch.no_grad():
            e1t, e1m, u = torch.zeros_like(uT[0]), torch.zeros_like(uT[0]), uT[0]
            rows = []
            for q in range(p):
                Jm = m.Jstate(th, uT[q])
                e1t = torch.einsum("nab,nb->na", JS[q], e1t) + eps[q]
                e1m = torch.einsum("nab,nb->na", Jm, e1m) + eps[q]
                u = m.Gt(th, u)
                e = uT[q + 1] - u
                if (q + 1) in (p // 4, p // 2, p):
                    rows.append(dict(t=(q + 1) * dt, rms_e1=float(rms(e1t)), rms_e1m=float(rms(e1m)), rms_e=float(rms(e)),
                                     rms_e1m_minus_e1=float(rms(e1m - e1t)), rms_e_minus_e1m=float(rms(e - e1m)),
                                     med_e1=float((e1t ** 2).sum(-1).sqrt().median()), med_e1m=float((e1m ** 2).sum(-1).sqrt().median()),
                                     med_e=float((e ** 2).sum(-1).sqrt().median())))
            res["M3"] = rows
        print("(M3) first-order error with true (e1) vs model (e1m) propagators vs measured e (rms; med = median)")
        print(f"{'t':>5} {'|e1|':>8} {'|e1m|':>8} {'|e|':>8} {'|e1m-e1|':>9} {'|e-e1m|':>9} {'med e1':>8} {'med e1m':>8} {'med e':>8}")
        for w in rows:
            print(f"{w['t']:5.1f} {w['rms_e1']:8.3g} {w['rms_e1m']:8.3g} {w['rms_e']:8.3g} {w['rms_e1m_minus_e1']:9.3g} {w['rms_e_minus_e1m']:9.3g} "
                  f"{w['med_e1']:8.3g} {w['med_e1m']:8.3g} {w['med_e']:8.3g}", flush=True)
        torch.save(res, os.path.join(out_dir, f"mech_{nm}.pt"))
        print(f"saved mech_{nm}.pt ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
