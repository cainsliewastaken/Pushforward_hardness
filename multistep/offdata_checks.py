"""Off-data checks on saved models (theory_reevaluation_2026-10-08.md, "What would actually test this").

For each run (final.pt written by train_kappa.py), on the held-out starts, CPU, float64:

 (1) Along free rollouts to --horizon: the model error AT THE PREDICTED STATE, Em(u_pred) = G(u_pred) - Gt(u_pred)
     (G the RK4 truth map), against the one-step error at the true state, eps = G(uT) - Gt(uT) = uT[q+1] - Gt(uT[q]),
     per horizon; and the first-order prediction Em(u_pred) ~= eps + Delta1(uT) e, with e = uT - u_pred the deviation
     and Delta1 = grad Gt - grad G the Jacobian error at the true state. Where |Em(u_pred)| ~ |eps| the objective is
     set by the on-attractor one-step error alone.

 (2) The Jacobian error Delta1 at the true states split along the local tangent plane of the attractor and its
     normal (local PCA of the held-out trajectory cloud), reported as |(grad H - grad f) v| for unit v = n, t1, t2,
     next to |grad f v|; and the growth of the model error under displacements delta*n (normal) and delta*t1
     (tangent) off the data, |Em(x + delta v)| against |Em(x)|.

Writes <out_root>/kappa/checks/offdata_<name>.pt per run and prints tables. New files only.
"""
import argparse
import os
import time

import torch
from torch.func import jacfwd, vmap

from msr.config import load_cfg
from msr.core import Arch, Model, make_data, nsteps


def rms(a):
    """a (..., n, 3) -> rms over the start axis of |a|."""
    return torch.sqrt((a * a).sum(-1).mean(-1))


def med(a):
    return (a * a).sum(-1).sqrt().median(-1).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs containing final.pt")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--horizon", type=float, default=2.56)
    ap.add_argument("--n", type=int, default=4096, help="held-out starts for (1)")
    ap.add_argument("--njac", type=int, default=2048, help="points for (2)")
    ap.add_argument("--knn", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--report-at", default="0.08,0.16,0.32,0.64,0.96,1.28,1.92,2.56")
    ap.add_argument("--deltas", default="0.003,0.01,0.03,0.1,0.3,1.0")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = "cpu"
    cfg = load_cfg("config.yaml")
    dt = a.dt
    report = [nsteps(float(t), dt) for t in a.report_at.split(",")]
    deltas = [float(x) for x in a.deltas.split(",")]

    cks = {r: torch.load(os.path.join(r, "final.pt"), map_location=dev) for r in a.runs}
    stats0 = next(iter(cks.values()))["stats"]
    for r, ck in cks.items():
        for k in stats0:
            assert torch.allclose(ck["stats"][k], stats0[k]), (r, k)
    t0 = time.time()
    data = make_data(cfg, dt, dev, train=False, stats=stats0, horizon=a.horizon)
    tr = data["truth"]
    uT = data["uT_test"][:, :a.n]                                   # (P+1, n, 3)
    P = uT.shape[0] - 1
    print(f"data: {uT.shape} ({time.time() - t0:.0f}s)", flush=True)
    jacG = vmap(jacfwd(lambda x: tr.G(x, dt)))                       # truth Jacobian grad G
    jacf = vmap(jacfwd(tr.f))

    # ---- local tangent/normal frame of the attractor at the first njac held-out starts (local PCA of the cloud)
    X = uT[0, :a.njac]
    cloud = uT.reshape(-1, 3)
    n_vec, t1_vec, t2_vec, thick, radius = [], [], [], [], []
    for i in range(0, X.shape[0], 64):
        d = torch.cdist(X[i:i + 64], cloud)                           # (64, cloud)
        idx = d.topk(a.knn + 1, largest=False).indices[:, 1:]        # drop the point itself
        nb = cloud[idx]                                              # (64, knn, 3)
        nb = nb - nb.mean(1, keepdim=True)
        C = torch.einsum("bki,bkj->bij", nb, nb) / a.knn
        w, V = torch.linalg.eigh(C)                                  # ascending
        n_vec.append(V[:, :, 0]); t2_vec.append(V[:, :, 1]); t1_vec.append(V[:, :, 2])
        thick.append(torch.sqrt(w[:, 0] / w[:, 2]))
        radius.append(d.gather(1, idx[:, -1:]).squeeze(1))
    n_vec, t1_vec, t2_vec = torch.cat(n_vec), torch.cat(t1_vec), torch.cat(t2_vec)
    thick, radius = torch.cat(thick), torch.cat(radius)
    fX = tr.f(X)
    flow_normal = ((fX * n_vec).sum(-1).abs() / fX.norm(dim=-1))
    print(f"frame: knn radius median {radius.median():.3f}, sqrt(l_min/l_max) median {thick.median():.3g}, "
          f"|n.f|/|f| median {flow_normal.median():.3g} (sanity: should be small)", flush=True)
    JS_X = jacG(X)
    gradf_X = jacf(X)

    out_dir = os.path.join(cfg["out_root"], "kappa", "checks")
    os.makedirs(out_dir, exist_ok=True)
    for r, ck in cks.items():
        name = os.path.basename(os.path.normpath(r))
        arch = Arch(cfg["hidden"], cfg["act"])
        model = Model(arch, "euler", dt, ck["stats"])
        th = ck["theta"].to(dev)
        Em = lambda x: tr.G(x, dt) - model.Gt(th, x)                   # model error as a function on state space
        print(f"\n===== {name} =====", flush=True)

        # ---- (1) along free rollouts
        rows = []
        u = uT[0]
        with torch.no_grad():
            for q in range(P):
                eps = uT[q + 1] - model.Gt(th, uT[q])
                Ep = Em(u)
                e = uT[q] - u
                if (q + 1) in report:
                    D1 = model.Jstate(th, uT[q]) - jacG(uT[q])
                    lin = eps + torch.einsum("nab,nb->na", D1, e)
                    ratio = (Ep * Ep).sum(-1).sqrt() / (eps * eps).sum(-1).sqrt()
                    rows.append(dict(q=q + 1, t=(q + 1) * dt, rms_eps=float(rms(eps)), rms_Ep=float(rms(Ep)),
                                     rms_diff=float(rms(Ep - eps)), rms_diff_lin_resid=float(rms(Ep - lin)),
                                     rms_e=float(rms(e)), med_e=float(med(e)), med_ratio=float(ratio.median()),
                                     q90_ratio=float(ratio.quantile(0.9)), med_eps=float(med(eps)), med_Ep=float(med(Ep))))
                u = model.Gt(th, u)
        print("(1) model error at predicted vs true states along free rollouts (rms over starts; med = median per start)")
        print(f"{'t':>5} {'|eps|':>9} {'|Em(up)|':>9} {'ratio':>7} {'|diff|':>9} {'lin.res':>9} {'|e|':>9} {'med|e|':>9} {'med ratio':>9} {'q90 ratio':>9}")
        for w in rows:
            print(f"{w['t']:5.2f} {w['rms_eps']:9.3g} {w['rms_Ep']:9.3g} {w['rms_Ep'] / w['rms_eps']:7.2f} {w['rms_diff']:9.3g} "
                  f"{w['rms_diff_lin_resid']:9.3g} {w['rms_e']:9.3g} {w['med_e']:9.3g} {w['med_ratio']:9.2f} {w['q90_ratio']:9.2f}", flush=True)

        # ---- (2) Jacobian error in the attractor frame, and error growth off the data
        with torch.no_grad():
            J_X = model.Jstate(th, X)
            D1 = (J_X - JS_X) / dt                                       # ~ grad H - grad f
            cols = {}
            for lab, v in (("n", n_vec), ("t1", t1_vec), ("t2", t2_vec)):
                dv = torch.einsum("nab,nb->na", D1, v).norm(dim=-1)
                fv = torch.einsum("nab,nb->na", gradf_X, v).norm(dim=-1)
                cols[lab] = dict(rms_D1v=float(dv.pow(2).mean().sqrt()), med_D1v=float(dv.median()),
                                 rms_gradf_v=float(fv.pow(2).mean().sqrt()), med_rel=float((dv / fv).median()))
            print("(2a) Jacobian error (grad H - grad f) applied to unit vectors of the local attractor frame")
            print(f"{'dir':>4} {'rms|D1 v|':>10} {'med|D1 v|':>10} {'rms|gradf v|':>12} {'med rel':>8}")
            for lab, c in cols.items():
                print(f"{lab:>4} {c['rms_D1v']:10.3g} {c['med_D1v']:10.3g} {c['rms_gradf_v']:12.3g} {c['med_rel']:8.3g}", flush=True)
            E0 = Em(X)
            grow = []
            for dl in deltas:
                en = 0.5 * (rms(Em(X + dl * n_vec)) + rms(Em(X - dl * n_vec)))
                et = 0.5 * (rms(Em(X + dl * t1_vec)) + rms(Em(X - dl * t1_vec)))
                grow.append(dict(delta=dl, rms_normal=float(en), rms_tangent=float(et)))
            print(f"(2b) model error off the data: rms|Em(x)| on data = {float(rms(E0)):.3g}; displaced by delta along n / t1")
            print(f"{'delta':>7} {'normal':>9} {'tangent':>9} {'normal/on':>9} {'tangent/on':>10}")
            for g in grow:
                print(f"{g['delta']:7.3f} {g['rms_normal']:9.3g} {g['rms_tangent']:9.3g} {g['rms_normal'] / float(rms(E0)):9.2f} "
                      f"{g['rms_tangent'] / float(rms(E0)):10.2f}", flush=True)
        torch.save(dict(run=r, rows=rows, jac_cols=cols, growth=grow, rms_E0=float(rms(E0)),
                        frame=dict(radius_med=float(radius.median()), thick_med=float(thick.median()),
                                   flow_normal_med=float(flow_normal.median()))),
                   os.path.join(out_dir, f"offdata_{name}.pt"))
        print(f"saved {os.path.join(out_dir, f'offdata_{name}.pt')} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
