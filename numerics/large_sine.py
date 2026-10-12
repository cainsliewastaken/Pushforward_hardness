"""Sine targets at larger size, bounds from the matrix-free Krylov series (Figure 'sine' of the paper).

Setup of sine.py (one hidden tanh layer, input weights x3, output layer zero, inputs uniform on
[-1, 1], full-batch GD on (1/2)||N - y||^2, so c_p = 1 and Pi = I), scaled up:
width 64 -> 512, n 200 -> 1000, 9 -> 16 targets, 3 -> 10 initializations. The learning rate is
scaled by 64 / width (0.0025), so lr * lambda_1 at initialization is as in the old runs.

At each checkpoint and for each run, with the kernel at that checkpoint:
  ||E||; lambda_1 (Ritz value); the Lanczos basis of V_M(E) for M <= MMAX;
  for gamma = g * lambda_1: the slow part ||E_gamma|| = F(s_gamma) and the step length
  |c_p| s_gamma = Upsilon_E(gamma)^(1/2), both by matrix-free conjugate gradients on
  (K + gamma I) w = E, the Krylov values F_M(s_gamma)
  for M in MS, and the floor R(U) - R(N) - sigma_R |c_p| s of the manifold high-pass (k = 8,
  m = 24, C_R = 1), which bounds F(s) from below for the same s.
Checks (exact eigendecomposition of the explicit kernel, not used in the figure): exact F(s) at the
same s, exact ||E_gamma||, exact lambda_1, and F_M <= F(s), floor <= F(s).

Output (results_large/): sine_rows.csv, sine_curves.csv, sine_meta.json."""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C
import sine as Sn
import krylov_mf as KM

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
CFG = dict(Sn.CFG)
CFG.update(width=512, n=1000, omegas=[1, 2, 5, 10, 15, 20, 30, 40, 60, 80, 100, 120, 160, 200, 250],
           seeds=list(range(10)), lr=0.02 * 64 / 512, steps=300000, ckpts=[3000, 30000, 300000],
           log_every=1000, knn=8, m_hp=24)
QUICK = dict(width=32, n=120, omegas=[1, 10, 40], seeds=[0, 1], lr=0.02, steps=600, ckpts=[60, 600],
             log_every=60)
MMAX, MS = 100, [1, 2, 5, 10, 20, 50, 100]
GAMMAS = [1e-4, 1e-2, 0.1, 0.3, 1.0]


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def analyse(cfg, X, Y, runs, saved, dev, out, exact=True):
    H, n = cfg["width"], cfg["n"]
    Xt = torch.tensor(X, dtype=torch.float64, device=dev)[:, :, None]   # (B, n, 1)
    Yt = torch.tensor(Y, dtype=torch.float64, device=dev)
    B = len(runs)
    hp = {}
    for r, (seed, _) in enumerate(runs):
        if seed not in hp:
            Bm, _ = C.graph_highpass_basis(X[r][:, None], k=cfg["knn"], m=cfg["m_hp"], n_out=1)
            hp[seed] = torch.tensor(Bm, dtype=torch.float64, device=dev)
    rows = []
    for k in sorted(saved):
        t0 = time.time()
        P = torch.tensor(np.array(saved[k][0]), dtype=torch.float64, device=dev)
        W1, b1, W2, b2 = P[:, :H, None], P[:, H:2 * H], P[:, 2 * H:3 * H][:, None, :], P[:, 3 * H:]
        net = KM.TanhNets(W1, b1, W2, b2, Xt)
        out_ = net.out()
        Es = (out_ - Yt) / np.sqrt(n)                                       # scaled error
        lam1 = KM.lam1_krylov(net.K, B, n, Es.dtype, dev)
        G, a, orth, nre = KM.lanczos(net.K, Es, MMAX)
        cg = {g: KM.cg_slow(net.K, Es, torch.tensor(g * lam1, dtype=Es.dtype, device=dev)) for g in GAMMAS}
        for r, (seed, name) in enumerate(runs):
            Bm = hp[seed]
            Pr = lambda u: u - Bm @ (Bm.T @ u)                               # high-pass projector
            ys, os_ = Yt[r] / np.sqrt(n), out_[r] / np.sqrt(n)
            RU, RN, RE = float(Pr(ys).norm()), float(Pr(os_).norm()), float(Pr(Es[r]).norm())
            A = net.psi_scaled(r)
            sigma = float(torch.linalg.matrix_norm(A - Bm @ (Bm.T @ A), ord=2))
            if exact:
                ex = KM.Exact(A)
                e_ex = ex.reduced(Es[r])
            err = float(Es[r].norm())
            for g in GAMMAS:
                gam = g * lam1[r]
                slow, ups, cres, cits = (x[r] for x in cg[g])
                cs = float(np.sqrt(ups))
                FM = KM.krylov_F(G[r], a[r], cs, MS)
                row = dict(seed=seed, target=name, omega=(0 if name == "random" else float(name[3:])), step=k,
                           gamma_rel=g, err=err, RU=RU, RN=RN, RE=RE, sigma=sigma, lam1=lam1[r], cs=cs,
                           E_gamma=float(slow), cg_res=float(cres), cg_its=int(cits),
                           floor=RU - RN - sigma * cs, b_R=RE - sigma * cs,
                           ritz_residual=KM.ritz_residual(G[r], a[r], gam), orth=orth[r], restarts=int(nre[r]),
                           **{f"F_M{M}": v for M, v in zip(MS, FM)})
                if exact:
                    row.update(lam1_exact=ex.lam_ext[0], F_exact=C.F_closed(ex.lam_ext, e_ex, cs)[0],
                               E_gamma_exact=float(np.linalg.norm(gam * e_ex / (ex.lam_ext + gam))),
                               ups_exact=C.upsilon(gam, ex.lam_ext, e_ex))
                rows.append(row)
        C.write_rows(os.path.join(out, "sine_rows.csv"), rows)
        log(f"checkpoint {k}: analysed {B} runs in {time.time() - t0:.0f}s")
    return rows


def checks(rows, exact):
    tol = lambda q: 1e-9 * max(q["err"], 1e-12)
    c = dict(n=len(rows),
             FM_monotone=sum(all(q[f"F_M{a}"] <= q[f"F_M{b}"] + tol(q) for a, b in zip(MS, MS[1:])) for q in rows),
             floor_le_bR=sum(q["floor"] <= q["b_R"] + tol(q) for q in rows),
             FM_le_err=sum(q[f"F_M{MS[-1]}"] <= q["err"] + tol(q) for q in rows),
             FM_le_Egamma=sum(q[f"F_M{MS[-1]}"] <= q["E_gamma"] * (1 + 1e-6) + tol(q) for q in rows),
             floor_le_Egamma=sum(q["floor"] <= q["E_gamma"] * (1 + 1e-6) + tol(q) for q in rows),
             max_cg_res=max(q["cg_res"] for q in rows),
             max_orth=max(q["orth"] for q in rows))
    if exact:
        c.update(FM_le_Fexact=sum(q[f"F_M{MS[-1]}"] <= q["F_exact"] + tol(q) for q in rows),
                 bR_le_Fexact=sum(q["b_R"] <= q["F_exact"] + tol(q) for q in rows),
                 max_rel_gap_FMmax_Fexact=max((q["F_exact"] - q[f"F_M{MS[-1]}"]) / max(q["err"], 1e-300) for q in rows),
                 max_rel_cg_vs_exact=max(abs(q["E_gamma"] - q["E_gamma_exact"]) / max(q["err"], 1e-300) for q in rows),
                 max_rel_lam1=max(abs(q["lam1"] / q["lam1_exact"] - 1) for q in rows))
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-exact", action="store_true")
    a = ap.parse_args()
    cfg = dict(CFG)
    out = os.path.join(HERE, "results_large")
    if a.quick:
        cfg.update(QUICK)
        out = os.path.join(HERE, "results_large_quick")
    os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device {dev} ({torch.cuda.get_device_name() if dev == 'cuda' else 'cpu'}), out {out}")
    log("config", {k: v for k, v in cfg.items() if k != "s_mults"})
    Xs, Ys, Ps, runs = Sn.setup(cfg, dev)
    log(f"training {len(runs)} runs, {max(cfg['ckpts'])} GD steps, lr {cfg['lr']}")
    t0 = time.time()
    curves, saved = Sn.train(cfg, Xs, Ys, Ps)
    log(f"training done in {time.time() - t0:.0f}s; nan runs: {int(np.isnan(curves[:, -1]).sum())}")
    crow = [dict(seed=s, target=nm, step=i * cfg["log_every"], err=float(curves[r, i]))
            for r, (s, nm) in enumerate(runs) for i in range(curves.shape[1])]
    C.write_rows(os.path.join(out, "sine_curves.csv"), crow)
    rows = analyse(cfg, Xs.cpu().numpy(), Ys.cpu().numpy(), runs, saved, dev, out, exact=not a.no_exact)
    ck = checks(rows, not a.no_exact)
    log("checks", ck)
    json.dump(dict(cfg=cfg, MMAX=MMAX, MS=MS, GAMMAS=GAMMAS, checks=ck, seconds=time.time() - T0),
              open(os.path.join(out, "sine_meta.json"), "w"), indent=1, default=float)
    last = max(cfg["ckpts"])
    log(f"last checkpoint, gamma = 0.3 lambda_1, means over seeds:")
    log(f"  {'target':8} {'R(U)':>6} {'err':>6} {'E_g(CG)':>7} {'F_1':>6} {'F_5':>6} {'F_20':>6} {'F_100':>6} "
        f"{'F exact':>7} {'floor':>6}")
    for nm in Sn.targets(cfg):
        q = [v for v in rows if v["target"] == nm and v["step"] == last and v["gamma_rel"] == 0.3]
        m = lambda key: np.mean([v[key] for v in q]) if key in q[0] else np.nan
        log(f"  {nm:8} {m('RU'):6.3f} {m('err'):6.3f} {m('E_gamma'):7.3f} {m('F_M1'):6.3f} {m('F_M5'):6.3f} "
            f"{m('F_M20'):6.3f} {m('F_M100'):6.3f} {m('F_exact'):7.3f} {m('floor'):6.3f}")
    log("ok")
