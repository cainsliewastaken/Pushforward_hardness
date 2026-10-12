"""Direct vs Euler steppers on Lorenz-63 at larger size, bounds from the matrix-free Krylov series
(Figure 'stepper' of the paper).

Protocol of soft_bounds.py part L (one-hidden-layer tanh network, output layer zero, inputs
standardized, targets centred and divided by their root-mean-square size, full-batch Adam with a
cosine schedule from 3e-3 and gradient clipping at 0.05, direct and Euler networks of one
initialization share their initial parameters), scaled up:
  widths 32, 128 -> 128, 512; training states 5000 -> 20000; 5 -> 10 initializations;
  6000 -> 20000 Adam steps; h grid adds 0.005;
  validation states (20000) come from independent trajectories (old: a random split of the same
  trajectories).
After training, with K = (1/n) Psi Psi^T, Psi = J Pi^(1/2), Pi = Adam's final preconditioner, in
state units: for u = E (error), V (centred target) and id (centred identity map), the Lanczos
basis of V_M(u) for M <= MMAX, lambda_1 (Ritz value), and for gamma = g * lambda_1 the slow part
||u_gamma|| = F(s_gamma) and the step length |c_p| s_gamma = Upsilon_u(gamma)^(1/2), both by
matrix-free conjugate gradients on (K + gamma I) w = u, and the Krylov values F_M(s_gamma), M in MS. F_M(s) is a lower bound on the smallest distance from u to the
output changes that a step of length s can produce (Theorem 'Krylov series').
Checks (exact eigendecomposition of the explicit kernel, not used in the figure): exact F(s) at
the same s, exact ||u_gamma||, exact lambda_1.

Output (results_large/): lorenz_rows_<tag>.csv (one row per run and gamma), lorenz_curves_<tag>.csv,
lorenz_targets_<tag>.csv, lorenz_meta_<tag>.json. Rows are written after every (h, width)."""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C
import replicate_rsk as RK
import krylov_mf as KM

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
CFG = dict(hs=[0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3], widths=[128, 512],
           inits=list(range(10)), n_tr=20000, n_va=20000, steps=20000, lr0=3e-3, clip=0.05, n_traj=100,
           curve_every=250, T_min=2000.0)
QUICK = dict(hs=[0.01, 0.1], widths=[16], inits=[0, 1, 2], n_tr=300, n_va=300, steps=300, n_traj=10,
             curve_every=50, T_min=100.0)
MMAX, MS = 200, [1, 2, 5, 10, 20, 50, 100, 200]
GAMMAS = [1e-5, 1e-4, 1e-3, 1e-2]


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def lorenz_pairs(h, n, seed, n_traj, T_min=2000.0):
    """n pairs (x, Phi_h(x)) drawn from n_traj independent trajectories on the attractor."""
    T = max(T_min, 2.0 * n * h)
    tr = RK.ensemble(T, h, n_traj, seed=seed)
    X0, X1 = tr[:, :-1].reshape(-1, 3), tr[:, 1:].reshape(-1, 3)
    idx = np.random.default_rng(seed).choice(len(X0), n, replace=False)
    return X0[idx], X1[idx]


def train(xt, Ys, H, inits, dev, steps, lr0, clip, curve_every):
    """Batched full-batch Adam (as soft_bounds.train_ckpt). Returns final parameters (W1, b1, W2, b2)
    stacked over the batch, Adam's final preconditioner (B, d), clip fraction, relative variation of
    the loss over the last 10% of steps, and the loss curve."""
    B = len(Ys)
    th = []
    for s in inits:
        g = torch.Generator().manual_seed(1000 + s)
        th.append((torch.randn(H, 3, generator=g, dtype=torch.float64),
                   torch.randn(H, generator=g, dtype=torch.float64)))
    W1 = torch.stack([a for a, _ in th]).to(dev).requires_grad_(True)
    b1 = torch.stack([b for _, b in th]).to(dev).requires_grad_(True)
    W2 = torch.zeros(B, 3, H, dtype=torch.float64, device=dev, requires_grad=True)
    b2 = torch.zeros(B, 3, dtype=torch.float64, device=dev, requires_grad=True)
    ps = [W1, b1, W2, b2]
    Y = torch.stack(Ys)
    m = [torch.zeros_like(p) for p in ps]
    v = [torch.zeros_like(p) for p in ps]
    b1c, b2c, eps = 0.9, 0.999, 1e-8
    nclip = torch.zeros(B, dtype=torch.float64, device=dev)
    curve, tail = [], []
    for k in range(steps):
        hdn = torch.tanh(torch.einsum("nd,bhd->bnh", xt, W1) + b1[:, None, :])
        out = torch.einsum("bnh,bkh->bnk", hdn, W2) + b2[:, None, :]
        lossb = ((out - Y) ** 2).mean((1, 2))
        if k % curve_every == 0:
            curve.append((k, lossb.detach().cpu().numpy()))
        if k >= steps - steps // 10:
            tail.append(lossb.detach())
        for p in ps:
            p.grad = None
        lossb.sum().backward()
        with torch.no_grad():
            gn = torch.sqrt(sum(p.grad.reshape(B, -1).pow(2).sum(1) for p in ps))
            fac = torch.clamp(clip / (gn + 1e-6), max=1.0)
            nclip += (gn > clip).double()
            lr = lr0 * 0.5 * (1 + np.cos(np.pi * k / steps))
            for i, p in enumerate(ps):
                gg = p.grad * fac.view(-1, *([1] * (p.dim() - 1)))
                m[i].mul_(b1c).add_(gg, alpha=1 - b1c)
                v[i].mul_(b2c).addcmul_(gg, gg, value=1 - b2c)
                mh = m[i] / (1 - b1c ** (k + 1))
                vh = v[i] / (1 - b2c ** (k + 1))
                p -= lr * mh / (vh.sqrt() + eps)
    vh = [vi / (1 - b2c ** steps) for vi in v]
    pi = torch.cat([(1 / (x.sqrt() + eps)).reshape(B, -1) for x in vh], 1)
    tail = torch.stack(tail)
    tvar = ((tail.max(0).values - tail.min(0).values) / tail.mean(0)).cpu().numpy()
    return [p.detach() for p in ps], pi, (nclip / steps).cpu().numpy(), tvar, curve


def diagnostics(net, us, tag_ok, exact):
    """us: dict name -> (B, N) scaled sampled functions; tag_ok: dict name -> bool mask of runs.
    Returns per-run list of dicts of results."""
    B, N = us["E"].shape
    lam1 = KM.lam1_krylov(net.K, B, N, us["E"].dtype, us["E"].device)
    res = [dict(lam1=lam1[b]) for b in range(B)]
    kry = {nm: KM.lanczos(net.K, u, MMAX) for nm, u in us.items()}
    gt = torch.tensor(lam1, dtype=us["E"].dtype, device=us["E"].device)
    cg = {(nm, g): KM.cg_slow(net.K, u, g * gt) for nm, u in us.items() for g in GAMMAS}
    for b in range(B):
        ex = None
        if exact:
            ex = KM.Exact(net.psi_scaled(b))
            res[b]["lam1_exact"] = ex.lam_ext[0]
            res[b]["rank_exact"] = len(ex.lam_ext) - 1
        for nm, u in us.items():
            if not tag_ok[nm][b]:
                continue
            G, a, orth, nre = (x[b] for x in kry[nm])
            res[b][f"{nm}_norm"] = float(u[b].norm())
            res[b][f"{nm}_orth"] = float(orth)
            res[b][f"{nm}_restarts"] = int(nre)
            e_ex = ex.reduced(u[b]) if ex is not None else None
            for g in GAMMAS:
                gam = g * lam1[b]
                slow, ups, cres, cits = (x[b] for x in cg[(nm, g)])
                cs = float(np.sqrt(ups))
                FM = KM.krylov_F(G, a, cs, MS)
                p = f"{nm}_g{g:g}"
                res[b][f"{p}_cs"] = cs
                res[b][f"{p}_slow"] = float(slow)
                res[b][f"{p}_cg_res"] = float(cres)
                res[b][f"{p}_cg_its"] = int(cits)
                res[b][f"{p}_ritz_res"] = KM.ritz_residual(G, a, gam)
                for M, val in zip(MS, FM):
                    res[b][f"{p}_F_M{M}"] = val
                if ex is not None:
                    res[b][f"{p}_F_exact"] = C.F_closed(ex.lam_ext, e_ex, cs)[0]
                    res[b][f"{p}_res_exact"] = float(np.linalg.norm(gam * e_ex / (ex.lam_ext + gam)))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--widths", default=None)
    ap.add_argument("--hs", default=None)
    ap.add_argument("--no-exact", action="store_true")
    a = ap.parse_args()
    cfg = dict(CFG)
    out = os.path.join(HERE, "results_large")
    if a.quick:
        cfg.update(QUICK)
        out = os.path.join(HERE, "results_large_quick")
    if a.widths:
        cfg["widths"] = [int(w) for w in a.widths.split(",")]
    if a.hs:
        cfg["hs"] = [float(h) for h in a.hs.split(",")]
    tag = "w" + "_".join(str(w) for w in cfg["widths"])
    os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device {dev} ({torch.cuda.get_device_name() if dev == 'cuda' else 'cpu'}), out {out}, tag {tag}")
    log("config", cfg)
    rows, crows, trows = [], [], []
    inits = cfg["inits"]
    for h in cfg["hs"]:
        t0 = time.time()
        hk = int(round(h * 1000))
        X0, X1 = lorenz_pairs(h, cfg["n_tr"], 1000 + hk, cfg["n_traj"], cfg["T_min"])
        V0, V1 = lorenz_pairs(h, cfg["n_va"], 50000 + hk, cfg["n_traj"], cfg["T_min"])
        log(f"h={h}: data in {time.time() - t0:.0f}s")
        mu, sd = X0.mean(0), X0.std(0)
        xt = torch.tensor((X0 - mu) / sd, dtype=torch.float64, device=dev)
        xv = torch.tensor((V0 - mu) / sd, dtype=torch.float64, device=dev)
        tg = {"direct": X1, "euler": X1 - X0}
        tgv = {"direct": V1, "euler": V1 - V0}
        cen = {s: tg[s].mean(0) for s in tg}
        sc = {s: float(np.sqrt(np.mean(np.sum((tg[s] - cen[s]) ** 2, 1)))) for s in tg}
        xc = X0 - X0.mean(0)
        trows.append(dict(h=h, size_direct=sc["direct"], size_euler=sc["euler"],
                          size_identity=float(np.sqrt(np.mean(np.sum(xc ** 2, 1)))),
                          size_increment=float(np.sqrt(np.mean(np.sum((X1 - X0) ** 2, 1))))))
        n = cfg["n_tr"]
        runs = [(s, i) for s in ("direct", "euler") for i in inits]
        for H in cfg["widths"]:
            t1 = time.time()
            Ys = [torch.tensor((tg[s] - cen[s]) / sc[s], dtype=torch.float64, device=dev) for s, _ in runs]
            ps, pi, fclip, tvar, curve = train(xt, Ys, H, inits * 2, dev, cfg["steps"], cfg["lr0"], cfg["clip"],
                                               cfg["curve_every"])
            ttrain = time.time() - t1
            for k, lv in curve:
                for b, (s, i) in enumerate(runs):
                    crows.append(dict(h=h, width=H, stepper=s, init=i, step=k, loss=float(lv[b])))
            scv = torch.tensor([sc[s] for s, _ in runs], dtype=torch.float64, device=dev)[:, None, None]
            cenv = torch.tensor(np.array([cen[s] for s, _ in runs]), dtype=torch.float64, device=dev)[:, None, :]
            tgt = torch.tensor(np.array([tg[s] for s, _ in runs]), dtype=torch.float64, device=dev)
            tgvt = torch.tensor(np.array([tgv[s] for s, _ in runs]), dtype=torch.float64, device=dev)
            with torch.no_grad():
                net = KM.TanhNets(*ps, xt, pi=pi)
                o_tr = net.out().reshape(len(runs), n, 3) * scv + cenv
                netv = KM.TanhNets(*ps, xv)
                o_va = netv.out().reshape(len(runs), cfg["n_va"], 3) * scv + cenv
                E = (o_tr - tgt).reshape(len(runs), -1) / np.sqrt(n)
                err_va = ((o_va - tgvt) ** 2).sum(2).mean(1).sqrt().cpu().numpy()
                V = (tgt - cenv).reshape(len(runs), -1) / np.sqrt(n)
                Id = torch.tensor(xc, dtype=torch.float64, device=dev).reshape(1, -1).expand(len(runs), -1) / np.sqrt(n)
                us = {"E": E, "V": V, "id": Id.contiguous()}
                ok = {"E": [True] * len(runs), "V": [True] * len(runs), "id": [s == "direct" for s, _ in runs]}
                t2 = time.time()
                res = diagnostics(net, us, ok, exact=not a.no_exact)
            for b, (s, i) in enumerate(runs):
                rows.append(dict(h=h, width=H, stepper=s, init=i, steps=cfg["steps"], scale=sc[s],
                                 err_train=float(E[b].norm()), err_val=float(err_va[b]), clip_frac=float(fclip[b]),
                                 tail_loss_var=float(tvar[b]), **res[b]))
            C.write_rows(os.path.join(out, f"lorenz_rows_{tag}.csv"), rows)
            C.write_rows(os.path.join(out, f"lorenz_curves_{tag}.csv"), crows)
            C.write_rows(os.path.join(out, f"lorenz_targets_{tag}.csv"), trows)
            med = lambda key, s: float(np.median([r[key] for r in rows if r["h"] == h and r["width"] == H
                                                  and r["stepper"] == s]))
            log(f"h={h} width={H}: train {ttrain:.0f}s, diagnostics {time.time() - t2:.0f}s | val err D "
                f"{med('err_val', 'direct'):.4f} E {med('err_val', 'euler'):.4f} | g=1e-4 slow(V) D "
                f"{med('V_g0.0001_slow', 'direct'):.4f} E {med('V_g0.0001_slow', 'euler'):.4f} id "
                f"{med('id_g0.0001_slow', 'direct'):.4f} | F_200(V) D {med('V_g0.0001_F_M200', 'direct'):.4f}"
                + ("" if a.no_exact else f" | exact F(V) D {med('V_g0.0001_F_exact', 'direct'):.4f}"))
    json.dump(dict(cfg=cfg, MMAX=MMAX, MS=MS, GAMMAS=GAMMAS, seconds=time.time() - T0),
              open(os.path.join(out, f"lorenz_meta_{tag}.json"), "w"), indent=1)
    log("ok")
