"""Part 2: sine targets (setup of the earlier paper).

One hidden tanh layer of width 64, scalar in/out, weights and biases N(0,1) with input weights x3,
output layer zero. n = 200 inputs uniform on [-1,1] (sorted), targets sin(w0 x) and N(0,1) labels.
Full-batch GD, lr 0.02, on the loss (1/2)||N - y||^2 = (1/2n) sum (N_i - y_i)^2, so the step is
-lr (1/n) J^T E (Pi = I). This is the direct fit: c_p = 1, U = y, E = N - y.

At each checkpoint (the parameters before step k, and the actual next step k -> k+1):
  ||E||; F(s), F_M(s) (M <= Mmax) with s = |theta_{k+1} - theta_k|, for the kernel at theta_k and for
  the kernel of the step-averaged Jacobian (Gauss-Legendre quadrature); ||E||_{K+} and its Krylov
  values; the slow-error floor of Theorem 'Local complexity bounds the injected error from below'
  with the manifold high-pass (k = 8, m = 24), scanned over the threshold lambda_c.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = dict(width=64, n=200, omegas=[1, 5, 10, 20, 40, 60, 90, 120], random_labels=True, lr=0.02,
           steps=300000, ckpts=[3000, 30000, 300000], seeds=[0, 1, 2], Mmax=50, knn=8, m_hp=24,
           quad=32, quad_check=64, in_scale=3.0, log_every=500, s_mults=[1, 100, 10000])
QUICK = dict(steps=2000, ckpts=[200, 2000], seeds=[0], omegas=[1, 20], log_every=100)


def targets(cfg):
    names = [f"sin{w}" for w in cfg["omegas"]] + (["random"] if cfg["random_labels"] else [])
    return names


def setup(cfg, dev):
    """Inputs, labels, and initial parameters for every (seed, target) run. Same init within a seed."""
    names = targets(cfg)
    H, n = cfg["width"], cfg["n"]
    X, Y, P, runs = [], [], [], []
    for seed in cfg["seeds"]:
        g = np.random.default_rng(seed)
        x = np.sort(g.uniform(-1, 1, n))
        W1 = cfg["in_scale"] * g.normal(size=H)
        b1 = g.normal(size=H)
        yr = g.normal(size=n)
        for name in names:
            y = yr if name == "random" else np.sin(float(name[3:]) * x)
            X.append(x)
            Y.append(y)
            P.append(np.concatenate([W1, b1, np.zeros(H), [0.0]]))
            runs.append((seed, name))
    t = lambda a: torch.tensor(np.array(a), dtype=torch.float64, device=dev)
    return t(X), t(Y), t(P), runs


def train(cfg, X, Y, P):
    """Batched full-batch GD with analytic gradients. Returns error curves and checkpoint params."""
    H, n, lr = cfg["width"], cfg["n"], cfg["lr"]
    W1, b1, W2, b2 = P[:, :H].clone(), P[:, H:2 * H].clone(), P[:, 2 * H:3 * H].clone(), P[:, 3 * H:].clone()
    last = max(cfg["ckpts"])
    nlog = last // cfg["log_every"] + 1
    curve = torch.zeros(X.shape[0], nlog, dtype=torch.float64, device=X.device)
    saved = {}
    flat = lambda: torch.cat([W1, b1, W2, b2], 1).cpu().numpy().copy()
    for k in range(last + 1):
        Z = X[:, :, None] * W1[:, None, :] + b1[:, None, :]
        Hh = torch.tanh(Z)
        out = (Hh * W2[:, None, :]).sum(-1) + b2
        Er = out - Y
        if k % cfg["log_every"] == 0:
            curve[:, k // cfg["log_every"]] = Er.pow(2).mean(1).sqrt()
        if k in cfg["ckpts"]:
            saved[k] = [flat()]
        r = Er / n
        gW2 = (Hh * r[:, :, None]).sum(1)
        gb2 = r.sum(1, keepdim=True)
        D = r[:, :, None] * W2[:, None, :] * (1 - Hh * Hh)
        gW1 = (D * X[:, :, None]).sum(1)
        gb1 = D.sum(1)
        W1 -= lr * gW1
        b1 -= lr * gb1
        W2 -= lr * gW2
        b2 -= lr * gb2
        if k in cfg["ckpts"]:
            saved[k].append(flat())
    return curve.cpu().numpy(), saved


def floor_scan(ker, lam_ext, e_ext, Ry, RN, sigR):
    """Theorem floor with C_R = 1: ||P_s E|| >= R(y) - R(N) - sigma_R/sqrt(lc) ||P_f E||, for every lc.
    Returns rows over the lambda_c grid (every kept eigenvalue, plus one below the smallest)."""
    lam = ker.lam
    e = e_ext[:-1]
    grid = np.concatenate([[lam.min() / 10], np.sort(lam)])
    rows = []
    for lc in grid:
        slow = lam <= lc
        Ps = np.sqrt(np.sum(e[slow] ** 2) + e_ext[-1] ** 2)
        Pf = np.sqrt(np.sum(e[~slow] ** 2))
        fl = Ry - RN - sigR / np.sqrt(lc) * Pf
        rows.append((lc, Ps, Pf, fl))
    return rows


def analyse(cfg, X, Y, runs, saved, chk):
    H, n, Mmax = cfg["width"], cfg["n"], cfg["Mmax"]
    net = C.MLP1(1, H, 1)
    tq, wq = C.gauss_legendre_01(cfg["quad"])
    tq2, wq2 = C.gauss_legendre_01(cfg["quad_check"])
    X, Y = X.cpu().numpy(), Y.cpu().numpy()
    hp_cache = {}
    scal, krow = [], []
    for r, (seed, name) in enumerate(runs):
        x, y = X[r][:, None], Y[r]
        if seed not in hp_cache:
            B, _ = C.graph_highpass_basis(x, k=cfg["knn"], m=cfg["m_hp"], n_out=1)
            hp_cache[seed] = C.ProjectorSeminorm(B, n, name="highpass")
        Rhp = hp_cache[seed]
        for k, (th, thn) in sorted(saved.items()):
            th, thn = th[r], thn[r]
            case = f"seed{seed}/{name}/step{k}"
            dth = thn - th
            s = float(np.linalg.norm(dth))
            J = net.jac(th, x)
            out = net.out(th, x)
            E = out - y
            nE = C.nrm(E, n)
            gd = -cfg["lr"] * J.T @ E / n
            Enew = net.out(thn, x) - y
            nEnew = C.nrm(Enew, n)
            Jb = sum(w * net.jac(th + t * dth, x) for t, w in zip(tq, wq))
            Jb2 = sum(w * net.jac(th + t * dth, x) for t, w in zip(tq2, wq2))
            idres = np.linalg.norm(E + Jb @ dth - Enew) / max(np.linalg.norm(Enew - E), 1e-300)
            ker, kb = C.Kernel(J, n), C.Kernel(Jb, n)
            lam, ee = ker.reduced(E)
            lb, eb = kb.reduced(E)
            Khat = ker.knorm(E / nE)
            kin = C.kinv_norm(lam, ee)
            kv, rM = C.krylov_kinv(lam, ee, Mmax)
            qE = C.distinct_count(lam, ee)
            Fb, _, _ = C.F_closed(lb, eb, s)
            # --- checks that concern the actual step
            chk.add_eq("sine: actual step = -lr grad (sanity)", case, np.linalg.norm(dth - gd), 0.0, 1e-9,
                       scale=max(s, 1e-300))
            chk.add_eq("sine: E_new = E + Psibar g (quadrature identity, rel resid)", case, idres, 0.0, 1e-8,
                       quad_rel_change=np.linalg.norm(Jb2 - Jb) / np.linalg.norm(Jb2))
            chk.add("sine: ||E_new|| >= F(s), averaged Jacobian (Prop. stepexact, reading i)", case, nEnew - Fb,
                    tol=1e-12 * nE, scale=nE, E_new=nEnew, F_avg=Fb, gain_avg=nE - Fb,
                    rel_to_gain=(nEnew - Fb) / max(nE - Fb, 1e-300))
            fin = np.isfinite(kv)
            chk.add("sine: ||E||_{K+,M} nondecreasing in M (Prop. krough (b))", case,
                    np.min(np.diff(np.where(fin, kv, 1e300))), tol=1e-9 * np.nanmax(kv[fin]), scale=np.nanmax(kv[fin]))
            # --- slow-error floor
            Ry, RN, sigR = Rhp(y), Rhp(out), Rhp.sigma(J)
            fl = floor_scan(ker, lam, ee, Ry, RN, sigR)
            margins = [Ps - f for (_, Ps, _, f) in fl]
            ib = int(np.argmax([f for (*_, f) in fl]))
            chk.add("sine: slow error >= slow-error floor, all lambda_c (Thm. floor)", case, min(margins),
                    tol=1e-12 * nE, scale=nE)
            # --- F(s), F_M(s): s = actual step length (mult 1); longer steps are supplementary
            for mult in cfg["s_mults"]:
                sm = s * mult
                cm = f"{case}/s_mult={mult:g}"
                F, gam, Es = C.F_closed(lam, ee, sm)
                FM, Q = C.krylov_F(lam, ee, sm, Mmax)
                if mult == 1:
                    FMb, _ = C.krylov_F(lb, eb, s, Mmax)
                    chk.add("info: ||E_new|| >= F(s), kernel at theta_k (first order; may fail)", cm, nEnew - F,
                            scale=nE, E_new=nEnew, F=F)
                else:
                    FMb = np.full(Mmax, np.nan)
                gain = nE - F
                chk.add("sine: F_M nondecreasing, F_M <= F (Thm. stepkrylov (a))", cm,
                        min(np.min(np.diff(FM)), np.min(F - FM)), tol=1e-12 * nE, scale=nE)
                chk.add("sine: F(s) >= ||E|| - s||E^||_K (Thm. stepsup (e))", cm, F - (nE - sm * Khat),
                        tol=1e-12 * nE, scale=nE)
                kap = np.nan
                if F > 0:
                    phi = Es / np.linalg.norm(Es)
                    dM = C.dist_to_span(Q, phi)
                    dM = np.concatenate([dM, np.full(Mmax - len(dM), dM[-1])])
                    chk.add("sine: two-sided Krylov bound (Thm. stepkrylov (b))", cm,
                            min(np.min((F - FM) - 0.5 * F * dM ** 2),
                                np.min((nE + sm * np.sqrt(ker.lam1)) * dM - (F - FM))),
                            tol=1e-12 * nE, scale=nE)
                    if gam is not None and np.isfinite(gam):
                        kap = (ker.lam1 + gam) / gam
                        rho = (np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)
                        chk.add("sine: Krylov rate d_M <= (kappa-1)/2 rho^(M-1) (Thm. stepkrylov (d))", cm,
                                np.min((kap - 1) / 2 * rho ** np.arange(Mmax) - dM), tol=1e-12, kappa=kap)
                else:
                    dM = np.full(Mmax, np.nan)
                # convergence of the series: smallest M with F - F_M <= 1% of the gain ||E|| - F
                rel = (F - FM) / max(gain, 1e-300)
                M99 = int(np.argmax(rel <= 0.01)) + 1 if np.any(rel <= 0.01) else -1
                row = dict(seed=seed, target=name, step=k, s_mult=mult, norm_E=nE, s=sm, F=F, F1=FM[0],
                           F_Mmax=FM[-1], gain=gain, gain_frac=gain / nE, F1_share=(nE - FM[0]) / max(gain, 1e-300),
                           gamma=gam if gam is not None else np.nan, kappa=kap, M_1pct=M99,
                           rank=ker.rank, null_part=ee[-1], q_E=qE, lam1=ker.lam1, lam_min_kept=ker.lam.min())
                if mult == 1:
                    relb = (Fb - FMb) / max(nE - Fb, 1e-300)
                    row.update(norm_E_next=nEnew, F_avg=Fb, M_1pct_avg=int(np.argmax(relb <= 0.01)) + 1
                               if np.any(relb <= 0.01) else -1, kinv=kin, kinv_M1=kv[0], kinv_Mmax=kv[-1],
                               kinv_Mmax_over_M1=kv[-1] / kv[0], Ry=Ry, RN=RN, sigma_hp=sigR,
                               floor_best=fl[ib][3], floor_best_lc=fl[ib][0], slow_err_at_best=fl[ib][1],
                               floor_at_lc_min=fl[0][3], quad_resid=idres)
                scal.append(row)
                for M in range(1, Mmax + 1):
                    krow.append(dict(seed=seed, target=name, step=k, s_mult=mult, M=M, F_M=FM[M - 1], F=F,
                                     F_M_avg=FMb[M - 1], F_avg=Fb if mult == 1 else np.nan, norm_E=nE,
                                     d_M=dM[M - 1], kinv_M=kv[M - 1]))
                print(f"  {cm}: ||E||={nE:.3e} s={sm:.3e} gain/||E||={gain / nE:.2e} M_1%={M99} q_E={qE} "
                      f"rank={ker.rank} null={ee[-1]:.2e} floor_best={fl[ib][3]:.3e}", flush=True)
    return scal, krow
def figures(cfg, scal, krow, curves, runs, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = targets(cfg)
    kr = {}
    for r in krow:
        kr.setdefault((r["seed"], r["target"], r["step"], r["s_mult"]), []).append(r)
    cols = {k: c for k, c in zip(sorted(cfg["ckpts"]), ["tab:blue", "tab:orange", "tab:green", "tab:red"])}
    nc = 3
    nr = int(np.ceil(len(names) / nc))
    for kind, mult in [("F", m) for m in cfg["s_mults"]] + [("kinv", 1)]:
        fig, axs = plt.subplots(nr, nc, figsize=(4 * nc, 3 * nr), squeeze=False)
        for i, name in enumerate(names):
            ax = axs[i // nc, i % nc]
            for (seed, tname, step, sm), rows in sorted(kr.items()):
                if tname != name or sm != mult:
                    continue
                M = np.array([r["M"] for r in rows])
                if kind == "F":
                    FM = np.array([r["F_M"] for r in rows])
                    F, nE = rows[0]["F"], rows[0]["norm_E"]
                    yv = (nE - FM) / max(nE - F, 1e-300)
                else:
                    yv = np.array([r["kinv_M"] for r in rows])
                ax.plot(M, yv, color=cols[step], alpha=1.0 if seed == cfg["seeds"][0] else 0.35,
                        label=f"step {step}" if seed == cfg["seeds"][0] else None)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("M")
            if kind == "F":
                ax.set_ylim(-0.05, 1.05)
                ax.set_ylabel("(||E|| - F_M) / (||E|| - F)")
            else:
                ax.set_yscale("log")
                ax.set_ylabel("||E||_{K+,M}")
            if i == 0:
                ax.legend(fontsize=8)
        for j in range(len(names), nr * nc):
            axs[j // nc, j % nc].axis("off")
        slab = "s = length of the next GD step" if mult == 1 else f"SUPPLEMENTARY: s = {mult:g} x next GD step"
        ttl = (f"Krylov series F_M(s), {slab}: fraction (||E|| - F_M)/(||E|| - F) of the certified decrease "
               "reached by M terms" if kind == "F" else
               "Krylov values of the error, ||E||_{K+,M} (seed 0 solid, others faint)")
        fig.suptitle(ttl, fontsize=10)
        fig.tight_layout()
        fname = f"fig_sine_FM{'' if mult == 1 else f'_x{mult:g}'}.png" if kind == "F" else "fig_sine_kinvM.png"
        fig.savefig(os.path.join(outdir, fname), dpi=130)
        plt.close(fig)
    # training curves
    fig, ax = plt.subplots(figsize=(6, 4))
    steps = np.arange(curves.shape[1]) * cfg["log_every"]
    for r, (seed, name) in enumerate(runs):
        if seed == cfg["seeds"][0]:
            ax.loglog(np.maximum(steps, 1), curves[r], label=name)
    ax.set_xlabel("GD step")
    ax.set_ylabel("||E|| (training)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_sine_training.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    cfg = dict(CFG, **(QUICK if a.quick else {}))
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    X, Y, P, runs = setup(cfg, dev)
    print(f"[sine] {len(runs)} runs on {dev}, {max(cfg['ckpts'])} steps", flush=True)
    curves, saved = train(cfg, X, Y, P)
    t1 = time.time()
    print(f"[sine] training {t1 - t0:.1f}s", flush=True)
    chk = C.Checks("2_sine")
    scal, krow = analyse(cfg, X, Y, runs, saved, chk)
    C.write_rows(os.path.join(a.out, "sine_checkpoints.csv"), scal)
    C.write_rows(os.path.join(a.out, "sine_krylov.csv"), krow)
    chk.write(os.path.join(a.out, "sine_checks.csv"))
    crow = [dict(seed=s, target=nm, step=i * cfg["log_every"], norm_E=float(curves[r, i]))
            for r, (s, nm) in enumerate(runs) for i in range(curves.shape[1])]
    C.write_rows(os.path.join(a.out, "sine_curves.csv"), crow)
    figures(cfg, scal, krow, curves, runs, a.out)
    chk.print_summary()
    with open(os.path.join(a.out, "sine_meta.json"), "w") as f:
        json.dump(dict(cfg=cfg, device=dev, train_s=t1 - t0, total_s=time.time() - t0), f, indent=1)
    print(f"[sine] total {time.time() - t0:.1f}s")

