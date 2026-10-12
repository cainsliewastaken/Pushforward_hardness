"""Soft (kernel-ridge) and Krylov versions of the two experiments of the paper.

Part S (sine targets, setup of sine.py): at each checkpoint, with the kernel at that checkpoint and
the manifold high-pass R (k = 8, m = 24, C_R = 1), for a grid of scales gamma:
    ||E_gamma||                      residual of the error at scale gamma  (= F(s_gamma))
    R(U) - R(N) - sigma_R Ups^(1/2)  floor of Theorem 'Local complexity bounds the injected error'
    R(E) - sigma_R Ups^(1/2)         step bound b_R(s_gamma)
    F_M(s_gamma), M = 1, 2, 5, 10    Krylov values
with Ups = Upsilon_E(gamma) and |c_p| s_gamma = Ups^(1/2). Chain: floor <= b_R <= F <= ||E||,
F_M <= F.

Part L (Lorenz-63 steppers, setup of replicate_rsk.py part C): direct and Euler networks trained
with full-batch Adam on centred, normalized targets, parameters and Adam preconditioner saved at
checkpoints. At each checkpoint, with the kernel K = (1/n) Psi Psi^T, Psi = J Pi^(1/2) (Pi = I at
step 0), in state units: the error E and the centred target V, their residuals at scale gamma,
the Krylov values of the residual, the relative removal rate, the hard slow part, and the
training and validation errors. Also the manifold high-pass roughness of the targets (m = 300).

Output: results/soft_sine.csv, results/soft_lorenz.csv, results/soft_lorenz_targets.csv,
results/fig_soft_sine.{png,pdf}, results/fig_soft_lorenz.{png,pdf}."""
import argparse
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C
import sine as Sn
import replicate_rsk as RK

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
T0 = time.time()
MS = [1, 2, 5, 10, 20, 50]


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def residual(lam, e, gam):
    """||u_gamma|| for reduced (lam, e); the null coordinate has lam = 0 and weight 1."""
    return float(np.linalg.norm(gam * e / (lam + gam)))


# ------------------------------------------------------------------------------------ part S
def part_sine(quick, dev):
    cfg = dict(Sn.CFG)
    if quick:
        cfg.update(Sn.QUICK)
    X, Y, P, runs = Sn.setup(cfg, dev)
    log(f"sine: training {len(runs)} runs, {max(cfg['ckpts'])} steps on {dev}")
    _, saved = Sn.train(cfg, X, Y, P)
    X, Y = X.cpu().numpy(), Y.cpu().numpy()
    H, n = cfg["width"], cfg["n"]
    net = C.MLP1(1, H, 1)
    grid = np.unique(np.concatenate([np.logspace(-6, 1, 29), [0.3]]))
    hp, rows = {}, []
    for r, (seed, name) in enumerate(runs):
        x, y = X[r][:, None], Y[r]
        if seed not in hp:
            B, _ = C.graph_highpass_basis(x, k=cfg["knn"], m=cfg["m_hp"], n_out=1)
            hp[seed] = C.ProjectorSeminorm(B, n, name="highpass")
        R = hp[seed]
        for k, (th_all, _) in sorted(saved.items()):
            th = th_all[r]
            J, out = net.jac(th, x), net.out(th, x)
            E = out - y
            ker = C.Kernel(J, n)
            lam, e = ker.reduced(E)
            RU, RN, RE, sig, nE = R(y), R(out), R(E), R.sigma(J), C.nrm(E, n)
            slow = lam <= 0.3 * ker.lam1
            hard_slow = float(np.sqrt(np.sum(e[slow] ** 2)))
            hard_floor = RU - RN - sig / np.sqrt(0.3 * ker.lam1) * float(np.sqrt(np.sum(e[~slow] ** 2)))
            for g in grid:
                gam = g * ker.lam1
                ups = np.sqrt(C.upsilon(gam, lam, e))
                FM, _ = C.krylov_F(lam, e, ups, max(MS))
                rows.append(dict(seed=seed, target=name, step=k, gamma_rel=g, err=nE, RU=RU, RN=RN, RE=RE,
                                 sigma=sig, lam1=ker.lam1, ups_sqrt=ups, E_gamma=residual(lam, e, gam),
                                 F_closed=C.F_closed(lam, e, ups)[0], floor=RU - RN - sig * ups,
                                 b_R=RE - sig * ups, hard_slow=hard_slow, hard_floor=hard_floor,
                                 **{f"F_M{M}": FM[M - 1] for M in MS}))
    C.write_rows(os.path.join(OUT, "soft_sine.csv"), rows)
    # checks of the chain
    bad = [q for q in rows if q["floor"] > q["b_R"] + 1e-9 or q["b_R"] > q["E_gamma"] + 1e-9
           or q["F_M10"] > q["E_gamma"] + 1e-9 or abs(q["F_closed"] - q["E_gamma"]) > 1e-6 * max(q["err"], 1e-12)]
    log(f"sine: {len(rows)} rows, chain floor <= b_R <= ||E_gamma||, F_M <= ||E_gamma||, F = ||E_gamma||: "
        f"{len(rows) - len(bad)}/{len(rows)} ok")
    last = max(cfg["ckpts"])
    names = Sn.targets(cfg)
    log("sine, last checkpoint, gamma = 0.3 lam1 (means over seeds):")
    log(f"  {'target':8} {'R(U)':>7} {'||E||':>7} {'||E_g||':>8} {'floor':>7} {'b_R':>7} {'F_1':>7} {'F_5':>7} "
        f"{'hard slow':>9} {'hard floor':>10}")
    for nm in names:
        q = [v for v in rows if v["target"] == nm and v["step"] == last and abs(v["gamma_rel"] - 0.3) < 1e-12]
        m = lambda key: np.mean([v[key] for v in q])
        log(f"  {nm:8} {m('RU'):7.3f} {m('err'):7.3f} {m('E_gamma'):8.3f} {m('floor'):7.3f} {m('b_R'):7.3f} "
            f"{m('F_M1'):7.3f} {m('F_M5'):7.3f} {m('hard_slow'):9.3f} {m('hard_floor'):10.3f}")
    fig_sine(cfg, rows, names)
    return rows


def fig_sine(cfg, rows, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    g03 = [v for v in rows if abs(v["gamma_rel"] - 0.3) < 1e-12]

    def stat(step, key):
        a = np.array([[v[key] for v in g03 if v["target"] == nm and v["step"] == step] for nm in names])
        return a.mean(1), a.min(1), a.max(1)
    cols = ["#9ecae1", "#3182bd", "#08306b"]
    last = max(cfg["ckpts"])
    for step, c in zip(sorted(cfg["ckpts"]), cols[-len(cfg["ckpts"]):]):
        x, _, _ = stat(step, "RU")
        o = np.argsort(x)
        for a, key in zip(ax[:2], ("err", "RN")):
            m, lo, hi = stat(step, key)
            a.errorbar(x[o], m[o], yerr=[(m - lo)[o], (hi - m)[o]], color=c, marker="o", ms=4, capsize=2,
                       label=f"{step} steps")
    ax[1].plot([0, 1], [0, 1], "k:", lw=0.8)
    x, _, _ = stat(last, "RU")
    o = np.argsort(x)
    for key, c, ls, lab in (("err", "k", "-", r"error $\|E\|$"),
                            ("E_gamma", "#08306b", "-", r"residual $\|E_\gamma\|$"),
                            ("F_M5", "#31a354", "--", r"Krylov value $F_5$"),
                            ("F_M1", "#a1d99b", "--", r"Krylov value $F_1$"),
                            ("floor", "#e6550d", ":", r"floor, manifold high-pass")):
        m, lo, hi = stat(last, key)
        ax[2].errorbar(x[o], m[o], yerr=[(m - lo)[o], (hi - m)[o]], color=c, ls=ls, marker="o", ms=4,
                       capsize=2, label=lab)
    ax[0].set_title(r"(a) Error $\|E\|$", fontsize=10)
    ax[1].set_title(r"(b) Roughness of the network output $\mathcal{R}_m(\mathcal{N}_\theta)$", fontsize=10)
    ax[2].set_title(rf"(c) Bounds at $\gamma = 0.3\lambda_1$, {last} steps", fontsize=10)
    for a in ax:
        a.set_xlabel(r"roughness of the target $\mathcal{R}_m(U)$")
        a.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_soft_sine.{ext}"), dpi=140)


# ------------------------------------------------------------------------------------ part L
def train_ckpt(xt, Ys, H, inits, dev, ckpts, steps=6000, lr0=3e-3, clip=0.05):
    """replicate_rsk.train_batch with the parameters and Adam's preconditioner saved at `ckpts`
    (before the update of that step) and after the last step."""
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
    saved = {}

    def snap(k):
        params = [tuple(p[b].detach().clone() for p in ps) for b in range(B)]
        if k == 0:
            pis = [None] * B
        else:
            vh = [vi / (1 - b2c ** k) for vi in v]
            pis = [torch.cat([(1 / (x[b].sqrt() + eps)).reshape(-1) for x in vh]) for b in range(B)]
        saved[k] = (params, pis)
    for k in range(steps):
        if k in ckpts:
            snap(k)
        hdn = torch.tanh(torch.einsum("nd,bhd->bnh", xt, W1) + b1[:, None, :])
        out = torch.einsum("bnh,bkh->bnk", hdn, W2) + b2[:, None, :]
        lossb = ((out - Y) ** 2).mean((1, 2))
        for p in ps:
            p.grad = None
        lossb.sum().backward()
        with torch.no_grad():
            gn = torch.sqrt(sum(p.grad.reshape(B, -1).pow(2).sum(1) for p in ps))
            fac = torch.clamp(clip / (gn + 1e-6), max=1.0)
            lr = lr0 * 0.5 * (1 + np.cos(np.pi * k / steps))
            for i, p in enumerate(ps):
                gg = p.grad * fac.view(-1, *([1] * (p.dim() - 1)))
                m[i].mul_(b1c).add_(gg, alpha=1 - b1c)
                v[i].mul_(b2c).addcmul_(gg, gg, value=1 - b2c)
                mh = m[i] / (1 - b1c ** (k + 1))
                vh = v[i] / (1 - b2c ** (k + 1))
                p -= lr * mh / (vh.sqrt() + eps)
    snap(steps)
    return saved


def reduce_fn(p, pi, xt):
    """Eigenvalues of K = (1/n) Psi Psi^T (descending, numerical rank) and a function that maps a
    sampled function u of shape (n, 3) in state units to its reduced vector (coefficients + null)."""
    n = xt.shape[0]
    J = RK.net_jac(p, xt)
    A = (J if pi is None else J * pi.sqrt()[None, :]) / np.sqrt(n)
    lam, W = torch.linalg.eigh(A.T @ A)
    lam, W = lam.flip(0), W.flip(1)
    keep = lam > C.RANK_RTOL * lam[0]
    lam, W = lam[keep], W[:, keep]
    lam_ext = np.append(lam.cpu().numpy(), 0.0)

    def red(u):
        ut = torch.as_tensor(u, dtype=A.dtype, device=A.device).reshape(-1) / np.sqrt(n)
        c = (W.T @ (A.T @ ut)) / lam.sqrt()
        null = torch.sqrt(torch.clamp(ut @ ut - c @ c, min=0.0))
        return np.append(c.cpu().numpy(), float(null))
    return lam_ext, red


def measures(lam, u, prefix, gam_rels):
    lam1 = lam[0]
    nu = float(np.linalg.norm(u))
    d = {f"{prefix}_norm": nu,
         f"{prefix}_rho_rel": float(np.sum(lam * u ** 2)) / (nu ** 2 * lam1) if nu > 0 else np.nan,
         f"{prefix}_hard_slow_1e-4": float(np.sqrt(np.sum(u[lam <= 1e-4 * lam1] ** 2))),
         f"{prefix}_null": float(abs(u[-1]))}
    for g in gam_rels:
        d[f"{prefix}_res_{g:g}"] = residual(lam, u, g * lam1)
    for tag, g in (("F", 1e-4), ("Fg2", 1e-2)):  # Krylov values of the residual at two scales
        ups = np.sqrt(C.upsilon(g * lam1, lam, u))
        FM, _ = C.krylov_F(lam, u, ups, max(MS))
        for M in MS:
            d[f"{prefix}_{tag}_M{M}"] = float(FM[M - 1])
    return d


def part_lorenz(quick, dev):
    hs = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]
    widths, inits, n_tr, n_va, steps, m_hp = [32, 128], list(range(5)), 5000, 5000, 6000, 300
    ckpts = [0, 50, 300, 1000, 3000]
    if quick:
        hs, widths, inits, n_tr, n_va, steps, m_hp, ckpts = [0.05], [32], [0, 1], 400, 400, 300, 30, [0, 50]
    gam_rels = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    rows, trows = [], []
    for h in hs:
        t0 = time.time()
        X0, X1, V0, V1 = RK.stepper_data(h, n_tr, n_va, seed=0)
        mu, sd = X0.mean(0), X0.std(0)
        xt = torch.tensor((X0 - mu) / sd, dtype=torch.float64, device=dev)
        xv = torch.tensor((V0 - mu) / sd, dtype=torch.float64, device=dev)
        tg = {"direct": X1, "euler": X1 - X0}
        tgv = {"direct": V1, "euler": V1 - V0}
        cen = {s: tg[s].mean(0) for s in tg}
        sc = {s: np.sqrt(np.mean(np.sum((tg[s] - cen[s]) ** 2, 1))) for s in tg}
        xc = X0 - X0.mean(0)
        Eg = RK.graph_basis_full((X0 - mu) / sd, k=10)[:, :m_hp]
        Wm = lambda u: np.linalg.norm(u - Eg @ (Eg.T @ u)) / np.sqrt(n_tr)
        trows.append(dict(h=h, size_direct=sc["direct"], size_euler=sc["euler"],
                          size_identity=np.sqrt(np.mean(np.sum(xc ** 2, 1))),
                          R_direct=Wm(tg["direct"] - cen["direct"]), R_euler=Wm(tg["euler"] - cen["euler"]),
                          R_identity=Wm(xc)))
        for H in widths:
            Ys = [torch.tensor((tg[s] - cen[s]) / sc[s], dtype=torch.float64, device=dev)
                  for s in ("direct", "euler") for _ in inits]
            saved = train_ckpt(xt, Ys, H, inits * 2, dev, ckpts, steps=steps)
            for k, (params, pis) in sorted(saved.items()):
                for b, (s, i) in enumerate([(s, i) for s in ("direct", "euler") for i in inits]):
                    p, pi = params[b], pis[b]
                    with torch.no_grad():
                        o_tr = RK.net_out(p, xt).cpu().numpy() * sc[s] + cen[s]
                        o_va = RK.net_out(p, xv).cpu().numpy() * sc[s] + cen[s]
                        E = o_tr - tg[s]
                        lam, red = reduce_fn(p, pi, xt)
                        row = dict(h=h, width=H, stepper=s, init=i, step=k, scale=sc[s],
                                   err_train=np.sqrt(np.mean(np.sum(E ** 2, 1))),
                                   err_val=np.sqrt(np.mean(np.sum((o_va - tgv[s]) ** 2, 1))),
                                   lam1=lam[0], rank=len(lam) - 1)
                        row.update(measures(lam, red(E), "E", gam_rels))
                        row.update(measures(lam, red(tg[s] - cen[s]), "V", gam_rels))
                        if s == "direct":
                            row.update(measures(lam, red(xc), "id", gam_rels))
                    rows.append(row)
            C.write_rows(os.path.join(OUT, "soft_lorenz.csv"), rows)
            C.write_rows(os.path.join(OUT, "soft_lorenz_targets.csv"), trows)
            log(f"lorenz h={h} width={H}: {time.time() - t0:.0f}s")
    summary_lorenz(rows, trows, steps, widths[0])
    fig_lorenz(rows, trows, steps, widths[0])
    return rows, trows


def med(rows, key, **sel):
    v = [r[key] for r in rows if all(r[a] == b for a, b in sel.items())]
    return float(np.median(v))


def ratio(rows, key, **sel):
    """Median for the direct step divided by the median for the Euler step (nan if the latter is 0)."""
    d, e = med(rows, key, stepper="direct", **sel), med(rows, key, stepper="euler", **sel)
    return d / e if e > 0 else float("nan")


def summary_lorenz(rows, trows, last, H):
    log(f"lorenz, width {H}, final step: ratio direct / Euler (medians over inits)")
    log(f"  {'h':5} {'R_m(V)':>7} {'hard slow':>9} {'||V_g||':>8} {'F_1(V)':>7} {'F_5(V)':>7} {'F_10(V)':>8} "
        f"{'train err':>9} {'val err':>8}")
    for t in trows:
        h = t["h"]
        r = lambda key: ratio(rows, key, h=h, width=H, step=last)
        log(f"  {h:<5} {t['R_direct'] / t['R_euler']:7.2f} {r('V_hard_slow_1e-4'):9.2f} {r('V_res_0.0001'):8.2f} "
            f"{r('V_F_M1'):7.2f} {r('V_F_M5'):7.2f} {r('V_F_M10'):8.2f} {r('err_train'):9.2f} {r('err_val'):8.2f}")
    steps = sorted({r["step"] for r in rows})
    for h in (trows[0]["h"], trows[min(5, len(trows) - 1)]["h"]):
        log(f"lorenz, width {H}, h = {h}, along training (medians): state-unit error ||E||, residual of the "
            f"error ||E_g|| at gamma = 1e-4 lam1, relative removal rate rho/lam1")
        for k in steps:
            a = {s: [med(rows, key, h=h, width=H, stepper=s, step=k) for key in ("err_train", "E_res_0.0001", "E_rho_rel")]
                 for s in ("direct", "euler")}
            log(f"  step {k:5d}: direct {a['direct'][0]:9.4f} {a['direct'][1]:9.4f} {a['direct'][2]:9.2e} | "
                f"euler {a['euler'][0]:9.4f} {a['euler'][1]:9.4f} {a['euler'][2]:9.2e}")


def fig_lorenz(rows, trows, last, H):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hs = [t["h"] for t in trows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    r = lambda key: [ratio(rows, key, h=h, width=H, step=last) for h in hs]
    ax[0].loglog(hs, [t["R_direct"] / t["R_euler"] for t in trows], color="#fd8d3c", marker="s", ms=4,
                 label="roughness of the target (manifold high-pass, data only)")
    ax[0].loglog(hs, r("V_res_0.0001"), color="#08306b", marker="o", ms=4,
                 label=r"residual of the target $\|V_\gamma\|$ (trained kernel)")
    ax[0].loglog(hs, r("V_F_M5"), color="#31a354", ls="--", marker="^", ms=4, label=r"Krylov value $F_5$")
    ax[0].loglog(hs, r("err_val"), "k-", marker="x", ms=5, lw=2, label="validation error")
    ax[0].axhline(1, color="gray", lw=0.6)
    ax[0].set_xlabel("step size $h$")
    ax[0].set_ylabel("direct / Euler")
    ax[0].set_title(f"(a) Ratio direct / Euler, width {H}", fontsize=10)
    ax[0].legend(fontsize=7)
    steps = sorted({q["step"] for q in rows})
    xs = [max(k, 1) for k in steps]
    h = hs[0]
    for s, c in (("direct", "tab:blue"), ("euler", "tab:red")):
        ax[1].loglog(xs, [med(rows, "err_train", h=h, width=H, stepper=s, step=k) for k in steps], color=c,
                     marker="o", ms=4, label=rf"{s}: error $\|E\|$")
        ax[1].loglog(xs, [med(rows, "E_res_0.0001", h=h, width=H, stepper=s, step=k) for k in steps], color=c,
                     ls="--", marker="^", ms=4, label=rf"{s}: residual $\|E_\gamma\|$")
    ax[1].set_xlabel("training step (step 0 shown at 1)")
    ax[1].set_ylabel("state units")
    ax[1].set_title(f"(b) Along training, h = {h}, width {H}", fontsize=10)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_soft_lorenz.{ext}"), dpi=140)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--part", default="SL")
    a = ap.parse_args()
    if a.quick:
        OUT = os.path.join(HERE, "results_quick_soft")
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device {dev}, quick={a.quick}, out={OUT}")
    if "S" in a.part:
        part_sine(a.quick, dev)
    if "L" in a.part:
        part_lorenz(a.quick, dev)
    log("ok")
