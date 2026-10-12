"""Krylov series of Theorem 'Krylov series' on the sine targets, at step lengths taken from training.

Same networks, data and gradient descent as sine.py. At the checkpoints 3000, 30000 and 300000 the
kernel K = (1/n) J J^T is taken at the checkpoint, and s is the parameter path length actually
travelled by gradient descent over the next K_steps in {100, 1000, 10000} steps. For each s we compute
F(s) (closed form), F_M(s) for M <= 50, and the upper bound on F - F_M from Theorem 'Krylov series'
(b) and (d): (||E|| + s lambda_1^(1/2)) (kappa - 1)/2 rho^(M-1). Plotted quantity: the part of the
certified decrease ||E|| - F(s) that M terms have not yet recovered, (F - F_M)/(||E|| - F).
Output: results/krylov_series.csv and fig_krylov_series.{png,pdf}.
"""
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C
import sine as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
CKPTS, KS, MMAX = [3000, 30000, 300000], [100, 1000, 10000], 50


def train(cfg, X, Y, P):
    H, n, lr = cfg["width"], cfg["n"], cfg["lr"]
    W1, b1, W2, b2 = P[:, :H].clone(), P[:, H:2 * H].clone(), P[:, 2 * H:3 * H].clone(), P[:, 3 * H:].clone()
    last = max(CKPTS) + max(KS)
    norms = torch.zeros(last, X.shape[0], dtype=torch.float64, device=X.device)
    errs, saved = {}, {}
    want_err = {k + K for k in CKPTS for K in KS} | set(CKPTS)
    for k in range(last + 1):
        Hh = torch.tanh(X[:, :, None] * W1[:, None, :] + b1[:, None, :])
        Er = (Hh * W2[:, None, :]).sum(-1) + b2 - Y
        if k in want_err:
            errs[k] = Er.pow(2).mean(1).sqrt().cpu().numpy()
        if k in CKPTS:
            saved[k] = torch.cat([W1, b1, W2, b2], 1).cpu().numpy().copy()
        if k == last:
            break
        r = Er / n
        g = [(r[:, :, None] * W2[:, None, :] * (1 - Hh * Hh) * X[:, :, None]).sum(1),
             (r[:, :, None] * W2[:, None, :] * (1 - Hh * Hh)).sum(1),
             (Hh * r[:, :, None]).sum(1), r.sum(1, keepdim=True)]
        norms[k] = lr * torch.sqrt(sum(gi.pow(2).sum(1) for gi in g))
        W1 -= lr * g[0]
        b1 -= lr * g[1]
        W2 -= lr * g[2]
        b2 -= lr * g[3]
    return saved, norms.cpu().numpy(), errs


def main():
    cfg = dict(S.CFG)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    X, Y, P, runs = S.setup(cfg, dev)
    saved, norms, errs = train(cfg, X, Y, P)
    print(f"trained {len(runs)} runs on {dev} in {time.time() - t0:.0f}s", flush=True)
    X, Y = X.cpu().numpy(), Y.cpu().numpy()
    n, net = cfg["n"], C.MLP1(1, cfg["width"], 1)
    rows = []
    chk = C.Checks("krylov_series")
    for r, (seed, name) in enumerate(runs):
        x, y = X[r][:, None], Y[r]
        for k in CKPTS:
            th = saved[k][r]
            E = net.out(th, x) - y
            nE = C.nrm(E, n)
            ker = C.Kernel(net.jac(th, x), n)
            lam, ee = ker.reduced(E)
            for K in KS:
                s = float(norms[k:k + K, r].sum())
                F, gam, Es = C.F_closed(lam, ee, s)
                FM, Q = C.krylov_F(lam, ee, s, MMAX)
                gain = nE - F
                if gam is not None and np.isfinite(gam):
                    kap = (ker.lam1 + gam) / gam
                    rho = (np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)
                    bound = (nE + s * np.sqrt(ker.lam1)) * (kap - 1) / 2 * rho ** np.arange(MMAX)
                else:
                    kap, bound = np.nan, np.full(MMAX, np.nan)
                case = f"seed{seed}/{name}/step{k}/K{K}"
                chk.add("F_M <= F and nondecreasing", case, min(np.min(F - FM), np.min(np.diff(FM))),
                        tol=1e-12 * nE, scale=nE)
                if np.isfinite(kap):
                    chk.add("F - F_M <= rate bound (b)+(d)", case, np.min(bound - (F - FM)), tol=1e-12 * nE,
                            scale=nE)
                for M in range(1, MMAX + 1):
                    rows.append(dict(seed=seed, target=name, step=k, K=K, s=s, M=M, norm_E=nE, F=F, F_M=FM[M - 1],
                                     gain=gain, rel_gap=(F - FM[M - 1]) / gain if gain > 0 else np.nan,
                                     rel_bound=bound[M - 1] / gain if gain > 0 else np.nan, kappa=kap,
                                     err_after_K=errs[k + K][r], q_E=C.distinct_count(lam, ee)))
        print(f"  {name} seed{seed} done", flush=True)
    C.write_rows(os.path.join(OUT, "krylov_series.csv"), rows)
    chk.write(os.path.join(OUT, "krylov_series_checks.csv"))
    chk.print_summary()
    figure(rows)
    # summary: M needed to recover all but 1% / 1e-6 of the certified decrease
    for K in KS:
        for k in CKPTS:
            sel = [r for r in rows if r["K"] == K and r["step"] == k]
            by = {}
            for r_ in sel:
                by.setdefault((r_["seed"], r_["target"]), []).append(r_)
            m1 = [min([q["M"] for q in v if q["rel_gap"] <= 1e-2] or [99]) for v in by.values()]
            m6 = [min([q["M"] for q in v if q["rel_gap"] <= 1e-6] or [99]) for v in by.values()]
            fr = np.median([v[0]["gain"] / v[0]["norm_E"] for v in by.values()])
            print(f"K={K:5d} step={k:6d}: median gain/||E|| {fr:.2e}; M for 1%: median {np.median(m1):.0f} max {max(m1)}; "
                  f"M for 1e-6: median {np.median(m6):.0f} max {max(m6)}", flush=True)


def figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    targets = ["sin5", "sin20", "sin60", "random"]
    cols = {100: "tab:blue", 1000: "tab:orange", 10000: "tab:green"}
    fig, axs = plt.subplots(len(CKPTS), len(targets), figsize=(12, 8), sharex=True, sharey=True)
    for i, k in enumerate(CKPTS):
        for j, tg in enumerate(targets):
            ax = axs[i, j]
            for K in KS:
                for seed in (0, 1, 2):
                    rs = [r for r in rows if r["step"] == k and r["target"] == tg and r["K"] == K and r["seed"] == seed]
                    M = [r["M"] for r in rs]
                    g = np.maximum([r["rel_gap"] for r in rs], 1e-16)
                    a = 1.0 if seed == 0 else 0.3
                    ax.semilogy(M, g, color=cols[K], alpha=a, label=f"next {K} steps" if seed == 0 else None)
                    if seed == 0:
                        ax.semilogy(M, [r["rel_bound"] for r in rs], color=cols[K], ls="--", lw=0.8,
                                    label="bound, Thm (b)+(d)" if K == KS[-1] else None)
            ax.set_ylim(1e-13, 3)
            ax.set_xlim(1, 50)
            if i == 0:
                ax.set_title(tg.replace("sin", r"$\sin$ ") + ("x" if tg != "random" else ""), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"step {k}\n" + r"$(F - F_M)/(\|E\| - F)$")
            if i == len(CKPTS) - 1:
                ax.set_xlabel("M")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(r"Krylov series: part of the certified decrease $\|E\| - F(s)$ not yet recovered by $F_M(s)$; "
                 r"$s$ = path length of the next 100 / 1000 / 10000 gradient steps (seed 0 solid)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_krylov_series.{ext}"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
