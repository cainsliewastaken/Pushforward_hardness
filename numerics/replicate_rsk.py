"""Replication of the experiments of the earlier paper 'Rough errors lie on the slow part of the
training kernel' (Overleaf: old_files/rough_slow_kernel.tex): Tables floor, ceiling, c, targets,
steppers, split and Figures fig_floor, fig_clusters, fig_stepper, following its Appendix
'Experimental details'. Output: results_rsk/.

Parts:  A  sine targets (Table floor, Figure floor, Table ceiling, Figure clusters)
        B  increment correlation of Lorenz-63 (Table c)
        C  direct vs Euler steppers on Lorenz-63 (Tables targets, steppers, split, Figure stepper)
Conventions of that paper: <a,b> = (1/n) sum a_i.b_i; K = (1/n) Phi Phi^T with Phi = J Pi^(1/2);
rough subspace A with projector P_A; W(u) = ||P_A u||^2; sigma^2 = lambda_max(P_A K P_A);
omega^2 = ||P_A Phi||_F^2 / n; slow projector onto eigenvalues <= lambda_c.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C
import sine as S

HERE = os.path.dirname(os.path.abspath(__file__))
log = lambda s: print(s, flush=True)


# ================================================================ A: sine targets
def cluster_basis(x, k, p):
    """Clusters of k consecutive (sorted) inputs; Euclidean-orthonormal basis of the polynomials of
    degree <= p on each cluster. P_A = I - B B^T subtracts the least-squares polynomial per cluster."""
    n = len(x)
    cols = []
    for s0 in range(0, n, k):
        idx = np.arange(s0, min(s0 + k, n))
        V = np.vander(x[idx] - x[idx].mean(), min(p + 1, len(idx)), increasing=True)
        Q, _ = np.linalg.qr(V)
        for j in range(Q.shape[1]):
            c = np.zeros(n)
            c[idx] = Q[:, j]
            cols.append(c)
    return np.array(cols).T


def rough_stats(B, J, n, U_f, lam_c):
    """sigma, omega, and ||P_f P_A|| for P_A = I - B B^T (all in the tilde coordinates u/sqrt(n))."""
    A = J / np.sqrt(n)
    PA_A = A - B @ (B.T @ A)
    sig = np.linalg.norm(PA_A, 2)
    om = np.linalg.norm(PA_A)
    PfPA = np.linalg.norm(U_f.T - (U_f.T @ B) @ B.T, 2) if U_f.shape[1] else 0.0
    return sig, om, PfPA


def part_A(out, quick):
    cfg = dict(S.CFG, **(S.QUICK if quick else {}))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, Y, P, runs = S.setup(cfg, dev)
    t0 = time.time()
    _, saved = S.train(cfg, X, Y, P)
    log(f"[A] trained {len(runs)} runs in {time.time() - t0:.0f}s")
    X, Y = X.cpu().numpy(), Y.cpu().numpy()
    n, H = cfg["n"], cfg["width"]
    net = C.MLP1(1, H, 1)
    m_grid = [2, 4, 8, 16, 24, 32, 48, 64, 96]
    cl_grid = [(k, p) for k in (2, 5, 10, 20, 25, 40, 50, 100) for p in (0, 1, 2)]
    rows, ceil_rows, m_rows = [], [], []
    last = max(cfg["ckpts"])
    for r, (seed, name) in enumerate(runs):
        x, y = X[r], Y[r]
        Bm = {m: C.graph_highpass_basis(x[:, None], k=8, m=m, n_out=1)[0] for m in m_grid}
        for k, (th, _) in sorted(saved.items()):
            th = th[r]
            J = net.jac(th, x[:, None])
            f = net.out(th, x[:, None])
            rv = f - y
            A = J / np.sqrt(n)
            Ua, sv, _ = np.linalg.svd(A, full_matrices=False)
            lam = sv ** 2
            lam_c = 0.3 * lam[0]
            U_f = Ua[:, lam > lam_c]
            til = lambda u: u / np.sqrt(n)
            Pf = lambda u: np.linalg.norm(U_f.T @ til(u))
            Ps = lambda u: np.sqrt(max(np.linalg.norm(til(u)) ** 2 - Pf(u) ** 2, 0.0))
            W = lambda B, u: np.linalg.norm(til(u) - B @ (B.T @ til(u)))  # W(u)^(1/2)
            B24 = Bm[24] if 24 in Bm else Bm[m_grid[0]]
            sig, om, PfPA = rough_stats(B24, J, n, U_f, lam_c)
            # budget identity sum_i lam_i W(v_i) = omega^2 (all eigenpairs; zero eigenvalues add nothing)
            PU = Ua - B24 @ (B24.T @ Ua)
            budget = float(np.sum(lam * np.sum(PU ** 2, 0)))
            WV, Wf = W(B24, y), W(B24, f)
            leak = sig / np.sqrt(lam_c) * Pf(rv)
            row = dict(seed=seed, target=name, step=k, size_V=np.linalg.norm(til(y)), err=np.linalg.norm(til(rv)),
                       WV=WV, Wf=Wf, floor=WV - Wf - leak, slow_err=Ps(rv), fast_err=Pf(rv), leak=leak,
                       sig_over_sqrt_lc=sig / np.sqrt(lam_c), om_over_sqrt_lc=om / np.sqrt(lam_c), PfPA=PfPA,
                       budget_lhs=budget, budget_rhs=om ** 2, budget_rel_diff=abs(budget - om ** 2) / om ** 2,
                       n_fast=U_f.shape[1])
            rows.append(row)
            # other m (Figure clusters a, b) and the ceiling table (only at the last checkpoint)
            if k != last:
                continue
            Vs = Ps(y)
            vs_vec = til(y) - U_f @ (U_f.T @ til(y))          # P_s V in tilde coordinates
            limit = np.linalg.norm(vs_vec - vs_vec.mean())    # ||P_s V - vbar||
            for m in m_grid:
                sg, _, pfpa = rough_stats(Bm[m], J, n, U_f, lam_c)
                bV = W(Bm[m], y) - sg / np.sqrt(lam_c) * Pf(y)
                fl = W(Bm[m], y) - W(Bm[m], f) - sg / np.sqrt(lam_c) * Pf(rv)
                m_rows.append(dict(seed=seed, target=name, kind="highpass", m=m, k=np.nan, p=np.nan,
                                   sig_over_sqrt_lc=sg / np.sqrt(lam_c), PfPA=pfpa, bound_V=bV, floor=fl,
                                   slow_err=Ps(rv)))
            for (kc, pc) in cl_grid:
                Bc = cluster_basis(x, kc, pc) / 1.0
                if Bc.shape[1] >= n:
                    continue  # subspace is {0}
                sg, _, pfpa = rough_stats(Bc, J, n, U_f, lam_c)
                bV = W(Bc, y) - sg / np.sqrt(lam_c) * Pf(y)
                m_rows.append(dict(seed=seed, target=name, kind="cluster", m=np.nan, k=kc, p=pc,
                                   sig_over_sqrt_lc=sg / np.sqrt(lam_c), PfPA=pfpa, bound_V=bV, floor=np.nan,
                                   slow_err=Ps(rv)))
            ceil_rows.append(dict(seed=seed, target=name, slow_target=Vs, limit=limit))
        log(f"  [A] {name} seed{seed} done")
    C.write_rows(os.path.join(out, "A_floor.csv"), rows)
    C.write_rows(os.path.join(out, "A_rough_subspaces.csv"), m_rows)
    C.write_rows(os.path.join(out, "A_ceiling.csv"), ceil_rows)
    return cfg, rows, m_rows, ceil_rows


def figs_A(cfg, rows, m_rows, ceil_rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = S.targets(cfg)
    last = max(cfg["ckpts"])
    agg = lambda sel, key: np.array([[r[key] for r in rows if r["target"] == nm and sel(r)] for nm in names])
    WV = agg(lambda r: r["step"] == last, "WV").mean(1)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    cols = ["#9ecae1", "#4292c6", "#08306b"]
    for c, k in zip(cols, sorted(cfg["ckpts"])):
        for i, key in enumerate(("err", "Wf")):
            v = agg(lambda r: r["step"] == k, key)
            ax[i].errorbar(WV, v.mean(1), yerr=[v.mean(1) - v.min(1), v.max(1) - v.mean(1)], color=c, marker="o",
                           ms=4, capsize=2, label=f"{k} steps")
    ax[1].plot([0, 1], [0, 1], "k:", lw=0.8, label="W(f) = W(V)")
    for key, c, mk, lab in (("slow_err", "#08306b", "o", "slow error ||P_s r||"),
                            ("floor", "#e6550d", "s", "slow-error floor")):
        v = agg(lambda r: r["step"] == last, key)
        ax[2].errorbar(WV, v.mean(1), yerr=[v.mean(1) - v.min(1), v.max(1) - v.mean(1)], color=c, marker=mk,
                       ms=4, capsize=2, ls="-" if key == "slow_err" else "--", label=lab)
    for a, t in zip(ax, ("(a) Total error ||r||", "(b) Network output W(f)^1/2", f"(c) Slow error and floor, {last} steps")):
        a.set_title(t, fontsize=10)
        a.set_xlabel("roughness of the target W(V)^1/2")
        a.legend(fontsize=7)
    ax[0].set_ylabel("RMS error")
    ax[1].set_ylabel("RMS roughness")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_floor.png"), dpi=140)
    plt.close(fig)
    # clusters figure
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(names))
    se = np.array([[r["slow_err"] for r in m_rows if r["target"] == nm and r["kind"] == "highpass" and r["m"] == 24]
                   for nm in names]).mean(1)
    ax[0].plot(x, se, "k-o", ms=4, label="slow error ||P_s r||")
    for m, c in zip((8, 24, 64), ("#fdae6b", "#e6550d", "#3182bd")):
        v = np.array([[r["floor"] for r in m_rows if r["target"] == nm and r["kind"] == "highpass" and r["m"] == m]
                      for nm in names]).mean(1)
        ax[0].plot(x, v, "--s", color=c, ms=4, label=f"floor, m = {m}")
    ax[0].set_xticks(x, names, rotation=45, fontsize=7)
    ax[0].set_title("(a) Slow error and slow-error floor", fontsize=10)
    ax[0].legend(fontsize=7)
    ms = sorted({r["m"] for r in m_rows if r["kind"] == "highpass"})
    sg = np.array([[r["sig_over_sqrt_lc"] for r in m_rows if r["kind"] == "highpass" and r["m"] == m] for m in ms])
    pf = np.array([[r["PfPA"] for r in m_rows if r["kind"] == "highpass" and r["m"] == m] for m in ms])
    ax[1].fill_between(ms, sg.min(1), sg.max(1), color="#3182bd", alpha=0.2)
    ax[1].loglog(ms, sg.mean(1), "-o", color="#3182bd", ms=4, label="bound sigma/sqrt(lambda_c)")
    ax[1].loglog(ms, pf.mean(1), "-o", color="#31a354", ms=4, label="measured ||P_f P_A||")
    ax[1].axhline(0.1, color="k", ls=":")
    ax[1].set_xlabel("smooth frequencies m")
    ax[1].set_title("(b) Fast part of A_m (27 runs)", fontsize=10)
    ax[1].legend(fontsize=7)
    tab = ceiling_table(cfg, m_rows, ceil_rows)
    ax[2].plot(x, [t["limit"] for t in tab], "k-o", ms=4, label="limit ||P_s V - vbar||")
    ax[2].plot(x, [t["b24"] for t in tab], "--s", color="#e6550d", ms=4, label="bound, m = 24")
    ax[2].plot(x, [t["best_b"] for t in tab], "--^", color="#3182bd", ms=4, label="bound, best m")
    ax[2].set_xticks(x, names, rotation=45, fontsize=7)
    ax[2].set_title("(c) Bound on the slow target", fontsize=10)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_clusters.png"), dpi=140)
    plt.close(fig)


def ceiling_table(cfg, m_rows, ceil_rows):
    names = S.targets(cfg)
    tab = []
    for nm in names:
        cr = [r for r in ceil_rows if r["target"] == nm]
        hp = {}
        cl = {}
        for r in m_rows:
            if r["target"] != nm:
                continue
            if r["kind"] == "highpass":
                hp.setdefault(r["m"], []).append(r["bound_V"])
            else:
                cl.setdefault((r["k"], r["p"]), []).append(r["bound_V"])
        bm = max(hp, key=lambda m: np.mean(hp[m]))
        bc = max(cl, key=lambda kp: np.mean(cl[kp]))
        tab.append(dict(target=nm, slow_target=np.mean([r["slow_target"] for r in cr]),
                        limit=np.mean([r["limit"] for r in cr]), b24=np.mean(hp.get(24, [np.nan])),
                        best_m=bm, best_b=np.mean(hp[bm]), best_kp=bc, best_cb=np.mean(cl[bc])))
    return tab


# ================================================================ B/C: Lorenz-63
SIG, RHO, BETA = 10.0, 28.0, 8.0 / 3.0


def lorenz_f(t, X):
    x, y, z = X.reshape(-1, 3).T
    return np.stack([SIG * (y - x), x * (RHO - z) - y, x * y - BETA * z], 1).ravel()


def ensemble(T_total, dt, n_traj, seed, spin=50.0, tol=1e-10):
    """n_traj independent trajectories, each T_total/n_traj long after a spin-up, sampled every dt.
    Returns array (n_traj, n_samples, 3). (The paper used one long trajectory; an ensemble of
    independent pieces samples the same invariant measure and is much faster to integrate.)"""
    g = np.random.default_rng(seed)
    x0 = np.array([1.0, 1.0, 20.0]) + g.normal(size=(n_traj, 3)) * 5
    kw = dict(method="DOP853", rtol=tol, atol=tol)
    xs = solve_ivp(lorenz_f, (0, spin), x0.ravel(), **kw).y[:, -1]
    L = T_total / n_traj
    ns = int(round(L / dt)) + 1
    te = np.arange(ns) * dt
    sol = solve_ivp(lorenz_f, (0, te[-1]), xs, t_eval=te, **kw)
    return sol.y.reshape(n_traj, 3, ns).transpose(0, 2, 1)


def part_B(out, quick):
    dt = 1e-3
    T = 200.0 if quick else 2000.0
    tr = ensemble(T, dt, 40, seed=7)
    hs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    # tau_v from xdot = f, xddot = Df f
    X = tr.reshape(-1, 3)
    f = lorenz_f(0, X.ravel()).reshape(-1, 3)
    x, y, z = X.T
    Df_f = np.stack([SIG * (f[:, 1] - f[:, 0]),
                     f[:, 0] * (RHO - z) - x * f[:, 2] - f[:, 1],
                     f[:, 0] * y + x * f[:, 1] - BETA * f[:, 2]], 1)
    v2, a2 = np.mean(np.sum(f ** 2, 1)), np.mean(np.sum(Df_f ** 2, 1))
    tau = np.sqrt(v2 / a2)
    rows = []
    for h in hs:
        k = int(round(h / dt))
        D = tr[:, k:, :] - tr[:, :-k, :]                 # Delta_t for every t on the grid
        c_meas = np.mean(np.sum(D[:, :-k] * D[:, k:], 2)) / np.mean(np.sum(D ** 2, 2))
        G1 = np.mean(np.sum((tr[:, k:] - tr[:, :-k]) ** 2, 2))
        G2 = np.mean(np.sum((tr[:, 2 * k:] - tr[:, :-2 * k]) ** 2, 2))
        rows.append(dict(h=h, c_measured=c_meas, c_from_G=G2 / (2 * G1) - 1, kappa=np.log2(G2 / G1),
                         c_second_order=1 - 0.5 * h ** 2 * a2 / v2, tau_v=tau))
        log(f"  [B] h={h}: c={c_meas:.4f} eq={G2 / (2 * G1) - 1:.4f} kappa={np.log2(G2 / G1):.3f}")
    C.write_rows(os.path.join(out, "B_increment_correlation.csv"), rows)
    return rows, tr


def c_of_h(tr, h, dt=1e-3):
    k = int(round(h / dt))
    if k >= tr.shape[1] // 3:
        return np.nan
    D = tr[:, k:, :] - tr[:, :-k, :]
    return np.mean(np.sum(D[:, :-k] * D[:, k:], 2)) / np.mean(np.sum(D ** 2, 2))


def stepper_data(h, n_tr, n_va, seed):
    T = 2000.0 if h <= 0.2 else 1e4 * h
    tr = ensemble(T, h, 20, seed=1000 + int(round(h * 1000)))
    X0, X1 = tr[:, :-1].reshape(-1, 3), tr[:, 1:].reshape(-1, 3)
    g = np.random.default_rng(seed)
    idx = g.choice(len(X0), n_tr + n_va, replace=False)
    return X0[idx[:n_tr]], X1[idx[:n_tr]], X0[idx[n_tr:]], X1[idx[n_tr:]]


def train_batch(xt, Ys, H, inits, dev, steps=6000, lr0=3e-3, clip=0.05):
    """Full-batch Adam on B = len(Ys) networks at once (independent losses, per-network clipping).
    Loss per network: mean squared error over all entries of the scaled target."""
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
    tail = []
    for k in range(steps):
        hdn = torch.tanh(torch.einsum("nd,bhd->bnh", xt, W1) + b1[:, None, :])
        out = torch.einsum("bnh,bkh->bnk", hdn, W2) + b2[:, None, :]
        lossb = ((out - Y) ** 2).mean((1, 2))
        if k >= steps - 500:
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
    vhat = [v[i] / (1 - b2c ** steps) for i in range(4)]
    tail = torch.stack(tail)
    tail_var = ((tail.max(0).values - tail.min(0).values) / tail.mean(0)).cpu().numpy()
    params = [(W1[b].detach(), b1[b].detach(), W2[b].detach(), b2[b].detach()) for b in range(B)]
    pis = [torch.cat([(1 / (vh[b].sqrt() + eps)).reshape(-1) for vh in vhat]) for b in range(B)]
    return params, pis, (nclip / steps).cpu().numpy(), tail_var


def net_out(p, xt):
    W1, b1, W2, b2 = p
    return torch.tanh(xt @ W1.T + b1) @ W2.T + b2


def net_jac(p, xt):
    """(3n x d) Jacobian, parameter order W1, b1, W2, b2 (matches the Pi vector)."""
    W1, b1, W2, b2 = p
    n = xt.shape[0]
    h = torch.tanh(xt @ W1.T + b1)
    A = W2[None] * (1 - h * h)[:, None, :]
    eye = torch.eye(3, dtype=h.dtype, device=h.device)
    dW1 = (A[..., None] * xt[:, None, None, :]).reshape(n, 3, -1)
    dW2 = (eye[None, :, :, None] * h[:, None, None, :]).reshape(n, 3, -1)
    return torch.cat([dW1, A, dW2, eye.expand(n, 3, 3)], 2).reshape(3 * n, -1)


def fast_basis(p, pi, xt, rel):
    """Euclidean-orthonormal basis (tilde coordinates) of the sampled eigenfunctions with
    eigenvalue > rel * lambda_1, from the d x d matrix Phi^T Phi / n."""
    n = xt.shape[0]
    Phi = net_jac(p, xt) * pi.sqrt()[None, :]
    A = Phi / np.sqrt(n)
    G = A.T @ A
    lam, Wv = torch.linalg.eigh(G)
    keep = lam > rel * lam[-1]
    Uf = (A @ Wv[:, keep]) / lam[keep].sqrt()
    return Uf, lam.flip(0), A


def graph_basis_full(Xs, k=10):
    """All eigenvectors of the kNN graph Laplacian (Section 'Manifold high-pass') on standardized states."""
    from scipy.spatial import cKDTree
    n = len(Xs)
    d, idx = cKDTree(Xs).query(Xs, k + 1)
    d, idx = d[:, 1:], idx[:, 1:]
    eps = np.median(d[:, -1])
    W = np.zeros((n, n))
    W[np.repeat(np.arange(n), k), idx.ravel()] = np.exp(-(d.ravel() / eps) ** 2)
    W = np.maximum(W, W.T)
    L = np.diag(W.sum(1)) - W
    mu, E = np.linalg.eigh(L)
    return E


def part_C(out, quick, tr_c):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hs = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75]
    widths, inits, n_tr, n_va, steps = [32, 128], list(range(5)), 5000, 5000, 6000
    ms_b = [3, 10, 30, 100, 300, 1000, 2000]
    if quick:
        hs, widths, inits, n_tr, n_va, steps, ms_b = [0.01, 0.2], [32], [0, 1], 400, 400, 300, [10, 30]
    rows, trows = [], []
    for h in hs:
        t0 = time.time()
        X0, X1, V0, V1 = stepper_data(h, n_tr, n_va, seed=0)
        mu, sd = X0.mean(0), X0.std(0)
        xt = torch.tensor((X0 - mu) / sd, dtype=torch.float64, device=dev)
        xv = torch.tensor((V0 - mu) / sd, dtype=torch.float64, device=dev)
        tg = {"direct": X1, "euler": X1 - X0}
        cen = {s: tg[s].mean(0) for s in tg}
        sc = {s: np.sqrt(np.mean(np.sum((tg[s] - cen[s]) ** 2, 1))) for s in tg}
        xc = X0 - X0.mean(0)
        # rough subspace (manifold high-pass, 10 neighbours) on the standardized training states
        E = graph_basis_full((X0 - mu) / sd, k=10)
        Et = torch.tensor(E[:, :max(ms_b)], dtype=torch.float64, device=dev)
        til = lambda u: u.reshape(-1) / np.sqrt(n_tr)

        def Wm(m, u):  # W(u)^(1/2) under A_m, u of shape (n, 3)
            Em = E[:, :m]
            return np.linalg.norm(u - Em @ (Em.T @ u)) / np.sqrt(n_tr)
        rough = {s: Wm(300 if not quick else 30, tg[s] - cen[s]) for s in tg}
        rough["identity"] = Wm(300 if not quick else 30, xc)
        trows.append(dict(h=h, c=c_of_h(tr_c, h), size_direct=sc["direct"], size_euler=sc["euler"],
                          size_identity=np.sqrt(np.mean(np.sum(xc ** 2, 1))),
                          W_direct=rough["direct"], W_euler=rough["euler"], W_identity=rough["identity"]))
        for H in widths:
            Ys = [torch.tensor((tg[s] - cen[s]) / sc[s], dtype=torch.float64, device=dev)
                  for s in ("direct", "euler") for _ in inits]
            params, pis, fclip, tail = train_batch(xt, Ys, H, inits * 2, dev, steps=steps)
            for b, (s, i) in enumerate([(s, i) for s in ("direct", "euler") for i in inits]):
                p = params[b]
                with torch.no_grad():
                    def pred(xs, X):
                        o = (net_out(p, xs).cpu().numpy() * sc[s] + cen[s])
                        return o if s == "direct" else X + o
                    err_tr = np.sqrt(np.mean(np.sum((pred(xt, X0) - X1) ** 2, 1)))
                    err_va = np.sqrt(np.mean(np.sum((pred(xv, V0) - V1) ** 2, 1)))
                    Uf, lam, A = fast_basis(p, pis[b], xt, 1e-4)
                    Uf3, _, _ = fast_basis(p, pis[b], xt, 1e-3)
                    Ufn = Uf.cpu().numpy()
                    Vc = (tg[s] - cen[s])
                    slow = lambda u, U=Ufn: np.sqrt(max(np.sum(til(u) ** 2) - np.sum((U.T @ til(u)) ** 2), 0))
                    Ps_vec = lambda u: til(u) - Ufn @ (Ufn.T @ til(u))
                    row = dict(h=h, width=H, stepper=s, init=i, scale=sc[s], err_train=err_tr, err_val=err_va,
                               slow_target=slow(Vc), slow_target_1e3=slow(Vc, Uf3.cpu().numpy()),
                               fast_frac=np.linalg.norm(Ufn.T @ til(Vc)) / np.linalg.norm(til(Vc)),
                               n_fast=Ufn.shape[1], clip_frac=fclip[b], tail_loss_var=tail[b])
                    if s == "direct":
                        a, d_ = Ps_vec(xc), Ps_vec(tg["euler"] - cen["euler"])
                        row.update(identity_part=np.linalg.norm(a), increment_part=np.linalg.norm(d_),
                                   cosine=float(a @ d_ / (np.linalg.norm(a) * np.linalg.norm(d_))))
                    # Theorem slow (d) on the target for several m: b = W(V)^1/2 - sigma/sqrt(lc) ||P_f V||
                    # sigma_m^2 = lambda_max(A^T A - sum_k (E_m^T A_k)^T (E_m^T A_k)), A_k = rows of component k
                    lc = 1e-4 * lam[0].item()
                    PfV = np.linalg.norm(Ufn.T @ til(Vc))
                    A3 = A.reshape(n_tr, 3, -1)
                    EtA = torch.einsum("nm,nkd->mkd", Et, A3)
                    G0 = A.T @ A
                    for m in ms_b:
                        Xm = EtA[:m].reshape(-1, A.shape[1])
                        sg = torch.linalg.eigvalsh(G0 - Xm.T @ Xm)[-1].clamp_min(0).sqrt().item()
                        row[f"bound_V_m{m}"] = Wm(m, Vc) - sg / np.sqrt(lc) * PfV
                        row[f"sig_over_sqrt_lc_m{m}"] = sg / np.sqrt(lc)
                rows.append(row)
            log(f"  [C] h={h} width={H}: {time.time() - t0:.0f}s")
        C.write_rows(os.path.join(out, "C_runs.csv"), rows)
        C.write_rows(os.path.join(out, "C_targets.csv"), trows)
    return rows, trows


def tables_C(rows, trows, out):
    from scipy.stats import mannwhitneyu
    med = lambda v: float(np.median(v))
    st, sp = [], []
    for h in sorted({r["h"] for r in rows}):
        for H in sorted({r["width"] for r in rows}):
            g = lambda s, k: [r[k] for r in rows if r["h"] == h and r["width"] == H and r["stepper"] == s]
            ed, ee = g("direct", "err_val"), g("euler", "err_val")
            p = mannwhitneyu(ed, ee, alternative="two-sided").pvalue if len(ed) > 1 else np.nan
            st.append(dict(h=h, width=H, val_direct=med(ed), val_euler=med(ee), slow_direct=med(g("direct", "slow_target")),
                           slow_euler=med(g("euler", "slow_target")), mw_p=p, ranges_disjoint=bool(max(ee) < min(ed) or max(ed) < min(ee)),
                           slow_direct_1e3=med(g("direct", "slow_target_1e3")), slow_euler_1e3=med(g("euler", "slow_target_1e3")),
                           clip_direct=med(g("direct", "clip_frac")), clip_euler=med(g("euler", "clip_frac")),
                           val_direct_min=min(ed), val_direct_max=max(ed), val_euler_min=min(ee), val_euler_max=max(ee),
                           slow_direct_min=min(g("direct", "slow_target")), slow_direct_max=max(g("direct", "slow_target")),
                           slow_euler_min=min(g("euler", "slow_target")), slow_euler_max=max(g("euler", "slow_target"))))
            sp.append(dict(h=h, width=H, identity_part=med(g("direct", "identity_part")),
                           increment_part=med(g("direct", "increment_part")), cosine=med(g("direct", "cosine")),
                           id_min=min(g("direct", "identity_part")), id_max=max(g("direct", "identity_part")),
                           slow_direct=med(g("direct", "slow_target")), slow_euler=med(g("euler", "slow_target"))))
    C.write_rows(os.path.join(out, "C_table_steppers.csv"), st)
    C.write_rows(os.path.join(out, "C_table_split.csv"), sp)
    return st, sp


def figs_C(st, sp, trows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for H, ls, fill in ((32, "-", True), (128, "--", False)):
        r = [x for x in st if x["width"] == H]
        if not r:
            continue
        hh = [x["h"] for x in r]
        for key, col, lab in (("direct", "#3182bd", "direct"), ("euler", "#e6550d", "Euler")):
            mfc = col if fill else "white"
            for i, (k, kmin, kmax) in enumerate(((f"val_{key}", f"val_{key}_min", f"val_{key}_max"),
                                                  (f"slow_{key}", f"slow_{key}_min", f"slow_{key}_max"))):
                y = np.array([x[k] for x in r])
                ax[i].errorbar(hh, y, yerr=[y - [x[kmin] for x in r], np.array([x[kmax] for x in r]) - y], color=col,
                               ls=ls, marker="o" if key == "direct" else "s", mfc=mfc, ms=4, capsize=2,
                               label=f"{lab}, width {H}")
        q = [x for x in sp if x["width"] == H]
        y = np.array([x["identity_part"] for x in q])
        ax[1].errorbar([x["h"] for x in q], y, yerr=[y - [x["id_min"] for x in q], np.array([x["id_max"] for x in q]) - y],
                       color="#31a354", ls=ls, marker="^", mfc="#31a354" if fill else "white", ms=4, capsize=2,
                       label=f"identity part, width {H}")
    hh = [t["h"] for t in trows]
    for k, col, lab in (("W_direct", "#3182bd", "direct target"), ("W_euler", "#e6550d", "Euler target"),
                        ("W_identity", "#31a354", "identity")):
        ax[2].plot(hh, [t[k] for t in trows], "-o", color=col, ms=4, label=lab)
    for a, t in zip(ax, ("(a) Validation error", "(b) Slow target ||P_s V||", "(c) Roughness W^1/2, m = 300")):
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("step size h")
        a.set_title(t, fontsize=10)
        a.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_stepper.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parts", default="ABC")
    ap.add_argument("--out", default=os.path.join(HERE, "results_rsk"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T0 = time.time()
    if "A" in a.parts:
        cfg, rows, m_rows, ceil_rows = part_A(a.out, a.quick)
        C.write_rows(os.path.join(a.out, "A_table_ceiling.csv"), ceiling_table(cfg, m_rows, ceil_rows))
        figs_A(cfg, rows, m_rows, ceil_rows, a.out)
        log(f"[A] done {time.time() - T0:.0f}s")
    if "B" in a.parts or "C" in a.parts:
        _, tr_c = part_B(a.out, a.quick)
        log(f"[B] done {time.time() - T0:.0f}s")
    if "C" in a.parts:
        rows, trows = part_C(a.out, a.quick, tr_c)
        st, sp = tables_C(rows, trows, a.out)
        figs_C(st, sp, trows, a.out)
        log(f"[C] done {time.time() - T0:.0f}s")
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(dict(parts=a.parts, quick=a.quick, total_s=time.time() - T0), f)
    log(f"[replicate] total {time.time() - T0:.0f}s")
