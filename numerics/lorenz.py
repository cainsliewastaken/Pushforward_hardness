"""Part 3: Lorenz-63, direct step vs Euler step (Sections 'Application: the direct step and the
Euler step' and 'Integration over training time').

Data: sigma=10, rho=28, beta=8/3; DOP853 at rtol=atol=1e-10 after a 50-unit spin-up. For each h,
n = 5000 states at random times on one long trajectory, each with its successor Phi_h(x) (integrated
separately from x, same tolerance). Network inputs are standardized; targets are in state units.
Network: 3 -> 32 tanh -> 3 (227 parameters), PyTorch default init for the hidden layer, output layer
zero; the direct and Euler networks of one (h, seed) share the initial parameters.
Targets: direct x_{t+1}; Euler x_{t+1} - x_t; 'raw' (unnormalized) and 'norm' (divided by the RMS
size a = ||target|| in the paper norm). In state units the method is G = c N (direct) or
G = x + c N (Euler), with c = 1 (raw) or c = a (norm): c is the paper's c_p for that network.
Training: full-batch Adam, 6000 steps, cosine lr 3e-3 -> 0, gradient-norm clipping at 0.05, loss
(1/2)||N - y/c||^2.

Pi = I throughout: Psi = J, K = J J^T / n, and the rescaled step is the raw parameter step.
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

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = dict(sig=10.0, rho=28.0, beta=8.0 / 3.0, spinup=50.0, T_traj=500.0, tol=1e-10, n=5000,
           hs=[0.01, 0.02, 0.05, 0.1, 0.2], seeds=[0, 1, 2], width=32, steps=6000, lr=3e-3, clip=0.05,
           rec_early=[0, 1, 2, 3, 5, 8, 12, 18, 25, 35], rec_every=50,
           ckpts=[0, 50, 300, 1000, 3000, 6000], Mmax=50, varrhos=[0.5, 0.1],
           G_times=[1, 2, 5, 12, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000],
           G_iters=300, curve_every=5)
QUICK = dict(n=400, hs=[0.05, 0.2], seeds=[0], steps=300, rec_every=25, ckpts=[0, 50, 300],
             G_times=[1, 5, 25, 50, 100, 200, 300], G_iters=60, T_traj=100.0)


# ---------------------------------------------------------------- data
def rhs(cfg):
    s, r, b = cfg["sig"], cfg["rho"], cfg["beta"]

    def f(t, X):
        x, y, z = X.reshape(-1, 3).T
        return np.stack([s * (y - x), x * (r - z) - y, x * y - b * z], 1).ravel()
    return f


def trajectory(cfg):
    f = rhs(cfg)
    kw = dict(method="DOP853", rtol=cfg["tol"], atol=cfg["tol"])
    x0 = solve_ivp(f, (0, cfg["spinup"]), np.array([1.0, 1.0, 1.0]), **kw).y[:, -1]
    return solve_ivp(f, (0, cfg["T_traj"]), x0, dense_output=True, **kw)


def dataset(cfg, sol, h, seed):
    g = np.random.default_rng(10_000 + 97 * seed + int(round(1e4 * h)))
    t = np.sort(g.uniform(0, cfg["T_traj"], cfg["n"]))
    X = sol.sol(t).T
    Y = solve_ivp(rhs(cfg), (0, h), X.ravel(), method="DOP853", rtol=cfg["tol"], atol=cfg["tol"]).y[:, -1]
    return X, Y.reshape(-1, 3)


# ---------------------------------------------------------------- network
def init_params(seed, H, dev):
    g = torch.Generator().manual_seed(seed)
    bnd = 1 / np.sqrt(3.0)  # PyTorch nn.Linear default for fan_in = 3
    W1 = (torch.rand(H, 3, generator=g, dtype=torch.float64) * 2 - 1) * bnd
    b1 = (torch.rand(H, generator=g, dtype=torch.float64) * 2 - 1) * bnd
    return torch.cat([W1.ravel(), b1, torch.zeros(3 * H + 3, dtype=torch.float64)]).to(dev)


def unpack(th, H):
    return th[:3 * H].view(H, 3), th[3 * H:4 * H], th[4 * H:7 * H].view(3, H), th[7 * H:]


def forward(th, xt, H):
    W1, b1, W2, b2 = unpack(th, H)
    return torch.tanh(xt @ W1.T + b1) @ W2.T + b2


def jac(th, xt, H):
    """Analytic Jacobian (3n x 227), point-major rows, same parameter order as core.MLP1."""
    W1, b1, W2, b2 = unpack(th, H)
    n = xt.shape[0]
    h = torch.tanh(xt @ W1.T + b1)
    A = W2[None] * (1 - h * h)[:, None, :]
    eye = torch.eye(3, dtype=th.dtype, device=th.device)
    dW1 = (A[..., None] * xt[:, None, None, :]).reshape(n, 3, -1)
    dW2 = (eye[None, :, :, None] * h[:, None, None, :]).reshape(n, 3, -1)
    db2 = eye.expand(n, 3, 3)
    return torch.cat([dW1, A, dW2, db2], 2).reshape(3 * n, -1)


def rec_steps(cfg):
    r = set(cfg["rec_early"]) | set(range(0, cfg["steps"] + 1, cfg["rec_every"])) | set(cfg["ckpts"])
    return sorted(k for k in r if k <= cfg["steps"])


def train(cfg, th0, xt, y_scaled, keep_J, dev):
    """Adam run. Returns per-step state-unit errors (scaled by caller), step lengths, params at the
    record steps, and (if keep_J) float32 Jacobians / sqrt(n) at the record steps."""
    H, T = cfg["width"], cfg["steps"]
    th = th0.clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=cfg["lr"])
    recs = set(rec_steps(cfg))
    err = torch.zeros(T + 1, dtype=torch.float64, device=dev)
    slen = torch.zeros(T, dtype=torch.float64, device=dev)
    params, Js = {}, []
    sqn = np.sqrt(xt.shape[0])
    for k in range(T + 1):
        if k in recs:
            params[k] = th.detach().clone()
            if keep_J:
                Js.append((jac(th.detach(), xt, H) / sqn).float())
        out = forward(th, xt, H)
        loss = 0.5 * ((out - y_scaled) ** 2).sum(1).mean()
        err[k] = torch.sqrt(2 * loss.detach())
        if k == T:
            break
        for gparam in opt.param_groups:
            gparam["lr"] = cfg["lr"] * 0.5 * (1 + np.cos(np.pi * k / T))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([th], cfg["clip"])
        prev = th.detach().clone()
        opt.step()
        slen[k] = torch.linalg.norm(th.detach() - prev)
    return err.cpu().numpy(), slen.cpu().numpy(), params, Js


# ---------------------------------------------------------------- G(t) by duality
def G_path(Js, w_all, idt, m_list, iters, log=None):
    """G(t) = sup_{||phi||<=1} <phi,id> - sum_k w_k ||phi||_{K_k}
             = min_{|z_k| <= w_k} || id - sum_k Psi_k z_k ||   (all in the paper norm).
    Js[k] = J_k / sqrt(n) (the kernel used for block k), w_k = |c| * (path length of block k).
    For each m in m_list (number of blocks), runs FISTA from the previous solution; returns
    (lower, upper): lower = value of the dual-feasible phi = residual direction (a certified lower bound
    on this Riemann-sum G), upper = residual norm of a feasible z."""
    dev = idt.device
    Kb, N, d = Js.shape
    z = torch.zeros(Kb, d, dtype=Js.dtype, device=dev)
    res = []
    idn = float(torch.linalg.norm(idt))

    def Aop(Am, zz):
        return torch.einsum("knd,kd->n", Am, zz)

    def ATop(Am, r):
        return torch.einsum("knd,n->kd", Am, r)

    def proj(zz, w):
        nz = torch.linalg.norm(zz, dim=1).clamp_min(1e-30)
        return zz * torch.clamp(w / nz, max=1.0)[:, None]

    def lower(Am, w, r):
        nr = torch.linalg.norm(r)
        if nr == 0:
            return 0.0
        phi = r / nr
        return float(phi @ idt - (w * torch.linalg.norm(ATop(Am, phi), dim=1)).sum())

    for m in m_list:
        if m == 0:
            res.append((idn, idn))
            continue
        Am, w = Js[:m], w_all[:m]
        v = torch.randn(m, d, dtype=Js.dtype, device=dev)
        for _ in range(15):
            v = ATop(Am, Aop(Am, v))
            v = v / torch.linalg.norm(v)
        Lf = 1.05 * float(torch.linalg.norm(ATop(Am, Aop(Am, v))))
        zc = z[:m].clone()
        y, tk = zc.clone(), 1.0
        best_lo, best_up = -np.inf, np.inf
        for it in range(iters):
            r = idt - Aop(Am, y)
            zn = proj(y + ATop(Am, r) / Lf, w)
            tk1 = 0.5 * (1 + np.sqrt(1 + 4 * tk * tk))
            if float(((y - zn) * (zn - zc)).sum()) > 0:  # adaptive restart
                tk1, y = 1.0, zn.clone()
            else:
                y = zn + ((tk - 1) / tk1) * (zn - zc)
            zc, tk = zn, tk1
            if it % 25 == 0 or it == iters - 1:
                r = idt - Aop(Am, zc)
                best_up = min(best_up, float(torch.linalg.norm(r)))
                best_lo = max(best_lo, lower(Am, w, r))
        z[:m] = zc
        res.append((max(best_lo, 0.0), best_up))
        if log:
            log(f"      G blocks={m}: [{best_lo:.4g}, {best_up:.4g}]")
    return res


# ---------------------------------------------------------------- analysis helpers
def krylov_bound_rows(kv, rM, nd, c, rho):
    """Eq. sDany / sDkrylov: (||u||_{K+,M} - (1+rho) r_M ||Phi_h - id||) / |c| for every M (nan if the
    Krylov value is infinite, where the maximizer is not attained)."""
    return np.where(np.isfinite(kv), (kv - (1 + rho) * rM * nd) / abs(c), np.nan)


def run_pair(cfg, sol, h, seed, dev, chk, out, log):
    H, n, Mmax = cfg["width"], cfg["n"], cfg["Mmax"]
    X, Y = dataset(cfg, sol, h, seed)
    mu, sd = X.mean(0), X.std(0)
    xt = torch.tensor((X - mu) / sd, dtype=torch.float64, device=dev)
    idv, Phi = X.ravel(), Y.ravel()
    dphi = Phi - idv
    nid, nPhi, nd = C.nrm(idv, n), C.nrm(Phi, n), C.nrm(dphi, n)
    th0 = init_params(seed, H, dev)
    tag = f"h={h}/seed{seed}"
    log(f"[{tag}] ||id||={nid:.4f} ||Phi_h||={nPhi:.4f} ||Phi_h - id||={nd:.4e}")

    # Jacobian sanity against autograd (a few rows)
    thp = th0 + 0.1 * torch.randn(th0.shape, generator=torch.Generator().manual_seed(1), dtype=th0.dtype).to(dev)
    Ja = torch.autograd.functional.jacobian(lambda t: forward(t, xt[:3], H).ravel(), thp)
    chk.add_eq("lorenz: analytic Jacobian = autograd", tag, float(torch.linalg.norm(jac(thp, xt[:3], H) - Ja)),
               0.0, 1e-10, scale=float(torch.linalg.norm(Ja)))

    # ------------------------------------------------ initialization (K_D = K_E, shared init)
    J0 = jac(th0, xt, H).cpu().numpy()
    ker0 = C.Kernel(J0, n)
    lam, idr = ker0.reduced(idv)
    d0 = idr[-1]
    d0_tol = {rt: C.Kernel(J0, n, rtol=rt).reduced(idv)[1][-1] for rt in (1e-8, 1e-16)}
    kv_id, rM_id = C.krylov_kinv(lam, idr, Mmax)
    kin_id = C.kinv_norm(lam, idr)
    fin = np.isfinite(kv_id)
    chk.add("lorenz: ||id||_{K+,M} nondecreasing in M (init)", tag, np.min(np.diff(np.where(fin, kv_id, 1e300))),
            tol=1e-9 * np.nanmax(kv_id[fin]), scale=np.nanmax(kv_id[fin]))
    _, eD = ker0.reduced(-Phi)
    _, eE = ker0.reduced(-dphi)
    aD, aE = nPhi, nd
    convs = {"paper(c_D=1,c_E=h)": (1.0, h), "raw(c_D=1,c_E=1)": (1.0, 1.0), "norm(c_D=a_D,c_E=a_E)": (aD, aE)}
    init_rows, init_krow = [], []
    for M in range(1, Mmax + 1):
        init_krow.append(dict(h=h, seed=seed, M=M, kinv_id_M=kv_id[M - 1], r_M=rM_id[M - 1], norm_id=nid))
    PfE = np.linalg.norm(eE[:-1])
    fperp = eE[-1]
    kin_Pf = float(np.sqrt(np.sum(eE[:-1] ** 2 / lam[:-1])))
    for cname, (cD, cE) in convs.items():
        for rho in cfg["varrhos"]:
            eps = rho * nd
            bnds = krylov_bound_rows(kv_id, rM_id, nd, cD, rho)
            best_a = np.nanmax(bnds) if np.any(np.isfinite(bnds)) else -np.inf
            cond_b = d0 > (1 + rho) * nd
            sD = C.s_needed(lam, eD, cD, eps)
            sE = C.s_needed(lam, eE, cE, eps)
            cond_c = fperp < rho * nd
            sE_up = (max(0.0, 1 - np.sqrt(max(rho ** 2 * nd ** 2 - fperp ** 2, 0)) / PfE) * kin_Pf / abs(cE)
                     if cond_c else np.inf)
            suff = bool((best_a > sE_up) or (cond_b and cond_c))
            case = f"{tag}/{cname}/rho={rho}"
            chk.add("lorenz: Krylov lower bound on s_D <= exact s_D (Thm. dirEustep (a))", case,
                    (sD - best_a) if np.isfinite(sD) else 1.0, tol=1e-9 * max(1.0, abs(best_a)),
                    scale=max(1.0, abs(best_a)))
            if cond_c:
                chk.add("lorenz: exact s_E <= upper bound (Thm. dirEustep (c))", case, sE_up - sE,
                        tol=1e-9 * max(sE_up, 1.0), scale=max(sE_up, 1e-300))
            if cond_b:
                chk.add("lorenz: (b) holds => exact s_D = inf (Thm. dirEustep (b))", case,
                        0.0 if np.isinf(sD) else -1.0)
            init_rows.append(dict(h=h, seed=seed, conv=cname, rho=rho, c_D=cD, c_E=cE, norm_id=nid,
                                  norm_Phi=nPhi, norm_dPhi=nd, d0=d0, d0_rtol1e8=d0_tol[1e-8],
                                  d0_rtol1e16=d0_tol[1e-16], rank_K0=ker0.rank, kinv_id=kin_id,
                                  kinv_id_M1=kv_id[0], kinv_id_Mmax=kv_id[-1],
                                  sD_bound_a_best=best_a, sD_bound_a_M1=bnds[0], cond_b=cond_b,
                                  sD_exact=sD, fperp=fperp, cond_c=cond_c, sE_upper=sE_up, sE_exact=sE,
                                  sufficient_condition_holds=suff, sD_gt_sE_exact=bool(sD > sE),
                                  kinv_rel_id=kin_id / nid, kinv_rel_f=(np.inf if fperp > 1e-10 * nd else kin_Pf / nd)))
    log(f"[{tag}] init: rank K0={ker0.rank} d0={d0:.4e} (2||Phi-id||={2 * nd:.3e}) "
        f"||id||_K+,M: M=1 {kv_id[0]:.3g}, M={Mmax} {kv_id[-1]:.3g}")
    # G0(t): fixed kernel K0, gradient flow on the state-unit error (rate c^2 K0)
    G0rows = []
    for cname, cD in (("raw", 1.0), ("norm", aD)):
        l1 = cD ** 2 * ker0.lam1
        for tt in np.logspace(-2, 8, 121) / l1:
            G0 = np.sqrt(np.sum(idr[:-1] ** 2 * np.exp(-2 * cD ** 2 * lam[:-1] * tt)) + d0 ** 2)
            G0rows.append(dict(h=h, seed=seed, scaling=cname, t=tt, t_lam1=tt * l1, G0=G0, two_dphi=2 * nd,
                               d0=d0, t_cor_a=np.log(nid / (2 * nd)) / l1))

    # ------------------------------------------------ training and checkpoints
    runs = {}
    ck_rows, ck_krow, G_rows, curve_rows, run_rows = [], [], [], [], []
    for scaling in ("raw", "norm"):
        for method in ("direct", "euler"):
            tgt = Phi if method == "direct" else dphi
            c = 1.0 if scaling == "raw" else (aD if method == "direct" else aE)
            ys = torch.tensor(tgt.reshape(-1, 3) / c, dtype=torch.float64, device=dev)
            t0 = time.time()
            errn, slen, params, Js = train(cfg, th0, xt, ys, keep_J=(method == "direct"), dev=dev)
            err = c * errn  # state units: ||c N - target||
            runs[(scaling, method)] = err
            inc = np.diff(err)
            run_rows.append(dict(h=h, seed=seed, scaling=scaling, method=method, c=c, err0=err[0],
                                 err_final=err[-1], err_min=err.min(), n_increases=int(np.sum(inc > 0)),
                                 max_rel_increase=float(np.max(inc / err[:-1])) if len(inc) else 0.0,
                                 ever_above_initial=bool(np.any(err > err[0] * (1 + 1e-12))),
                                 path_length=float(slen.sum()), wall=time.time() - t0))
            for k in range(0, cfg["steps"] + 1, cfg["curve_every"]):
                curve_rows.append(dict(h=h, seed=seed, scaling=scaling, method=method, step=k, err=err[k]))
            if method == "euler":
                chk.add("lorenz: ||E_E(t)|| <= ||Phi_h - id|| for all t (Thm. integrate)", f"{tag}/{scaling}",
                        np.min(nd - err), scale=nd, n_increases=run_rows[-1]["n_increases"])
            # checkpoints: Corollary 'Any training time'
            for k in cfg["ckpts"]:
                th = params[k]
                Jk = jac(th, xt, H).cpu().numpy()
                kk = C.Kernel(Jk, n)
                Ns = c * forward(th, xt, H).detach().cpu().numpy().ravel()  # state units c N
                row = dict(h=h, seed=seed, scaling=scaling, method=method, step=k, c=c, rank=kk.rank,
                           err=err[k])
                if method == "direct":
                    u = idv - Ns
                    lu, ur = kk.reduced(u)
                    kv_u, rM_u = C.krylov_kinv(lu, ur, Mmax)
                    ED = Ns - Phi
                    le, er_ = kk.reduced(ED)
                    kv_E, _ = C.krylov_kinv(le, er_, Mmax)
                    for rho in cfg["varrhos"]:
                        b = krylov_bound_rows(kv_u, rM_u, nd, c, rho)
                        sD = C.s_needed(le, er_, c, rho * nd)
                        ba = np.nanmax(b) if np.any(np.isfinite(b)) else -np.inf
                        row[f"sD_bound_c_best_rho{rho}"] = ba
                        row[f"sD_exact_rho{rho}"] = sD
                        chk.add("lorenz: Cor. anytime (c) Krylov bound <= exact s_D", f"{tag}/{scaling}/step{k}/rho={rho}",
                                (sD - ba) if np.isfinite(sD) else 1.0, tol=1e-9 * max(1.0, abs(ba)),
                                scale=max(1.0, abs(ba)))
                    row.update(norm_u=C.nrm(u, n), u_null=ur[-1], kinv_u_M1=kv_u[0] / c, kinv_u_Mmax=kv_u[-1] / c,
                               remove_len=C.kinv_norm(le, er_) / c, remove_len_M1=kv_E[0] / c,
                               remove_len_Mmax=kv_E[-1] / c)
                    b05 = krylov_bound_rows(kv_u, rM_u, nd, c, 0.5)
                    for M in range(1, Mmax + 1):
                        ck_krow.append({"h": h, "seed": seed, "scaling": scaling, "method": method, "step": k,
                                        "M": M, "kinv_u_M": kv_u[M - 1] / c, "kinv_E_M": kv_E[M - 1] / c,
                                        "sD_bound_M_rho0.5": b05[M - 1]})
                else:
                    EE = Ns - dphi
                    le, er_ = kk.reduced(EE)
                    kv_E, _ = C.krylov_kinv(le, er_, Mmax)
                    for rho in cfg["varrhos"]:
                        row[f"sE_exact_rho{rho}"] = C.s_needed(le, er_, c, rho * nd)
                    row.update(remove_len=C.kinv_norm(le, er_) / c, remove_len_M1=kv_E[0] / c,
                               remove_len_Mmax=kv_E[-1] / c, E_null=er_[-1])
                    for M in range(1, Mmax + 1):
                        ck_krow.append(dict(h=h, seed=seed, scaling=scaling, method=method, step=k, M=M,
                                            kinv_E_M=kv_E[M - 1] / c))
                ck_rows.append(row)
            # G(t) for the direct path
            if method == "direct":
                rs = rec_steps(cfg)
                Jst = torch.stack(Js)                      # (len(rs), N, d), J/sqrt(n) at record steps
                L = np.array([slen[rs[i]:rs[i + 1]].sum() for i in range(len(rs) - 1)])
                w = torch.tensor(abs(c) * L, dtype=Jst.dtype, device=dev)
                idt = torch.tensor(idv / np.sqrt(n), dtype=Jst.dtype, device=dev)
                m_list = [rs.index(t) for t in cfg["G_times"] if t in rs]
                # Theorem integrate (b): G >= ||id|| - sum_k |c| L_k ||id||_{K_k}/||id||
                rid = (torch.linalg.norm(torch.einsum("knd,n->kd", Jst, idt), dim=1) / float(torch.linalg.norm(idt))).cpu().numpy()
                Gb_left = nid - np.concatenate([[0], np.cumsum(abs(c) * L * rid[:-1])])
                Gb_right = nid - np.concatenate([[0], np.cumsum(abs(c) * L * rid[1:])])
                t0 = time.time()
                gl = G_path(Jst[:-1], w, idt, m_list, cfg["G_iters"])
                gr = G_path(Jst[1:], w, idt, m_list, cfg["G_iters"])
                log(f"    G(t) {scaling}: {time.time() - t0:.1f}s; G(end) left [{gl[-1][0]:.4g},{gl[-1][1]:.4g}] "
                    f"right [{gr[-1][0]:.4g},{gr[-1][1]:.4g}]")
                for m, (lo_l, up_l), (lo_r, up_r) in zip(m_list, gl, gr):
                    t = rs[m]
                    G_rows.append(dict(h=h, seed=seed, scaling=scaling, step=t, G_left_lo=lo_l, G_left_up=up_l,
                                       G_right_lo=lo_r, G_right_up=up_r, G_b_left=Gb_left[m],
                                       G_b_right=Gb_right[m], err_D=err[t], two_dphi=2 * nd, dphi=nd))
                del Jst, Js
                if dev == "cuda":
                    torch.cuda.empty_cache()
    # comparisons that need both methods
    for scaling in ("raw", "norm"):
        errD, errE = runs[(scaling, "direct")], runs[(scaling, "euler")]
        for g in G_rows:
            if g["scaling"] != scaling or g["h"] != h or g["seed"] != seed:
                continue
            t = g["step"]
            g["err_E"] = errE[t]
            Glo = min(g["G_left_lo"], g["G_right_lo"])
            g["cond_2dphi_lt_G"] = bool(2 * nd < Glo)
            case = f"{tag}/{scaling}/step{t}"
            for rule in ("left", "right"):
                chk.add(f"lorenz: ||E_D(t)|| >= G(t) - ||Phi_h - id|| (G ~ Riemann sum, {rule} rule, "
                        f"G lower cert.)", case, errD[t] - (g[f"G_{rule}_lo"] - nd), scale=max(errD[t], 1e-300))
                chk.add(f"lorenz: ||E_D(t)|| >= G(t) - ||Phi_h - id|| (G ~ Riemann sum, {rule} rule, "
                        f"G upper value: strict)", case, errD[t] - (g[f"G_{rule}_up"] - nd),
                        scale=max(errD[t], 1e-300))
                chk.add(f"info: G(t) solver gap (upper - lower) / ||id|| <= 1% ({rule} rule)", case,
                        0.01 - (g[f"G_{rule}_up"] - g[f"G_{rule}_lo"]) / nid)
            if g["cond_2dphi_lt_G"]:
                chk.add("lorenz: where 2||Phi_h - id|| < G(t): ||E_E(t)|| < ||E_D(t)|| (measured)", case,
                        errD[t] - errE[t], scale=max(errD[t], 1e-300))
    return init_rows, init_krow, G0rows, ck_rows, ck_krow, G_rows, curve_rows, run_rows


def figures(cfg, out, init_krow, G_rows, curve_rows, G0rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hs, seeds = cfg["hs"], cfg["seeds"]
    colors = dict(zip(hs, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]))
    # Krylov values of the identity at init
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for h in hs:
        for s in seeds:
            rows = [r for r in init_krow if r["h"] == h and r["seed"] == s]
            M = [r["M"] for r in rows]
            v = np.array([r["kinv_id_M"] for r in rows])
            v = np.where(np.isfinite(v), v, np.nan)
            ax.semilogy(M, v, color=colors[h], alpha=1 if s == seeds[0] else 0.35,
                        label=f"h={h}" if s == seeds[0] else None)
    ax.set_xlabel("M")
    ax.set_ylabel("||id||_{K0+,M}")
    ax.set_title("Krylov values of the identity, kernel at initialization\n(line ends where the value becomes infinite)",
                 fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_lorenz_id_krylov.png"), dpi=130)
    plt.close(fig)
    # G(t) vs measured errors
    fig, axs = plt.subplots(2, len(hs), figsize=(3.6 * len(hs), 7), squeeze=False)
    for j, h in enumerate(hs):
        for i, scaling in enumerate(("raw", "norm")):
            ax = axs[i, j]
            for s in seeds:
                a = 1 if s == seeds[0] else 0.3
                for method, col in (("direct", "tab:red"), ("euler", "tab:blue")):
                    rows = [r for r in curve_rows if r["h"] == h and r["seed"] == s and r["scaling"] == scaling
                            and r["method"] == method]
                    ax.loglog([max(r["step"], 0.5) for r in rows], [r["err"] for r in rows], color=col, alpha=a,
                              lw=1, label=f"||E_{method[0].upper()}(t)|| measured" if s == seeds[0] else None)
                g = [r for r in G_rows if r["h"] == h and r["seed"] == s and r["scaling"] == scaling]
                if g:
                    t = [r["step"] for r in g]
                    ax.loglog(t, [max(r["G_left_lo"], 1e-12) for r in g], "k-", marker=".", ms=3, alpha=a,
                              label="G(t) (left Riemann, certified lower)" if s == seeds[0] else None)
                    ax.loglog(t, [max(r["G_right_lo"], 1e-12) for r in g], "k--", alpha=a,
                              label="G(t) (right Riemann)" if s == seeds[0] else None)
                    ax.axhline(g[0]["two_dphi"], color="tab:green", ls=":", alpha=a,
                               label="2||Phi_h - id||" if s == seeds[0] else None)
            ax.set_ylim(bottom=1e-3)
            ax.set_title(f"h={h}, {scaling}", fontsize=9)
            ax.set_xlabel("Adam step")
            if j == 0:
                ax.set_ylabel("state units")
            if i == 0 and j == 0:
                ax.legend(fontsize=6)
    fig.suptitle("Unbuilt identity G(t) (path cost from Jacobians every ~50 steps: an approximation) vs "
                 "measured training errors; seed 0 solid, others faint", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_lorenz_G.png"), dpi=130)
    plt.close(fig)
    # G0
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for i, scaling in enumerate(("raw", "norm")):
        ax = axs[i]
        for h in hs:
            rows = [r for r in G0rows if r["h"] == h and r["seed"] == seeds[0] and r["scaling"] == scaling]
            ax.loglog([r["t_lam1"] for r in rows], [r["G0"] for r in rows], color=colors[h], label=f"h={h}")
            ax.axhline(rows[0]["two_dphi"], color=colors[h], ls=":")
        ax.set_xlabel("c^2 lambda_1(K0) t (gradient-flow time, fixed kernel)")
        ax.set_ylabel("G0(t) = ||exp(-c^2 K0 t) id||")
        ax.set_title(f"Corollary 'Kernel held fixed', {scaling} (dotted: 2||Phi_h - id||), seed {seeds[0]}", fontsize=9)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_lorenz_G0.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    cfg = dict(CFG, **(QUICK if a.quick else {}))
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T0 = time.time()
    log = lambda s: print(s, flush=True)
    sol = trajectory(cfg)
    log(f"[lorenz] trajectory {time.time() - T0:.1f}s, device {dev}")
    chk = C.Checks("3_lorenz")
    acc = [[] for _ in range(8)]
    for h in cfg["hs"]:
        for seed in cfg["seeds"]:
            t0 = time.time()
            res = run_pair(cfg, sol, h, seed, dev, chk, a.out, log)
            for lst, r in zip(acc, res):
                lst.extend(r)
            log(f"[h={h}/seed{seed}] done {time.time() - t0:.1f}s (total {time.time() - T0:.0f}s)")
    names = ["lorenz_init", "lorenz_init_krylov", "lorenz_G0", "lorenz_ckpt", "lorenz_ckpt_krylov", "lorenz_G",
             "lorenz_curves", "lorenz_runs"]
    for nm, rows in zip(names, acc):
        C.write_rows(os.path.join(a.out, nm + ".csv"), rows)
    chk.write(os.path.join(a.out, "lorenz_checks.csv"))
    figures(cfg, a.out, acc[1], acc[5], acc[6], acc[2])
    chk.print_summary()
    with open(os.path.join(a.out, "lorenz_meta.json"), "w") as f:
        json.dump(dict(cfg=cfg, device=dev, total_s=time.time() - T0), f, indent=1)
    log(f"[lorenz] total {time.time() - T0:.1f}s")
