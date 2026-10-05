"""Measurements M1-M7 for one model at one parameter vector on the held-out starts, and the exact
identity checks. Everything is matrix-free: per-start 3x3 state Jacobians, Jacobian-vector and
vector-Jacobian products in theta, and Lanczos (scipy eigsh) for spectral norms / top eigenvalues.

Sizes (config): M1, M2, M4, M5 on all n_test held-out starts; M3 and M6 on the first n_grad;
M7 (rollout kernel) on the first n_kernel.
"""
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.func import grad, jvp, vjp, vmap

from .core import gl_nodes, norm, nsteps, sq


def _cos(a, b):
    return float(a @ b / (torch.linalg.norm(a) * torch.linalg.norm(b)))


def _top_eig(matvec, N, dev, tol=1e-8):
    """Largest eigenvalue of a symmetric PSD operator given by a torch matvec (Lanczos)."""
    op = LinearOperator((N, N), dtype=np.float64,
                        matvec=lambda x: matvec(torch.as_tensor(x, device=dev).ravel()).cpu().numpy())
    v0 = np.random.default_rng(0).standard_normal(N)
    return float(eigsh(op, k=1, which="LA", tol=tol, v0=v0, return_eigenvectors=False)[0])


def forward_quantities(m, th, uT, truth, n_gl):
    """Rollout, errors e, one-step errors eps, J_i, Bbar_i (Gauss-Legendre), q_i, the propagated sums
    S_p = sum_i P(t_p,t_i) eps_i and Q_p = sum_i P(t_p,t_i) q_i (S_{i+1} = J_i S_i + eps_i), and the
    contraction factors rho_i, rhobar_i, rho^f_i along n_i = e_i/|e_i| (per start), i >= 1."""
    pmax = uT.shape[0] - 1
    dev = uT.device
    tau, wgl = gl_nodes(n_gl, dev)
    with torch.no_grad():
        up = m.rollout(th, uT[0], pmax)
    e = uT - up
    eps = uT[1:] - m.Gt(th, uT[:-1])
    J = torch.empty(pmax, *uT.shape[1:], 3, device=dev)
    S, Q = torch.zeros_like(uT), torch.zeros_like(uT)
    rho, rhob, rhof = (torch.full(uT.shape[:2], float("nan"), device=dev)[:pmax] for _ in range(3))
    for i in range(pmax):
        with torch.no_grad():
            J[i] = m.Jstate(th, up[i])
            B = sum(w * m.Jstate(th, up[i] + t * e[i]) for t, w in zip(tau, wgl))
            qi = torch.einsum("nab,nb->na", B - J[i], e[i])
            S[i + 1] = torch.einsum("nab,nb->na", J[i], S[i]) + eps[i]
            Q[i + 1] = torch.einsum("nab,nb->na", J[i], Q[i]) + qi
        if i >= 1:
            nv = e[i] / torch.linalg.norm(e[i], dim=-1, keepdim=True)
            rho[i] = torch.einsum("na,nab,nb->n", nv, J[i], nv)
            rhob[i] = torch.einsum("na,nab,nb->n", nv, B, nv)
            _, Gn = jvp(lambda u: truth.G(u, m.dt), (uT[i],), (nv,))
            rhof[i] = (nv * Gn).sum(-1)
    cumeps = torch.cat([torch.zeros_like(uT[:1]), torch.cumsum(eps.detach(), 0)])
    return dict(up=up, e=e.detach(), eps=eps.detach(), J=J, S=S, Q=Q, cumeps=cumeps,
                rho=rho, rhob=rhob, rhof=rhof)


def measure(m, th, data, cfg, checks=False):
    truth, dt = data["truth"], m.dt
    uT = data["uT_test"]
    dev = uT.device
    pmax = uT.shape[0] - 1
    Ts = cfg["horizons"]
    ps = [nsteps(T, dt) for T in Ts]
    th = th.detach()
    F = forward_quantities(m, th, uT, truth, cfg["gl_points"])
    e = F["e"]
    out = dict(T=np.array(Ts), p=np.array(ps))

    # per-step curves
    out["e_norm_t"] = np.array([norm(e[i]) for i in range(pmax + 1)])
    out["I_t"] = np.array([norm(F["cumeps"][i]) for i in range(pmax + 1)])
    out["eps_norm_t"] = np.array([norm(F["eps"][i]) for i in range(pmax)])

    # M1, M4 and the reconstruction identity e(t_p) = S_p + Q_p
    eN = np.array([norm(e[p]) for p in ps])
    I = np.array([norm(F["cumeps"][p]) for p in ps])
    out.update(e_norm=eN, I=I, R=eN - I,
               S_eps=np.array([norm(F["S"][p]) for p in ps]),
               S_q=np.array([norm(F["Q"][p]) for p in ps]))
    out["ratio_q"] = out["S_q"] / eN
    out["recon_relerr"] = np.array([norm(F["S"][p] + F["Q"][p] - e[p]) / norm(e[p]) for p in ps])
    out["recon_relerr_t"] = np.array([norm(F["S"][i] + F["Q"][i] - e[i]) / norm(e[i]) for i in range(1, pmax + 1)])

    # M2 (windows i = 1..p-1) and M5
    rho, rhob, rhof = F["rho"], F["rhob"], F["rhof"]
    ei, ub, v = e[1:pmax], uT[1:pmax], F["up"][1:pmax]
    en = torch.linalg.norm(ei, dim=-1)
    nv = ei / en[..., None]
    with torch.no_grad():
        dH = m.H(th, v) - m.H(th, ub)
        dfdt = (truth.G(v, dt) - v) / dt - (uT[2:pmax + 1] - ub) / dt   # f_dt(v) - f_dt(ubar)
    Re = (nv * dH).sum(-1).abs() / en
    Re_f = (nv * (dH - dfdt)).sum(-1).abs() / en
    one_m_rho = (1 - rho[1:]) / dt

    def win(a, p, fn):                       # a indexed from i = 1
        return float(fn(a[:p - 1])) if p > 1 else np.nan

    out["rho_m1"] = np.array([win(rho[1:] - 1, p, torch.mean) for p in ps])      # Euler: = dt n.gradH.n
    out["rhob_m1"] = np.array([win(rhob[1:] - 1, p, torch.mean) for p in ps])
    out["rhof_m1"] = np.array([win(rhof[1:] - 1, p, torch.mean) for p in ps])
    lnr = torch.log(rhob[1:].abs())
    out["sum_ln_rhob"] = np.array([float(lnr[:p - 1].sum(0).mean()) if p > 1 else 0.0 for p in ps])
    out["frac_rhob_nonpos"] = np.array([win((rhob[1:] <= 0).double(), p, torch.mean) for p in ps])
    out["Re_H"] = np.array([win(Re, p, torch.max) for p in ps])
    out["Re_H_minus_f"] = np.array([win(Re_f, p, torch.max) for p in ps])
    out["max_1m_rho_dt"] = np.array([win(one_m_rho, p, torch.max) for p in ps])
    out["rho_m1_t"] = (rho[1:] - 1).mean(1).cpu().numpy()
    out["rhob_m1_t"] = (rhob[1:] - 1).mean(1).cpu().numpy()
    out["rhof_m1_t"] = (rhof[1:] - 1).mean(1).cpu().numpy()

    # M6: M has rows n_i^T (d_theta H(v_i) - d_theta H(ubar_i)) / |e_k(t_i)| over (k, i <= p-1)
    ng = cfg["n_grad"]
    v6, u6, n6, e6 = v[:, :ng], ub[:, :ng], nv[:, :ng], en[:, :ng]

    def rows_fn(t, i_hi):
        dHt = m.H(t, v6[:i_hi]) - m.H(t, u6[:i_hi])
        return ((n6[:i_hi] * dHt).sum(-1) / e6[:i_hi]).reshape(-1)

    def row1(t, a, b, nn, ee):
        return (nn * (m.H(t, a[None]) - m.H(t, b[None]))[0]).sum() / ee

    prow = vmap(grad(row1), in_dims=(None, 0, 0, 0, 0))
    rownorm = torch.empty(pmax - 1, ng, device=dev)
    for i in range(pmax - 1):
        for s in range(0, ng, cfg["chunk"]):
            sl = slice(s, s + cfg["chunk"])
            rownorm[i, sl] = torch.linalg.norm(prow(th, v6[i, sl], u6[i, sl], n6[i, sl], e6[i, sl]), dim=1)
    spec, mrow = [], []
    for p in ps:
        if p <= 1:
            spec.append(np.nan)
            mrow.append(np.nan)
            continue
        _, vjp_fn = vjp(lambda t: rows_fn(t, p - 1), th)
        mv = lambda x: vjp_fn(jvp(lambda t: rows_fn(t, p - 1), (th,), (x,))[1])[0]
        spec.append(np.sqrt(_top_eig(mv, m.P, dev)))
        mrow.append(float(rownorm[:p - 1].max()))
    out["M6_spec"], out["M6_maxrow"] = np.array(spec), np.array(mrow)

    # M3: cosines of the one-step velocity g^(1) with per-horizon g^(q) and cumulative g^(Sigma)(p)
    g = per_horizon_velocities(m, th, uT[:, :ng], cfg["chunk_cot"])
    gsum = torch.cumsum(g, 0) / torch.arange(1, pmax + 1, device=dev)[:, None]
    out["c_q"] = np.array([_cos(g[0], g[q]) for q in range(pmax)])
    out["c_Sigma"] = np.array([_cos(g[0], gsum[q]) for q in range(pmax)])
    out["g_norm_q"] = torch.linalg.norm(g, dim=1).cpu().numpy()

    # M7: rollout kernel K^(p) = (1/n) Psi Psi^T on the first n_kernel starts
    nk = cfg["n_kernel"]
    lam, trK, PTe = [], [], []
    for p in ps:
        f = lambda t: m.rollout(t, uT[0, :nk], p)[-1].reshape(-1)
        _, vjp_fn = vjp(f, th)
        Kv = lambda x: jvp(f, (th,), (vjp_fn(x)[0],))[1] / nk
        lam.append(_top_eig(Kv, 3 * nk, dev))
        tr = 0.0
        eye = torch.eye(3 * nk, device=dev)
        for s in range(0, 3 * nk, cfg["chunk_cot"]):
            tr += float((vmap(vjp_fn)(eye[s:s + cfg["chunk_cot"]])[0] ** 2).sum())
        trK.append(tr / nk)
        # |P^T ehat|: adjoint pass a_{p-1} = ehat, a_i = J_{i+1}^T a_{i+1}
        ek = e[p, :nk]
        a = ek / norm(ek)
        acc = sq(a)
        for i in range(p - 2, -1, -1):
            a = torch.einsum("nba,nb->na", F["J"][i + 1, :nk], a)
            acc = acc + sq(a)
        PTe.append(float(torch.sqrt(acc)))
    out["K_lam_max"] = np.array(lam)
    out["K_tr_over_lam"] = np.array(trK) / np.array(lam)
    out["PT_ehat"] = np.array(PTe)

    if checks:
        out.update(identity_checks(m, th, uT[:, :nk], cfg))
    return out


def per_horizon_velocities(m, th, uT, chunk):
    """g^(q) = -grad (1/2)|e(t_q)|^2 for q = 1..pmax: one forward graph, batched backward passes."""
    pmax = uT.shape[0] - 1
    th_ = th.detach().clone().requires_grad_(True)
    u, Ls = uT[0], []
    for q in range(1, pmax + 1):
        u = m.Gt(th_, u)
        Ls.append(0.5 * sq(uT[q] - u))
    L = torch.stack(Ls)
    eye = torch.eye(pmax, device=uT.device)
    G = []
    for s in range(0, pmax, chunk):
        G.append(torch.autograd.grad(L, th_, grad_outputs=eye[s:s + chunk], is_grads_batched=True,
                                     retain_graph=True)[0])
    return -torch.cat(G).detach()


def identity_checks(m, th, uT, cfg, Tlist=(0.32, 1.28), seed=0):
    """(1) Psi^(p) g vs central FD of the rollout; (2) g^(Sigma)(p) . g vs -FD of L_p along g;
    random unit directions g. Max relative errors."""
    from .core import loss_unrolled
    gen = torch.Generator().manual_seed(seed)
    h = cfg["fd_step"]
    dev = uT.device
    e1 = e2 = 0.0
    L = lambda t, p: loss_unrolled(m.arch, t, m.c, uT, p)
    for T in Tlist:
        p = nsteps(T, m.dt)
        for _ in range(3):
            gdir = torch.randn(m.P, generator=gen).to(dev)
            gdir /= torch.linalg.norm(gdir)
            f = lambda t: m.rollout(t, uT[0], p)[-1]
            _, Psig = jvp(f, (th,), (gdir,))
            fd = (f(th + h * gdir) - f(th - h * gdir)) / (2 * h)
            e1 = max(e1, float(torch.linalg.norm(Psig - fd) / torch.linalg.norm(Psig)))
            gS = -grad(lambda t: L(t, p))(th)
            fdL = (L(th + h * gdir, p) - L(th - h * gdir, p)) / (2 * h)
            e2 = max(e2, float(abs(-gS @ gdir - fdL) / abs(fdL)))
    return dict(chk_Psi_fd=e1, chk_gSigma_fd=e2)
