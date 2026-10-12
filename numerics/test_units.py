"""Part 1: unit tests of the one-step results on small random matrices.

Theorem 'Supremum over roughness functions' (thm:stepsup), Proposition 'Kernel roughness is the
supremum over roughness functions' (prop:krough), Theorem 'Krylov series' (thm:stepkrylov), and
Proposition 'Every step of length at most s' (prop:stepexact) on a small nonlinear network.
Writes results/unit_checks.csv (one row per checked inequality) and results/unit_values.csv.
"""
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
chk = C.Checks("1_unit")
values = []


def make_J(rng, N, d, r, lam_ratio, n):
    """J with rank r and kernel eigenvalues spread over lam_ratio (log-spaced)."""
    Q1, _ = np.linalg.qr(rng.normal(size=(N, r)))
    Q2, _ = np.linalg.qr(rng.normal(size=(d, r)))
    sig = np.logspace(0, -0.5 * np.log10(lam_ratio), r)
    return np.sqrt(n) * (Q1 * sig) @ Q2.T


def F_direct(E, J, c, s, n, rng, starts=4):
    """min_{|g|<=s} ||E + c J g|| by SLSQP (feasible point, so the value is >= F(s))."""
    f = lambda g: np.sum((E + c * J @ g) ** 2) / n
    gr = lambda g: 2 * c * J.T @ (E + c * J @ g) / n
    cons = dict(type="ineq", fun=lambda g: s ** 2 - g @ g, jac=lambda g: -2 * g)
    best = np.inf
    for k in range(starts):
        g0 = np.zeros(J.shape[1]) if k == 0 else rng.normal(size=J.shape[1])
        g0 *= min(1.0, 0.9 * s / (np.linalg.norm(g0) + 1e-300))
        r = minimize(f, g0, jac=gr, constraints=[cons], method="SLSQP",
                     options=dict(ftol=1e-15, maxiter=2000))
        g = r.x * min(1.0, s / max(np.linalg.norm(r.x), 1e-300))
        best = min(best, np.sqrt(f(g)))
    return best


def F_dual(E, J, c, s, n, rng, starts=4):
    """max_{||phi||<=1} <phi,E> - |c| s ||phi||_K by SLSQP (feasible point, so the value is <= F(s))."""
    kn = lambda p: np.linalg.norm(J.T @ p) / n
    h = lambda p: -(p @ E / n - abs(c) * s * kn(p))

    def gh(p):
        v = J.T @ p
        return -(E / n - abs(c) * s * (J @ v) / (n * max(np.linalg.norm(v), 1e-300)))
    cons = dict(type="ineq", fun=lambda p: 1 - p @ p / n, jac=lambda p: -2 * p / n)
    best = -np.inf
    for k in range(starts):
        p0 = E.copy() if k == 0 else rng.normal(size=E.shape)
        p0 /= np.linalg.norm(p0) / np.sqrt(n)
        r = minimize(h, p0, jac=gh, constraints=[cons], method="SLSQP",
                     options=dict(ftol=1e-15, maxiter=2000))
        p = r.x / max(1.0, np.linalg.norm(r.x) / np.sqrt(n))
        best = max(best, -h(p))
    return max(best, 0.0)


def seminorm_family(x, n, n_out, rng, J):
    fam = []
    for m in (3, 8):
        B, _ = C.graph_highpass_basis(x, k=8, m=m, n_out=n_out)
        fam.append(C.ProjectorSeminorm(B, n, name=f"highpass_m{m}"))
    N = n * n_out
    for k in (5, 20, 40):  # projector onto a random k-dim subspace: P = I - B B^T with B spanning the complement
        Q, _ = np.linalg.qr(rng.normal(size=(N, N)))
        fam.append(C.ProjectorSeminorm(Q[:, k:], n, name=f"randproj_k{k}"))
    fam.append(C.LipschitzSeminorm(x, n_out))
    return fam


def test_case(name, J, E, x, n, n_out, c, rng):
    N = J.shape[0]
    ker = C.Kernel(J, n)
    lam_ext, e_ext = ker.reduced(E)
    nE = C.nrm(E, n)
    Khat = ker.knorm(E / nE)
    pos = lam_ext > 0
    s_full = np.sqrt(np.sum(e_ext[pos] ** 2 / lam_ext[pos])) / abs(c)
    qE = C.distinct_count(lam_ext, e_ext)
    fam = seminorm_family(x, n, n_out, rng, J)
    sig = {R.name: R.sigma(J) for R in fam}
    print(f"[{name}] N={N} d={J.shape[1]} rank={ker.rank} c_p={c} ||E||={nE:.4f} "
          f"null part={e_ext[-1]:.4f} q_E={qE} s_full={s_full:.4g}", flush=True)

    # ---------------- Theorem stepsup and Theorem stepkrylov, over a range of step lengths
    for frac in (0.02, 0.1, 0.3, 0.6, 0.9, 1.3):
        s = frac * s_full
        cs = abs(c) * s
        case = f"{name}/s={frac}s_full"
        F, gam, Es = C.F_closed(lam_ext, e_ext, cs)
        Fd = F_direct(E, J, c, s, n, rng)
        Fu = F_dual(E, J, c, s, n, rng)
        values.append(dict(case=case, F_closed=F, F_direct=Fd, F_dual=Fu, gamma=gam, norm_E=nE))
        chk.add_eq("stepsup(d): closed form = direct min over |g|<=s (SLSQP)", case, F, Fd, 1e-6, scale=nE)
        chk.add_eq("stepsup(d): closed form = dual max over phi (SLSQP)", case, F, Fu, 1e-6, scale=nE)
        chk.add("stepsup: dual <= F <= direct (weak duality)", case,
                min(F - Fu, Fd - F), tol=1e-9 * nE, scale=nE)
        for R in fam:
            b = (R(E) - cs * sig[R.name]) / R.C
            chk.add("stepsup(a): b_R(s) <= F(s)", f"{case}/{R.name}", F - b, tol=1e-10 * nE,
                    scale=nE, b_R=b, F=F)
        chk.add("stepsup(e): F(s) >= ||E|| - |c|s||E^||_K", case, F - (nE - cs * Khat),
                tol=1e-10 * nE, scale=nE)
        if F <= 1e-12 * nE:
            chk.add("stepkrylov: F_M = 0 when F = 0", case, -np.max(C.krylov_F(lam_ext, e_ext, cs, qE + 2)[0]),
                    tol=1e-10 * nE, scale=nE)
            continue
        # rank-one seminorm phi* attains F (thm stepsup (b)); phi* built in full coordinates
        Estar_full = np.sqrt(n) * (ker.U @ Es[:-1])
        null_full = np.sqrt(n) * (E / np.sqrt(n) - ker.U @ ker.reduce(E)[0])
        if e_ext[-1] > 0:
            Estar_full = Estar_full + null_full * (Es[-1] / e_ext[-1])
        phis = Estar_full / C.nrm(Estar_full, n)
        R1 = C.RankOneSeminorm(phis, n)
        b1 = C.b_R(R1, E, J, cs)
        chk.add_eq("stepsup(b): rank-one phi* attains F", case, b1, F, 1e-8, scale=nE)
        gstar = np.linalg.lstsq(c * J, Estar_full - E, rcond=None)[0]
        chk.add_eq("stepsup(d): E* = E + c Psi g* for some |g*| <= s (residual)", case,
                   C.nrm(E + c * J @ gstar - Estar_full, n), 0.0, 1e-8, scale=nE)
        chk.add("stepsup(d): E* = E + c Psi g* for some |g*| <= s (length)", case,
                s - np.linalg.norm(gstar), tol=1e-8 * s, scale=s)
        # Krylov series
        Mmax = qE + 3
        FM, Q = C.krylov_F(lam_ext, e_ext, cs, Mmax)
        values.append(dict(case=case, **{f"F_M{M}": FM[M - 1] for M in range(1, Mmax + 1)}))
        chk.add("stepkrylov(a): F_M nondecreasing in M", case, np.min(np.diff(FM)), tol=1e-10 * nE, scale=nE)
        chk.add("stepkrylov(a): F_M <= F", case, np.min(F - FM), tol=1e-10 * nE, scale=nE)
        chk.add_eq("stepkrylov(a): F_1 = max(0, ||E|| - |c|s||E^||_K)", case, FM[0], max(0, nE - cs * Khat),
                   1e-9, scale=nE)
        chk.add_eq("stepkrylov(c): F_M = F at M = q_E", case, FM[qE - 1], F, 1e-8, scale=nE, q_E=qE)
        phi_red = Es / np.linalg.norm(Es)
        dM = C.dist_to_span(Q, phi_red)
        dM = np.concatenate([dM, np.full(Mmax - len(dM), dM[-1])])
        lo = 0.5 * F * dM ** 2
        hi = (nE + cs * np.sqrt(ker.lam1)) * dM
        chk.add("stepkrylov(b): F - F_M >= F d_M^2 / 2", case, np.min((F - FM) - lo), tol=1e-9 * nE, scale=nE)
        chk.add("stepkrylov(b): F - F_M <= (||E|| + |c|s lam1^1/2) d_M", case, np.min(hi - (F - FM)),
                tol=1e-9 * nE, scale=nE)
        if gam is not None and np.isfinite(gam):
            kap = (ker.lam1 + gam) / gam
            rho = (np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)
            bound = (kap - 1) / 2 * rho ** np.arange(Mmax)
            chk.add("stepkrylov(d): d_M <= (kappa-1)/2 rho^(M-1)", case, np.min(bound - dM), tol=1e-10,
                    scale=1.0, kappa=kap)

    # ---------------- Proposition krough
    w = rng.normal(size=N)
    u_in = J @ (J.T @ w) / n                                 # in the range of K
    u_rand = rng.normal(size=N)                              # generic (outside the range unless K is full rank)
    Kfull = J @ J.T / n
    Kp = np.linalg.pinv(Kfull, rcond=1e-12, hermitian=True)
    for uname, u in (("u=Kw", u_in), ("u=random", u_rand)):
        case = f"{name}/{uname}"
        lam_u, u_ext = ker.reduced(u)
        kin = C.kinv_norm(lam_u, u_ext)
        resid = np.linalg.norm(Kfull @ (Kp @ u) - u) / np.linalg.norm(u)
        ref = np.sqrt(u @ Kp @ u / n) if resid < 1e-8 else np.inf
        chk.add_eq("krough: ||u||_{K+} (thin SVD) = pinv(K) reference (inf = outside range)", case, kin, ref,
                   1e-7, scale=ref if np.isfinite(ref) else 1.0)
        qu = C.distinct_count(lam_u, u_ext)
        Mmax = qu + 3
        vals, rM = C.krylov_kinv(lam_u, u_ext, Mmax)
        values.append(dict(case=case, kinv=kin, q_u=qu, **{f"kinvM{M}": vals[M - 1] for M in range(1, Mmax + 1)}))
        finite = np.isfinite(vals)
        dif = np.diff(np.where(finite, vals, 1e300))
        chk.add("krough(b): ||u||_{K+,M} nondecreasing in M", case, np.min(dif), tol=1e-9 * np.nanmax(vals[finite]),
                scale=np.nanmax(vals[finite]))
        chk.add_eq("krough(b): ||u||_{K+,M} = ||u||_{K+} at M = q_u (inf = outside range)", case, vals[qu - 1],
                   kin, 1e-7, scale=kin if np.isfinite(kin) else 1.0, q_u=qu)
        if qu > 1:
            chk.add("info: ||u||_{K+,M} < ||u||_{K+} at M = q_u - 1 (not reached early; not a claim of the paper)", case,
                    (kin - vals[qu - 2]) if np.isfinite(kin) else (0.0 if np.isfinite(vals[qu - 2]) else -1.0),
                    scale=kin if np.isfinite(kin) else 1.0)
        if np.isfinite(kin):
            nz = np.abs(u_ext[:-1]) > 1e-12 * np.linalg.norm(u_ext)
            ku = lam_u[:-1][nz].max() / lam_u[:-1][nz].min()
            rho = (np.sqrt(ku) - 1) / (np.sqrt(ku) + 1)
            Ms = np.arange(1, Mmax + 1)
            cg = kin ** 2 * (1 - 4 * rho ** (2 * Ms))
            chk.add("krough(c): CG bound ||u||_{K+,M}^2 >= ||u||^2(1 - 4 rho^2M)", case,
                    np.min(vals ** 2 - cg), tol=1e-9 * kin ** 2, scale=kin ** 2, kappa_u=ku)
        # seminorm ratios (prop krough (a)) and the rank-one functional K+ u
        for R in fam:
            sR = R.sigma(J)
            if sR > 0:
                ratio = R(u) / sR
                chk.add("krough(a): R(u)/sigma_R <= ||u||_{K+}", f"{case}/{R.name}",
                        (kin - ratio) if np.isfinite(kin) else 1.0, scale=ratio if ratio else 1.0,
                        ratio=ratio, kinv=kin)
        if np.isfinite(kin):
            R1 = C.RankOneSeminorm(Kp @ u, n)
            ratio = R1(u) / R1.sigma(J)
            chk.add_eq("krough(a): equality for R(v) = |<K+u, v>|", case, ratio, kin, 1e-8, scale=kin)
        # cross-check the Lanczos Krylov values against an explicit monomial Krylov basis (small M)
        for M in range(1, min(5, qu) + 1):
            Vk = np.stack([np.linalg.matrix_power(Kfull, j) @ u for j in range(M)], 1)
            Qe, _ = np.linalg.qr(Vk)
            a = Qe.T @ u
            G = Qe.T @ Kfull @ Qe
            # <phi,u> = phi.u/n, ||phi||_K^2 = phi.K phi/n with phi = Qe c -> sup^2 = a^T G^-1 a / n
            ref = np.sqrt(a @ np.linalg.solve(G, a) / n)
            chk.add_eq("krough: Lanczos Krylov value = explicit Krylov basis (M<=5)", f"{case}/M={M}",
                       vals[M - 1], ref, 1e-6, scale=ref)


def test_stepexact(rng):
    """Prop. stepexact on a small nonlinear network with real steps (averaged Jacobian by quadrature)."""
    n, din, H, dout = 25, 2, 5, 2
    net = C.MLP1(din, H, dout)
    x = rng.uniform(-1, 1, size=(n, din))
    y = np.stack([np.sin(2 * x[:, 0]) * np.cos(x[:, 1]), np.tanh(x[:, 0] - x[:, 1] ** 2)], 1).ravel()
    th = rng.normal(size=net.d)
    E = net.out(th, x) - y
    J0 = net.jac(th, x)
    ker0 = C.Kernel(J0, n)
    nE = C.nrm(E, n)
    Bhp, _ = C.graph_highpass_basis(x, k=8, m=5, n_out=dout)
    Rhp = C.ProjectorSeminorm(Bhp, n, name="highpass_m5")
    grad = J0.T @ E / n
    steps = [("gd", eta, -eta * grad) for eta in (0.1 / ker0.lam1, 1 / ker0.lam1, 5 / ker0.lam1, 30 / ker0.lam1)]
    for L in (0.01, 0.1, 0.5, 1.0, 2.0):
        g = rng.normal(size=net.d)
        steps.append(("rand", L, L * g / np.linalg.norm(g)))
    print(f"[stepexact] N={n * dout} d={net.d} ||E||={nE:.4f}", flush=True)
    for kind, par, dth in steps:
        s = np.linalg.norm(dth)
        case = f"stepexact/{kind}/{par:.3g}/s={s:.3g}"
        Enew = net.out(th + dth, x) - y
        nEnew = C.nrm(Enew, n)
        Jb = {}
        for q in (32, 64):
            t, w = C.gauss_legendre_01(q)
            Jb[q] = sum(wi * net.jac(th + ti * dth, x) for ti, wi in zip(t, w))
        idres = np.linalg.norm(E + Jb[64] @ dth - Enew) / max(np.linalg.norm(Enew - E), 1e-300)
        qdiff = np.linalg.norm(Jb[64] - Jb[32]) / np.linalg.norm(Jb[64])
        chk.add_eq("stepexact: E_new = E + Psibar g (quadrature identity, rel resid)", case, idres, 0.0, 1e-10,
                   quad32_vs_64=qdiff)
        kb = C.Kernel(Jb[64], n)
        lb, eb = kb.reduced(E)
        Fbar, _, Esb = C.F_closed(lb, eb, s)
        chk.add("stepexact: ||E_new|| >= F(s) with averaged Jacobian", case, nEnew - Fbar, tol=1e-10 * nE,
                scale=nE, E_new=nEnew, F_bar=Fbar)
        l0, e0 = ker0.reduced(E)
        F0 = C.F_closed(l0, e0, s)[0]
        chk.add("info: ||E_new|| >= F(s) with Jacobian at theta (first order only; may fail)", case,
                nEnew - F0, scale=nE, E_new=nEnew, F_theta=F0)
        # rigorous form with kappa_bar(s) = max over the ball B_s: sampled (an underestimate of the max,
        # so the right side used here is >= the true right side; passing here implies the true bound holds)
        pts = [th + t * dth for t in np.linspace(0, 1, 33)]
        for k in range(300):
            g = rng.normal(size=net.d)
            pts.append(th + s * g / np.linalg.norm(g) * (1.0 if k % 2 else rng.uniform() ** (1 / net.d)))
        Js = [net.jac(p, x) for p in pts]

        def full_phi(red, kr, u):
            c, u0 = kr.reduce(u)
            v = np.sqrt(n) * (kr.U @ red[:-1])
            if u0 > 0:
                v = v + np.sqrt(n) * (u / np.sqrt(n) - kr.U @ c) * red[-1] / u0
            return v
        phis = {"phi*_avg": full_phi(Esb, kb, E), "E_hat": E.copy(), "random": rng.normal(size=E.shape)}
        phis["phi*_theta"] = full_phi(C.F_closed(l0, e0, s)[2], ker0, E)
        for pn, p in phis.items():
            p = p / C.nrm(p, n)
            kbar = max(np.linalg.norm(Jp.T @ p) / n for Jp in Js)
            rhs = C.ip(p, E, n) - s * kbar
            chk.add("stepexact: ||E_new|| >= <phi,E> - s kappabar_phi(s)", f"{case}/{pn}", nEnew - rhs,
                    tol=1e-10 * nE, scale=nE, rhs=rhs)
        sbar = max(Rhp.sigma(Jp) for Jp in Js)
        rhs = (Rhp(E) - s * sbar) / Rhp.C
        chk.add("stepexact: ||E_new|| >= (R(E) - s sigmabar_R(s))/C_R", f"{case}/highpass_m5", nEnew - rhs,
                tol=1e-10 * nE, scale=nE, rhs=rhs)


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(12345)
    n, n_out = 25, 2
    N = n * n_out
    x = rng.uniform(size=(n, 2))
    E = rng.normal(size=N)
    test_case("A_d20_rank20", make_J(rng, N, 20, 20, 1e4, n), E, x, n, n_out, 1.0, rng)
    test_case("B_d20_rank12", make_J(rng, N, 20, 12, 1e4, n), E, x, n, n_out, 1.0, rng)
    test_case("C_d80_fullrankK_c0.37", make_J(rng, N, 80, 50, 1e3, n), E, x, n, n_out, 0.37, rng)
    test_stepexact(rng)
    chk.write(os.path.join(OUT, "unit_checks.csv"))
    C.write_rows(os.path.join(OUT, "unit_values.csv"), values)
    chk.print_summary()
    print(f"wall {time.time() - t0:.1f}s")
