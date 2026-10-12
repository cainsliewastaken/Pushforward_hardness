"""Numerical validation of training_theory.tex, GPU version with Hessian-vector products (no full Hessian tensors).

float64 throughout. Tests, selected with --test:
  1  one step: F(s), Krylov F_M(s), seminorm floors, best step vs F +/- remainder, directional remainder   (Thm 1, Prop 1, Thm 2)
  2  along GD: Theorem 9 floor in path / displacement / directional form, true remainder, per-direction floors,
     kernel growth and drive on rough directions                                                             (Thm 8, Thm 9, Prop kgrowth)
  3  descending optimizers: SGD descent-check margins; heavy ball; Adam with M_P (preconditioned Hessian norm)
     against M*lam_max(P), metric drift length against its a priori bound                                   (Part II)
  4  Euler vs direct on Lorenz-63: Cor 5(b) in path and displacement form                                    (Cor 5)
"""
import argparse, math, os, time
import torch
from torch.func import functional_call, jacrev, jvp, grad, vmap

torch.set_default_dtype(torch.float64)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NPTS = None


class MLP(torch.nn.Module):
    def __init__(self, din, width, dout, zero_last=False):
        super().__init__()
        self.l1 = torch.nn.Linear(din, width); self.l2 = torch.nn.Linear(width, width); self.l3 = torch.nn.Linear(width, dout)
        if zero_last:
            torch.nn.init.zeros_(self.l3.weight); torch.nn.init.zeros_(self.l3.bias)

    def forward(self, x):
        return self.l3(torch.tanh(self.l2(torch.tanh(self.l1(x)))))


class Net:
    def __init__(self, model, X):
        self.model, self.X = model.to(DEV), X.to(DEV)
        self.names, self.shapes, vals = [], [], []
        for nme, p in self.model.named_parameters():
            self.names.append(nme); self.shapes.append(p.shape); vals.append(p.detach().reshape(-1))
        self.theta0 = torch.cat(vals); self.d = self.theta0.numel()
        self.n, self.ny = X.shape[0], model.l3.out_features; self.m = self.n * self.ny

    def _unflat(self, theta):
        out, i = {}, 0
        for nme, s in zip(self.names, self.shapes):
            k = 1
            for q in s: k *= q
            out[nme] = theta[i:i + k].reshape(s); i += k
        return out

    def outflat(self, theta):
        return functional_call(self.model, self._unflat(theta), (self.X,)).reshape(-1)

    def Psi(self, theta):
        return jacrev(self.outflat)(theta)

    def dir2(self, theta, v):                       # (m,): nabla^2 N_i [v, v] for every output entry
        return jvp(lambda th: jvp(self.outflat, (th,), (v,))[1], (theta,), (v,))[1]

    def S_mv(self, theta, phi, w):                  # S(phi) w = (1/n) sum_i phi_i nabla^2 N_i w
        f = lambda th: (phi @ self.outflat(th)) / NPTS
        return jvp(grad(f), (theta,), (w,))[1]

    def H2(self, theta, pts=None, iters=12):
        """RMS over points of max over outputs of the spectral norm of the per-output Hessian (power iteration)."""
        if pts is None: pts = torch.arange(self.n, device=DEV)
        sel = (pts[:, None] * self.ny + torch.arange(self.ny, device=DEV)[None, :]).reshape(-1)
        oh = torch.zeros(len(sel), self.m, device=DEV); oh[torch.arange(len(sel), device=DEV), sel] = 1.0
        V = torch.randn(len(sel), self.d, device=DEV); V = V / V.norm(dim=1, keepdim=True)
        def Hv(o, v):
            return grad(lambda th: (jvp(self.outflat, (th,), (v,))[1] * o).sum())(theta)
        hv = vmap(Hv)
        for _ in range(iters):
            HV = hv(oh, V); V = HV / HV.norm(dim=1, keepdim=True).clamp_min(1e-300)
        lam = (hv(oh, V) * V).sum(1).abs().reshape(len(pts), self.ny).max(1).values
        return float(torch.sqrt((lam ** 2).mean()))

    def h2_dir(self, theta0, Delta, taus=(0.0, 0.25, 0.5, 0.75, 1.0)):
        """Directional second derivative along the segment theta0 + tau*Delta: max_tau ||nabla^2 N[Dhat, Dhat]|| (sampled-function norm)."""
        r = float(Delta.norm())
        if r == 0: return 0.0
        Dh = Delta / r
        return max(nrm(self.dir2(theta0 + t * Delta, Dh)) for t in taus)

    def remainder(self, theta0, Psi0, theta):       # N(theta) - N(theta0) - Psi0 (theta - theta0)
        return self.outflat(theta) - self.outflat(theta0) - Psi0 @ (theta - theta0)


def ip(a, b): return float((a * b).sum() / NPTS)
def nrm(a): return math.sqrt(max(ip(a, a), 0.0))
def kernel(Psi): return Psi @ Psi.T / NPTS
def comps(V, E): return (V.T @ E) / NPTS


def eig(K):
    lam, U = torch.linalg.eigh(K)
    return lam.clamp_min(0.0).flip(0), (U * math.sqrt(NPTS)).flip(1)


def opnorm(mv, d, iters=30):
    v = torch.randn(d, device=DEV); v = v / v.norm()
    for _ in range(iters):
        w = mv(v); v = w / w.norm().clamp_min(1e-300)
    return float(abs(v @ mv(v)))


def loss_hvp(net, Y):
    def L(th):
        E = net.outflat(th) - Y
        return 0.5 * (E @ E) / NPTS
    return lambda th, w: jvp(grad(L), (th,), (w,))[1]


def ridge_F(lam, ehat, s, tol=1e-12):
    pos = lam > tol
    U0 = float((ehat[pos] ** 2 / lam[pos]).sum())
    if s ** 2 >= U0:
        return math.sqrt(float((ehat[~pos] ** 2).sum())), None
    def Ups(g): return float((lam * ehat ** 2 / (lam + g) ** 2).sum())
    lo, hi = -30.0, 30.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Ups(math.exp(mid)) > s ** 2: lo = mid
        else: hi = mid
    g = math.exp(0.5 * (lo + hi))
    return math.sqrt(float(((g * ehat / (lam + g)) ** 2).sum())), g


def krylov_F(K, E, s, M, iters=3000):
    vecs, v = [], E.clone()
    for _ in range(M):
        for q in vecs: v = v - ip(q, v) * q
        nv = nrm(v)
        if nv < 1e-10: break
        v = v / nv; vecs.append(v); v = K @ v
    Q = torch.stack(vecs, 1)
    b = Q.T @ E / NPTS; G = Q.T @ K @ Q / NPTS
    c = b / b.norm(); best = 0.0
    step = 0.5 / (1.0 + s * float(torch.linalg.eigvalsh(G).max()) ** 0.5)
    for _ in range(iters):
        qf = torch.sqrt(c @ G @ c + 1e-30)
        val = float(b @ c - s * qf); best = max(best, val)
        c = c + step * (b - s * (G @ c) / qf)
        if c.norm() > 1: c = c / c.norm()
    return best


def graph_highpass_projector(X, m, kg=4):
    n = X.shape[0]
    D2 = torch.cdist(X, X) ** 2
    order = D2.sort(dim=1)
    rg = order.values[:, kg].sqrt().median()
    W = torch.exp(-D2 / rg ** 2)
    mask = torch.zeros_like(W); mask.scatter_(1, order.indices[:, 1:kg + 1], 1.0)
    mask = torch.maximum(mask, mask.T); W = W * mask
    Lg = torch.diag(W.sum(1)) - W
    _, U = torch.linalg.eigh(Lg)
    return torch.eye(n, device=DEV) - U[:, :m] @ U[:, :m].T


def cluster_projector(n, size):
    P = torch.eye(n, device=DEV)
    for c in range(0, n, size):
        idx = torch.arange(c, min(c + size, n), device=DEV)
        P[idx[:, None], idx[None, :]] -= 1.0 / len(idx)
    return P


def seminorm_floor(P, K, E, s):
    sig = math.sqrt(max(float(torch.linalg.eigvalsh(P @ K @ P).max()), 0.0))
    return nrm(P @ E) - s * sig, sig


def f(x): return f"{x: .4e}"


def sine_setup(width, n, k):
    X = torch.linspace(-1, 1, n).unsqueeze(1); Y = torch.sin(k * math.pi * X).reshape(-1).to(DEV)
    torch.manual_seed(0)
    return Net(MLP(1, width, 1), X), X.to(DEV), Y


# ----------------------------------------------------------------------------- test 1
def test1(out, width=64, n=512, ks=(1, 4, 8), steps=(0.02, 0.05, 0.1, 0.2, 0.5)):
    global NPTS; NPTS = n
    L = [f"# Test 1: width={width} n={n}; sampled-function norms (RMS over the n points).",
         "# cols: k state s | F(s) F_1 F_3 F_10 | highpass(m=16) cluster(8) | err_best_step  upper=F+0.5*H2bar*s^2  upper_dir=F+0.5*h2(g*)*s^2  true_rem | err_GD_step"]
    for k in ks:
        net, X, Y = sine_setup(width, n, k)
        Php = graph_highpass_projector(X, 16); Pcl = cluster_projector(n, 8)
        th = net.theta0.clone()
        states = {"init": th.clone()}
        Psi = net.Psi(th); lam1 = float(eig(kernel(Psi))[0][0]); eta = 0.5 / lam1
        for j in range(300):
            E = net.outflat(th) - Y; Psi = net.Psi(th); th = th - eta * (Psi.T @ E / n)
        states["mid300"] = th.clone()
        for sname, th in states.items():
            E = net.outflat(th) - Y; Psi = net.Psi(th); K = kernel(Psi); lam, V = eig(K); e = comps(V, E)
            H2c = net.H2(th)
            H2b = H2c
            for _ in range(3):
                dth = torch.randn(net.d, device=DEV); dth = 0.5 * dth / dth.norm(); H2b = max(H2b, net.H2(th + dth))
            for s in steps:
                Fs, g = ridge_F(lam, e, s)
                F1, F3, F10 = (krylov_F(K, E, s, M) for M in (1, 3, 10))
                hp, _ = seminorm_floor(Php, K, E, s); cl, _ = seminorm_floor(Pcl, K, E, s)
                if g is None:
                    gbest = -torch.linalg.lstsq(Psi, E).solution
                    if gbest.norm() > s: gbest = gbest * (s / gbest.norm())
                else:
                    gbest = -Psi.T @ torch.linalg.solve(K + g * torch.eye(n, device=DEV), E) / n
                Eb = net.outflat(th + gbest) - Y
                rem = nrm(net.remainder(th, Psi, th + gbest)); h2 = net.h2_dir(th, gbest)
                gd = -Psi.T @ E / n; gd = gd * (s / gd.norm()); Egd = net.outflat(th + gd) - Y
                L.append(" ".join([str(k), sname, f"{s:.3f}", "|", f(Fs), f(F1), f(F3), f(F10), "|", f(hp), f(cl), "|",
                                   f(nrm(Eb)), f(Fs + 0.5 * H2b * s ** 2), f(Fs + 0.5 * h2 * s ** 2), f(rem), "|", f(nrm(Egd))]))
            L.append(f"#   k={k} {sname}: ||E||={nrm(E):.4e} lam1={float(lam[0]):.3e} lam_min={float(lam[-1]):.3e} H2(center)={H2c:.3e} H2bar(r=0.5)={H2b:.3e}")
            print("\n".join(L[-len(steps) - 1:]), flush=True)
    open(os.path.join(out, "test1.txt"), "w").write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- test 2
def test2(out, width=64, n=512, ks=(4, 1), nsteps=4000, every=100, rebase=1000, opt="gd"):
    """opt: 'gd' (eta = 0.5/lam1), 'eos' (eta = 1.8/lam1(0), gradient descent that reaches the edge of stability),
    'adam' (full-batch Adam, eta = 1e-3, bias-corrected). The floors are path-independent, so the same bookkeeping applies."""
    global NPTS; NPTS = n
    L = []
    for k in ks:
        net, X, Y = sine_setup(width, n, k)
        th0 = net.theta0.clone(); E0 = net.outflat(th0) - Y; Psi0 = net.Psi(th0); K0 = kernel(Psi0); lam0, V0 = eig(K0); e0 = comps(V0, E0)
        H2_0 = net.H2(th0); lam1 = float(lam0[0]); eta = {"gd": 0.5, "eos": 1.8, "adam": 1e-3 * lam1}[opt] / lam1
        am = torch.zeros(net.d, device=DEV); av = torch.zeros(net.d, device=DEV); L.append(f"\n# optimizer: {opt}")
        base = dict(j=0, th=th0.clone(), lam=lam0, e=e0)          # state the rebased floor is taken from (Theorem time0 from a later state)
        Phps = {m: graph_highpass_projector(X, m) for m in (16, 64)}  # high-pass roughness functions R_m(phi) = ||P_m phi||; sigma_R = sqrt(lam_max(P K P))
        L.append("# feature roughness: sigR16 sigR64 at each check (peak feature roughness of the high-pass seminorms); per direction R16(phi) R64(phi) and the budget bounds sigR/R(phi) on ||phi||_K (Cor budget)")
        L += [f"\n# Test 2: k={k} width={width} n={n} eta={eta:.4e} H2(0)={H2_0:.4e} lam1(0)={lam1:.4e} ||E0||={nrm(E0):.4e} rebase every {rebase}",
              "# cols: step t ||E|| ell r=|theta-theta0| | F0(ell) F0(r) | true_rem 0.5*H2bar*ell^2 0.5*H2bar*r^2 0.5*h2*r^2 | floor_path floor_disp floor_dir (ratio to ||E||) | rebased: base_step r_b floor_dir_b (ratio) | H2bar(visited) kappa_S kappa_G"]
        share = (e0 ** 2) / float((e0 ** 2).sum())
        slow = [i for i in range(n) if float(lam0[i]) < 1e-6 * lam1]
        slow = sorted(slow, key=lambda i: -float(share[i]))[:3]
        dirs = {"Ehat0": E0 / nrm(E0)}
        for i in slow: dirs[f"v{i}(lam={float(lam0[i]):.1e},share={float(share[i]):.2f})"] = V0[:, i]
        L.append("# per direction: name step | <phi,E> exact_floor frozen_floor | eff_cos | ||phi||_K^2/||phi||_K0^2 firstorder_pred | <q_phi,E> || growth: |g_phi| |g_phi(t)-g_phi(0)| sum||S(phi)dtheta|| sum D2(phi;theta_check)|dtheta| D2(phi;theta_t) | align=sum(a.d)/sum|a||d| || closure: drive_self=<phi,E> g^T S g  rate=-2 drive_self/|g|^2  rayleigh=ghat^T S ghat")
        th = th0.clone(); ell = 0.0; H2bar = H2_0; dlast = None
        cost = {nm: 0.0 for nm in dirs}; fo = {nm: 0.0 for nm in dirs}; fob = {nm: 0.0 for nm in dirs}; k0 = {}
        g0 = {}; scost = {nm: 0.0 for nm in dirs}; d2cost = {nm: 0.0 for nm in dirs}; D2last = {}
        for nm, phi in dirs.items():
            g0[nm] = Psi0.T @ phi / n; k0[nm] = float(g0[nm].norm() ** 2)
            D2last[nm] = opnorm(lambda w, p=phi: net.S_mv(th0, p, w), net.d, iters=20)
        hv = loss_hvp(net, Y)
        for j in range(nsteps + 1):
            E = net.outflat(th) - Y; Psi = net.Psi(th); g = Psi.T @ E / n
            if j % every == 0:
                H2bar = max(H2bar, net.H2(th)); K = kernel(Psi)
                KE = K @ E; EK2 = float((Psi.T @ E / n).norm() ** 2)
                kap = ip(E, Psi @ hv(th, g)) / EK2 if EK2 > 0 else 0.0  # <E, Psi Hess g>/||E||_K^2  (Hess g = nabla^2 L g)
                kapG = ip(KE, KE) / EK2 if EK2 > 0 else 0.0; kapS = kap - kapG
                Delta = th - th0; r = float(Delta.norm())
                rem = nrm(net.remainder(th0, Psi0, th)); h2 = net.h2_dir(th0, Delta)
                F0l = ridge_F(lam0, e0, ell)[0]; F0r = ridge_F(lam0, e0, r)[0]
                fp = F0l - 0.5 * H2bar * ell ** 2; fd = F0r - 0.5 * H2bar * r ** 2; fdir = F0r - 0.5 * h2 * r ** 2
                nE = nrm(E)
                if rebase and j % rebase == 0 and j > 0:
                    Kb = kernel(Psi); lamb, Vb = eig(Kb); base = dict(j=j, th=th.clone(), lam=lamb, e=comps(Vb, E))
                Db = th - base["th"]; rb = float(Db.norm()); h2b = net.h2_dir(base["th"], Db) if rb > 0 else 0.0
                fdb = ridge_F(base["lam"], base["e"], rb)[0] - 0.5 * h2b * rb ** 2
                sigR = {m: math.sqrt(max(float(torch.linalg.eigvalsh(P @ K @ P).max()), 0.0)) for m, P in Phps.items()}
                L.append(f"roughness {j} | sigR16 {sigR[16]: .4e} sigR64 {sigR[64]: .4e}")
                glazy = {}
                if opt == "gd":
                    # lazy (frozen-kernel) path from the initial state: theta_lazy(j) = theta0 - (1/n) Psi0^T sum_k ehat_k s_k(j) v_k,
                    # s_k(j) = (1 - (1 - eta lam_k)^j)/lam_k (eta j on the null space); kernel sizes and error evaluated EXACTLY there.
                    lamt = lam0.clamp_min(0.0); dec = (1 - eta * lamt) ** j
                    s = torch.where(lamt > 1e-12, (1 - dec) / lamt.clamp_min(1e-300), torch.full_like(lamt, eta * j))
                    Dl = -(Psi0.T @ (V0 @ (e0 * s))) / n
                    thl = th0 + Dl; Psil = net.Psi(thl); El = net.outflat(thl) - Y; Efro = V0 @ (e0 * dec)
                    glazy = {nm: float((Psil.T @ phi / n).norm()) for nm, phi in dirs.items()}
                    L.append(f"lazy {j} | |D_lazy| {f(float(Dl.norm()))} |theta-theta_lazy| {f(float((th - th0 - Dl).norm()))} ||E_frozen|| {f(nrm(Efro))} ||E(theta_lazy)|| {f(nrm(El))} ||E|| {f(nrm(E))}")
                L.append(" ".join([str(j), f"{j * eta:.3f}", f(nE), f(ell), f(r), "|", f(F0l), f(F0r), "|", f(rem), f(0.5 * H2bar * ell ** 2), f(0.5 * H2bar * r ** 2), f(0.5 * h2 * r ** 2), "|",
                                   f(fp), f(fd), f(fdir), f"({fp / nE:.3f} {fd / nE:.3f} {fdir / nE:.3f})", "|", str(base["j"]), f(rb), f(fdb), f"({fdb / nE:.3f})", "|", f(H2bar), f(kapS), f(kapG)]))
                for nm, phi in dirs.items():
                    gphi = Psi.T @ phi / n; kn = float(gphi.norm() ** 2)
                    q = Psi @ net.S_mv(th, phi, gphi)
                    pe0 = ip(phi, E0); pe = ip(phi, E)
                    effcos = (pe0 - pe) / cost[nm] if cost[nm] > 0 else 0.0
                    if j > 0: D2last[nm] = opnorm(lambda w, p=phi: net.S_mv(th, p, w), net.d, iters=20)
                    align = fo[nm] / fob[nm] if fob[nm] > 0 else 0.0
                    a_phi = net.S_mv(th, phi, gphi); drive = float(a_phi @ g)            # <q_phi,E> = a_phi . g_E
                    ray = float(a_phi @ gphi) / kn if kn > 0 else 0.0                    # Rayleigh quotient ghat^T S(phi) ghat
                    selfdrive = pe * float(a_phi @ gphi)                                  # self term: g_E ~ <phi,E> g_phi
                    # tower of directional derivatives of f_phi = <phi,N>/n along the last step direction (Prop tower), and first-layer weight scale
                    u = (dlast / dlast.norm()) if dlast is not None else gphi / gphi.norm()
                    fphi = lambda t2, p=phi: (p @ net.outflat(t2)) / n
                    d1f = lambda t2: jvp(fphi, (t2,), (u,))[1]; d2f = lambda t2: jvp(d1f, (t2,), (u,))[1]
                    d3f = lambda t2: jvp(d2f, (t2,), (u,))[1]; d4f = lambda t2: jvp(d3f, (t2,), (u,))[1]
                    tower = [abs(float(fn(th))) for fn in (d1f, d2f, d3f, d4f)]
                    nw1 = net.shapes[0].numel(); w1 = th[:nw1]
                    L.append(" ".join([nm, str(j), "|", f(pe), f(pe0 - cost[nm]), f(pe0 - ell * math.sqrt(k0[nm])), "|", f"{effcos: .3f}", "|",
                                       f"{kn / k0[nm]: .4e}", f"{1 + fo[nm] / k0[nm]: .4e}", f(drive), "||",
                                       f(math.sqrt(kn)), f(float((gphi - g0[nm]).norm())), f(scost[nm]), f(d2cost[nm]), f(D2last[nm]), "|", f"{align: .3f}", "||",
                                       f(selfdrive), f(-2 * selfdrive / kn if kn > 0 else 0.0), f(ray), "|| tower", *[f(x) for x in tower],
                                       "| w1", f(float(w1.norm()) / math.sqrt(nw1)), f(float(w1.abs().max())),
                                       "|| rough", *[f(nrm(Phps[m] @ phi)) for m in (16, 64)], "| budget", *[f(sigR[m] / max(nrm(Phps[m] @ phi), 1e-300)) for m in (16, 64)],
                                       "|| lazy|g|", f(glazy.get(nm, float('nan')))]))
                print("\n".join(L[-len(dirs) - 1:]), flush=True)
            if j == nsteps: break
            if opt == "adam":
                am = 0.9 * am + 0.1 * g; av = 0.999 * av + 0.001 * g ** 2
                d = -eta * (am / (1 - 0.9 ** (j + 1))) / (torch.sqrt(av / (1 - 0.999 ** (j + 1))) + 1e-8)
            else:
                d = -eta * g
            dn = float(d.norm())
            for nm, phi in dirs.items():
                gphi = Psi.T @ phi / n
                cost[nm] += float(gphi.norm()) * dn
                a = net.S_mv(th, phi, gphi)                                   # a_phi = S(phi) g_phi, the kernel-growth direction
                fo[nm] += 2.0 * float(a @ d); fob[nm] += 2.0 * float(a.norm()) * dn   # first-order change of ||phi||_K^2 and its bound
                scost[nm] += float(net.S_mv(th, phi, d).norm())              # exact directional cost: sum ||S(phi) dtheta||  (Lemma drift(a))
                d2cost[nm] += D2last[nm] * dn                                # local a priori bound: D2(phi; last check state) * path length
            th = th + d; ell += dn; dlast = d
    open(os.path.join(out, "test2.txt"), "w").write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- test 3
def test3(out, width=64, n=512, k=4, nsteps=3000, batch=32, seeds=3, check=300, nres=256):
    global NPTS; NPTS = n
    net, X, Y = sine_setup(width, n, k)
    th0 = net.theta0.clone(); E0 = net.outflat(th0) - Y; Psi0 = net.Psi(th0); K0 = kernel(Psi0); lam0, V0 = eig(K0); e0 = comps(V0, E0)
    L0 = 0.5 * nrm(E0) ** 2; H2_0 = net.H2(th0); hv = loss_hvp(net, Y)
    M0 = opnorm(lambda w: hv(th0, w), net.d)
    share = (e0 ** 2) / float((e0 ** 2).sum())
    slow = sorted([i for i in range(n) if float(lam0[i]) < 1e-6 * float(lam0[0])], key=lambda i: -float(share[i]))[:2]
    dirs = {f"v{i}(share={float(share[i]):.2f})": V0[:, i] for i in slow}
    L = [f"# Test 3: k={k} width={width} n={n} batch={batch} seeds={seeds} steps={nsteps} L0={L0:.4e} ||E0||={nrm(E0):.4e} M(theta0)={M0:.3e} lam1(0)={float(lam0[0]):.3e} H2(0)={H2_0:.3e}"]
    opts = {"sgd": dict(eta=0.5 / float(lam0[0]), beta=0.0),
            "heavyball": dict(eta=0.05 / float(lam0[0]), beta=0.9),
            "adam": dict(eta=1e-3, beta=0.9, beta2=0.999, eps=1e-8)}
    for name, cfg in opts.items():
        eta, beta = cfg["eta"], cfg["beta"]
        finalE, pathlen, driftlen, driftmet, sb2s, margins, adam_stats, numet = [], [], [], [], [], [], [], []
        Mmax = 0.0; MPmax = 0.0; cmin_old = float('inf'); cmin_new = float('inf'); cmin_new1 = float('inf'); dcost = {nm: [0.0, 0.0] for nm in dirs}
        dfinal = {nm: [] for nm in dirs}
        for sd in range(seeds):
            gen = torch.Generator(device='cpu').manual_seed(100 + sd)
            th = th0.clone(); d_prev = torch.zeros(net.d, device=DEV); m = torch.zeros(net.d, device=DEV); v = torch.zeros(net.d, device=DEV)
            Pbar_prev = None; ell = 0.0; ellbar = 0.0; ellmet = 0.0
            for j in range(nsteps):
                E = net.outflat(th) - Y; Psi = net.Psi(th); gfull = Psi.T @ E / n
                idx = torch.randperm(n, generator=gen)[:batch].to(DEV); W = torch.zeros(n, device=DEV); W[idx] = n / batch
                ghat = Psi.T @ (W * E) / n
                if j % check == 0:
                    a = Psi * E[:, None]
                    varj = float(((a - gfull[None, :]) ** 2).sum(1).mean())
                    sb2 = (n - batch) / (batch * (n - 1)) * varj; sb2s.append(sb2)
                    Mj = opnorm(lambda w: hv(th, w), net.d); Mmax = max(Mmax, Mj)
                    if name == "sgd":
                        Ls = []
                        for _ in range(nres):
                            i2 = torch.randperm(n, generator=gen)[:batch].to(DEV); W2 = torch.zeros(n, device=DEV); W2[i2] = n / batch
                            Ls.append(0.5 * nrm(net.outflat(th - eta * (Psi.T @ (W2 * E) / n)) - Y) ** 2)
                        Lt = torch.tensor(Ls); lhs = float(Lt.mean()); se = float(Lt.std() / math.sqrt(nres))
                        Mseg = max(Mj, opnorm(lambda w: hv(th - eta * gfull, w), net.d))
                        rhs = 0.5 * nrm(E) ** 2 - eta * (1 - eta * Mseg / 2) * float(gfull @ gfull) + 0.5 * eta ** 2 * Mseg * sb2
                        margins.append((j, lhs - rhs, se, 0.5 * nrm(E) ** 2 - lhs))
                if name == "sgd":
                    u = gfull; d = -eta * ghat; etap = eta
                elif name == "heavyball":
                    u = gfull - (beta / eta) * d_prev; d = -eta * ghat + beta * d_prev; etap = eta
                else:
                    b2 = cfg["beta2"]
                    if j == 0:
                        vhat_prev = ghat ** 2                                   # P_{-1} = P_0 (Lemma adam), bias-corrected
                    else:
                        vhat_prev = v / (1 - b2 ** j)
                    Pbar = 1.0 / (torch.sqrt(vhat_prev) + cfg["eps"])           # lagged, bias-corrected preconditioner
                    etap = eta * (1 - beta)
                    u = Pbar * gfull - (beta / etap) * Pbar * (d_prev / (Pbar_prev if Pbar_prev is not None else Pbar))   # u_j = Pbar g - (beta/eta') R_j d_{j-1}
                    m = beta * m + (1 - beta) * ghat; v = b2 * v + (1 - b2) * ghat ** 2
                    P = 1.0 / (torch.sqrt(v / (1 - b2 ** (j + 1))) + cfg["eps"])
                    d = -eta / (1 - beta ** (j + 1)) * P * m
                    if j % check == 0:
                        sP = torch.sqrt(Pbar)
                        MP = opnorm(lambda w: sP * hv(th, sP * w), net.d); MPmax = max(MPmax, MP)
                        lmP = float(Pbar.max()); rho = float((Pbar / Pbar_prev).max()) if Pbar_prev is not None else 1.0
                        s_j = float((Pbar * ghat.abs()).max())
                        c_old = 1 - 0.5 * beta * (1 + rho) - 0.5 * etap * Mj * lmP
                        c_new = 1 - 0.5 * beta * (1 + rho) - 0.5 * etap * MP
                        cmin_old = min(cmin_old, c_old); cmin_new = min(cmin_new, c_new)
                        if j > 0: cmin_new1 = min(cmin_new1, c_new)
                        xi2 = []                                                # E_j |xi|^2_{Pbar}, measured over resampled batches
                        for _ in range(64):
                            i2 = torch.randperm(n, generator=gen)[:batch].to(DEV); W2 = torch.zeros(n, device=DEV); W2[i2] = n / batch
                            xi = Psi.T @ (W2 * E) / n - gfull; xi2.append(float((xi * Pbar * xi).sum()))
                        numet.append(sum(xi2) / len(xi2))
                        adam_stats.append((sd, j, lmP, rho, s_j, Mj, MP, c_old, c_new, numet[-1]))
                    ellmet += etap * float((u / torch.sqrt(Pbar)).norm())      # |u|_{Pbar^{-1}}
                    for nm, phi in dirs.items():
                        gphi = Psi.T @ phi / n
                        dcost[nm][1] += etap * float((torch.sqrt(Pbar) * gphi).norm()) * float((u / torch.sqrt(Pbar)).norm())
                    Pbar_prev = Pbar
                for nm, phi in dirs.items():
                    dcost[nm][0] += etap * float((Psi.T @ phi / n).norm()) * float(u.norm())
                ell += float(d.norm()); ellbar += etap * float(u.norm())
                d_prev = d; th = th + d
            E = net.outflat(th) - Y
            for nm, phi in dirs.items(): dfinal[nm].append(ip(phi, E))
            finalE.append(nrm(E)); pathlen.append(ell); driftlen.append(ellbar); driftmet.append(ellmet)
            print(f"{name} seed {sd}: final ||E||={nrm(E):.4e} path={ell:.3e} drift={ellbar:.3e}", flush=True)
        T = nsteps * eta; sb2 = sum(sb2s) / len(sb2s)
        if name == "sgd":
            c = 1 - eta * Mmax / 2; nu = 0.5 * eta * Mmax * sb2; etap = eta
        elif name == "heavyball":
            c = 1 - beta - eta * Mmax / 2; nu = (beta + eta * Mmax / 2) * sb2; etap = eta
        else:
            etap = eta * (1 - beta); T = nsteps * etap
            lmaxP = max(x[2] for x in adam_stats)
            c = cmin_new / lmaxP if cmin_new > 0 else float('nan')
            nu = (1 - cmin_new) * lmaxP * sb2 if cmin_new > 0 else float('nan')
        ok = (c == c) and c > 0
        elltilde = math.sqrt(T * (L0 + T * nu) / c) if ok else float('inf')
        F0t = ridge_F(lam0, e0, elltilde)[0] if ok else float('nan')
        L.append(f"\n## {name}: eta={eta:.3e} (eta'={etap:.3e}) T={T:.3f} M_max(visited)={Mmax:.3e} sigma_b^2={sb2:.3e} c={c:.4f} nu_bar={nu:.3e}")
        L.append(f"   final ||E||: mean={sum(finalE)/len(finalE):.4e} min={min(finalE):.3e} max={max(finalE):.3e};  mean final L={0.5*sum(x**2 for x in finalE)/len(finalE):.4e};  bound E L_N <= L0+T nu = {L0 + T * nu:.4e}")
        L.append(f"   path length sum|dtheta| mean={sum(pathlen)/len(pathlen):.4e};  drift length sum eta'|u| mean={sum(driftlen)/len(driftlen):.4e};  a priori ell_tilde (Euclidean)={elltilde:.4e};  F0(ell_tilde)={F0t:.4e}")
        if name == "sgd":
            L.append(f"   expected-descent check at {len(margins)} states ({nres} resampled batches each): margin = E_j L_(j+1) - rhs (negative = holds), se = standard error of E_j L_(j+1), drop = L_j - E_j L_(j+1)")
            for (j, mg, se, drop) in margins:
                L.append(f"     j={j}: margin={mg: .3e} se={se:.1e} drop={drop:.3e} {'VIOLATED' if mg > 2 * se else 'ok'}")
        for nm in dirs:
            L.append(f"   direction {nm}: <phi,E0>={ip(dirs[nm], E0): .4e}  mean final <phi,E_N>={sum(dfinal[nm])/len(dfinal[nm]): .4e};  Euclidean cost sum eta'||phi||_K|u| = {dcost[nm][0]/seeds:.4e}" + (f";  metric cost sum eta'||phi||_(K^P)|u|_(P^-1) = {dcost[nm][1]/seeds:.4e}" if name == "adam" else ""))
        if name == "adam":
            cA = cmin_new1; nu_met = (1 - cA) * sum(numet) / len(numet) if cA < 1 else 0.0
            T1 = (nsteps - 1) * etap
            ellmet_apriori = math.sqrt(T1 * (L0 + T1 * nu_met) / cA) if cA > 0 else float('inf')
            L.append(f"   metric drift length sum eta'|u|_(Pbar^-1) mean={sum(driftmet)/len(driftmet):.4e};  a priori bound sqrt(T(V0+T nu)/c_A) from j=1 with c_A=min_(j>=1) c_new={cA:.4f}, measured nu_bar=(1-c_A) E|xi|^2_Pbar={nu_met:.4e}: {ellmet_apriori:.4e}  (V0+T nu = {L0 + T1 * nu_met:.4e} vs mean final L)")
            L.append(f"   c_j: min with M*lam_max(Pbar) = {cmin_old:.3f};  min with M_P = {cmin_new:.3f} (j=0 included), {cmin_new1:.3f} (j>=1);  max M_P={MPmax:.3e} vs max M*lam_max(P)={max(x[5]*x[2] for x in adam_stats):.3e};  E|xi|^2_Pbar mean={sum(numet)/len(numet):.3e} vs lam_max(Pbar) sigma_b^2={lmaxP*sb2:.3e}")
            for (sd, j, lmP, rho, s_j, Mj, MP, c_old, c_new, xm) in adam_stats:
                L.append(f"   seed {sd} j={j}: lam_max(Pbar)={lmP:.3e} rho={rho:.3f} spike s_j={s_j:.3f} M={Mj:.3e} M_P={MP:.3e} c_old={c_old: .3f} c_new={c_new: .3f} E|xi|^2_P={xm:.3e}")
        print("\n".join(L), flush=True)
    open(os.path.join(out, "test3.txt"), "w").write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- test 5
def test5(out, width=64, n=512, ks=(2, 4), orders=(1, 2, 3, 4), nsteps=10000, every=250):
    """Taylorized training: gradient descent on the order-M Taylor expansion of the network about theta0,
    f_M(D) = sum_{m<=M} (1/m!) nabla^m N(theta0)[D,...,D] (exact nested forward-mode derivatives), run in lockstep with the
    real network at the same step size. Logged at each check: model error, true network error at theta0 + D_M, |D_M|,
    |D_M - Delta_real|, and the kernel size of each rough direction at theta0 + D_M (exact) and under the model."""
    global NPTS; NPTS = n
    L = []
    for k in ks:
        net, X, Y = sine_setup(width, n, k)
        th0 = net.theta0.clone(); E0 = net.outflat(th0) - Y; Psi0 = net.Psi(th0); K0 = kernel(Psi0); lam0, V0 = eig(K0); e0 = comps(V0, E0)
        lam1 = float(lam0[0]); eta = 0.5 / lam1
        share = (e0 ** 2) / float((e0 ** 2).sum())
        slow = sorted([i for i in range(n) if float(lam0[i]) < 1e-6 * lam1], key=lambda i: -float(share[i]))[:2]
        dirs = {f"v{i}(share={float(share[i]):.2f})": V0[:, i] for i in slow}
        N0 = net.outflat(th0)
        def taylor(M):
            def fM(D):
                val = N0; cur = net.outflat; fact = 1.0
                for m in range(1, M + 1):
                    cur = (lambda c: (lambda t: jvp(c, (t,), (D,))[1]))(cur); fact *= m
                    val = val + cur(th0) / fact
                return val
            return fM
        L += [f"\n# Test 5: Taylorized training, k={k} width={width} n={n} eta={eta:.4e} ||E0||={nrm(E0):.4e} orders={orders}",
              "# cols: M step | ||E_real|| ||f_M-Y|| ||N(theta0+D_M)-Y|| | |Delta_real| |D_M| |D_M-Delta_real| | per direction: |g_v| real, exact at theta0+D_M, under the model"]
        # real run first, storing checkpoints
        th = th0.clone(); real = {}
        for j in range(nsteps + 1):
            if j % every == 0: real[j] = th.clone()
            if j == nsteps: break
            E = net.outflat(th) - Y; th = th - eta * (net.Psi(th).T @ E / n)
        for M in orders:
            fM = taylor(M)
            lossM = lambda D: 0.5 * ((fM(D) - Y) @ (fM(D) - Y)) / n
            gradM = grad(lossM)
            D = torch.zeros(net.d, device=DEV); diverged = False
            for j in range(nsteps + 1):
                if j % every == 0:
                    thr = real[j]; Er = net.outflat(thr) - Y; Psir = net.Psi(thr)
                    thm = th0 + D; Em = net.outflat(thm) - Y; Psim = net.Psi(thm); fm = fM(D)
                    row = [str(M), str(j), "|", f(nrm(Er)), f(nrm(fm - Y)), f(nrm(Em)), "|", f(float((thr - th0).norm())), f(float(D.norm())), f(float((D - (thr - th0)).norm())), "|"]
                    for nm, phi in dirs.items():
                        gmodel = grad(lambda Dd, p=phi: (p @ fM(Dd)) / n)(D)
                        row += [nm, f(float((Psir.T @ phi / n).norm())), f(float((Psim.T @ phi / n).norm())), f(float(gmodel.norm()))]
                    L.append(" ".join(row)); print(L[-1], flush=True)
                if j == nsteps or diverged: break
                gD = gradM(D)
                if not torch.isfinite(gD).all() or float(gD.norm()) > 1e6:
                    L.append(f"{M} {j} | model diverged"); print(L[-1], flush=True); diverged = True; continue
                D = D - eta * gD
    open(os.path.join(out, "test5.txt"), "w").write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- test 6
class MLP1(torch.nn.Module):
    """One hidden tanh layer: N(x) = a . tanh(w x + b) + c. Named l1/l3 so Net's bookkeeping (shapes[0] = l1.weight) is unchanged."""
    def __init__(self, din, width, dout):
        super().__init__()
        self.l1 = torch.nn.Linear(din, width); self.l3 = torch.nn.Linear(width, dout)

    def forward(self, x):
        return self.l3(torch.tanh(self.l1(x)))


def test6(out, width=256, n=512, ks=(2, 4, 8), nsteps=20000, every=500, layers=1, gamma_rel=1e-3, frozen_mode="C"):
    """Frozen-residual prediction of the plateau. Real GD is run and stored. Two predicted parameter paths are integrated with the
    SAME update rule but with the residual E replaced by a prescribed function of time, which makes the units independent of each
    other (one hidden layer) and uses no expansion in the displacement:
      A: E_A(t) = E_lazy(t) = sum_k (1 - eta lam_k)^t ehat_k v_k   (lazy/frozen-kernel residual, from K_0 and E_0 only)
      C: E_C   = E_gamma, the slow part of E_0 at gamma = gamma_rel*lam_1 (residual frozen at the rough part of the initial error)
    At each check: for real, A, C: ||N(theta)-Y||, Rayleigh quotient root rho^(1/2) of K(theta) at the frozen direction Ehat_C,
    the root of the Rayleigh quotient at the dominant slow eigenvector v, <v,E_real>, |theta_pred - theta_real|, max|w1|.
    Crossing step = first check at which the root of the Rayleigh quotient at v exceeds 0.3, real vs predicted."""
    global NPTS; NPTS = n
    L = []
    for k in ks:
        X = torch.linspace(-1, 1, n).unsqueeze(1); Y = torch.sin(k * math.pi * X).reshape(-1).to(DEV)
        torch.manual_seed(0)
        net = Net(MLP1(1, width, 1) if layers == 1 else MLP(1, width, 1), X); X = X.to(DEV)
        th0 = net.theta0.clone(); E0 = net.outflat(th0) - Y; Psi0 = net.Psi(th0); K0 = kernel(Psi0); lam0, V0 = eig(K0); e0 = comps(V0, E0)
        lam1 = float(lam0[0]); eta = 0.5 / lam1
        share = (e0 ** 2) / float((e0 ** 2).sum())
        slow = sorted([i for i in range(n) if float(lam0[i]) < 1e-6 * lam1], key=lambda i: -float(share[i]))[:1]
        v = V0[:, slow[0]]
        gam = gamma_rel * lam1; EC = V0 @ (e0 * gam / (lam0.clamp_min(0) + gam)); EChat = EC / nrm(EC)
        # variant D (replaces C when frozen_mode == "D"): rough part of the residual frozen at E_gamma, smooth part self-consistent,
        # E_D(theta) = E_gamma + P_smooth (N(theta) - Y) with P_smooth = I - V_slow V_slow^T/n over the eigenvectors with lam < gamma
        rough = lam0.clamp_min(0) < gam; Vs = V0[:, rough]
        def E_D(th_):
            Et = net.outflat(th_) - Y
            return EC + (Et - Vs @ (Vs.T @ Et) / n)
        nw1 = net.shapes[0].numel()
        L += [f"\n# Test 6: frozen-residual prediction, layers={layers} width={width} n={n} k={k} eta={eta:.4e} ||E0||={nrm(E0):.4e} ||E_gamma||={nrm(EC):.4e} (gamma={gam:.2e}) rough dir v{slow[0]} share={float(share[slow[0]]):.2f} |g_v(0)|={float((Psi0.T @ v / n).norm()):.3e}",
              "# cols: step | ||E|| real A C | rho^(1/2) at Ehat_C: real A C | |g_v|: real A C | <v,E_real> | |th_A-th_real| |th_C-th_real| |th_real-th0| | max|w1| real A C"]
        th = th0.clone(); thA = th0.clone(); thC = th0.clone(); lamt = lam0.clamp_min(0.0)
        cross = {"real": None, "A": None, "C": None}
        for j in range(nsteps + 1):
            if j % every == 0:
                row = [str(j), "|"]; vals = {}
                for nm, t_ in (("real", th), ("A", thA), ("C", thC)):
                    Et = net.outflat(t_) - Y; Pt = net.Psi(t_)
                    vals[nm] = (nrm(Et), float((Pt.T @ EChat / n).norm()), float((Pt.T @ v / n).norm()), float(t_[:nw1].abs().max()))
                    if cross[nm] is None and vals[nm][2] > 0.3: cross[nm] = j
                row += [f(vals[x][0]) for x in ("real", "A", "C")] + ["|"] + [f(vals[x][1]) for x in ("real", "A", "C")] + ["|"] + [f(vals[x][2]) for x in ("real", "A", "C")]
                row += ["|", f(ip(v, net.outflat(th) - Y)), "|", f(float((thA - th).norm())), f(float((thC - th).norm())), f(float((th - th0).norm())), "|"] + [f(vals[x][3]) for x in ("real", "A", "C")]
                L.append(" ".join(row)); print(L[-1], flush=True)
            if j == nsteps: break
            E = net.outflat(th) - Y; th = th - eta * (net.Psi(th).T @ E / n)
            EA = V0 @ (e0 * (1 - eta * lamt) ** j)
            thA = thA - eta * (net.Psi(thA).T @ EA / n)
            thC = thC - eta * (net.Psi(thC).T @ (E_D(thC) if frozen_mode == "D" else EC) / n)
        L.append(f"# k={k}: crossing step (|g_v| > 0.3): real={cross['real']} A(lazy residual)={cross['A']} C({'rough frozen, smooth self-consistent' if frozen_mode == 'D' else 'frozen rough residual'})={cross['C']}")
        print(L[-1], flush=True)
    open(os.path.join(out, "test6.txt"), "w").write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- test 7
def test7(out, width=64, n=512, ks=(4,), nsteps=20000, every=100, opt="gd", seeds=16, layers=2, thresh=0.3):
    """Seed study of the crossing time. For each initialization: the initial Rayleigh-quotient roots |g_Y(0)| (at the normalized
    target direction) and |g_v(0)| (dominant slow eigenvector of K_0), the largest initial input weight, the first step at which each
    root exceeds `thresh` (crossing time), the path length and displacement at that step, the error curve every `every` steps.
    Lean loop: one Jacobian per step, no Hessians. opt = gd (eta = 0.5/lam1) or adam (eta = 1e-3, bias-corrected)."""
    global NPTS; NPTS = n
    X = torch.linspace(-1, 1, n).unsqueeze(1)
    for k in ks:
        Y = torch.sin(k * math.pi * X).reshape(-1).to(DEV); Yhat = Y / nrm(Y)
        rows = [f"# Test 7 seed study: opt={opt} k={k} width={width} layers={layers} n={n} nsteps={nsteps} thresh={thresh}",
                "# seed | eta | |g_Y(0)| |g_v(0)| lam_v share_v w1max0 | T_Y T_v (steps; -1 = no crossing) | ell@T_Y r@T_Y | ||E(0)|| ||E(T_Y)|| ||E(end)|| | w1max@T_Y w1max(end)"]
        curves = ["# step then one column per seed: ||E||"]
        allcurves = []; envcurves = []
        mixrows = ["# mode mixing in the K_0 eigenbasis. seed step | ||E|| | offdiag/total Frobenius, ||M_SS||_F ||M_SF||_F ||M_FF||_F | rho_v(t) max_S diag  top3 eig of M_SS | mix_v = offdiag share of column v | sum_S diag, ||off_SS||_F ||off_SF||_F"]
        for sd in range(seeds):
            torch.manual_seed(sd)
            net = Net(MLP1(1, width, 1) if layers == 1 else MLP(1, width, 1), X)
            th = net.theta0.clone(); E0 = net.outflat(th) - Y; Psi0 = net.Psi(th); K0 = kernel(Psi0); lam0, V0 = eig(K0); e0 = comps(V0, E0)
            lam1 = float(lam0[0]); eta = (0.5 / lam1) if opt == "gd" else 1e-3
            share = (e0 ** 2) / float((e0 ** 2).sum())
            slow = sorted([i for i in range(n) if float(lam0[i]) < 1e-6 * lam1], key=lambda i: -float(share[i]))
            iv = slow[0] if slow else int(torch.argmin(lam0)); v = V0[:, iv]
            nw1 = net.shapes[0].numel()
            gY0 = float((Psi0.T @ Yhat / n).norm()); gv0 = float((Psi0.T @ v / n).norm()); w1max0 = float(th[:nw1].abs().max())
            am = torch.zeros(net.d, device=DEV); av = torch.zeros(net.d, device=DEV)
            ell = 0.0; TY = -1; Tv = -1; ellY = rY = EY = wY = float('nan'); curve = []
            # mode mixing: the kernel in the fixed eigenbasis of K_0, M(t) = V0^T K(t) V0 / n (= diag(lam0) at t = 0),
            # at a few checkpoints; slow block S = {lam < 1e-3 lam1}
            cps = sorted(set(int(nsteps * fr) for fr in (0.1, 0.25, 0.5, 0.75, 1.0)))
            S = lam0.clamp_min(0) < 1e-3 * lam1; Fm = ~S
            for j in range(nsteps + 1):
                E = net.outflat(th) - Y; Psi = net.Psi(th); g = Psi.T @ E / n
                if j % every == 0: curve.append(nrm(E))
                if j in cps:
                    K = kernel(Psi); M = V0.T @ K @ V0 / n; dg = torch.diagonal(M); off = M - torch.diag(dg)
                    mu = torch.linalg.eigvalsh(M[S][:, S]).flip(0)[:3] if int(S.sum()) >= 3 else torch.zeros(3, device=DEV)
                    colv = M[:, iv]; mixv = float((colv.norm() ** 2 - colv[iv] ** 2).clamp_min(0).sqrt() / colv.norm().clamp_min(1e-300))
                    mixrows.append(" ".join([str(sd), str(j), "|", f(nrm(E)), "|", f(float(off.norm() / M.norm())), f(float(M[S][:, S].norm())), f(float(M[S][:, Fm].norm())), f(float(M[Fm][:, Fm].norm())), "|",
                                             f(float(dg[iv])), f(float(dg[S].max())), f(float(mu[0])), f(float(mu[1])), f(float(mu[2])), "|", f"{mixv:.3f}", "|", f(float(dg[S].sum())), f(float(off[S][:, S].norm())), f(float(off[S][:, Fm].norm()))]))
                    if j == nsteps and opt == "gd":
                        # envelope curves from E_0 under: (i) K_0, (ii) diag of trained kernel in the K_0 basis (no mixing), (iii) full trained kernel
                        steps_grid = torch.arange(0, nsteps + 1, every, device=DEV, dtype=torch.float64)
                        def lazy_curve(lams, coeffs):
                            dec = (1 - eta * lams.clamp(min=0.0, max=2.0 / eta))[None, :] ** steps_grid[:, None]
                            return torch.sqrt(((coeffs[None, :] * dec) ** 2).sum(1))
                        ci = lazy_curve(lam0.clamp_min(0), e0); cii = lazy_curve(dg.clamp_min(0), e0)
                        muM, W = torch.linalg.eigh((M + M.T) / 2); ciii = lazy_curve(muM.clamp_min(0), W.T @ e0)
                        envcurves.append((ci, cii, ciii))
                gY = float((Psi.T @ Yhat / n).norm()); gv = float((Psi.T @ v / n).norm())
                if TY < 0 and gY > thresh: TY = j; ellY = ell; rY = float((th - net.theta0).norm()); EY = nrm(E); wY = float(th[:nw1].abs().max())
                if Tv < 0 and gv > thresh: Tv = j
                if j == nsteps: break
                if opt == "adam":
                    am = 0.9 * am + 0.1 * g; av = 0.999 * av + 0.001 * g ** 2
                    d = -eta * (am / (1 - 0.9 ** (j + 1))) / (torch.sqrt(av / (1 - 0.999 ** (j + 1))) + 1e-8)
                else:
                    d = -eta * g
                th = th + d; ell += float(d.norm())
            Eend = nrm(net.outflat(th) - Y)
            rows.append(" ".join([str(sd), "|", f"{eta:.3e}", "|", f(gY0), f(gv0), f"{float(lam0[iv]):.2e}", f"{float(share[iv]):.2f}", f(w1max0), "|", str(TY), str(Tv), "|", f(ellY), f(rY), "|", f(nrm(E0)), f(EY), f(Eend), "|", f(wY), f(float(th[:nw1].abs().max()))]))
            print(rows[-1], flush=True); allcurves.append(curve)
        for i in range(len(allcurves[0])):
            curves.append(" ".join([str(i * every)] + [f"{c[i]:.4e}" for c in allcurves]))
        open(os.path.join(out, f"test7_{opt}_k{k}.txt"), "w").write("\n".join(rows) + "\n")
        open(os.path.join(out, f"test7_{opt}_k{k}_curves.txt"), "w").write("\n".join(curves) + "\n")
        open(os.path.join(out, f"test7_{opt}_k{k}_mix.txt"), "w").write("\n".join(mixrows) + "\n")
        if envcurves:
            env = ["# envelope: step | mean measured ||E|| | mean lazy curve with K_0 | with diag of trained kernel (no mixing) | with full trained kernel"]
            for i in range(len(allcurves[0])):
                m = sum(c[i] for c in allcurves) / len(allcurves)
                env.append(" ".join([str(i * every), "|", f"{m:.4e}", "|"] + [f"{float(sum(ec[q][i] for ec in envcurves) / len(envcurves)):.4e}" for q in range(3)]))
            open(os.path.join(out, f"test7_{opt}_k{k}_env.txt"), "w").write("\n".join(env) + "\n")


# ----------------------------------------------------------------------------- test 8
def test8(out, width=64, n=512, ks=(4,), nsteps=20000, every=50, opt="gd", seeds=8, seed0=0, layers=2):
    """Statistics of rho_E(t) and of the trained kernel's rough block, over initializations.
    Per seed, every `every` steps: ||E||, rho_E = <Ehat, K Ehat> = |g|^2/||E||^2, and (GD) the frozen-kernel prediction of rho_E.
    At five checkpoints: the kernel in the K_0 eigenbasis restricted to the slow block S (lam < 1e-3 lam1): top-5 eigenvalues,
    low-rank Frobenius fractions (r = 1, 2, 3, 5), overlaps (cos^2) of the top-3 eigenvectors with the S-projections of the target
    Yhat and of its quadrature companion cos(k pi x) (spiked-kernel test), off-diagonal share of M_SS, and the full-trained-kernel
    lazy curve value at that step (GD)."""
    global NPTS; NPTS = n
    X = torch.linspace(-1, 1, n).unsqueeze(1)
    for k in ks:
        Y = torch.sin(k * math.pi * X).reshape(-1).to(DEV); Yhat = Y / nrm(Y); Cq = torch.cos(k * math.pi * X).reshape(-1).to(DEV); Cq = Cq / nrm(Cq)
        rows = [f"# Test 8: opt={opt} k={k} width={width} layers={layers} n={n} nsteps={nsteps} seeds={seed0}..{seed0+seeds-1}",
                "# seed step | ||E|| rho_E rho_E_lazy | top5 eig of M_SS | lowrank frac r=1,2,3,5 | cos^2(u1,Y_S) cos^2(u2,Y_S) cos^2(u3,Y_S) | cos^2(u1,C_S) cos^2(u2,C_S) cos^2(u3,C_S) | offshare_SS | lazy_fullK ||E||"]
        traces = ["# step then per seed: ||E|| rho_E (pairs)"]
        alltr = []
        for sd in range(seed0, seed0 + seeds):
            torch.manual_seed(sd)
            net = Net(MLP1(1, width, 1) if layers == 1 else MLP(1, width, 1), X)
            th = net.theta0.clone(); E0 = net.outflat(th) - Y; Psi0 = net.Psi(th); K0 = kernel(Psi0); lam0, V0 = eig(K0); e0 = comps(V0, E0)
            lam1 = float(lam0[0]); eta = (0.5 / lam1) if opt == "gd" else 1e-3
            S = lam0.clamp_min(0) < 1e-3 * lam1
            yS = comps(V0, Yhat)[S]; yS = yS / yS.norm(); cS = comps(V0, Cq)[S]; cS = cS / cS.norm()
            cps = sorted(set(int(nsteps * fr) for fr in (0.1, 0.25, 0.5, 0.75, 1.0)))
            am = torch.zeros(net.d, device=DEV); av = torch.zeros(net.d, device=DEV); tr = []
            for j in range(nsteps + 1):
                E = net.outflat(th) - Y; Psi = net.Psi(th); g = Psi.T @ E / n; nE2 = float(E @ E) / n
                rhoE = float(g @ g) / nE2 if nE2 > 0 else 0.0
                if j % every == 0:
                    if opt == "gd":
                        w = e0 ** 2 * (1 - eta * lam0.clamp_min(0)) ** (2 * j); rlazy = float((lam0.clamp_min(0) * w).sum() / w.sum().clamp_min(1e-300))
                    else:
                        rlazy = float('nan')
                    tr.append((math.sqrt(nE2), rhoE, rlazy))
                if j in cps:
                    K = kernel(Psi); M = V0.T @ K @ V0 / n; MS = (M[S][:, S] + M[S][:, S].T) / 2
                    mu, U = torch.linalg.eigh(MS); mu = mu.flip(0); U = U.flip(1)
                    fro2 = float((MS ** 2).sum()); lr = [float((mu[:r] ** 2).sum()) / fro2 for r in (1, 2, 3, 5)]
                    oy = [float((U[:, i] @ yS) ** 2) for i in range(3)]; oc = [float((U[:, i] @ cS) ** 2) for i in range(3)]
                    offshare = float((MS - torch.diag(torch.diagonal(MS))).norm() / MS.norm())
                    if opt == "gd":
                        muM, W = torch.linalg.eigh((M + M.T) / 2); cW = W.T @ e0
                        lazyfull = float(torch.sqrt(((cW * (1 - eta * muM.clamp(min=0.0, max=2.0 / eta)) ** j) ** 2).sum()))
                    else:
                        lazyfull = float('nan')
                    rows.append(" ".join([str(sd), str(j), "|", f(math.sqrt(nE2)), f(rhoE), f(tr[-1][2]), "|", *[f(float(x)) for x in mu[:5]], "|", *[f"{x:.3f}" for x in lr], "|", *[f"{x:.3f}" for x in oy], "|", *[f"{x:.3f}" for x in oc], "|", f"{offshare:.3f}", "|", f(lazyfull)]))
                    print(rows[-1], flush=True)
                if j == nsteps: break
                if opt == "adam":
                    am = 0.9 * am + 0.1 * g; av = 0.999 * av + 0.001 * g ** 2
                    d = -eta * (am / (1 - 0.9 ** (j + 1))) / (torch.sqrt(av / (1 - 0.999 ** (j + 1))) + 1e-8)
                else:
                    d = -eta * g
                th = th + d
            alltr.append(tr)
        for i in range(len(alltr[0])):
            traces.append(" ".join([str(i * every)] + [f"{t[i][0]:.4e} {t[i][1]:.4e}" for t in alltr] + ["|", f"{alltr[0][i][2]:.4e}"]))
        open(os.path.join(out, f"test8_{opt}_k{k}_w{width}.txt"), "w").write("\n".join(rows) + "\n")
        open(os.path.join(out, f"test8_{opt}_k{k}_w{width}_traces.txt"), "w").write("\n".join(traces) + "\n")


# ----------------------------------------------------------------------------- test 4
def lorenz_rk4(x, h, sub=10):
    def rhs(x):
        return torch.stack([10 * (x[:, 1] - x[:, 0]), x[:, 0] * (28 - x[:, 2]) - x[:, 1], x[:, 0] * x[:, 1] - 8 / 3 * x[:, 2]], 1)
    dt = h / sub
    for _ in range(sub):
        k1 = rhs(x); k2 = rhs(x + dt / 2 * k1); k3 = rhs(x + dt / 2 * k2); k4 = rhs(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def test4(out, width=64, n=1024, hs=(0.005, 0.01, 0.02, 0.05), nsteps=3000):
    global NPTS; NPTS = n
    x = torch.tensor([[1.0, 1.0, 1.0]]); pts = []
    for i in range(2000 + 20 * n):
        x = lorenz_rk4(x, 0.01, sub=2)
        if i >= 2000 and (i - 2000) % 20 == 0: pts.append(x.clone())
    Xs = torch.cat(pts)[:n]; scale = float(Xs.abs().max()); X = (Xs / scale)
    idf = X.reshape(-1).to(DEV)
    L = ["# Test 4: Euler (Q=id) vs direct (Q=0), Lorenz-63 (states scaled by max|x|), zero output init, GD, equal eta and steps.",
         "# cols: h T | ||E_E(0)|| ||E_E(T)|| max_t||E_E|| | ||E_D(0)|| ||E_D(T)|| | ell_D r_D | F0id(ell_D) F0id(r_D) | floor_path floor_disp floor_dir (each = F0id - drift - M1 h) | 2 M1 h | h*_path h*_disp h*_dir"]
    for h in hs:
        Phi = lorenz_rk4(Xs, h) / scale
        U = {"E": (Phi - X).reshape(-1).to(DEV), "D": Phi.reshape(-1).to(DEV)}
        M1 = float(((Phi - X) / h).norm(dim=1).max())
        nets = {}
        for name in ("E", "D"):
            torch.manual_seed(1); nets[name] = Net(MLP(3, width, 3, zero_last=True), X)
        lam1 = {nm: float(eig(kernel(nt.Psi(nt.theta0)))[0][0]) for nm, nt in nets.items()}
        eta = 0.5 / max(lam1.values()); T = nsteps * eta
        res = {}
        sub = torch.randperm(n)[:128].to(DEV)
        for name, net in nets.items():
            th = net.theta0.clone(); E0 = net.outflat(th) - U[name]; Psi0 = net.Psi(th)
            lam0, V0 = eig(kernel(Psi0))
            ell = 0.0; H2bar = net.H2(th, pts=sub); errs = [nrm(E0)]; mid = None
            for j in range(nsteps):
                E = net.outflat(th) - U[name]; Psi = net.Psi(th); g = Psi.T @ E / n
                if j % 500 == 0 and j > 0: H2bar = max(H2bar, net.H2(th, pts=sub))
                if j == nsteps // 2:
                    lamm, Vm = eig(kernel(Psi)); mid = dict(th=th.clone(), lam=lamm, e=comps(Vm, E), E=nrm(E))
                th = th - eta * g; ell += eta * float(g.norm()); errs.append(nrm(net.outflat(th) - U[name]))
            Delta = th - net.theta0; r = float(Delta.norm()); h2 = net.h2_dir(net.theta0, Delta)
            rem = nrm(net.remainder(net.theta0, Psi0, th))
            Dm = th - mid["th"]; rm = float(Dm.norm()); h2m = net.h2_dir(mid["th"], Dm)
            floor_mid = ridge_F(mid["lam"], mid["e"], rm)[0] - 0.5 * h2m * rm ** 2
            res[name] = dict(E0=errs[0], ET=errs[-1], maxE=max(errs), ell=ell, r=r, h2=h2, rem=rem, H2bar=H2bar, lam0=lam0, V0=V0, Emid=mid["E"], rmid=rm, floor_mid=floor_mid)
        eh_id = comps(res["D"]["V0"], idf); D = res["D"]
        F0l = ridge_F(D["lam0"], eh_id, D["ell"])[0]; F0r = ridge_F(D["lam0"], eh_id, D["r"])[0]
        fl = [F0l - 0.5 * D["H2bar"] * D["ell"] ** 2 - M1 * h, F0r - 0.5 * D["H2bar"] * D["r"] ** 2 - M1 * h, F0r - 0.5 * D["h2"] * D["r"] ** 2 - M1 * h]
        # h* for each form: largest h with 2 M1 h < F0id(.) - drift, i.e. h* = (F0id - drift)/(2 M1), with ell_D, r_D as measured at this h
        hstar = [max(0.0, (x + M1 * h) / (2 * M1)) for x in fl]
        L.append(" ".join([f"{h:.3f}", f"{T:.3f}", "|", f(res["E"]["E0"]), f(res["E"]["ET"]), f(res["E"]["maxE"]), "|", f(D["E0"]), f(D["ET"]), "|", f(D["ell"]), f(D["r"]), "|",
                           f(F0l), f(F0r), "|", f(fl[0]), f(fl[1]), f(fl[2]), "|", f(2 * M1 * h), "|", f(hstar[0]), f(hstar[1]), f(hstar[2])]))
        L.append(f"#   h={h}: M1={M1:.3e} ||id||={nrm(idf):.3e} eta={eta:.3e} direct: H2bar(subsampled)={D['H2bar']:.3e} h2(dir)={D['h2']:.3e} true_rem={D['rem']:.3e} 0.5*H2bar*ell^2={0.5*D['H2bar']*D['ell']**2:.3e} 0.5*h2*r^2={0.5*D['h2']*D['r']**2:.3e};  Euler: r={res['E']['r']:.3e} rem={res['E']['rem']:.3e}")
        L.append(f"#   h={h}: rebased from T/2: direct ||E||(T/2)={D['Emid']:.3e} r from mid={D['rmid']:.3e} floor_dir from mid={D['floor_mid']: .3e} (actual {D['ET']:.3e});  Euler ||E||(T/2)={res['E']['Emid']:.3e} floor_dir from mid={res['E']['floor_mid']: .3e} (actual {res['E']['ET']:.3e})")
        print("\n".join(L[-2:]), flush=True)
    open(os.path.join(out, "test4.txt"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", type=int, required=True); ap.add_argument("--out", default=".")
    ap.add_argument("--width", type=int); ap.add_argument("--n", type=int); ap.add_argument("--nsteps", type=int)
    ap.add_argument("--ks", type=str); ap.add_argument("--hs", type=str); ap.add_argument("--rebase", type=int); ap.add_argument("--every", type=int); ap.add_argument("--opt", type=str); ap.add_argument("--layers", type=int); ap.add_argument("--gamma", type=float); ap.add_argument("--seeds", type=int); ap.add_argument("--seed0", type=int)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    kw = {k: v for k, v in dict(width=a.width, n=a.n, nsteps=a.nsteps, rebase=a.rebase, every=a.every, opt=a.opt).items() if v is not None}
    if a.ks: kw["ks"] = tuple(int(x) for x in a.ks.split(","))
    if a.hs: kw["hs"] = tuple(float(x) for x in a.hs.split(","))
    if a.test == 3: kw = {k: v for k, v in kw.items() if k in ("width", "n", "nsteps")}
    if a.test == 4: kw = {k: v for k, v in kw.items() if k in ("width", "n", "nsteps", "hs")}
    if a.test == 1: kw = {k: v for k, v in kw.items() if k in ("width", "n", "ks")}
    if a.test == 5: kw = {k: v for k, v in kw.items() if k in ("width", "n", "ks", "nsteps", "every")}
    if a.test == 6:
        kw = {k: v for k, v in kw.items() if k in ("width", "n", "ks", "nsteps", "every")}
        if a.layers is not None: kw["layers"] = a.layers
        if a.opt: kw["frozen_mode"] = a.opt
        if a.gamma is not None: kw["gamma_rel"] = a.gamma
    if a.test == 7:
        kw = {k: v for k, v in kw.items() if k in ("width", "n", "ks", "nsteps", "every", "opt")}
        if a.layers is not None: kw["layers"] = a.layers
        if a.seeds is not None: kw["seeds"] = a.seeds
    if a.test == 8:
        kw = {k: v for k, v in kw.items() if k in ("width", "n", "ks", "nsteps", "every", "opt")}
        if a.layers is not None: kw["layers"] = a.layers
        if a.seeds is not None: kw["seeds"] = a.seeds
        if a.seed0 is not None: kw["seed0"] = a.seed0
    print(f"# device {DEV} args {kw}", flush=True)
    t0 = time.time()
    {1: test1, 2: test2, 3: test3, 4: test4, 5: test5, 6: test6, 7: test7, 8: test8}[a.test](a.out, **kw)
    print(f"# done test {a.test} in {time.time() - t0:.1f}s", flush=True)
