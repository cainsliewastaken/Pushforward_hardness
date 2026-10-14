"""KronQ, fast form (2026-10-12): the per-layer Kronecker-eigenbasis optimizer with every component developed in the
optimizer line (kronq2 / kronq3h / kronsign / kadam_front) and the rotations paid where they are cheap.

Layer l has weights Wt (o x i+1, bias folded) stored IN THE CURRENT EIGENBASIS (train_ks --reparam): the forward rotates
the input, a~ = U_A^T [a; 1], computes z~ = Wt a~, and rotates back, z = U_S z~ (two thin products, 2 m i^2 + 2 m o^2;
with blocks of b, 2 m i b + 2 m o b). autograd then gives the rotated gradient Gr = dL/dWt directly: no w^3 per step.
Factor statistics are accumulated in the current basis (F~ = E[a~ a~^T], E[d~ d~^T]); at a refresh the eigh of F~ gives the
basis change V relative to the current one, and Wt, momentum, factors and per-mode state are transported by V (exact
within blocks). Per-mode curvature lam_ij = e^S_i e^A_j (Kronecker) or the EKFAC diagonal E[(d~_i)^2 (a~_j)^2] (--ekfac).
Rules on the rotated momentum C (b1): sign push D = w sign(C) (soft sign with --sign_floor), w = (lam/lam_max)^(q/2) or
exp(u) with u learned by the one-step hypergradient (kronq3h; --hyper, --horizon, --hyper_shape), capped at the GN step
|Gr|/(lam lr eta_l L) (--cap); or the Adam rule D = C_hat / (sqrt(V_hat) + eps) with V = EMA Gr^2 (--rule adam, = the
field-problem winner 'ks_adam'). Step length: lr eta_l with eta_l from the function-space remainder controller (--lr_kappa,
round-robin jvp+forward every --check_every), or the output-change budget (--rem), or the exact-GN line search (--gnls).
Refresh: every --refresh steps or error-controlled (--adapt_refresh kappa: e = relative change of the first-order decrease
sum |Gr| under the basis change); eigh or warm-started subspace iteration (--eig_iters). Stats every --stats_every steps from
the MC probe (one random +-1 output combination; --S_mode mc) or the empirical signal (--S_mode emp, rescaled to the GN
trace once). --last_S identity: no S factor on the output layer. --centred: train f - f_0. Memory: C (bf16 opt), u / Dprev
(bf16 or int8 signs), EKFAC diagonal (bf16 opt), V for the adam rule; factors and bases per block.
"""
import math, torch
from torch.func import jvp

EIG_DTYPE = None
CHOLQR = True
def solve_gamma(Cabs, lam, target2, iters=40):
    """gamma with sum (Cabs / (lam + gamma))^2 = target2 (decreasing in gamma); bisection in log gamma."""
    lo, hi = -30.0, 30.0
    for _ in range(iters):
        g = 0.5 * (lo + hi); v = float((Cabs / (lam + math.exp(g))).pow(2).sum())
        if v > target2: lo = g
        else: hi = g
    return math.exp(0.5 * (lo + hi))
class exact_fp32:
    """TF32 off inside: the remainder f(W+d) - f(W) - J d is below TF32 rounding at small steps, so the controller and the
    shape diagnostic must measure it with full fp32 matmuls."""
    def __enter__(self):
        self.t = torch.backends.cuda.matmul.allow_tf32; self.p = torch.get_float32_matmul_precision()
        torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
    def __exit__(self, *e):
        torch.backends.cuda.matmul.allow_tf32 = self.t; torch.set_float32_matmul_precision(self.p)
def gelu_dd(z):
    """second derivative of the exact GELU z Phi(z): phi(z) (2 - z^2)."""
    return torch.exp(-0.5 * z * z) * (2.0 - z * z) * 0.3989422804014327
def cholqr(Y, passes=3):
    """orthonormalise the columns of Y (batched ... x n x k) by CholeskyQR2: G = Y^T Y in float64, V = Y L^-T, twice.
    Falls back to QR if the Gram is not positive definite. 10x+ faster than batched QR on the GPU."""
    V = Y
    for _ in range(passes):
        G = (V.transpose(-1, -2).double() @ V.double()); G = G + 1e-12 * G.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
        L, info = torch.linalg.cholesky_ex(G)
        if int(info.max()) != 0: return torch.linalg.qr(Y)[0]
        V = torch.linalg.solve_triangular(L, V.double().transpose(-1, -2), upper=False).transpose(-1, -2).to(Y.dtype)
    return V
def _fused_update(g, C, u, r_, c_, Dprev, Se, Ae, lr_eta, lam_scale, b1, beta_u, u_clamp, reltol, L):   # lr_eta: 0-d tensor (varies per step)
    """the reference configuration's per-layer update in one graph: momentum, one-step hypergradient with the rank-one
    normaliser, sign push with the learned weight, live mask, cap. Returns D (float32); C, u, r_, c_ updated in place."""
    C.mul_(b1).add_(g.to(C.dtype), alpha=1 - b1); Cf = C.float()
    if Dprev is not None:
        hg = -(lr_eta * g) * Dprev.float(); hg2 = hg * hg
        r_.mul_(0.99).add_(hg2.mean(1), alpha=0.01); c_.mul_(0.99).add_(hg2.mean(0), alpha=0.01)
        den = (r_[:, None] * c_[None, :] / (r_.mean() + 1e-30)).sqrt()
        un = u.float() - beta_u * hg / (den + 1e-30); u.copy_(un.clamp(-u_clamp, u_clamp).to(u.dtype))
    lam = Se[:, None] * Ae[None, :] * lam_scale; lmax = lam.max(); live = lam > reltol * lmax
    D = torch.where(live, torch.exp(u.float()) * torch.sign(Cf), torch.zeros_like(Cf))
    capv = g.abs() / (lam.clamp(min=1e-30) * (lr_eta * L))
    D = torch.sign(D) * torch.minimum(D.abs(), capv)
    return D, lam, live
try:
    fused_update = torch.compile(_fused_update, dynamic=False)
except Exception:
    fused_update = _fused_update
def safe_eigh(F):
    """eigh of a (batch of) symmetric matrices with a float64 CPU fallback when cusolver fails to converge."""
    Fs = 0.5 * (F + F.transpose(-1, -2))
    if EIG_DTYPE is not None: Fs = Fs.to(EIG_DTYPE)
    try:
        ev, V = torch.linalg.eigh(Fs); return ev.to(F.dtype), V.to(F.dtype)
    except Exception:
        ev, V = torch.linalg.eigh(Fs.double().cpu()); return ev.to(F.device, F.dtype), V.to(F.device, F.dtype)
class Basis:
    """Block-diagonal orthogonal basis on one side (dim): batched full blocks + one remainder block; or none (diag mode)."""
    def __init__(self, dim, block, max_dim, dev):
        self.dim = dim; self.diag = dim > max_dim; self.dev = dev
        b = block if 0 < block < dim else dim
        self.b, self.nf = b, dim // b; self.rem = dim - self.nf * b
        self.U = None; self.Ur = None       # None = identity (before the first refresh, or diag mode)
        self.F = None; self.Fr = None; self.e = None
    # --- apply a block-diagonal matrix B (Bf: nf x b x b, Br: rem x rem) to rows ('L') or columns ('R') of M
    def _apply(self, M, side, Bf, Br, transpose):
        nb = self.nf * self.b
        if side == 'L':
            n = M.shape[1]; out = torch.bmm(Bf.transpose(1, 2) if transpose else Bf, M[:nb].reshape(self.nf, self.b, n)).reshape(nb, n)
            if self.rem: out = torch.cat([out, (Br.T if transpose else Br) @ M[nb:]], 0)
        else:
            n = M.shape[0]; Mf = M[:, :nb].reshape(n, self.nf, self.b)
            out = torch.einsum('nkb,kbc->nkc' if transpose else 'nkb,kcb->nkc', Mf, Bf).reshape(n, nb)
            if self.rem: out = torch.cat([out, M[:, nb:] @ (Br if transpose else Br.T)], 1)
        return out
    def rows_to_basis(self, M):      # U^T M   (M: dim x n)
        return M if self.U is None else self._apply(M, 'L', self.U, self.Ur, True)
    def rows_from_basis(self, M):    # U M     (M: dim x n)
        return M if self.U is None else self._apply(M, 'L', self.U, self.Ur, False)
    def rows_to_basis_sq(self, M):   # (U (.) U)^T M   (M: dim x n): the entrywise-squared basis, for second-derivative sums
        return M if self.U is None else self._apply(M, 'L', self.U * self.U, (self.Ur * self.Ur) if self.Ur is not None else None, True)
    def cols_to_basis(self, M):      # M U     (M: n x dim)
        return M if self.U is None else self._apply(M, 'R', self.U, self.Ur, True)
    def cols_from_basis(self, M):    # M U^T   (M: n x dim)
        return M if self.U is None else self._apply(M, 'R', self.U, self.Ur, False)
    # --- statistics in the current basis
    def update(self, Zt, beta):      # Zt: samples x dim, already in the current basis
        m = Zt.shape[0]
        if self.diag:
            new = (Zt * Zt).mean(0)
            self.F = new if self.F is None else self.F.mul_(beta).add_(new, alpha=1 - beta); return
        nb = self.nf * self.b; Zf = Zt[:, :nb].reshape(m, self.nf, self.b)
        newF = torch.einsum('mkb,mkc->kbc', Zf, Zf) / m; newR = (Zt[:, nb:].T @ Zt[:, nb:] / m) if self.rem else None
        if self.F is None: self.F, self.Fr = newF, newR
        else:
            self.F.mul_(beta).add_(newF, alpha=1 - beta)
            if self.rem: self.Fr.mul_(beta).add_(newR, alpha=1 - beta)
    def refresh(self, iters=0):
        """Diagonalise the factor in the current basis; returns the basis change (Vf, Vr) relative to the current basis and
        updates U <- U V and F <- diag(e)."""
        if self.diag: self.e = self.F.clamp(min=0).clone(); return None
        if self.F is None: return None
        if iters <= 0:
            ev, V = safe_eigh(self.F); e = ev.clamp(min=0).reshape(-1)
            if self.rem: evr, Vr = safe_eigh(self.Fr); e = torch.cat([e, evr.clamp(min=0)])
            else: Vr = None
        else:
            V = torch.eye(self.b, device=self.dev).expand(self.nf, self.b, self.b).clone()
            for _ in range(iters): V = cholqr(torch.bmm(self.F, V)) if CHOLQR else torch.linalg.qr(torch.bmm(self.F, V))[0]
            e = torch.einsum('kbc,kbd,kdc->kc', V, self.F, V).clamp(min=0).reshape(-1)
            if self.rem:
                Vr = torch.eye(self.rem, device=self.dev)
                for _ in range(iters): Vr = cholqr(self.Fr @ Vr) if CHOLQR else torch.linalg.qr(self.Fr @ Vr)[0]
                e = torch.cat([e, torch.diagonal(Vr.T @ self.Fr @ Vr).clamp(min=0)])
            else: Vr = None
        # new basis U V; factor becomes (nearly) diagonal in it
        if self.U is None:
            self.U = V; self.Ur = Vr; self.nacc = 0
        else:
            self.U = torch.bmm(self.U, V); self.nacc = getattr(self, 'nacc', 0) + 1
            if self.rem: self.Ur = self.Ur @ Vr
            if CHOLQR and self.nacc % 10 == 0:                      # re-orthonormalise the accumulated basis (Cholesky-QR drift ~1e-4 per refresh)
                self.U = cholqr(self.U)
                if self.rem: self.Ur = cholqr(self.Ur)
        self.F = torch.bmm(torch.bmm(V.transpose(1, 2), self.F), V)
        if self.rem: self.Fr = Vr.T @ self.Fr @ Vr
        self.e = e
        return (V, Vr)
    def transport(self, T, V, side, squared=False):
        """T expressed in the old basis -> new basis: rows ('L', T: dim x n): V^T T; columns ('R', T: n x dim): T V.
        squared=True transports a second moment by the squared overlaps."""
        if V is None: return T
        Vf, Vr = V
        if squared: Vf, Vr = Vf * Vf, (Vr * Vr if Vr is not None else None)
        return self._apply(T, side, Vf, Vr, True)
    def state_bytes(self):
        n = lambda x: 0 if x is None else x.numel() * x.element_size()
        return n(self.F) + n(self.Fr) + n(self.U) + n(self.Ur)


class KronQFast:
    def __init__(self, lins, a, dev, nout, L_act, draw_eps):
        """lins: list of nn.Linear (their weights are replaced by rotated Wt); a: args; L_act: activation for hidden layers."""
        self.a, self.dev, self.nout, self.L = a, dev, nout, len(lins); self.act = L_act; self.draw_eps = draw_eps
        self.Wt = [torch.nn.Parameter(torch.cat([lin.weight.detach(), lin.bias.detach()[:, None]], 1).clone()) for lin in lins]
        self.A = [Basis(lin.weight.shape[1] + 1, a.block, (0 if a.A_identity else a.max_dim), dev) for lin in lins]
        self.S = [Basis(lin.weight.shape[0], a.block, (0 if (a.S_identity or (a.last_S == 'identity' and l == self.L - 1)) else a.max_dim), dev) for l, lin in enumerate(lins)]
        bf = torch.bfloat16 if a.compress else torch.float32
        self.C = [torch.zeros_like(W, dtype=torch.bfloat16 if a.mom_bf16 else torch.float32) for W in self.Wt]
        self.u = [torch.zeros_like(W, dtype=bf) for W in self.Wt] if a.hyper else None
        self.hv = [torch.zeros_like(W, dtype=torch.float32) if not a.compress else (torch.zeros(W.shape[0], device=dev), torch.zeros(W.shape[1], device=dev)) for W in self.Wt] if a.hyper else None
        self.Dprev = [None] * self.L; self.z_tr = [None] * self.L   # eligibility traces (--horizon > 1)
        self.V2 = [torch.zeros_like(W, dtype=bf) for W in self.Wt] if a.rule == 'adam' else None
        self.lamx = [None] * self.L if a.ekfac else None
        if a.mom_adapt and self.V2 is None: self.V2 = [torch.zeros(W.shape, device=dev, dtype=torch.bfloat16 if a.compress else torch.float32) for W in self.Wt]
        self.Tk_log = [None] * self.L; self.m2s = [None] * self.L   # smoothed de-biased signal^2 per mode (mom_adapt)
        self.Dabs = [None] * self.L                                    # |D| of the previous step per mode (staleness limit)
        self.sc = [None] * self.L; self.sc2 = [None] * self.L; self.Dprev2 = [None] * self.L   # smoothed lag-1 / lag-2 sign correlations and the sign two steps back (signcorr rule)
        self.cres = [None] * self.L if (a.alloc2 or a.curv) else None   # residual curvature per mode c_ij = <E, f_uu> (LOOKAHEAD_THEORY Prop. 10)
        self.cres_n = [0] * self.L                                        # samples since the basis last changed: running mean, then EMA (not transportable)
        self.T_eff = 1.0
        self.use_fused = bool(a.fused) and a.hyper and a.cap and a.compress and a.rule == 'sign' and a.wrule == 'learned' and not a.dprev_bits and a.horizon <= 1 \
            and not a.hyper_shape and a.q == 0 and a.qa == 0 and a.q_in == 0 and a.sign_floor <= 0 and a.topk <= 0 and not a.alloc and not a.alloc2 \
            and not a.mom_adapt and a.hyper_rule == 'idbd' and not a.gnls and not a.ekfac and not a.lr_kappa and not a.sphere and not a.radius_scan and not a.check_curv
        self.nref = [0] * self.L; self._lr_eta_t = torch.zeros((), device=dev); self.capL = float(self.L) if a.cap_L else 1.0; self.c_ema = None; self.rem_dyn = None
        self.xv_fresh = None   # (x, y) fresh batch for --rem_val, set by the harness each step
        self.zbud = [float(x) for x in a.zbud.split(',')] if a.zbud else []
        self.scan_steps = set(int(x) for x in a.radius_scan.split(',')) if a.radius_scan else set()
        self.tracker = None
        self.acf_steps = set(int(x) for x in a.acf.split(',')) if a.acf else set(); self.acf_buf = None; self.acf_idx = None; self.acf_t0 = None
        self.eta = [1.0] * self.L; self.refresh_cur = a.refresh; self.emp_scale = [None] * self.L
        self.period = [a.refresh] * self.L; self.next_ref = [1] * self.L; self.rf = [None] * self.L
        self.last_gamma = [None] * self.L; self.kappa = a.lr_kappa if a.lr_kappa > 0 else a.sphere
        global EIG_DTYPE; EIG_DTYPE = torch.float64 if a.eig_dtype == 'float64' else None
        self.cache = {}; self.last_Gr = None; self.diag_out = {}
        self.f0 = None
    def parameters(self): return self.Wt
    # ---------------- forward in the rotated parametrisation
    def forward(self, x, keep=False):
        h = x; at_list = []; zt_list = []; h_list = []
        for l in range(self.L):
            a_aug = torch.cat([h, torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype)], 1)
            at = self.A[l].cols_to_basis(a_aug)                  # a~ = U_A^T [a;1]  (rows are samples)
            zt = at @ self.Wt[l].T                                 # z~ = Wt a~
            if keep: zt.retain_grad(); at_list.append(at); zt_list.append(zt)
            z = self.S[l].cols_from_basis(zt)                      # z = U_S z~
            h = self.act(z) if l < self.L - 1 else z
            if keep and l < self.L - 1: h.retain_grad(); h_list.append(h)
        if keep: self.cache = dict(at=at_list, zt=zt_list, h=h_list)
        return h
    @torch.no_grad()
    def preacts(self, x):
        """hidden-layer preactivations z = U_S Wt U_A^T [h;1] from the trained (rotated) weights, for the diagnostics."""
        h = x; zs = []
        for l in range(self.L - 1):
            a_aug = torch.cat([h, torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype)], 1)
            z = self.S[l].cols_from_basis(self.A[l].cols_to_basis(a_aug) @ self.Wt[l].T); zs.append(z); h = self.act(z)
        return zs
    @torch.no_grad()
    def params_orig(self):
        """the trained weights in the original coordinates: [W0, b0, W1, b1, ...], W = U_S Wt U_A^T split into weight and bias."""
        out = []
        for l in range(self.L):
            W = self.S[l].rows_from_basis(self.A[l].cols_from_basis(self.Wt[l].data)); out += [W[:, :-1].contiguous(), W[:, -1].contiguous()]
        return out
    @torch.no_grad()
    def rotate_grads(self, G_aug):
        """original-coordinate gradients (o x i+1 per layer) -> the current Kronecker basis, with the lam arrays."""
        R = [self.A[l].cols_to_basis(self.S[l].rows_to_basis(G_aug[l])) for l in range(self.L)]
        lams = [None if (self.S[l].e is None or self.A[l].e is None) else self.S[l].e[:, None] * self.A[l].e[None, :] for l in range(self.L)]
        return R, lams
    def check_curv(self, xb, yb, loss_fn, emit, step, nmodes=4, hgrad=None):
        """exact check of the residual curvature: for a few modes (i, j) of each hidden layer, the loss Hessian along the
        mode (a Hessian-vector product; the mode is the (i, j) entry of Wt) minus the exact Gauss-Newton part (a jvp)
        against the factorised estimate cres[l][i, j]; also lam[i, j] against the exact GN part."""
        from torch.autograd.functional import hvp
        a = self.a
        for l in range(self.L - 1):
            if self.cres is None or self.cres[l] is None or self.S[l].e is None: continue
            lam = self.S[l].e[:, None] * self.A[l].e[None, :] * (2.0 / self.nout); o, i = self.Wt[l].shape
            idx = [(0, 0), (0, i // 2), (o // 2, 0)] + [(int(torch.randint(o, (1,))), int(torch.randint(i, (1,)))) for _ in range(nmodes - 3)]
            W0 = self.Wt[l].data.clone()
            with exact_fp32(), torch.enable_grad():
                for (ii, jj) in idx:
                    v = torch.zeros_like(W0); v[ii, jj] = 1.0
                    _, Hv = hvp(lambda Wl: loss_fn(self.forward_with(xb, l, Wl), yb), W0, v)
                    d2 = float(Hv[ii, jj])
                    _, Jv = jvp(lambda Wl: self.forward_with(xb, l, Wl), (W0,), (v,))
                    gn = float((Jv * Jv).sum()) * 2.0 / (self.nout * xb.shape[0])
                    emit(f'curvcheck_l{l}_i{ii}_j{jj}_c_true', d2 - gn, step); emit(f'curvcheck_l{l}_i{ii}_j{jj}_c_est', float(self.cres[l][ii, jj]), step)
                    emit(f'curvcheck_l{l}_i{ii}_j{jj}_gn_true', gn, step); emit(f'curvcheck_l{l}_i{ii}_j{jj}_lam_est', float(lam[ii, jj]), step)
                    # brute force of the formula with autograd for dL/dh and dense bases
                    Sb = self.S[l]; Ab = self.A[l]
                    US = torch.block_diag(*[Sb.U[k] for k in range(Sb.nf)] + ([Sb.Ur] if Sb.Ur is not None else [])) if Sb.U is not None else torch.eye(o, device=W0.device)
                    hcur = xb
                    for ll in range(l): hcur = self.act(self.S[ll].cols_from_basis(self.A[ll].cols_to_basis(torch.cat([hcur, torch.ones(hcur.shape[0], 1, device=hcur.device)], 1)) @ self.Wt[ll].detach().T))
                    a_aug = torch.cat([hcur, torch.ones(hcur.shape[0], 1, device=hcur.device)], 1); at_ = Ab.cols_to_basis(a_aug)
                    z_ = Sb.cols_from_basis(at_ @ W0.T); h_ = self.act(z_).requires_grad_(True); hh = h_
                    for ll in range(l + 1, self.L):
                        a2 = torch.cat([hh, torch.ones(hh.shape[0], 1, device=hh.device)], 1); zz = self.S[ll].cols_from_basis(self.A[ll].cols_to_basis(a2) @ self.Wt[ll].detach().T)
                        hh = self.act(zz) if ll < self.L - 1 else zz
                    gh, = torch.autograd.grad(loss_fn(hh, yb), h_)
                    c_bf = float(((US[:, ii] ** 2)[None, :] * gh * gelu_dd(z_) * (at_[:, jj] ** 2)[:, None]).sum())
                    emit(f'curvcheck_l{l}_i{ii}_j{jj}_c_bf', c_bf, step)
                    # the vectorised product on the current cache, before EMA and transport
                    zc = Sb.cols_from_basis(self.cache['zt'][l].detach()); rc = hgrad[l] * gelu_dd(zc); atc = self.cache['at'][l].detach()
                    newc = Sb.rows_to_basis_sq(rc.T @ (atc * atc)); emit(f'curvcheck_l{l}_i{ii}_j{jj}_c_vec', float(newc[ii, jj]), step)
                    emit(f'curvcheck_l{l}_i{ii}_j{jj}_c_vecdense', float(((US ** 2).T @ (rc.T @ (atc * atc)))[ii, jj]), step)
                    emit(f'curvcheck_l{l}_i{ii}_j{jj}_gh_ratio', float((gh - hgrad[l]).norm() / (gh.norm() + 1e-30)), step)
                    emit(f'curvcheck_l{l}_i{ii}_j{jj}_at_ratio', float((at_ - atc).norm() / (at_.norm() + 1e-30)), step)
    def radius_scan(self, xb, l, dirs, s0, emit, step, nl=13):
        """Taylor radius along several directions at the current point: for each unit direction u, the relative remainder
        rf(s) = ||f(W + s u) - f(W) - s J u|| / (s ||J u||) on a log grid of lengths s around the step's length s0, the
        leverage ||J u|| (output change per unit length), and the length s* at which rf first reaches 0.1, in parameter
        units and in output units (s* ||J u||). TF32 off."""
        W0 = self.Wt[l].data; grid = [s0 * 10 ** (-3 + 5.0 * k / (nl - 1)) for k in range(nl)]
        with exact_fp32():
            f0 = self.forward_with(xb, l, W0)
            for name, Dir in dirs.items():
                u = Dir / (float(Dir.norm()) + 1e-30)
                _, Ju = jvp(lambda Wl: self.forward_with(xb, l, Wl), (W0,), (u,)); lev = float(Ju.norm())
                E0 = (f0 - self._scan_yb).flatten(); cosE = float((E0 @ Ju.flatten()) / (E0.norm() * Ju.norm() + 1e-30))   # alignment of the descent step's output response (-J u) with -E
                emit(f'rscan_{name}_l{l}_cosE', cosE, step)
                rfs = []
                for k, s_ in enumerate(grid):
                    f1 = self.forward_with(xb, l, W0 + s_ * u); rf = float((f1 - f0 - s_ * Ju).norm() / (s_ * lev + 1e-30)); rfs.append(rf)
                    emit(f'rscan_{name}_l{l}_rf{k}', rf, step)
                sstar = float('nan')
                for k in range(1, nl):
                    if rfs[k - 1] < 0.1 <= rfs[k]:
                        a0, a1 = math.log(grid[k - 1]), math.log(grid[k]); r0, r1 = math.log(max(rfs[k - 1], 1e-12)), math.log(rfs[k])
                        sstar = math.exp(a0 + (math.log(0.1) - r0) * (a1 - a0) / (r1 - r0)); break
                if math.isnan(sstar) and rfs[0] >= 0.1: sstar = grid[0]
                emit(f'rscan_{name}_l{l}_sstar', sstar, step); emit(f'rscan_{name}_l{l}_lev', lev, step)
                if l < self.L - 1 and sstar == sstar:
                    at_ = self.cache['at'][l].detach(); zt_ = self.cache['zt'][l].detach()
                    emit(f'rscan_{name}_l{l}_dzrel', float((sstar * (at_ @ u.T)).norm() / (zt_.norm() + 1e-30)), step)   # ||dz|| / ||z|| at the radius
                emit(f'rscan_{name}_l{l}_sstar_out', sstar * lev, step); emit(f'rscan_{name}_l{l}_s0', s0, step)
    def forward_with(self, x, lw, Wl):
        """forward using Wl in place of Wt[lw] (functional; for jvp)."""
        h = x
        for l in range(self.L):
            a_aug = torch.cat([h, torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype)], 1)
            at = self.A[l].cols_to_basis(a_aug); zt = at @ (Wl if l == lw else self.Wt[l].detach()).T
            z = self.S[l].cols_from_basis(zt); h = self.act(z) if l < self.L - 1 else z
        return h
    def state_bytes(self):
        n = lambda x: 0 if x is None else (sum(t.numel() * t.element_size() for t in x) if isinstance(x, tuple) else x.numel() * x.element_size())
        tot = sum(B.state_bytes() for B in self.A + self.S) + sum(n(C) for C in self.C)
        for arr in (self.u, self.hv, self.V2, self.lamx, self.Dprev):
            if arr is not None: tot += sum(n(t) for t in arr)
        return tot
    def direction(self, l, g, Cf, lr, update_state=True):
        """per-mode step direction D (in units of lr) for layer l from the rotated gradient g and momentum Cf."""
        a = self.a; L = self.L
        if self.S[l].e is None or self.A[l].e is None: lam = torch.ones_like(g); lmax = 1.0
        else:
            lam = self.S[l].e[:, None] * self.A[l].e[None, :] * (2.0 / self.nout)
            if self.lamx is not None and self.lamx[l] is not None: lam = self.lamx[l].float() * (2.0 / self.nout)
            lmax = float(lam.max())
        live = lam > a.reltol * lmax
        if a.topk > 0:
            thr = lam.flatten().kthvalue(max(1, lam.numel() - a.topk))[0]; live = live & (lam >= thr)
        if a.alloc:                                   # derived T-step allocation on the per-step length budget R = lr eta sqrt(N)
            T_rem = max(10, a.steps - self.t_step + 1); R = lr * self.eta[l] * math.sqrt(g.numel())
            gam = solve_gamma(Cf.abs(), lam, (T_rem * R) ** 2)
            D = Cf / (lam + gam) / T_rem / max(lr * self.eta[l], 1e-30)
            if a.sphere > 0: D = D * (math.sqrt(D.numel()) / (float(D.norm()) + 1e-30))
            self.last_gamma[l] = gam / lmax
            return D, lam, live
        if a.alloc2:                                  # LOOKAHEAD_THEORY Thm 7: full-curvature ridge on convex modes, full-radius push on concave ones
            c = self.cres[l].float() if (self.cres is not None and self.cres[l] is not None) else torch.zeros_like(lam)
            cur = lam + a.cfac * c; beta = a.box; unit = max(lr * self.eta[l], 1e-30) * (1.0 if a.alloc2_oneshot else self.T_eff)
            def D_of(mu):
                conv = cur + mu > 0
                Dc = (Cf / (cur + mu).clamp(min=1e-30) / unit).clamp(-beta, beta)
                return torch.where(conv, Dc, beta * torch.sign(Cf))
            mu = 0.0; D = D_of(0.0); N = float(D.numel())
            if beta > 1 and float((D * D).sum()) > N:       # sphere budget sum D^2 <= N binds: raise the damping mu
                lo, hi = -30.0, 30.0
                for _ in range(40):
                    g_ = 0.5 * (lo + hi); Dm = D_of(math.exp(g_))
                    if float((Dm * Dm).sum()) > N: lo = g_
                    else: hi = g_
                mu = math.exp(hi); D = D_of(mu)
            self.last_gamma[l] = mu / lmax
            return D, lam, torch.ones_like(live)
        if a.rule == 'adam':
            V2 = self.V2[l]; V2f = V2.float() if a.mom_adapt else V2.float() * a.b2 + (g * g) * (1 - a.b2)
            if update_state and not a.mom_adapt: self.V2[l] = V2f.to(V2.dtype)
            bc1 = 1 - a.b1 ** self.t_step; bc2 = 1 - a.b2 ** self.t_step
            D = (Cf / bc1) / ((V2f / bc2).sqrt() + a.eps)
            if not a.adam_fair: return D, lam, live
            if a.hyper: D = D * torch.exp(self.u[l].float())                      # fair comparison: the learned weight and the cap on the Adam rule too
            if a.cap:
                capv = g.abs() / (lam.clamp(min=1e-30) * max(lr * self.eta[l], 1e-30) * L); D = torch.sign(D) * torch.minimum(D.abs(), capv)
            return D, lam, live
        w = torch.ones_like(g)
        if a.wrule == 'derived':                      # MECHANISM.md section 3: w ~ e_k erf(SNR_k / sqrt 2) / s_k with the Kronecker SNR
            snr = math.sqrt(a.batch) * Cf.abs() / (lam.clamp(min=1e-30) * self.nout / 2.0).sqrt() * math.sqrt(1.0 / (1 - a.b1))   # momentum SNR
            w = (Cf.abs() / lam.clamp(min=1e-30)) * torch.erf(snr / math.sqrt(2.0))
            w = w / (w.median() + 1e-30); w = w.clamp(a.wrule_min, a.wrule_max)
        qq = a.q_in if (l == 0 and a.q_in != 0) else a.q                 # per-mode weight exponent; --q_in overrides it on the input layer
        if a.qa != 0 and l < L - 1 and self.A[l].e is not None:            # TRUST_REGION: the layer's radius is a sphere in pre-activation space,
            al = self.A[l].e; w = w * (al / float(al.max())).clamp(min=a.q_floor)[None, :] ** (0.5 * a.qa)   # i.e. ~ 1/sqrt(alpha_j) per mode
        if qq != 0: w = w * (lam / lmax).clamp(min=1e-300) ** (0.5 * qq)
        if a.hyper: w = w * torch.exp(self.u[l].float())
        sgn = torch.sign(Cf) if a.sign_floor <= 0 else Cf / (Cf.abs() + a.sign_floor * float(Cf.abs().max()) + 1e-30)
        D = torch.where(live, w * sgn, torch.zeros_like(g))
        if a.sphere > 0: D = D * (math.sqrt(D.numel()) / (float(D.norm()) + 1e-30))   # unit-rms direction (length lr eta sqrt(N)), then the cap
        if a.cap:
            capv = g.abs() / (lam.clamp(min=1e-30) * max(lr * self.eta[l], 1e-30) * self.capL)
            D = torch.sign(D) * torch.minimum(D.abs(), capv)
        return D, lam, live
    # ---------------- one optimizer step; call after loss.backward() on forward(xb, keep=need_keep)
    def step(self, step, lr, out, xb, yb, loss_fn, tm=None, t=None, emit=None):
        a = self.a; L = self.L; dev = self.dev
        need_stats = (step - 1) % max(1, a.stats_every) == 0
        self.t_step = step
        T_rem = a.steps - step + 1                      # effective steps left in units of the current lr (linear decay to 0 unless a controller holds lr)
        if a.rem_ctrl or a.sphere: self.T_eff = float(T_rem)
        else: self.T_eff = max(1.0, 0.5 * (T_rem + 1.0))   # sum of the remaining linear-decay factors over the current one: (T_rem + 1) / 2
        due = [l for l in range(L) if step >= self.next_ref[l]]
        Gr = [W.grad for W in self.Wt]
        if self.acf_steps:
            Wn = a.acf_len
            if step in self.acf_steps and self.acf_buf is None and all(self.S[l].e is not None for l in range(L)):
                self.acf_idx = []; self.acf_buf = []; self.acf_t0 = step
                for l in range(L):                                   # 200 modes per lambda decile, random within the decile
                    lam_ = (self.S[l].e[:, None] * self.A[l].e[None, :]).flatten(); order = torch.argsort(lam_, descending=True); n_ = order.numel()
                    idx = torch.cat([order[dec * n_ // 10:(dec + 1) * n_ // 10][torch.randperm(max(1, n_ // 10), device=order.device)[:200]] for dec in range(10)])
                    self.acf_idx.append(idx); self.acf_buf.append(torch.zeros(Wn, idx.numel(), device=self.dev))
            if self.acf_buf is not None:
                due = []                                              # hold the basis fixed for the window so a mode stays the same mode
                t_ = step - self.acf_t0
                for l in range(L): self.acf_buf[l][t_] = Gr[l].flatten()[self.acf_idx[l]]
                if t_ == Wn - 1 and emit is not None:
                    for l in range(L):
                        G = self.acf_buf[l]; nper = G.shape[1] // 10
                        for dec in range(10):
                            g_ = G[:, dec * nper:(dec + 1) * nper]; den = (g_ * g_).mean()
                            for tau in (1, 2, 4, 8, 16, 32, Wn - 1):
                                emit(f'acf_d{dec}_l{l}_tau{tau}', float((g_[:-tau] * g_[tau:]).mean() / (den + 1e-30)), self.acf_t0)
                            # the persistent (signal) fraction from the time mean: mean^2 / mean of squares, per mode, averaged
                            emit(f'acf_d{dec}_l{l}_sig', float((g_.mean(0) ** 2).mean() / (den + 1e-30)), self.acf_t0)
                    self.acf_buf = None; self.acf_idx = None
        # loss gradients at the cached intermediates, snapshotted BEFORE the probe backward: a retained .grad accumulates
        # every backward that passes through it (the probe's included, ~m ny / 2 times larger), so it cannot be read later
        hgrad = [h.grad.clone() if h.grad is not None else None for h in self.cache['h']] if (self.cres is not None or (emit is not None and (step % a.eval_every == 0 or step == a.steps))) else None
        ztgrad = [zt.grad.clone() if zt.grad is not None else None for zt in self.cache['zt']] if a.S_mode == 'emp' else None
        diag_now = emit is not None and (step % a.eval_every == 0 or step == a.steps or step in self.scan_steps)
        # ---- statistics in the current basis
        probe_now = a.gnls or (need_stats and (not a.S_identity or bool(due) or any(S_.F is None for S_ in self.S)))   # T1: probe only at refresh steps
        if need_stats or a.gnls:
            t0 = t() if t else None
            with torch.enable_grad():
                if probe_now and (a.S_mode == 'mc' or a.gnls or (a.S_mode == 'emp' and need_stats and any(sc is None for sc in self.emp_scale))):
                    eps = self.draw_eps(out.shape)
                    gzt = torch.autograd.grad((out * eps).sum(), self.cache['zt'], retain_graph=True)
                if a.S_mode == 'emp':
                    gzt_e = ztgrad
            if need_stats:
                for l in range(L):
                    at = self.cache['at'][l].detach()
                    if a.S_mode == 'emp':
                        d = gzt_e[l].detach()
                        if self.emp_scale[l] is None:       # rescale the empirical signal to the GN trace once (from the probe)
                            self.emp_scale[l] = float(gzt[l].detach().pow(2).sum().sqrt() / (d.pow(2).sum().sqrt() + 1e-30))
                        d = d * self.emp_scale[l]
                    else: d = gzt[l].detach() if probe_now else None
                    self.A[l].update(at, a.beta_stats)
                    if probe_now: self.S[l].update(d, a.beta_stats)
                    if diag_now and emit is not None and l < L - 1 and self.S[l].e is not None and self.A[l].e is not None and hgrad is not None and hgrad[l] is not None:
                        # per-sample spread of the mode's gradient term (MECHANISM Lemma 1 / assumption S): z_x,ij = d~_x,i a~_x,j, true residual
                        zt_ = self.cache['zt'][l].detach(); z_ = self.S[l].cols_from_basis(zt_)
                        gd1 = 0.5 * (1 + torch.erf(z_ / math.sqrt(2.0))) + z_ * torch.exp(-0.5 * z_ * z_) * 0.3989422804014327   # gelu'
                        dtil = self.S[l].cols_to_basis(hgrad[l] * gd1)                                  # m x o, rotated dL/dz
                        lam_ = self.S[l].e[:, None] * self.A[l].e[None, :] * (2.0 / self.nout); order = torch.argsort(lam_.flatten(), descending=True); n_ = order.numel()
                        for dec in range(10):
                            seg = order[dec * n_ // 10:(dec + 1) * n_ // 10][torch.randperm(max(1, n_ // 10), device=order.device)[:100]]
                            ii_ = seg // lam_.shape[1]; jj_ = seg % lam_.shape[1]
                            zx = dtil[:, ii_] * at[:, jj_]                                              # m x 100 per-sample terms
                            mu = zx.mean(0); sd = zx.std(0); lam_s = lam_.flatten()[seg]
                            emit(f'spread_tau_d{dec}_l{l}', float((sd / lam_s.clamp(min=1e-30).sqrt()).median()), step)        # tau_k = std / s_k: flat if (S)
                            emit(f'spread_snr_d{dec}_l{l}', float((mu.abs() / (sd / math.sqrt(zx.shape[0]) + 1e-30)).median()), step)   # per-batch SNR
                            emit(f'spread_snrk_d{dec}_l{l}', float((math.sqrt(zx.shape[0]) * mu.abs() / lam_s.clamp(min=1e-30).sqrt()).median()), step)   # the Kronecker estimate
                    if self.cres is not None and l < L - 1 and hgrad[l] is not None:
                        # c_ij = sum_x sum_a (q_i)_a^2 (dL/dh_x)_a act''(z_xa) (a~_xj)^2: the mode's second derivative on the residual
                        zt_ = self.cache['zt'][l]; z_ = self.S[l].cols_from_basis(zt_.detach()); dlh = hgrad[l]
                        r_ = dlh * gelu_dd(z_); newc = self.S[l].rows_to_basis_sq(r_.T @ (at * at)); self.cres_n[l] += 1
                        w_ = max(1.0 / self.cres_n[l], 1 - a.beta_stats)
                        if self.cres[l] is None: self.cres[l] = newc.to(torch.bfloat16 if a.compress else torch.float32)
                        else: self.cres[l] = (self.cres[l].float() * (1 - w_) + newc * w_).to(self.cres[l].dtype)
                    if a.ekfac:                              # exact GN diagonal in the current basis: E[d~_i^2 a~_j^2]
                        new = (d * d).T @ (at * at) / at.shape[0]
                        if self.lamx[l] is None: self.lamx[l] = new.to(torch.bfloat16 if a.compress else torch.float32)
                        else: self.lamx[l] = (self.lamx[l].float() * a.beta_stats + new * (1 - a.beta_stats)).to(self.lamx[l].dtype)
            if tm is not None: tm['stats'] += t() - t0
        if a.check_curv and emit is not None and step in (2, a.steps // 2):   # after the first refresh (step 1) the modes are the eigenbasis
            self.check_curv(xb, yb, loss_fn, emit, step, hgrad=hgrad)
        # ---- refresh of the bases (per-layer period; adaptive by the g.D criterion of kronq2 or the overlap criterion of kronsign)
        if due:
            t0 = t() if t else None
            with torch.no_grad():
                for l in due:
                    if self.S[l].F is None or self.A[l].F is None: continue
                    old_ok = self.S[l].e is not None and self.A[l].e is not None
                    if a.adapt_refresh > 0 and old_ok:
                        D_old, _, _ = self.direction(l, Gr[l], self.C[l].float(), lr, update_state=False); gd_old = float((Gr[l] * D_old).sum())
                    self.nref[l] += 1
                    if a.S_every > 1 and self.S[l].U is not None and (self.nref[l] % a.S_every) != 0: VS = None   # S side refreshed every k-th time
                    else: VS = self.S[l].refresh(a.eig_iters)
                    if a.A0_fixed and l == 0 and self.A[0].U is not None and self.nref[0] > a.A0_fixed: VA = None   # T3: the input layer's A basis frozen after a.A0_fixed refreshes
                    else: VA = self.A[l].refresh(a.eig_iters)
                    if emit is not None:                                                        # refresh need per side: mean |diagonal| of the basis change
                        if VA is not None and VA[0] is not None: emit(f'refresh_ovA_l{l}', float(torch.diagonal(VA[0], dim1=1, dim2=2).abs().mean()), step)
                        if VS is not None and VS[0] is not None: emit(f'refresh_ovS_l{l}', float(torch.diagonal(VS[0], dim1=1, dim2=2).abs().mean()), step)
                    tr = lambda T, sq=False: self.A[l].transport(self.S[l].transport(T, VS, 'L', sq), VA, 'R', sq)
                    self.Wt[l].data = tr(self.Wt[l].data); self.C[l] = tr(self.C[l].float()).to(self.C[l].dtype)
                    Gr[l] = tr(Gr[l])
                    if self.u is not None:
                        self.u[l] = tr(self.u[l].float(), True).to(self.u[l].dtype)
                        if a.compress: self.hv[l] = (self.S[l].transport(self.hv[l][0][:, None], VS, 'L', True)[:, 0], self.A[l].transport(self.hv[l][1][None, :], VA, 'R', True)[0])
                        else: self.hv[l] = tr(self.hv[l], True)
                        self.Dprev[l] = None; self.z_tr[l] = None
                    if self.V2 is not None: self.V2[l] = tr(self.V2[l].float(), True).to(self.V2[l].dtype)
                    if self.m2s[l] is not None: self.m2s[l] = tr(self.m2s[l].float(), True).to(self.m2s[l].dtype)
                    if self.Dabs[l] is not None: self.Dabs[l] = tr(self.Dabs[l].float(), True).to(self.Dabs[l].dtype)
                    if self.sc[l] is not None: self.sc[l] = tr(self.sc[l].float(), True).to(self.sc[l].dtype); self.sc2[l] = tr(self.sc2[l].float(), True).to(self.sc2[l].dtype)
                    if self.Dprev2[l] is not None: self.Dprev2[l] = None
                    if self.lamx is not None and self.lamx[l] is not None: self.lamx[l] = tr(self.lamx[l].float(), True).to(self.lamx[l].dtype)
                    if self.cres is not None: self.cres[l] = None; self.cres_n[l] = 0   # (p_j . a)^2 has cross terms under a basis change: restart in the new basis
                    if old_ok and a.adapt_refresh > 0:              # kronq2: relative change of the first-order decrease g.D
                        D_new, _, _ = self.direction(l, Gr[l], self.C[l].float(), lr, update_state=False); gd_new = float((Gr[l] * D_new).sum())
                        e = abs(gd_new - gd_old) / max(abs(gd_new), 1e-300)
                        self.period[l] = int(min(max(self.period[l] * min(max(a.adapt_refresh / max(e, 1e-12), 0.5), 2.0), a.refresh_min), 1000))
                        if emit is not None: emit(f'refresh_err_l{l}', e, step); emit(f'refresh_period_l{l}', self.period[l], step)
                    elif old_ok and a.eig_adapt:                    # kronsign: double the period while the basis overlap stays above 0.99
                        ov = float(torch.diagonal(VA[0], dim1=1, dim2=2).abs().mean()) if VA is not None else 1.0
                        if VS is not None: ov = 0.5 * (ov + float(torch.diagonal(VS[0], dim1=1, dim2=2).abs().mean()))
                        if ov > 0.99: self.period[l] = min(2 * self.period[l], a.eig_max)
                        elif ov < 0.9: self.period[l] = max(self.period[l] // 2, 1)
                        if emit is not None: emit(f'refresh_period_l{l}', self.period[l], step); emit(f'refresh_overlap_l{l}', ov, step)
                    self.next_ref[l] = step + self.period[l]
            if tm is not None: tm['eig'] += t() - t0
        # ---- per-layer step
        t0 = t() if t else None
        with torch.no_grad():
            Dws = []
            for l in range(L):
                g = Gr[l]; C = self.C[l]
                if self.use_fused and self.S[l].e is not None and self.A[l].e is not None:
                    self._lr_eta_t.fill_(float(lr * self.eta[l]))
                    D, lam, live = fused_update(g, C, self.u[l], self.hv[l][0], self.hv[l][1], self.Dprev[l], self.S[l].e, self.A[l].e, self._lr_eta_t,
                                                2.0 / self.nout, a.b1, a.beta_u, a.u_clamp, a.reltol, self.capL)
                    Cf = C.float(); self.Dprev[l] = D.to(self.u[l].dtype); Dws.append(D)
                    if diag_now:
                        emit(f'livefrac_l{l}', float(live.float().mean()), step); emit(f'wnorm_l{l}', float(self.Wt[l].norm()), step)
                        emit(f'u_q50_l{l}', float(self.u[l].float().median()), step)
                    continue
                if a.mom_adapt:                               # per-mode window from the mode's own signal-to-noise (user, 2026-10-13)
                    V2 = self.V2[l]; v = V2.float() * a.b2 + (g * g) * (1 - a.b2); self.V2[l] = v.to(V2.dtype)
                    # de-biased signal estimate: an EMA with window T carries noise variance sigma^2 / (2T - 1), so
                    # m^2 = (C^2 (2T - 1) - v) / (2T - 2) with T the window used so far (else a short window locks itself in)
                    Tprev = self.Tk_log[l] if self.Tk_log[l] is not None else torch.full_like(g, 1.0 / (1 - a.b1))
                    C2 = C.float() ** 2; m2_inst = (C2 * (2 * Tprev - 1) - v) / (2 * Tprev - 2)        # unbiased, very noisy per step
                    if self.m2s[l] is None: self.m2s[l] = m2_inst.clamp(min=0).to(C.dtype)
                    else: self.m2s[l] = (self.m2s[l].float() * a.mom_snr_beta + m2_inst * (1 - a.mom_snr_beta)).to(C.dtype)
                    m2 = self.m2s[l].float().clamp(min=0)
                    T_noise = (v - m2).clamp(min=0) / (m2 + 1e-30)                                   # batches needed for SNR 1 on this mode
                    # staleness, online: the mode's gradient changes by |d_k| lam_k / |g_k| per step (step over completion length)
                    # in the per-mode model, and the layer's basis is valid for ~ its adaptive refresh period
                    if self.Dabs[l] is not None and self.S[l].e is not None and self.A[l].e is not None:
                        lam_ = self.S[l].e[:, None] * self.A[l].e[None, :] * (2.0 / self.nout)
                        rho = (self.Dabs[l].float() * lr * self.eta[l] * lam_ / (g.abs() + 1e-30)).clamp(min=1e-6)
                        T_stale = torch.minimum(1.0 / rho, torch.full_like(rho, float(max(2, 2 * self.period[l]))))
                    else: T_stale = torch.full_like(g, float(max(2, 2 * self.period[l])))
                    Tk = torch.minimum(T_noise, T_stale).clamp(a.mom_T_min, a.mom_T_max)
                    bk = 1 - 1.0 / Tk; C.copy_((C.float() * bk + g * (1 - bk)).to(C.dtype)); self.Tk_log[l] = Tk
                else: C.mul_(a.b1).add_(g.to(C.dtype), alpha=1 - a.b1)
                Cf = C.float()
                if a.hyper and a.hyper_rule == 'signcorr' and self.Dprev[l] is not None:  # sign-correlation rule (ACF item 24): the smoothed
                    sp = torch.sign(self.Dprev[l].float()); sg = torch.sign(g)              # lag-1 and lag-2 sign correlations: 0 for white
                    if self.sc[l] is None: self.sc[l] = torch.zeros_like(g, dtype=self.u[l].dtype); self.sc2[l] = torch.zeros_like(g, dtype=self.u[l].dtype)
                    c_ = self.sc[l].float() * a.sc_beta + (sp * sg) * (1 - a.sc_beta); self.sc[l] = c_.to(self.sc[l].dtype)   # noise, > 0 persistent,
                    if self.Dprev2[l] is not None:                                                                             # < 0 at lag 2 for the
                        c2 = self.sc2[l].float() * a.sc_beta + (torch.sign(self.Dprev2[l].float()) * sg) * (1 - a.sc_beta)    # period-4 oscillation
                        self.sc2[l] = c2.to(self.sc2[l].dtype)
                    else: c2 = torch.zeros_like(g)
                    self.u[l] = (self.u[l].float() + a.sc_rate * (c_ + c2)).clamp(-a.u_clamp, a.u_clamp).to(self.u[l].dtype)
                elif a.hyper and (a.rule != 'adam' or a.adam_fair) and self.Dprev[l] is not None:        # kronq3h one-step / traced hypergradient on u
                    Dp = self.Dprev[l].float() if not a.dprev_bits else self.Dprev[l].float() * torch.exp(self.u[l].float())
                    if a.horizon > 1:
                        gam = 1 - 1.0 / a.horizon
                        self.z_tr[l] = Dp if self.z_tr[l] is None else self.z_tr[l] * gam + Dp
                        Dp = self.z_tr[l]
                    hg = -lr * self.eta[l] * g * Dp
                    if a.compress:
                        r_, c_ = self.hv[l]; r_.mul_(0.99).add_((hg * hg).mean(1), alpha=0.01); c_.mul_(0.99).add_((hg * hg).mean(0), alpha=0.01)
                        den = (r_[:, None] * c_[None, :] / (r_.mean() + 1e-30)).sqrt()
                    else:
                        self.hv[l].mul_(0.99).add_(hg * hg, alpha=0.01); den = self.hv[l].sqrt()
                    un = self.u[l].float() - a.beta_u * hg / (den + 1e-30)
                    if a.hyper_shape: un = un - un.mean()
                    self.u[l] = un.clamp(-a.u_clamp, a.u_clamp).to(self.u[l].dtype)
                D, lam, live = self.direction(l, g, Cf, lr)
                if a.mom_adapt: self.Dabs[l] = D.abs().to(torch.bfloat16 if a.compress else torch.float32)
                if a.hyper and a.hyper_rule == 'signcorr': self.Dprev2[l] = self.Dprev[l]
                if a.hyper and (a.rule != 'adam' or a.adam_fair): self.Dprev[l] = (torch.sign(g).to(torch.int8) if a.hyper_rule == 'signcorr' else (D.to(self.u[l].dtype) if not a.dprev_bits else torch.sign(D).to(torch.int8)))
                if a.gnls:                                   # exact-GN line search along D on the batch
                    at = self.cache['at'][l].detach(); dz = self.cache['zt'][l]
                    sx = (gzt[l].detach() * (at @ D.T)).sum(1); q = float((sx * sx).mean()) * (2.0 / self.nout); gD = float((g * D).sum())
                    alpha = min(1.0, gD / (lr * self.eta[l] * q + 1e-30)) if gD > 0 else 1.0; D = alpha * D
                    if diag_now: emit(f'gnls_alpha_l{l}', alpha, step); emit(f'gnls_qratio_l{l}', q / (float((lam * D * D).sum()) + 1e-30), step)
                if diag_now:
                    emit(f'livefrac_l{l}', float(live.float().mean()), step); emit(f'wnorm_l{l}', float(self.Wt[l].norm()), step)
                    if self.cres is not None and self.cres[l] is not None:
                        cur_ = lam + self.cres[l].float(); conc = cur_ <= 0; emit(f'frac_concave_l{l}', float(conc.float().mean()), step)
                        emit(f'c_over_lam_q50_l{l}', float((self.cres[l].float().abs() / lam.clamp(min=1e-30)).median()), step)
                        if a.hyper and conc.any() and (~conc).any():
                            emit(f'u_q50_concave_l{l}', float(self.u[l].float()[conc].median()), step); emit(f'u_q50_convex_l{l}', float(self.u[l].float()[~conc].median()), step)
                    if a.alloc2: emit(f'alloc2_mu_over_lmax_l{l}', self.last_gamma[l], step)
                    if a.alloc and self.last_gamma[l] is not None: emit(f'alloc_gamma_over_lmax_l{l}', self.last_gamma[l], step); emit(f'alloc_fitfrac_l{l}', float((lam > self.last_gamma[l] * float(lam.max())).float().mean()), step)
                    if a.radius_scan and step in self.scan_steps and l < L - 1:
                        lmax_ = float(lam.max()); top = torch.zeros_like(g); ij = torch.argmax(g.abs()); top.view(-1)[ij] = 1.0
                        rs_dirs = {'step': D, 'grad': g, 'sign': torch.sign(g), 'ridge2': g / (lam + 1e-2 * lmax_), 'ridge4': g / (lam + 1e-4 * lmax_), 'rand': torch.randn_like(g), 'mode': top}
                        order = torch.argsort(lam.flatten(), descending=True); nm = order.numel()
                        for dec in range(10):                        # one mode per lambda decile (the mode with the largest |g| in the decile)
                            seg = order[dec * nm // 10:(dec + 1) * nm // 10]; k = seg[torch.argmax(g.flatten()[seg].abs())]
                            md = torch.zeros_like(g); md.view(-1)[k] = 1.0; rs_dirs[f'moded{dec}'] = md
                            if a.hyper: emit(f'u_q50_d{dec}_l{l}', float(self.u[l].float().flatten()[seg].median()), step)   # learned weight per decile
                            if self.A[l].e is not None and self.S[l].e is not None:
                                ii_, jj_ = int(k) // g.shape[1], int(k) % g.shape[1]
                                emit(f'alpha_d{dec}_l{l}', float(self.A[l].e[jj_]), step); emit(f'sigma_d{dec}_l{l}', float(self.S[l].e[ii_]), step)
                        if self.A[l].e is not None:                       # one mode per ALPHA decile (input-side eigenvalue), largest |g| in the decile
                            aord = torch.argsort(self.A[l].e, descending=True); na = aord.numel()
                            for dec in range(10):
                                cols = aord[dec * na // 10:(dec + 1) * na // 10]; sub = g[:, cols].abs(); kk = int(torch.argmax(sub))
                                ii_, jj_ = kk // sub.shape[1], int(cols[kk % sub.shape[1]])
                                md = torch.zeros_like(g); md[ii_, jj_] = 1.0; rs_dirs[f'amoded{dec}'] = md
                                emit(f'aalpha_d{dec}_l{l}', float(self.A[l].e[jj_]), step); emit(f'asigma_d{dec}_l{l}', float(self.S[l].e[ii_]), step)
                            emit(f'lam_q50_d{dec}_l{l}', float(lam.flatten()[seg].median()), step)
                        self._scan_yb = yb; self.radius_scan(xb, l, rs_dirs, float(lr * self.eta[l] * D.norm()), emit, step)
                    if l == (1 if L > 2 else 0):               # shape of the Taylor radius: remainder and leverage along five directions at the step's length
                        lmax_ = float(lam.max()); dirs = {'step': D, 'grad': g, 'rand': torch.randn_like(g),
                                                          'ridge2': g / (lam + 1e-2 * lmax_), 'ridge4': g / (lam + 1e-4 * lmax_)}
                        nrm = float(lr * self.eta[l] * D.norm()); W_now = self.Wt[l].data
                        with exact_fp32():
                            f0 = self.forward_with(xb, l, W_now)
                            for name, Dir in dirs.items():
                                d = -Dir * (nrm / (float(Dir.norm()) + 1e-30))
                                _, Jd = jvp(lambda Wl: self.forward_with(xb, l, Wl), (W_now,), (d,))
                                f1 = self.forward_with(xb, l, W_now + d)
                                emit(f'shape_rf_{name}_l{l}', float((f1 - f0 - Jd).norm() / (Jd.norm() + 1e-30)), step)
                                emit(f'shape_lev_{name}_l{l}', float(Jd.norm() / (d.norm() + 1e-30)), step)
                    if a.hyper: emit(f'u_q50_l{l}', float(self.u[l].float().median()), step)
                    # per-decile profile of the actual step (--diag): learned weight, |D| (units of lr eta), |D| / w, and the share of the
                    # step's Kronecker output energy sum lam D^2 per decile (where the budget is spent)
                    order = torch.argsort(lam.flatten(), descending=True); n_ = order.numel(); Df = D.flatten(); lamf = lam.flatten()
                    tot = float((lamf * Df * Df).sum()) + 1e-30; uf = self.u[l].float().flatten() if a.hyper else None
                    for dec in range(10):
                        seg = order[dec * n_ // 10:(dec + 1) * n_ // 10]; Dd = Df[seg].abs()
                        emit(f'prof_Dabs_d{dec}_l{l}', float(Dd.median()), step); emit(f'prof_eshare_d{dec}_l{l}', float((lamf[seg] * Df[seg] ** 2).sum()) / tot, step)
                        if uf is not None:
                            emit(f'prof_w_d{dec}_l{l}', float(torch.exp(uf[seg]).median()), step); emit(f'prof_Dw_d{dec}_l{l}', float((Dd / (torch.exp(uf[seg]) + 1e-30)).median()), step)
                    if a.hyper and a.hyper_rule == 'signcorr' and self.sc[l] is not None:
                        order = torch.argsort(lam.flatten(), descending=True); n_ = order.numel(); scf = self.sc[l].float().flatten(); uf = self.u[l].float().flatten()
                        for dec in range(10):
                            seg = order[dec * n_ // 10:(dec + 1) * n_ // 10]; emit(f'sc_q50_d{dec}_l{l}', float(scf[seg].median()), step); emit(f'u_q50_d{dec}_l{l}', float(uf[seg].median()), step)
                            if self.sc2[l] is not None: emit(f'sc2_q50_d{dec}_l{l}', float(self.sc2[l].float().flatten()[seg].median()), step)
                    if a.mom_adapt and self.Tk_log[l] is not None:
                        Tk = self.Tk_log[l].flatten(); order = torch.argsort(lam.flatten(), descending=True); n_ = order.numel()
                        emit(f'Tk_q50_l{l}', float(Tk.median()), step)
                        for dec in range(10): emit(f'Tk_q50_d{dec}_l{l}', float(Tk[order[dec * n_ // 10:(dec + 1) * n_ // 10]].median()), step)
                Dws.append(D)
            if getattr(a, 'track', 0) and diag_now and step % a.track_every == 0:
                if self.tracker is None:
                    from tracking import Tracker; self.tracker = Tracker(self)
                t1 = t() if t else None; self.tracker.run(xb, emit, step, Dws)
                if tm is not None: tm['stats'] += t() - t1
            # ---- step length controllers
            g_all = 1.0
            if a.rem > 0 or self.kappa > 0:
                f_before = out.detach()
            for l in range(L): self.Wt[l].data.sub_(lr * self.eta[l] * Dws[l])
            if a.zbud:                                       # per-layer pre-activation budget (TRUST_REGION: the layer's radius is a sphere in z-space)
                for l in range(L - 1):
                    at = self.cache['at'][l].detach(); zt = self.cache['zt'][l].detach()
                    dz = (lr * self.eta[l]) * (at @ Dws[l].T); rz = float(dz.norm() / (zt.norm() + 1e-30))
                    zb = self.zbud[min(l, len(self.zbud) - 1)]
                    if rz > zb: self.Wt[l].data.add_((1.0 - zb / rz) * lr * self.eta[l] * Dws[l])
                    if diag_now: emit(f'zbud_rho_l{l}', rz, step)
            rem_now = self.rem_dyn if self.rem_dyn is not None else a.rem          # rem_dyn: set by --rem_val_track or by slow_adapt.SlowAdapt
            if a.rem > 0:
                f_after = self.forward(xb); rho = float((f_after - f_before).norm() / ((f_before - yb).norm() + 1e-30))
                self.last_rho = rho                                        # before the budget acts (slow_adapt measurement)
                if rho > rem_now:
                    sc = 1.0 - rem_now / rho
                    for l in range(L): self.Wt[l].data.add_(sc * lr * self.eta[l] * Dws[l])
                if diag_now: emit('rem_rho', rho, step); emit('rem_now', rem_now, step)
            if (a.rem_val > 0 or a.ls_val or (a.rem_val_track > 0 and step % a.rem_val_every == 0)) and self.xv_fresh is not None:   # REPORT 61 item 30: the budget on a FRESH batch (the cross-kernel effect, Cor. 24a)
                xv_, yv_ = self.xv_fresh; sc_b = (1.0 - rem_now / rho) if (a.rem > 0 and rho > rem_now) else 0.0   # fraction already undone by the batch budget
                steps_taken = [(1.0 - sc_b) * lr * self.eta[l] * Dws[l] for l in range(L)]                   # the step as applied so far
                f1v = self.forward(xv_)
                for l in range(L): self.Wt[l].data.add_(steps_taken[l])                                       # back to the pre-step point
                f0v = self.forward(xv_)
                for l in range(L): self.Wt[l].data.sub_(steps_taken[l])
                rho_v = float((f1v - f0v).norm() / ((f0v - yv_).norm() + 1e-30))
                if a.rem_val_track > 0:                        # the budget tracks the held-out break-even: rem_t = m x 2 c_t, c_t = EMA of the alignment
                    dfv = f1v - f0v; c_now = float(-((f0v - yv_) * dfv).sum() / ((f0v - yv_).norm() * dfv.norm() + 1e-30))
                    self.c_ema = c_now if self.c_ema is None else 0.9 * self.c_ema + 0.1 * c_now
                    target = max(0.02, min(0.6, a.rem_val_track * 2.0 * self.c_ema)); self.rem_dyn = target   # applied to the batch output change every step
                    if rho_v > target:
                        scv = 1.0 - target / rho_v
                        for l in range(L): self.Wt[l].data.add_(scv * steps_taken[l])
                    if diag_now: emit('remtrack_target', target, step); emit('remtrack_c', self.c_ema, step)
                elif a.ls_val:                                 # REPORT 61 item 31: exact line search on the fresh batch, alpha = <E_v, -df> / ||df||^2
                    dfv = f1v - f0v; alpha = float(-((f0v - yv_) * dfv).sum() / ((dfv * dfv).sum() + 1e-30)); alpha = min(max(alpha, 0.0), 1.0)
                    for l in range(L): self.Wt[l].data.add_((1.0 - alpha) * steps_taken[l])
                    if diag_now: emit('lsv_alpha', alpha, step); emit('lsv_cos', float(-((f0v - yv_) * dfv).sum() / ((f0v - yv_).norm() * dfv.norm() + 1e-30)), step)
                elif rho_v > a.rem_val:
                    scv = 1.0 - a.rem_val / rho_v
                    for l in range(L): self.Wt[l].data.add_(scv * steps_taken[l])
                if diag_now: emit('remv_rho', rho_v, step)
            if self.kappa > 0 and step % a.check_every == 0:   # per-layer eta from the function-space remainder (round-robin)
                l = (step // a.check_every) % L; d = -lr * self.eta[l] * Dws[l]
                W_now = self.Wt[l].data.clone(); W_pre = W_now - d                           # the pre-step point
                with exact_fp32():                                                           # TF32 noise would pin eta at its floor
                    f0 = self.forward_with(xb, l, W_pre)
                    _, Jd = jvp(lambda Wl: self.forward_with(xb, l, Wl), (W_pre,), (d,))
                    f1 = self.forward_with(xb, l, W_now)
                lin = float(Jd.norm())
                if lin > 0:
                    rf = float((f1 - f0 - Jd).norm()) / lin; self.rf[l] = rf if self.rf[l] is None else 0.7 * self.rf[l] + 0.3 * rf
                    if lr > 0.05 * a.lr and l < L - 1:                       # kronq2: hidden layers only; the output layer is linear in its weights
                        if self.rf[l] > 1.5 * self.kappa: self.eta[l] *= 0.7
                        elif self.rf[l] < self.kappa / 1.5: self.eta[l] *= 1.25
                        self.eta[l] = min(max(self.eta[l], 1e-3), 1e3)
                    if l == L - 2: self.eta[L - 1] = self.eta[L - 2]            # output layer follows the last hidden layer
                    if emit is not None and (diag_now or step % a.eval_every < a.check_every): emit(f'rf_l{l}', self.rf[l], step); emit(f'eta_l{l}', self.eta[l], step)
        if tm is not None: tm['upd'] += t() - t0
