"""KronQ: THE MAIN OPTIMIZER (2026-10-14). Self-contained; does not import the research code.

The method, and nothing else (OPTIMIZER_PLAN.md section 2b):
  - per layer, the Kronecker eigenbasis of the batch-averaged input covariance A = avg([a;1][a;1]^T) and probe output-gradient
    covariance S = avg(delta delta^T), in diagonal blocks of `block`, statistics every step (EMA beta_stats);
  - the weights are fp32 master weights in ORDINARY coordinates (the network is a plain MLP: one product per layer);
  - each step the gradient is rotated into the basis, there: momentum (b1), push = learned per-mode weight x sign(momentum), the learned
    log-weight updated by the one-step hypergradient with a rank-one normaliser (memory 100), clamped at +-u_clamp; the push is rotated
    back and applied with the scheduled learning rate; the previous push the hypergradient needs is rebuilt from the momentum and the
    log-weight before they are updated, not stored (optimizer state: momentum + log-weight, bf16 each, plus the factors);
  - optional decoupled weight decay, given as the decay per step at full lr (`weight_decay`; it follows the lr schedule) - use it on
    finite / replayed data;
  - bases refreshed by `eig_iters` subspace iterations warm-started from the previous basis, on a per-layer adaptive period (relative change
    of the first-order decrease, target `adapt_refresh`, floor `refresh_min`); only the optimizer state is moved to a new basis;
  - numerics: the statistics, the refresh and the gradient / update rotations in full fp32 (the network's own products may run in reduced
    precision); every Cholesky-QR rotation checked for orthonormality and redone by Householder QR in float64 when it fails.
NOT in it (tested and dropped; see REPORT / SPIKE_RESULT / XFOLD_RESULT): output budget, per-mode cap, slow adaptation, cross-fitting,
weight shrink, gated decay, tracking lr, budget-aware hypergradient, mean-sign shrinkage, rotated-weight parametrisation.
"""
import contextlib, math, torch


@contextlib.contextmanager
def _fp32():
    """matrix products inside run in full fp32 (no TF32)."""
    prev = torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32 = False
    try: yield
    finally: torch.backends.cuda.matmul.allow_tf32 = prev


def _cholqr(Y, passes=3, tol=1e-5):
    """orthonormal columns of Y (batched): CholeskyQR in float64, three passes; any block not orthonormal to `tol` (the jitter lets an
    ill-conditioned Gram pass the Cholesky) is redone by Householder QR in float64."""
    V = Y
    for _ in range(passes):
        G = V.transpose(-1, -2).double() @ V.double()
        G = G + 1e-12 * G.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
        L, info = torch.linalg.cholesky_ex(G)
        if int(info.max()) != 0: return torch.linalg.qr(Y.double())[0].to(Y.dtype)
        V = torch.linalg.solve_triangular(L, V.double().transpose(-1, -2), upper=False).transpose(-1, -2).to(Y.dtype)
    Vd = V.double(); err = (Vd.transpose(-1, -2) @ Vd - torch.eye(V.shape[-1], device=V.device, dtype=torch.float64)).abs().amax(dim=(-2, -1))
    bad = err > tol
    if bool(bad.any()):
        if V.dim() == 2: return torch.linalg.qr(Y.double())[0].to(Y.dtype)
        V = V.clone(); V[bad] = torch.linalg.qr(Y[bad].double())[0].to(Y.dtype)
    return V


def _update(g, C, u, r_, c_, has_prev, Se, Ae, lr_t, lam_scale, b1, beta_u, u_clamp, reltol):
    """one layer in the basis: momentum, hypergradient step on the log-weights, push. C, u, r_, c_ updated in place; returns D.
    The previous push D_{t-1} = exp(u_{t-1}) sign(C_{t-1}) on the live modes is not stored: on entry C and u still hold C_{t-1}
    and u_{t-1}, and the live mask changes only at a refresh, after which has_prev is False (2026-10-14; it used to be kept in
    bf16, 2 bytes per parameter; the hypergradient now uses exp(u) unrounded, a difference at bf16 rounding)."""
    lam = Se[:, None] * Ae[None, :] * lam_scale; live = lam > reltol * lam.max()
    if has_prev:
        hg = -(lr_t * g) * torch.where(live, torch.exp(u.float()) * torch.sign(C.float()), torch.zeros_like(g)); hg2 = hg * hg
    C.mul_(b1).add_(g.to(C.dtype), alpha=1 - b1); Cf = C.float()
    if has_prev:
        r_.mul_(0.99).add_(hg2.mean(1), alpha=0.01); c_.mul_(0.99).add_(hg2.mean(0), alpha=0.01)
        den = (r_[:, None] * c_[None, :] / (r_.mean() + 1e-30)).sqrt()
        u.copy_((u.float() - beta_u * hg / (den + 1e-30)).clamp(-u_clamp, u_clamp).to(u.dtype))
    return torch.where(live, torch.exp(u.float()) * torch.sign(Cf), torch.zeros_like(Cf))
try:
    _update_c = torch.compile(_update, dynamic=False)
except Exception:
    _update_c = _update


class _Basis:
    """one side of one layer: block-diagonal statistics and eigenbasis in ordinary coordinates (nf blocks of b, one remainder block)."""
    def __init__(self, dim, block, dev):
        self.dev = dev; b = block if 0 < block < dim else dim
        self.b, self.nf = b, dim // b; self.rem = dim - self.nf * b
        self.F = self.Fr = self.U = self.Ur = self.e = None

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
    def rows_to(self, M): return M if self.U is None else self._apply(M, 'L', self.U, self.Ur, True)      # U^T M
    def rows_from(self, M): return M if self.U is None else self._apply(M, 'L', self.U, self.Ur, False)   # U M
    def cols_to(self, M): return M if self.U is None else self._apply(M, 'R', self.U, self.Ur, True)      # M U
    def cols_from(self, M): return M if self.U is None else self._apply(M, 'R', self.U, self.Ur, False)   # M U^T
    def transport(self, T, V, side, squared=False):
        """T in the old basis -> new basis (V = U_old^T U_new); squared=True moves a second moment by the squared overlaps."""
        Vf, Vr = V
        if squared: Vf, Vr = Vf * Vf, (Vr * Vr if Vr is not None else None)
        return self._apply(T, side, Vf, Vr, True)

    def update(self, Z, beta):
        """add a batch (samples x dim, ordinary coordinates) to the block statistics."""
        m = Z.shape[0]; nb = self.nf * self.b; Zf = Z[:, :nb].reshape(m, self.nf, self.b)
        newF = torch.einsum('mkb,mkc->kbc', Zf, Zf) / m; newR = (Z[:, nb:].T @ Z[:, nb:] / m) if self.rem else None
        if self.F is None: self.F, self.Fr = newF, newR
        else:
            self.F.mul_(beta).add_(newF, alpha=1 - beta)
            if self.rem: self.Fr.mul_(beta).add_(newR, alpha=1 - beta)

    def refresh(self, iters):
        """new basis from the statistics (subspace iteration warm-started from the current basis); returns V = U_old^T U_new."""
        it = max(1, iters)
        U = self.U if self.U is not None else torch.eye(self.b, device=self.dev).expand(self.nf, self.b, self.b).clone()
        for _ in range(it): U = _cholqr(torch.bmm(self.F, U))
        e = torch.einsum('kbc,kbd,kdc->kc', U, self.F, U).clamp(min=0).reshape(-1)
        V = U if self.U is None else torch.bmm(self.U.transpose(1, 2), U)
        Ur = Vr = None
        if self.rem:
            Ur = self.Ur if self.Ur is not None else torch.eye(self.rem, device=self.dev)
            for _ in range(it): Ur = _cholqr(self.Fr @ Ur)
            e = torch.cat([e, torch.diagonal(Ur.T @ self.Fr @ Ur).clamp(min=0)])
            Vr = Ur if self.Ur is None else self.Ur.T @ Ur
        self.U, self.Ur, self.e = U, Ur, e
        return (V, Vr)

    def nbytes(self):
        return sum(0 if x is None else x.numel() * x.element_size() for x in (self.F, self.Fr, self.U, self.Ur))


class KronQ:
    """the main optimizer for an MLP given as a list of nn.Linear (hidden activation `act`, linear output). Usage per step:
        out = opt.forward(x, keep=True); loss = mse(out, y); loss.backward(retain_graph=True); opt.step(lr, out)
    `draw_probe(shape)` returns random +-1 output signs for the curvature probe; `lr_max` is the schedule's peak (for the decay)."""

    def __init__(self, lins, act, nout, draw_probe, lr_max, dev, block=256, b1=0.9, beta_u=0.01, u_clamp=4.6, reltol=1e-12, beta_stats=0.95,
                 eig_iters=2, refresh=20, adapt_refresh=0.1, refresh_min=5, stats_every=1, weight_decay=0.0, mom_bf16=True):
        self.L, self.act, self.nout, self.draw_probe, self.lr_max, self.dev = len(lins), act, nout, draw_probe, lr_max, dev
        self.h = dict(b1=b1, beta_u=beta_u, u_clamp=u_clamp, reltol=reltol, beta_stats=beta_stats, eig_iters=eig_iters,
                      adapt_refresh=adapt_refresh, refresh_min=refresh_min, stats_every=stats_every, weight_decay=weight_decay)
        self.Wt = [torch.nn.Parameter(torch.cat([lin.weight.detach(), lin.bias.detach()[:, None]], 1).clone()) for lin in lins]  # [W | b]
        self.A = [_Basis(lin.weight.shape[1] + 1, block, dev) for lin in lins]
        self.S = [_Basis(lin.weight.shape[0], block, dev) for lin in lins]
        self.C = [torch.zeros_like(W, dtype=torch.bfloat16 if mom_bf16 else torch.float32) for W in self.Wt]
        self.u = [torch.zeros_like(W, dtype=torch.bfloat16) for W in self.Wt]
        self.hv = [(torch.zeros(W.shape[0], device=dev), torch.zeros(W.shape[1], device=dev)) for W in self.Wt]
        self.has_prev = [False] * self.L                                    # the previous push is rebuilt in _update, not stored
        self.period = [refresh] * self.L; self.next_ref = [1] * self.L
        self.t = 0; self._lr_t = torch.zeros((), device=dev); self.cache = {}

    # ---------------- the network (ordinary coordinates)
    def forward(self, x, keep=False):
        h = x; a_list = []; z_list = []
        for l in range(self.L):
            a_aug = torch.cat([h, torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype)], 1)
            z = a_aug @ self.Wt[l].T
            if keep: a_list.append(a_aug.detach()); z_list.append(z)
            h = self.act(z) if l < self.L - 1 else z
        if keep: self.cache = dict(a=a_list, z=z_list)
        return h

    def parameters(self): return self.Wt

    @torch.no_grad()
    def preacts(self, x):
        """hidden-layer pre-activations (diagnostics)."""
        h = x; zs = []
        for l in range(self.L - 1):
            z = torch.cat([h, torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype)], 1) @ self.Wt[l].T; zs.append(z); h = self.act(z)
        return zs

    @torch.no_grad()
    def params_orig(self):
        out = []
        for W in self.Wt: out += [W.data[:, :-1].contiguous(), W.data[:, -1].contiguous()]
        return out

    def state_bytes(self):
        n = lambda x: 0 if x is None else x.numel() * x.element_size()
        return sum(n(x) for x in self.C + self.u) + sum(n(r) + n(c) for r, c in self.hv) + sum(B.nbytes() for B in self.A + self.S)

    # ---------------- helpers
    def _to(self, l, M): return self.A[l].cols_to(self.S[l].rows_to(M))          # U_S^T M U_A
    def _from(self, l, M): return self.S[l].rows_from(self.A[l].cols_from(M))      # U_S M U_A^T
    def _push(self, l, g):
        """the current push direction (units of lr), for the adaptive refresh criterion."""
        if self.S[l].e is None or self.A[l].e is None: lam = torch.ones_like(g)
        else: lam = self.S[l].e[:, None] * self.A[l].e[None, :]
        return torch.where(lam > self.h['reltol'] * float(lam.max()), torch.exp(self.u[l].float()) * torch.sign(self.C[l].float()), torch.zeros_like(g))

    # ---------------- one step (call after loss.backward on the output of forward(x, keep=True))
    def step(self, lr, out):
        h = self.h; L = self.L; self.t += 1; t = self.t
        G = [W.grad for W in self.Wt]
        if (t - 1) % max(1, h['stats_every']) == 0:                       # statistics: activations and the MC curvature probe
            with torch.enable_grad():
                gz = torch.autograd.grad((out * self.draw_probe(out.shape)).sum(), self.cache['z'])
            with torch.no_grad(), _fp32():
                for l in range(L):
                    self.A[l].update(self.cache['a'][l], h['beta_stats']); self.S[l].update(gz[l].detach(), h['beta_stats'])
        with torch.no_grad():
            with _fp32(): Gr = [self._to(l, G[l]) for l in range(L)]
            due = [l for l in range(L) if t >= self.next_ref[l]]
            for l in due:                                                   # refresh: new bases from the statistics; only the state moves
                if self.S[l].F is None or self.A[l].F is None: continue
                old_ok = self.S[l].e is not None and self.A[l].e is not None
                if old_ok: gd_old = float((Gr[l] * self._push(l, Gr[l])).sum())
                with _fp32():
                    VS = self.S[l].refresh(h['eig_iters']); VA = self.A[l].refresh(h['eig_iters'])
                    tr = lambda T, sq=False: self.A[l].transport(self.S[l].transport(T, VS, 'L', sq), VA, 'R', sq)
                    self.C[l] = tr(self.C[l].float()).to(self.C[l].dtype)
                    self.u[l] = tr(self.u[l].float(), True).to(self.u[l].dtype)
                    self.hv[l] = (self.S[l].transport(self.hv[l][0][:, None], VS, 'L', True)[:, 0], self.A[l].transport(self.hv[l][1][None, :], VA, 'R', True)[0])
                    Gr[l] = self._to(l, G[l])
                self.has_prev[l] = False
                if old_ok and h['adapt_refresh'] > 0:
                    gd_new = float((Gr[l] * self._push(l, Gr[l])).sum()); e_ = abs(gd_new - gd_old) / max(abs(gd_new), 1e-300)
                    self.period[l] = int(min(max(self.period[l] * min(max(h['adapt_refresh'] / max(e_, 1e-12), 0.5), 2.0), h['refresh_min']), 1000))
                self.next_ref[l] = t + self.period[l]
            self._lr_t.fill_(float(lr))
            for l in range(L):                                              # the update in the basis, rotated back and applied
                D = _update_c(Gr[l], self.C[l], self.u[l], self.hv[l][0], self.hv[l][1], self.has_prev[l], self.S[l].e, self.A[l].e,
                              self._lr_t, 2.0 / self.nout, h['b1'], h['beta_u'], h['u_clamp'], h['reltol'])
                self.has_prev[l] = True
                with _fp32(): self.Wt[l].data.sub_(lr * self._from(l, D))
            if h['weight_decay'] > 0:                                       # decoupled decay on the weights (not the bias), following the schedule
                gam = h['weight_decay'] * lr / self.lr_max
                for l in range(L): self.Wt[l].data[:, :-1].mul_(1.0 - gam)
