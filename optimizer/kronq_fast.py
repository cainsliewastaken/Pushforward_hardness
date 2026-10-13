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
            for _ in range(iters): V, _ = torch.linalg.qr(torch.bmm(self.F, V))
            e = torch.einsum('kbc,kbd,kdc->kc', V, self.F, V).clamp(min=0).reshape(-1)
            if self.rem:
                Vr = torch.eye(self.rem, device=self.dev)
                for _ in range(iters): Vr, _ = torch.linalg.qr(self.Fr @ Vr)
                e = torch.cat([e, torch.diagonal(Vr.T @ self.Fr @ Vr).clamp(min=0)])
            else: Vr = None
        # new basis U V; factor becomes (nearly) diagonal in it
        if self.U is None:
            self.U = V; self.Ur = Vr
        else:
            self.U = torch.bmm(self.U, V)
            if self.rem: self.Ur = self.Ur @ Vr
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
        self.A = [Basis(lin.weight.shape[1] + 1, a.block, a.max_dim, dev) for lin in lins]
        self.S = [Basis(lin.weight.shape[0], a.block, (0 if (a.last_S == 'identity' and l == self.L - 1) else a.max_dim), dev) for l, lin in enumerate(lins)]
        bf = torch.bfloat16 if a.compress else torch.float32
        self.C = [torch.zeros_like(W, dtype=torch.bfloat16 if a.mom_bf16 else torch.float32) for W in self.Wt]
        self.u = [torch.zeros_like(W, dtype=bf) for W in self.Wt] if a.hyper else None
        self.hv = [torch.zeros_like(W, dtype=torch.float32) if not a.compress else (torch.zeros(W.shape[0], device=dev), torch.zeros(W.shape[1], device=dev)) for W in self.Wt] if a.hyper else None
        self.Dprev = [None] * self.L; self.z_tr = [None] * self.L   # eligibility traces (--horizon > 1)
        self.V2 = [torch.zeros_like(W, dtype=bf) for W in self.Wt] if a.rule == 'adam' else None
        self.lamx = [None] * self.L if a.ekfac else None
        self.eta = [1.0] * self.L; self.refresh_cur = a.refresh; self.emp_scale = [None] * self.L
        self.period = [a.refresh] * self.L; self.next_ref = [1] * self.L; self.rf = [None] * self.L
        global EIG_DTYPE; EIG_DTYPE = torch.float64 if a.eig_dtype == 'float64' else None
        self.cache = {}; self.last_Gr = None; self.diag_out = {}
        self.f0 = None
    def parameters(self): return self.Wt
    # ---------------- forward in the rotated parametrisation
    def forward(self, x, keep=False):
        h = x; at_list = []; zt_list = []
        for l in range(self.L):
            a_aug = torch.cat([h, torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype)], 1)
            at = self.A[l].cols_to_basis(a_aug)                  # a~ = U_A^T [a;1]  (rows are samples)
            zt = at @ self.Wt[l].T                                 # z~ = Wt a~
            if keep: zt.retain_grad(); at_list.append(at); zt_list.append(zt)
            z = self.S[l].cols_from_basis(zt)                      # z = U_S z~
            h = self.act(z) if l < self.L - 1 else z
        if keep: self.cache = dict(at=at_list, zt=zt_list)
        return h
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
        if a.rule == 'adam':
            V2 = self.V2[l]; V2f = V2.float() * a.b2 + (g * g) * (1 - a.b2)
            if update_state: self.V2[l] = V2f.to(V2.dtype)
            bc1 = 1 - a.b1 ** self.t_step; bc2 = 1 - a.b2 ** self.t_step
            return (Cf / bc1) / ((V2f / bc2).sqrt() + a.eps), lam, live
        w = torch.ones_like(g)
        if a.q != 0: w = w * (lam / lmax).clamp(min=1e-300) ** (0.5 * a.q)
        if a.hyper: w = w * torch.exp(self.u[l].float())
        sgn = torch.sign(Cf) if a.sign_floor <= 0 else Cf / (Cf.abs() + a.sign_floor * float(Cf.abs().max()) + 1e-30)
        D = torch.where(live, w * sgn, torch.zeros_like(g))
        if a.cap:
            capv = g.abs() / (lam.clamp(min=1e-30) * max(lr * self.eta[l], 1e-30) * L)
            D = torch.sign(D) * torch.minimum(D.abs(), capv)
        return D, lam, live
    # ---------------- one optimizer step; call after loss.backward() on forward(xb, keep=need_keep)
    def step(self, step, lr, out, xb, yb, loss_fn, tm=None, t=None, emit=None):
        a = self.a; L = self.L; dev = self.dev
        need_stats = (step - 1) % max(1, a.stats_every) == 0
        self.t_step = step
        due = [l for l in range(L) if step >= self.next_ref[l]]
        Gr = [W.grad for W in self.Wt]
        diag_now = emit is not None and (step % a.eval_every == 0 or step == a.steps)
        # ---- statistics in the current basis
        if need_stats or a.gnls:
            t0 = t() if t else None
            with torch.enable_grad():
                if a.S_mode == 'mc' or a.gnls or (a.S_mode == 'emp' and need_stats and any(sc is None for sc in self.emp_scale)):
                    eps = self.draw_eps(out.shape)
                    gzt = torch.autograd.grad((out * eps).sum(), self.cache['zt'], retain_graph=True)
                if a.S_mode == 'emp':
                    gzt_e = [zt.grad for zt in self.cache['zt']]
            if need_stats:
                for l in range(L):
                    at = self.cache['at'][l].detach()
                    if a.S_mode == 'emp':
                        d = gzt_e[l].detach()
                        if self.emp_scale[l] is None:       # rescale the empirical signal to the GN trace once (from the probe)
                            self.emp_scale[l] = float(gzt[l].detach().pow(2).sum().sqrt() / (d.pow(2).sum().sqrt() + 1e-30))
                        d = d * self.emp_scale[l]
                    else: d = gzt[l].detach()
                    self.A[l].update(at, a.beta_stats)
                    if not self.S[l].diag: self.S[l].update(d, a.beta_stats)
                    else: self.S[l].update(d, a.beta_stats)
                    if a.ekfac:                              # exact GN diagonal in the current basis: E[d~_i^2 a~_j^2]
                        new = (d * d).T @ (at * at) / at.shape[0]
                        if self.lamx[l] is None: self.lamx[l] = new.to(torch.bfloat16 if a.compress else torch.float32)
                        else: self.lamx[l] = (self.lamx[l].float() * a.beta_stats + new * (1 - a.beta_stats)).to(self.lamx[l].dtype)
            if tm is not None: tm['stats'] += t() - t0
        # ---- refresh of the bases (per-layer period; adaptive by the g.D criterion of kronq2 or the overlap criterion of kronsign)
        if due:
            t0 = t() if t else None
            with torch.no_grad():
                for l in due:
                    if self.S[l].F is None or self.A[l].F is None: continue
                    old_ok = self.S[l].e is not None and self.A[l].e is not None
                    if a.adapt_refresh > 0 and old_ok:
                        D_old, _, _ = self.direction(l, Gr[l], self.C[l].float(), lr, update_state=False); gd_old = float((Gr[l] * D_old).sum())
                    VS = self.S[l].refresh(a.eig_iters); VA = self.A[l].refresh(a.eig_iters)
                    tr = lambda T, sq=False: self.A[l].transport(self.S[l].transport(T, VS, 'L', sq), VA, 'R', sq)
                    self.Wt[l].data = tr(self.Wt[l].data); self.C[l] = tr(self.C[l].float()).to(self.C[l].dtype)
                    Gr[l] = tr(Gr[l])
                    if self.u is not None:
                        self.u[l] = tr(self.u[l].float(), True).to(self.u[l].dtype)
                        if a.compress: self.hv[l] = (self.S[l].transport(self.hv[l][0][:, None], VS, 'L', True)[:, 0], self.A[l].transport(self.hv[l][1][None, :], VA, 'R', True)[0])
                        else: self.hv[l] = tr(self.hv[l], True)
                        self.Dprev[l] = None; self.z_tr[l] = None
                    if self.V2 is not None: self.V2[l] = tr(self.V2[l].float(), True).to(self.V2[l].dtype)
                    if self.lamx is not None and self.lamx[l] is not None: self.lamx[l] = tr(self.lamx[l].float(), True).to(self.lamx[l].dtype)
                    if old_ok and a.adapt_refresh > 0:              # kronq2: relative change of the first-order decrease g.D
                        D_new, _, _ = self.direction(l, Gr[l], self.C[l].float(), lr, update_state=False); gd_new = float((Gr[l] * D_new).sum())
                        e = abs(gd_new - gd_old) / max(abs(gd_new), 1e-300)
                        self.period[l] = int(min(max(self.period[l] * min(max(a.adapt_refresh / max(e, 1e-12), 0.5), 2.0), 1), 1000))
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
                g = Gr[l]; C = self.C[l]; C.mul_(a.b1).add_(g.to(C.dtype), alpha=1 - a.b1); Cf = C.float()
                if a.hyper and a.rule != 'adam' and self.Dprev[l] is not None:        # kronq3h one-step / traced hypergradient on u
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
                if a.hyper and a.rule != 'adam': self.Dprev[l] = (D.to(self.u[l].dtype) if not a.dprev_bits else torch.sign(D).to(torch.int8))
                if a.gnls:                                   # exact-GN line search along D on the batch
                    at = self.cache['at'][l].detach(); dz = self.cache['zt'][l]
                    sx = (gzt[l].detach() * (at @ D.T)).sum(1); q = float((sx * sx).mean()) * (2.0 / self.nout); gD = float((g * D).sum())
                    alpha = min(1.0, gD / (lr * self.eta[l] * q + 1e-30)) if gD > 0 else 1.0; D = alpha * D
                    if diag_now: emit(f'gnls_alpha_l{l}', alpha, step); emit(f'gnls_qratio_l{l}', q / (float((lam * D * D).sum()) + 1e-30), step)
                if diag_now:
                    emit(f'livefrac_l{l}', float(live.float().mean()), step); emit(f'wnorm_l{l}', float(self.Wt[l].norm()), step)
                    if a.hyper: emit(f'u_q50_l{l}', float(self.u[l].float().median()), step)
                Dws.append(D)
            # ---- step length controllers
            g_all = 1.0
            if a.rem > 0 or a.lr_kappa > 0:
                f_before = out.detach()
            for l in range(L): self.Wt[l].data.sub_(lr * self.eta[l] * Dws[l])
            if a.rem > 0:
                f_after = self.forward(xb); rho = float((f_after - f_before).norm() / ((f_before - yb).norm() + 1e-30))
                if rho > a.rem:
                    sc = 1.0 - a.rem / rho
                    for l in range(L): self.Wt[l].data.add_(sc * lr * self.eta[l] * Dws[l])
                if diag_now: emit('rem_rho', rho, step)
            if a.lr_kappa > 0 and step % a.check_every == 0:   # per-layer eta from the function-space remainder (round-robin)
                l = (step // a.check_every) % L; d = -lr * self.eta[l] * Dws[l]
                W_now = self.Wt[l].data.clone(); W_pre = W_now - d                           # the pre-step point
                f0 = self.forward_with(xb, l, W_pre)
                _, Jd = jvp(lambda Wl: self.forward_with(xb, l, Wl), (W_pre,), (d,))
                f1 = self.forward_with(xb, l, W_now)
                lin = float(Jd.norm())
                if lin > 0:
                    rf = float((f1 - f0 - Jd).norm()) / lin; self.rf[l] = rf if self.rf[l] is None else 0.7 * self.rf[l] + 0.3 * rf
                    if lr > 0.05 * a.lr and l < L - 1:                       # kronq2: hidden layers only; the output layer is linear in its weights
                        if self.rf[l] > 1.5 * a.lr_kappa: self.eta[l] *= 0.7
                        elif self.rf[l] < a.lr_kappa / 1.5: self.eta[l] *= 1.25
                        self.eta[l] = min(max(self.eta[l], 1e-3), 1e3)
                    if l == L - 2: self.eta[L - 1] = self.eta[L - 2]            # output layer follows the last hidden layer
                    if emit is not None and (diag_now or step % a.eval_every < a.check_every): emit(f'rf_l{l}', self.rf[l], step); emit(f'eta_l{l}', self.eta[l], step)
        if tm is not None: tm['upd'] += t() - t0
