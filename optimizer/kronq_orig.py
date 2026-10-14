"""KronQ with the weights in ORIGINAL coordinates (the refactor, 2026-10-14).

KronQFast stores the weights in the rotating Kronecker basis: every refresh re-expresses them (and the basis is the product of every
rotation since the start), so rounding in those products compounds in the weights and the basis drifts; the forward pass also applies
the two basis rotations to the activations in every layer. Here:
  - fp32 master weights W (out x in+1, bias column last) in ordinary coordinates; the network uses them directly (one product per layer);
  - curvature statistics (blocks of the input covariance A and the probe output-gradient covariance S) in ordinary coordinates;
  - at a refresh the basis is computed from the statistics (subspace iteration warm-started from the previous basis), never accumulated;
    only the optimizer state (momentum, learned log-weights, hypergradient normaliser) is moved to the new basis; the weights are not
    touched, so a refresh cannot change the function;
  - each step: rotate the gradient into the basis, run the unchanged fused KronQ update there, rotate the update back, apply it,
    then decoupled decay and the output budget exactly as KronQFast does.
In exact arithmetic the update equals KronQFast's. Only the reference configuration is supported (sign rule, learned weights, compressed
normaliser, MC probe); any other option raises instead of being ignored.
"""
import math, torch, contextlib

@contextlib.contextmanager
def full_fp32(on):
    """matmuls inside run in full fp32 (no TF32) when on; restores the previous setting."""
    if not on: yield; return
    prev = torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32 = False
    try: yield
    finally: torch.backends.cuda.matmul.allow_tf32 = prev
import kronq_fast as kf
from kronq_fast import Basis, fused_update


class OBasis(Basis):
    """Block statistics and eigenbasis in ORIGINAL coordinates; refresh() recomputes the basis from the statistics."""
    def _orth(self, Y):
        return kf.cholqr(Y) if kf.CHOLQR else torch.linalg.qr(Y)[0]

    def refresh(self, iters=2):
        """new basis by `iters` subspace iterations on F warm-started from the current basis; returns the change (V, Vr) = U_old^T U_new
        (relative to the identity on the first call) for transporting the optimizer state."""
        if self.F is None: return None
        it = max(1, iters)
        U = self.U if self.U is not None else torch.eye(self.b, device=self.dev).expand(self.nf, self.b, self.b).clone()
        for _ in range(it): U = self._orth(torch.bmm(self.F, U))
        e = torch.einsum('kbc,kbd,kdc->kc', U, self.F, U).clamp(min=0).reshape(-1)
        V = U if self.U is None else torch.bmm(self.U.transpose(1, 2), U)
        Ur = Vr = None
        if self.rem:
            Ur = self.Ur if self.Ur is not None else torch.eye(self.rem, device=self.dev)
            for _ in range(it): Ur = self._orth(self.Fr @ Ur)
            e = torch.cat([e, torch.diagonal(Ur.T @ self.Fr @ Ur).clamp(min=0)])
            Vr = Ur if self.Ur is None else self.Ur.T @ Ur
        Ud = U.double(); self.sd_Uacc = float((torch.bmm(Ud.transpose(1, 2), Ud) - torch.eye(self.b, device=self.dev, dtype=torch.float64)).abs().max())
        self.U, self.Ur, self.e = U, Ur, e
        return (V, Vr)


class KronQOrig:
    UNSUPPORTED = ('xfold', 'wd_gate', 'hyper_budget', 'lr_track', 'alloc', 'alloc2', 'mom_adapt', 'ekfac', 'sphere', 'gnls', 'track',
                   'acf', 'zbud', 'rem_val', 'ls_val', 'rem_val_track', 'S_identity', 'A_identity', 'freeze_after', 'rem_ctrl', 'lr_kappa',
                   'radius_scan', 'check_curv', 'alloc2_oneshot', 'curv', 'topk', 'q', 'qa', 'q_in', 'sign_floor', 'dprev_bits', 'hyper_shape',
                   'clip_grad', 'A0_fixed')

    def __init__(self, lins, a, dev, nout, L_act, draw_eps):
        bad = [k for k in self.UNSUPPORTED if getattr(a, k, 0)]
        if a.rule != 'sign' or not a.hyper or not a.compress or a.wrule != 'learned' or a.S_mode != 'mc' or getattr(a, 'horizon', 1) > 1 \
                or a.hyper_rule != 'idbd' or a.last_S != 'stats' or a.S_every != 1 or a.eig_adapt:
            bad.append('non-reference rule / statistics option')
        if bad: raise SystemExit(f'--orig 1 supports the reference KronQ configuration only; unsupported here: {bad}')
        self.a, self.dev, self.nout, self.L, self.act, self.draw_eps = a, dev, nout, len(lins), L_act, draw_eps
        self.Wt = [torch.nn.Parameter(torch.cat([lin.weight.detach(), lin.bias.detach()[:, None]], 1).clone()) for lin in lins]  # ORIGINAL coordinates
        self.A = [OBasis(lin.weight.shape[1] + 1, a.block, a.max_dim, dev) for lin in lins]
        self.S = [OBasis(lin.weight.shape[0], a.block, a.max_dim, dev) for lin in lins]
        self.C = [torch.zeros_like(W, dtype=torch.bfloat16 if a.mom_bf16 else torch.float32) for W in self.Wt]
        self.u = [torch.zeros_like(W, dtype=torch.bfloat16) for W in self.Wt]
        self.hv = [(torch.zeros(W.shape[0], device=dev), torch.zeros(W.shape[1], device=dev)) for W in self.Wt]
        self.Dprev = [None] * self.L
        self.capL = (float(self.L) if a.cap_L else 1.0) if a.cap else 0.0
        self.period = [a.refresh] * self.L; self.next_ref = [1] * self.L; self.nref = [0] * self.L
        self._lr_eta_t = torch.zeros((), device=dev)
        self.E0_norm = None; self.kept_prev = 1.0; self.last_rho = None; self.last_rho_real = float('nan'); self.rem_dyn = None
        self.cache = {}; self.xv_fresh = None; self.eta = [1.0] * self.L
        kf.QR_GUARD = bool(a.qr_guard); kf.QR_TOL = a.qr_tol; kf.CHOLQR = bool(a.cholqr)
        hp = set(x for x in a.orig_hp.split(',') if x); self.hp_rot = 'rot' in hp; self.hp_stats = 'stats' in hp; self.hp_refresh = 'refresh' in hp

    def parameters(self): return self.Wt

    # ---------------- network in ordinary coordinates
    def forward(self, x, keep=False):
        h = x; a_list = []; z_list = []
        for l in range(self.L):
            a_aug = torch.cat([h, torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype)], 1)
            z = a_aug @ self.Wt[l].T
            if keep: a_list.append(a_aug.detach()); z_list.append(z)
            h = self.act(z) if l < self.L - 1 else z
        if keep: self.cache = dict(a=a_list, z=z_list)
        return h

    @torch.no_grad()
    def preacts(self, x):
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
        s = sum(n(x) for x in self.C) + sum(n(x) for x in self.u) + sum(n(x) for x in self.Dprev) + sum(n(r) + n(c) for r, c in self.hv)   # Dprev counted as in KronQFast
        return s + sum(B.state_bytes() for B in self.A + self.S)

    # ---------------- basis helpers
    def to_basis(self, l, M):                                                    # U_S^T M U_A
        with full_fp32(self.hp_rot): return self.A[l].cols_to_basis(self.S[l].rows_to_basis(M))
    def from_basis(self, l, M):                                                  # U_S M U_A^T
        with full_fp32(self.hp_rot): return self.S[l].rows_from_basis(self.A[l].cols_from_basis(M))

    def _dir(self, l, g, lr):
        """the push direction (units of lr) for the adaptive-refresh criterion, as KronQFast.direction in the reference configuration."""
        a = self.a
        if self.S[l].e is None or self.A[l].e is None: lam = torch.ones_like(g); lmax = 1.0
        else: lam = self.S[l].e[:, None] * self.A[l].e[None, :] * (2.0 / self.nout); lmax = float(lam.max())
        D = torch.where(lam > a.reltol * lmax, torch.exp(self.u[l].float()) * torch.sign(self.C[l].float()), torch.zeros_like(g))
        if a.cap:
            capv = g.abs() / (lam.clamp(min=1e-30) * max(lr, 1e-30) * self.capL); D = torch.sign(D) * torch.minimum(D.abs(), capv)
        return D

    # ---------------- one step
    def step(self, step, lr, out, xb, yb, loss_fn, tm=None, t=None, emit=None):
        a = self.a; L = self.L
        need_stats = (step - 1) % max(1, a.stats_every) == 0
        diag_now = emit is not None and (step % a.eval_every == 0 or step == a.steps)
        G = [W.grad for W in self.Wt]
        due = [l for l in range(L) if step >= self.next_ref[l]]
        sd_now = a.spike_diag > 0 and step >= a.spike_diag and getattr(self, 'sd_x', None) is not None
        # statistics in ordinary coordinates (input-side from the activations, output-side from the MC probe)
        if need_stats:
            t0 = t() if t else None
            with torch.enable_grad():
                eps = self.draw_eps(out.shape)
                gz = torch.autograd.grad((out * eps).sum(), self.cache['z'])
            with full_fp32(self.hp_stats):
                for l in range(L):
                    self.A[l].update(self.cache['a'][l], a.beta_stats); self.S[l].update(gz[l].detach(), a.beta_stats)
            if tm is not None: tm['stats'] += t() - t0
        with torch.no_grad():
            Gr = [self.to_basis(l, G[l]) for l in range(L)]
        # refresh: new bases from the statistics; only the optimizer state moves
        if due:
            t0 = t() if t else None
            with torch.no_grad():
                sd_pre = self.forward(self.sd_x[0]) if sd_now else None
                for l in due:
                    if self.S[l].F is None or self.A[l].F is None: continue
                    old_ok = self.S[l].e is not None and self.A[l].e is not None
                    if a.adapt_refresh > 0 and old_ok: gd_old = float((Gr[l] * self._dir(l, Gr[l], lr)).sum())
                    self.nref[l] += 1
                    with full_fp32(self.hp_refresh): VS = self.S[l].refresh(a.eig_iters); VA = self.A[l].refresh(a.eig_iters)
                    tr = lambda T, sq=False: self.A[l].transport(self.S[l].transport(T, VS, 'L', sq), VA, 'R', sq)
                    self.C[l] = tr(self.C[l].float()).to(self.C[l].dtype)
                    self.u[l] = tr(self.u[l].float(), True).to(self.u[l].dtype)
                    self.hv[l] = (self.S[l].transport(self.hv[l][0][:, None], VS, 'L', True)[:, 0], self.A[l].transport(self.hv[l][1][None, :], VA, 'R', True)[0])
                    self.Dprev[l] = None
                    Gr[l] = self.to_basis(l, G[l])                         # recomputed in the new basis (not transported)
                    if old_ok and a.adapt_refresh > 0:
                        gd_new = float((Gr[l] * self._dir(l, Gr[l], lr)).sum()); e_ = abs(gd_new - gd_old) / max(abs(gd_new), 1e-300)
                        self.period[l] = int(min(max(self.period[l] * min(max(a.adapt_refresh / max(e_, 1e-12), 0.5), 2.0), a.refresh_min), 1000))
                        if emit is not None and diag_now: emit(f'refresh_period_l{l}', self.period[l], step)
                    self.next_ref[l] = step + self.period[l]
                if sd_pre is not None:
                    sd_post = self.forward(self.sd_x[0]); den = float((sd_pre - self.sd_x[1]).norm()) + 1e-30
                    self.sd_last = dict(refresh_df=float((sd_post - sd_pre).norm()) / den, nref=float(len(due)),
                                        Uacc=max(max(getattr(self.S[l], 'sd_Uacc', 0.0), getattr(self.A[l], 'sd_Uacc', 0.0)) for l in range(L)))
            if tm is not None: tm['eig'] += t() - t0
        # the KronQ update in the basis, rotated back
        t0 = t() if t else None
        with torch.no_grad():
            Dws = []
            for l in range(L):
                self._lr_eta_t.fill_(float(lr))
                D, lam, live = fused_update(Gr[l], self.C[l], self.u[l], self.hv[l][0], self.hv[l][1], self.Dprev[l], self.S[l].e, self.A[l].e,
                                            self._lr_eta_t, 2.0 / self.nout, a.b1, a.beta_u, a.u_clamp, a.reltol, self.capL, a.u_decay, a.u_floor)
                self.Dprev[l] = D.to(self.u[l].dtype)
                Dws.append(self.from_basis(l, D))
                if diag_now: emit(f'livefrac_l{l}', float(live.float().mean()), step); emit(f'u_q50_l{l}', float(self.u[l].float().median()), step)
            f_before = out.detach()
            for l in range(L): self.Wt[l].data.sub_(lr * Dws[l])
            if a.wd > 0 or a.wd_abs > 0:
                gam_wd = a.wd_abs * (lr / a.lr) if a.wd_abs > 0 else lr * a.wd
                for l in range(L): self.Wt[l].data[:, :-1].mul_(1.0 - gam_wd)
            rem_now = self.rem_dyn if self.rem_dyn is not None else a.rem
            if a.rem > 0:
                En = float((f_before - yb).norm())
                if a.rem_anchor in ('init', 'min'):
                    if self.E0_norm is None: self.E0_norm = En
                    En = self.E0_norm if a.rem_anchor == 'init' else min(En, self.E0_norm)
                f_after = self.forward(xb); rho = float((f_after - f_before).norm() / (En + 1e-30))
                self.last_rho = rho; self.kept_prev = 1.0
                if rho > rem_now:
                    sc = 1.0 - rem_now / rho; self.kept_prev = rem_now / rho
                    for l in range(L): self.Wt[l].data.add_(sc * lr * Dws[l])
                    if a.rem_check or a.rem_iter > 0:
                        kept = rem_now / rho; it = 0
                        while True:
                            f2 = self.forward(xb); rho2 = float((f2 - f_before).norm() / (En + 1e-30)); self.last_rho_real = rho2
                            if it >= a.rem_iter or rho2 <= 1.5 * rem_now: break
                            sc2 = (1.0 - rem_now / rho2) * kept
                            for l in range(L): self.Wt[l].data.add_(sc2 * lr * Dws[l])
                            kept *= rem_now / rho2; it += 1; self.kept_prev = kept
                if diag_now: emit('rem_rho', rho, step); emit('rem_now', rem_now, step)
        if tm is not None: tm['upd'] += t() - t0
