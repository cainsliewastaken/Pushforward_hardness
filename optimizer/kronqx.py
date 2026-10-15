"""KronQX: TEST VARIANTS of the main optimizer's curvature statistics (2026-10-15). Does not edit kronq_main.py: subclasses KronQ and
replaces only how the two Kronecker factors are obtained; the push (momentum, learned multiplier x sign, hypergradient, clamp), the
refresh, the transport, the numerics and the decay are KronQ's own (step() below is kronq_main.KronQ.step with the statistics part
swapped and, for 'aonly', the output side left in ordinary coordinates).

mode
  probe     reference: KronQ exactly (random +-1 probe, one extra backward to the layer outputs)
  fused     same statistics as probe, but the loss and probe backward run as ONE hand-written backward over 2B stacked rows (the
            MLP is plain: z = [a, 1] W^T, gelu); the forward keeps no autograd graph. Call fused_backward(out, y) instead of
            loss.backward(). Same arithmetic; tests the per-step overhead
  emp       empirical Fisher: the output-side factor from the loss's own per-sample output gradients (z.grad from the ordinary
            backward, no extra backward); the input side as KronQ
  gradfac   Shampoo / SOAP factors: L = EMA(G G^T) (output side), R = EMA(G^T G) (input side) from the batch-mean gradient
            G = [dW | db]; no activation statistics, no probe
  aonly     input side only: A from the activations, no output-side basis (rows stay in ordinary coordinates, unit output
            factor in the live mask); no probe
  severy    A every statistics step, the probe (and S) only every s_every steps
  kfra      no probe: S from the top down (KFRA, Botev, Ritter & Barber 2017): S_top = I (the probe's expectation for squared
            loss), S_l = (W_{l+1}^T S_{l+1} W_{l+1}) o E[b b^T], b = gelu'(z_l), with E[b b^T] ~ bbar bbar^T + diag(var b) from
            EMAs of the per-neuron mean and mean square of b (forward only); full matrices, recomputed every kfra_every steps,
            their diagonal blocks set as the S factors (KronQ's refresh then works on them as usual)
probe_rows (fused only): the probe runs on the first probe_rows rows of the batch (0 = all)
kdrift_every: every k steps, the EXACT empirical kernel on a fixed probe set (kd_x, kd_y; m samples) for nr fixed random
            output directions r: K = sum_r sum_l (a_l a_l^T) o (d_l^r d_l^r^T), d_l^r the per-sample backprop of r^T f to z_l (exact
            for an MLP: the per-sample gradient of layer l is d (x) [a, 1]). Logged per measurement (kdlog): step, steps since the last
            one, sum of lr over them, ||dK||_F / ||K_prev||_F, the off-diagonal share of V_prev^T dK V_prev (rotation in the previous
            eigenframe), the subspace distance of the top-8 / 32 / 128 eigenvectors (1 - ||V1^T V2||_F^2 / k), lambda_1, and the
            share of the residual energy (sum_r (V^T e_r)^2) beyond rank 8 / 32 / 128 (where the residual sits in the spectrum)
switch: (t_switch, period): from step t_switch on, every layer's refresh period is `period` (fixed; the switched-schedule test)
log_refresh: at every basis refresh record, per layer and side, the off-diagonal share of the current factor in the OLD basis,
            sqrt(1 - ||diag(U_old^T F U_old)||^2 / ||U_old^T F U_old||^2) over all blocks (the factor-level form of the frame
            rotation, THEORY.md sec. 18), the steps since the last refresh and the period chosen; rlog = [t, layer, side, off, dt, period]
"""
import math, torch
from kronq_main import KronQ, _update_c, _fp32

MODES = ('probe', 'fused', 'emp', 'gradfac', 'aonly', 'severy', 'kfra')


def _gelu_d(z):
    """derivative of the exact (erf) gelu."""
    return 0.5 * (1.0 + torch.erf(z * 0.7071067811865476)) + z * torch.exp(-0.5 * z * z) * 0.3989422804014327


class KronQX(KronQ):
    def __init__(self, *args, mode='probe', s_every=4, probe_rows=0, kfra_every=10, **kw):
        assert mode in MODES, mode
        super().__init__(*args, **kw)
        self.mode = mode; self._gz = None; self.s_every, self.probe_rows, self.kfra_every = s_every, probe_rows, kfra_every
        if mode in ('fused', 'kfra'): assert self.act is torch.nn.functional.gelu, 'written for gelu'
        self.bm = [None] * self.L; self.bq = [None] * self.L          # kfra: EMA of the per-neuron mean and mean square of gelu'(z)
        self.log_refresh = False; self.rlog = []; self._last = [0] * self.L; self.switch = None
        self.kdrift_every = 0; self.kd_x = self.kd_y = self.kd_r = None; self.kdlog = []; self._kd_prev = None; self._lrsum = 0.0

    @staticmethod
    def _offshare(B):
        """off-diagonal share of B's current factor in B's current (old) basis; None before the first basis."""
        if B.U is None or B.F is None: return None
        M = torch.bmm(torch.bmm(B.U.transpose(1, 2), B.F), B.U); tot = float((M * M).sum()); dg = float((M.diagonal(dim1=1, dim2=2) ** 2).sum())
        if B.rem and B.Ur is not None:
            Mr = B.Ur.T @ B.Fr @ B.Ur; tot += float((Mr * Mr).sum()); dg += float((Mr.diagonal() ** 2).sum())
        return math.sqrt(max(1.0 - dg / max(tot, 1e-300), 0.0))

    def forward(self, x, keep=False):
        if self.mode == 'fused' and keep:
            with torch.no_grad(): return super().forward(x, keep=True)
        out = super().forward(x, keep)
        if keep and self.mode == 'emp':
            for z in self.cache['z']: z.retain_grad()
        return out

    def stats_now(self):
        """is the coming step a statistics step (same rule as KronQ.step)."""
        return self.t % max(1, self.h['stats_every']) == 0

    @torch.no_grad()
    def fused_backward(self, out, y):
        """mean-squared-error gradients of every [W | b] and, on statistics steps, the probe's output-side gradients, in one backward
        over stacked rows [loss; probe]. Sets Wt[l].grad and self._gz."""
        B = out.shape[0]; st = self.stats_now(); k = self.probe_rows if 0 < self.probe_rows < B else B
        d = 2.0 * (out - y) / out.numel()
        if st: d = torch.cat([d, self.draw_probe((k, out.shape[1]))], 0)
        a, z = self.cache['a'], self.cache['z']; gz = [None] * self.L
        for l in range(self.L - 1, -1, -1):
            W = self.Wt[l]
            g = d[:B].T @ a[l]
            if W.grad is None: W.grad = g
            else: W.grad.copy_(g)
            if st: gz[l] = d[B:]
            if l > 0:
                gd = _gelu_d(z[l - 1]); d = (d @ W[:, :-1]) * (torch.cat([gd, gd[:k]], 0) if st else gd)
        self._gz = gz if st else None

    @torch.no_grad()
    def _kfra(self):
        """S factors of every layer from the top down (module docstring)."""
        with _fp32():
            n = self.Wt[-1].shape[0]; S = torch.eye(n, device=self.dev)
            for l in range(self.L - 1, -1, -1):
                B_ = self.S[l]; nb = B_.nf * B_.b
                B_.F = torch.stack([S[i * B_.b:(i + 1) * B_.b, i * B_.b:(i + 1) * B_.b] for i in range(B_.nf)]).clone()
                B_.Fr = S[nb:, nb:].clone() if B_.rem else None
                if l == 0: break
                W = self.Wt[l][:, :-1].float(); M = W.T @ S @ W
                bm, bq = self.bm[l - 1], self.bq[l - 1]
                S = M * (bm[:, None] * bm[None, :]); S.diagonal().add_(M.diagonal() * (bq - bm * bm).clamp(min=0))

    @torch.no_grad()
    def kernel(self, x, y, R):
        """exact empirical kernel (m x m) of the projected outputs R (nr x nout) on x, eigenpairs (descending), residual energy per mode."""
        with _fp32():
            a, z, h = [], [], x
            for l in range(self.L):
                at = torch.cat([h, torch.ones(h.shape[0], 1, device=h.device)], 1); a.append(at); zz = at @ self.Wt[l].T; z.append(zz)
                h = self.act(zz) if l < self.L - 1 else zz
            AG = [ai @ ai.T for ai in a]; K = torch.zeros(x.shape[0], x.shape[0], device=x.device, dtype=torch.float64)
            E = (h - y) @ R.T                                               # residual along each direction (m x nr)
            for r in R:
                d = r[None, :].expand(x.shape[0], -1)
                for l in range(self.L - 1, -1, -1):
                    K += (AG[l] * (d @ d.T)).double()
                    if l > 0: d = (d @ self.Wt[l][:, :-1]) * _gelu_d(z[l - 1])
            ev, V = torch.linalg.eigh(K); ev, V = ev.flip(0), V.flip(1)
            c2 = ((V.T @ E.double()) ** 2).sum(1)
        return K, ev, V, c2

    def _kdrift(self, t):
        K, ev, V, c2 = self.kernel(self.kd_x, self.kd_y, self.kd_r)
        tot = float(c2.sum()) + 1e-300; front = [round(float(c2[k:].sum()) / tot, 5) for k in (8, 32, 128)]
        if self._kd_prev is not None:
            Kp, Vp, tp, lp = self._kd_prev; dK = K - Kp; M = Vp.T @ dK @ Vp; m2 = float((M * M).sum()) + 1e-300
            off = math.sqrt(max(1.0 - float((M.diagonal() ** 2).sum()) / m2, 0.0))
            sub = [round(1.0 - float(((Vp[:, :k].T @ V[:, :k]) ** 2).sum()) / k, 6) for k in (8, 32, 128)]
            self.kdlog.append([t, t - tp, round(self._lrsum - lp, 8), round(float(dK.norm() / Kp.norm()), 6), round(off, 5)] + sub + [float(ev[0])] + front)
        else:
            self.kdlog.append([t, 0, 0.0, None, None, None, None, None, float(ev[0])] + front)
        self._kd_prev = (K, V, t, self._lrsum)

    def _lam(self, l):
        Se = self.S[l].e if self.mode != 'aonly' else torch.ones(self.Wt[l].shape[0], device=self.dev)
        return Se, self.A[l].e

    def _push(self, l, g):
        Se, Ae = self._lam(l)
        if Se is None or Ae is None: lam = torch.ones_like(g)
        else: lam = Se[:, None] * Ae[None, :]
        return torch.where(lam > self.h['reltol'] * float(lam.max()), torch.exp(self.u[l].float()) * torch.sign(self.C[l].float()), torch.zeros_like(g))

    def step(self, lr, out):
        h = self.h; L = self.L; self.t += 1; t = self.t
        G = [W.grad for W in self.Wt]
        if (t - 1) % max(1, h['stats_every']) == 0:                       # statistics
            gz = None
            if self.mode == 'probe':
                with torch.enable_grad():
                    gz = torch.autograd.grad((out * self.draw_probe(out.shape)).sum(), self.cache['z'])
            elif self.mode == 'fused': gz = self._gz
            elif self.mode == 'emp': gz = [z.grad * z.shape[0] for z in self.cache['z']]
            elif self.mode == 'severy' and (t - 1) % max(1, self.s_every) == 0:
                with torch.enable_grad():
                    gz = torch.autograd.grad((out * self.draw_probe(out.shape)).sum(), self.cache['z'])
            elif self.mode == 'kfra':
                with torch.no_grad():
                    for l in range(L - 1):
                        b = _gelu_d(self.cache['z'][l].detach().float()); m_, q_ = b.mean(0), (b * b).mean(0)
                        if self.bm[l] is None: self.bm[l], self.bq[l] = m_, q_
                        else: self.bm[l].mul_(h['beta_stats']).add_(m_, alpha=1 - h['beta_stats']); self.bq[l].mul_(h['beta_stats']).add_(q_, alpha=1 - h['beta_stats'])
                if (t - 1) % max(1, self.kfra_every) == 0: self._kfra()
            with torch.no_grad(), _fp32():
                for l in range(L):
                    if self.mode == 'gradfac':
                        Gf = G[l].float(); self.A[l].update(Gf, h['beta_stats']); self.S[l].update(Gf.T.contiguous(), h['beta_stats'])
                        continue
                    self.A[l].update(self.cache['a'][l].float(), h['beta_stats'])
                    if gz is not None: self.S[l].update(gz[l].detach().float(), h['beta_stats'])
        with torch.no_grad():
            with _fp32(): Gr = [self._to(l, G[l]) for l in range(L)]
            due = [l for l in range(L) if t >= self.next_ref[l]]
            for l in due:
                if self.A[l].F is None or (self.mode != 'aonly' and self.S[l].F is None): continue
                Se0, Ae0 = self._lam(l); old_ok = Se0 is not None and Ae0 is not None
                if self.log_refresh:
                    with _fp32():
                        offs = [('A', self._offshare(self.A[l]))] + ([('S', self._offshare(self.S[l]))] if self.mode != 'aonly' else [])
                if old_ok: gd_old = float((Gr[l] * self._push(l, Gr[l])).sum())
                with _fp32():
                    VA = self.A[l].refresh(h['eig_iters'])
                    if self.mode != 'aonly':
                        VS = self.S[l].refresh(h['eig_iters'])
                        tr = lambda T, sq=False: self.A[l].transport(self.S[l].transport(T, VS, 'L', sq), VA, 'R', sq)
                        self.hv[l] = (self.S[l].transport(self.hv[l][0][:, None], VS, 'L', True)[:, 0], self.A[l].transport(self.hv[l][1][None, :], VA, 'R', True)[0])
                    else:
                        tr = lambda T, sq=False: self.A[l].transport(T, VA, 'R', sq)
                        self.hv[l] = (self.hv[l][0], self.A[l].transport(self.hv[l][1][None, :], VA, 'R', True)[0])
                    self.C[l] = tr(self.C[l].float()).to(self.C[l].dtype)
                    self.u[l] = tr(self.u[l].float(), True).to(self.u[l].dtype)
                    Gr[l] = self._to(l, G[l])
                self.has_prev[l] = False
                if old_ok and h['adapt_refresh'] > 0:
                    gd_new = float((Gr[l] * self._push(l, Gr[l])).sum()); e_ = abs(gd_new - gd_old) / max(abs(gd_new), 1e-300)
                    if math.isfinite(e_):
                        self.period[l] = int(min(max(self.period[l] * min(max(h['adapt_refresh'] / max(e_, 1e-12), 0.5), 2.0), h['refresh_min']), 1000))
                if self.log_refresh:
                    for side, o in offs:
                        if o is not None: self.rlog.append([t, l, side, round(o, 6), t - self._last[l], self.period[l]])
                    self._last[l] = t
                if self.switch is not None and t >= self.switch[0]: self.period[l] = self.switch[1]
                self.next_ref[l] = t + self.period[l]
            if self.switch is not None and t == self.switch[0]:
                for l in range(L): self.period[l] = self.switch[1]; self.next_ref[l] = min(self.next_ref[l], t + self.switch[1])
            self._lr_t.fill_(float(lr))
            for l in range(L):
                Se, Ae = self._lam(l)
                D = _update_c(Gr[l], self.C[l], self.u[l], self.hv[l][0], self.hv[l][1], self.has_prev[l], Se, Ae,
                              self._lr_t, 2.0 / self.nout, h['b1'], h['beta_u'], h['u_clamp'], h['reltol'])
                self.has_prev[l] = True
                with _fp32(): self.Wt[l].data.sub_(lr * self._from(l, D))
            if h['weight_decay'] > 0:
                gam = h['weight_decay'] * lr / self.lr_max
                for l in range(L): self.Wt[l].data[:, :-1].mul_(1.0 - gam)
            self._lrsum += float(lr)
            if self.kdrift_every and (t == 1 or t % self.kdrift_every == 0): self._kdrift(t)
