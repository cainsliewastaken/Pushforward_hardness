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
"""
import math, torch
from kronq_main import KronQ, _update_c, _fp32

MODES = ('probe', 'fused', 'emp', 'gradfac', 'aonly')


def _gelu_d(z):
    """derivative of the exact (erf) gelu."""
    return 0.5 * (1.0 + torch.erf(z * 0.7071067811865476)) + z * torch.exp(-0.5 * z * z) * 0.3989422804014327


class KronQX(KronQ):
    def __init__(self, *args, mode='probe', **kw):
        assert mode in MODES, mode
        super().__init__(*args, **kw)
        self.mode = mode; self._gz = None
        if mode == 'fused': assert self.act is torch.nn.functional.gelu, 'fused backward is written for gelu'

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
        B = out.shape[0]; st = self.stats_now()
        d = 2.0 * (out - y) / out.numel()
        if st: d = torch.cat([d, self.draw_probe(out.shape)], 0)
        a, z = self.cache['a'], self.cache['z']; gz = [None] * self.L
        for l in range(self.L - 1, -1, -1):
            W = self.Wt[l]
            g = d[:B].T @ a[l]
            if W.grad is None: W.grad = g
            else: W.grad.copy_(g)
            if st: gz[l] = d[B:]
            if l > 0:
                gd = _gelu_d(z[l - 1]); d = (d @ W[:, :-1]) * (torch.cat([gd, gd], 0) if st else gd)
        self._gz = gz if st else None

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
                self.next_ref[l] = t + self.period[l]
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
