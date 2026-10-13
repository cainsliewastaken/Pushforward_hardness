"""Krylov (matrix-free ridge) step on the batch (2026-10-13; COLLECTIVE_RULE.md section 4).
The one-step optimum of Thm stepsup is the ridge step d = -J^T (K + gamma)^-1 E, K = J J^T on the batch; Thm stepkrylov
says its restriction to the Krylov space span{E, K E, ..., K^(M-1) E} is M-1 conjugate-gradient iterations. Each CG
iteration applies K once: one vjp (J^T p, a backward with p as the output cotangent) and one jvp (J v). No factor
statistics, no eigendecomposition, no blocks; every interaction between parameter directions is in K exactly.
Length: the output-space trust region of the scan (REPORT 61 item 15): the linear output change ||K x|| is scaled to at
most rem * ||E|| (rem ~ 0.2 = the measured radius on the rough target), times the decay. Momentum (heavy ball on the
scaled step) is the minibatch averaging. Cost per step ~ (M + 1) forward-backward. State: momentum buffer (1 x P)."""
import math, torch
from torch.func import jvp, vjp

class KrylovStep:
    def __init__(self, lins, a, dev, act):
        self.a, self.dev, self.act = a, dev, act; self.L = len(lins)
        self.params = [p for lin in lins for p in (lin.weight, lin.bias)]
        self.mom = [torch.zeros_like(p) for p in self.params] if a.kr_mom > 0 else None
        self.gamma_rel = a.kr_gamma
    def fwd(self, ps, x):
        h = x
        for l in range(self.L):
            z = h @ ps[2 * l].T + ps[2 * l + 1]; h = self.act(z) if l < self.L - 1 else z
        return h
    def state_bytes(self):
        return 0 if self.mom is None else sum(m.numel() * m.element_size() for m in self.mom)
    @torch.no_grad()
    def step(self, step, lr_fac, xb, yb, emit=None, diag=False):
        a = self.a; ps = tuple(p.detach() for p in self.params); M = a.kr_M
        with torch.enable_grad():
            out, vjp_fn = vjp(lambda *q: self.fwd(q, xb), *ps)
        E = (out - yb).detach(); nE = float(E.norm())
        Kop = lambda p: jvp(lambda *q: self.fwd(q, xb), ps, vjp_fn(p))[1]          # K p = J (J^T p)
        # Rayleigh scale for the damping: gamma = gamma_rel * (E^T K E / E^T E)
        KE = Kop(E); ray = float((E * KE).sum() / (nE ** 2 + 1e-30)); gamma = self.gamma_rel * ray
        # CG on (K + gamma) x = E from x = 0: x stays in the Krylov space; accumulate K x alongside
        x = torch.zeros_like(E); Kx = torch.zeros_like(E); r = E.clone(); p = r.clone(); rr = float((r * r).sum()); Kp = KE
        for it in range(M):
            Ap = Kp + gamma * p; alpha = rr / (float((p * Ap).sum()) + 1e-30)
            x = x + alpha * p; Kx = Kx + alpha * Kp; r = r - alpha * Ap
            rr_new = float((r * r).sum())
            if it + 1 < M:
                beta = rr_new / (rr + 1e-30); p = r + beta * p; Kp = Kop(p)
            rr = rr_new
        # parameter step d = -J^T x; output change (linear) = -K x; trust region ||K x|| <= rem ||E|| (times the decay)
        d = vjp_fn(x); nKx = float(Kx.norm()); cap = a.rem * nE * lr_fac
        sc = min(1.0, cap / (nKx + 1e-30))
        cosE = float((E * Kx).sum() / (nE * nKx + 1e-30))
        upds = [-sc * di for di in d]
        if self.mom is not None:                       # heavy ball on the scaled step, then the trust region on the accumulated update (one jvp)
            for i in range(len(upds)): self.mom[i].mul_(a.kr_mom).add_(upds[i])
            with torch.enable_grad(): Jm = jvp(lambda *q: self.fwd(q, xb), ps, tuple(self.mom))[1]
            scm = min(1.0, cap / (float(Jm.norm()) + 1e-30)); upds = [scm * m_ for m_ in self.mom]
            if emit is not None and diag: emit('kr_mom_scale', scm, step)
        for pi, upd in zip(self.params, upds): pi.add_(upd)
        if emit is not None and diag:
            emit('kr_cosE', cosE, step); emit('kr_scale', sc, step); emit('kr_Kx_over_E', nKx / (nE + 1e-30), step)
            emit('kr_cg_res', math.sqrt(rr) / (nE + 1e-30), step); emit('kr_gamma_over_ray', self.gamma_rel, step)
            emit('kr_ray', ray, step); emit('kr_dnorm', float(math.sqrt(sum(float((di * di).sum()) for di in d))) * sc, step)
