"""KronRoot: eigendecomposition-free Kronecker whitening (Newton-Schulz matrix roots) with Adam in whitened coordinates.

Same Gauss-Newton Kronecker factors as KronSign (A = E[a~ a~^T], S = E[d d^T], statistics machinery inherited), but no
eigenbasis.  Each nn.Linear is reparametrised as

    W = S^{-1/4} W~ A^{-1/4}        (W = [weight | bias], o x i1;  roots of the damped factors, symmetric)

so the forward pass computes z = S^{-1/4} W~ A^{-1/4} x~, autograd returns g~ = S^{-1/4} G A^{-1/4}, and plain gradient
descent on W~ is the preconditioned step  dW = -lr S^{-1/2} G A^{-1/2}  (K-FAC with power -1/2, i.e. G_GN^{-1/2} under the
Kronecker approximation: every Gauss-Newton mode moves at a rate ~ sqrt(lambda)).  Adam runs elementwise on W~.

Roots: coupled Newton-Schulz iteration for M^{1/2}, M^{-1/2} on the damped, normalised factor, applied twice for the
quarter roots.  Only matrix products: no eigh, no QR, no sorting, no degenerate-eigenspace ambiguity.
At a root refresh W~, the momentum and the second moment are re-expressed so that W is unchanged:
    W~' = S'^{1/4} S^{-1/4} W~ A^{-1/4} A'^{1/4};   m' = S'^{-1/4} S^{1/4} m A^{1/4} A'^{-1/4};   v' by squared transforms.
Last layer under squared loss: S = I exactly (last_S='identity'), so only the input side is whitened there.
"""
import torch, time
import torch.nn.functional as Fn
from kronsign import KronSign


def ns_sqrt(M, iters):
    """Coupled Newton-Schulz: returns (M^{1/2}, M^{-1/2}) for SPD M with spectrum in (0, 1]."""
    I = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    Y, Z = M, I
    for _ in range(iters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y, Z = Y @ T, T @ Z
    return Y, Z


def quarter_roots(F, damp, iters):
    """(F^{1/4}, F^{-1/4}) of the damped factor F + damp * mean(diag F) * I, via two Newton-Schulz square roots."""
    d = F.shape[0]
    Fd = F + damp * (torch.trace(F) / d + 1e-30) * torch.eye(d, dtype=F.dtype, device=F.device)
    c = torch.linalg.matrix_norm(Fd, ord='fro')                    # spectrum of Fd / c in (0, 1]
    R, _ = ns_sqrt(Fd / c, iters)                                   # (Fd/c)^{1/2}
    R = 0.5 * (R + R.T)
    c2 = torch.linalg.matrix_norm(R, ord='fro')
    P, Q = ns_sqrt(R / c2, iters)                                   # (R/c2)^{1/2}, (R/c2)^{-1/2}
    s = (c ** 0.25) * (c2 ** 0.5)                                   # undo both normalisations
    P = 0.5 * (P + P.T) * s
    Q = 0.5 * (Q + Q.T) / s
    return P, Q


class KronRoot(KronSign):
    def __init__(self, model, lr, b1=0.9, b2=0.999, eps=1e-8, beta_stats=0.95, stats_every=10, root_every=10, ns_iters=20,
                 damp=1e-3, S_mode='mc', last_S='identity', seed=0):
        super().__init__(model, lr, rule='adam', b1=b1, b2=b2, eps=eps, beta_stats=beta_stats, stats_every=stats_every,
                         eig_every=root_every, S_mode=S_mode, seed=seed, last_S=last_S, stats='gn')
        self.root_every, self.ns_iters, self.damp = root_every, ns_iters, damp
        for li, m in enumerate(self.layers):
            assert isinstance(m, torch.nn.Linear), 'KronRoot: Linear layers only'
            st = self.st[li]; o, i1 = st['o'], st['i1']
            W = m.weight.detach(); Wt = W if m.bias is None else torch.cat([W, m.bias.detach()[:, None]], 1)
            m.weight.requires_grad_(False)
            if m.bias is not None: m.bias.requires_grad_(False)
            m.register_parameter('_Wt', torch.nn.Parameter(Wt.clone()))
            I_ = lambda k: torch.eye(k, dtype=W.dtype, device=W.device)
            st.update(Ap=I_(i1), Am=I_(i1), Sp=None if st['S_id'] else I_(o), Sm=None if st['S_id'] else I_(o),
                      m=torch.zeros_like(Wt), v=torch.zeros_like(Wt), carry=None, n_root=0)
            m._kr_st = st
            def fwd(mod, x):
                st_ = mod._kr_st; lead = x.shape[:-1]; xr = x.reshape(-1, x.shape[-1])
                if mod.bias is not None: xr = torch.cat([xr, torch.ones(xr.shape[0], 1, dtype=xr.dtype, device=xr.device)], 1)
                h = (xr @ st_['Am']) @ mod._Wt.T                            # x~ A^{-1/4} W~^T
                if st_['Sm'] is not None: h = h @ st_['Sm']                  # ... S^{-1/4}   (symmetric)
                return h.reshape(*lead, h.shape[-1])
            m.forward = fwd.__get__(m, type(m))

    @torch.no_grad()
    def _refresh_roots(self, li):
        st = self.st[li]; m = self.layers[li]; o, i1 = st['o'], st['i1']
        Ap, Am = quarter_roots(st['A'], self.damp, self.ns_iters)
        self.flops += 2 * self.ns_iters * 3 * 2.0 * i1 ** 3
        if st['S_id']: Sp = Sm = None
        else:
            Sp, Sm = quarter_roots(st['S'], self.damp, self.ns_iters)
            self.flops += 2 * self.ns_iters * 3 * 2.0 * o ** 3
        # re-express W~, m, v so that W = S^{-1/4} W~ A^{-1/4} is unchanged
        LA = st['Am'] @ Ap                       # A^{-1/4} A'^{1/4}   (right factor for W~)
        LAg = st['Ap'] @ Am                      # A^{1/4} A'^{-1/4}   (right factor for gradient-like objects)
        if st['S_id']: LS = LSg = None
        else:
            LS = Sp @ st['Sm']                   # S'^{1/4} S^{-1/4}
            LSg = Sm @ st['Sp']                  # S'^{-1/4} S^{1/4}
        def tr(X, L, R): return (X if L is None else L @ X) @ R
        m._Wt.data = tr(m._Wt.data, LS, LA)
        st['m'] = tr(st['m'], LSg, LAg)
        st['v'] = tr(st['v'], None if LSg is None else LSg ** 2, LAg ** 2).clamp(min=0)
        st['carry'] = (LSg, LAg)                 # for this step's gradient (taken in the old coordinates)
        st['Ap'], st['Am'], st['Sp'], st['Sm'] = Ap, Am, Sp, Sm
        st['n_root'] += 1
        self.flops += 8.0 * o * i1 * (o + i1)

    @torch.no_grad()
    def step(self, stats_x=None, emp_factors=None, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1; t = self.t
        t0 = self._tic()
        if (t - 1) % self.stats_every == 0:
            if emp_factors is not None: self._update_factors(*emp_factors)
            elif stats_x is not None: self._update_factors(*self.compute_stats(stats_x))
            for st in self.st:
                self.flops += 2.0 * self._last_B * (st['i1'] ** 2 + (0 if st['S_id'] else st['o'] ** 2))
        self._toc('stats', t0)
        tr0 = self._tic()
        for li, st in enumerate(self.st):
            if st['n_root'] == 0 or (t - 1) % self.root_every == 0: self._refresh_roots(li)
        self._toc('roots', tr0)
        ts = self._tic()
        bc1 = 1 - self.b1 ** t; bc2 = 1 - self.b2 ** t
        for li, (m, st) in enumerate(zip(self.layers, self.st)):
            o, i1 = st['o'], st['i1']
            g = m._Wt.grad
            if st['carry'] is not None:
                LSg, LAg = st['carry']; g = (g if LSg is None else LSg @ g) @ LAg; st['carry'] = None
                self.flops += 2.0 * o * i1 * ((0 if LSg is None else o) + i1)
            st['m'].mul_(self.b1).add_(g, alpha=1 - self.b1)
            st['v'].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            m._Wt.data.sub_(lr * (st['m'] / bc1) / ((st['v'] / bc2).sqrt() + self.eps))
            self.flops += 4.0 * self._last_B * (i1 * i1 + (0 if st['S_id'] else o * o)) + 12.0 * o * i1
        self._toc('step', ts)
        if self._step_hooked:
            self._detach(); self._acts.clear(); self._gouts.clear(); self._step_hooked = False
