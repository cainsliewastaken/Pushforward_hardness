"""KronSign: per-layer K-FAC Gauss-Newton eigenbasis optimizer for networks of nn.Linear layers.

For each nn.Linear (weight W: out x in, optional bias b) the parameters are treated as one matrix Wt = [W | b] (out x (in+1)).
  A = E[a~ a~^T]   input covariance, a~ = [a; 1]                (in+1) x (in+1)
  S = E[d d^T]      backprop-signal covariance at the layer output   out x out
  G_l ~ A (x) S     (K-FAC); its eigenmodes are u_i v_j^T with U_S = eig(S), U_A = eig(A).
Step on the momentum M of Wt, rotated into the basis C = U_S^T M U_A:
  rule='sign': Wt -= lr * U_S sign(C) U_A^T                            (Signum in the K-FAC eigenbasis)
  rule='adam': Wt -= lr * U_S [C_hat / (sqrt(V_hat) + eps)] U_A^T       (Adam in the K-FAC eigenbasis, EKFAC-style;
               V = EMA of (U_S^T G U_A)^2, carried into a new basis by squared overlaps)
Statistics (every stats_every steps, EMA with beta_stats) come from a separate pass over a data subset:
  S_mode='mc'    : one random +-1 combination of the outputs per sample (unbiased GN estimate for squared loss)
  S_mode='exact' : one backward pass per output (exact GN factor; n_out passes)
  S_mode='emp'   : signals captured from the training backward pass (residual-weighted, free; call capture_emp())
Eigendecompositions every eig_every steps: method 'eigh' (exact) or 'qr' (one warm-started power step + QR).
Parameters not in nn.Linear layers get Adam with the same lr.
"""
import torch, time
from concurrent.futures import ThreadPoolExecutor


class KronSign:
    def __init__(self, model, lr, rule='sign', b1=0.9, b2=0.999, eps=1e-8, beta_stats=0.95, stats_every=10, eig_every=10,
                 S_mode='mc', eig='eigh', seed=0):
        self.model, self.lr, self.rule, self.b1, self.b2, self.eps = model, lr, rule, b1, b2, eps
        self.beta_stats, self.stats_every, self.eig_every, self.S_mode, self.eig = beta_stats, stats_every, eig_every, S_mode, eig
        self.layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        lin_params = {id(p) for m in self.layers for p in m.parameters()}
        self.other = [p for p in model.parameters() if id(p) not in lin_params and p.requires_grad]
        self.st = [dict(A=None, S=None, UA=None, US=None, M=None, V=None) for _ in self.layers]
        self.ost = [dict(m=torch.zeros_like(p), v=torch.zeros_like(p)) for p in self.other]
        self.t = 0
        self.gen = None; self.seed = seed
        self._acts = {}; self._gouts = {}; self._hooks = []
        self.emp_buffer = None
        self.profile = False; self.times = {}
        self.async_eig = False; self._pool = None; self._pending = None

    # ---------------------------------------------------------------- statistics
    def _attach(self, keep_out, grad_hook):
        """Forward hooks: record each layer's input (detached) and, if keep_out, its output tensor;
        if grad_hook, also record d(loss)/d(output) through a tensor hook during the next backward."""
        self._acts.clear(); self._outs = {}; self._gouts.clear()
        def fh(mod, inp, out, i):
            self._acts[i] = inp[0].detach()
            if keep_out: self._outs[i] = out
            if grad_hook and out.requires_grad:
                out.register_hook(lambda g, i=i: self._gouts.__setitem__(i, g.detach()))
        for i, m in enumerate(self.layers):
            self._hooks.append(m.register_forward_hook(lambda mod, inp, out, i=i: fh(mod, inp, out, i)))

    def _detach(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()

    @staticmethod
    def _aug(a):
        a = a.reshape(-1, a.shape[-1])
        return torch.cat([a, torch.ones(a.shape[0], 1, dtype=a.dtype, device=a.device)], 1)

    def _factors_from(self, acts, gouts_list):
        As, Ss = [], []
        for i in range(len(self.layers)):
            at = self._aug(acts[i]); As.append(at.T @ at / at.shape[0])
            S = 0
            for g in gouts_list:
                gi = g[i].reshape(-1, g[i].shape[-1]); S = S + gi.T @ gi
            Ss.append(S / at.shape[0])
        return As, Ss

    def compute_stats(self, x):
        """GN factors on input batch x (a data subset), S_mode 'mc' or 'exact'. Does not touch parameter gradients."""
        if self.gen is None:
            self.gen = torch.Generator(device=x.device); self.gen.manual_seed(1000 + self.seed)
        self._attach(keep_out=True, grad_hook=False)
        try:
            with torch.enable_grad():
                out = self.model(x)
                outs = [self._outs[i] for i in range(len(self.layers))]
                probes = []
                if self.S_mode == 'exact':
                    for c in range(out.shape[-1]):
                        e = torch.zeros_like(out); e[..., c] = 1.0; probes.append(e)
                else:
                    probes.append(torch.randint(0, 2, out.shape, generator=self.gen, device=x.device).to(out.dtype) * 2 - 1)
                gl = []
                for k, pr in enumerate(probes):
                    gz = torch.autograd.grad(out, outs, grad_outputs=pr, retain_graph=k < len(probes) - 1)
                    gl.append({i: g.detach() for i, g in enumerate(gz)})
                acts = dict(self._acts)
        finally:
            self._detach(); self._outs = {}
        return self._factors_from(acts, gl)

    def begin_train_stats(self):
        """Call before the training forward on a statistics step (S_mode 'mc'): records layer inputs and outputs."""
        if self.gen is None:
            dev = next(self.model.parameters()).device
            self.gen = torch.Generator(device=dev); self.gen.manual_seed(1000 + self.seed)
        self._attach(keep_out=True, grad_hook=False)

    def end_train_stats(self, out):
        """Call after the training forward and BEFORE loss.backward(): one random +-1 probe backward on the same graph."""
        try:
            outs = [self._outs[i] for i in range(len(self.layers))]
            pr = torch.randint(0, 2, out.shape, generator=self.gen, device=out.device).to(out.dtype) * 2 - 1
            gz = torch.autograd.grad(out, outs, grad_outputs=pr, retain_graph=True)
            acts = dict(self._acts)
        finally:
            self._detach(); self._outs = {}
        return self._factors_from(acts, [{i: g.detach() for i, g in enumerate(gz)}])

    def capture_emp(self):
        """Call before the training forward pass when S_mode='emp': records inputs and output-gradients of that pass."""
        self._attach(keep_out=False, grad_hook=True)

    def finish_emp(self, loss_scale=1.0):
        """Call after loss.backward() when S_mode='emp'. loss_scale undoes the loss's 1/batch normalisation."""
        acts, gouts = dict(self._acts), {i: g * loss_scale for i, g in self._gouts.items()}
        self._detach()
        return self._factors_from(acts, [gouts])

    def _update_factors(self, As, Ss):
        for st, A, S in zip(self.st, As, Ss):
            if st['A'] is None: st['A'], st['S'] = A.clone(), S.clone()
            else:
                st['A'].mul_(self.beta_stats).add_(A, alpha=1 - self.beta_stats)
                st['S'].mul_(self.beta_stats).add_(S, alpha=1 - self.beta_stats)

    def _basis(self, F, U_old):
        if self.eig == 'qr' and U_old is not None:
            Q, _ = torch.linalg.qr(F @ U_old)
            ev = (Q * (F @ Q)).sum(0); order = torch.argsort(ev, descending=True)
            return Q[:, order]
        _, U = torch.linalg.eigh(F)
        return U.flip(1)

    def _tic(self):
        if self.profile:
            if torch.cuda.is_available(): torch.cuda.synchronize()
            return time.perf_counter()
    def _toc(self, key, t0):
        if self.profile:
            if torch.cuda.is_available(): torch.cuda.synchronize()
            self.times[key] = self.times.get(key, 0.0) + time.perf_counter() - t0

    # ---------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, stats_x=None, emp_factors=None, lr=None):
        """stats_x: data subset for 'mc'/'exact' statistics (used when due). emp_factors: output of finish_emp() for 'emp'."""
        lr = self.lr if lr is None else lr
        self.t += 1; t = self.t
        t0 = self._tic()
        if (t - 1) % self.stats_every == 0:
            if emp_factors is not None:
                self._update_factors(*emp_factors)
            elif self.S_mode == 'emp':
                pass
            elif stats_x is not None:
                self._update_factors(*self.compute_stats(stats_x))
        self._toc('stats', t0)
        bc1 = 1 - self.b1 ** t; bc2 = 1 - self.b2 ** t
        for m, st in zip(self.layers, self.st):
            W = m.weight; b = m.bias
            G = W.grad if b is None else torch.cat([W.grad, b.grad[:, None]], 1)
            if st['M'] is None: st['M'] = torch.zeros_like(G)
            st['M'].mul_(self.b1).add_(G, alpha=1 - self.b1)
            new_basis = None
            if st['UA'] is None:
                te = self._tic(); new_basis = (self._basis(st['A'], None), self._basis(st['S'], None))
            elif self.async_eig:
                te = self._tic()
                if self._pending is not None and self._pending[0].done() and st is self.st[-1]:
                    pass
                if st.get('fut') is not None and st['fut'].done():
                    new_basis = tuple(x.to(G.device) for x in st['fut'].result()); st['fut'] = None
                if (t - 1) % self.eig_every == 0 and st.get('fut') is None:
                    if self._pool is None: self._pool = ThreadPoolExecutor(max_workers=1)
                    Ac, Sc = st['A'].to('cpu', copy=True), st['S'].to('cpu', copy=True)
                    st['fut'] = self._pool.submit(lambda Ac=Ac, Sc=Sc: (torch.linalg.eigh(Ac)[1].flip(1), torch.linalg.eigh(Sc)[1].flip(1)))
                self._toc('eig', te)
            elif (t - 1) % self.eig_every == 0:
                te = self._tic(); new_basis = (self._basis(st['A'], st['UA']), self._basis(st['S'], st['US']))
            if new_basis is not None:
                UA, US = new_basis
                if self.rule == 'adam':
                    if st['V'] is None: st['V'] = torch.zeros_like(G)
                    elif st['UA'] is not None:
                        OS = (US.T @ st['US']) ** 2; OA = (UA.T @ st['UA']) ** 2
                        st['V'] = OS @ st['V'] @ OA.T
                st['UA'], st['US'] = UA, US
                if not self.async_eig: self._toc('eig', te)
            tr = self._tic()
            C = st['US'].T @ st['M'] @ st['UA']
            if self.rule == 'sign':
                D = torch.sign(C)
            else:
                Gr = st['US'].T @ G @ st['UA']
                st['V'].mul_(self.b2).addcmul_(Gr, Gr, value=1 - self.b2)
                D = (C / bc1) / ((st['V'] / bc2).sqrt() + self.eps)
            upd = st['US'] @ D @ st['UA'].T
            if b is None: W.sub_(lr * upd)
            else:
                W.sub_(lr * upd[:, :-1]); b.sub_(lr * upd[:, -1])
            self._toc('rotate_step', tr)
        for p, s in zip(self.other, self.ost):
            if p.grad is None: continue
            s['m'].mul_(self.b1).add_(p.grad, alpha=1 - self.b1); s['v'].mul_(self.b2).addcmul_(p.grad, p.grad, value=1 - self.b2)
            p.sub_(lr * (s['m'] / bc1) / ((s['v'] / bc2).sqrt() + self.eps))

    def zero_grad(self):
        for p in self.model.parameters(): p.grad = None
