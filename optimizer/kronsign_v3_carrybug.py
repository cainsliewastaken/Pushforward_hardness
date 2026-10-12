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
block=b > 0: block-diagonal factors (Shampoo-style). Each side of each layer is split into blocks of b units (the last
  block padded with zero coordinates, which never move); each block has its own eigenbasis, so U_A and U_S are
  block-diagonal and the rotation cost drops from o*i*(o+i) to 2*o*i*b per side. block=0 (or b >= both dims) is the
  full method.
"""
import torch, time
from concurrent.futures import ThreadPoolExecutor


class KronSign:
    def __init__(self, model, lr, rule='sign', b1=0.9, b2=0.999, eps=1e-8, beta_stats=0.95, stats_every=10, eig_every=10,
                 S_mode='mc', eig='eigh', seed=0, block=0):
        self.model, self.lr, self.rule, self.b1, self.b2, self.eps = model, lr, rule, b1, b2, eps
        self.beta_stats, self.stats_every, self.eig_every, self.S_mode, self.eig = beta_stats, stats_every, eig_every, S_mode, eig
        self.layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        lin_params = {id(p) for m in self.layers for p in m.parameters()}
        self.other = [p for p in model.parameters() if id(p) not in lin_params and p.requires_grad]
        self.st = [dict(A=None, S=None, UA=None, US=None, C=None, V=None) for _ in self.layers]
        self.ost = [dict(m=torch.zeros_like(p), v=torch.zeros_like(p)) for p in self.other]
        self.t = 0
        self.gen = None; self.seed = seed
        self._acts = {}; self._gouts = {}; self._hooks = []
        self.emp_buffer = None
        self.profile = False; self.times = {}
        self.async_eig = False; self._pool = None; self._pending = None
        self.lowrank = False; self._step_hooked = False
        self.flops = 0.0
        self.block = int(block)

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

    def begin_step(self, keep_out=False):
        """Call before the training forward. Records each layer's input and d(loss)/d(output) for the low-rank rotation,
        and (keep_out) the outputs for training-pass statistics."""
        if self.gen is None:
            dev = next(self.model.parameters()).device
            self.gen = torch.Generator(device=dev); self.gen.manual_seed(1000 + self.seed)
        self._attach(keep_out=keep_out, grad_hook=True); self._step_hooked = True

    def emp_from_step(self, loss_scale=1.0):
        """Empirical factors from the current step's captured signals (use with begin_step)."""
        return self._factors_from(dict(self._acts), [{i: g * loss_scale for i, g in self._gouts.items()}])

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
            if not self._step_hooked: self._detach()
            self._outs = {}
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

    # ---------------------------------------------------------------- blocked factors
    def _bdims(self, o, i1):
        """Block layout for a layer: (b_out, n_out, P_out, b_in, n_in, P_in); None when unblocked."""
        if self.block <= 0 or (self.block >= o and self.block >= i1): return None
        bo, bi = min(self.block, o), min(self.block, i1)
        no, ni = -(-o // bo), -(-i1 // bi)
        return bo, no, no * bo, bi, ni, ni * bi

    @staticmethod
    def _diag_blocks(F, b, n):
        """Diagonal b x b blocks of F (d x d, zero-padded to n*b): tensor (n, b, b)."""
        P = n * b; d = F.shape[0]
        if P != d: F = torch.nn.functional.pad(F, (0, P - d, 0, P - d))
        return F.view(n, b, n, b).diagonal(dim1=0, dim2=2).permute(2, 0, 1).contiguous()

    def _basis_blocked(self, Fb, U_old):
        """Per-block eigenbasis, Fb (n, b, b); U_old (n, b, b) or None. Columns ordered by descending eigenvalue."""
        if self.eig == 'qr' and U_old is not None:
            Q, _ = torch.linalg.qr(Fb @ U_old)
            ev = (Q * (Fb @ Q)).sum(-2); order = torch.argsort(ev, dim=-1, descending=True)
            return torch.gather(Q, 2, order[:, None, :].expand_as(Q))
        _, U = torch.linalg.eigh(Fb)
        return U.flip(-1)

    @staticmethod
    def _rot(L, X, R, lt, rt):
        """Block-diagonal rotation  blockdiag(L)[^T] X blockdiag(R)[^T]  (lt / rt: transpose the L / R blocks).
        L (nb, bo, bo), R (na, bi, bi), X (nb*bo, na*bi)."""
        nb, bo = L.shape[0], L.shape[1]; na, bi = R.shape[0], R.shape[1]
        X4 = X.view(nb, bo, na, bi)
        T = torch.einsum('kqp,kqlr->kplr', L, X4) if lt else torch.einsum('kpq,kqlr->kplr', L, X4)
        Y = torch.einsum('kplr,lrs->kpls', T, R) if rt else torch.einsum('kpls,lrs->kplr', T, R)
        return Y.reshape(nb * bo, na * bi)

    @staticmethod
    def _pad(X, Po, Pa):
        return X if X.shape == (Po, Pa) else torch.nn.functional.pad(X, (0, Pa - X.shape[1], 0, Po - X.shape[0]))

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
        self._last_B = getattr(self, '_last_B', 1)
        t0 = self._tic()
        if (t - 1) % self.stats_every == 0:
            if emp_factors is not None:
                self._update_factors(*emp_factors)
            elif self.S_mode == 'emp':
                pass
            elif stats_x is not None:
                self._update_factors(*self.compute_stats(stats_x))
            for m in self.layers:
                o_, i_ = m.weight.shape[0], m.weight.shape[1] + (m.bias is not None)
                self.flops += 2.0 * self._last_B * (i_ * i_ + o_ * o_)
        self._toc('stats', t0)
        bc1 = 1 - self.b1 ** t; bc2 = 1 - self.b2 ** t
        for li, (m, st) in enumerate(zip(self.layers, self.st)):
            W = m.weight; b = m.bias
            out_d, in_d1 = W.shape[0], W.shape[1] + (0 if b is None else 1)
            bd = self._bdims(out_d, in_d1)
            if bd is not None:
                self._step_blocked(li, m, st, bd, lr, bc1, bc2); continue
            new_basis = None
            if st['UA'] is None:
                te = self._tic(); new_basis = (self._basis(st['A'], None), self._basis(st['S'], None))
            elif (t - 1) % self.eig_every == 0:
                te = self._tic(); new_basis = (self._basis(st['A'], st['UA']), self._basis(st['S'], st['US']))
            if new_basis is not None:
                UA, US = new_basis
                if st['C'] is not None:                       # carry the rotated momentum into the new basis (exact)
                    st['C'] = (US.T @ st['US']) @ st['C'] @ (st['UA'].T @ UA)
                if self.rule == 'adam':
                    if st['V'] is None: st['V'] = torch.zeros(out_d, in_d1, dtype=W.dtype, device=W.device)
                    elif st['UA'] is not None:
                        OS = (US.T @ st['US']) ** 2; OA = (UA.T @ st['UA']) ** 2
                        st['V'] = OS @ st['V'] @ OA.T
                st['UA'], st['US'] = UA, US
                self.flops += (2 + 4 / 3) * (out_d ** 3 + in_d1 ** 3) + 2 * (out_d ** 3 + in_d1 ** 3) + 2 * (out_d * out_d * in_d1 + out_d * in_d1 * in_d1) * (2 if self.rule == 'adam' else 1)
                self._toc('eig', te)
            tr = self._tic()
            # rotated gradient: low-rank (batch side) when cheaper, else from the parameter gradient
            Gr = None
            if self.lowrank and li in self._gouts and li in self._acts:
                Dl = self._gouts[li].reshape(-1, out_d); Bn = Dl.shape[0]
                if Bn * (out_d * out_d + in_d1 * in_d1 + out_d * in_d1) < out_d * out_d * in_d1 + out_d * in_d1 * in_d1:
                    At = self._acts[li].reshape(-1, W.shape[1])
                    if b is not None: At = self._aug(At)
                    Gr = (st['US'].T @ Dl.T) @ (At @ st['UA'])
                    self.flops += 2.0 * (out_d * out_d * Bn + Bn * in_d1 * in_d1 + out_d * Bn * in_d1)
                    self.n_lowrank = getattr(self, 'n_lowrank', 0) + 1
            if Gr is None:
                G = W.grad if b is None else torch.cat([W.grad, b.grad[:, None]], 1)
                Gr = st['US'].T @ G @ st['UA']
                self.flops += 2.0 * (out_d * out_d * in_d1 + out_d * in_d1 * in_d1)
            if st['C'] is None: st['C'] = torch.zeros_like(Gr)
            st['C'].mul_(self.b1).add_(Gr, alpha=1 - self.b1)
            if self.rule == 'sign':
                D = torch.sign(st['C'])
            else:
                st['V'].mul_(self.b2).addcmul_(Gr, Gr, value=1 - self.b2)
                D = (st['C'] / bc1) / ((st['V'] / bc2).sqrt() + self.eps)
            upd = st['US'] @ D @ st['UA'].T
            self.flops += 2.0 * (out_d * out_d * in_d1 + out_d * in_d1 * in_d1) + 12.0 * out_d * in_d1
            if b is None: W.sub_(lr * upd)
            else:
                W.sub_(lr * upd[:, :-1]); b.sub_(lr * upd[:, -1])
            self._toc('rotate_step', tr)
        if self._step_hooked:
            self._detach(); self._acts.clear(); self._gouts.clear(); self._step_hooked = False
        for p, s in zip(self.other, self.ost):
            if p.grad is None: continue
            s['m'].mul_(self.b1).add_(p.grad, alpha=1 - self.b1); s['v'].mul_(self.b2).addcmul_(p.grad, p.grad, value=1 - self.b2)
            p.sub_(lr * (s['m'] / bc1) / ((s['v'] / bc2).sqrt() + self.eps))

    @torch.no_grad()
    def _step_blocked(self, li, m, st, bd, lr, bc1, bc2):
        """One layer's update with block-diagonal factors. Rotated momentum C and V live in the padded (Po x Pa) space."""
        t = self.t; W = m.weight; b = m.bias
        out_d, in_d1 = W.shape[0], W.shape[1] + (0 if b is None else 1)
        bo, nb, Po, bi, na, Pa = bd
        new_basis = None
        if st['UA'] is None:
            te = self._tic(); new_basis = (self._basis_blocked(self._diag_blocks(st['A'], bi, na), None),
                                           self._basis_blocked(self._diag_blocks(st['S'], bo, nb), None))
        elif (t - 1) % self.eig_every == 0:
            te = self._tic(); new_basis = (self._basis_blocked(self._diag_blocks(st['A'], bi, na), st['UA']),
                                           self._basis_blocked(self._diag_blocks(st['S'], bo, nb), st['US']))
        if new_basis is not None:
            UA, US = new_basis
            if st['C'] is not None:                           # carry C (exact) and V (squared overlaps) into the new basis
                MS = torch.einsum('kqp,kqr->kpr', US, st['US'])   # US_new^T US_old per block
                MA = torch.einsum('lrq,lrs->lqs', st['UA'], UA)   # UA_old^T UA_new per block
                st['C'] = self._rot(MS, st['C'], MA, False, False)
                if self.rule == 'adam': st['V'] = self._rot(MS ** 2, st['V'], MA ** 2, False, False)
            if self.rule == 'adam' and st['V'] is None:
                st['V'] = torch.zeros(Po, Pa, dtype=W.dtype, device=W.device)
            st['UA'], st['US'] = UA, US
            self.flops += (2 + 4 / 3) * (nb * bo ** 3 + na * bi ** 3) + 2 * (nb * bo ** 3 + na * bi ** 3) \
                + 2 * Po * Pa * (bo + bi) * (2 if self.rule == 'adam' else 1)
            self._toc('eig', te)
        tr = self._tic()
        Gr = None
        if self.lowrank and li in self._gouts and li in self._acts:
            Dl = self._gouts[li].reshape(-1, out_d); Bn = Dl.shape[0]
            if Bn * (Po * bo + Pa * bi + Po * Pa) < 2 * Po * Pa * (bo + bi):
                At = self._acts[li].reshape(-1, W.shape[1])
                if b is not None: At = self._aug(At)
                Dl = torch.nn.functional.pad(Dl, (0, Po - out_d)); At = torch.nn.functional.pad(At, (0, Pa - in_d1))
                L = torch.einsum('kqp,nkq->kpn', st['US'], Dl.view(Bn, nb, bo))      # US^T Dl^T, blocks   (nb, bo, Bn)
                R = torch.einsum('nlr,lrs->nls', At.view(Bn, na, bi), st['UA'])      # At UA, blocks       (Bn, na, bi)
                Gr = torch.einsum('kpn,nls->kpls', L, R).reshape(Po, Pa)
                self.flops += 2.0 * Bn * (Po * bo + Pa * bi + Po * Pa)
                self.n_lowrank = getattr(self, 'n_lowrank', 0) + 1
        if Gr is None:
            G = W.grad if b is None else torch.cat([W.grad, b.grad[:, None]], 1)
            Gr = self._rot(st['US'], self._pad(G, Po, Pa), st['UA'], True, True)
            self.flops += 2.0 * Po * Pa * (bo + bi)
        if st['C'] is None: st['C'] = torch.zeros_like(Gr)
        st['C'].mul_(self.b1).add_(Gr, alpha=1 - self.b1)
        if self.rule == 'sign':
            D = torch.sign(st['C'])
        else:
            st['V'].mul_(self.b2).addcmul_(Gr, Gr, value=1 - self.b2)
            D = (st['C'] / bc1) / ((st['V'] / bc2).sqrt() + self.eps)
        upd = self._rot(st['US'], D, st['UA'], False, False)[:out_d, :in_d1]
        self.flops += 2.0 * Po * Pa * (bo + bi) + 12.0 * out_d * in_d1
        if b is None: W.sub_(lr * upd)
        else:
            W.sub_(lr * upd[:, :-1]); b.sub_(lr * upd[:, -1])
        self._toc('rotate_step', tr)

    def zero_grad(self):
        for p in self.model.parameters(): p.grad = None
