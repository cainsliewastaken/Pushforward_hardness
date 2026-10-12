"""KronSign: per-layer Kronecker-eigenbasis optimizer for networks of nn.Linear and nn.Conv2d layers.

Each layer's parameters form one matrix Wt = [W | b]  (o x i1; a convolution is unfolded, o = out channels,
i1 = in_channels*kh*kw + 1).  Two factor matrices give a Kronecker eigenbasis U_S (o x o), U_A (i1 x i1):
  stats='gn'       K-FAC Gauss-Newton: A = E[a~ a~^T] (layer inputs, +1 for the bias), S = E[d d^T] with d the backprop
                   signal of one random +-1 output probe ('mc': one extra backward per statistics step), of every
                   output ('exact') or of the training loss ('emp').
  stats='shampoo'  A = EMA[G^T G], S = EMA[G G^T] from the parameter gradient (no hooks). With rule='adam' this is SOAP.
Update rules on the rotated gradient Gr = U_S^T G U_A:
  rule='sign'   Wt -= lr U_S sign(C) U_A^T,  C = momentum of Gr, carried exactly across basis refreshes.
  rule='adam'   Wt -= lr U_S [C_hat / (sqrt(V_hat)+eps)] U_A^T, V = EMA of Gr^2 per mode (EKFAC-style), carried by
                squared overlaps.
  rule='kronv'  rank-1 second moment V ~ v_r v_c^T / mean(v_r): the step becomes a preconditioner product
                Wt -= lr sqrt(mu) P_S M P_A with P_S = U_S diag(1/(sqrt(v_r)+e)) U_S^T, P_A likewise, M the momentum in
                parameter space; v_r, v_c are refreshed from one rotated gradient at every basis refresh. Half the
                per-step rotation cost of 'adam' and no rotated state.
Options
  block=b          block-diagonal factors: each side split into blocks of b units (last block zero-padded); the
                   rotation cost drops from o*i*(o+i) to 2*o*i*b per side. block=0 is the full method.
  last_S='identity' for the squared loss the GN output factor of the last layer is E[probe probe^T] = I, so its output
                   side needs no statistics, eigendecomposition or rotation ('gn' stats, 'mc'/'exact' modes only).
  eig_batch        one batched eigendecomposition for all due blocks of equal size (across layers).
  eig_adapt        per-layer refresh period: doubles (up to eig_max) while the basis overlap stays above 0.99, halves
                   when it drops below 0.9.
  rot_dtype        dtype for the rotation products (e.g. torch.float32, torch.bfloat16); None = parameter dtype.
  ddp              factors averaged across ranks; eigendecompositions sharded by layer and broadcast.
Eigendecompositions: 'eigh' (exact) or 'qr' (one warm-started power step + QR). Statistics every stats_every steps
(EMA with beta_stats). Parameters outside Linear/Conv2d layers get Adam with the same lr.
"""
import math, time, torch
import torch.nn.functional as Fn


class KronSign:
    def __init__(self, model, lr, rule='sign', b1=0.9, b2=0.999, eps=1e-8, beta_stats=0.95, stats_every=10, eig_every=10,
                 S_mode='mc', eig='eigh', seed=0, block=0, last_S='stats', stats='gn', eig_batch=False, eig_adapt=False,
                 eig_max=80, rot_dtype=None, ddp=False, clip=1.0, reparam=False):
        assert rule in ('sign', 'adam', 'kronv') and stats in ('gn', 'shampoo') and S_mode in ('mc', 'exact', 'emp')
        self.clip = clip                     # kronv only: per-layer RMS cap of the normalised update (Adafactor convention)
        self.model, self.lr, self.rule, self.b1, self.b2, self.eps = model, lr, rule, b1, b2, eps
        self.beta_stats, self.stats_every, self.eig_every, self.S_mode, self.eig = beta_stats, stats_every, eig_every, S_mode, eig
        self.block, self.stats, self.eig_batch, self.eig_adapt, self.eig_max, self.rot_dtype = int(block), stats, eig_batch, eig_adapt, eig_max, rot_dtype
        self.layers = [m for m in model.modules() if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))]
        for m in self.layers:
            if isinstance(m, torch.nn.Conv2d): assert m.groups == 1, 'grouped convolutions are not supported'
        lin_params = {id(p) for m in self.layers for p in m.parameters()}
        self.other = [p for p in model.parameters() if id(p) not in lin_params and p.requires_grad]
        self.ost = [dict(m=torch.zeros_like(p), v=torch.zeros_like(p)) for p in self.other]
        self.ddp = bool(ddp) and torch.distributed.is_available() and torch.distributed.is_initialized()
        if self.ddp:
            self.world, self.rank = torch.distributed.get_world_size(), torch.distributed.get_rank()
        self.st = []
        for li, m in enumerate(self.layers):
            o, i1 = self._dims(m)
            S_id = (last_S == 'identity' and li == len(self.layers) - 1 and stats == 'gn' and S_mode in ('mc', 'exact'))
            self.st.append(dict(A=None, S=None, UA=None, US=None, C=None, V=None, M=None, vr=None, vc=None, PS=None, PA=None,
                                nv=0, scale=1.0, o=o, i1=i1, S_id=S_id, bd=self._bdims(o, i1, S_id), period=eig_every, next=1))
        self.t = 0
        self.gen = None; self.seed = seed
        self._acts = {}; self._gouts = {}; self._outs = {}; self._hooks = []
        self.profile = False; self.times = {}
        self.async_eig = False
        self.lowrank = False; self._step_hooked = False
        self.flops = 0.0; self.n_lowrank = 0; self.n_refresh = 0
        self.reparam = bool(reparam)
        if self.reparam:
            assert rule in ('sign', 'adam') and not self.ddp, 'reparam: rules sign/adam, no DDP'
            self._reparametrize()

    # ---------------------------------------------------------------- layer geometry
    @staticmethod
    def _dims(m):
        hb = m.bias is not None
        if isinstance(m, torch.nn.Linear): return m.weight.shape[0], m.weight.shape[1] + hb
        return m.weight.shape[0], m.weight[0].numel() + hb

    def _bdims(self, o, i1, S_id):
        """(b_out, n_out, P_out, b_in, n_in, P_in): block layout, one block per side when unblocked."""
        if self.block <= 0:
            return o, 1, o, i1, 1, i1
        bo = o if S_id else min(self.block, o); bi = min(self.block, i1)
        nb, na = -(-o // bo), -(-i1 // bi)
        return bo, nb, nb * bo, bi, na, na * bi

    def _grad_mat(self, m):
        G = m.weight.grad.reshape(m.weight.shape[0], -1)
        return G if m.bias is None else torch.cat([G, m.bias.grad[:, None]], 1)

    @staticmethod
    def _apply(m, upd, lr):
        W = m.weight; iw = W[0].numel()
        W.sub_(lr * upd[:, :iw].reshape(W.shape))
        if m.bias is not None: m.bias.sub_(lr * upd[:, iw])

    def _rows_in(self, li, a):
        """Layer input as rows (samples x i1), bias column appended; convolutions are unfolded into patches."""
        m = self.layers[li]
        if isinstance(m, torch.nn.Conv2d):
            kh, kw = m.kernel_size; ph, pw = m.padding
            if m.padding_mode != 'zeros':
                a = Fn.pad(a, (pw, pw, ph, ph), mode=m.padding_mode); ph = pw = 0
            a = Fn.unfold(a, (kh, kw), dilation=m.dilation, padding=(ph, pw), stride=m.stride).transpose(1, 2).reshape(-1, m.weight[0].numel())
        else:
            a = a.reshape(-1, a.shape[-1])
        if m.bias is None: return a
        return torch.cat([a, torch.ones(a.shape[0], 1, dtype=a.dtype, device=a.device)], 1)

    def _rows_out(self, li, g):
        m = self.layers[li]
        if isinstance(m, torch.nn.Conv2d): return g.permute(0, 2, 3, 1).reshape(-1, g.shape[1])
        return g.reshape(-1, g.shape[-1])

    # ---------------------------------------------------------------- reparametrisation
    def _reparametrize(self):
        """Linear layers train Wt in rotated (padded) coordinates: W = (blockdiag(U_S) Wt blockdiag(U_A)^T)[:o, :i1].
        The forward pass applies U_A and U_S; autograd returns the rotated gradient; the step is elementwise on Wt."""
        for li, m in enumerate(self.layers):
            if not isinstance(m, torch.nn.Linear): continue
            st = self.st[li]; bo, nb, Po, bi, na, Pa = st['bd']; o, i1 = st['o'], st['i1']
            W = m.weight.detach(); Wt = W if m.bias is None else torch.cat([W, m.bias.detach()[:, None]], 1)
            Wt = Fn.pad(Wt, (0, Pa - i1, 0, Po - o)).clone()
            m.weight.requires_grad_(False)
            if m.bias is not None: m.bias.requires_grad_(False)
            m.register_parameter('_Wt', torch.nn.Parameter(Wt))
            eye = lambda n, b: torch.eye(b, dtype=W.dtype, device=W.device).expand(n, b, b).clone()
            st['UA'] = eye(na, bi); st['US'] = None if st['S_id'] else eye(nb, bo)
            st['rp'] = True; st['fresh'] = True; st['carry_ops'] = None
            m._ks_st = st
            def fwd(mod, x):
                st_ = mod._ks_st; bo_, nb_, Po_, bi_, na_, Pa_ = st_['bd']; o_ = st_['o']
                lead = x.shape[:-1]; xr = x.reshape(-1, x.shape[-1])
                if mod.bias is not None: xr = torch.cat([xr, torch.ones(xr.shape[0], 1, dtype=xr.dtype, device=xr.device)], 1)
                if xr.shape[1] != Pa_: xr = Fn.pad(xr, (0, Pa_ - xr.shape[1]))
                h = torch.einsum('nlr,lrs->nls', xr.view(-1, na_, bi_), st_['UA']).reshape(-1, Pa_) @ mod._Wt.T
                if st_['US'] is not None: h = torch.einsum('nkq,kpq->nkp', h.view(-1, nb_, bo_), st_['US']).reshape(-1, Po_)
                return h[:, :o_].reshape(*lead, o_)
            m.forward = fwd.__get__(m, type(m))

    # ---------------------------------------------------------------- statistics
    def _attach(self, keep_out, grad_hook):
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

    def _factors_from(self, acts, gouts_list):
        As, Ss = [], []
        for i in range(len(self.layers)):
            at = self._rows_in(i, acts[i]); n = at.shape[0]
            As.append(at.T @ at / n)
            if self.st[i]['S_id']: Ss.append(None); continue
            S = 0
            for g in gouts_list:
                gi = self._rows_out(i, g[i]); S = S + gi.T @ gi
            Ss.append(S / n)
        return As, Ss

    def _gen(self, dev):
        if self.gen is None:
            self.gen = torch.Generator(device=dev); self.gen.manual_seed(1000 + self.seed + (1000003 * self.rank if self.ddp else 0))
        return self.gen

    def compute_stats(self, x):
        """GN factors on input batch x (a data subset), S_mode 'mc' or 'exact'. Does not touch parameter gradients."""
        self._gen(x.device)
        self._attach(keep_out=True, grad_hook=False)
        try:
            with torch.enable_grad():
                out = self.model(x)
                outs = [self._outs[i] for i in range(len(self.layers))]
                probes = []
                if self.S_mode == 'exact':
                    for c in range(out.shape[1]):
                        e = torch.zeros_like(out); e[:, c] = 1.0; probes.append(e)
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
        """Call before the training forward. Records layer inputs and d(loss)/d(output) for the low-rank rotation."""
        self._gen(next(self.model.parameters()).device)
        self._attach(keep_out=keep_out, grad_hook=True); self._step_hooked = True

    def emp_from_step(self, loss_scale=1.0):
        return self._factors_from(dict(self._acts), [{i: g * loss_scale for i, g in self._gouts.items()}])

    def begin_train_stats(self):
        self._gen(next(self.model.parameters()).device)
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
        self._attach(keep_out=False, grad_hook=True)

    def finish_emp(self, loss_scale=1.0):
        acts, gouts = dict(self._acts), {i: g * loss_scale for i, g in self._gouts.items()}
        self._detach()
        return self._factors_from(acts, [gouts])

    def _update_factors(self, As, Ss):
        if self.ddp and self.stats == 'gn':
            for F in list(As) + [S for S in Ss if S is not None]:
                torch.distributed.all_reduce(F); F.div_(self.world)
        for st, A, S in zip(self.st, As, Ss):
            if st['A'] is None:
                st['A'] = A.clone(); st['S'] = None if S is None else S.clone()
            else:
                st['A'].mul_(self.beta_stats).add_(A, alpha=1 - self.beta_stats)
                if S is not None: st['S'].mul_(self.beta_stats).add_(S, alpha=1 - self.beta_stats)

    # ---------------------------------------------------------------- blocked linear algebra
    @staticmethod
    def _diag_blocks(F, b, n):
        P = n * b; d = F.shape[0]
        if P != d: F = Fn.pad(F, (0, P - d, 0, P - d))
        return F.view(n, b, n, b).diagonal(dim1=0, dim2=2).permute(2, 0, 1).contiguous()

    def _basis_blocked(self, Fb, U_old):
        """Per-block eigenbasis of Fb (n, b, b); columns ordered by descending eigenvalue."""
        if self.eig == 'qr' and U_old is not None:
            Q, _ = torch.linalg.qr(Fb @ U_old)
            ev = (Q * (Fb @ Q)).sum(-2); order = torch.argsort(ev, dim=-1, descending=True)
            return torch.gather(Q, 2, order[:, None, :].expand_as(Q))
        _, U = torch.linalg.eigh(Fb)
        return U.flip(-1)

    def _rot(self, L, X, R, lt, rt):
        """blockdiag(L)[^T] X blockdiag(R)[^T];  L (nb, bo, bo) or None (identity), R (na, bi, bi), X (nb*bo, na*bi)."""
        na, bi = R.shape[0], R.shape[1]
        dt = X.dtype
        if self.rot_dtype is not None:
            X = X.to(self.rot_dtype); R = R.to(self.rot_dtype); L = None if L is None else L.to(self.rot_dtype)
        if L is None:
            T = X.view(1, X.shape[0], na, bi)
        else:
            nb, bo = L.shape[0], L.shape[1]
            X4 = X.view(nb, bo, na, bi)
            T = torch.einsum('kqp,kqlr->kplr', L, X4) if lt else torch.einsum('kpq,kqlr->kplr', L, X4)
        Y = torch.einsum('kplr,lrs->kpls', T, R) if rt else torch.einsum('kpls,lrs->kplr', T, R)
        return Y.reshape(X.shape[0], na * bi).to(dt)

    @staticmethod
    def _pad(X, Po, Pa):
        return X if X.shape == (Po, Pa) else Fn.pad(X, (0, Pa - X.shape[1], 0, Po - X.shape[0]))

    def _tic(self):
        if self.profile:
            if torch.cuda.is_available(): torch.cuda.synchronize()
            return time.perf_counter()
    def _toc(self, key, t0):
        if self.profile:
            if torch.cuda.is_available(): torch.cuda.synchronize()
            self.times[key] = self.times.get(key, 0.0) + time.perf_counter() - t0

    # ---------------------------------------------------------------- basis refresh
    def _refresh(self, due):
        """Recompute the eigenbases of the layers in `due` (batched across equal block sizes if eig_batch; sharded by
        layer and broadcast if ddp), carry the optimizer state into the new bases, adapt the refresh periods."""
        te = self._tic(); t = self.t
        items = []                                   # (li, side, Fb, U_old)
        for li in due:
            st = self.st[li]; bo, nb, Po, bi, na, Pa = st['bd']
            fresh = st.get('fresh', False)
            items.append((li, 'A', self._diag_blocks(st['A'], bi, na), None if fresh else st['UA']))
            if not st['S_id']: items.append((li, 'S', self._diag_blocks(st['S'], bo, nb), None if fresh else st['US']))
        mine = [it for it in items if not self.ddp or it[0] % self.world == self.rank]
        new = {}
        if self.eig_batch:
            groups = {}
            for it in mine: groups.setdefault((it[2].shape[1], it[3] is None), []).append(it)
            for (b, fresh), its in groups.items():
                Fb = torch.cat([it[2] for it in its]); Uo = None if fresh else torch.cat([it[3] for it in its])
                U = self._basis_blocked(Fb, Uo); k = 0
                for it in its:
                    n = it[2].shape[0]; new[(it[0], it[1])] = U[k:k + n]; k += n
        else:
            for it in mine: new[(it[0], it[1])] = self._basis_blocked(it[2], it[3])
        if self.ddp:
            for it in items:
                key = (it[0], it[1])
                if key not in new: new[key] = torch.empty_like(it[2])
                torch.distributed.broadcast(new[key], src=it[0] % self.world)
        for li in due:
            st = self.st[li]; bo, nb, Po, bi, na, Pa = st['bd']
            UA, US = new[(li, 'A')], (None if st['S_id'] else new[(li, 'S')])
            if st['UA'] is not None:                 # carry the state into the new basis
                MA = torch.einsum('lrq,lrs->lqs', st['UA'], UA)                      # UA_old^T UA_new per block
                MS = None if st['S_id'] else torch.einsum('kqp,kqr->kpr', US, st['US'])   # US_new^T US_old per block
                if st['C'] is not None: st['C'] = self._rot(MS, st['C'], MA, False, True)          # US_new^T US_old C UA_old^T UA_new
                if st['V'] is not None: st['V'] = self._rot(MS ** 2 if MS is not None else None, st['V'], MA ** 2, False, True)
                if st.get('rp'):
                    m_ = self.layers[li]
                    m_._Wt.data = self._rot(MS, m_._Wt.data, MA, False, True); st['carry_ops'] = (MS, MA)
                    self.flops += 2.0 * Po * Pa * (bi + (0 if st['S_id'] else bo))
                if st['vr'] is not None:
                    st['vc'] = torch.einsum('lqs,lq->ls', MA ** 2, st['vc'].view(na, bi)).reshape(-1)
                    if MS is not None: st['vr'] = torch.einsum('kpr,kr->kp', MS ** 2, st['vr'].view(nb, bo)).reshape(-1)
                if self.eig_adapt and not st.get('fresh', False):
                    ov = torch.einsum('lqs,lqs->ls', st['UA'], UA).abs().mean()
                    if MS is not None: ov = 0.5 * (ov + torch.einsum('kpq,kpq->kq', st['US'], US).abs().mean())
                    ov = float(ov)
                    if ov > 0.99: st['period'] = min(2 * st['period'], self.eig_max)
                    elif ov < 0.9: st['period'] = max(st['period'] // 2, 1)
            st['UA'], st['US'] = UA, US; st['fresh'] = False
            st['next'] = t + st['period']
            carry = 2 * Po * Pa * (bi + (0 if st['S_id'] else bo))
            self.flops += (2 + 4 / 3 + 2) * (na * bi ** 3 + (0 if st['S_id'] else nb * bo ** 3)) \
                + carry * {'sign': 1, 'adam': 2, 'kronv': 0}[self.rule]
            if self.rule == 'kronv': st['PS'] = None   # rebuilt from the next gradient
        self.n_refresh += 1
        self._toc('eig', te)

    # ---------------------------------------------------------------- rotated gradient
    def _rot_grad(self, li, m, st):
        bo, nb, Po, bi, na, Pa = st['bd']; o, i1, S_id = st['o'], st['i1'], st['S_id']
        if self.lowrank and li in self._gouts and li in self._acts:
            Dl = self._rows_out(li, self._gouts[li]); Bn = Dl.shape[0]
            if Bn * ((0 if S_id else Po * bo) + Pa * bi + Po * Pa) < Po * Pa * ((0 if S_id else bo) + bi):
                At = self._rows_in(li, self._acts[li])
                Dl = Fn.pad(Dl, (0, Po - o)); At = Fn.pad(At, (0, Pa - i1))
                if self.rot_dtype is not None: Dl, At = Dl.to(self.rot_dtype), At.to(self.rot_dtype)
                UA = st['UA'] if self.rot_dtype is None else st['UA'].to(self.rot_dtype)
                L = Dl.T.view(1, Po, Bn) if S_id else torch.einsum('kqp,nkq->kpn', st['US'] if self.rot_dtype is None else st['US'].to(self.rot_dtype), Dl.view(Bn, nb, bo))
                R = torch.einsum('nlr,lrs->nls', At.view(Bn, na, bi), UA)
                self.flops += 2.0 * Bn * ((0 if S_id else Po * bo) + Pa * bi + Po * Pa)
                self.n_lowrank += 1
                return torch.einsum('kpn,nls->kpls', L, R).reshape(Po, Pa).to(m.weight.dtype)
        G = self._grad_mat(m)
        self.flops += 2.0 * Po * Pa * ((0 if S_id else bo) + bi)
        return self._rot(st['US'], self._pad(G, Po, Pa), st['UA'], True, True)

    # ---------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, stats_x=None, emp_factors=None, lr=None):
        """stats_x: data subset for 'mc'/'exact' statistics (used when due). emp_factors: output of finish_emp() for 'emp'."""
        lr = self.lr if lr is None else lr
        self.t += 1; t = self.t
        self._last_B = getattr(self, '_last_B', 1)
        t0 = self._tic()
        if (t - 1) % self.stats_every == 0:
            if self.stats == 'shampoo':
                As, Ss = [], []
                for m, st in zip(self.layers, self.st):
                    G = self._grad_mat(m); As.append(G.T @ G); Ss.append(G @ G.T)
                    self.flops += 2.0 * st['o'] * st['i1'] * (st['o'] + st['i1'])
                self._update_factors(As, Ss)
            else:
                if emp_factors is not None: self._update_factors(*emp_factors)
                elif self.S_mode == 'emp': pass
                elif stats_x is not None: self._update_factors(*self.compute_stats(stats_x))
                for st in self.st:
                    self.flops += 2.0 * self._last_B * (st['i1'] ** 2 + (0 if st['S_id'] else st['o'] ** 2))
        self._toc('stats', t0)
        due = [li for li, st in enumerate(self.st) if st['UA'] is None or t >= st['next']]
        if due: self._refresh(due)
        bc1 = 1 - self.b1 ** t; bc2 = 1 - self.b2 ** t
        for li, (m, st) in enumerate(zip(self.layers, self.st)):
            tr = self._tic()
            bo, nb, Po, bi, na, Pa = st['bd']; o, i1 = st['o'], st['i1']
            if self.rule == 'kronv':
                G = self._grad_mat(m)
                if st['M'] is None: st['M'] = torch.zeros_like(G)
                st['M'].mul_(self.b1).add_(G, alpha=1 - self.b1)
                if st['PS'] is None or li in due:          # refresh the rank-1 second moment and the preconditioners
                    Gr = self._rot_grad(li, m, st)
                    if st['vr'] is None:
                        st['vr'] = torch.zeros(Po, dtype=G.dtype, device=G.device); st['vc'] = torch.zeros(Pa, dtype=G.dtype, device=G.device)
                    G2 = Gr * Gr
                    st['vr'].mul_(self.b2).add_(G2.mean(1), alpha=1 - self.b2); st['vc'].mul_(self.b2).add_(G2.mean(0), alpha=1 - self.b2)
                    st['nv'] += 1; bcv = 1 - self.b2 ** st['nv']
                    vr, vc = st['vr'] / bcv, st['vc'] / bcv
                    mu = vr[:o].mean(); e = 1e-4 * float(mu.sqrt()) + 1e-30      # relative floor: no mode amplified > 1e4x the mean scale
                    dr = 1.0 / (vr.sqrt() + e); dc = 1.0 / (vc.sqrt() + e)
                    st['scale'] = float(mu.sqrt())
                    st['PA'] = torch.einsum('lqr,lr,lsr->lqs', st['UA'], dc.view(na, bi), st['UA'])
                    st['PS'] = dr if st['S_id'] else torch.einsum('kpq,kq,krq->kpr', st['US'], dr.view(nb, bo), st['US'])
                    self.flops += 2.0 * (na * bi ** 3 + (0 if st['S_id'] else nb * bo ** 3))
                Mp = self._pad(st['M'], Po, Pa) / bc1
                if st['S_id']: upd = st['PS'][:, None] * self._rot(None, Mp, st['PA'], False, False)
                else: upd = self._rot(st['PS'], Mp, st['PA'], False, False)
                upd = upd[:o, :i1] * st['scale']
                if self.clip > 0:                               # Adafactor-style update clipping: RMS(upd) <= clip
                    upd = upd / torch.clamp(upd.pow(2).mean().sqrt() / self.clip, min=1.0)
                self.flops += 2.0 * Po * Pa * (bi + (0 if st['S_id'] else bo)) + 8.0 * o * i1
            else:
                if st.get('rp'):
                    Gr = m._Wt.grad
                    if st['carry_ops'] is not None:            # gradient was taken in the pre-refresh basis
                        Gr = self._rot(st['carry_ops'][0], Gr, st['carry_ops'][1], False, True); st['carry_ops'] = None
                        self.flops += 2.0 * Po * Pa * (bi + (0 if st['S_id'] else bo))
                    self.flops += 4.0 * self._last_B * (Pa * bi + (0 if st['S_id'] else Po * bo)) + 6.0 * self._last_B * (Po * Pa - o * i1)
                else:
                    Gr = self._rot_grad(li, m, st)
                if st['C'] is None: st['C'] = torch.zeros_like(Gr)
                st['C'].mul_(self.b1).add_(Gr, alpha=1 - self.b1)
                if self.rule == 'sign':
                    D = torch.sign(st['C'])
                else:
                    if st['V'] is None: st['V'] = torch.zeros_like(Gr)
                    st['V'].mul_(self.b2).addcmul_(Gr, Gr, value=1 - self.b2)
                    D = (st['C'] / bc1) / ((st['V'] / bc2).sqrt() + self.eps)
                if st.get('rp'):
                    m._Wt.data.sub_(lr * D); self.flops += 12.0 * o * i1
                    self._toc('rotate_step', tr); continue
                upd = self._rot(st['US'], D, st['UA'], False, False)[:o, :i1]
                self.flops += 2.0 * Po * Pa * (bi + (0 if st['S_id'] else bo)) + 12.0 * o * i1
            self._apply(m, upd, lr)
            self._toc('rotate_step', tr)
        if self._step_hooked:
            self._detach(); self._acts.clear(); self._gouts.clear(); self._step_hooked = False
        for p, s in zip(self.other, self.ost):
            if p.grad is None: continue
            s['m'].mul_(self.b1).add_(p.grad, alpha=1 - self.b1); s['v'].mul_(self.b2).addcmul_(p.grad, p.grad, value=1 - self.b2)
            p.sub_(lr * (s['m'] / bc1) / ((s['v'] / bc2).sqrt() + self.eps))

    def zero_grad(self):
        for p in self.model.parameters(): p.grad = None
