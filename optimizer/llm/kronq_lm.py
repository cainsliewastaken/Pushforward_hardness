"""KronQ for transformers / language models (optimizer/llm/). The method is the main optimizer's (../kronq_main.py, class KronQ):
this file imports its basis, QR and per-mode update (_Basis, _update_c, _fp32) and does not edit them. What it adds is only what a
language model needs:

  - any nn.Linear in any model, found by module traversal and read through forward hooks (inputs a, outputs z) instead of
    kronq_main's own MLP forward; bias handled as kronq_main does ([W | b], input augmented by 1); weight sharing is refused;
  - the parameter split used by SOAP / Muon: the KronQ step on the hidden matrices (attention q/k/v/o, MLP), AdamW on everything
    else (embeddings, a tied or untied output head, LayerNorm / RMSNorm gains, any Linear listed in `exclude` or wider than
    `max_dim`);
  - the curvature probe for the loss in use: loss='mse' -> random +-1 output signs (Gauss-Newton of squared loss, exactly
    kronq_main's probe); loss='ce' -> labels sampled from the model's own softmax (Monte Carlo Fisher, as in K-FAC); probe='emp'
    -> the batch loss's own output gradients (empirical Fisher: no extra backward, a different curvature; NOT the method, a cost
    option to measure);
  - SCALE: the statistics come from ONE micro-batch per statistics step (gradient accumulation over the others as usual), so the
    probe's extra backward costs ~1/3 of one micro-batch, not of the whole step; the statistics are consumed by tensor hooks as
    the probe's gradients pass each layer (no layer-output gradient is kept), and the step runs one layer at a time (no full
    fp32 copy of all gradients). Optional `stats_tokens` row subsampling;
  - bf16 autocast: activations and probe gradients are cast to fp32 for the statistics; parameters must stay fp32 (master
    weights); the basis work runs in full fp32 as in kronq_main;
  - data parallel (torch DDP): the gradients are averaged by DDP; the factor statistics are all-reduced here so every rank holds
    the same factors; each layer's basis refresh is computed by one rank (layer index mod world size) and broadcast, so the bases
    are identical on every rank and the refresh work is spread across ranks (as in distributed Shampoo).

Usage (see README.md and test_kronq_lm.py):
    one batch per step:                           micro-batches (gradient accumulation):
        logits = model(x)                             for i, (x, y) in enumerate(micro):
        loss = ce(logits, y)                              with opt.capture(i == 0):          # statistics from this one
        opt.backward(loss, logits)                            logits = model(x)
        opt.step(lr_t); opt.zero_grad()                   opt.backward(ce(logits, y) / len(micro), logits)
                                                      opt.step(lr_t); opt.zero_grad()
`lr` is the KronQ learning rate at this step; the AdamW part runs at lr_adam x lr / lr_max, and the decay (weight_decay, per step at
peak lr, as kronq_main) follows the same factor.

NOT here (yet): optimizer-state sharding (ZeRO / FSDP), pure-bf16 parameters, activation checkpointing inside the statistics
micro-batch (the probe would recompute it).
"""
import contextlib, os, sys, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kronq_main import _Basis, _update_c, _fp32          # the main optimizer's own pieces (unchanged)


def _no_autocast(x):
    """statistics run in fp32 even when the forward runs under autocast (the hooks fire inside the autocast region)."""
    return torch.autocast(device_type=x.device.type, enabled=False)


def _batch_factor(B, Z):
    """the batch part of _Basis.update: block factors of the rows of Z (same arithmetic)."""
    m = Z.shape[0]; nb = B.nf * B.b; Zf = Z[:, :nb].reshape(m, B.nf, B.b)
    return torch.einsum('mkb,mkc->kbc', Zf, Zf) / m, ((Z[:, nb:].T @ Z[:, nb:] / m) if B.rem else None)


class _Layer:
    """KronQ state of one nn.Linear: the two bases and the per-mode state, all on [W | b] (bias column when the layer has one)."""
    def __init__(self, name, mod, block, refresh, mom_bf16):
        self.name, self.mod, self.has_bias = name, mod, mod.bias is not None
        o, i = mod.weight.shape; dev = mod.weight.device; n = i + int(self.has_bias)
        self.A, self.S = _Basis(n, block, dev), _Basis(o, block, dev)
        self.C = torch.zeros(o, n, device=dev, dtype=torch.bfloat16 if mom_bf16 else torch.float32)
        self.u = torch.zeros(o, n, device=dev, dtype=torch.bfloat16)
        self.hv = (torch.zeros(o, device=dev), torch.zeros(n, device=dev))
        self.has_prev, self.period, self.next_ref = False, refresh, 1
        self.z = None; self.order = -1; self.nA = self.nS = None     # captured output; pending batch factors

    def grad(self):
        g = self.mod.weight.grad.float()
        return torch.cat([g, self.mod.bias.grad.float()[:, None]], 1) if self.has_bias else g

    def apply(self, M):
        """weights -= M (M on [W | b])."""
        i = self.mod.weight.shape[1]
        self.mod.weight.data.sub_(M[:, :i].to(self.mod.weight.dtype))
        if self.has_bias: self.mod.bias.data.sub_(M[:, i].to(self.mod.bias.dtype))

    def to_basis(self, M): return self.A.cols_to(self.S.rows_to(M))       # U_S^T M U_A
    def from_basis(self, M): return self.S.rows_from(self.A.cols_from(M))   # U_S M U_A^T


class KronQLM:
    def __init__(self, model, lr, lr_max=None, loss='ce', probe='fisher', draw_probe=None,
                 lr_adam=None, adam_betas=(0.9, 0.95), adam_eps=1e-8, adam_wd=0.0,
                 exclude=('lm_head',), max_dim=16384, block=256, b1=0.9, beta_u=0.01, u_clamp=4.6, reltol=1e-12, beta_stats=0.95,
                 eig_iters=2, refresh=20, adapt_refresh=0.1, refresh_min=5, stats_every=1, weight_decay=0.0, mom_bf16=True,
                 stats_tokens=0, probe_tokens=0, lam_scale=1.0, stream=True, seed=0):
        assert loss in ('ce', 'mse') and probe in ('fisher', 'emp')
        net = model.module if hasattr(model, 'module') else model            # DDP-wrapped or plain
        self.model, self.lr, self.lr_max = net, lr, (lr if lr_max is None else lr_max)
        self.loss, self.probe, self.draw_probe, self.stream = loss, probe, draw_probe, stream
        self.h = dict(b1=b1, beta_u=beta_u, u_clamp=u_clamp, reltol=reltol, beta_stats=beta_stats, eig_iters=eig_iters,
                      adapt_refresh=adapt_refresh, refresh_min=refresh_min, stats_every=stats_every, weight_decay=weight_decay)
        self.stats_tokens, self.probe_tokens, self.lam_scale = stats_tokens, probe_tokens, lam_scale
        self.ddp = dist.is_available() and dist.is_initialized()
        self.rank, self.world = (dist.get_rank(), dist.get_world_size()) if self.ddp else (0, 1)

        # the split: KronQ on hidden Linear layers, AdamW on the rest
        tied = {id(m.weight) for m in net.modules() if isinstance(m, nn.Embedding)}
        excl = lambda name: any(name == e or name.endswith('.' + e) for e in exclude)
        self.layers = []
        for name, m in net.named_modules():
            if isinstance(m, nn.Linear) and not excl(name) and id(m.weight) not in tied and max(m.weight.shape) <= max_dim:
                if m.weight.dtype in (torch.bfloat16, torch.float16): raise ValueError(f'{name}: KronQ needs fp32 parameters (use autocast for bf16)')
                self.layers.append(_Layer(name, m, block, refresh, mom_bf16))
        if not self.layers: raise ValueError('no nn.Linear layer selected for KronQ')
        kron = {id(p) for L in self.layers for p in L.mod.parameters()}
        rest = [p for p in net.parameters() if p.requires_grad and id(p) not in kron]
        self.adam = None
        if rest:
            groups = [g for g in (dict(params=[p for p in rest if p.dim() >= 2], weight_decay=adam_wd),
                                  dict(params=[p for p in rest if p.dim() < 2], weight_decay=0.0)) if g['params']]
            self.lr_adam = lr if lr_adam is None else lr_adam
            self.adam = torch.optim.AdamW(groups, lr=self.lr_adam, betas=adam_betas, eps=adam_eps)
        self.kron_names = [L.name for L in self.layers]
        for L in self.layers: L.mod.register_forward_hook(self._hook(L))
        for m in net.modules():                                    # embedding outputs: the anchor the probe backward runs to
            if isinstance(m, nn.Embedding): m.register_forward_hook(self._emb_hook)

        dev = self.layers[0].mod.weight.device
        self.gen = torch.Generator(device=dev); self.gen.manual_seed(seed)     # probe labels and token subsets (same on every rank)
        self.t = 0; self._lr_t = torch.zeros((), device=dev)
        self._cap = None; self._ncap = 0; self._anchors = []; self._hooks = []

    # ---------------- capture (one micro-batch per statistics step)
    def stats_due(self): return self.t % max(1, self.h['stats_every']) == 0     # is the NEXT step a statistics step

    def _capturing(self):
        if not torch.is_grad_enabled() or not self.stats_due(): return False
        return True if self._cap is None else self._cap

    @contextlib.contextmanager
    def capture(self, on=True):
        """forwards inside take the statistics (only on statistics steps); forwards outside a capture block are not recorded
        once capture() has been used. Without capture(), every forward on a statistics step is recorded (one batch per step)."""
        prev = self._cap; self._cap = bool(on)
        try: yield
        finally: self._cap = prev if prev is not None else False

    def _rows(self, Z):
        n = self.stats_tokens
        if n and Z.shape[0] > n: Z = Z[torch.randperm(Z.shape[0], device=Z.device, generator=self.gen)[:n]]
        return Z

    def _hook(self, L):
        def h(m, inp, out):
            if not self._capturing(): return
            if L.z is not None or L.nA is not None:
                raise RuntimeError(f'{L.name}: recorded twice in one statistics step (weight sharing, or capture on more than one '
                                   'micro-batch) - not supported')
            with torch.no_grad(), _fp32(), _no_autocast(inp[0]):   # input factor now (in fp32 even under autocast)
                a = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                if L.has_bias: a = torch.cat([a, torch.ones(a.shape[0], 1, device=a.device)], 1)
                L.nA = _batch_factor(L.A, self._rows(a))
            L.z, L.order = out, self._ncap; self._ncap += 1
            if self.probe == 'emp': self._hooks.append(out.register_hook(self._s_hook(L)))
        return h

    def _emb_hook(self, m, inp, out):
        if self._capturing() and out.requires_grad: self._anchors.append(out)

    def _s_hook(self, L):
        def f(g):
            with torch.no_grad(), _fp32(), _no_autocast(g):
                L.nS = _batch_factor(L.S, self._rows(g.detach().reshape(-1, g.shape[-1]).float()))
        return f

    # ---------------- backward (+ the probe on the captured micro-batch)
    def _probe_loss(self, out):
        if self.loss == 'mse':                                   # Gauss-Newton of squared loss: random +-1 output signs (kronq_main)
            xi = self.draw_probe(out.shape) if self.draw_probe is not None else \
                torch.randint(0, 2, out.shape, device=out.device, generator=self.gen).to(out.dtype) * 2 - 1
            return (out * xi).sum()
        z = out.reshape(-1, out.shape[-1])                       # cross entropy: Monte Carlo Fisher, labels drawn from the softmax
        if self.probe_tokens and z.shape[0] > self.probe_tokens:
            z = z[torch.randperm(z.shape[0], device=z.device, generator=self.gen)[:self.probe_tokens]]
        with torch.no_grad():
            y = torch.multinomial(torch.softmax(z.detach().float(), -1), 1, generator=self.gen)[:, 0]
        return F.cross_entropy(z.float(), y, reduction='sum')

    def backward(self, loss, out=None):
        """loss.backward() for one (micro-)batch; on the captured micro-batch also the probe (Fisher / Gauss-Newton), which then
        needs `out` (the model output of that forward)."""
        captured = [L for L in self.layers if L.z is not None]
        if not captured or self.probe == 'emp':
            loss.backward()
            for hk in self._hooks: hk.remove()
            self._hooks = []
        else:
            if out is None: raise RuntimeError('this micro-batch was captured for statistics: pass its output, opt.backward(loss, out)')
            loss.backward(retain_graph=True)
            self._probe(out)
        for L in self.layers: L.z = None
        self._anchors = []

    def _probe(self, out):
        cap = [L for L in self.layers if L.z is not None]
        with torch.enable_grad():
            pl = self._probe_loss(out)
            if not self.stream:                                  # reference path: every layer-output gradient at once
                gz = torch.autograd.grad(pl, [L.z for L in cap])
                with torch.no_grad(), _fp32():
                    for L, g in zip(cap, gz): L.nS = _batch_factor(L.S, self._rows(g.reshape(-1, g.shape[-1]).float()))
                return
            hks = [L.z.register_hook(self._s_hook(L)) for L in cap]
            anchors = self._anchors or [min(cap, key=lambda L: L.order).z]   # embeddings, else the first layer's output
            try:
                torch.autograd.grad(pl, anchors, allow_unused=True)            # gradients flow through every layer; hooks consume them
            finally:
                for hk in hks: hk.remove()
        missing = [L.name for L in cap if L.nS is None]
        if missing: raise RuntimeError(f'probe gradient did not reach {missing[:3]}: these layers are not upstream of the model output '
                                       'through the embeddings / first layer; use stream=False')

    def zero_grad(self):
        for p in self.model.parameters(): p.grad = None

    # ---------------- statistics: average over ranks, then the EMA (as _Basis.update)
    def _ema(self, B, nF, beta):
        newF, newR = nF
        if self.world > 1:
            dist.all_reduce(newF); newF /= self.world
            if B.rem: dist.all_reduce(newR); newR /= self.world
        if B.F is None: B.F, B.Fr = newF, newR
        else:
            B.F.mul_(beta).add_(newF, alpha=1 - beta)
            if B.rem: B.Fr.mul_(beta).add_(newR, alpha=1 - beta)

    # ---------------- refresh (one owner rank per layer computes, then broadcast)
    def _sync(self, B, Uold, Urold, owner):
        if self.world > 1:
            if self.rank != owner:
                B.U = torch.empty(B.nf, B.b, B.b, device=B.dev); B.e = torch.empty(B.nf * B.b + B.rem, device=B.dev)
                if B.rem: B.Ur = torch.empty(B.rem, B.rem, device=B.dev)
            dist.broadcast(B.U, owner); dist.broadcast(B.e, owner)
            if B.rem: dist.broadcast(B.Ur, owner)
        V = B.U if Uold is None else torch.bmm(Uold.transpose(1, 2), B.U)     # as _Basis.refresh
        Vr = None
        if B.rem: Vr = B.Ur if Urold is None else Urold.T @ B.Ur
        return (V, Vr)

    def _push(self, L, g):
        if L.S.e is None or L.A.e is None: lam = torch.ones_like(g)
        else: lam = L.S.e[:, None] * L.A.e[None, :]
        return torch.where(lam > self.h['reltol'] * float(lam.max()), torch.exp(L.u.float()) * torch.sign(L.C.float()), torch.zeros_like(g))

    # ---------------- one step (after the backward passes of all micro-batches)
    def step(self, lr=None):
        h = self.h; self.t += 1; t = self.t
        lr = self.lr if lr is None else lr; frac = lr / self.lr_max
        if (t - 1) % max(1, h['stats_every']) == 0:
            pend = [L.name for L in self.layers if L.nA is None or L.nS is None]
            if pend: raise RuntimeError(f'statistics step {t} without statistics for {pend[:3]}: capture one micro-batch and pass its '
                                        'output to opt.backward(loss, out)')
            with torch.no_grad(), _fp32():
                for L in self.layers:
                    self._ema(L.A, L.nA, h['beta_stats']); self._ema(L.S, L.nS, h['beta_stats']); L.nA = L.nS = None
        with torch.no_grad():
            self._lr_t.fill_(float(lr))
            due = [l for l, L in enumerate(self.layers) if t >= L.next_ref and L.S.F is not None and L.A.F is not None]
            old = {}
            for l in due:                                            # phase 1: first-order decrease in the old basis; owners refresh
                L = self.layers[l]; ok = L.S.e is not None and L.A.e is not None; gd = 0.0
                if ok:
                    with _fp32(): G = L.to_basis(L.grad())
                    gd = float((G * self._push(L, G)).sum()); del G
                old[l] = (ok, gd, L.S.U, L.S.Ur, L.A.U, L.A.Ur)
                if l % self.world == self.rank:
                    with _fp32(): L.S.refresh(h['eig_iters']); L.A.refresh(h['eig_iters'])
            for l, L in enumerate(self.layers):                      # phase 2, one layer at a time: refresh bookkeeping, then the step
                if l in old:
                    ok, gd_old, SU, SUr, AU, AUr = old.pop(l); owner = l % self.world
                    with _fp32():
                        VS = self._sync(L.S, SU, SUr, owner); VA = self._sync(L.A, AU, AUr, owner)
                        tr = lambda T, sq=False: L.A.transport(L.S.transport(T, VS, 'L', sq), VA, 'R', sq)
                        L.C = tr(L.C.float()).to(L.C.dtype)
                        L.u = tr(L.u.float(), True).to(L.u.dtype)
                        L.hv = (L.S.transport(L.hv[0][:, None], VS, 'L', True)[:, 0], L.A.transport(L.hv[1][None, :], VA, 'R', True)[0])
                    L.has_prev = False
                    with _fp32(): G = L.to_basis(L.grad())
                    if ok and h['adapt_refresh'] > 0:
                        gd_new = float((G * self._push(L, G)).sum()); e_ = abs(gd_new - gd_old) / max(abs(gd_new), 1e-300)
                        L.period = int(min(max(L.period * min(max(h['adapt_refresh'] / max(e_, 1e-12), 0.5), 2.0), h['refresh_min']), 1000))
                    if self.world > 1:                               # the owner's period, so every rank refreshes on the same steps
                        p_ = torch.tensor([L.period], device=L.C.device); dist.broadcast(p_, owner); L.period = int(p_.item())
                    L.next_ref = t + L.period
                else:
                    with _fp32(): G = L.to_basis(L.grad())
                D = _update_c(G, L.C, L.u, L.hv[0], L.hv[1], L.has_prev, L.S.e, L.A.e, self._lr_t, self.lam_scale,
                              h['b1'], h['beta_u'], h['u_clamp'], h['reltol'])
                del G
                L.has_prev = True
                with _fp32(): L.apply(lr * L.from_basis(D))
                del D
            if h['weight_decay'] > 0:                                # decoupled decay on the KronQ weights (not biases), following the schedule
                gam = h['weight_decay'] * frac
                for L in self.layers: L.mod.weight.data.mul_(1.0 - gam)
        if self.adam is not None:
            for g in self.adam.param_groups: g['lr'] = self.lr_adam * frac
            self.adam.step()
        self._cap = None if self._cap is None else False; self._ncap = 0

    # ---------------- bookkeeping
    def state_bytes(self, split=False):
        """optimizer state in bytes: KronQ per-mode state, KronQ factors + bases, AdamW state."""
        n = lambda x: 0 if x is None else x.numel() * x.element_size()
        per_mode = sum(n(L.C) + n(L.u) + n(L.hv[0]) + n(L.hv[1]) for L in self.layers)
        factors = sum(L.A.nbytes() + L.S.nbytes() + n(L.A.e) + n(L.S.e) for L in self.layers)
        adam = sum(n(v) for st in self.adam.state.values() for v in st.values() if torch.is_tensor(v)) if self.adam is not None else 0
        return dict(per_mode=per_mode, factors=factors, adam=adam) if split else per_mode + factors + adam

    def u_stats(self):
        """fraction of learned log-weights at the clamp (upper, lower) and the median |u|, per KronQ layer."""
        c = self.h['u_clamp']; out = {}
        for L in self.layers:
            u = L.u.float()
            out[L.name] = dict(at_upper=float((u >= c - 1e-2).float().mean()), at_lower=float((u <= -c + 1e-2).float().mean()),
                               med_abs=float(u.abs().median()))
        return out
