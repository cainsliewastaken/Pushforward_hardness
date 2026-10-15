"""KronMLM: TEST VARIANT of KronQLM (kronq_lm.py) with KronM's per-mode heavy-ball momentum (../kronm_main.py) in place of the learned
multiplier x sign. Reuses KronQLM unchanged (hooks, Conv2d as KFC, probe, statistics, refresh, decay, AdamW part) and swaps only the
per-layer update in the basis. The velocity is fp32 (mom_bf16 forced off)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kronq_lm
from kronq_lm import KronQLM
from kronm_main import make_update, swapped


class KronMLM(KronQLM):
    """the curvature of a Conv2d layer is KFC's: T (A x S) with T the number of output positions (Grosse & Martens 2016), since the
    statistics average over positions; KronQ never needed T (it uses lam only through lam / lam_max), the momentum step divides by it."""
    def __init__(self, *args, rule='crit', beta_max=0.999, **kw):
        kw['mom_bf16'] = False
        super().__init__(*args, **kw)
        self.rule, self.beta_max = rule, beta_max; inner = make_update(rule, beta_max)
        self.T = {L.name: 1 for L in self.layers}
        for L in self.layers:
            if L.conv: L.mod.register_forward_hook(lambda m, i, o, n=L.name: self.T.__setitem__(n, int(o.shape[-1] * o.shape[-2])))
        self._scl = None
        def upd(g, C, u, r_, c_, has_prev, Se, Ae, lr_t, lam_scale, *rest):     # called once per layer, in layer order
            return inner(g, C, u, r_, c_, has_prev, Se, Ae, lr_t, next(self._scl), *rest)
        self._upd = upd

    def _push(self, L, g):
        if L.S.e is None or L.A.e is None: return L.C.float()
        lam = L.S.e[:, None] * L.A.e[None, :]
        return torch.where(lam > self.h['reltol'] * float(lam.max()), L.C.float() / (float(lam.max()) * self.lam_scale * self.T[L.name] + 1e-30), torch.zeros_like(g))

    def step(self, lr=None):
        dev = self.layers[0].C.device
        self._scl = iter([torch.tensor(float(self.lam_scale * self.T[L.name]), device=dev) for L in self.layers])
        with swapped(kronq_lm, self._upd): super().step(lr)
