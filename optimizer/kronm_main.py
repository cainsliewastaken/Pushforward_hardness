"""KronM: TEST VARIANT of the main optimizer (2026-10-14). Per-mode heavy-ball momentum in KronQ's Kronecker eigenbasis, in place of
KronQ's learned multiplier x sign. Does not edit kronq_main.py or llm/kronq_lm.py: it reuses their classes and swaps only the per-layer
update in the basis, so the basis, statistics, refresh, state transport, numerics and decoupled weight decay are KronQ's own.

The update, per layer, in the basis (lam_k = eS_i eA_j x lam_scale, the Kronecker curvature of mode k; lam_max its layer maximum):
  a_k    = min(1, lr_t lam_k / lam_max)                    (lr_t the scheduled lr, read as a fraction of the top mode's Newton step)
  beta_k = min((1 - sqrt(a_k))^2, beta_max)   rule 'crit'  (critical damping of heavy ball on a mode of curvature lam_k at step lr/lam_max)
  beta_k = b1                                 rule 'const' (control: one momentum for every mode)
  v_k    = beta_k v_k + g_k                                (heavy ball; the gradient's magnitude is kept, no sign)
  step   = lr v_k / lam_max                                (on live modes, lam_k > reltol lam_max; zero on the others)
So lr is dimensionless: lr = 1 is a full Newton step on the layer's stiffest mode. The velocity is kept in fp32 (a bf16 velocity cannot
represent beta near 1: 0.999 x v rounds back to v). State: the velocity only (u and the hypergradient normaliser stay unused at zero).
"""
import contextlib, functools, torch
import kronq_main
from kronq_main import KronQ


def _update_m(g, C, u, r_, c_, has_prev, Se, Ae, lr_t, lam_scale, b1, beta_u, u_clamp, reltol, rule='crit', beta_max=0.999):
    """same signature as kronq_main._update (u, r_, c_, has_prev, beta_u, u_clamp unused); C (fp32) updated in place; returns D."""
    lam = Se[:, None] * Ae[None, :] * lam_scale; lmax = lam.max().clamp(min=1e-30); live = lam > reltol * lmax
    if rule == 'crit':
        a = (lr_t * lam / lmax).clamp(0.0, 1.0); beta = ((1.0 - a.sqrt()) ** 2).clamp(max=beta_max)
        C.mul_(beta).add_(g)
    else:
        C.mul_(b1).add_(g)
    return torch.where(live, C / lmax, torch.zeros_like(C))


def make_update(rule, beta_max):
    f = functools.partial(_update_m, rule=rule, beta_max=beta_max)
    try: return torch.compile(f, dynamic=False)
    except Exception: return f


@contextlib.contextmanager
def swapped(module, fn):
    """module._update_c -> fn for the duration (the host class looks the name up at call time)."""
    prev = module._update_c; module._update_c = fn
    try: yield
    finally: module._update_c = prev


class KronM(KronQ):
    """kronq_main.KronQ with the per-mode momentum update (module docstring). Same constructor plus rule, beta_max; mom_bf16 is forced off."""
    def __init__(self, *args, rule='crit', beta_max=0.999, **kw):
        kw['mom_bf16'] = False
        super().__init__(*args, **kw)
        self.rule, self.beta_max = rule, beta_max; self._upd = make_update(rule, beta_max)

    def _push(self, l, g):
        """the current step direction (units of lr), for KronQ's adaptive refresh criterion."""
        if self.S[l].e is None or self.A[l].e is None: return self.C[l].float()
        lam = self.S[l].e[:, None] * self.A[l].e[None, :]
        return torch.where(lam > self.h['reltol'] * float(lam.max()), self.C[l].float() / (float(lam.max()) * 2.0 / self.nout + 1e-30), torch.zeros_like(g))

    def step(self, lr, out):
        with swapped(kronq_main, self._upd): super().step(lr, out)

    def m_stats(self):
        """per layer: median and max |velocity| (diagnostic)."""
        return {l: (float(C.float().abs().median()), float(C.float().abs().max())) for l, C in enumerate(self.C)}
