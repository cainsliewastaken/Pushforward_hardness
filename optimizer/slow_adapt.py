"""Slow adaptation of the target-dependent constants from measurements (2026-10-13; REPORT 61 item 41 plan).

The fixed constants of the reference configuration that the measurements say should move: the output budget (the
break-even 2 c_t falls 5-10x over a run, item 22 / 38), the position in the bracket (kappa_pred rises with the fast-mode
reversal, item 38) and the regime (output-side statistic; preflight.py sets it once at step 0). The tracked rule of item 39
(alignment every 10 steps, EMA 0.9, budget = m x 2 c directly) failed by lag early and by feedback late. This version
measures slowly, uses a margin and limits the rate of change:

every `every` steps a window of `win` steps on ONE fixed fresh batch (x_r, y_r) with the true kernel's top-k Ritz
directions P at the window start (Lanczos, one vjp + one jvp per matvec; then one jvp per step in the window) gives
  c_t    = cos(-E_r, J d_t)                       held-out alignment of the step (the break-even is 2 c_t);
  r_t    = -<P J d_{t+1}, P J d_t> / ||P J d_t||^2  fast-mode reversal (how much of a step's fast-mode output change the next
                                                   step cancels);
  ff     = ||P E_r||^2 / ||E_r||^2 at the window start  the regime measure of preflight.py (rough 0.004, smooth 0.026 at init),
           used as the concentration ff N / k over the uniform share k / N (field N = 65536: rough 8x, smooth 53x), so that
           problems with a small kernel (one output: N = batch) are measured on the same scale;
At the window end:
  kappa  = clip(1 / (2 (1 - median r)), 0.5, kappa_max)  the two-stage formula where there is reversal (r > 0), the one-step
                                                   optimum 1/2 where consecutive steps reinforce (r <= 0, item 38 smooth);
  budget = kappa x 2 x median c, clipped to [rem_min, rem_max], then limited to [1/2, 3/2] of the current budget;
  regime -> 'smooth' once ff N / k >= thresh (20) (one way: the held rough part is removed, not re-created), which switches the
           output-side statistic from the MC probe to the empirical signal (as preflight does for a smooth target).
The budget is applied by KronQFast as rem_dyn (the batch output change is held to it every step); the lr schedule is left
to --decay (linear: the budget mostly binds early; none: the budget is the only length control)."""
import math, torch
from torch.func import jvp, vjp
from mechanism import lanczos_topk, fwd_lins


class SlowAdapt:
    def __init__(self, act, L, dev, a, KQ, k=32, m=48, seed=29):
        self.act, self.L, self.dev, self.a, self.KQ, self.k, self.m = act, L, dev, a, KQ, k, m
        self.every, self.win = a.sa_every, a.sa_win
        self.gen = torch.Generator(device=dev); self.gen.manual_seed(seed)
        self.ritz = None; self.prev = None; self.cs = []; self.rs = []; self.ff = None; self.step_fast = []
        self.regime = 'rough'; self.rem = a.rem; self.nwin = 0
        KQ.rem_dyn = (a.rem if a.rem > 0 else None) if not a.sa_measure else None

    def f(self, plist, x): return fwd_lins(plist, x, self.act, self.L)
    def active(self, step): return (step - 1) % self.every < self.win
    def window_start(self, step): return (step - 1) % self.every == 0

    @torch.no_grad()
    def start(self, plist, xr, yr):
        self.xr, self.yr = xr, yr; pb = tuple(plist)
        with torch.enable_grad():
            out, vjp_fn = vjp(lambda *q: self.f(q, xr), *pb)
        N = out.numel(); E = (out - yr).detach().reshape(-1)
        def matvec(v):
            g = vjp_fn(v.reshape(out.shape))
            with torch.enable_grad():
                _, Kv = jvp(lambda *q: self.f(q, xr), pb, g)
            return Kv.reshape(-1)
        _, self.ritz = lanczos_topk(matvec, N, self.dev, self.m, self.k, self.gen)
        cE = self.ritz.T @ E; self.ff = float((cE @ cE) / (E @ E + 1e-30)); self.conc = self.ff * N / self.ritz.shape[1]   # concentration over the uniform share k / N
        self.prev = None; self.cs = []; self.rs = []; self.step_fast = []; self.E0 = E.clone(); self.Jds = []; self.ratios = []

    @torch.no_grad()
    def run(self, p_before, p_after, emit, step):
        pb = tuple(p_before); d = tuple(a_ - b_ for a_, b_ in zip(p_after, pb))
        with torch.enable_grad():
            out, Jd = jvp(lambda *q: self.f(q, self.xr), pb, d)
        E = (out - self.yr).reshape(-1); Jd = Jd.reshape(-1); coef = self.ritz.T @ Jd; PJd = self.ritz @ coef
        self.cs.append(float(-(E @ Jd) / (E.norm() * Jd.norm() + 1e-30)))
        self.step_fast.append(float((PJd @ PJd) / (Jd @ Jd + 1e-30)))
        if self.prev is not None: self.rs.append(float(-(PJd @ self.prev) / (self.prev @ self.prev + 1e-30)))
        self.prev = PJd; self.Jds.append(Jd)
        rho_b = getattr(self.KQ, 'last_rho', None)                      # the batch output change the budget acts on, vs the held-out one
        if rho_b is not None: self.ratios.append(rho_b / (float(Jd.norm() / (E.norm() + 1e-30)) + 1e-30))
        if (step - 1) % self.every == self.win - 1: self.update(emit, step)

    def window_kappas(self):
        """kappa over horizons W: the net first-order output displacement D of W consecutive steps against the residual at the
        block start (E_0 + the displacement so far), kappa_W = (||D|| / ||E||) / (2 cos(-E, D)); W = 1 is the per-step rule,
        W = 2 the two-stage formula's setting; medians over the non-overlapping blocks in the window."""
        out = {}
        for W in (1, 2, 5, 10, 20, 40):
            ks = []; E = self.E0.clone(); i = 0
            while i + W <= len(self.Jds):
                D = torch.stack(self.Jds[i:i + W]).sum(0); cD = float(-(E @ D) / (E.norm() * D.norm() + 1e-30)); rD = float(D.norm() / (E.norm() + 1e-30))
                ks.append(rD / (2.0 * cD) if cD > 1e-4 else float('inf')); E = E + D; i += W
            if ks: out[W] = float(torch.tensor(ks).median())
        return out

    def update(self, emit, step):
        a = self.a; self.nwin += 1
        for W, kW in self.window_kappas().items(): emit(f'sa_kappaW{W}', kW, step)
        if self.ratios: emit('sa_rho_batch_over_heldout', float(torch.tensor(self.ratios).median()), step)
        if a.sa_measure:                                                   # measurement only: nothing is changed
            c = float(torch.tensor(self.cs).median()); r = float(torch.tensor(self.rs).median()) if self.rs else 0.0
            emit('sa_c', c, step); emit('sa_r', r, step); emit('sa_fastfrac', self.ff, step); emit('sa_conc', self.conc, step)
            kw = self.window_kappas(); print(f"slow_adapt measure step {step}: c {c:.3f} r {r:+.3f} conc {self.conc:.1f}x; kappa_W " + ' '.join(f'{W}:{v:.2f}' for W, v in kw.items()) + (f"; rho batch/held-out {float(torch.tensor(self.ratios).median()):.2f}" if self.ratios else ''), flush=True)
            return
        c = float(torch.tensor(self.cs).median()); r = float(torch.tensor(self.rs).median()) if self.rs else 0.0
        kappa = min(a.sa_kappa_max, max(0.5, 1.0 / (2.0 * (1.0 - min(r, 0.49)))))
        target = min(a.sa_rem_max, max(a.sa_rem_min, kappa * 2.0 * c))
        if self.rem is not None and self.rem > 0: target = min(1.5 * self.rem, max(0.5 * self.rem, target))   # rate limit per window
        self.rem = target; self.KQ.rem_dyn = target
        switched = 0.0
        if self.regime == 'rough' and self.ff is not None and self.conc >= a.sa_thresh:
            self.regime = 'smooth'; switched = 1.0
            if a.S_mode != 'emp':
                a.S_mode = 'emp'; self.KQ.emp_scale = [None] * self.L     # the empirical signal is rescaled to the GN trace at the next statistics step
        for name, v in (('sa_c', c), ('sa_r', r), ('sa_kappa', kappa), ('sa_rem', target), ('sa_fastfrac', self.ff), ('sa_conc', self.conc),
                        ('sa_step_fastshare', float(torch.tensor(self.step_fast).median())), ('sa_smooth', 1.0 if self.regime == 'smooth' else 0.0), ('sa_switched', switched)):
            emit(name, v, step)
        print(f"slow_adapt step {step}: c {c:.3f} r {r:+.3f} kappa {kappa:.2f} -> budget {target:.3f}; residual fast fraction {self.ff:.4f} ({self.conc:.1f}x uniform) -> regime {self.regime}", flush=True)
