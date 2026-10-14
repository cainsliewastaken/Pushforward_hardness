"""Per-step mechanism diagnostics for any optimizer in scale_study.py (2026-10-13; REPORT 61 item 21 plan).
Everything is computed in the ORIGINAL weight coordinates (list [W0, b0, W1, b1, ...]) with a functional forward, so the
step d = theta_after - theta_before is meaningful for the rotated fast path as well (its weights are mapped back).
At a diagnostic step, on the training batch:
  1. loss decomposition: dL_actual = L(theta_after) - L(theta_before); first = 2 <E, J d> / N; gn = ||J d||^2 / N;
     remainder = dL_actual - first - gn  (one jvp + one forward);
  2. residual split by the current kernel: Lanczos (m matvecs of K = J J^T, each one vjp + one jvp) -> top-k Ritz pairs;
     fastfrac = energy of E on the top-k Ritz vectors / ||E||^2; lam_1, lam_k; rayleigh = <E, K E> / ||E||^2;
  3. leverage of the residual direction E_hat: ||J^T E_hat|| before and after the step (two vjps); its change split into
     Q = ||(J_after - J_before)^T E_hat||^2 and Cross = total - Q (the stage identity on E_hat);
  4. minibatch signal-to-noise of the gradient from two half-batches: ||g1 + g2|| / ||g1 - g2||, overall and (fast path)
     per lambda decile in the Kronecker basis;
  5. alignment cos(E, J d) and ||J d|| / ||E||."""
import math, torch
from torch.func import jvp, vjp

def fwd_lins(plist, x, act, L):
    h = x
    for l in range(L):
        z = h @ plist[2 * l].T + plist[2 * l + 1]; h = act(z) if l < L - 1 else z
    return h

def lanczos_topk(matvec, N, dev, m=48, k=32, gen=None):
    V = []; al = []; be = []
    v = torch.randn(N, device=dev, generator=gen); v = v / v.norm(); V.append(v); beta = 0.0; vprev = None
    for j in range(m):
        w = matvec(V[j])
        if vprev is not None: w = w - beta * vprev
        alpha = float(w @ V[j]); w = w - alpha * V[j]
        for _ in range(2):                            # full reorthogonalisation, twice (once is not enough near breakdown in float32)
            for u in V: w = w - float(w @ u) * u
        beta = float(w.norm()); al.append(alpha)
        if beta < 1e-6 * max(abs(x) for x in al) or j == m - 1: break   # relative breakdown test (small kernels, e.g. one output, exhaust the Krylov space)
        be.append(beta); vprev = V[j]; V.append(w / beta)
    n = len(al); T = torch.diag(torch.tensor(al, dtype=torch.float64)) + torch.diag(torch.tensor(be[:n - 1], dtype=torch.float64), 1) + torch.diag(torch.tensor(be[:n - 1], dtype=torch.float64), -1)
    ev, Q = torch.linalg.eigh(T); kk = min(k, n); idx = torch.argsort(ev, descending=True)[:kk]
    Vm = torch.stack(V[:n], 1)                           # N x n
    ritz = Vm @ Q[:, idx].to(Vm.dtype).to(dev)         # N x kk
    ritz, _ = torch.linalg.qr(ritz)                    # exact orthonormality (projections onto the Ritz space must be <= 1)
    return ev[idx].to(dev, torch.float32), ritz

class Mechanism:
    def __init__(self, act, L, dev, k=32, m=48, seed=7):
        self.act, self.L, self.dev, self.k, self.m = act, L, dev, k, m
        self.gen = torch.Generator(device=dev); self.gen.manual_seed(seed)
    def f(self, plist, x): return fwd_lins(plist, x, self.act, self.L)
    @torch.no_grad()
    def run(self, p_before, p_after, xb, yb, loss_fn, emit, step, rotate=None, xv=None, yv=None):
        """p_before / p_after: lists of tensors in original coordinates (same layout). rotate: optional fn(G_aug_list) ->
        (rotated gradient list, lam list) for the per-decile SNR in the fast path's basis."""
        N = yb.numel(); pb = tuple(p_before); pa = tuple(p_after); d = tuple(a_ - b_ for a_, b_ in zip(pa, pb))
        with torch.enable_grad():
            out_b, vjp_b = vjp(lambda *q: self.f(q, xb), *pb)
        E = (out_b - yb).detach(); nE = float(E.norm()); Lb = float(loss_fn(out_b, yb))
        with torch.enable_grad():
            _, Jd = jvp(lambda *q: self.f(q, xb), pb, d)
        out_a = self.f(pa, xb); La = float(loss_fn(out_a, yb))
        first = 2.0 * float((E * Jd).sum()) / N; gn = float((Jd * Jd).sum()) / N; dL = La - Lb
        emit('mech_dL', dL, step); emit('mech_first', first, step); emit('mech_gn', gn, step); emit('mech_rem', dL - first - gn, step)
        emit('mech_cosE', -float((E * Jd).sum()) / (nE * float(Jd.norm()) + 1e-30), step); emit('mech_Jd_over_E', float(Jd.norm()) / (nE + 1e-30), step)
        emit('mech_dnorm', math.sqrt(sum(float((x_ * x_).sum()) for x_ in d)), step); emit('mech_L_before', Lb, step)
        # 1b. the same on held-out data: the population-side first-order gain and alignment of the step
        if xv is not None:
            with torch.enable_grad():
                out_vb, Jd_v = jvp(lambda *q: self.f(q, xv), pb, d)
            Ev = (out_vb - yv).detach(); nEv = float(Ev.norm()); Nv = yv.numel(); out_va = self.f(pa, xv)
            emit('mechv_dL', float(loss_fn(out_va, yv)) - float(loss_fn(out_vb, yv)), step)
            emit('mechv_first', 2.0 * float((Ev * Jd_v).sum()) / Nv, step); emit('mechv_gn', float((Jd_v * Jd_v).sum()) / Nv, step)
            emit('mechv_cosE', -float((Ev * Jd_v).sum()) / (nEv * float(Jd_v.norm()) + 1e-30), step); emit('mechv_Jd_over_E', float(Jd_v.norm()) / (nEv + 1e-30), step)
            # alignment of the batch output response with the population residual, and of the batch residual with it
            emit('mechv_cosEb_Ev', 0.0, step)
        # 2. kernel split (at theta_before)
        def matvec(v):
            g = vjp_b(v.reshape(E.shape))
            with torch.enable_grad():
                _, Kv = jvp(lambda *q: self.f(q, xb), pb, g)
            return Kv.reshape(-1)
        ev, ritz = lanczos_topk(matvec, N, self.dev, self.m, self.k, self.gen)
        Ef = E.reshape(-1); coef = ritz.T @ Ef; fast = float((coef * coef).sum()) / (nE ** 2 + 1e-30)
        emit('mech_fastfrac', fast, step); emit('mech_lam1', float(ev[0]), step); emit('mech_lamk', float(ev[-1]), step)
        KE = matvec(Ef); emit('mech_rayleigh', float(Ef @ KE) / (nE ** 2 + 1e-30), step)
        # the step's output response: fraction on the fast subspace
        Jf = Jd.reshape(-1); cj = ritz.T @ Jf; emit('mech_Jd_fastfrac', float((cj * cj).sum()) / (float(Jf @ Jf) + 1e-30), step)
        # first-order decrease split: fast / slow part of E
        first_fast = -2.0 * float(coef @ cj) / N; emit('mech_first_fast', first_fast, step); emit('mech_first_slow', -first - first_fast if False else (-first) - first_fast, step)
        # 3. leverage of E_hat before and after
        Eh = (E / (nE + 1e-30))
        gb = vjp_b(Eh); lev_b2 = sum(float((x_ * x_).sum()) for x_ in gb)
        with torch.enable_grad():
            _, vjp_a = vjp(lambda *q: self.f(q, xb), *pa)
        ga = vjp_a(Eh); lev_a2 = sum(float((x_ * x_).sum()) for x_ in ga)
        Q = sum(float(((x_ - y_) ** 2).sum()) for x_, y_ in zip(ga, gb))
        emit('mech_levE', math.sqrt(lev_b2), step); emit('mech_dlev2', lev_a2 - lev_b2, step); emit('mech_Q_E', Q, step); emit('mech_cross_E', lev_a2 - lev_b2 - Q, step)
        # 4. minibatch SNR at theta_after from two half-batches
        m = xb.shape[0] // 2
        with torch.enable_grad():
            pa_r = tuple(p.detach().clone().requires_grad_(True) for p in pa)
            g1 = torch.autograd.grad(loss_fn(self.f(pa_r, xb[:m]), yb[:m]), pa_r)
            g2 = torch.autograd.grad(loss_fn(self.f(pa_r, xb[m:]), yb[m:]), pa_r)
        s2 = sum(float(((x_ + y_) ** 2).sum()) for x_, y_ in zip(g1, g2)); n2 = sum(float(((x_ - y_) ** 2).sum()) for x_, y_ in zip(g1, g2))
        emit('mech_snr', math.sqrt(s2 / (n2 + 1e-30)), step)
        if rotate is not None:
            G1 = [torch.cat([g1[2 * l], g1[2 * l + 1][:, None]], 1) for l in range(self.L)]; G2 = [torch.cat([g2[2 * l], g2[2 * l + 1][:, None]], 1) for l in range(self.L)]
            R1, lams = rotate(G1); R2, _ = rotate(G2)
            for l in range(self.L - 1):
                if lams[l] is None: continue
                order = torch.argsort(lams[l].flatten(), descending=True); n = order.numel(); r1 = R1[l].flatten(); r2 = R2[l].flatten()
                for dec in range(10):
                    seg = order[dec * n // 10:(dec + 1) * n // 10]
                    emit(f'mech_snr_d{dec}_l{l}', math.sqrt(float(((r1[seg] + r2[seg]) ** 2).sum()) / (float(((r1[seg] - r2[seg]) ** 2).sum()) + 1e-30)), step)


class KappaDiag:
    """Reversal fraction r of the chain's second stage (side-chat plan, 2026-10-13): on a fixed fresh batch x_r and the
    top-k directions of the true kernel at the window start, r_t = -<P J d_{t+1}, P J d_t> / ||P J d_t||^2 (how much of step t's
    fast-mode output change step t+1 cancels), plus the same without the projection, the held-out alignment of each step on
    the fast / slow parts, and kappa_pred = 1 / (2 (1 - r)). One jvp per step in the window; one Lanczos at the start."""
    def __init__(self, act, L, dev, k=32, m=48, seed=23):
        self.act, self.L, self.dev, self.k, self.m = act, L, dev, k, m
        self.gen = torch.Generator(device=dev); self.gen.manual_seed(seed); self.ritz = None; self.prev = None; self.xr = None
    def f(self, plist, x): return fwd_lins(plist, x, self.act, self.L)
    @torch.no_grad()
    def start(self, plist, xr, yr):
        self.xr, self.yr = xr, yr; pb = tuple(plist)
        with torch.enable_grad():
            out, vjp_fn = vjp(lambda *q: self.f(q, xr), *pb)
        N = out.numel()
        def matvec(v):
            g = vjp_fn(v.reshape(out.shape))
            with torch.enable_grad():
                _, Kv = jvp(lambda *q: self.f(q, xr), pb, g)
            return Kv.reshape(-1)
        _, self.ritz = lanczos_topk(matvec, N, self.dev, self.m, self.k, self.gen); self.prev = None
    @torch.no_grad()
    def run(self, p_before, p_after, emit, step):
        pb = tuple(p_before); d = tuple(a_ - b_ for a_, b_ in zip(p_after, pb))
        with torch.enable_grad():
            out, Jd = jvp(lambda *q: self.f(q, self.xr), pb, d)
        E = (out - self.yr).reshape(-1); Jd = Jd.reshape(-1); coef = self.ritz.T @ Jd; PJd = self.ritz @ coef
        cE = self.ritz.T @ E; E_fast = self.ritz @ cE; E_slow = E - E_fast; Jd_slow = Jd - PJd
        emit('kd_cos_fast', float(-(E_fast @ PJd) / (E_fast.norm() * PJd.norm() + 1e-30)), step)
        emit('kd_cos_slow', float(-(E_slow @ Jd_slow) / (E_slow.norm() * Jd_slow.norm() + 1e-30)), step)
        emit('kd_cos_all', float(-(E @ Jd) / (E.norm() * Jd.norm() + 1e-30)), step)
        emit('kd_fast_energy_frac', float((PJd @ PJd) / (Jd @ Jd + 1e-30)), step)
        if self.prev is not None:
            pJ, pP = self.prev
            emit('kd_r_fast', float(-(PJd @ pP) / (pP @ pP + 1e-30)), step); emit('kd_r_all', float(-(Jd @ pJ) / (pJ @ pJ + 1e-30)), step)
        self.prev = (Jd, PJd)
