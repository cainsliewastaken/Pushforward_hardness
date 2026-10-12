"""rollout_monitor: diagnostics of multistep (unrolled) training as a function of the number of physical steps.

Drop-in for any project with a one-step model written functionally,

    u_next = step_fn(params, u)          # u: [B, *state_shape], params: any pytree of tensors

(for an nn.Module use `from_module`). Everything is computed at the current parameters, for every window length
p = 1..N at once from one rollout of N physical steps, along one fixed parameter direction (by default the
descent direction of a long-horizon target loss, optionally on held-out starts). Notation follows the theory note
(multistep_training_theory.tex): window loss L_p = (1/p) sum_{q<=p} (1/2) mean_b |e_q|^2, rescaled gradients
g = Pi^{1/2} grad, update theta + z Pi^{1/2} d with |d| = 1.

Per physical step q (free or one forward-mode pass):
  err_rms, err_quantiles  distribution over starts of |e_q|
  tail_share              share of sum_b |e_q,b|^2 carried by the worst `tail_frac` of starts
  align                   a_q = -<e_q, Y1_q>: contribution of physical step q to the derivative of the window loss
                          along d (negative = including step q pushes toward the target's descent direction)
Per window length p (accumulated over q <= p):
  ell[m-1]                coefficients l_m(p) of L_p(theta + z Pi^{1/2} d) = sum_m l_m z^m, m = 1..order
                          (l_1 = <g_p, d>; l_2 = Gauss-Newton part + error-weighted part)
  gn_curv, ew_curv        the two parts of l_2
  rho_lin                 |l_1 / l_2|: update size at which the second-order term equals the first
  rho_fit                 radius of convergence in z estimated from the decay of |l_m|, m = 1..order
On a coarse grid of p (optional, a few backward / kernel products each):
  cos_target              cos(g_p, g_target): alignment of window p's own gradient with the target gradient
  higher_order_ratio      |g_p - g1_p| / |g1_p|, g1_p the first-order part (injections propagated by the
                          learned Jacobians along the TRUE trajectory)
  krylov_lo, krylov_up    lower bound (Krylov functionals) and upper bound (best step in a Krylov subspace) on
                          F_p(s) / ||e||_W, the smallest window error reachable by one update of rescaled length s,
                          with the error linearized in the update (the roughness paper's single-step value)
"""
import math
import numpy as np
import torch
from torch.func import jvp, vjp, grad as fgrad
from torch.utils._pytree import tree_map, tree_flatten


# ---------------------------------------------------------------------------------------------- pytree helpers
def _leaves(a):
    return tree_flatten(a)[0]


def _dot(a, b):
    return sum((x * y).sum() for x, y in zip(_leaves(a), _leaves(b)))


def _norm(a):
    return torch.sqrt(_dot(a, a))


def _mul(a, b):
    return tree_map(lambda x, y: x * y, a, b)


def _scale(a, c):
    return tree_map(lambda x: x * c, a)


def _axpy(a, b, c):                     # a + c * b
    return tree_map(lambda x, y: x + c * y, a, b)


def from_module(module, step=None):
    """(step_fn, params) for an nn.Module; `step(module_fn, u)` maps the module output to the next state
    (default: the module output itself). Example for an Euler model: step=lambda f, u: u + dt * f(u)."""
    from torch.func import functional_call
    params = {k: v.detach() for k, v in module.named_parameters()}
    def step_fn(p, u):
        f = lambda x: functional_call(module, p, (x,))
        return f(u) if step is None else step(f, u)
    return step_fn, params


# ---------------------------------------------------------------------------------------------- rollouts and losses
def rollout(step_fn, params, u0, n):
    """Predicted states u_1..u_n, shape [n, B, *S]."""
    out, u = [], u0
    for _ in range(n):
        u = step_fn(params, u)
        out.append(u)
    return torch.stack(out)


def window_loss(step_fn, params, u_true, p):
    """L_p = (1/p) sum_{q=1..p} (1/2) mean_b |e_q|^2 with u_true of shape [>=p+1, B, *S]."""
    e = u_true[1:p + 1] - rollout(step_fn, params, u_true[0], p)
    return 0.5 * (e * e).flatten(start_dim=2).sum(-1).mean(-1).sum() / p


def _mean_dot(a, b):                     # mean over starts of the state inner product, per time: [n, B, *S] -> [n]
    return (a * b).flatten(start_dim=2).sum(-1).mean(-1)


# ---------------------------------------------------------------------------------------------- the monitor
class RolloutMonitor:
    def __init__(self, step_fn, params, precond=None):
        """precond: optional pytree (same structure as params) of positive per-parameter step sizes Pi,
        e.g. Adam's current 1/(sqrt(v)+eps); default identity."""
        self.f, self.params = step_fn, tree_map(lambda x: x.detach(), params)
        self.sq = None if precond is None else tree_map(lambda x: x.detach().sqrt(), precond)

    def _rescale(self, g):              # Pi^{1/2} applied to a parameter-space tensor
        return g if self.sq is None else _mul(g, self.sq)

    def grad_window(self, u_true, p):
        """Rescaled gradient g_p = Pi^{1/2} grad L_p."""
        return self._rescale(fgrad(lambda P: window_loss(self.f, P, u_true, p))(self.params))

    def taylor(self, u0, n, d, order):
        """Sensitivities Y^(m) = d^m/dz^m rollout(theta + z Pi^{1/2} d) at z = 0, m = 1..order (nested forward mode)."""
        dtheta = self._rescale(d)
        z0 = torch.zeros((), dtype=_leaves(self.params)[0].dtype, device=_leaves(self.params)[0].device)
        def f(z):
            return rollout(self.f, _axpy(self.params, dtheta, z), u0, n)
        out, fun = [], f
        for _ in range(order):
            fun = (lambda g: (lambda z: jvp(g, (z,), (torch.ones_like(z),))[1]))(fun)
            out.append(fun(z0).detach())
        return out

    # -------------------------------------------------------------------------------- first-order gradient
    def first_order_grad(self, u_true, p):
        """g1_p = -Pi^{1/2} Phi^T L^T W L eps: one-step errors at the true states propagated by the learned
        Jacobians along the true trajectory (exact for the first-order part, any state shape)."""
        B = u_true.shape[1]
        P = self.params
        with torch.no_grad():
            eps = [u_true[i + 1] - self.f(P, u_true[i]) for i in range(p)]
        e1, E1 = torch.zeros_like(u_true[0]), []
        for i in range(p):
            Je = jvp(lambda u: self.f(P, u), (u_true[i],), (e1,))[1].detach() if i > 0 else torch.zeros_like(e1)
            e1 = Je + eps[i]
            E1.append(e1)
        y = [None] * p
        y[p - 1] = E1[p - 1] / (p * B)
        for i in range(p - 2, -1, -1):
            _, back = vjp(lambda u: self.f(P, u), u_true[i + 1])
            y[i] = E1[i] / (p * B) + back(y[i + 1])[0].detach()
        g1 = None
        for i in range(p):
            _, back = vjp(lambda Q: self.f(Q, u_true[i]), P)
            gi = back(y[i])[0]
            g1 = _scale(gi, -1.0) if g1 is None else _axpy(g1, gi, -1.0)
        return self._rescale(g1)

    # -------------------------------------------------------------------------------- Krylov single-step value
    def _kernel_ops(self, u_true, p):
        B, P, u0 = u_true.shape[1], self.params, u_true[0]
        w = torch.full((p,), 1.0 / p, dtype=u_true.dtype, device=u_true.device)
        def Wt(v):                       # W v for sampled functions v: [p, B, *S]
            return v * (w / B).view(-1, *([1] * (v.dim() - 1)))
        def ip(a, b):                    # <a, b>_W
            return (Wt(a) * b).sum()
        def PsiT_W(v):                   # g_v = Pi^{1/2} Psi^T W v
            _, back = vjp(lambda Q: rollout(self.f, Q, u0, p), P)
            return self._rescale(back(Wt(v))[0])
        def Psi(g):                      # Psi g = jvp of the rollout along Pi^{1/2} g
            return jvp(lambda Q: rollout(self.f, Q, u0, p), (P,), (self._rescale(g),))[1]
        return ip, PsiT_W, Psi

    def krylov_value(self, u_true, p, s, M=6):
        """Bounds on F_p(s)/||e||_W, F_p(s) = min_{|g|<=s} ||e - Psi g||_W (error linearized in the update).
        Lower: max over functionals phi in the Krylov space V_M of <phi,e> - s ||phi||_K, ||phi||_W <= 1 (each is a
        valid bound). Upper: best update of length <= s in span{Psi^T W K^j e}."""
        ip, PsiT_W, Psi = self._kernel_ops(u_true, p)
        with torch.no_grad():
            e = u_true[1:p + 1] - rollout(self.f, self.params, u_true[0], p)
        en = torch.sqrt(ip(e, e))
        Q, G_vec = [], []
        v = e
        for _ in range(M):               # Krylov basis of e, Ke, K^2 e, ... orthonormal in <,>_W
            for qv in Q:
                v = v - ip(qv, v) * qv
            nv = torch.sqrt(ip(v, v))
            if nv <= 1e-12 * en:
                break
            v = v / nv
            Q.append(v.detach())
            g = PsiT_W(v)
            G_vec.append(tree_map(lambda x: x.detach(), g))
            v = Psi(g).detach()          # K v = Psi Psi^T W v
        k = len(Q)
        G = torch.tensor([[float(_dot(G_vec[i], G_vec[j])) for j in range(k)] for i in range(k)], dtype=torch.float64)
        b = torch.tensor([float(ip(Q[i], e)) for i in range(k)], dtype=torch.float64)
        lo = 0.0
        for gam in [0.0] + list(10.0 ** np.linspace(-12, 4, 81) * float(G.diagonal().max().clamp_min(1e-300))):
            try:
                c = torch.linalg.solve(G + gam * torch.eye(k, dtype=G.dtype), b) if gam > 0 else b.clone()
            except Exception:
                continue
            c = c / c.norm().clamp_min(1e-300)
            lo = max(lo, float(b @ c - s * torch.sqrt((c @ G @ c).clamp_min(0))))
        # upper bound: updates g = sum beta_i G_vec[i]; residual ||e - Psi g||^2 = |e|^2 - 2 beta.h + beta^T A beta
        ge = PsiT_W(e)
        h = torch.tensor([float(_dot(G_vec[i], ge)) for i in range(k)], dtype=torch.float64)
        PsiG = [Psi(G_vec[i]).detach() for i in range(k)]
        A = torch.tensor([[float(ip(PsiG[i], PsiG[j])) for j in range(k)] for i in range(k)], dtype=torch.float64)
        e2 = float(ip(e, e))
        def resid(beta):
            return math.sqrt(max(e2 - 2 * float(beta @ h) + float(beta @ A @ beta), 0.0))
        def beta_of(lam):
            return torch.linalg.lstsq(A + lam * G, h.unsqueeze(1)).solution.squeeze(1)
        def length(beta):
            return math.sqrt(max(float(beta @ G @ beta), 0.0))
        up = math.sqrt(e2)
        b0 = beta_of(0.0)
        if length(b0) <= s:
            up = min(up, resid(b0))
        else:
            lo_l, hi_l = -12.0, 12.0
            scale = float(A.diagonal().max().clamp_min(1e-300)) / float(G.diagonal().max().clamp_min(1e-300))
            for _ in range(80):          # bisection on log lambda for |g| = s
                mid = 0.5 * (lo_l + hi_l)
                if length(beta_of(scale * 10 ** mid)) > s:
                    lo_l = mid
                else:
                    hi_l = mid
            bb = beta_of(scale * 10 ** hi_l)
            up = min(up, resid(bb))
        return lo / float(en), up / float(en)

    # -------------------------------------------------------------------------------- main entry point
    def profile(self, u_true, n, target_u_true=None, direction=None, order=4, quantiles=(0.5, 0.9, 0.99),
                tail_frac=0.01, grid=None, krylov=False, krylov_M=6, step_size=None, higher_order=False):
        """u_true: [>= n+1, B, *S] true trajectories of the starts to monitor; target_u_true: trajectories for the
        target loss L_n (e.g. held-out starts; default u_true). direction: rescaled unit pytree d, or None for the
        target's descent direction -g_target/|g_target|. grid: window lengths for the per-p extras."""
        out = {"n": n}
        tgt = u_true if target_u_true is None else target_u_true
        g_tgt = self.grad_window(tgt, n)
        out["target_loss"] = float(window_loss(self.f, self.params, tgt, n))
        out["target_grad_norm"] = float(_norm(g_tgt))
        d = _scale(g_tgt, -1.0 / float(_norm(g_tgt))) if direction is None else direction
        with torch.no_grad():
            pred = rollout(self.f, self.params, u_true[0], n)
            e = u_true[1:n + 1] - pred
            en = torch.sqrt((e * e).flatten(start_dim=2).sum(-1))             # [n, B]
        out["err_rms"] = torch.sqrt((en ** 2).mean(-1)).cpu().numpy()
        out["err_quantiles"] = {qq: torch.quantile(en, qq, dim=-1).cpu().numpy() for qq in quantiles}
        k = max(1, int(round(tail_frac * en.shape[1])))
        sq = en ** 2
        out["tail_share"] = (sq.topk(k, dim=-1).values.sum(-1) / sq.sum(-1).clamp_min(1e-300)).cpu().numpy()
        Y = self.taylor(u_true[0], n, d, order)
        fact = [math.factorial(m) for m in range(order + 1)]
        c = torch.zeros(order, n, dtype=e.dtype, device=e.device)             # per-step contributions to l_m
        for m in range(1, order + 1):
            c[m - 1] = -_mean_dot(e, Y[m - 1] / fact[m])
            for a in range(1, m):
                c[m - 1] += 0.5 * _mean_dot(Y[a - 1] / fact[a], Y[m - a - 1] / fact[m - a])
        steps = torch.arange(1, n + 1, dtype=e.dtype, device=e.device)
        ell = (c.cumsum(-1) / steps).cpu().numpy()                             # l_m(p), p = 1..n
        out["align"] = c[0].cpu().numpy()
        out["ell"] = ell
        if order >= 2:
            gn = (0.5 * _mean_dot(Y[0], Y[0])).cumsum(-1) / steps
            out["gn_curv"] = gn.cpu().numpy()
            out["ew_curv"] = ell[1] - out["gn_curv"]
            out["rho_lin"] = np.abs(ell[0]) / np.maximum(np.abs(ell[1]), 1e-300)
        if order >= 3:
            ms = np.arange(1, order + 1)
            rho = np.full(n, np.nan)
            for p in range(n):
                a = np.abs(ell[:, p])
                ok = a > 0
                if ok.sum() >= 3:
                    slope = np.polyfit(ms[ok], np.log(a[ok]), 1)[0]
                    rho[p] = math.exp(-slope)
            out["rho_fit"] = rho
        if grid:
            grid = [p for p in grid if 1 <= p <= n]
            out["grid"] = np.array(grid)
            cos, hor, klo, kup = [], [], [], []
            for p in grid:
                gp = self.grad_window(u_true, p)
                cos.append(float(_dot(gp, g_tgt) / (_norm(gp) * _norm(g_tgt))))
                if higher_order:
                    g1 = self.first_order_grad(u_true, p)
                    hor.append(float(_norm(_axpy(gp, g1, -1.0)) / _norm(g1)))
                if krylov:
                    s = step_size if step_size is not None else 0.5 * float(out["rho_lin"][p - 1])
                    lo, up = self.krylov_value(u_true, p, s, krylov_M)
                    klo.append(lo)
                    kup.append(up)
            out["cos_target"] = np.array(cos)
            if higher_order:
                out["higher_order_ratio"] = np.array(hor)
            if krylov:
                out["krylov_lo"], out["krylov_up"] = np.array(klo), np.array(kup)
        return out


def report(out, dt=None, every=1):
    """Plain-text table of a profile."""
    n = out["n"]
    t = (lambda p: f"{p * dt:6.2f}") if dt else (lambda p: f"{p:6d}")
    qs = sorted(out["err_quantiles"])
    hdr = (f"{'p' if not dt else 'p*dt':>6} {'rms|e|':>9} " + " ".join(f"{'q' + str(q):>9}" for q in qs)
           + f" {'tail':>6} {'align_q':>10} {'l1(p)':>10} {'l2(p)':>10} {'GN':>10} {'rho_lin':>9} {'rho_fit':>9}")
    lines = [f"target L_n = {out['target_loss']:.4g}, |g_target| = {out['target_grad_norm']:.4g}", hdr]
    for p in range(1, n + 1, every):
        i = p - 1
        row = f"{t(p)} {out['err_rms'][i]:9.3g} " + " ".join(f"{out['err_quantiles'][q][i]:9.3g}" for q in qs)
        row += f" {out['tail_share'][i]:6.2f} {out['align'][i]:10.3g} {out['ell'][0][i]:10.3g}"
        row += f" {out['ell'][1][i]:10.3g} {out['gn_curv'][i]:10.3g} {out['rho_lin'][i]:9.3g}" if "gn_curv" in out else ""
        row += f" {out['rho_fit'][i]:9.3g}" if "rho_fit" in out else ""
        lines.append(row)
    if "grid" in out:
        lines.append(f"\n{'p':>6} {'cos(g_p,g_target)':>18}" + (f" {'|g-g1|/|g1|':>12}" if "higher_order_ratio" in out else "")
                     + (f" {'F_lo/|e|':>9} {'F_up/|e|':>9}" if "krylov_lo" in out else ""))
        for j, p in enumerate(out["grid"]):
            row = f"{t(p)} {out['cos_target'][j]:18.3f}"
            if "higher_order_ratio" in out:
                row += f" {out['higher_order_ratio'][j]:12.3g}"
            if "krylov_lo" in out:
                row += f" {out['krylov_lo'][j]:9.3f} {out['krylov_up'][j]:9.3f}"
            lines.append(row)
    return "\n".join(lines)
