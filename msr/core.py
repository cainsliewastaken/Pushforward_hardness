"""Lorenz-63 truth, data, the two model classes, rollouts and losses (float64 throughout).

Notation follows README.md: u_T(t_i) true trajectory (RK4 truth map G), u_p(t_i) predicted
trajectory under the learned map Gt, e(t_i) = u_T(t_i) - u_p(t_i), eps_i = G(u_T(t_i)) - Gt(u_T(t_i)).
Norm over starts: |a| = (mean_k |a_k|^2)^(1/2).

Both model classes are written as Gt(u) = alpha*u + beta + gamma * N((u - mu)/sd; theta):
  euler : alpha = 1, beta = 0,  gamma = dt * sf   ->  Gt(u) = u + dt H(u),  H = sf * N
  direct: alpha = 0, beta = mu, gamma = sd        ->  Gt(u) = mu + sd * N
(mu, sd, sf fixed constants from the training data, so the network output is O(1)). The tendency
of either model is H(u) = (Gt(u) - u)/dt, written without cancellation for Euler.
"""
import math
import numpy as np
import torch
import torch.nn.functional as Fn
from torch.func import jacfwd, vmap

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------- truth
class Truth:
    def __init__(self, cfg):
        self.s, self.r, self.b = cfg["sigma"], cfg["rho"], cfg["beta"]
        self.sub = cfg["substeps"]

    def f(self, u):
        x, y, z = u[..., 0], u[..., 1], u[..., 2]
        return torch.stack([self.s * (y - x), x * (self.r - z) - y, x * y - self.b * z], -1)

    def G(self, u, dt):
        """RK4 truth map over one emulator step dt, with `substeps` RK4 substeps."""
        h = dt / self.sub
        for _ in range(self.sub):
            k1 = self.f(u)
            k2 = self.f(u + 0.5 * h * k1)
            k3 = self.f(u + 0.5 * h * k2)
            k4 = self.f(u + h * k3)
            u = u + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return u

    def traj(self, u0, dt, p):
        """(p+1, n, 3): u_T(t_0..t_p)."""
        out = [u0]
        for _ in range(p):
            out.append(self.G(out[-1], dt))
        return torch.stack(out)


def attractor_starts(truth, n, seed, transient, dev, dt_spin=0.01):
    """n independent states on the attractor: random initial conditions run for `transient` time units."""
    g = np.random.default_rng(seed)          # numpy: the first k of n draws equal a draw of k
    u = torch.tensor(g.standard_normal((n, 3)) * 8.0 + np.array([0.0, 0.0, 25.0]), device=dev)
    for _ in range(int(round(transient / dt_spin))):
        u = truth.G(u, dt_spin)
    return u


def nsteps(T, dt):
    p = int(round(T / dt))
    assert abs(p * dt - T) < 1e-9, (T, dt)
    return p


def train_steps(pr, dt):
    """Unrolled steps of a protocol: fixed physical window (T_train) or fixed step count (p_train)."""
    if "p_train" in pr:
        return pr["p_train"]
    return nsteps(pr["T_train"], dt) if "T_train" in pr else None


def make_data(cfg, dt, dev, train=True, stats=None, horizon=None):
    """Training starts with true trajectories over the longest training window at this dt, held-out
    starts with true trajectories up to `horizon` (default: the longest measurement horizon), and the
    dt-independent normalization (first 4096 training starts on a dt = 0.01 grid over `stats_window`)
    unless `stats` is given (measurements reuse the normalization saved with the checkpoints)."""
    tr = Truth(cfg)
    ptr = max([train_steps(pr, dt) or 0 for pr in cfg["protocols"].values()] + [nsteps(cfg["onestep_window"], dt)])
    if train or stats is None:
        u0_tr = attractor_starts(tr, cfg["n_train"] if train else 4096, cfg["data_seed_train"], cfg["transient"], dev)
        ref = tr.traj(u0_tr[:4096], 0.01, nsteps(cfg.get("stats_window", 0.64), 0.01)).reshape(-1, 3)
        stats = dict(mu=ref.mean(0), sd=ref.std(0), sf=tr.f(ref).std(0))
    u0_te = attractor_starts(tr, cfg["n_test"], cfg["data_seed_test"], cfg["transient"], dev)
    pmax = nsteps(horizon or max(cfg["horizons"]), dt)
    d = dict(truth=tr, dt=dt, pmax=pmax, stats=stats, uT_test=tr.traj(u0_te, dt, pmax))
    if train:
        d["uT_train"] = tr.traj(u0_tr, dt, ptr)
    return d


# ---------------------------------------------------------------- network and model classes
class Arch:
    """MLP 3 -> hidden... -> 3 with a smooth activation; parameters are one flat vector."""

    def __init__(self, hidden, act):
        self.sizes = [3] + list(hidden) + [3]
        self.act = dict(silu=Fn.silu, tanh=torch.tanh, gelu=Fn.gelu)[act]
        self.shapes = [(o, i) for i, o in zip(self.sizes[:-1], self.sizes[1:])]
        self.P = sum(o * i + o for o, i in self.shapes)

    def unpack(self, th):
        out, k = [], 0
        for o, i in self.shapes:
            W = th[k:k + o * i].view(o, i)
            k += o * i
            out.append((W, th[k:k + o]))
            k += o
        return out

    def init(self, seed, dev):
        """PyTorch-default uniform hidden layers, zero output layer."""
        g = torch.Generator().manual_seed(seed)
        parts = []
        for li, (o, i) in enumerate(self.shapes):
            bnd = 1 / math.sqrt(i)
            if li == len(self.shapes) - 1:
                parts += [torch.zeros(o * i), torch.zeros(o)]
            else:
                parts += [(torch.rand(o * i, generator=g) * 2 - 1) * bnd, (torch.rand(o, generator=g) * 2 - 1) * bnd]
        return torch.cat(parts).to(dev)

    def net(self, th, z):
        L = self.unpack(th)
        for W, b in L[:-1]:
            z = self.act(z @ W.T + b)
        W, b = L[-1]
        return z @ W.T + b


def model_consts(cls, dt, stats):
    mu, sd, sf = stats["mu"], stats["sd"], stats["sf"]
    if cls == "euler":
        return dict(alpha=torch.ones((), device=mu.device), beta=torch.zeros_like(mu), gamma=dt * sf, mu=mu, sd=sd)
    return dict(alpha=torch.zeros((), device=mu.device), beta=mu.clone(), gamma=sd.clone(), mu=mu, sd=sd)


def stack_consts(cs):
    return {k: torch.stack([c[k] for c in cs]) for k in cs[0]}


class Model:
    """One learned stepper: architecture + class constants. Functions take a flat theta (P,)."""

    def __init__(self, arch, cls, dt, stats):
        self.arch, self.cls, self.dt = arch, cls, dt
        self.c = model_consts(cls, dt, stats)
        self.P = arch.P

    def Gt(self, th, u):
        return gt(self.arch, th, u, self.c)

    def H(self, th, u):
        c = self.c
        N = self.arch.net(th, (u - c["mu"]) / c["sd"])
        if self.cls == "euler":
            return (c["gamma"] / self.dt) * N
        return ((c["alpha"] - 1) * u + c["beta"] + c["gamma"] * N) / self.dt

    def Jstate(self, th, u):
        """State Jacobian grad Gt(u), shape (..., 3, 3), by forward mode per sample."""
        U = u.reshape(-1, 3)
        J = vmap(jacfwd(lambda x: self.Gt(th, x)))(U)
        return J.reshape(*u.shape[:-1], 3, 3)

    def rollout(self, th, u0, p):
        out = [u0]
        for _ in range(p):
            out.append(self.Gt(th, out[-1]))
        return torch.stack(out)


def gt(arch, th, u, c):
    return c["alpha"] * u + c["beta"] + c["gamma"] * arch.net(th, (u - c["mu"]) / c["sd"])


# ---------------------------------------------------------------- losses (single model; vmapped over models)
def sq(a):
    """|a|^2 = mean_k |a_k|^2 over the start axis (second to last)."""
    return (a * a).sum(-1).mean(-1)


def loss_onestep(arch, th, c, X, Y):
    return 0.5 * sq(Y - gt(arch, th, X, c))


def loss_unrolled(arch, th, c, uT, p, w=None, cap=None):
    """L_p = sum_q w_q (1/2)|e(t_q)|^2 / sum_q w_q from the starts uT[0], q = 1..p. Default w_q = 1, i.e.
    (1/p) sum_q; damped training uses w_q = exp(-gamma t_q) (see loss_weights).
    cap (state units): a start contributes only up to the last step before its error first exceeds cap
    (the mask is not differentiated; the average over starts keeps the full start count)."""
    u, L, alive = uT[0], 0.0, None
    for q in range(1, p + 1):
        u = gt(arch, th, u, c)
        d = uT[q] - u
        if cap is not None:
            ok = (d.detach() ** 2).sum(-1) <= cap * cap
            alive = ok if alive is None else alive & ok
            d = torch.where(alive[..., None], d, torch.zeros_like(d))
        L = L + (1.0 if w is None else w[q - 1]) * 0.5 * sq(d)
    return L / (p if w is None else sum(w))


def loss_weights(gamma, dt, p):
    """Per-step weights exp(-gamma t_q), t_q = q dt: damping per unit physical time (alpha = exp(-gamma dt) per step)."""
    return None if not gamma else [math.exp(-gamma * q * dt) for q in range(1, p + 1)]


def gl_nodes(k, dev):
    x, w = np.polynomial.legendre.leggauss(k)
    return torch.tensor(0.5 * (x + 1), device=dev), torch.tensor(0.5 * w, device=dev)


def norm(a):
    return float(torch.sqrt(sq(a)))
