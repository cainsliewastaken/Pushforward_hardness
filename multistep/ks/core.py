"""Kuramoto-Sivashinsky truth (ETDRK4, periodic [0, L)), data, the Euler-class MLP stepper and the training losses.
Same conventions as msr/core.py (Lorenz-63): float64, flat parameter vector, Gt(u) = u + dt H(u), H = sf N((u - mu)/sd)
with scalar normalization constants, one-step error eps = G(uT) - Gt(uT), rollout error e = uT - u_pred.
"""
import math

import torch
import torch.nn.functional as Fn
from torch.func import jacfwd, vmap

torch.set_default_dtype(torch.float64)


class KS:
    """u_t = -u u_x - u_xx - u_xxxx on [0, L) with N grid points; ETDRK4 (Kassam & Trefethen 2005) with step h."""

    def __init__(self, L=22.0, N=64, h=0.25, dev="cpu"):
        self.L, self.N, self.h = float(L), int(N), float(h)
        k = 2 * math.pi * torch.arange(N // 2 + 1, device=dev) / L
        Lh = k ** 2 - k ** 4
        self.k, self.Lh = k, Lh
        self.E, self.E2 = torch.exp(h * Lh), torch.exp(h * Lh / 2)
        M = 32
        r = torch.exp(1j * math.pi * (torch.arange(1, M + 1, device=dev) - 0.5) / M)
        LR = h * Lh[:, None].to(torch.complex128) + r[None, :]
        eLR = torch.exp(LR)
        self.Q = (h * ((torch.exp(LR / 2) - 1) / LR).mean(1)).real
        self.f1 = (h * ((-4 - LR + eLR * (4 - 3 * LR + LR ** 2)) / LR ** 3).mean(1)).real
        self.f2 = (h * ((2 + LR + eLR * (-2 + LR)) / LR ** 3).mean(1)).real
        self.f3 = (h * ((-4 - 3 * LR - LR ** 2 + eLR * (4 - LR)) / LR ** 3).mean(1)).real
        self.g = -0.5j * k

    def _nl(self, v):
        u = torch.fft.irfft(v, n=self.N)
        return self.g * torch.fft.rfft(u * u)

    def _step(self, v):
        Nv = self._nl(v)
        a = self.E2 * v + self.Q * Nv
        Na = self._nl(a)
        b = self.E2 * v + self.Q * Na
        Nb = self._nl(b)
        c = self.E2 * a + self.Q * (2 * Nb - Nv)
        Nc = self._nl(c)
        return self.E * v + Nv * self.f1 + 2 * (Na + Nb) * self.f2 + Nc * self.f3

    def G(self, u, dt):
        """Truth map over one emulator step dt (a whole number of solver steps h)."""
        sub = int(round(dt / self.h))
        assert abs(sub * self.h - dt) < 1e-12, (dt, self.h)
        v = torch.fft.rfft(u)
        for _ in range(sub):
            v = self._step(v)
        return torch.fft.irfft(v, n=self.N)

    def f(self, u):
        """Tendency u_t (for the normalization constant sf and sanity checks)."""
        v = torch.fft.rfft(u)
        return torch.fft.irfft(self.Lh * v + self._nl(v), n=self.N)

    def traj(self, u0, dt, p):
        out = [u0]
        for _ in range(p):
            out.append(self.G(out[-1], dt))
        return torch.stack(out)

    def jac_fd(self, u, dt, eps=1e-6):
        """Jacobian of G(., dt) at each row of u (n, N) by central differences, (n, N, N); J[:, :, j] = dG/du_j."""
        n, N = u.shape
        I = torch.eye(N, device=u.device, dtype=u.dtype)
        up = u[:, None, :] + eps * I[None]
        um = u[:, None, :] - eps * I[None]
        d = (self.G(up.reshape(-1, N), dt) - self.G(um.reshape(-1, N), dt)) / (2 * eps)
        return d.reshape(n, N, N).transpose(1, 2)


def attractor_starts(ks, n, seed, transient, dev):
    """n states on the attractor: random smooth initial fields run for `transient` time units."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.arange(ks.N, device=dev) * ks.L / ks.N
    u = torch.zeros(n, ks.N, device=dev)
    for m in range(1, 5):
        a = torch.randn(n, 2, generator=g).to(dev)
        u = u + a[:, :1] * torch.cos(2 * math.pi * m * x / ks.L) + a[:, 1:] * torch.sin(2 * math.pi * m * x / ks.L)
    for _ in range(int(round(transient / ks.h))):
        u = ks.G(u, ks.h)
    return u


def make_data(cfg, dev, horizon=None, train=True, stats=None):
    """Training starts with true trajectories over the longest training window (p_train emulator steps) and held-out
    starts with trajectories up to `horizon` (time units); scalar normalization (mu, sd, sf) from the training data."""
    ks = KS(cfg["L"], cfg["N"], cfg["h"], dev)
    dt = cfg["dt"]
    d = dict(ks=ks, dt=dt)
    if train or stats is None:
        u0 = attractor_starts(ks, cfg["n_train"] if train else 4096, cfg["seed_train"], cfg["transient"], dev)
        ref = ks.traj(u0[:2048], dt, cfg["p_train"])
        stats = dict(mu=ref.mean(), sd=ref.std(), sf=ks.f(ref.reshape(-1, ks.N)).std())
        if train:
            d["uT_train"] = ks.traj(u0, dt, cfg["p_train"])
    d["stats"] = stats
    u0t = attractor_starts(ks, cfg["n_test"], cfg["seed_test"], cfg["transient"], dev)
    pmax = int(round((horizon or cfg["horizon"]) / dt))
    d["uT_test"] = ks.traj(u0t, dt, pmax)
    return d


class Arch:
    """MLP N -> hidden... -> N with a smooth activation; parameters are one flat vector."""

    def __init__(self, N, hidden, act="silu"):
        self.sizes = [N] + list(hidden) + [N]
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


class Model:
    """Euler-class stepper Gt(u) = u + dt sf N((u - mu)/sd; theta)."""

    def __init__(self, arch, dt, stats):
        self.arch, self.dt, self.c = arch, dt, {k: v.clone() for k, v in stats.items()}

    def Gt(self, th, u):
        c = self.c
        return u + self.dt * c["sf"] * self.arch.net(th, (u - c["mu"]) / c["sd"])

    def Jstate(self, th, u):
        return vmap(jacfwd(lambda x: self.Gt(th, x)))(u)

    def rollout(self, th, u0, p):
        out = [u0]
        for _ in range(p):
            out.append(self.Gt(th, out[-1]))
        return torch.stack(out)


def sq(a):
    """|a|^2 = mean over starts of |a_k|^2 (rms-over-starts convention of the Lorenz code)."""
    return (a * a).sum(-1).mean(-1)


def rollout_kappa(model, th, uT, p, kap):
    """Rows 1..p of the nudged rollout: x_i = kap uT_i + (1 - kap) u_i before each step, gradient through (1-kap) u_i.
    kap None: free rollout. kap float: the same weight at every step."""
    out, u = [], uT[0]
    for q in range(1, p + 1):
        if kap is not None and q > 1:
            u = kap * uT[q - 1] + (1 - kap) * u
        u = model.Gt(th, u)
        out.append(u)
    return torch.stack(out)


def loss_kappa(model, th, uT, p, kap):
    """(1/p) sum_q (1/2)|uT_q - row q|^2; kap = 1 is the one-step loss on the window's states, kap = 0 the free loss."""
    seq = rollout_kappa(model, th, uT, p, kap)
    return 0.5 * sq(uT[1:p + 1] - seq).mean()


@torch.no_grad()
def eval_free(model, th, uT, p):
    """Per-start free rollout errors |e(t_q)|, shape (p, n)."""
    u, errs = uT[0], []
    for q in range(1, p + 1):
        u = model.Gt(th, u)
        errs.append(torch.sqrt(((uT[q] - u) ** 2).sum(-1)))
    return torch.stack(errs)
