"""2D turbulence prototype: data, the Euler-class FNO stepper with a flat parameter vector, losses, evaluation.
Conventions as in ks/core.py: Gt(w) = w + dt * sf * N((w - mu)/sd; theta), eps = G(uT) - Gt(uT), e = uT - u_pred.
Fields are (…, N, N) real; the model and data are float32 for memory, the truth map runs in float64.
"""
import math
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as Fn

from turb2d.solver import Turb2D


# ---------------------------------------------------------------- data
def truncate(w, n):
    """Spectral truncation of (…, M, M) fields to n x n (M >= n)."""
    W = torch.fft.rfft2(w.double())
    k = n // 2
    out = torch.zeros(*w.shape[:-2], n, n // 2 + 1, dtype=W.dtype, device=w.device)
    out[..., :k, : k + 1] = W[..., :k, : k + 1]
    out[..., -k:, : k + 1] = W[..., -k:, : k + 1]
    return torch.fft.irfft2(out, s=(n, n)) * (n / w.shape[-1]) ** 2


def starts_from_h5(path, n, seed, stride, lo_frac=0.0, hi_frac=1.0, dev="cpu"):
    """n snapshots from the stored 256^2 series, within the index range [lo_frac, hi_frac) of the series and at
    least `stride` indices apart, in sorted order."""
    with h5py.File(path) as h:
        T = h["omega"].shape[0]
        lo, hi = int(lo_frac * T), int(hi_frac * T)
        g = np.random.default_rng(seed)
        idx = np.sort(g.choice(np.arange(lo, hi, stride), size=n, replace=False))
        w = np.stack([h["omega"][int(i)] for i in idx]).astype(np.float64)
    return torch.tensor(w, device=dev), idx


def make_data(cfg, dev, train=True, stats=None, horizon=None, batch=256):
    """Starts: stored 256^2 snapshots (train / test from disjoint halves of the series), truncated to N, then
    re-equilibrated on the N^2 attractor for `settle` time units. Trajectories over p_train (train) or `horizon`
    (test) emulator steps dt. Scalar normalization (mu, sd, sf) from the training trajectories. float32 output."""
    ks = Turb2D(cfg["N"], cfg["Re"], cfg["fkx"], cfg["fky"], cfg["alpha"], cfg["beta"], cfg["h"], dev)
    dt = cfg["dt"]
    d = dict(ks=ks, dt=dt)

    def prep(n, seed, lo_frac, hi_frac, p):
        w, _ = starts_from_h5(cfg["h5"], n, seed, cfg["stride"], lo_frac, hi_frac, dev="cpu")
        w = truncate(w, cfg["N"]).to(dev)
        out = []
        for i in range(0, n, batch):
            u = ks.G(w[i:i + batch], cfg["settle"]) if cfg["settle"] > 0 else w[i:i + batch]
            out.append(ks.traj(u, dt, p).float().cpu())
        return torch.cat(out, 1)                                       # (p+1, n, N, N) float32 on cpu

    if train or stats is None:
        uTr = prep(cfg["n_train"] if train else 512, cfg["seed_train"], 0.0, 0.5, cfg["p_train"])
        ref = uTr[:, :512].double()
        nref = ref.shape[1]
        f = torch.cat([ks.f_tendency(ref[q, i:i + 64].to(dev)).cpu() for q in range(0, ref.shape[0], 4) for i in range(0, nref, 64)])
        stats = dict(mu=ref.mean(), sd=ref.std(), sf=f.std())
        if train:
            d["uT_train"] = uTr
    d["stats"] = {k: v.float() for k, v in stats.items()}
    pmax = int(round((horizon or cfg["horizon"]) / dt))
    d["uT_test"] = prep(cfg["n_test"], cfg["seed_test"], 0.5, 1.0, pmax)
    return d


def _tendency(self, omega):
    """u_t for the normalization constant (same RHS terms as one substep, in physical space)."""
    om = torch.fft.rfft2(omega.double())
    u_hat, v_hat = self.uv_hat(om)
    c = self.convection(om, u_hat, v_hat)
    rhs = -c - self.nu * self.Ksq * om - self.Fk_hat + self.beta * v_hat - self.alpha * om
    return torch.fft.irfft2(rhs, s=(self.N, self.N))


Turb2D.f_tendency = _tendency


# ---------------------------------------------------------------- FNO with a flat parameter vector
class FNO:
    """2D FNO: lift (1 + 4 coordinate channels -> W), L layers [spectral conv (two corner blocks) + 1x1 conv], SiLU,
    project W -> 128 -> 1. Parameters live in one flat real vector theta (complex weights as real/imag pairs)."""

    def __init__(self, N, modes=16, width=24, layers=4, dev="cpu"):
        self.N, self.m, self.W, self.L = N, modes, width, layers
        self.shapes = [("lift_w", (width, 5)), ("lift_b", (width,))]
        for l in range(layers):
            self.shapes += [(f"sp{l}_a", (2, width, width, modes, modes)), (f"sp{l}_b", (2, width, width, modes, modes)),
                            (f"pw{l}_w", (width, width)), (f"pw{l}_b", (width,))]
        self.shapes += [("p1_w", (128, width)), ("p1_b", (128,)), ("p2_w", (1, 128)), ("p2_b", (1,))]
        self.sizes = [int(np.prod(s)) for _, s in self.shapes]
        self.P = sum(self.sizes)
        x = torch.arange(N, device=dev) * 2 * math.pi / N
        X, Y = x[:, None].expand(N, N), x[None, :].expand(N, N)
        self.coords = torch.stack([torch.cos(X), torch.sin(X), torch.cos(Y), torch.sin(Y)]).float()   # (4, N, N)

    def unpack(self, th):
        out, k = {}, 0
        for (name, s), n in zip(self.shapes, self.sizes):
            out[name] = th[k:k + n].view(*s)
            k += n
        return out

    def init(self, seed, dev):
        g = torch.Generator().manual_seed(seed)
        parts = []
        for (name, s), n in zip(self.shapes, self.sizes):
            if name.startswith("sp"):
                parts.append((torch.rand(n, generator=g) * 2 - 1) * (1.0 / (self.W * self.W)))
            elif name.endswith("_b"):
                parts.append(torch.zeros(n))
            elif name == "p2_w":
                parts.append(torch.zeros(n))                                   # zero output layer (as in ks/)
            else:
                fan_in = s[1]
                parts.append((torch.rand(n, generator=g) * 2 - 1) / math.sqrt(fan_in))
        return torch.cat(parts).to(dev)

    def spectral(self, z, a, b):
        """z (B, W, N, N) real -> spectral conv with corner weights a (low kx) and b (high kx), modes m."""
        m = self.m
        zf = torch.fft.rfft2(z, norm="ortho")
        wa = torch.complex(a[0], a[1])
        wb = torch.complex(b[0], b[1])
        out = torch.zeros_like(zf)
        out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", zf[:, :, :m, :m], wa)
        out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", zf[:, :, -m:, :m], wb)
        return torch.fft.irfft2(out, s=(self.N, self.N), norm="ortho")

    def net(self, th, z):
        """z (B, N, N) normalized vorticity -> (B, N, N)."""
        p = self.unpack(th)
        B = z.shape[0]
        x = torch.cat([z[:, None], self.coords[None].expand(B, -1, -1, -1).to(z.dtype)], 1)   # (B, 5, N, N)
        h = torch.einsum("bcxy,wc->bwxy", x, p["lift_w"]) + p["lift_b"][None, :, None, None]
        for l in range(self.L):
            s = self.spectral(h, p[f"sp{l}_a"], p[f"sp{l}_b"])
            w = torch.einsum("bixy,oi->boxy", h, p[f"pw{l}_w"]) + p[f"pw{l}_b"][None, :, None, None]
            h = Fn.silu(s + w) if l < self.L - 1 else s + w
        h = Fn.silu(torch.einsum("bwxy,hw->bhxy", h, p["p1_w"]) + p["p1_b"][None, :, None, None])
        return (torch.einsum("bhxy,oh->boxy", h, p["p2_w"]) + p["p2_b"][None, :, None, None])[:, 0]


class Model:
    """Euler-class stepper Gt(w) = w + dt sf N((w - mu)/sd; theta)."""

    def __init__(self, arch, dt, stats):
        self.arch, self.dt, self.c = arch, dt, {k: v.clone() for k, v in stats.items()}

    def Gt(self, th, u):
        c = self.c
        return u + self.dt * c["sf"] * self.arch.net(th, (u - c["mu"]) / c["sd"])

    def jvp(self, th, u, v):
        """Directional derivative of Gt at u along v (same shapes), forward mode."""
        return torch.func.jvp(lambda x: self.Gt(th, x), (u,), (v,))[1]


# ---------------------------------------------------------------- losses and evaluation
def sq(a):
    """Mean square per grid point, averaged over starts (so errors are in vorticity units per point)."""
    return (a * a).flatten(-2).mean(-1).mean(-1)


def rollout_kappa(model, th, uT, p, kap):
    out, u = [], uT[0]
    for q in range(1, p + 1):
        if kap is not None and q > 1:
            u = kap * uT[q - 1] + (1 - kap) * u
        u = model.Gt(th, u)
        out.append(u)
    return torch.stack(out)


def loss_kappa(model, th, uT, p, kap):
    seq = rollout_kappa(model, th, uT, p, kap)
    return 0.5 * sq(uT[1:p + 1] - seq).mean()


@torch.no_grad()
def eval_free(model, th, uT, p):
    u, errs = uT[0], []
    for q in range(1, p + 1):
        u = model.Gt(th, u)
        errs.append(torch.sqrt(((uT[q] - u) ** 2).flatten(-2).mean(-1)))
    return torch.stack(errs)
