"""2D turbulence truth in torch, matching py2d (Py2D_solver.py, NoSGS, dealias=True):

  vorticity omega on [0, 2pi)^2, N x N grid, rfft2 layout (axis 0 = x = Kx full, axis 1 = y = Ky half);
  psi_hat = omega_hat / K^2 (omega = -Laplacian psi), u = d psi/dy, v = -d psi/dx;
  convection: 0.5 * (conservative + non-conservative forms), products dealiased by 3/2 padding;
  per solver step h (AB2 for convection, Crank-Nicolson for viscosity):
    RHS   = omega1 - h*(1.5 c1 - 0.5 c0) + 0.5 h nu (-K^2 omega1) - h Fk_hat + h beta v1_hat
    omega = RHS / (1 + h alpha + 0.5 h nu K^2)
  forcing Fk = fky cos(fky y) + fkx cos(fkx x) (py2d sign convention: enters the RHS with a minus sign).
Batched over a leading start axis; float64; differentiable (torch.fft).
The map G(omega, dt) over an emulator step restarts AB2 with c0 = c1 at its first substep (py2d's it == 0 rule),
so it is a function of the state alone.
"""
import math

import torch

torch.set_default_dtype(torch.float64)


class Turb2D:
    def __init__(self, N=128, Re=1000.0, fkx=4, fky=4, alpha=0.1, beta=20.0, h=1e-3, dev="cpu"):
        self.N, self.nu, self.alpha, self.beta, self.h = int(N), 1.0 / Re, float(alpha), float(beta), float(h)
        self.L = 2 * math.pi
        k = torch.fft.fftfreq(N, d=1.0 / N, device=dev) * (2 * math.pi / self.L)       # 0..N/2-1, -N/2..-1
        kr = k[: N // 2 + 1].clone()
        kr[-1] = -kr[-1] if False else kr[-1]                                           # rfft half axis: 0..N/2
        self.Kx = k[:, None].expand(N, N // 2 + 1).clone()
        self.Ky = kr[None, :].expand(N, N // 2 + 1).clone()
        self.Ksq = self.Kx ** 2 + self.Ky ** 2
        self.invKsq = torch.where(self.Ksq > 0, 1.0 / self.Ksq.clamp(min=1e-300), torch.zeros_like(self.Ksq))
        x = torch.arange(N, device=dev) * self.L / N
        X, Y = x[:, None].expand(N, N), x[None, :].expand(N, N)
        Fk = fky * torch.cos(fky * Y) + fkx * torch.cos(fkx * X)
        self.Fk_hat = torch.fft.rfft2(Fk)
        self.denom = 1 + self.h * self.alpha + 0.5 * self.h * self.nu * self.Ksq
        # dealiasing (3/2 padding) index sets
        self.Np = 3 * N // 2
        self.kn = N // 2
        self.dev = dev

    # ---- dealiased product of two spectral fields (py2d multiply_dealias_spectral)
    def _pad(self, a_hat):
        N, Np, kn = self.N, self.Np, self.kn
        out = torch.zeros(*a_hat.shape[:-2], Np, Np // 2 + 1, dtype=a_hat.dtype, device=a_hat.device)
        s = (Np / N) ** 2
        out[..., : kn + 1, : kn + 1] = s * a_hat[..., : kn + 1, : kn + 1]
        out[..., Np - kn + 1 :, : kn + 1] = s * a_hat[..., N - kn + 1 :, : kn + 1]
        out[..., kn, :] = 0
        out[..., :, kn] = 0
        return out

    def _trunc(self, ab_hat_fine):
        N, Np, kn = self.N, self.Np, self.kn
        out = torch.zeros(*ab_hat_fine.shape[:-2], N, N // 2 + 1, dtype=ab_hat_fine.dtype, device=ab_hat_fine.device)
        s = (N / Np) ** 2
        out[..., : kn + 1, : kn + 1] = s * ab_hat_fine[..., : kn + 1, : kn + 1]
        out[..., N - kn + 1 :, : kn + 1] = s * ab_hat_fine[..., Np - kn + 1 :, : kn + 1]
        return out

    def mul_dealias(self, a_hat, b_hat):
        Np = self.Np
        a = torch.fft.irfft2(self._pad(a_hat), s=(Np, Np))
        b = torch.fft.irfft2(self._pad(b_hat), s=(Np, Np))
        return self._trunc(torch.fft.rfft2(a * b))

    def convection(self, om_hat, u_hat, v_hat):
        Kx, Ky = self.Kx, self.Ky
        c_cons = 1j * Kx * self.mul_dealias(u_hat, om_hat) + 1j * Ky * self.mul_dealias(v_hat, om_hat)
        c_non = self.mul_dealias(u_hat, 1j * Kx * om_hat) + self.mul_dealias(v_hat, 1j * Ky * om_hat)
        return 0.5 * (c_cons + c_non)

    def uv_hat(self, om_hat):
        psi = om_hat * self.invKsq
        return 1j * self.Ky * psi, -1j * self.Kx * psi

    def substep(self, om_hat, c0):
        u_hat, v_hat = self.uv_hat(om_hat)
        c1 = self.convection(om_hat, u_hat, v_hat)
        c = 1.5 * c1 - 0.5 * c0 if c0 is not None else c1
        rhs = om_hat - self.h * c - 0.5 * self.h * self.nu * self.Ksq * om_hat - self.h * self.Fk_hat + self.h * self.beta * v_hat
        return rhs / self.denom, c1

    def G(self, omega, dt):
        """Truth map over dt (a whole number of solver steps) applied to real fields (..., N, N)."""
        sub = int(round(dt / self.h))
        assert abs(sub * self.h - dt) < 1e-12, (dt, self.h)
        om = torch.fft.rfft2(omega)
        c0 = None
        for _ in range(sub):
            om, c0 = self.substep(om, c0)
        return torch.fft.irfft2(om, s=(self.N, self.N))

    def traj(self, om0, dt, p):
        out = [om0]
        for _ in range(p):
            out.append(self.G(out[-1], dt))
        return torch.stack(out)

    def jvp_fd(self, omega, dt, v, eps=1e-6):
        """Directional derivative of G at omega along v (same shape), central differences."""
        s = eps / (v.flatten(-2).norm(dim=-1)[..., None, None] + 1e-300)
        return (self.G(omega + s * v, dt) - self.G(omega - s * v, dt)) / (2 * s)

    def energy_enstrophy(self, omega):
        om = torch.fft.rfft2(omega)
        psi = om * self.invKsq
        w = torch.ones_like(self.Ksq)
        w[:, 1:-1] = 2.0                                                            # rfft half-axis weights
        E = 0.5 * (w * (self.Ksq * psi.abs() ** 2)).sum((-2, -1)) / self.N ** 4
        Z = 0.5 * (w * om.abs() ** 2).sum((-2, -1)) / self.N ** 4
        return E, Z
