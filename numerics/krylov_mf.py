"""Matrix-free Krylov values (Section 'The best roughness function', Theorem 'Krylov series') for
batches of one-hidden-layer tanh networks.

The kernel K = (1/n) Psi Psi^T, Psi = J Pi^(1/2), is applied as v -> Psi (Psi^T v) / n with one
vector-Jacobian and one Jacobian-vector product. K is never formed and no eigendecomposition of K
is used for the reported values. Sampled functions are stored divided by sqrt(n), so the paper's
<a, b> = a.b / n is the Euclidean product of the stored vectors, and K is a symmetric matrix.

For a sampled function u, lanczos() builds an orthonormal basis Q_M of the Krylov space
V_M = span{u, K u, ..., K^(M-1) u} (full reorthogonalization), and the projected data
G_M = Q_M^T K Q_M (formed explicitly from the stored products K q_i, not from the Lanczos
recurrence) and a_M = Q_M^T u. The Krylov value
    F_M(s) = max_{phi in V_M, ||phi|| <= 1} <phi, -u> - |c_p| s ||phi||_K
equals, by duality, min_{|g| <= s} ||Q_M^T (u + c_p Psi g)||, the kernel-ridge closed form of the
projected problem (core.F_closed on the eigenpairs of G_M). This is <= F(s) for any orthonormal
Q_M, so every value is a lower bound whatever the rounding in the recurrence; the loss of
orthogonality max |Q^T Q - I| is recorded as a check.

Exact values (eigendecomposition of the explicit kernel) are computed separately, only as checks.
"""
import numpy as np
import torch

import core as C


class TanhNets:
    """B networks out = W2 tanh(W1 x + b1) + b2. W1 (B,H,din), b1 (B,H), W2 (B,dout,H), b2 (B,dout);
    x (n,din) shared or (B,n,din). pi: None (Pi = I) or (B,d), parameter order W1, b1, W2, b2,
    row-major (as replicate_rsk.net_jac and core.MLP1)."""

    def __init__(self, W1, b1, W2, b2, x, pi=None):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
        B, H, din = W1.shape
        dout = W2.shape[1]
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(B, *x.shape)
        self.x, self.pi = x, pi
        self.B, self.H, self.din, self.dout, self.n = B, H, din, dout, x.shape[1]
        self.N = self.n * dout
        self.sizes = [H * din, H, dout * H, dout]
        self.d = sum(self.sizes)
        self.h = torch.tanh(torch.einsum("bnd,bhd->bnh", x, W1) + b1[:, None, :])
        self.hp = 1 - self.h * self.h

    def out(self):
        """(B, N) point-major network outputs (not scaled)."""
        return (torch.einsum("bnh,bkh->bnk", self.h, self.W2) + self.b2[:, None, :]).reshape(self.B, -1)

    def vjp(self, v):
        """J^T v for v (B, N) -> (B, d)."""
        v = v.reshape(self.B, self.n, self.dout)
        gW2 = torch.einsum("bnk,bnh->bkh", v, self.h)
        gb2 = v.sum(1)
        r = torch.einsum("bnk,bkh->bnh", v, self.W2) * self.hp
        gW1 = torch.einsum("bnh,bnd->bhd", r, self.x)
        gb1 = r.sum(1)
        return torch.cat([gW1.reshape(self.B, -1), gb1, gW2.reshape(self.B, -1), gb2], 1)

    def jvp(self, g):
        """J g for g (B, d) -> (B, N)."""
        a, b, c = np.cumsum(self.sizes)[:3]
        gW1 = g[:, :a].reshape(self.B, self.H, self.din)
        gb1 = g[:, a:b]
        gW2 = g[:, b:c].reshape(self.B, self.dout, self.H)
        gb2 = g[:, c:]
        dz = torch.einsum("bnd,bhd->bnh", self.x, gW1) + gb1[:, None, :]
        o = (torch.einsum("bnh,bkh->bnk", self.hp * dz, self.W2) + torch.einsum("bnh,bkh->bnk", self.h, gW2)
             + gb2[:, None, :])
        return o.reshape(self.B, -1)

    def K(self, v):
        """K v = Psi Psi^T v / n for v (B, N)."""
        g = self.vjp(v)
        if self.pi is not None:
            g = g * self.pi
        return self.jvp(g) / self.n

    def psi_scaled(self, b):
        """Explicit Psi / sqrt(n) of network b, shape (N, d), so that K = A A^T. Checks only."""
        x, h, hp, W2 = self.x[b], self.h[b], self.hp[b], self.W2[b]
        n, dout = self.n, self.dout
        A = W2[None] * hp[:, None, :]                                     # (n, dout, H)
        dW1 = (A[..., None] * x[:, None, None, :]).reshape(n, dout, -1)
        eye = torch.eye(dout, dtype=h.dtype, device=h.device)
        dW2 = (eye[None, :, :, None] * h[:, None, None, :]).reshape(n, dout, -1)
        J = torch.cat([dW1, A, dW2, eye.expand(n, dout, dout)], 2).reshape(n * dout, -1)
        if self.pi is not None:
            J = J * self.pi[b].sqrt()[None, :]
        return J / np.sqrt(n)


def lanczos(Kop, u, M, gen_seed=0, tol=1e-13):
    """Batched Lanczos with full reorthogonalization. u (B, N) (scaled sampled functions).
    Returns G (B,M,M) = Q^T K Q, a (B,M) = Q^T u, orth (B,) = max|Q Q^T - I|, nrestart (B,) = number
    of breakdowns (the Krylov space became invariant; continued with random orthogonal vectors,
    which keeps F_M a lower bound and does not change it once V_M is invariant)."""
    B, N = u.shape
    M = min(M, N)
    dt, dev = u.dtype, u.device
    gen = torch.Generator(device=dev).manual_seed(gen_seed)
    Q = torch.zeros(B, M, N, dtype=dt, device=dev)
    KQ = torch.zeros(B, M, N, dtype=dt, device=dev)
    nu = u.norm(dim=1)
    q0 = torch.where(nu[:, None] > 0, u / nu.clamp(min=1e-300)[:, None],
                     torch.randn(B, N, generator=gen, dtype=dt, device=dev))
    Q[:, 0] = q0 / q0.norm(dim=1, keepdim=True)
    nrestart = torch.zeros(B, dtype=torch.long, device=dev)
    scale = torch.zeros(B, dtype=dt, device=dev)
    for k in range(M):
        w = Kop(Q[:, k])
        KQ[:, k] = w
        scale = torch.maximum(scale, w.norm(dim=1))
        if k == M - 1:
            break
        Qk = Q[:, :k + 1]
        for _ in range(2):
            w = w - torch.einsum("bm,bmn->bn", torch.einsum("bmn,bn->bm", Qk, w), Qk)
        beta = w.norm(dim=1)
        bad = beta <= tol * scale.clamp(min=1e-300)
        if bad.any():
            r = torch.randn(B, N, generator=gen, dtype=dt, device=dev)
            for _ in range(2):
                r = r - torch.einsum("bm,bmn->bn", torch.einsum("bmn,bn->bm", Qk, r), Qk)
            w = torch.where(bad[:, None], r, w)
            nrestart += bad.long()
        Q[:, k + 1] = w / w.norm(dim=1, keepdim=True)
    G = torch.einsum("bmn,bkn->bmk", Q, KQ)
    G = 0.5 * (G + G.transpose(1, 2))
    a = torch.einsum("bmn,bn->bm", Q, u)
    eye = torch.eye(M, dtype=dt, device=dev)
    orth = (torch.einsum("bmn,bkn->bmk", Q, Q) - eye).abs().amax((1, 2))
    return G.cpu().numpy(), a.cpu().numpy(), orth.cpu().numpy(), nrestart.cpu().numpy()


def projected(G, a, M):
    """Eigenpairs of the leading M x M block: (mu_ext, b_ext) in the reduced form of core
    (eigenvalues clipped at 0; the last coordinate is the part of a_M outside the range, weight 1)."""
    mu, W = np.linalg.eigh(G[:M, :M])
    mu = np.clip(mu, 0.0, None)
    b = W.T @ a[:M]
    return np.append(mu, 0.0), np.append(b, 0.0)


def krylov_F(G, a, cs, Ms):
    """F_M(s) for M in Ms, cs = |c_p| s."""
    return [C.F_closed(*projected(G, a, min(M, len(a))), cs)[0] for M in Ms]


def ritz_upsilon(G, a, gam):
    """Gauss-quadrature (Lanczos) estimate of Upsilon_u(gamma) = <u, K (K + gamma)^-2 u> from all
    of V_M (a convergence check; the step length itself comes from cg_slow)."""
    mu, b = projected(G, a, len(a))
    return C.upsilon(gam, mu, b)


def ritz_residual(G, a, gam):
    """Lanczos estimate of ||u_gamma|| (not a bound; reported as a convergence check)."""
    mu, b = projected(G, a, len(a))
    return float(np.linalg.norm(gam * b / (mu + gam)))


def cg_slow(Kop, u, gam, tol=1e-10, maxit=5000):
    """Matrix-free slow part by batched conjugate gradients: w = (K + gamma I)^-1 u, gam (B,) tensor.
    Returns ||u_gamma|| = gamma |w| (Eq. slow part), Upsilon_u(gamma) = <w, K w>, the final relative
    residual and the iteration count, all as numpy arrays over the batch. With |c_p| s = Upsilon^(1/2)
    the smallest error that a step of length s can reach is F(s) = ||u_gamma|| (Proposition 4)."""
    B = u.shape[0]
    w = torch.zeros_like(u)
    r = u.clone()
    p = r.clone()
    rr = (r * r).sum(1)
    r0 = rr.sqrt().clamp(min=1e-300)
    active = torch.ones(B, dtype=torch.bool, device=u.device)
    its = torch.zeros(B, dtype=torch.long, device=u.device)
    for it in range(maxit):
        Ap = Kop(p) + gam[:, None] * p
        pAp = (p * Ap).sum(1)
        alpha = torch.where(active, rr / pAp.clamp(min=1e-300), torch.zeros_like(rr))
        w += alpha[:, None] * p
        r -= alpha[:, None] * Ap
        rr_new = (r * r).sum(1)
        its += active.long()
        active = active & (rr_new.sqrt() > tol * r0)
        if not active.any():
            break
        beta = torch.where(active, rr_new / rr.clamp(min=1e-300), torch.zeros_like(rr))
        p = torch.where(active[:, None], r + beta[:, None] * p, p)
        rr = torch.where(active, rr_new, rr)
    Kw = Kop(w)
    res = (u - Kw - gam[:, None] * w).norm(dim=1) / r0                    # true residual
    ups = (w * Kw).sum(1)
    slow = gam * w.norm(dim=1)
    f = lambda t: t.detach().cpu().numpy()
    return f(slow), f(ups), f(res), f(its)


def lam1_krylov(Kop, B, N, dtype, dev, M=40, seed=1):
    """Largest Ritz value from a random start (<= lambda_1, converges fast)."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    r = torch.randn(B, N, generator=gen, dtype=dtype, device=dev)
    G, _, _, _ = lanczos(Kop, r, M, gen_seed=seed + 1)
    return np.array([np.linalg.eigvalsh(g)[-1] for g in G])


class Exact:
    """Exact reduced form of one network's kernel from the explicit A = Psi / sqrt(n). Checks only."""

    def __init__(self, A):
        N, d = A.shape
        self.A = A
        if N <= d:
            lam, V = torch.linalg.eigh(A @ A.T)
            lam, V = lam.flip(0), V.flip(1)
            keep = lam > C.RANK_RTOL * lam[0]
            self.lam, self.V, self.W = lam[keep], V[:, keep], None
        else:
            lam, W = torch.linalg.eigh(A.T @ A)
            lam, W = lam.flip(0), W.flip(1)
            keep = lam > C.RANK_RTOL * lam[0]
            self.lam, self.W, self.V = lam[keep], W[:, keep], None
        self.lam_ext = np.append(self.lam.cpu().numpy(), 0.0)

    def reduced(self, u):
        """u (N,) scaled sampled function -> reduced vector (eigen-coefficients + null norm)."""
        c = self.V.T @ u if self.V is not None else (self.W.T @ (self.A.T @ u)) / self.lam.sqrt()
        null = torch.sqrt(torch.clamp(u @ u - c @ c, min=0.0))
        return np.append(c.cpu().numpy(), float(null))
