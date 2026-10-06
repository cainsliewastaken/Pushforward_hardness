"""Shared numerics for the one-step results of the paper (Section "One training step").

Conventions (paper, Sections "Setting" and "Lower bound on the injected error"):
  * a sampled function is a vector of length N = n * n_out, point-major (index i*n_out + j);
  * <a, b> = a.b / n, ||a|| = <a, a>^(1/2);
  * K = (1/n) J J^T with J the network Jacobian at the training inputs (Pi = I).

K is never formed. Everything goes through the thin SVD J/sqrt(n) = U S V^T:
eigenvalues lam_i = S_i^2, eigenvectors v_i = sqrt(n) U_i (orthonormal in <.,.>), and a sampled
function u has eigen-coefficients u_hat = U^T u / sqrt(n) plus a null-space part of norm
u0 = ||u - sum_i u_hat_i v_i||. Every quantity below depends on u only through
(lam, u_hat, u0), and the Krylov spaces of u live in span{v_i} + span{null part of u}, so all
Krylov computations are done exactly in that (r+1)-dimensional "reduced" space in which K is
diag(lam_1, ..., lam_r, 0).

Numerical rank: eigenvalues below RANK_RTOL * lam_1 are treated as zero (part of the null space).
"""
import numpy as np
from scipy.optimize import brentq

RANK_RTOL = 1e-12


# ---------------------------------------------------------------- kernel via thin SVD
class Kernel:
    def __init__(self, J, n, rtol=RANK_RTOL):
        J = np.asarray(J, dtype=np.float64)
        self.n, self.N, self.d = n, J.shape[0], J.shape[1]
        U, S, Vt = np.linalg.svd(J / np.sqrt(n), full_matrices=False)
        lam = S ** 2
        keep = lam > rtol * lam[0] if lam.size and lam[0] > 0 else np.zeros_like(lam, bool)
        self.U, self.S, self.Vt, self.lam = U[:, keep], S[keep], Vt[keep], lam[keep]
        self.lam_dropped_max = lam[~keep].max() if (~keep).any() else 0.0
        self.rank = int(keep.sum())

    @property
    def lam1(self):
        return self.lam[0] if self.rank else 0.0

    def reduce(self, u):
        """(u_hat, u0): eigen-coefficients and norm of the null-space part (paper norm)."""
        ut = np.asarray(u, np.float64) / np.sqrt(self.n)
        c = self.U.T @ ut
        return c, float(np.linalg.norm(ut - self.U @ c))

    def reduced(self, u):
        """Reduced vector and reduced diagonal of K: (lam_ext, u_ext)."""
        c, u0 = self.reduce(u)
        return np.append(self.lam, 0.0), np.append(c, u0)

    def knorm(self, phi):
        """||phi||_K = <phi, K phi>^(1/2) = |J^T phi| / n."""
        return float(np.linalg.norm(self.S * (self.U.T @ (np.asarray(phi) / np.sqrt(self.n)))))


def ip(a, b, n):
    return float(np.dot(a, b) / n)


def nrm(a, n):
    return float(np.linalg.norm(a) / np.sqrt(n))


# ---------------------------------------------------------------- F(s): kernel-ridge closed form
def upsilon(gam, lam, e):
    return float(np.sum(lam * e ** 2 / (lam + gam) ** 2))


def solve_gamma(lam, e, cs):
    """gamma > 0 with Upsilon(gamma) = cs^2, or None if cs^2 >= Upsilon(0+) (constraint inactive)."""
    pos = lam > 0
    lam, e = lam[pos], e[pos]
    if lam.size == 0 or not np.any(e != 0):
        return None
    ups0 = float(np.sum(e ** 2 / lam))
    if cs ** 2 >= ups0:
        return None
    if cs == 0:
        return np.inf
    target = 2 * np.log(cs)
    f = lambda t: np.log(upsilon(np.exp(t), lam, e)) - target
    lo = np.log(lam.min()) - 5
    for _ in range(40):
        if f(lo) > 0:
            break
        lo -= 5
    hi = np.log(max(lam.max(), np.sqrt(np.sum(lam * e ** 2)) / cs)) + 1
    while f(hi) > 0:
        hi += 2
    return float(np.exp(brentq(f, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=500)))


def F_closed(lam_ext, e_ext, cs):
    """Theorem 'Supremum over roughness functions' (d). lam_ext/e_ext are reduced (null coordinate
    has lam = 0). cs = |c_p| s. Returns (F, gamma, Estar_reduced)."""
    gam = solve_gamma(lam_ext, e_ext, cs)
    if gam is None:  # constraint inactive: only the null-space part is left
        Es = np.where(lam_ext > 0, 0.0, e_ext)
        return float(np.linalg.norm(Es)), None, Es
    if np.isinf(gam):
        return float(np.linalg.norm(e_ext)), gam, e_ext.copy()
    Es = gam * e_ext / (lam_ext + gam)
    return float(np.linalg.norm(Es)), gam, Es


def projected_spectrum(Qm, lam_ext, sig_rtol):
    """SVD of C = Qm^T [diag(sqrt lam); 0] (the projected features): returns (W, mu) with W the left
    singular vectors whose singular values exceed sig_rtol * sigma_1 and mu their squares.
    Using the SVD of C (not eigh of C C^T) keeps relative accuracy on small eigenvalues."""
    r = len(lam_ext) - 1
    Cm = Qm[:r, :].T * np.sqrt(lam_ext[:r])[None, :]
    if Cm.size == 0:
        return np.zeros((Qm.shape[1], 0)), np.zeros(0)
    W, sv, _ = np.linalg.svd(Cm, full_matrices=False)
    keep = sv > sig_rtol * max(sv.max(), 1e-300)
    return W[:, keep], sv[keep] ** 2


def F_projected(Qm, lam_ext, e_ext, cs, sig_rtol=1e-15):
    """min_{|h|<=s} |Qm^T (e + c B h)|, cs = |c| s: the kernel-ridge closed form on the projected problem."""
    a = Qm.T @ e_ext
    W, mu = projected_spectrum(Qm, lam_ext, sig_rtol)
    b = W.T @ a
    rest = np.linalg.norm(a - W @ b)  # part of a outside the range of the projected features
    return F_closed(np.append(mu, 0.0), np.append(b, rest), cs)[0]


# ---------------------------------------------------------------- Krylov (Lanczos on the reduced diagonal)
def lanczos(dvec, v, M, tol=1e-14):
    """Orthonormal basis (columns) of span{v, Dv, ..., D^{M-1} v}, D = diag(dvec), full reorth.
    Stops early at (numerical) breakdown, where the Krylov space has become invariant."""
    nv = np.linalg.norm(v)
    if nv == 0:
        return np.zeros((len(v), 0))
    Q = np.zeros((len(v), M))
    Q[:, 0] = v / nv
    scale = max(np.abs(dvec).max(), 1e-300)
    k = 1
    for k in range(1, M):
        w = dvec * Q[:, k - 1]
        for _ in range(2):
            w -= Q[:, :k] @ (Q[:, :k].T @ w)
        b = np.linalg.norm(w)
        if b <= tol * scale:
            return Q[:, :k]
        Q[:, k] = w / b
    else:
        k = M
    return Q[:, :k]


def krylov_F(lam_ext, e_ext, cs, Mmax):
    """F_M(s) for M = 1..Mmax (Eq. FMstep), via duality on V_M:
    F_M = min_{|h|<=s} |Q_M^T (e + c B h)|, B = [diag(sqrt(lam)); 0]."""
    Q = lanczos(lam_ext, e_ext, Mmax)
    out = np.empty(Mmax)
    for M in range(1, Mmax + 1):
        out[M - 1] = F_projected(Q[:, :min(M, Q.shape[1])], lam_ext, e_ext, cs)
    return out, Q


def dist_to_span(Q, phi):
    """d_M = min_{psi in span Q[:, :M]} |psi - phi| for M = 1..ncols."""
    # residual norms by modified Gram-Schmidt (sqrt(|phi|^2 - sum c^2) would floor at ~1e-8)
    r = np.array(phi, dtype=np.float64)
    out = np.empty(Q.shape[1])
    for M in range(Q.shape[1]):
        r = r - Q[:, M] * (Q[:, M] @ r)
        out[M] = np.linalg.norm(r)
    return out


def kinv_norm(lam_ext, u_ext, in_range_rtol=1e-10):
    """||u||_{K+} (inf if u has a null-space part above in_range_rtol * |u|)."""
    nu = np.linalg.norm(u_ext)
    if nu == 0:
        return 0.0
    if u_ext[-1] > in_range_rtol * nu:
        return np.inf
    pos = lam_ext > 0
    return float(np.sqrt(np.sum(u_ext[pos] ** 2 / lam_ext[pos])))


def krylov_kinv(lam_ext, u_ext, Mmax, sig_rtol=1e-10, null_rtol=1e-12):
    """Krylov values ||u||_{K+,M} (Eq. kroughM) for M = 1..Mmax, and r_M = ||phi_M|| / ||phi_M||_K
    for the maximizer phi_M (nan where the value is infinite). On V_M with orthonormal basis Q:
    sup <phi,u>/||phi||_K = sqrt(a^T G^+ a) (a = Q^T u, G = Q^T K Q) if a is in the range of G, else inf.
    Singular values of Q^T sqrt(K) below sig_rtol * sigma_1 (eigenvalues below 1e-20 lam_1) count as zero."""
    Q = lanczos(lam_ext, u_ext, Mmax)
    vals, rM = np.empty(Mmax), np.empty(Mmax)
    for M in range(1, Mmax + 1):
        Qm = Q[:, :min(M, Q.shape[1])]
        a = Qm.T @ u_ext
        W, mu = projected_spectrum(Qm, lam_ext, sig_rtol)
        b = W.T @ a
        if np.linalg.norm(a - W @ b) > null_rtol * np.linalg.norm(a):
            vals[M - 1], rM[M - 1] = np.inf, np.nan
            continue
        c = W @ (b / mu)                     # maximizer phi = Q c (up to scale)
        kn = np.sqrt(np.sum(b ** 2 / mu))    # = sqrt(a^T G^+ a) = <phi,u>/||phi||_K = ||phi||_K
        vals[M - 1] = kn
        rM[M - 1] = np.linalg.norm(c) / kn
    return vals, rM


def distinct_count(lam_ext, u_ext, comp_rtol=1e-12, eig_rtol=1e-9):
    """q_u: number of distinct eigenvalues (including 0) on which u has a nonzero component."""
    nu = np.linalg.norm(u_ext)
    ls = np.sort(lam_ext[np.abs(u_ext) > comp_rtol * nu])
    if ls.size == 0:
        return 0
    q, last = 1, ls[0]
    for l in ls[1:]:
        if (last == 0 and l > 0) or l > last * (1 + eig_rtol):
            q += 1
            last = l
    return q


# ---------------------------------------------------------------- seminorms
class ProjectorSeminorm:
    """R(u) = ||P u|| for an orthogonal projector P = I - B B^T on sampled functions
    (B has Euclidean-orthonormal columns). C_R = 1, sigma_R^2 = lambda_max(P K P)."""
    name = "projector"

    def __init__(self, B, n, name=None):
        self.B, self.n = B, n
        if name:
            self.name = name
        self.C = 1.0

    def P(self, u):
        return u - self.B @ (self.B.T @ u)

    def __call__(self, u):
        return nrm(self.P(u), self.n)

    def sigma(self, J):
        A = J / np.sqrt(self.n)
        return float(np.linalg.norm(A - self.B @ (self.B.T @ A), 2))


def graph_highpass_basis(x, k=8, m=24, n_out=1):
    """Manifold high-pass (Definition 'Manifold high-pass'): kNN graph with Gaussian weights,
    r_g = median distance to the k-th neighbour, symmetrized by max; returns the Euclidean-
    orthonormal basis of the m smoothest eigenvectors, lifted to all output components."""
    x = np.asarray(x, np.float64).reshape(len(x), -1)
    n = x.shape[0]
    D = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    rg = np.median(D[np.arange(n), idx[:, -1]])
    W = np.zeros((n, n))
    rows = np.repeat(np.arange(n), k)
    W[rows, idx.ravel()] = np.exp(-(D[rows, idx.ravel()] / rg) ** 2)
    W = np.maximum(W, W.T)
    L = np.diag(W.sum(1)) - W
    mu, V = np.linalg.eigh(L)
    Vm = V[:, :m]
    B = np.kron(Vm, np.eye(n_out))  # point-major layout
    return B, mu


class LipschitzSeminorm:
    """Lip_n(u) = max_{i != j} |u_i - u_j| / |x_i - x_j| (Eq. lipn).
    C_R = sqrt(2n)/d_min (attained by u_i = -u_j on the closest pair; Prop. lipconst gives >= sqrt(n)/d_min).
    sigma_R = max_{i<j} sigma_max(Psi_i - Psi_j) / d_ij."""
    name = "lipschitz"

    def __init__(self, x, n_out):
        x = np.asarray(x, np.float64).reshape(len(x), -1)
        self.n, self.n_out = x.shape[0], n_out
        self.iu = np.triu_indices(self.n, 1)
        self.dij = np.linalg.norm(x[self.iu[0]] - x[self.iu[1]], axis=1)
        self.C = np.sqrt(2 * self.n) / self.dij.min()

    def __call__(self, u):
        U = np.asarray(u).reshape(self.n, self.n_out)
        diff = np.linalg.norm(U[self.iu[0]] - U[self.iu[1]], axis=1)
        return float(np.max(diff / self.dij))

    def sigma(self, J):
        Jb = J.reshape(self.n, self.n_out, -1)
        best = 0.0
        for a, b, d in zip(self.iu[0], self.iu[1], self.dij):
            best = max(best, np.linalg.norm(Jb[a] - Jb[b], 2) / d)
        return float(best)


class RankOneSeminorm:
    """R(u) = |<phi, u>|; C_R = ||phi||, sigma_R = ||phi||_K."""
    name = "rank_one"

    def __init__(self, phi, n, name=None):
        self.phi, self.n = phi, n
        if name:
            self.name = name
        self.C = nrm(phi, n)

    def __call__(self, u):
        return abs(ip(self.phi, u, self.n))

    def sigma(self, J):
        return float(np.linalg.norm(J.T @ self.phi) / self.n)


def b_R(R, E, J, cs):
    """Step bound b_R(s) = (R(E) - |c_p| s sigma_R) / C_R (Eq. bR)."""
    return (R(E) - cs * R.sigma(J)) / R.C


# ---------------------------------------------------------------- one-hidden-layer tanh MLP (numpy)
class MLP1:
    """out = W2 tanh(W1 x + b1) + b2. Flat parameter order: W1 (H x din), b1, W2 (dout x H), b2."""

    def __init__(self, din, H, dout):
        self.din, self.H, self.dout = din, H, dout
        self.sizes = [H * din, H, dout * H, dout]
        self.d = sum(self.sizes)

    def unpack(self, th):
        a, b, c, _ = np.cumsum(self.sizes)
        return (th[:a].reshape(self.H, self.din), th[a:b], th[b:c].reshape(self.dout, self.H), th[c:])

    def forward(self, th, x):
        W1, b1, W2, b2 = self.unpack(th)
        return np.tanh(x @ W1.T + b1) @ W2.T + b2  # (n, dout)

    def out(self, th, x):
        return self.forward(th, x).ravel()  # point-major sampled function

    def jac(self, th, x):
        W1, b1, W2, b2 = self.unpack(th)
        n = x.shape[0]
        h = np.tanh(x @ W1.T + b1)
        hp = 1 - h ** 2
        A = W2[None, :, :] * hp[:, None, :]                        # (n, dout, H)
        dW1 = (A[:, :, :, None] * x[:, None, None, :]).reshape(n, self.dout, -1)
        eye = np.eye(self.dout)
        dW2 = (eye[None, :, :, None] * h[:, None, None, :]).reshape(n, self.dout, -1)
        db2 = np.broadcast_to(eye, (n, self.dout, self.dout))
        return np.concatenate([dW1, A, dW2, db2], axis=2).reshape(n * self.dout, self.d)


# ---------------------------------------------------------------- check recorder
class Checks:
    """One row per checked inequality: margin = (side that should be larger) - (other side).
    pass = margin >= -tol. 'scale' makes margins comparable (rel_margin = margin / scale)."""

    def __init__(self, part):
        self.part, self.rows = part, []

    def add(self, check, case, margin, tol=0.0, scale=1.0, **info):
        margin = float(margin)
        row = dict(part=self.part, check=check, case=case, margin=margin,
                   rel_margin=margin / scale if scale else np.nan, tol=tol,
                   passed=bool(margin >= -tol))
        row.update(info)
        self.rows.append(row)
        return row["passed"]

    def add_eq(self, check, case, a, b, rtol, scale=1.0, **info):
        """Equality check |a - b| <= rtol * scale; records rel_diff = |a - b| / scale."""
        a, b = float(a), float(b)
        if np.isinf(a) and np.isinf(b):
            diff = 0.0
        else:
            diff = abs(a - b)
        return self.add(check, case, rtol * scale - diff, scale=scale, kind="eq",
                        rel_diff=diff / scale if scale else np.nan, lhs=a, rhs=b, **info)

    def write(self, path):
        import csv
        keys = []
        for r in self.rows:
            keys += [k for k in r if k not in keys]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)

    def summary(self):
        """check -> [passed, total, min rel_margin (inequalities) or max rel_diff (equalities), kind]"""
        out = {}
        for r in self.rows:
            eq = r.get("kind") == "eq"
            o = out.setdefault(r["check"], [0, 0, -np.inf if eq else np.inf, "eq" if eq else "ineq"])
            o[0] += r["passed"]
            o[1] += 1
            o[2] = max(o[2], r["rel_diff"]) if eq else min(o[2], r["rel_margin"])
        return out

    def print_summary(self):
        for k, (p, t, m, kind) in self.summary().items():
            lab = "max rel diff" if kind == "eq" else "min rel margin"
            print(f"{'PASS' if p == t else 'FAIL'} {p:4d}/{t:<4d} {lab} {m: .3e}  {k}", flush=True)


def write_rows(path, rows):
    import csv
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- misc
def gauss_legendre_01(k):
    t, w = np.polynomial.legendre.leggauss(k)
    return 0.5 * (t + 1), 0.5 * w


def s_needed(lam_ext, e_ext, c, eps):
    """s(eps) = min{s >= 0 : F(s) <= eps} (Corollary 'Step length needed'); inf if unreachable."""
    if e_ext[-1] > eps:
        return np.inf
    if np.linalg.norm(e_ext) <= eps:
        return 0.0
    lo, hi = 0.0, 1.0
    pos = lam_ext > 0
    s_full = np.sqrt(np.sum(e_ext[pos] ** 2 / lam_ext[pos])) / abs(c)
    hi = s_full
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if F_closed(lam_ext, e_ext, abs(c) * mid)[0] <= eps:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-13 * hi:
            break
    return hi
