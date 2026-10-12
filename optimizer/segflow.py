"""SegFlow: a generic trust-region optimizer for least-squares training that integrates the *linearised* gradient flow
exactly inside a Krylov space of the current kernel, for as long as the Taylor radius allows, then re-expands.

Model: residual r(theta) = f(theta) - y in R^N, loss = ||r||^2/(2n), gradient flow theta' = -J^T r / n.
Inside the linearisation at theta_i the residual obeys r' = -K r with K = J J^T / n (the NTK on the data), so
    r(t) = exp(-K t) r_i,   dtheta(t) = -J^T phi(K t) t r_i / n,   phi(z) = (1-e^{-z})/z.
The residual lives in a narrow band of eigenvalues at any time (it migrates from large to small eigenvalues as the big
ones are fitted), so m Lanczos vectors started from r_i capture exp(-Kt) r_i for modest m.  Everything is matrix-free:
one Jacobian-vector product (forward mode) and one vector-Jacobian product (reverse mode) per Lanczos iteration.
The segment length t is the largest time whose displacement ||dtheta(t)|| stays inside the trust radius delta
(the Taylor radius); delta adapts by the usual actual/predicted-decrease ratio.  No learning rate.
"""
import math, time, torch

def lanczos(matvec, v0, m_max, tol=1e-10):
    """Symmetric Lanczos with full reorthogonalisation. Returns Q (N x m), T (m x m) with Q^T K Q = T."""
    N = v0.numel(); nv = torch.linalg.norm(v0); Q = torch.zeros(N, m_max, dtype=v0.dtype, device=v0.device)
    alpha = []; beta = []; q = v0/nv; Q[:, 0] = q
    for j in range(m_max):
        w = matvec(q); a = torch.dot(q, w); alpha.append(a)
        w = w - a*q - (beta[-1]*Q[:, j-1] if j > 0 else 0)
        w = w - Q[:, :j+1] @ (Q[:, :j+1].T @ w)          # full reorthogonalisation (twice for safety)
        w = w - Q[:, :j+1] @ (Q[:, :j+1].T @ w)
        b = torch.linalg.norm(w)
        if j == m_max-1 or b < tol*nv:
            m = j+1; break
        beta.append(b); q = w/b; Q[:, j+1] = q
    T = torch.diag(torch.stack(alpha[:m]))
    if m > 1:
        bb = torch.stack(beta[:m-1]); T = T + torch.diag(bb, 1) + torch.diag(bb, -1)
    return Q[:, :m], T

class SegFlow:
    def __init__(self, f, theta0, y, n=None, delta0=0.25, delta_max=1e3, m_max=60, lanczos_tol=1e-10,
                 grow=1.6, shrink=0.5, rho_good=0.75, rho_bad=0.25, t_cap=None, verbose=False, filt='flow', so_radius=False, so_correct=False, kappa=0.1, recycle_k=0, precond=None, beta2=0.9, eps=1e-8, outer_momentum=0.0, kernel_fn=None):
        """f: flat parameter vector -> output vector (N,), differentiable with torch.func.  y: targets (N,).
        n: normalisation of the loss (number of samples; default N).  delta0: initial trust radius in ||theta||."""
        self.f, self.y = f, y; self.theta = theta0.clone(); self.n = float(n if n is not None else y.numel())
        self.delta, self.delta_max, self.m_max, self.tol = float(delta0), delta_max, m_max, lanczos_tol
        self.grow, self.shrink, self.rho_good, self.rho_bad, self.t_cap, self.verbose, self.filt = grow, shrink, rho_good, rho_bad, t_cap, verbose, filt
        self.so_radius, self.so_correct, self.kappa = so_radius, so_correct, kappa   # second-order (H2) radius rule and corrector
        self.kappa_eff = kappa
        self.counts_extra = dict(d2=0); self.recycle_k = recycle_k; self.Vrec = None; self.KVrec = None; self.refresh = 2; self.refresh_ptr = 0   # recycled Ritz vectors and their (stale) K-images
        # 'adam' preconditioner: segments integrate theta' = -P J^T r/n with P = diag(1/(sqrt(v_hat)+eps)) normalised to mean 1,
        # v = EMA of squared gradients at segment starts (Adam's second moment); the kernel becomes K_P = J P J^T / n
        self.precond, self.beta2, self.eps = precond, beta2, eps; self.v = None; self.vk = 0
        self.outer_momentum = outer_momentum; self.prev_step = None
        # explicit kernel: kernel_fn(theta) -> (K as N x N tensor, flops spent) built from the network's layer structure
        # (Hadamard products of activation and backprop Gram matrices); matvecs are then dense N^2 and counted in flops
        self.kernel_fn = kernel_fn; self.flops_extra = 0.0   # momentum across segments: extrapolate past the linear radius along the last accepted step
        self.t = 0.0; self.counts = dict(forward=0, jvp=0, vjp=0, segments=0, rejected=0); self.hist = []
        self.r = self.residual(self.theta)
    # ---- model access (matrix-free) ----
    def residual(self, theta):
        self.counts['forward'] += 1; return self.f(theta) - self.y
    def jvp(self, theta, v):
        self.counts['jvp'] += 1; return torch.func.jvp(self.f, (theta,), (v,))[1]
    def vjp(self, theta, u):
        self.counts['vjp'] += 1
        _, pull = torch.func.vjp(self.f, theta); return pull(u)[0]
    def d2(self, theta, v):
        """Second directional derivative f''[v,v] (nested forward mode): the exact Taylor remainder leader, and the
        object the roughness theory's H2 bounds (H2 = ||f''[u,.]||_op/sqrt(n) along the unit gradient)."""
        self.counts_extra['d2'] += 1
        g = lambda th: torch.func.jvp(self.f, (th,), (v,))[1]
        return torch.func.jvp(g, (theta,), (v,))[1]
    def rms(self, r): return float(torch.sqrt(torch.dot(r, r)/self.n))
    # ---- one segment ----
    def step(self, t_remaining=math.inf):
        theta, r, n = self.theta, self.r, self.n
        if self.precond == 'adam':
            g = self.vjp(theta, r)/n                                          # gradient at the segment start
            self.v = g*g if self.v is None else self.beta2*self.v + (1-self.beta2)*g*g; self.vk += 1
            vhat = self.v/(1-self.beta2**self.vk); P = 1.0/(torch.sqrt(vhat) + self.eps); P = P/P.mean()
        elif self.precond == 'layer':                                         # fixed per-layer metric: P constant on each parameter block
            P = self.layer_P
        else:
            P = None
        if self.kernel_fn is not None and P is None:
            Kmat, fl = self.kernel_fn(theta); self.flops_extra += fl; N = r.numel()
            def matvec(u):
                self.flops_extra += 2*N*N; return Kmat @ u
        else:
            matvec = (lambda u: self.jvp(theta, P*self.vjp(theta, u))/n) if P is not None else (lambda u: self.jvp(theta, self.vjp(theta, u))/n)   # K u = J P J^T u / n
        if self.recycle_k and self.Vrec is not None:
            # Krylov recycling with deflation. V = k Ritz vectors kept from the last segment with K-images KV (stale by one
            # segment, O(delta) error like the accepted Taylor error; a few are refreshed each segment). New Lanczos vectors are
            # built for the DEFLATED operator (I-VV^T) K (I-VV^T) from the residual outside span(V), so they stay orthogonal
            # to V and their tridiagonal is exact; the cross block uses the stored KV. Cost: m_max (+refresh) matvecs instead of ~30.
            V, KV = self.Vrec, self.KVrec; k = V.shape[1]
            if self.refresh:
                j = [(self.refresh_ptr + i) % k for i in range(self.refresh)]; self.refresh_ptr = (self.refresh_ptr + self.refresh) % k
                for jj in j: KV[:, jj] = matvec(V[:, jj])
            proj = lambda u: u - V @ (V.T @ u)
            r_perp = proj(r)
            if float(torch.linalg.norm(r_perp)) > 1e-10*float(torch.linalg.norm(r)):
                Qn, Tn = lanczos(lambda u: proj(matvec(proj(u))), r_perp, self.m_max, self.tol)
                Q = torch.cat([V, Qn], 1)
                T = torch.zeros(Q.shape[1], Q.shape[1], dtype=r.dtype)
                T[:k, :k] = 0.5*(V.T @ KV + (V.T @ KV).T); T[:k, k:] = KV.T @ Qn; T[k:, :k] = T[:k, k:].T; T[k:, k:] = Tn
            else:
                Q = V; T = 0.5*(V.T @ KV + (V.T @ KV).T)
        else:
            Q, T = lanczos(matvec, r, self.m_max, self.tol)
        m = T.shape[0]
        try:
            th, S = torch.linalg.eigh(T)
        except Exception:                                                     # ill-conditioned recycled block: rebuild from scratch this segment
            self.Vrec = None; Q, T = lanczos(matvec, r, max(self.m_max, 30), self.tol); th, S = torch.linalg.eigh(T)
        th = torch.clamp(th, min=0.0)                                          # Ritz values (eigenvalues of K seen by r)
        c = S.T @ (Q.T @ r)                                                  # residual in the Ritz basis
        def coeffs(t):
            if self.filt == 'lm': return 1.0/(th + 1.0/t)                  # Levenberg-Marquardt filter, damping mu = 1/t
            z = th*t; phi = torch.where(z > 1e-12, (1-torch.exp(-z))/torch.where(z > 1e-12, z, 1.0), 1.0 - z/2)
            return phi*t                                                     # dtheta = -J^T Q S (coeffs*c) / n
        def disp(t):
            w = coeffs(t)*c; return float(torch.sqrt(torch.dot(w, th*w)/n))  # ||J^T Q S w||^2 = n w^T T w (exact)
        # largest t with disp(t) <= delta (disp is increasing in t), capped by remaining time and t_cap
        t_hi = min(t_remaining, self.t_cap if self.t_cap else math.inf)
        def pick_t(delta):
            if math.isfinite(t_hi) and disp(t_hi) <= delta: return t_hi
            lo, hi = 1e-12, (t_hi if math.isfinite(t_hi) else 1.0)
            if not math.isfinite(t_hi):
                while disp(hi) < delta and hi < 1e30: hi *= 4
            for _ in range(100): tm = math.sqrt(lo*hi); lo, hi = ((tm, hi) if disp(tm) < delta else (lo, tm))
            return lo
        t = pick_t(self.delta)
        w = coeffs(t)*c; u = Q @ (S @ w); dtheta = -self.vjp(theta, u)/n; dtheta = dtheta if P is None else P*dtheta
        h2 = None
        if self.so_radius or self.so_correct:
            nd = float(torch.linalg.norm(dtheta)); h2d = self.d2(theta, dtheta/nd)      # f''[d,d] along the unit step direction
            if self.so_radius:
                # a-priori radius: Taylor remainder 1/2 delta^2 ||f''[d,d]|| = kappa * (predicted residual decrease at that
                # radius); solved by bisection in the Krylov space (the H2-type rule of the theory, remainder vs gain)
                nh = max(float(torch.linalg.norm(h2d)), 1e-300); rn = float(torch.linalg.norm(r))
                def excess(delta):
                    tt = pick_t(delta); rp = float(torch.linalg.norm((1 - th*coeffs(tt))*c)); return 0.5*delta**2*nh - self.kappa_eff*(rn - rp)
                lo, hi = 1e-6*self.delta, 1e3*self.delta
                for _ in range(60): md = math.sqrt(lo*hi); lo, hi = ((md, hi) if excess(md) < 0 else (lo, md))
                delta_star = min(max(lo, 0.25*self.delta), 4*self.delta, self.delta_max)
                self.delta = delta_star; t = pick_t(self.delta)
                w = coeffs(t)*c; u = Q @ (S @ w); dtheta = -self.vjp(theta, u)/n; dtheta = dtheta if P is None else P*dtheta; nd = float(torch.linalg.norm(dtheta))
            h2 = nd**2*h2d                                                             # f''[dtheta,dtheta] (direction ~ unchanged)
            if self.so_correct:
                # corrector: cancel the leading nonlinearity inside the fitted subspace with the same filter
                cc = S.T @ (Q.T @ (0.5*h2)); corr = self.vjp(theta, Q @ (S @ (coeffs(t)*cc)))/n; dtheta = dtheta + (corr if P is None else P*corr)
        r_pred = Q @ (S @ ((1 - th*coeffs(t))*c)) + (r - Q @ (Q.T @ r))        # linear-model prediction (both filters)
        if h2 is not None and not self.so_correct: r_pred = r_pred + 0.5*h2         # quadratic prediction (rho then measures 3rd order)
        if getattr(self, 'record_mid', False):                                   # state half-way along the segment (for measurements on the path)
            wm = coeffs(0.5*t)*c; dm = self.vjp(theta, Q @ (S @ wm))/n; self.theta_mid = theta - (dm if P is None else P*dm); self.t_mid = self.t + 0.5*t
        if self.recycle_k:
            # keep the k Ritz pairs that carry the residual's energy plus the largest Ritz values (fast modes re-excited each segment)
            kk = min(self.recycle_k, th.numel())
            if getattr(self, 'recycle_mode', 'fast') == 'fast':
                idx = torch.argsort(th, descending=True)[:kk]                     # only the converged fast pairs (top Ritz values)
            else:
                half = kk//2; idx_fast = torch.argsort(th, descending=True)[:half]
                wgt = c**2; wgt[idx_fast] = -1; idx_res = torch.argsort(wgt, descending=True)[:kk-half]
                idx = torch.unique(torch.cat([idx_fast, idx_res]))
            Vn = Q @ S[:, idx]; Vq, R = torch.linalg.qr(Vn)                     # re-orthonormalise the stored basis
            self.Vrec = Vq; self.KVrec = torch.linalg.solve_triangular(R, (Vn*th[idx]).T, upper=True, left=False).T if False else (Vn*th[idx]) @ torch.linalg.inv(R)   # K v ~ theta v (Ritz) mapped through the QR
        if self.outer_momentum and self.prev_step is not None: dtheta = dtheta + self.outer_momentum*self.prev_step
        theta_new = theta + dtheta; r_new = self.residual(theta_new)
        dec_pred = float(torch.dot(r, r) - torch.dot(r_pred, r_pred)); dec_act = float(torch.dot(r, r) - torch.dot(r_new, r_new))
        rho = dec_act/dec_pred if dec_pred > 0 else (1.0 if dec_act >= 0 else -1.0)
        # band where the residual sits: energy-weighted mean log Ritz value, and Krylov representation quality
        wgt = c**2/torch.dot(c, c); cum = torch.cumsum(wgt, 0); lam_mean = float(th[min(int(torch.searchsorted(cum, torch.tensor(0.5, dtype=cum.dtype))), th.numel()-1)])  # median eigenvalue of the residual energy
        accepted = rho > 0 and dec_act > -1e-12*float(torch.dot(r, r))
        if accepted:
            self.theta, self.r, self.t = theta_new, r_new, self.t + t; self.prev_step = dtheta
        elif self.outer_momentum: self.prev_step = None
        else:
            self.counts['rejected'] += 1
        if not self.so_radius:
            if rho > self.rho_good: self.delta = min(self.delta*self.grow, self.delta_max)
            elif rho < self.rho_bad: self.delta *= self.shrink
        else:
            self.kappa_eff = 0.5*self.kappa_eff if not accepted else min(self.kappa, 1.25*self.kappa_eff)   # back off the tolerance after a rejection
            if not accepted: self.delta *= self.shrink
        self.counts['segments'] += 1
        rec = dict(t=self.t, dt=t, E=self.rms(self.r), E_pred=self.rms(r_pred), rho=rho, delta=self.delta, m=m,
                   disp=float(torch.linalg.norm(dtheta)), lam_band=lam_mean, accepted=accepted, **self.counts, **self.counts_extra)
        self.hist.append(rec)
        if self.verbose: print('  seg %4d t=%.4g dt=%.3g E=%.5f pred %.5f rho %.2f delta %.3g m %d band %.3g %s' % (
            rec['segments'], rec['t'], t, rec['E'], rec['E_pred'], rho, self.delta, m, lam_mean, '' if accepted else 'REJECT'), flush=True)
        return rec
    def run(self, T=math.inf, E_target=0.0, max_segments=100000, max_wall=math.inf):
        t0 = time.time()
        while self.t < T and self.rms(self.r) > E_target and self.counts['segments'] < max_segments and time.time()-t0 < max_wall:
            self.step(t_remaining=T - self.t)
        return self.hist
