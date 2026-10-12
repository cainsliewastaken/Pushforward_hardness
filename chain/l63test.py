"""
Test of the 'held rough error' section on a Lorenz-63 learned time stepper.
Rank-one roughness functions along Krylov functionals phi_gamma (Theorem 1(c)/2),
for which C_R = 1, sigma_R = ||phi||_K and Assumption 6 holds with c = 1.

numpy only. Conventions: RMS inner product <u,v> = (1/n) sum_i u_i . v_i on stacked
vectors of length n*ny; K = Psi Psi^T / n; gradient flow theta_dot = -(1/n) Psi^T E.
"""
import numpy as np, os, sys, time, json

rng = np.random.default_rng(0)

# ---------------- Lorenz-63 data ----------------
def l63(x, s=10., r=28., b=8/3):
    return np.array([s*(x[1]-x[0]), x[0]*(r-x[2])-x[1], x[0]*x[1]-b*x[2]])

def rk4(x, dt, f=l63):
    k1=f(x); k2=f(x+dt/2*k1); k3=f(x+dt/2*k2); k4=f(x+dt*k3)
    return x + dt/6*(k1+2*k2+2*k3+k4)

def flow(x, Delta, dt=0.005):
    m = int(round(Delta/dt))
    for _ in range(m): x = rk4(x, dt)
    return x

def make_data(n, Delta, seed=1):
    r = np.random.default_rng(seed)
    x = np.array([1., 1., 1.])
    for _ in range(4000): x = rk4(x, 0.01)          # burn-in onto attractor
    X = []
    for _ in range(n):
        for _ in range(r.integers(20, 60)): x = rk4(x, 0.01)   # decorrelate samples
        X.append(x.copy())
    X = np.array(X)
    Y = np.array([flow(xi, Delta) for xi in X])
    mu, sd = X.mean(0), X.std(0)
    Xn, Yn = (X-mu)/sd, (Y-mu)/sd
    if os.environ.get('L63_TARGET', 'direct') == 'euler':     # Euler target: temporal difference divided by the step, same normalisation
        Yn = (Yn - Xn)/Delta
    return Xn, Yn

# ---------------- MLP (tanh) ----------------
class MLP:
    def __init__(self, din, H, L, dout, seed=0):
        r = np.random.default_rng(seed)
        self.sizes = [din]+[H]*L+[dout]
        self.W = []; self.b = []
        for a, c in zip(self.sizes[:-1], self.sizes[1:]):
            self.W.append(r.normal(0, 1/np.sqrt(a), (c, a)))
            self.b.append(np.zeros(c))
        self.L = L
    def pack(self):
        return np.concatenate([p.ravel() for p in self.W+self.b])
    def unpack(self, v):
        i = 0
        for W in self.W:
            W[...] = v[i:i+W.size].reshape(W.shape); i += W.size
        for b in self.b:
            b[...] = v[i:i+b.size]; i += b.size
    def forward(self, X):
        A = [X]; Z = []
        a = X
        for l in range(self.L):
            z = a@self.W[l].T + self.b[l]; Z.append(z); a = np.tanh(z); A.append(a)
        out = a@self.W[-1].T + self.b[-1]
        return out, A, Z
    def jacobian(self, X):
        """Per-sample Jacobian: rows (i,c) for sample i, output c; shape (n*dout, d)."""
        out, A, Z = self.forward(X)
        n, dout = out.shape
        d = self.pack().size
        J = np.zeros((n, dout, d))
        for c in range(dout):
            delta = np.zeros((n, dout)); delta[:, c] = 1.
            grads_W = [None]*len(self.W); grads_b = [None]*len(self.b)
            # output layer
            grads_W[-1] = delta[:, :, None]*A[-1][:, None, :]
            grads_b[-1] = delta
            dl = delta@self.W[-1]            # n x H
            for l in range(self.L-1, -1, -1):
                dl = dl*(1-A[l+1]**2)
                grads_W[l] = dl[:, :, None]*A[l][:, None, :]
                grads_b[l] = dl
                if l > 0: dl = dl@self.W[l]
            J[:, c, :] = np.concatenate([g.reshape(n, -1) for g in grads_W+grads_b], axis=1)
        return J.reshape(n*dout, d), out

def rms_ip(u, v, n): return (u@v)/n
def rms_norm(u, n): return np.sqrt(max(u@u, 0)/n)

def kernel(J, n): return J@J.T/n           # (n*ny) x (n*ny)

def krylov_phi(K, E, gamma, n, M=60, tol=1e-10):
    """phi_gamma = E_gamma/||E_gamma||, E_gamma = gamma (K+gamma I)^{-1} E, by CG (Krylov).
    Returns phi (RMS-unit), E_gamma, and the CG iteration count."""
    A = K + gamma*np.eye(K.shape[0])
    x = np.zeros_like(E); r = E.copy(); p = r.copy(); rr = r@r
    for k in range(M):
        Ap = A@p; alpha = rr/(p@Ap); x += alpha*p; r -= alpha*Ap
        rr_new = r@r
        if np.sqrt(rr_new) < tol*np.sqrt(E@E): break
        p = r + (rr_new/rr)*p; rr = rr_new
    Eg = gamma*x
    return Eg/rms_norm(Eg, n), Eg, k+1

def along_dir_second(net, X, theta, u, h=1e-4):
    """Returns Psi_u = d/dh Psi(theta + h u): matrix with rows nabla^2 N(x_i)[u, .]."""
    net.unpack(theta+h*u); Jp, _ = net.jacobian(X)
    net.unpack(theta-h*u); Jm, _ = net.jacobian(X)
    net.unpack(theta)
    return (Jp-Jm)/(2*h)

def run(depth, H, X, Y, T_total, gammas_rel, log_every, seed=0, eta_frac=0.4, tag=''):
    n, ny = Y.shape
    net = MLP(3, H, depth, ny, seed=seed)
    theta = net.pack()
    Yv = Y.reshape(-1)
    J, out = net.jacobian(X)
    E = out.reshape(-1) - Yv
    K = kernel(J, n)
    lam, V = np.linalg.eigh(K); lam = lam[::-1]; V = V[:, ::-1]
    lam1_0 = lam[0]
    # Krylov functionals at time 0
    gammas = [g*lam1_0 for g in gammas_rel]
    phis = []; info = []
    for g in gammas:
        phi, Eg, it = krylov_phi(K, E, g, n)
        # F_M(s): the one-step floor at the step length s_gamma = Upsilon^{1/2}
        Ups = sum(l*(V[:, k]@E)**2/n/(l+g)**2 for k, l in enumerate(lam) if l > 1e-14)
        phis.append(phi)
        info.append(dict(gamma=g, cg_iters=it, Egamma_norm=rms_norm(Eg, n),
                         phiE0=rms_ip(phi, E, n), sigma0=np.sqrt(rms_ip(phi, K@phi, n)),
                         Upsilon=Ups))
    print(f'[{tag}] d_theta={theta.size}, lam1(0)={lam1_0:.3e}, ||E0||={rms_norm(E,n):.4f}')
    for inf in info:
        print(f"   gamma/lam1={inf['gamma']/lam1_0:.0e}: <phi,E0>={inf['phiE0']:.4f}, "
              f"sigma0^2/lam1={inf['sigma0']**2/lam1_0:.3e}, ||E_gamma||={inf['Egamma_norm']:.4f}, cg={inf['cg_iters']}")

    # H2 along the gradient direction and along each phi (operator-norm estimates)
    def H2_along(u):
        Pu = along_dir_second(net, X, theta, u/np.linalg.norm(u))
        return np.linalg.norm(Pu, 2)/np.sqrt(n)        # ||Psi_u||_op/sqrt(n) <= H2
    gE = J.T@E/n
    H2_g0 = H2_along(gE)
    H2_rand = max(H2_along(rng.normal(size=theta.size)) for _ in range(3))
    print(f'   H2 along g_E: {H2_g0:.3e}, along random dirs (max of 3): {H2_rand:.3e}')

    eta = eta_frac/lam1_0
    t = 0.; step = 0
    logs = []
    t0 = time.time()
    while t < T_total:
        if step % log_every == 0:
            J, out = net.jacobian(X); E = out.reshape(-1) - Yv
            K = kernel(J, n)
            lam1 = np.linalg.eigvalsh(K)[-1]
            rec = dict(t=t, step=step, E=rms_norm(E, n), lam1=lam1, eta=eta)
            for j, phi in enumerate(phis):
                pE = rms_ip(phi, E, n)
                Kphi = K@phi
                Es = E - pE*phi
                rec[f'phiE{j}'] = pE
                rec[f'sig2_{j}'] = rms_ip(phi, Kphi, n)
                rec[f'cross{j}'] = rms_ip(Kphi, Es, n)        # <phi, K E_s>
                rec[f'Es{j}'] = rms_norm(Es, n)
            # realized drive and H2 along g_E every 5th log
            if (step // log_every) % 5 == 0:
                gE = J.T@E/n
                if np.linalg.norm(gE) > 0:
                    rec['H2_g'] = H2_along(gE)
                    Pu = along_dir_second(net, X, theta, gE/np.linalg.norm(gE))
                    # exact rate of sigma^2 along phi_j:  d/dt <phi,K phi> = -2 g_phi . S(phi) g_E
                    # = -(2/n) phi^T Psi_{g_E} g_phi * |g_E|  (Prop. 10(b) / 16)
                    for j, phi in enumerate(phis):
                        gphi = J.T@phi/n
                        rec[f'drive{j}'] = -2*(phi@(Pu@gphi))/n*np.linalg.norm(gE)
            logs.append(rec)
            if (step // log_every) % 20 == 0:
                print(f'   t={t:8.2f} step={step:6d} ||E||={rec["E"]:.4f} lam1={lam1:.3e} '
                      + ' '.join(f'<phi{j},E>={rec[f"phiE{j}"]:+.4f}' for j in range(len(phis)))
                      + f'  ({time.time()-t0:.0f}s)', flush=True)
            # keep eta below the descent threshold as lambda_1 moves
            if eta*lam1 > 1.0: eta = 0.5/lam1
        # GD step (gradient-flow proxy): theta <- theta - eta * (1/n) Psi^T E
        out, A, Z = net.forward(X); E = out.reshape(-1) - Yv
        delta = (out - Y)/n                           # n x ny, dL/d out
        grads_W = [None]*len(net.W); grads_b = [None]*len(net.b)
        grads_W[-1] = delta.T@A[-1]; grads_b[-1] = delta.sum(0)
        dl = delta@net.W[-1]
        for l in range(net.L-1, -1, -1):
            dl = dl*(1-A[l+1]**2)
            grads_W[l] = dl.T@A[l]; grads_b[l] = dl.sum(0)
            if l > 0: dl = dl@net.W[l]
        g = np.concatenate([q.ravel() for q in grads_W+grads_b])
        theta -= eta*g; net.unpack(theta)
        t += eta; step += 1
    return dict(logs=logs, info=info, lam1_0=lam1_0, H2_g0=H2_g0, H2_rand=H2_rand,
                d_theta=theta.size, E0=logs[0]['E'], phis=np.array(phis), gammas=gammas)

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    Delta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    T_total = float(sys.argv[3]) if len(sys.argv) > 3 else 2000.
    depths = [int(s) for s in (sys.argv[4] if len(sys.argv) > 4 else '1,3').split(',')]
    # cluster additions (optional args 5-7): hidden width H, log_every, output directory
    H = int(sys.argv[5]) if len(sys.argv) > 5 else 96
    log_every = int(sys.argv[6]) if len(sys.argv) > 6 else 200
    outdir = sys.argv[7] if len(sys.argv) > 7 else '.'
    X, Y = make_data(n, Delta)
    gammas_rel = [1e-1, 1e-2, 1e-3, 1e-4]
    results = {}
    for depth in depths:
        tag = f'H{H}_D{Delta}_depth{depth}'
        res = run(depth, H, X, Y, T_total, gammas_rel, log_every=log_every, tag=tag)
        res['H'] = H; res['Delta'] = Delta; res['n'] = n; res['log_every'] = log_every
        results[depth] = res
        np.save(f'{outdir}/res_{tag}.npy', res, allow_pickle=True)
    print('done')
