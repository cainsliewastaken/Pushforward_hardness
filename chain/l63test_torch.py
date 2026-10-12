"""
Torch/GPU port of l63test.py (same quantities, same logging keys, float64).
Data generation and the network initialization are taken verbatim from l63test.py
(numpy, same seeds), so a run is directly comparable with the numpy version.

usage: python l63test_torch.py N DELTA T_TOTAL DEPTHS [H] [LOG_EVERY] [OUTDIR]
       device: cuda if available (set CUDA_VISIBLE_DEVICES to pick a GPU), else cpu
"""
import numpy as np, sys, time, os
import torch
from l63test import make_data, MLP as NpMLP          # same data and init as the numpy script

torch.set_default_dtype(torch.float64)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTSCALE = float(os.environ.get('L63_OUTSCALE', '1'))   # N_s = OUTSCALE * N (lazy limit for large OUTSCALE)
rng = np.random.default_rng(0)


class MLP:
    """tanh MLP; parameters held in lists of torch tensors (no autograd, manual backprop)."""
    def __init__(self, din, H, L, dout, seed=0):
        ref = NpMLP(din, H, L, dout, seed=seed)       # numpy init -> identical parameters
        self.W = [torch.tensor(W, device=DEV) for W in ref.W]
        self.b = [torch.tensor(b, device=DEV) for b in ref.b]
        self.L = L
        self.out0 = None
    def pack(self):
        return torch.cat([p.reshape(-1) for p in self.W + self.b])
    def unpack(self, v):
        i = 0
        for W in self.W:
            W.copy_(v[i:i+W.numel()].reshape(W.shape)); i += W.numel()
        for b in self.b:
            b.copy_(v[i:i+b.numel()]); i += b.numel()
    def forward(self, X):
        A = [X]; a = X
        for l in range(self.L):
            a = torch.tanh(a @ self.W[l].T + self.b[l]); A.append(a)
        raw = a @ self.W[-1].T + self.b[-1]
        if OUTSCALE != 1.0:                       # centred lazy/rich scaling: N_s = s (N(theta) - N(theta_0))
            if self.out0 is None: self.out0 = raw.detach().clone()
            raw = raw - self.out0
        return OUTSCALE*raw, A
    def grad(self, X, delta, A):
        """Flat gradient of sum_i delta_i . N(x_i) w.r.t. theta (delta: n x dout)."""
        gW = [None]*len(self.W); gb = [None]*len(self.b)
        gW[-1] = delta.T @ A[-1]; gb[-1] = delta.sum(0)
        dl = delta @ self.W[-1]
        for l in range(self.L-1, -1, -1):
            dl = dl*(1-A[l+1]**2)
            gW[l] = dl.T @ A[l]; gb[l] = dl.sum(0)
            if l > 0: dl = dl @ self.W[l]
        return OUTSCALE*torch.cat([q.reshape(-1) for q in gW + gb])
    def jacobian(self, X):
        """Per-sample Jacobian, rows (i,c): shape (n*dout, d). Same row order as l63test.py."""
        out, A = self.forward(X)
        n, dout = out.shape
        Js = []
        for c in range(dout):
            delta = torch.zeros(n, dout, device=DEV); delta[:, c] = 1.
            gW = [None]*len(self.W); gb = [None]*len(self.b)
            gW[-1] = delta[:, :, None]*A[-1][:, None, :]; gb[-1] = delta
            dl = delta @ self.W[-1]
            for l in range(self.L-1, -1, -1):
                dl = dl*(1-A[l+1]**2)
                gW[l] = dl[:, :, None]*A[l][:, None, :]; gb[l] = dl
                if l > 0: dl = dl @ self.W[l]
            Js.append(torch.cat([g.reshape(n, -1) for g in gW + gb], dim=1))
        J = OUTSCALE*torch.stack(Js, dim=1)              # n x dout x d
        return J.reshape(n*dout, -1), out


def rms_ip(u, v, n): return float(u @ v)/n
def rms_norm(u, n): return float(torch.sqrt(torch.clamp(u @ u, min=0)/n))
def kernel(J, n): return J @ J.T / n

def krylov_phi(K, E, gamma, n, M=60, tol=1e-10):
    A = K + gamma*torch.eye(K.shape[0], device=DEV)
    x = torch.zeros_like(E); r = E.clone(); p = r.clone(); rr = float(r @ r); EE = float(E @ E)
    for k in range(M):
        Ap = A @ p; alpha = rr/float(p @ Ap); x += alpha*p; r -= alpha*Ap
        rr_new = float(r @ r)
        if np.sqrt(rr_new) < tol*np.sqrt(EE): break
        p = r + (rr_new/rr)*p; rr = rr_new
    Eg = gamma*x
    return Eg/rms_norm(Eg, n), Eg, k+1

def along_dir_second(net, X, theta, u, h=1e-4):
    net.unpack(theta + h*u); Jp, _ = net.jacobian(X)
    net.unpack(theta - h*u); Jm, _ = net.jacobian(X)
    net.unpack(theta)
    return (Jp - Jm)/(2*h)

def opnorm(M):
    """Largest singular value via the small Gram matrix (M is 600 x d, d large)."""
    return float(torch.sqrt(torch.clamp(torch.linalg.eigvalsh(M @ M.T)[-1], min=0)))


def run(depth, H, X, Y, T_total, gammas_rel, log_every, seed=0, eta_frac=0.4, tag=''):
    n, ny = Y.shape
    net = MLP(3, H, depth, ny, seed=seed)
    theta = net.pack()
    Yv = Y.reshape(-1)
    J, out = net.jacobian(X)
    E = out.reshape(-1) - Yv
    K = kernel(J, n)
    lam, V = torch.linalg.eigh(K); lam = lam.flip(0); V = V.flip(1)
    lam1_0 = float(lam[0])
    gammas = [g*lam1_0 for g in gammas_rel]
    phis = []; info = []
    for g in gammas:
        phi, Eg, it = krylov_phi(K, E, g, n)
        mask = lam > 1e-14
        VE = (V.T @ E)
        Ups = float((lam[mask]*VE[mask]**2/n/(lam[mask]+g)**2).sum())
        phis.append(phi)
        info.append(dict(gamma=g, cg_iters=it, Egamma_norm=rms_norm(Eg, n),
                         phiE0=rms_ip(phi, E, n), sigma0=np.sqrt(rms_ip(phi, K @ phi, n)),
                         Upsilon=Ups))
    print(f'[{tag}] d_theta={theta.numel()}, lam1(0)={lam1_0:.3e}, ||E0||={rms_norm(E,n):.4f} (device {DEV})')
    for inf in info:
        print(f"   gamma/lam1={inf['gamma']/lam1_0:.0e}: <phi,E0>={inf['phiE0']:.4f}, "
              f"sigma0^2/lam1={inf['sigma0']**2/lam1_0:.3e}, ||E_gamma||={inf['Egamma_norm']:.4f}, cg={inf['cg_iters']}")

    def H2_along(u):
        Pu = along_dir_second(net, X, theta, u/torch.linalg.norm(u))
        return opnorm(Pu)/np.sqrt(n)
    gE = J.T @ E/n
    H2_g0 = H2_along(gE)
    H2_rand = max(H2_along(torch.tensor(rng.normal(size=theta.numel()), device=DEV)) for _ in range(3))
    print(f'   H2 along g_E: {H2_g0:.3e}, along random dirs (max of 3): {H2_rand:.3e}', flush=True)

    eta = eta_frac/lam1_0
    t = 0.; step = 0
    logs = []
    t0 = time.time()
    while t < T_total:
        if step % log_every == 0:
            J, out = net.jacobian(X); E = out.reshape(-1) - Yv
            K = kernel(J, n)
            lam1 = float(torch.linalg.eigvalsh(K)[-1])
            rec = dict(t=t, step=step, E=rms_norm(E, n), lam1=lam1, eta=eta)
            for j, phi in enumerate(phis):
                pE = rms_ip(phi, E, n)
                Kphi = K @ phi
                Es = E - pE*phi
                rec[f'phiE{j}'] = pE
                rec[f'sig2_{j}'] = rms_ip(phi, Kphi, n)
                rec[f'cross{j}'] = rms_ip(Kphi, Es, n)
                rec[f'Es{j}'] = rms_norm(Es, n)
            if (step // log_every) % 5 == 0:
                gE = J.T @ E/n
                gnorm = float(torch.linalg.norm(gE))
                if gnorm > 0:
                    Pu = along_dir_second(net, X, theta, gE/gnorm)
                    rec['H2_g'] = opnorm(Pu)/np.sqrt(n)
                    for j, phi in enumerate(phis):
                        gphi = J.T @ phi/n
                        rec[f'drive{j}'] = -2*float(phi @ (Pu @ gphi))/n*gnorm
            logs.append(rec)
            if (step // log_every) % 20 == 0:
                print(f'   t={t:8.2f} step={step:6d} ||E||={rec["E"]:.4f} lam1={lam1:.3e} '
                      + ' '.join(f'<phi{j},E>={rec[f"phiE{j}"]:+.4f}' for j in range(len(phis)))
                      + f'  ({time.time()-t0:.0f}s)', flush=True)
            if eta*lam1 > 1.0: eta = 0.5/lam1
        out, A = net.forward(X)
        g = net.grad(X, (out - Y)/n, A)
        theta -= eta*g; net.unpack(theta)
        t += eta; step += 1
    return dict(logs=logs, info=info, lam1_0=lam1_0, H2_g0=H2_g0, H2_rand=H2_rand,
                d_theta=theta.numel(), E0=logs[0]['E'],
                phis=torch.stack(phis).cpu().numpy(), gammas=gammas,
                device=str(DEV), wall_s=time.time()-t0, steps=step)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    Delta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    T_total = float(sys.argv[3]) if len(sys.argv) > 3 else 2000.
    depths = [int(s) for s in (sys.argv[4] if len(sys.argv) > 4 else '1,3').split(',')]
    H = int(sys.argv[5]) if len(sys.argv) > 5 else 96
    log_every = int(sys.argv[6]) if len(sys.argv) > 6 else 200
    outdir = sys.argv[7] if len(sys.argv) > 7 else '.'
    Xn, Yn = make_data(n, Delta)
    X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV)
    gammas_rel = [1e-1, 1e-2, 1e-3, 1e-4]
    for depth in depths:
        tag = f'H{H}_D{Delta}_depth{depth}'
        res = run(depth, H, X, Y, T_total, gammas_rel, log_every=log_every, tag=tag)
        res.update(H=H, Delta=Delta, n=n, log_every=log_every)
        np.save(f'{outdir}/res_{tag}.npy', res, allow_pickle=True)
        print(f'   wall {res["wall_s"]:.0f}s for {res["steps"]} steps', flush=True)
    print('done')
