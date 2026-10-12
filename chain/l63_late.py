"""
Late-stage training diagnostics for the Lorenz-63 one-step map (torch, float64, GPU).
Same data, init, network and gradient-flow proxy as l63test.py / l63test_torch.py.
No frozen-kernel functionals: everything is measured against the CURRENT kernel K(t).

Logged on a time grid (geometric + linear, ~NLOG points):
  t, step, eta, ||E||, lam (all 600 eigenvalues of K(t)),
  eE2 (energy of E(t) per K(t)-eigenmode, 600), yV2 (energy of Y per eigenmode, 600),
  lam_eff = <E,KE>/||E||^2, c_top = lam_eff/lam1,
  rough_E (energy of E per graph-Laplacian mode, 200), rough_top (roughness index of top-5 kernel eigvecs),
  drive_e = d/dt <e,K e> along e=E/||E|| (Prop 15/16 exact rate), H2_g (||Psi_u||_op/sqrt n along g_E),
  gnorm = |g_E|,
  + K(t) snapshots (float32) at NSNAP geometric times, and E(t) at every log.
usage: python l63_late.py N DELTA T_TOTAL DEPTH H OUTDIR [SEED] [NLOG] [NSNAP]
"""
import numpy as np, sys, time, os, math
import torch
from l63test import make_data
from l63test_torch import MLP, kernel, along_dir_second, opnorm, DEV

torch.set_default_dtype(torch.float64)


def graph_laplacian_basis(Xn, k=10):
    """Symmetric normalized Laplacian of the kNN graph on the (standardized) inputs; eigvecs U (n x n),
    ascending eigenvalues (mode 0 = smoothest). Roughness basis independent of the network."""
    n = Xn.shape[0]
    D2 = ((Xn[:, None, :]-Xn[None, :, :])**2).sum(-1)
    idx = np.argsort(D2, axis=1)[:, 1:k+1]
    W = np.zeros((n, n))
    sig = np.median(np.sqrt(D2[np.arange(n)[:, None], idx]))
    for i in range(n):
        W[i, idx[i]] = np.exp(-D2[i, idx[i]]/sig**2)
    W = np.maximum(W, W.T)
    d = W.sum(1); Dm = 1/np.sqrt(d)
    Lap = np.eye(n) - Dm[:, None]*W*Dm[None, :]
    mu, U = np.linalg.eigh(Lap)
    return mu, U


class GraphStepper:
    """One GD step captured in a CUDA graph (eager fallback on CPU)."""
    def __init__(self, net, X, Y, n, eta0):
        self.net, self.X, self.Y, self.n = net, X, Y, n
        theta0 = net.pack().clone()                    # initial parameters, restored after warmup/capture
        self.theta = theta0.clone()
        self.eta = torch.full((), eta0, device=DEV)
        self.use_graph = (DEV.type == 'cuda') and os.environ.get('L63_NOGRAPH', '0') != '1'
        if self.use_graph:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): self._step()
            torch.cuda.current_stream().wait_stream(s)
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g):
                self._step()
            torch.cuda.synchronize()
            self.theta.copy_(theta0); self.net.unpack(self.theta)
    def _step(self):
        out, A = self.net.forward(self.X)
        g = self.net.grad(self.X, (out - self.Y)/self.n, A)
        self.theta.sub_(self.eta*g)
        self.net.unpack(self.theta)
    def run(self, k):
        if self.use_graph:
            for _ in range(k): self.g.replay()
        else:
            for _ in range(k): self._step()


def main():
    n = int(sys.argv[1]); Delta = float(sys.argv[2]); T_total = float(sys.argv[3])
    depth = int(sys.argv[4]); H = int(sys.argv[5]); outdir = sys.argv[6]
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    NLOG = int(sys.argv[8]) if len(sys.argv) > 8 else 400
    NSNAP = int(sys.argv[9]) if len(sys.argv) > 9 else 30
    eta_frac = 0.4
    dseed = int(os.environ.get('L63_DATASEED', '1'))
    tag = f'H{H}_D{Delta}_depth{depth}_s{seed}' + (f'_n{n}' if n != 200 else '') + (f'_ds{dseed}' if dseed != 1 else '') + (f"_os{os.environ['L63_OUTSCALE']}" if os.environ.get('L63_OUTSCALE', '1') != '1' else '')
    os.makedirs(outdir, exist_ok=True)

    Xn, Yn = make_data(n, Delta, seed=dseed)
    mu, U = graph_laplacian_basis(Xn)
    X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV); Yv = Y.reshape(-1)
    ny = Y.shape[1]
    Ut = torch.tensor(U, device=DEV)                      # n x n, columns = modes
    def rough_coeffs(v):                                   # v: n*ny stacked (i,c) -> energy per Laplacian mode
        M = v.reshape(n, ny)
        return ((Ut.T @ M)**2).sum(1)                      # n
    net = MLP(3, H, depth, ny, seed=seed)
    d_theta = net.pack().numel()

    # time grid
    t_geo = np.geomspace(1e-2, T_total, int(NLOG*0.6)); t_lin = np.linspace(0, T_total, int(NLOG*0.4)+1)[1:]
    t_grid = np.unique(np.concatenate([[0.], t_geo, t_lin]))
    snap_times = np.geomspace(1e-1, T_total, NSNAP)
    snap_idx = set(np.searchsorted(t_grid, snap_times))

    J, out = net.jacobian(X); E = out.reshape(-1) - Yv; K = kernel(J, n)
    lam1_0 = float(torch.linalg.eigvalsh(K)[-1])
    QUAD = os.environ.get('L63_QUAD', '0') == '1'
    SOF = os.environ.get('L63_SOF', '0') == '1'      # second-order-feature spectrum per current mode
    J0 = J.clone() if QUAD else None
    eta0 = eta_frac/lam1_0
    st = GraphStepper(net, X, Y, n, eta0)
    print(f'[{tag}] d_theta={d_theta} lam1(0)={lam1_0:.3e} ||E0||={float(torch.sqrt(E@E/n)):.4f} eta0={eta0:.2e} '
          f'graph={st.use_graph} device={DEV}', flush=True)

    logs = dict(t=[], step=[], eta=[], E=[], lam=[], eE2=[], yV2=[], e0V2=[], lam_eff=[], rough_E=[], rough_top=[],
                drive_e=[], H2_g=[], gnorm=[], Evec=[], snap_t=[], snap_K=[], snap_V=[], snap_Q=[], snap_X=[], pathlen=[], sof_rate=[], sof_q=[], sof_cos=[])
    t = 0.; step = 0; eta = eta0; t0 = time.time(); ilog = 0; pathlen = 0.
    rough_Y = rough_coeffs(Yv).cpu().numpy()
    E0vec = E.clone()
    while ilog < len(t_grid):
        # ---- log at t_grid[ilog]
        J, out = net.jacobian(X); E = out.reshape(-1) - Yv; K = kernel(J, n)
        lam, V = torch.linalg.eigh(K)                       # ascending
        En = float(torch.sqrt(E@E/n)); e = E/torch.sqrt(E@E)
        VE = V.T @ E; VY = V.T @ Yv; VE0 = V.T @ E0vec
        lam_eff = float(e @ (K @ e))
        gE = J.T @ E/n; gnorm = float(torch.linalg.norm(gE))
        Pu = along_dir_second(net, X, st.theta, gE/gnorm) if gnorm > 0 else None
        H2g = opnorm(Pu)/math.sqrt(n) if Pu is not None else float('nan')
        ge = J.T @ e/n
        drive_e = -2*float(e @ (Pu @ ge))/n*gnorm if Pu is not None else float('nan')   # d/dt <e,K e>, e frozen
        top = V[:, -5:]
        rough_top = torch.stack([rough_coeffs(top[:, k]) for k in range(5)], 0)         # 5 x n
        mu_t = torch.tensor(mu, device=DEV)
        rough_idx_top = (rough_top @ mu_t)/rough_top.sum(1)                              # mean Laplacian eigenvalue
        logs['t'].append(t); logs['step'].append(step); logs['eta'].append(eta); logs['E'].append(En)
        logs['lam'].append(lam.cpu().numpy()); logs['eE2'].append((VE**2/n).cpu().numpy()); logs['yV2'].append((VY**2/n).cpu().numpy()); logs['e0V2'].append((VE0**2/n).cpu().numpy())
        logs['lam_eff'].append(lam_eff); logs['rough_E'].append(rough_coeffs(E).cpu().numpy()/n)
        logs['rough_top'].append(rough_idx_top.cpu().numpy()); logs['drive_e'].append(drive_e); logs['H2_g'].append(H2g)
        logs['gnorm'].append(gnorm); logs['Evec'].append(E.cpu().numpy())
        logs['pathlen'].append(pathlen)
        if SOF and Pu is not None:
            # dlam_k/dt = -2|g| v_k^T Pu J^T v_k / n  (Hellmann-Feynman; Pu = dPsi/dh along g_E/|g_E|, theta_dot = -g_E)
            A = Pu.T @ V; Wk = J.T @ V                       # d x 600 each
            dot = (A*Wk).sum(0); nA = torch.linalg.norm(A, dim=0); nW = torch.linalg.norm(Wk, dim=0)
            logs['sof_rate'].append((-2*gnorm*dot/n).cpu().numpy()); logs['sof_q'].append((nA**2/n).cpu().numpy())
            logs['sof_cos'].append((dot/torch.clamp(nA*nW, min=1e-300)).cpu().numpy())
        if ilog in snap_idx:
            if QUAD:
                dJ = J - J0
                logs['snap_Q'].append((dJ @ dJ.T/n).cpu().numpy().astype(np.float32))
                logs['snap_X'].append(((J0 @ dJ.T + dJ @ J0.T)/n).cpu().numpy().astype(np.float32))
            logs['snap_t'].append(t); logs['snap_K'].append(K.cpu().numpy().astype(np.float32)); logs['snap_V'].append(V[:, -20:].cpu().numpy().astype(np.float32))
        if ilog % 40 == 0:
            print(f'   t={t:9.3f} step={step:8d} ||E||={En:.4f} lam1={float(lam[-1]):.3e} lam_eff={lam_eff:.3e} '
                  f'c={lam_eff/float(lam[-1]):.3f} H2={H2g:.2e} ({time.time()-t0:.0f}s)', flush=True)
        if eta*float(lam[-1]) > 1.0:
            eta = 0.5/float(lam[-1]); st.eta.fill_(eta)
        ilog += 1
        if ilog >= len(t_grid): break
        # ---- advance to the next grid time
        k = max(1, int(math.ceil((t_grid[ilog]-t)/eta - 1e-9)))
        st.run(k); t += k*eta; step += k
        pathlen += gnorm*k*eta
    res = {k: (np.array(v) if k not in ('snap_K', 'snap_V') else np.array(v)) for k, v in logs.items()}
    res.update(dict(tag=tag, H=H, depth=depth, Delta=Delta, n=n, seed=seed, d_theta=d_theta, lam1_0=lam1_0,
                    lap_mu=mu, lap_U=U, rough_Y=rough_Y/n, X=Xn, Y=Yn, wall_s=time.time()-t0, steps=step))
    np.save(f'{outdir}/late_{tag}.npy', res, allow_pickle=True)
    print(f'   wall {time.time()-t0:.0f}s for {step} steps; saved late_{tag}.npy', flush=True)


if __name__ == '__main__':
    main()
