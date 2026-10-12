"""Design study: train + validation error vs budget, with everything an init-time / few-step predictor could use.
Saves: E_tr(t), E_val(t), lam(t) (600), eE2(t), yV2(t), E0 decomposition; at t=0: K0 (train), K_vt (val x train),
outputs at init on train/val, HF per-mode rates (q_k, cos_k, rate_k), H2, |g|, chi = ||Kdot||/lam1^2;
early kernel snapshots at kernel times tau = lam1(0) t in TAUS (float32).
usage: python l63_design.py N DELTA T_TOTAL DEPTH H OUTDIR [SEED] [NLOG]
env: L63_OUTSCALE (lazy/rich), L63_DATASEED, L63_NVAL (default 200)"""
import numpy as np, sys, time, os, math, torch
from l63test import rk4, flow
from l63test_torch import MLP, kernel, along_dir_second, opnorm, DEV, OUTSCALE
from l63_late import GraphStepper
torch.set_default_dtype(torch.float64)
TAUS = [float(x) for x in os.environ.get('L63_TAUS', '1,3,10,30,100,300').split(',')]

def make_data_stats(n, Delta, seed):
    r = np.random.default_rng(seed); x = np.array([1., 1., 1.])
    for _ in range(4000): x = rk4(x, 0.01)
    X = []
    for _ in range(n):
        for _ in range(r.integers(20, 60)): x = rk4(x, 0.01)
        X.append(x.copy())
    X = np.array(X); Y = np.array([flow(xi, Delta) for xi in X])
    return X, Y

def main():
    n = int(sys.argv[1]); Delta = float(sys.argv[2]); T_total = float(sys.argv[3]); depth = int(sys.argv[4]); H = int(sys.argv[5]); outdir = sys.argv[6]
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0; NLOG = int(sys.argv[8]) if len(sys.argv) > 8 else 250
    dseed = int(os.environ.get('L63_DATASEED', '1')); nval = int(os.environ.get('L63_NVAL', '200'))
    tag = f'H{H}_D{Delta}_depth{depth}_s{seed}' + (f'_n{n}' if n != 200 else '') + (f'_ds{dseed}' if dseed != 1 else '') + (f'_os{OUTSCALE:g}' if OUTSCALE != 1 else '')
    os.makedirs(outdir, exist_ok=True)
    Xr, Yr = make_data_stats(n, Delta, dseed); mu, sd = Xr.mean(0), Xr.std(0)
    Xvr, Yvr = make_data_stats(nval, Delta, dseed+100)
    X = torch.tensor((Xr-mu)/sd, device=DEV); Y = torch.tensor((Yr-mu)/sd, device=DEV)
    Xv = torch.tensor((Xvr-mu)/sd, device=DEV); Yv_ = torch.tensor((Yvr-mu)/sd, device=DEV)
    Yv = Y.reshape(-1); Yvv = Yv_.reshape(-1); ny = Y.shape[1]
    net = MLP(3, H, depth, ny, seed=seed)
    if OUTSCALE != 1.0:                 # centre on train AND val with the same init function
        net.forward(X); out0_tr = net.out0.clone(); net.out0 = None; net.forward(Xv); out0_v = net.out0.clone()
        class _C: pass
        net.out0 = out0_tr; net._out0_val = out0_v
    def val_out():
        if OUTSCALE != 1.0:
            saved = net.out0; net.out0 = net._out0_val; o, _ = net.forward(Xv); net.out0 = saved; return o
        return net.forward(Xv)[0]
    J, out = net.jacobian(X); E = out.reshape(-1) - Yv; K = kernel(J, n)
    lam, V = torch.linalg.eigh(K); lam1_0 = float(lam[-1]); eta0 = 0.4/lam1_0
    # cross kernel val x train at init and init outputs
    if OUTSCALE != 1.0: net.out0 = net._out0_val
    Jv, outv = net.jacobian(Xv); K_vt = Jv @ J.T / n; Ev = outv.reshape(-1) - Yvv
    if OUTSCALE != 1.0: net.out0 = out0_tr
    # HF rates at init
    gE = J.T @ E/n; gnorm = float(torch.linalg.norm(gE)); theta = net.pack()
    Pu = along_dir_second(net, X, theta, gE/gnorm); H2 = opnorm(Pu)/math.sqrt(n)
    A = Pu.T @ V; Wk = J.T @ V; dot = (A*Wk).sum(0); nA = torch.linalg.norm(A, dim=0); nW = torch.linalg.norm(Wk, dim=0)
    Kdot = -gnorm*(Pu @ J.T + J @ Pu.T)/n; chi = float(torch.linalg.matrix_norm(Kdot, 2))/lam1_0**2
    init = dict(K0=K.cpu().numpy().astype(np.float32), K_vt=K_vt.cpu().numpy().astype(np.float32), E0=E.cpu().numpy(), Ev0=Ev.cpu().numpy(),
                lam0=lam.cpu().numpy(), yV2_0=((V.T @ Yv)**2/n).cpu().numpy(), e0V2_0=((V.T @ E)**2/n).cpu().numpy(),
                rate0=(-2*gnorm*dot/n).cpu().numpy(), q0=(nA**2/n).cpu().numpy(), cos0=(dot/torch.clamp(nA*nW, min=1e-300)).cpu().numpy(),
                H2_0=H2, gnorm0=gnorm, chi=chi, lam1_0=lam1_0, d_theta=theta.numel())
    print(f'[{tag}] d={theta.numel()} lam1(0)={lam1_0:.3e} ||E0||={float(torch.sqrt(E@E/n)):.4f} ||Ev0||={float(torch.sqrt(Ev@Ev/nval)):.4f} H2={H2:.2e} chi={chi:.3e} (device {DEV})', flush=True)
    del Jv, Pu, A, Wk, Kdot
    st = GraphStepper(net, X, Y, n, eta0)
    t_grid = np.unique(np.concatenate([[0.], np.geomspace(1e-2, T_total, int(NLOG*0.6)), np.linspace(0, T_total, int(NLOG*0.4)+1)[1:]]))
    snap_t = [tau/lam1_0 for tau in TAUS if tau/lam1_0 < T_total]; snap_i = set(np.searchsorted(t_grid, snap_t))
    logs = dict(t=[], step=[], E=[], Ev=[], lam=[], eE2=[], yV2=[], e0V2=[], lam_eff=[], snap_t=[], snap_K=[], snap_Kvt=[], snap_E=[], snap_Ev=[],
                snap_Kl=[], snap_svW=[], snap_svdW=[], snap_spike_rough=[], snap_spike_amp=[], snap_factor_rough=[])
    LAYERS = os.environ.get('L63_LAYERS', '0') == '1'
    W0 = [W.clone() for W in net.W]
    # column index of each layer's block in the flat Jacobian (order: W_0..W_L, then b_0..b_L)
    sizesW = [W.numel() for W in net.W]; sizesB = [b.numel() for b in net.b]; offB = sum(sizesW)
    idx_l = []
    o = 0; ob = offB
    for l in range(len(net.W)):
        idx_l.append(torch.cat([torch.arange(o, o+sizesW[l]), torch.arange(ob, ob+sizesB[l])]).to(DEV)); o += sizesW[l]; ob += sizesB[l]
    Ut = torch.tensor(np.linalg.eigh(np.eye(1))[1], device=DEV)  # placeholder
    from l63_late import graph_laplacian_basis
    lap_mu, lap_U = graph_laplacian_basis((Xr-mu)/sd); lap_mu_t = torch.tensor(lap_mu, device=DEV); lap_U_t = torch.tensor(lap_U, device=DEV)
    def rough_idx(v):            # v: n-vector over samples
        c = (lap_U_t.T @ v)**2; return float((c @ lap_mu_t)/c.sum())
    def rough_mat(M):            # energy-weighted roughness of the columns of M (n x m)
        C = (lap_U_t.T @ M)**2; return float((lap_mu_t @ C).sum()/C.sum())
    E0vec = E.clone(); t = 0.; step = 0; eta = eta0; t0 = time.time(); ilog = 0
    while ilog < len(t_grid):
        J, out = net.jacobian(X); E = out.reshape(-1) - Yv; K = kernel(J, n); lam, V = torch.linalg.eigh(K)
        Ev = val_out().reshape(-1) - Yvv
        VE = V.T @ E; e = E/torch.sqrt(E@E)
        logs['t'].append(t); logs['step'].append(step); logs['E'].append(float(torch.sqrt(E@E/n))); logs['Ev'].append(float(torch.sqrt(Ev@Ev/nval)))
        logs['lam'].append(lam.cpu().numpy()); logs['eE2'].append((VE**2/n).cpu().numpy()); logs['yV2'].append(((V.T @ Yv)**2/n).cpu().numpy()); logs['e0V2'].append(((V.T @ E0vec)**2/n).cpu().numpy())
        logs['lam_eff'].append(float(e @ (K @ e)))
        if ilog in snap_i:
            logs['snap_t'].append(t); logs['snap_K'].append(K.cpu().numpy().astype(np.float32))
            if OUTSCALE != 1.0: net.out0 = net._out0_val
            Jv_s, _ = net.jacobian(Xv)
            if OUTSCALE != 1.0: net.out0 = out0_tr
            logs['snap_Kvt'].append((Jv_s @ J.T / n).cpu().numpy().astype(np.float32)); logs['snap_E'].append(E.cpu().numpy()); logs['snap_Ev'].append(Ev.cpu().numpy()); del Jv_s
            if LAYERS:
                _, Acts = net.forward(X)
                Kl = []; svW = []; svdW = []; rough = []; amp = []
                for l in range(len(net.W)):
                    Jl = J[:, idx_l[l]]; Kl.append((Jl @ Jl.T/n).cpu().numpy().astype(np.float32))
                    svW.append(torch.linalg.svdvals(net.W[l]).cpu().numpy())
                    dW = net.W[l] - W0[l]; Uu, Ss, Vh = torch.linalg.svd(dW, full_matrices=False); svdW.append(Ss.cpu().numpy())
                    v = Vh[0]; u = Uu[:, 0]; a_in = Acts[l]                       # layer l input activations (n x fan_in)
                    f_in = a_in @ v; z_out = (a_in @ net.W[l].T + net.b[l]) @ u   # spike's input feature and output-side feature
                    rough.append([rough_idx(f_in), rough_idx(z_out), rough_idx(torch.tanh(a_in @ net.W[l].T + net.b[l]) @ u) if l < net.L else float('nan')])
                    amp.append(float(Ss[0]))
                fr = []
                for c in range(ny):                      # backprop factors delta_l for output c
                    dl = torch.zeros(n, ny, device=DEV); dl[:, c] = 1.; dl = dl @ net.W[-1]
                    row = []
                    for l in range(net.L-1, -1, -1):
                        dl = dl*(1-Acts[l+1]**2); row.append([rough_mat(Acts[l+1]), rough_mat(1-Acts[l+1]**2), rough_mat(dl)])
                        if l > 0: dl = dl @ net.W[l]
                    fr.append(row[::-1])                  # layer order 0..L-1
                logs['snap_factor_rough'].append(np.array(fr).mean(0))   # (L_hidden, 3): [act, tanh', delta]
                logs['snap_Kl'].append(np.array(Kl)); logs['snap_svW'].append(svW); logs['snap_svdW'].append(svdW); logs['snap_spike_rough'].append(np.array(rough)); logs['snap_spike_amp'].append(np.array(amp))
        if ilog % 50 == 0: print(f'   t={t:9.3f} step={step:8d} ||E||={logs["E"][-1]:.4f} ||Ev||={logs["Ev"][-1]:.4f} lam1={float(lam[-1]):.3e} ({time.time()-t0:.0f}s)', flush=True)
        if eta*float(lam[-1]) > 1.0: eta = 0.5/float(lam[-1]); st.eta.fill_(eta)
        ilog += 1
        if ilog >= len(t_grid): break
        k = max(1, int(math.ceil((t_grid[ilog]-t)/eta - 1e-9))); st.run(k); t += k*eta; step += k
    res = {k: (np.array(v) if k not in ('snap_svW', 'snap_svdW') else v) for k, v in logs.items()}
    res['lap_mu'] = lap_mu; res['lap_U'] = lap_U
    res.update(init); res.update(dict(tag=tag, H=H, depth=depth, Delta=Delta, n=n, nval=nval, seed=seed, dseed=dseed, outscale=OUTSCALE, Y=Y.cpu().numpy(), Yval=Yv_.cpu().numpy(), wall_s=time.time()-t0, steps=step))
    np.save(f'{outdir}/' + ('lay_' if LAYERS else 'des_') + f'{tag}.npy', res, allow_pickle=True)
    print(f'   wall {time.time()-t0:.0f}s for {step} steps; saved des_{tag}.npy  E_tr={res["E"][-1]:.4f} E_val={res["Ev"][-1]:.4f}', flush=True)

if __name__ == '__main__':
    main()
