"""Run SegFlow on the L63 designs and compare with the saved full-batch GD curves (runs/heldout/des_*.npy).
usage: python segflow_l63.py TAG [delta0] [m_max] [verbose]   e.g. H48_D0.5_depth2_s1"""
import sys, os, re, time, math, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
torch.set_default_dtype(torch.float64); torch.set_num_threads(int(os.environ.get('SEGFLOW_THREADS', '8'))); torch.set_num_interop_threads(1)
from l63_design import make_data_stats
from l63test_torch import MLP
from segflow import SegFlow
tag = sys.argv[1]; delta0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25; m_max = int(sys.argv[3]) if len(sys.argv) > 3 else 60
verbose = len(sys.argv) > 4 and sys.argv[4] == '1'
MODE = os.environ.get('SEGFLOW_MODE', 'target')
FILT = os.environ.get('SEGFLOW_FILT', 'flow'); SO = os.environ.get('SEGFLOW_SO', '')   # '', 'radius', 'correct', 'both'
KAPPA = float(os.environ.get('SEGFLOW_KAPPA', '0.1')); REC = int(os.environ.get('SEGFLOW_RECYCLE', '0')); PRE = os.environ.get('SEGFLOW_PRECOND', '') or None; OM = float(os.environ.get('SEGFLOW_OM', '0'))   # 'target': run until E <= E_GD(T); 'time': run to flow time T
mm = re.match(r'H(\d+)_D([\d.]+)_depth(\d+)_s(\d+)(_os([\d.]+))?', tag); H, Delta, depth, seed = int(mm[1]), float(mm[2]), int(mm[3]), int(mm[4])
osc = float(mm[6]) if mm[6] else 1.0
d = np.load(f'../runs/heldout/des_{tag}.npy', allow_pickle=True).item(); n = d['n']; T = float(d['t'][-1])
Xr, Yr = make_data_stats(n, Delta, 1); mu, sd = Xr.mean(0), Xr.std(0)
X = torch.tensor((Xr-mu)/sd); Y = torch.tensor((Yr-mu)/sd).reshape(-1)
net = MLP(3, H, depth, 3, seed=seed); shapes = [w.shape for w in net.W] + [b.shape for b in net.b]
theta0 = torch.cat([p.reshape(-1) for p in net.W + net.b]).clone()
def unpack(theta):
    out = []; i = 0
    for s in shapes: k = int(np.prod(s)); out.append(theta[i:i+k].reshape(s)); i += k
    return out[:depth+1], out[depth+1:]
def raw(theta):
    Ws, bs = unpack(theta); a = X
    for l in range(depth): a = torch.tanh(a @ Ws[l].T + bs[l])
    return (a @ Ws[-1].T + bs[-1]).reshape(-1)
nval = int(d['nval']); Xvr, Yvr = make_data_stats(nval, Delta, 1+100); Xv = torch.tensor((Xvr-mu)/sd); Yv = torch.tensor((Yvr-mu)/sd).reshape(-1)
def raw_on(theta, Xin):
    Ws, bs = unpack(theta); a = Xin
    for l in range(depth): a = torch.tanh(a @ Ws[l].T + bs[l])
    return (a @ Ws[-1].T + bs[-1]).reshape(-1)
out0 = raw(theta0).detach(); out0v = raw_on(theta0, Xv).detach()
def val_err(theta):
    o = raw_on(theta, Xv); o = o if osc == 1.0 else osc*(o - out0v); return float(torch.sqrt(torch.sum((o-Yv)**2)/nval))
f = (lambda th: raw(th)) if osc == 1.0 else (lambda th: osc*(raw(th) - out0))
assert abs(float(torch.sqrt(torch.sum((f(theta0)-Y)**2)/n)) - d['E'][0]) < 1e-6, 'init mismatch with saved GD run'
EXPL = os.environ.get('SEGFLOW_EXPLICIT', '0') == '1'
def kernel_explicit(theta):
    """K = J J^T / n for the tanh MLP without forming J: per layer l, K^(l)[(i,c),(j,c')] = (d_l^c_i . d_l^c'_j)(a_{l-1,i} . a_{l-1,j}),
    with d_l^c the backprop signal of output c at layer l and a_{l-1} the layer input (with a 1 for the bias). Returns (K, flops)."""
    Ws, bs = unpack(theta); A = [X]; a = X
    for l in range(depth): a = torch.tanh(a @ Ws[l].T + bs[l]); A.append(a)
    N = n*3; flops = 2*n*theta.numel()                                    # one forward
    K = torch.zeros(N, N); ones3 = torch.ones(3, 3)
    # output layer: d = one-hot(c); K^(out)[(i,c),(j,c')] = delta_cc' * (abar_L,i . abar_L,j)
    Ab = torch.cat([A[-1], torch.ones(n, 1)], 1); GA = Ab @ Ab.T; flops += 2*n*n*Ab.shape[1]
    K += torch.kron(GA, torch.eye(3)); flops += N*N
    D = torch.stack([Ws[-1][c].expand(n, -1) for c in range(3)], 1)        # (n, 3, H): dL/d(pre-activation of last hidden)... start of backprop
    for l in range(depth-1, -1, -1):
        D = D*(1 - A[l+1]**2)[:, None, :]                                   # (n,3,H_l): backprop signal at layer l
        Dm = D.reshape(N, -1); GD = Dm @ Dm.T; flops += 2*N*N*Dm.shape[1]
        Ab = torch.cat([A[l], torch.ones(n, 1)], 1); GA = Ab @ Ab.T; flops += 2*n*n*Ab.shape[1]
        K += GD*torch.kron(GA, ones3); flops += 2*N*N
        if l > 0: D = D @ Ws[l]; flops += 2*n*3*Ws[l].numel()
    sc = 1.0 if osc == 1.0 else osc**2
    return sc*K/n, flops
opt_kwargs = dict(kernel_fn=kernel_explicit if EXPL else None); opt = SegFlow(f, theta0, Y, n=n, delta0=delta0, m_max=m_max, verbose=verbose, filt=FILT, so_radius=SO in ('radius','both'), so_correct=SO in ('correct','both'), kappa=KAPPA, recycle_k=REC, precond=PRE, outer_momentum=OM, **opt_kwargs)
LW = os.environ.get('SEGFLOW_LAYERW', '')
if LW:   # per-layer weights for the metric, applied to weights and biases of each layer, normalised to unit mean
    lw = [float(x) for x in LW.split(',')]; assert len(lw) == depth+1
    blocks = [torch.full((int(np.prod(s)),), lw[i % (depth+1)]) for i, s in enumerate(shapes)]
    Pl = torch.cat(blocks); opt.layer_P = Pl/Pl.mean()
opt.refresh = int(os.environ.get('SEGFLOW_REFRESH', '2')); opt.recycle_mode = os.environ.get('SEGFLOW_RECMODE', 'fast')
if EXPL:
    Kx, flx = kernel_explicit(theta0); Jt = torch.func.jacrev(f)(theta0); Kj = Jt @ Jt.T/n
    print(f'  explicit kernel check: rel err {float(torch.linalg.norm(Kx-Kj)/torch.linalg.norm(Kj)):.2e}; cost {flx/(2*n*theta0.numel()):.1f} fwd-eq (vs 150 for 30 implicit matvecs)')
# validation of the Krylov exponential against the exact kernel at init
J = torch.func.jacrev(f)(theta0); K = J @ J.T / n; lam, V = torch.linalg.eigh(K); r0 = opt.r
from segflow import lanczos
Q, Tm = lanczos(lambda u: J @ (J.T @ u)/n, r0, m_max); th, S = torch.linalg.eigh(Tm)
for t in [0.3/lam[-1], 3/lam[-1], 30/lam[-1], 300/lam[-1]]:
    ex = V @ (torch.exp(-lam*t)*(V.T @ r0)); kr = Q @ (S @ (torch.exp(-th*t)*(S.T @ (Q.T @ r0))))
    print(f'  Krylov-vs-exact exp(-Kt)r at t*lam1={t*float(lam[-1]):g}: rel err {float(torch.linalg.norm(ex-kr)/torch.linalg.norm(ex)):.2e} (m={Tm.shape[0]})')
t0 = time.time(); hist = opt.run(T=T) if MODE == 'time' else opt.run(E_target=float(d['E'][-1]), max_segments=2000, max_wall=1500); wall = time.time()-t0
# compare with GD curve at matching flow times
tg, Eg = np.asarray(d['t']), np.asarray(d['E'])
ts = np.array([h['t'] for h in hist]); Es = np.array([h['E'] for h in hist])
Eg_at = np.interp(ts, tg, Eg); sel = ts > 0
ratio = Es[sel]/Eg_at[sel]
Ev_sf = val_err(opt.theta); Ev_gd = float(np.asarray(d['Ev'])[-1])
assert abs(val_err(theta0) - float(np.asarray(d['Ev'])[0])) < 1e-6, 'val init mismatch'
print(f'{tag} [{MODE},{FILT},so={SO},kappa={KAPPA},rec={REC},pre={PRE},om={OM},explicit={EXPL}]: val E: GD {Ev_gd:.4f} SegFlow {Ev_sf:.4f} | GD E(T)={Eg[-1]:.4f} ({d["steps"]} steps, {d["wall_s"]:.0f}s GPU) | SegFlow E={Es[-1]:.4f} at t={ts[-1]:.4g} (T={T:.4g})  segments {opt.counts["segments"]} (rejected {opt.counts["rejected"]}) '
      f'jvp {opt.counts["jvp"]} vjp {opt.counts["vjp"]} fwd {opt.counts["forward"]} d2 {opt.counts_extra["d2"]} FLOPeq {2*opt.counts["jvp"]+3*opt.counts["vjp"]+opt.counts["forward"]+4*opt.counts_extra["d2"]+opt.flops_extra/(2*n*theta0.numel()):.0f} (explicit-kernel part {opt.flops_extra/(2*n*theta0.numel()):.0f})  wall {wall:.0f}s CPU | track E/E_GD at same t: median {np.median(ratio):.3f} range [{ratio.min():.3f},{ratio.max():.3f}]')
# migration of the residual band and segment lengths
if verbose or True:
    idx = np.unique(np.geomspace(1, len(hist), 12).astype(int)-1)
    print('   seg      t        dt       E      E_GD(t)   band lam   delta    m')
    for i in idx:
        h = hist[i]; print('  %4d  %9.3g  %8.3g  %.4f  %.4f   %8.3g  %7.3g  %3d' % (i+1, h['t'], h['dt'], h['E'], np.interp(h['t'], tg, Eg), h['lam_band'], h['delta'], h['m']))
np.save(f'../runs/segflow/{tag}_{MODE}_{FILT}{("_so"+SO+f"_k{KAPPA:g}") if SO else ""}{f"_rec{REC}" if REC else ""}{f"_pre{PRE}" if PRE else ""}{f"_om{OM:g}" if OM else ""}{"_explicit" if EXPL else ""}_m{m_max}_d{delta0:g}.npy', dict(hist=hist, Ev_sf=Ev_sf, Ev_gd=Ev_gd, tag=tag, delta0=delta0, m_max=m_max, wall=wall, E_gd=Eg, t_gd=tg), allow_pickle=True)
