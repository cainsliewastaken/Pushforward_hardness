"""Measure the roughness-theory quantities along a SegFlow path (cheap stand-in for the gradient-flow path):
 sigma_m(t)^2 = lambda_max(P_m K P_m)   (peak feature roughness for the graph-Laplacian high-pass above mode m)
 ||P_m E||, <E_r, K E_r>, c_m = <E_r,KE_r>/(sigma_m^2 ||E_r||^2)   (efficiency constant of the closure)
 lambda_1, H2 = ||f''[g_hat, .]||_op / sqrt(n)  (second-derivative constant), b = sqrt(lambda_1) H2,
 realized drive Theta = d(sigma^2)/dt / 2 vs its bound sigma * b * ||E||.
usage: python segflow_rough.py TAG [every]"""
import sys, os, re, time, math, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
torch.set_default_dtype(torch.float64); torch.set_num_threads(int(os.environ.get('SEGFLOW_THREADS', '8'))); torch.set_num_interop_threads(1)
from l63_design import make_data_stats
from l63test_torch import MLP
from l63_late import graph_laplacian_basis
from segflow import SegFlow, lanczos
tag = sys.argv[1]; every = int(sys.argv[2]) if len(sys.argv) > 2 else 2
mm = re.match(r'H(\d+)_D([\d.]+)_depth(\d+)_s(\d+)(_os([\d.]+))?', tag); H, Delta, depth, seed = int(mm[1]), float(mm[2]), int(mm[3]), int(mm[4]); osc = float(mm[6]) if mm[6] else 1.0
d = np.load(f'../runs/heldout/des_{tag}.npy', allow_pickle=True).item(); n = d['n']
Xr, Yr = make_data_stats(n, Delta, 1); mu, sd = Xr.mean(0), Xr.std(0); Xn = (Xr-mu)/sd
X = torch.tensor(Xn); Y = torch.tensor((Yr-mu)/sd).reshape(-1)
lap_mu, lap_U = graph_laplacian_basis(Xn); U = torch.tensor(lap_U)
MS = [5, 10, 20, 40, 80]
def P_m(v, m):            # high-pass: remove the first m Laplacian modes, per output component (v is (n*3,), row order (i,c))
    V = v.reshape(n, 3); return (V - U[:, :m] @ (U[:, :m].T @ V)).reshape(-1)
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
out0 = raw(theta0).detach(); f = (lambda th: raw(th)) if osc == 1.0 else (lambda th: osc*(raw(th) - out0))
KAPPA = float(os.environ.get('SEGFLOW_KAPPA', '0.1'))
opt = SegFlow(f, theta0, Y, n=n, delta0=0.25, m_max=30, so_radius=True, kappa=KAPPA); opt.record_mid = True
def measure(theta, r):
    jvp = lambda v: torch.func.jvp(f, (theta,), (v,))[1]
    _, pull = torch.func.vjp(f, theta); vjp = lambda u: pull(u)[0]
    K = lambda u: jvp(vjp(u))/n
    rec = dict(E=float(torch.sqrt(r@r/n))); gfull = vjp(r); rec['EKE'] = float(gfull@gfull)/n; rec['rate_id'] = 2*rec['EKE']/float(r@r)   # exact: -d ln||E||^2/dt on the flow
    # lambda_1 (Lanczos from a random start, 15 iterations)
    g0 = torch.randn(r.numel()); Q, T = lanczos(K, g0, 15); rec['lam1'] = float(torch.linalg.eigvalsh(T)[-1])
    # H2 along the unit gradient: ||f''[g_hat,.]||_op/sqrt(n) by power iteration on P P^T, P v = d/de J(theta+e g) v
    g = vjp(r); ghat = g/torch.linalg.norm(g)
    Pv = lambda v: torch.func.jvp(lambda th: torch.func.jvp(f, (th,), (v,))[1], (theta,), (ghat,))[1]
    PTu = lambda u: torch.func.jvp(lambda th: torch.func.vjp(f, th)[1](u)[0], (theta,), (ghat,))[1]
    u = torch.randn(r.numel()); u /= torch.linalg.norm(u)
    for _ in range(12): w = Pv(PTu(u)); s2 = float(torch.linalg.norm(w)); u = w/s2
    rec['H2'] = math.sqrt(s2)/math.sqrt(n); rec['b'] = math.sqrt(max(rec['lam1'], 0))*rec['H2']
    for m in MS:
        Er = P_m(r, m); nEr = float(torch.linalg.norm(Er))
        gr = vjp(Er); rec[f'EKE_{m}'] = float(gr@gr)/n          # <E_r, K E_r> = ||J^T E_r||^2/n
        rec[f'Er_{m}'] = nEr/math.sqrt(n)
        v0 = P_m(torch.randn(r.numel()), m); Qm, Tm = lanczos(lambda v: P_m(K(P_m(v, m)), m), v0, 12)
        rec[f'sig2_{m}'] = float(torch.linalg.eigvalsh(Tm)[-1])
        rec[f'c_{m}'] = rec[f'EKE_{m}']/max(rec[f'sig2_{m}']*nEr**2, 1e-300)
    return rec
recs = []; t0 = time.time(); seg = 0
rec = measure(opt.theta, opt.r); rec['t'] = 0.0; recs.append(rec)
E_target = float(d['E'][-1])
while opt.rms(opt.r) > E_target and opt.counts['segments'] < 3000:
    h = opt.step(); seg += 1
    if h['accepted'] and seg % every == 0:
        th_m = opt.theta_mid; r_m = f(th_m) - Y                                  # true residual at the mid-segment point
        rec = measure(th_m, r_m); rec['t'] = opt.t_mid; rec['passes'] = opt.counts['jvp']; recs.append(rec)
rec = measure(opt.theta, opt.r); rec['t'] = opt.t; recs.append(rec)
print(f'{tag}: {len(recs)} checkpoints, {opt.counts["segments"]} segments, wall {time.time()-t0:.0f}s', flush=True)
ts = np.array([q['t'] for q in recs]); Es = np.array([q['E'] for q in recs]); slope = -np.gradient(np.log(Es**2), ts)
print('   t        E     -dlnE2/dt  2<E,KE>/|E|2   lam1    H2      b    | m=10: Er    sig2     c    | m=40: Er    sig2     c')
for k in np.unique(np.geomspace(1, len(recs), 14).astype(int)-1):
    r_ = recs[k]; print('%8.3g  %.4f  %9.2e  %9.2e  %7.2f  %6.3f  %6.2f | %.4f  %.2e  %.3f | %.4f  %.2e  %.3f' % (r_['t'], r_['E'], slope[k], r_['rate_id'], r_['lam1'], r_['H2'], r_['b'], r_['Er_10'], r_['sig2_10'], r_['c_10'], r_['Er_40'], r_['sig2_40'], r_['c_40']))
os.makedirs('../runs/rough', exist_ok=True); np.save(f'../runs/rough/{tag}.npy', dict(recs=recs, tag=tag, MS=MS, lap_mu=lap_mu, E_gd=np.asarray(d['E']), t_gd=np.asarray(d['t'])), allow_pickle=True)
