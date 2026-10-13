"""KronQ: scalable sign-push optimizer in a per-layer Kronecker (K-FAC) Gauss-Newton eigenbasis, with the exponent q set
online by the function-space Taylor radius and a per-mode Gauss-Newton cap. Memory ~ Adam + sum_l (in_l^2 + out_l^2);
no exact Jacobian, no eigendecomposition larger than one layer's input or output width.

Per layer (Wt = [W | b], out x (in+1)):  A = E[a~ a~^T] (input covariance), S = E[sum_c d_c d_c^T] (output-side backprop
covariance, exact over the n_y outputs on a stats subset), EMA'd; eigenbases U_A, U_S refreshed every --eig_every steps.
Mode (i, j) has GN eigenvalue lam_ij = eS_i eA_j (K-FAC), leverage s_ij = sqrt(lam_ij).
Step (rotated coordinates):  C = U_S^T M U_A (momentum),  Gr = U_S^T G U_A (gradient),
   D_ij = (s_ij / s_max)^q_l sign(C_ij)            (null modes, lam_ij < reltol lam_max, get 0)
   |D_ij| <= |Gr_ij| / (lam_ij lr L)                 (Gauss-Newton cap; L = number of layers)
   W -= lr U_S D U_A^T.
Online q_l (per layer, round-robin): every --check_every steps one layer l is moved alone by its step on a check subset;
   rf_l = ||f(th + d_l) - f(th) - J_l d_l|| / ||J_l d_l|| (one jvp + one forward); q_l moves to hold rf_l at --q_kappa.
Full batch or minibatch (--batch). Biases ride in Wt.
"""
import os, sys, json, math, time, argparse, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
from l63_design import make_data_stats
from l63test_torch import MLP
p = argparse.ArgumentParser()
p.add_argument('--n', type=int, default=20000); p.add_argument('--nval', type=int, default=5000)
p.add_argument('--H', type=int, default=96); p.add_argument('--depth', type=int, default=2); p.add_argument('--Delta', type=float, default=1.0)
p.add_argument('--steps', type=int, default=3000); p.add_argument('--seed', type=int, default=1); p.add_argument('--dataseed', type=int, default=1)
p.add_argument('--lr', type=float, default=3e-3); p.add_argument('--b1', type=float, default=0.9); p.add_argument('--decay', type=int, default=1)
p.add_argument('--q', type=float, default=0.0, help='initial / fixed exponent'); p.add_argument('--q_kappa', type=float, default=0.0, help='>0: online q per layer')
p.add_argument('--check_every', type=int, default=10); p.add_argument('--cap', type=int, default=1)
p.add_argument('--stats_every', type=int, default=10); p.add_argument('--eig_every', type=int, default=10); p.add_argument('--stats_n', type=int, default=2000)
p.add_argument('--beta_stats', type=float, default=0.95); p.add_argument('--reltol', type=float, default=1e-12)
p.add_argument('--batch', type=int, default=0, help='0 = full batch'); p.add_argument('--data_cache', default=''); p.add_argument('--out', required=True)
p.add_argument('--log_every', type=int, default=10)
a = p.parse_args()
torch.set_default_dtype(torch.float64); DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; t0w = time.time()
if a.data_cache and os.path.exists(a.data_cache):
    zc = np.load(a.data_cache); Xr, Yr, Xvr, Yvr = zc['X'], zc['Y'], zc['Xv'], zc['Yv']
else:
    Xr, Yr = make_data_stats(a.n, a.Delta, a.dataseed); Xvr, Yvr = make_data_stats(a.nval, a.Delta, a.dataseed + 100)
mx, sx = Xr.mean(0), Xr.std(0)
X = torch.tensor((Xr - mx) / sx, device=DEV); Y = torch.tensor((Yr - mx) / sx, device=DEV)
Xv = torch.tensor((Xvr - mx) / sx, device=DEV); Yv = torch.tensor((Yvr - mx) / sx, device=DEV)
net = MLP(3, a.H, a.depth, 3, seed=a.seed); L1 = a.depth + 1
Wt = [torch.cat([w.to(DEV), b.to(DEV)[:, None]], 1).clone() for w, b in zip(net.W, net.b)]   # out x (in+1)
P = sum(w.numel() for w in Wt); FWD = 2 * a.n * P
def forward(Ws, Xin, keep=False):
    acts = []; h = Xin
    for l in range(L1):
        at = torch.cat([h, torch.ones(h.shape[0], 1, device=DEV)], 1); acts.append(at)
        z = at @ Ws[l].T; h = torch.tanh(z) if l < L1 - 1 else z
    return (h, acts) if keep else h
def grads(Ws, Xb, Yb):
    Wr = [w.detach().requires_grad_(True) for w in Ws]; out = forward(Wr, Xb); r = out - Yb
    L = 0.5 * (r * r).sum() / Xb.shape[0]; gs = torch.autograd.grad(L, Wr); return float(L), [g.detach() for g in gs], r.detach()
def factor_stats(Ws, Xs):
    """A_l = E[a~ a~^T]; S_l = E[sum_c d_c d_c^T] with d_c = d out_c / d z_l (exact, n_y backward passes)."""
    Wr = [w.detach().requires_grad_(True) for w in Ws]; h = Xs; zs = []; acts = []
    for l in range(L1):
        at = torch.cat([h, torch.ones(h.shape[0], 1, device=DEV)], 1); acts.append(at.detach())
        z = at @ Wr[l].T; z.retain_grad(); zs.append(z); h = torch.tanh(z) if l < L1 - 1 else z
    m = Xs.shape[0]; S = [torch.zeros(w.shape[0], w.shape[0], device=DEV) for w in Ws]
    for c in range(h.shape[1]):
        gz = torch.autograd.grad(h[:, c].sum(), zs, retain_graph=True)
        for l in range(L1): S[l] += gz[l].T @ gz[l] / m
    A = [at.T @ at / m for at in acts]
    return A, S
st = [dict(A=None, S=None, UA=None, US=None, lam=None, q=a.q) for _ in range(L1)]
M = [torch.zeros_like(w) for w in Wt]; flop = 0.0; rows = []; prev = None; rf_hist = []
gen = torch.Generator(device=DEV); gen.manual_seed(7 + a.seed)
def emit(metric, value, step=None):
    rows.append(dict(opt='kronq', args=' '.join(sys.argv[1:]), n=a.n, H=a.H, Delta=a.Delta, step=step, metric=metric, value=value))
for step in range(1, a.steps + 1):
    lr = a.lr * ((1 - (step - 1) / a.steps) if a.decay else 1.0)
    if a.batch: idx = torch.randint(0, a.n, (a.batch,), device=DEV, generator=gen); Xb, Yb = X[idx], Y[idx]
    else: Xb, Yb = X, Y
    Lb, G, r = grads(Wt, Xb, Yb); flop += 3 * Xb.shape[0] / a.n
    if (step - 1) % a.stats_every == 0:
        sidx = torch.randint(0, a.n, (min(a.stats_n, a.n),), device=DEV, generator=gen)
        A, S = factor_stats(Wt, X[sidx]); flop += 7 * len(sidx) / a.n
        for l, s_ in enumerate(st):
            if s_['A'] is None: s_['A'], s_['S'] = A[l], S[l]
            else: s_['A'].mul_(a.beta_stats).add_(A[l], alpha=1 - a.beta_stats); s_['S'].mul_(a.beta_stats).add_(S[l], alpha=1 - a.beta_stats)
    if (step - 1) % a.eig_every == 0:
        for s_ in st:
            eA, UA = torch.linalg.eigh(s_['A']); eS, US = torch.linalg.eigh(s_['S'])
            s_['UA'], s_['US'] = UA, US; s_['lam'] = eS.clamp(min=0)[:, None] * eA.clamp(min=0)[None, :]
            flop += 10 * (UA.shape[0] ** 3 + US.shape[0] ** 3) / FWD
    D_all = []
    with torch.no_grad():
        for l, (w, g, m, s_) in enumerate(zip(Wt, G, M, st)):
            m.mul_(a.b1).add_(g, alpha=1 - a.b1)
            C = s_['US'].T @ m @ s_['UA']; Gr = s_['US'].T @ g @ s_['UA']; lam = s_['lam']; lmax = float(lam.max())
            live = lam > a.reltol * lmax
            wq = torch.where(live, (lam.clamp(min=1e-300) / lmax) ** (0.5 * s_['q']), torch.zeros_like(lam))
            Dr = wq * torch.sign(C)
            if a.cap:
                capv = Gr.abs() / (lam.clamp(min=1e-300) * max(lr, 1e-300) * L1)
                Dr = torch.sign(Dr) * torch.minimum(Dr.abs(), capv)
            D = s_['US'] @ Dr @ s_['UA'].T; D_all.append(D)
            flop += 8 * w.shape[0] * w.shape[1] * (w.shape[0] + w.shape[1]) / FWD
        if step % a.check_every == 0:              # round-robin per-layer Taylor remainder on a check subset
            l = (step // a.check_every) % L1
            cidx = torch.randint(0, a.n, (min(2000, a.n),), device=DEV, generator=gen); Xc = X[cidx]
            tang = tuple((-lr * D_all[k]) if k == l else torch.zeros_like(Wt[k]) for k in range(L1))
            with torch.enable_grad():
                f_old, Jd = torch.func.jvp(lambda *ws: forward(list(ws), Xc), tuple(Wt), tang)
            f_l = forward([w - lr * D_all[k] if k == l else w for k, w in enumerate(Wt)], Xc)
            lin = float(Jd.norm()); flop += 3 * len(cidx) / a.n
            if lin > 0:
                rf = float((f_l - f_old - Jd).norm()) / lin; st[l]['rf'] = rf if st[l].get('rf') is None else 0.7 * st[l]['rf'] + 0.3 * rf
                if a.q_kappa > 0 and lr > 0.05 * a.lr:
                    if st[l]['rf'] > 1.5 * a.q_kappa: st[l]['q'] = min(st[l]['q'] + 0.1, 0.5)
                    elif st[l]['rf'] < a.q_kappa / 1.5: st[l]['q'] = max(st[l]['q'] - 0.025, -1.0)
        for w, D in zip(Wt, D_all): w.sub_(lr * D)
    if step % a.log_every == 0 or step == a.steps:
        with torch.no_grad(): rr = forward(Wt, X) - Y
        emit('E_train', float(torch.sqrt((rr * rr).sum() / a.n)), step); emit('flop_eq', flop, step)
        if step % 100 == 0:
            for l, s_ in enumerate(st):
                emit(f'q_l{l}', s_['q'], step)
                if s_.get('rf') is not None: emit(f'rf_l{l}', s_['rf'], step)
with torch.no_grad():
    rr = forward(Wt, X) - Y; Ef = float(torch.sqrt((rr * rr).sum() / a.n)); rv = forward(Wt, Xv) - Yv; Evf = float(torch.sqrt((rv * rv).sum() / a.nval))
for k_, v_ in (('E_train_final', Ef), ('E_val_final', Evf), ('flop_eq_final', flop), ('wall_s', time.time() - t0w), ('P', P)): emit(k_, v_, a.steps)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, 'a') as fh:
    for rw in rows: fh.write(json.dumps(rw) + '\n')
print(f'kronq q0={a.q} q_kappa={a.q_kappa} lr={a.lr} batch={a.batch} H={a.H} n={a.n} Delta={a.Delta}: E_train={Ef:.4g} E_val={Evf:.4g} '
      f'flop_eq={flop:.4g} q_end={[round(s_["q"], 3) for s_ in st]} wall={time.time() - t0w:.0f}s', flush=True)
