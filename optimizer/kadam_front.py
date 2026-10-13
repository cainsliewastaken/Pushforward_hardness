"""[kadam_front.py: copy of kadam.py (2026-10-12) with a front-locked push window (--rho) and front/per-mode critically
damped momentum in mode space (--beta_mode); only the kadam branch differs.]
KAdam: Adam whose per-coordinate normalisation is done in the eigenbasis of the data kernel K = J J^T / n.

Adam normalises each parameter coordinate of the momentum by the running RMS of that coordinate's gradient, so a
pattern moves at the same rate whatever its gradient size. KAdam does the same per kernel eigen-direction:
  J = Q diag(s) V^T  (s_k = sqrt(n lambda_k)),  gradient g = J^T r / n  has mode components g_k = V_k^T g = s_k (Q^T r)_k / n,
  m   <- b1 m + (1-b1) g                       (parameter space, basis-free)
  v_k <- b2 v_k + (1-b2) g_k^2                 (per kernel mode; rotated with the basis at every refresh)
  step = -lr [ V (m_k_hat / (sqrt(v_k_hat)+eps)) + Adam step on the part of m outside span(V) ].
The basis (Q, s, V) is recomputed every R steps from the exact Jacobian; between refreshes V is held fixed.
On refresh the second moment is carried over as v_new = (V_new^T V_old)^2 v_old  (+ the per-coordinate Adam moment for
the part of the new basis the old one did not span).

Baselines in the same harness: Adam (torch), Muon (Nesterov 0.95 + Newton-Schulz(5), biases by Adam, as in
muon_roughness.py). L63 one-step map, tanh MLP, full batch, squared loss, linear lr decay to 0 over the budget.
Output: one JSON line per (run, metric) to the file given by --out; training curve in the same file.
"""
import os, sys, json, math, time, argparse, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
from l63_design import make_data_stats
from l63test_torch import MLP

p = argparse.ArgumentParser()
p.add_argument('--opt', required=True, choices=['adam', 'muon', 'kadam', 'soapm', 'soap', 'signum', 'kron'])
p.add_argument('--kron_stats_every', type=int, default=1, help='kron: recompute the Kronecker factor statistics every k steps (EMA held in between)')
p.add_argument('--kron_subsample', type=int, default=0, help='kron: estimate the factors on this many random training points (0 = all)')
p.add_argument('--kron_S', default='exact', choices=['exact', 'mc', 'emp', 'none'], help='kron: S from per-output signals (exact), one random +-1 output direction (mc), residual-weighted per-sample signals (emp = empirical K-FAC), or identity (none)')
p.add_argument('--soap_sign', type=int, default=0, help='soap: 1 = sign of rotated momentum instead of Adam (SPlus-like)')
p.add_argument('--kron_beta', type=float, default=0.95, help='kron: EMA of the Kronecker factors A=E[a a^T], S=E[sum_c d d^T]')
p.add_argument('--kron_every', type=int, default=10, help='kron: steps between factor eigendecompositions')
p.add_argument('--kron_topk', type=int, default=-1, help='kron: sign-normalised modes per layer by eigenvalue product; -1 = those above reltol*max, capped at min(N, P_l); 0 = all')
p.add_argument('--kron_check', type=int, default=0, help='kron: every this many steps, log how much of each exact per-layer GN band the Kronecker basis captures')
p.add_argument('--sign_floor', type=float, default=0.01, help='KAdam signmode 2: core = m_k / (|m_k| + floor * max|m|)')
p.add_argument('--shampoo_beta', type=float, default=0.95, help='soap: EMA of G G^T and G^T G')
p.add_argument('--signmode', type=int, default=0, help='KAdam: 1 = step wq*sign(m_k) per mode (no running RMS; Muon-like normalisation in the GN basis); 2 = gated sign m_k/(|m_k|+floor*max|m|)')
p.add_argument('--basis_every', type=int, default=10, help='soapm: steps between SVD-basis refreshes of the momentum')
p.add_argument('--lr', type=float, required=True)
p.add_argument('--n', type=int, default=2000)
p.add_argument('--nval', type=int, default=2000)
p.add_argument('--H', type=int, default=160)
p.add_argument('--depth', type=int, default=2)
p.add_argument('--Delta', type=float, default=0.5)
p.add_argument('--steps', type=int, default=3000)
p.add_argument('--seed', type=int, default=1)
p.add_argument('--dataseed', type=int, default=1)
p.add_argument('--refresh', type=int, default=50, help='KAdam: steps between eigenbasis refreshes')
p.add_argument('--qmode', type=float, default=0.0, help='KAdam: weight mode k by (s_k/s_1)^q; 0 = equal push on every mode, 1 = GD-like rates')
p.add_argument('--blocks', default='global', choices=['global', 'layer'], help='KAdam: one global eigenbasis, or one per layer (W_l and b_l together)')
p.add_argument('--topk', type=int, default=0, help='KAdam: keep only the k largest modes per block (0 = all above reltol); the rest goes to the Adam leftover')
p.add_argument('--sign_from', type=int, default=0, help='KAdam signmode 1: modes ranked < sign_from use the per-mode running-RMS step instead of sign')
p.add_argument('--sign_to', type=int, default=0, help='KAdam signmode 1: modes ranked >= sign_to (if >0) also use the running-RMS step; sign only on [sign_from, sign_to)')
p.add_argument('--reltol', type=float, default=1e-12, help='KAdam: keep modes with lambda_k > reltol * lambda_1')
p.add_argument('--b1', type=float, default=0.9)
p.add_argument('--rho', type=float, default=float('inf'), help='front window: modes with lam < lam_front/rho get push (lam/lam_lo)^0.5 (inf = equal push)')
p.add_argument('--beta_mode', default='fixed', choices=['fixed', 'front', 'mode'], help='mode-space momentum: fixed b1; one beta per block critically damping the front band; per-mode critical damping floored at the window edge')
p.add_argument('--rho_d', type=float, default=0.0, help='beta_mode=mode: damping cut a_front/rho_d (0 = use --rho)')
p.add_argument('--beta_max', type=float, default=0.995)
p.add_argument('--centred', type=int, default=0, help='1: train f(theta) - f(theta_0) (target shifted by the initial output; J unchanged)')
p.add_argument('--q_kappa', type=float, default=0.0, help='>0: set q online, holding the Taylor remainder at q_kappa (q starts at --qmode)')
p.add_argument('--cap_gn', type=int, default=0, help='1: cap each mode step at the Gauss-Newton step |g_k|/lam_k / (number of blocks)')
p.add_argument('--tr_kappa', type=float, default=0.0, help='>0: trust-region lr controller holding the nonlinear (Taylor) remainder at kappa x first-order decrease')
p.add_argument('--decay', type=int, default=1, help='1: linear lr decay to 0; 0: constant base lr (controller only)')
p.add_argument('--log_taylor', type=int, default=1, help='1: log the GN-predicted vs actual loss change (one jvp per step)')
p.add_argument('--boost', type=float, default=1.0, help='front=wave: push factor for modes just below the wave, lam in [q25/rho, q25)')
p.add_argument('--front', default='rayleigh', choices=['rayleigh', 'median', 'wave'], help='front eigenvalue: residual Rayleigh quotient, or energy-weighted median eigenvalue of the residual (energy per mode = n g_k^2/lam_k, free)')
p.add_argument('--beta_min', type=float, default=0.5)
p.add_argument('--b2', type=float, default=0.999)
p.add_argument('--eps', type=float, default=1e-8)
p.add_argument('--data_cache', default='', help='npz path: load (X, Y, Xv, Yv) if present, else generate and save')
p.add_argument('--warmup', type=int, default=0, help='linear lr warmup over this many steps (then linear decay to 0)')
p.add_argument('--log_every', type=int, default=10)
p.add_argument('--out', required=True)
a = p.parse_args()

torch.set_default_dtype(torch.float64)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(a.seed)
t_wall0 = time.time()

if a.data_cache and os.path.exists(a.data_cache):
    _z = np.load(a.data_cache); Xr, Yr, Xvr, Yvr = _z['X'], _z['Y'], _z['Xv'], _z['Yv']
    assert Xr.shape[0] == a.n and Xvr.shape[0] == a.nval and float(_z['Delta']) == a.Delta and int(_z['dataseed']) == a.dataseed
else:
    Xr, Yr = make_data_stats(a.n, a.Delta, a.dataseed)
    Xvr, Yvr = make_data_stats(a.nval, a.Delta, a.dataseed + 100)
    if a.data_cache:
        np.savez(a.data_cache, X=Xr, Y=Yr, Xv=Xvr, Yv=Yvr, Delta=a.Delta, dataseed=a.dataseed)
mu, sd = Xr.mean(0), Xr.std(0)
X = torch.tensor((Xr - mu) / sd, device=DEV); Y = torch.tensor((Yr - mu) / sd, device=DEV).reshape(-1)
Xv = torch.tensor((Xvr - mu) / sd, device=DEV); Yv = torch.tensor((Yvr - mu) / sd, device=DEV).reshape(-1)
net = MLP(3, a.H, a.depth, 3, seed=a.seed)
Ws = [w.clone().to(DEV) for w in net.W]; bs = [b.clone().to(DEV) for b in net.b]
shapes = [w.shape for w in Ws] + [b.shape for b in bs]
theta = torch.cat([q.reshape(-1) for q in Ws + bs]).clone()
P = theta.numel(); N = Y.numel(); FWD = 2 * a.n * P            # one forward pass, in flops

def unpack(th):
    out = []; i = 0
    for sh in shapes:
        k = int(np.prod(sh)); out.append(th[i:i + k].reshape(sh)); i += k
    return out[:a.depth + 1], out[a.depth + 1:]

def f_flat(th, Xin):
    Wl, bl = unpack(th); h = Xin
    for l in range(a.depth): h = torch.tanh(h @ Wl[l].T + bl[l])
    return (h @ Wl[-1].T + bl[-1]).reshape(-1)

def loss_grad(th):
    th = th.detach().requires_grad_(True)
    r = f_flat(th, X) - Y; L = 0.5 * torch.dot(r, r) / a.n
    g, = torch.autograd.grad(L, th)
    return r.detach(), L.detach(), g

def jacobian(th):
    # rows ordered (sample, output) to match f_flat's reshape
    def f1(t, x): return f_flat(t, x[None, :])
    Jp = torch.cat([torch.func.vmap(torch.func.jacrev(f1), in_dims=(None, 0))(th, X[i:i + 2000]) for i in range(0, X.shape[0], 2000)], 0)   # n x 3 x P, chunked
    return Jp.reshape(N, P)

def ns5(G, steps=5, eps=1e-7):
    aa, bb, cc = (3.4445, -4.7750, 2.0315); Xm = G / (G.norm() + eps); tr = G.shape[0] > G.shape[1]
    if tr: Xm = Xm.T
    for _ in range(steps):
        A = Xm @ Xm.T; B = bb * A + cc * (A @ A); Xm = aa * Xm + B @ Xm
    return Xm.T if tr else Xm

rows = []
def emit(metric, value, step=None):
    rows.append(dict(opt=a.opt, lr=a.lr, b2=a.b2, warmup=a.warmup, args=' '.join(sys.argv[1:]), n=a.n, H=a.H, depth=a.depth, Delta=a.Delta, seed=a.seed, steps=a.steps,
                     refresh=a.refresh if a.opt == 'kadam' else None, qmode=a.qmode if a.opt in ('kadam', 'soapm', 'soap') else None, kron=dict(beta=a.kron_beta, every=a.kron_every, topk=a.kron_topk, stats_every=a.kron_stats_every, subsample=a.kron_subsample, S=a.kron_S) if a.opt == 'kron' else None, soap_sign=a.soap_sign if a.opt == 'soap' else None, sign_floor=a.sign_floor if a.signmode == 2 else None, topk=a.topk if a.opt == 'kadam' else None, sign_from=a.sign_from or None, sign_to=a.sign_to or None, signmode=a.signmode if a.opt == 'kadam' else None, blocks=a.blocks if a.opt == 'kadam' else None, step=step, metric=metric, value=value))

m = torch.zeros(P, device=DEV); vp = torch.zeros(P, device=DEV)        # Adam moments (parameter space)
nW = sum(w.numel() for w in Ws); mbuf = torch.zeros(nW, device=DEV)    # Muon momentum on weight matrices
flop_eq = 0.0; n_refresh = 0; t_refresh = 0.0
qcur = a.qmode; prevF = None; rf_ema = None; lrmult = 0.01 if a.tr_kappa > 0 else 1.0; prevTR = None; nu_ema = None; num_ema = 0.0; den_ema = 0.0   # controller starts small and grows
theta0 = theta.clone()
offs = np.cumsum([0] + [int(np.prod(sh)) for sh in shapes])     # theta layout: W_0..W_depth, then b_0..b_depth
L1 = a.depth + 1
def layer_idx(l):
    return torch.cat([torch.arange(offs[l], offs[l + 1]), torch.arange(offs[L1 + l], offs[L1 + l + 1])]).to(DEV)
if a.blocks == 'global':
    blocks = [dict(idx=None, V=None, vk=None, wq=None, lam=None, mk=None, rem=None)]
else:
    blocks = [dict(idx=layer_idx(l), V=None, vk=None, wq=None, lam=None, mk=None, rem=None) for l in range(L1)]
spike_steps = {300, 1000, a.steps}
soap_state = [dict(U=None, V=None, v=None, wq=None, L=None, R=None) for _ in range(a.depth + 1)]
kron_state = [dict(A=None, S=None, UA=None, US=None, mask=None, prod=None) for _ in range(a.depth + 1)]
kron_gen = torch.Generator(device=DEV); kron_gen.manual_seed(1000 + a.seed)
def kron_factors(th, r):
    # per layer: A = E[a~ a~^T] (inputs with bias 1); S = E[sum_c delta^c delta^c^T] (exact), its one-direction estimate (mc),
    # the residual-weighted per-sample version E[delta^r delta^r^T] (emp), or the identity (none). Optionally on a random subset.
    Wl, bl = unpack(th)
    if a.kron_subsample and a.kron_subsample < a.n:
        ii = torch.randperm(a.n, generator=kron_gen, device=DEV)[:a.kron_subsample]
    else:
        ii = torch.arange(a.n, device=DEV)
    ns = ii.numel(); h = [X[ii]]
    for l in range(a.depth): h.append(torch.tanh(h[-1] @ Wl[l].T + bl[l]))
    As, Ss = [None] * L1, [None] * L1
    if a.kron_S == 'exact':
        D = torch.eye(3, device=DEV).expand(ns, 3, 3)                       # (ns, c, out)
    elif a.kron_S == 'mc':
        D = (torch.randint(0, 2, (ns, 1, 3), generator=kron_gen, device=DEV).double() * 2 - 1)
    elif a.kron_S == 'emp':
        D = r.reshape(a.n, 3)[ii][:, None, :].clone()                       # residual-weighted: one signal per sample
    else:
        D = None
    for l in range(a.depth, -1, -1):
        at = torch.cat([h[l], torch.ones(ns, 1, device=DEV)], 1)
        As[l] = at.T @ at / ns
        out_d = shapes[l][0]
        Ss[l] = torch.einsum('ncq,ncr->qr', D, D) / ns if D is not None else torch.eye(out_d, device=DEV)
        if l > 0 and D is not None: D = (D @ Wl[l]) * (1 - h[l] ** 2)[:, None, :]
    return As, Ss
def kron_band_check(th, step):
    # exact per-layer GN modes vs the Kronecker basis: fraction of each exact band captured by the kept Kronecker modes
    with torch.enable_grad():
        J = jacobian(th)
    for l in range(L1):
        st = kron_state[l]; ix = layer_idx(l); Jb = J[:, ix]; Pb = Jb.shape[1]; out_d, in_d = shapes[l]
        if Pb <= N:
            lam, Ve = torch.linalg.eigh(Jb.T @ Jb / a.n); lam = lam.flip(0); Ve = Ve.flip(1)
        else:
            lam, Qb = torch.linalg.eigh(Jb @ Jb.T / a.n); lam = lam.flip(0); Qb = Qb.flip(1)
            keep = lam > 1e-12 * lam[0]; lam = lam[keep]; Ve = (Jb.T @ Qb[:, keep]) / (a.n * lam).sqrt()
        keep = lam > 1e-12 * lam[0]; lam = lam[keep]; Ve = Ve[:, keep]
        # Kronecker modes as flattened parameter vectors (W row-major, then bias), kept ones only
        I, Jx = torch.nonzero(st['mask'], as_tuple=True)
        Mk = st['US'][:, I].T[:, :, None] * st['UA'][:, Jx].T[:, None, :]          # k x out x (in+1)
        Vk = torch.cat([Mk[:, :, :in_d].reshape(Mk.shape[0], -1), Mk[:, :, in_d]], 1).T   # P_l x k
        O = Ve.T @ Vk                                                              # exact x kron
        cap = (O ** 2).sum(1)                                                      # per exact mode: fraction in kron span
        for lo, hi in ((0, 32), (32, 128), (128, 512), (512, 2048), (2048, cap.numel())):
            if lo < cap.numel(): emit(f'kron_capture_l{l}_{lo}_{hi}', float(cap[lo:min(hi, cap.numel())].mean()), step)
        emit(f'exact_modes_l{l}', int(cap.numel()), step)
        del Mk, Vk, O
    del J
def spike_log(step):
    # per-layer weight change: norm and effective rank (spike structure)
    for l in range(L1):
        dW = (theta[offs[l]:offs[l + 1]] - theta0[offs[l]:offs[l + 1]]).reshape(shapes[l])
        sv = torch.linalg.svdvals(dW); e = sv ** 2; tot = float(e.sum())
        emit(f'dW_norm_l{l}', float(dW.norm()), step)
        if tot > 0:
            c = torch.cumsum(e, 0) / tot
            emit(f'dW_pr_l{l}', tot ** 2 / float((e ** 2).sum()), step)
            emit(f'dW_n90_l{l}', int((c < 0.90).sum()) + 1, step); emit(f'dW_n99_l{l}', int((c < 0.99).sum()) + 1, step)

if a.centred:                          # centred outputs f - f(theta_0): same J, targets shifted by the initial output
    with torch.no_grad(): Y = Y + f_flat(theta, X); Yv = Yv + f_flat(theta, Xv)
for step in range(1, a.steps + 1):
    fac = ((1 - (step - 1) / a.steps) if a.decay else 1.0) * (min(1.0, step / a.warmup) if a.warmup > 0 else 1.0); lr = a.lr * fac * lrmult
    r, L, g = loss_grad(theta); flop_eq += 3
    if prevF is not None and a.opt == 'kadam':               # function-space Taylor remainder ||df - J dtheta|| / ||J dtheta||
        r_prev, lJd = prevF; lin = float(lJd.norm())
        if lin > 0:
            rf = float(((r - r_prev) + lJd).norm()) / lin
            rf_ema = rf if rf_ema is None else 0.9 * rf_ema + 0.1 * rf
    if prevTR is not None and a.opt == 'kadam':
        L0, pred1, predGN = prevTR                      # nu = Taylor remainder of the last step / its first-order decrease
        num_ema = 0.9 * num_ema + 0.1 * abs(float(L) - L0 - predGN); den_ema = 0.9 * den_ema + 0.1 * abs(pred1)
        nu_ema = num_ema / max(den_ema, 1e-300)          # Taylor remainder / first-order decrease, ratio of running means
        if a.tr_kappa > 0:
            if nu_ema > 1.5 * a.tr_kappa: lrmult *= 0.8
            elif nu_ema < a.tr_kappa / 1.5: lrmult *= 1.05
            lrmult = min(max(lrmult, 1e-3), 1e3)
        if a.q_kappa > 0 and rf_ema is not None and lr > 0.05 * a.lr:   # online q from the function-space remainder
            if rf_ema > 1.5 * a.q_kappa: qcur = min(qcur + 0.02, 0.5)
            elif rf_ema < a.q_kappa / 1.5: qcur = max(qcur - 0.005, -1.0)
        if step % 100 == 0: emit('nu', nu_ema, step); emit('lrmult', lrmult, step); emit('lr_eff', lr, step); emit('q', qcur, step)
        if step % 100 == 0 and rf_ema is not None: emit('rf', rf_ema, step)
    E = math.sqrt(2 * float(L))
    if not math.isfinite(E) or E > 1e3:
        emit('diverged', 1, step); break
    with torch.no_grad():
        if a.opt == 'adam' or a.opt == 'kadam':
            m.mul_(a.b1).add_(g, alpha=1 - a.b1)
            bc1 = 1 - a.b1 ** step; bc2 = 1 - a.b2 ** step
        if a.opt == 'adam':
            vp.mul_(a.b2).addcmul_(g, g, value=1 - a.b2)
            theta -= lr * (m / bc1) / ((vp / bc2).sqrt() + a.eps)
        elif a.opt == 'kron':
            # Kronecker (K-FAC Gauss-Newton) per-layer basis, sign step per mode, Adam on the leftover
            m.mul_(a.b1).add_(g, alpha=1 - a.b1); vp.mul_(a.b2).addcmul_(g, g, value=1 - a.b2)
            bc1 = 1 - a.b1 ** step; bc2 = 1 - a.b2 ** step
            new_stats = kron_state[0]['A'] is None or (step - 1) % a.kron_stats_every == 0
            if new_stats:
                As, Ss = kron_factors(theta, r)
            upd = torch.empty_like(theta)
            for l in range(L1):
                st = kron_state[l]
                if new_stats:
                    if st['A'] is None: st['A'], st['S'] = As[l].clone(), Ss[l].clone()
                    else:
                        st['A'].mul_(a.kron_beta).add_(As[l], alpha=1 - a.kron_beta); st['S'].mul_(a.kron_beta).add_(Ss[l], alpha=1 - a.kron_beta)
                if st['UA'] is None or (step - 1) % a.kron_every == 0:
                    eA, UA = torch.linalg.eigh(st['A']); eS, US = torch.linalg.eigh(st['S'])
                    prod = eS.clamp(min=0)[:, None] * eA.clamp(min=0)[None, :]
                    Pl = prod.numel(); nrel = int((prod > a.reltol * prod.max()).sum()); kk = Pl if a.kron_topk == 0 else (min(N, Pl, nrel) if a.kron_topk < 0 else min(a.kron_topk, Pl))
                    emit(f'kron_modes_l{l}', kk, step) if (step - 1) % 500 == 0 else None
                    mask = torch.zeros(Pl, device=DEV, dtype=torch.bool); mask[torch.topk(prod.reshape(-1), kk).indices] = True
                    st['UA'], st['US'], st['mask'], st['prod'] = UA, US, mask.reshape(prod.shape), prod
                ix = layer_idx(l); out_d, in_d = shapes[l]
                Ml = torch.cat([m[offs[l]:offs[l + 1]].reshape(out_d, in_d), m[offs[L1 + l]:offs[L1 + l + 1]].reshape(out_d, 1)], 1)
                C = st['US'].T @ Ml @ st['UA']
                Cm = C * st['mask']
                kept = st['US'] @ Cm @ st['UA'].T
                sgn = st['US'] @ (torch.sign(C) * st['mask']) @ st['UA'].T
                rest = torch.cat([(Ml - kept)[:, :in_d].reshape(-1), (Ml - kept)[:, in_d]])
                vpl = vp[ix]
                stepl = torch.cat([sgn[:, :in_d].reshape(-1), sgn[:, in_d]]) + (rest / bc1) / ((vpl / bc2).sqrt() + a.eps)
                upd[ix] = stepl
            theta -= lr * upd
            if a.kron_check and (step == 1 or step % a.kron_check == 0):
                kron_band_check(theta, step)
        elif a.opt == 'signum':
            # sign descent with momentum in the parameter basis: the control for KAdam --signmode 1
            m.mul_(a.b1).add_(g, alpha=1 - a.b1)
            theta -= lr * torch.sign(m)
        elif a.opt == 'soap':
            # SOAP: Adam in the eigenbasis of accumulated G G^T (left) and G^T G (right) per weight matrix;
            # second moments carried into a new basis by squared overlaps; momentum in parameter space, rotated every step
            Wl, bl = unpack(g); gW = torch.cat([w.reshape(-1) for w in Wl]); gb = g[nW:]
            mbuf.mul_(a.b1).add_(gW, alpha=1 - a.b1)
            m[nW:].mul_(a.b1).add_(gb, alpha=1 - a.b1); vp[nW:].mul_(a.b2).addcmul_(gb, gb, value=1 - a.b2)
            bc1 = 1 - a.b1 ** step; bc2 = 1 - a.b2 ** step; upd = torch.zeros(P, device=DEV); i = 0
            for li, w in enumerate(Ws):
                k = w.numel(); G = gW[i:i + k].reshape(w.shape); M = mbuf[i:i + k].reshape(w.shape)
                st = soap_state[li]
                if st['L'] is None:
                    st['L'] = torch.zeros(w.shape[0], w.shape[0], device=DEV); st['R'] = torch.zeros(w.shape[1], w.shape[1], device=DEV)
                    st['v'] = torch.zeros(w.shape, device=DEV)
                st['L'].mul_(a.shampoo_beta).add_(G @ G.T, alpha=1 - a.shampoo_beta)
                st['R'].mul_(a.shampoo_beta).add_(G.T @ G, alpha=1 - a.shampoo_beta)
                if st['U'] is None or (step - 1) % a.basis_every == 0:
                    eL, U = torch.linalg.eigh(st['L']); eL = eL.flip(0); U = U.flip(1)
                    eR, Vb = torch.linalg.eigh(st['R']); eR = eR.flip(0); Vb = Vb.flip(1)
                    if st['U'] is not None:
                        OL = (U.T @ st['U']) ** 2; OR = (Vb.T @ st['V']) ** 2
                        st['v'] = OL @ st['v'] @ OR.T
                    st['U'], st['V'] = U, Vb
                    dL = eL.clamp(min=1e-12 * float(eL[0])).sqrt(); dR = eR.clamp(min=1e-12 * float(eR[0])).sqrt()
                    st['wq'] = ((dL[:, None] / dL[0]) * (dR[None, :] / dR[0])) ** (a.qmode / 2)
                Gr = st['U'].T @ G @ st['V']; Mr = st['U'].T @ M @ st['V']
                st['v'].mul_(a.b2).addcmul_(Gr, Gr, value=1 - a.b2)
                Dr = st['wq'] * torch.sign(Mr) if a.soap_sign else st['wq'] * (Mr / bc1) / ((st['v'] / bc2).sqrt() + a.eps)
                upd[i:i + k] = (st['U'] @ Dr @ st['V'].T).reshape(-1); i += k
            upd[nW:] = (m[nW:] / bc1) / ((vp[nW:] / bc2).sqrt() + a.eps)      # biases: Adam at the same lr
            theta -= lr * upd
        elif a.opt == 'soapm':
            # Muon's basis (SVD of the Nesterov momentum per weight matrix), Adam's running RMS in that basis, q weighting
            Wl, bl = unpack(g); gW = torch.cat([w.reshape(-1) for w in Wl]); gb = g[nW:]
            mbuf.mul_(0.95).add_(gW); gn = gW + 0.95 * mbuf
            m[nW:].mul_(a.b1).add_(gb, alpha=1 - a.b1); vp[nW:].mul_(a.b2).addcmul_(gb, gb, value=1 - a.b2)
            bc2 = 1 - a.b2 ** step; upd = torch.zeros(P, device=DEV); i = 0
            for li, w in enumerate(Ws):
                k = w.numel(); M = gn[i:i + k].reshape(w.shape); G = gW[i:i + k].reshape(w.shape)
                st = soap_state[li]
                if st['U'] is None or (step - 1) % a.basis_every == 0:
                    U, S, Vh = torch.linalg.svd(M, full_matrices=True)
                    d_out = torch.full((w.shape[0],), float(S[-1]), device=DEV); d_out[:S.numel()] = S
                    d_in = torch.full((w.shape[1],), float(S[-1]), device=DEV); d_in[:S.numel()] = S
                    d_out = d_out.clamp(min=1e-3 * float(S[0])); d_in = d_in.clamp(min=1e-3 * float(S[0]))
                    st['U'], st['V'] = U, Vh.T
                    st['wq'] = ((d_out[:, None] / S[0]) * (d_in[None, :] / S[0])) ** (a.qmode / 2)
                    if st['v'] is None: st['v'] = torch.zeros(w.shape, device=DEV)
                Gr = st['U'].T @ G @ st['V']; Mr = st['U'].T @ M @ st['V']
                st['v'].mul_(a.b2).addcmul_(Gr, Gr, value=1 - a.b2)
                Dr = st['wq'] * Mr / ((st['v'] / bc2).sqrt() + a.eps)
                upd[i:i + k] = (st['U'] @ Dr @ st['V'].T).reshape(-1); i += k
            bc1 = 1 - a.b1 ** step
            upd[nW:] = 3e-3 / max(a.lr, 1e-300) * (m[nW:] / bc1) / ((vp[nW:] / bc2).sqrt() + a.eps)   # bias lr fixed 3e-3
            theta -= lr * upd
        elif a.opt == 'muon':
            Wl, bl = unpack(g); gW = torch.cat([w.reshape(-1) for w in Wl]); gb = g[nW:]
            mbuf.mul_(0.95).add_(gW); gn = gW + 0.95 * mbuf
            # biases: Adam
            m[nW:].mul_(a.b1).add_(gb, alpha=1 - a.b1); vp[nW:].mul_(a.b2).addcmul_(gb, gb, value=1 - a.b2)
            upd = torch.zeros(P, device=DEV); i = 0
            for w in Ws:
                k = w.numel(); G = gn[i:i + k].reshape(w.shape)
                upd[i:i + k] = (ns5(G) * max(1.0, w.shape[0] / w.shape[1]) ** 0.5).reshape(-1); i += k
            bc1 = 1 - a.b1 ** step; bc2 = 1 - a.b2 ** step
            upd[nW:] = 3e-3 / max(a.lr, 1e-300) * (m[nW:] / bc1) / ((vp[nW:] / bc2).sqrt() + a.eps)   # bias lr fixed 3e-3
            theta -= lr * upd
        else:  # kadam (global block, or one block per layer)
            vp.mul_(a.b2).addcmul_(g, g, value=1 - a.b2)       # per-coordinate moment, used for the out-of-span part
            if blocks[0]['V'] is None or (step - 1) % a.refresh == 0:
                t0 = time.time()
                with torch.enable_grad():
                    J = jacobian(theta)
                flop_eq += 3 * 3                                # Jacobian ~ n_y backward passes
                for bi, B in enumerate(blocks):
                    Jb = J if B['idx'] is None else J[:, B['idx']]
                    Pb = Jb.shape[1]
                    if Pb <= N:                                 # small block: diagonalise the Pb x Pb Gauss-Newton block
                        G = (Jb.T @ Jb) / a.n
                        lam, Vn = torch.linalg.eigh(G); lam = lam.flip(0); Vn = Vn.flip(1)
                        keep = lam > a.reltol * lam[0]; lam = lam[keep]; Vn = Vn[:, keep]
                        if a.topk: lam = lam[:a.topk]; Vn = Vn[:, :a.topk]
                        flop_eq += (2 * N * Pb * Pb + 10 * Pb ** 3) / FWD
                        Qb = None
                    else:                                       # large block: diagonalise the N x N kernel block
                        Kb = (Jb @ Jb.T) / a.n
                        lam, Qb = torch.linalg.eigh(Kb); lam = lam.flip(0); Qb = Qb.flip(1)
                        keep = lam > a.reltol * lam[0]; lam = lam[keep]; Qb = Qb[:, keep]
                        if a.topk: lam = lam[:a.topk]; Qb = Qb[:, :a.topk]
                        Vn = (Jb.T @ Qb) / (a.n * lam).sqrt()
                        flop_eq += (2 * N * N * Pb + 10 * N ** 3 + 2 * N * Pb * Qb.shape[1]) / FWD
                        del Kb
                    sb = (a.n * lam).sqrt()
                    vpb = vp if B['idx'] is None else vp[B['idx']]
                    if B['V'] is None:
                        vk_new = torch.zeros(Vn.shape[1], device=DEV)
                    else:
                        O2 = (Vn.T @ B['V']) ** 2
                        covered = O2.sum(1).clamp(max=1.0)
                        vk_new = O2 @ B['vk'] + (1 - covered) * ((Vn ** 2).T @ vpb)
                        B['rem'] = None if B['rem'] is None else O2 @ B['rem']
                    mk_new = None if B['mk'] is None else Vn.T @ (B['V'] @ B['mk'])      # carry the mode-space momentum into the new basis
                    B['V'], B['vk'], B['wq'] = Vn, vk_new, (sb / sb[0]) ** a.qmode
                    B['sr'] = sb / sb[0]
                    B['lam'], B['mk'] = lam, mk_new
                    if Qb is not None and bi == (1 if a.blocks == 'layer' else 0) and ((step - 1) % 500 == 0 or step + a.refresh > a.steps):
                        cr2 = (Qb.T @ r) ** 2; lfl = float((lam * cr2).sum() / cr2.sum())
                        emit('lamf_refresh', lfl, step)
                        for rk in (1, 10, 100, 300, 1000, 2000):
                            if rk <= lam.numel(): emit(f'lam_r{rk}', float(lam[rk - 1]), step)
                        for fct in (3, 10, 100): emit(f'res_below_f{fct}', float(cr2[lam < lfl / fct].sum() / cr2.sum()), step)
                    emit(f'k_modes_b{bi}', int(Vn.shape[1]), step); emit(f'lambda1_b{bi}', float(lam[0]), step)
                    if bi == 0 and B['idx'] is None and Qb is not None:
                        cr = Qb.T @ r; emit('rayleigh_res', float((lam * cr ** 2).sum() / (r @ r)), step)
                del J
                n_refresh += 1; t_refresh += time.time() - t0
            step_vec = torch.empty_like(theta)
            for B in blocks:
                ix = B['idx']; Vb = B['V']
                gb = g if ix is None else g[ix]; mb = m if ix is None else m[ix]; vpb = vp if ix is None else vp[ix]
                gk = Vb.T @ gb
                B['vk'].mul_(a.b2).addcmul_(gk, gk, value=1 - a.b2)
                mkp = Vb.T @ mb; m_rest = mb - Vb @ mkp
                lamk = B['lam']
                lf = a.n * float(gb @ gb) / max(float(r @ r), 1e-300)       # block front: <r, K_b r>/|r|^2 (free)
                if a.front == 'median':                                     # half the residual energy above, half below
                    ek = (B['vk'] / bc2) / lamk.clamp(min=1e-300); ce = torch.cumsum(ek, 0) / ek.sum().clamp(min=1e-300)
                    lf = float(lamk[min(int((ce < 0.5).sum()), lamk.numel() - 1)])
                q25 = lf
                if a.front == 'wave' and B['rem'] is not None and float(B['rem'].clamp(min=0).sum()) > 0:
                    wts = B['rem'].clamp(min=0).flip(0); la = lamk.flip(0); cw = torch.cumsum(wts, 0) / wts.sum()
                    qi = lambda p: float(la[min(int(torch.searchsorted(cw, torch.tensor(p, device=DEV, dtype=cw.dtype))), la.numel() - 1)])
                    q25, lf = qi(0.25), qi(0.5)
                sq = (B['vk'] / bc2).sqrt() + a.eps
                ak = (lr * lamk / sq).clamp(max=1.0)                         # per-mode step x eigenvalue of the normalised update
                win = (lamk >= lf / 3) & (lamk <= lf * 3)
                if bool(win.any()): af = float(ak[win].median())
                else: af = float(ak[torch.argmin((lamk.clamp(min=1e-300).log() - math.log(max(lf, 1e-300))).abs())])
                if a.beta_mode == 'fixed': bk = torch.full_like(lamk, a.b1)
                elif a.beta_mode == 'front': bk = torch.full_like(lamk, min(max((1 - math.sqrt(af)) ** 2, a.beta_min), a.beta_max))
                else:
                    rd = a.rho_d if a.rho_d > 0 else a.rho
                    fl = af / rd if math.isfinite(rd) else 0.0
                    bk = ((1 - ak.clamp(min=fl).sqrt()) ** 2).clamp(max=a.beta_max)
                if B['mk'] is None: B['mk'] = gk.clone()
                else: B['mk'].mul_(bk).add_((1 - bk) * gk)
                mk = B['mk']
                w = torch.ones_like(lamk)
                if math.isfinite(a.rho):
                    lo = q25 / a.rho; w = torch.where(lamk < lo, (lamk / lo).clamp(min=0).sqrt(), w)
                    if a.boost != 1.0: w = torch.where((lamk >= lo) & (lamk < q25), torch.full_like(w, a.boost), w)
                core = torch.sign(mk) if a.signmode == 1 else mk / sq
                remk = gk * core                                            # this mode's share of the first-order loss removal
                B['rem'] = remk.clone() if B['rem'] is None else B['rem'].mul_(0.9).add_(remk, alpha=0.1)
                if a.q_kappa > 0: B['wq'] = B['sr'] ** qcur
                eff = B['wq'] * w * core
                if a.cap_gn:                                                # never move a mode past its own Gauss-Newton step
                    cap = gk.abs() / (lamk * max(lr, 1e-300) * len(blocks))
                    eff = torch.sign(eff) * torch.minimum(eff.abs(), cap)
                    if step % 100 == 0 and B is blocks[min(1, len(blocks) - 1)]: emit('capped_frac', float((eff.abs() < (B['wq'] * w * core).abs()).float().mean()), step)
                sv = Vb @ eff + (m_rest / bc1) / ((vpb / bc2).sqrt() + a.eps)
                if step % 100 == 0 and B is blocks[min(1, len(blocks) - 1)] and B['rem'] is not None and float(B['rem'].clamp(min=0).sum()) > 0:
                    wt = B['rem'].clamp(min=0).flip(0); cw = torch.cumsum(wt, 0) / wt.sum(); la = lamk.flip(0)
                    emit('wave_med_log', float(la[min(int((cw < 0.5).sum()), la.numel() - 1)]), step)
                if step % 100 == 0 and B is blocks[min(1, len(blocks) - 1)]:   # residual spectrum slope gamma(t): e_k ~ v_k / lam_k
                    ek = ((gk ** 2) / lamk.clamp(min=1e-300)).cpu().numpy(); lk = lamk.cpu().numpy(); okk = lk > 1e-12 * lk[0]
                    lb = np.floor(np.log10(lk[okk])); xs = []; ys = []
                    for bb in np.unique(lb):
                        mm = lb == bb
                        if mm.sum() >= 5: xs.append(np.log10(lk[okk][mm]).mean()); ys.append(np.log10(ek[okk][mm].mean() + 1e-300))
                    if len(xs) >= 3: emit('gamma_t', float(np.polyfit(xs, ys, 1)[0]), step)
                if step % 100 == 0 and B is blocks[min(1, len(blocks) - 1)]:
                    emit('lamf', lf, step); emit('a_front', af, step); emit('wave_q25', q25, step)
                    emit('beta_front', float(bk[win].mean()) if bool(win.any()) else float(bk.mean()), step)
                if ix is None: step_vec = sv
                else: step_vec[ix] = sv
                flop_eq += 2 * 2 * Vb.shape[0] * Vb.shape[1] / FWD
            if a.log_taylor or a.tr_kappa > 0:
                with torch.enable_grad():
                    _, Jd = torch.func.jvp(lambda th: f_flat(th, X), (theta,), (step_vec,))
                flop_eq += 2
                pred1 = -lr * float(g @ step_vec); prevTR = (float(L), pred1, pred1 + 0.5 * lr ** 2 * float(Jd @ Jd) / a.n)
                prevF = (r.clone(), lr * Jd.detach())
            theta -= lr * step_vec
    if step in spike_steps:
        spike_log(step)
    if step % a.log_every == 0 or step == a.steps:
        emit('E_train', E, step); emit('flop_eq', flop_eq, step)

with torch.no_grad():
    r = f_flat(theta, X) - Y; Ef = float(torch.sqrt(torch.dot(r, r) / a.n))
    rv = f_flat(theta, Xv) - Yv; Evf = float(torch.sqrt(torch.dot(rv, rv) / a.nval))
emit('E_train_final', Ef, a.steps); emit('E_val_final', Evf, a.steps); emit('flop_eq_final', flop_eq, a.steps)
emit('wall_s', time.time() - t_wall0); emit('refresh_wall_s', t_refresh); emit('n_refresh', n_refresh); emit('P', P)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, 'a') as fh:
    for rw in rows: fh.write(json.dumps(rw) + '\n')
print(f'{a.opt} rho={a.rho:g} beta={a.beta_mode} rho_d={a.rho_d:g} lr={a.lr:g} q={a.qmode:g} blocks={a.blocks} sign={a.signmode} depth={a.depth} Delta={a.Delta} n={a.n}: E_train={Ef:.4g} E_val={Evf:.4g} '
      f'flop_eq={flop_eq:.4g} wall={time.time()-t_wall0:.0f}s refreshes={n_refresh} ({t_refresh:.0f}s)', flush=True)
