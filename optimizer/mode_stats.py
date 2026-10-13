"""Per-mode statistics along a KronQ2-style run (sign push, q 0, GN cap; Kronecker eigenbasis per layer; full batch) and
what they predict. At checkpoints t (every 100 steps), for every Kronecker mode (i, j) of each hidden layer:
  lam = eS_i eA_j; persistence p = |m_ij| / sqrt(v_ij) (momentum and second moment of the rotated gradient, kept in the
  current basis); secant curvature h = <dGr_ij / d_ij> (consecutive full-batch gradients over the last 20 steps);
  residual energy proxy e = Gr_ij^2 / lam.
Outcome over [t, t + 100]: lift = lam(t+100)/lam(t) and fit = e(t+100)/e(t), modes matched by the dominant overlap of the
factor eigenvectors. Reports rank correlations of p, h/lam, lam with log lift and log fit, within eigenvalue decades."""
import sys, os, math, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
torch.set_default_dtype(torch.float64); torch.set_num_threads(8)
from l63_design import make_data_stats
from l63test_torch import MLP
from scipy.stats import spearmanr
n, H, depth, Delta, lr, steps = 2000, 48, 2, float(sys.argv[1]) if len(sys.argv) > 1 else 1.0, 3e-3, 1600
Xr, Yr = make_data_stats(n, Delta, 1); mx, sx = Xr.mean(0), Xr.std(0); X = torch.tensor((Xr - mx) / sx); Y = torch.tensor((Yr - mx) / sx)
net = MLP(3, H, depth, 3, seed=1); L1 = depth + 1
Wt = [torch.cat([w.cpu(), b.cpu()[:, None]], 1).clone() for w, b in zip(net.W, net.b)]
def forward(Ws, Xin):
    h = Xin
    for l in range(L1):
        z = torch.cat([h, torch.ones(h.shape[0], 1)], 1) @ Ws[l].T; h = torch.tanh(z) if l < L1 - 1 else z
    return h
def grads(Ws):
    Wr = [w.detach().requires_grad_(True) for w in Ws]; r = forward(Wr, X) - Y; L = 0.5 * (r * r).sum() / n
    return float(L), [g.detach() for g in torch.autograd.grad(L, Wr)]
def factors(Ws):
    Wr = [w.detach().requires_grad_(True) for w in Ws]; h = X; zs = []; acts = []
    for l in range(L1):
        at = torch.cat([h, torch.ones(h.shape[0], 1)], 1); acts.append(at.detach()); z = at @ Wr[l].T; zs.append(z); h = torch.tanh(z) if l < L1 - 1 else z
    S = [torch.zeros(w.shape[0], w.shape[0]) for w in Ws]
    for c in range(3):
        gz = torch.autograd.grad(h[:, c].sum(), zs, retain_graph=True)
        for l in range(L1): S[l] += gz[l].T @ gz[l] / n
    return [at.T @ at / n for at in acts], S
M = [torch.zeros_like(w) for w in Wt]; V = [torch.zeros_like(w) for w in Wt]; Afac = Sfac = None; basis = [None] * L1
snap = {}; prevG = None; prevD = None; sec_num = [None] * L1; sec_den = [None] * L1
for step in range(1, steps + 1):
    lr_t = lr * (1 - (step - 1) / steps); Lval, G = grads(Wt)
    if (step - 1) % 10 == 0:
        A, S = factors(Wt)
        Afac = A if Afac is None else [0.95 * a_ + 0.05 * b_ for a_, b_ in zip(Afac, A)]; Sfac = S if Sfac is None else [0.95 * a_ + 0.05 * b_ for a_, b_ in zip(Sfac, S)]
        newb = []
        for l in range(L1):
            eA, UA = torch.linalg.eigh(Afac[l]); eS, US = torch.linalg.eigh(Sfac[l]); newb.append((US, UA, eS.clamp(min=0)[:, None] * eA.clamp(min=0)[None, :]))
        if basis[0] is not None:   # carry rotated moments into the new basis by squared overlaps (as SOAP/KronSign)
            for l in range(L1):
                OS = (newb[l][0].T @ basis[l][0]) ** 2; OA = (newb[l][1].T @ basis[l][1]) ** 2
                V[l] = OS @ V[l] @ OA.T; sec_num[l] = OS @ sec_num[l] @ OA.T if sec_num[l] is not None else None; sec_den[l] = OS @ sec_den[l] @ OA.T if sec_den[l] is not None else None
        basis = newb
    D_all = []; Gr_all = []
    for l in range(L1):
        US, UA, lam = basis[l]; M[l].mul_(0.9).add_(G[l], alpha=0.1)
        C = US.T @ M[l] @ UA; Gr = US.T @ G[l] @ UA; V[l].mul_(0.999).add_(Gr * Gr, alpha=0.001); Gr_all.append(Gr)
        lmax = float(lam.max()); live = lam > 1e-12 * lmax
        Dr = torch.where(live, torch.sign(C), torch.zeros_like(C)); Dr = torch.sign(Dr) * torch.minimum(Dr.abs(), Gr.abs() / (lam.clamp(min=1e-300) * lr_t * L1))
        D_all.append(Dr)
        if prevG is not None and prevD is not None and (step - 1) % 10 != 0:   # secant within one basis: dGr = H_rot (-lr dprev)
            dG = Gr - prevG[l]; dd = -lr_t * prevD[l]
            sn = dG * dd; sd = dd * dd
            sec_num[l] = sn if sec_num[l] is None else 0.95 * sec_num[l] + 0.05 * sn; sec_den[l] = sd if sec_den[l] is None else 0.95 * sec_den[l] + 0.05 * sd
    prevG = Gr_all; prevD = D_all
    for l in range(L1):
        US, UA, _ = basis[l]; Wt[l] -= lr_t * (US @ D_all[l] @ UA.T)
    if step % 100 == 0:
        rec = []
        for l in range(L1 - 1):
            US, UA, lam = basis[l]; C = US.T @ M[l] @ UA; Gr = Gr_all[l]
            pers = (C.abs() / (V[l].sqrt() + 1e-300)).clamp(max=10); hsec = sec_num[l] / (sec_den[l] + 1e-300) if sec_num[l] is not None else torch.zeros_like(lam)
            e = Gr * Gr / lam.clamp(min=1e-300)
            rec.append(dict(US=US.clone(), UA=UA.clone(), lam=lam.clone(), pers=pers.clone(), h=hsec.clone(), e=e.clone()))
        snap[step] = (rec, Lval)
print(f'Delta {Delta}: final E_rms {math.sqrt(2 * Lval):.4f}')
for t in sorted(snap):
    if t + 100 not in snap: continue
    r0, _ = snap[t]; r1, _ = snap[t + 100]
    for l in range(L1 - 1):
        a0, b0 = r0[l], r1[l]
        mS = (b0['US'].T @ a0['US']).abs().argmax(0); mA = (b0['UA'].T @ a0['UA']).abs().argmax(0)   # old index -> new index
        lam0 = a0['lam']; lam1 = b0['lam'][mS][:, mA]; e0 = a0['e']; e1 = b0['e'][mS][:, mA]
        lmax = float(lam0.max()); ok = (lam0 > 1e-10 * lmax) & (lam1 > 0) & (e0 > 0) & (e1 > 0)
        lift = torch.log(lam1[ok] / lam0[ok]).numpy(); fit = torch.log(e1[ok] / e0[ok]).numpy()
        p = a0['pers'][ok].numpy(); hr = (a0['h'][ok] / lam0[ok]).numpy(); ll = torch.log10(lam0[ok] / lmax).numpy()
        out = []
        for name, x in (('persistence', p), ('secant h/lam', hr), ('log lam', ll)):
            # within-decade rank correlation, averaged over decades with >= 30 modes (removes the trivial eigenvalue trend)
            cl, cf = [], []
            for dec in np.unique(np.floor(ll)):
                m_ = np.floor(ll) == dec
                if m_.sum() >= 30: cl.append(spearmanr(x[m_], lift[m_])[0]); cf.append(spearmanr(x[m_], fit[m_])[0])
            out.append(f'{name}: lift {np.nanmean(cl):+.2f} fit {np.nanmean(cf):+.2f}')
        if t in (100, 400, 800, 1200):
            print(f'  t {t:4d} layer {l}: modes {ok.sum()}; median log-lift {np.median(lift):+.2f}; within-decade rank corr with log lift / log fit (fit < 0 = residual removed): ' + ' | '.join(out))
