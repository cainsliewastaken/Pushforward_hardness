"""Init-only adaptability features per design, without full Jacobians (double autodiff + power iteration).
For each GD/squared-loss design file: rebuild the init network, take K0, E0 from the file, and compute
  H2 along the descent direction d = J^T E0 / n            (should reproduce the saved H2_0)
  H2 along the slow-mode direction d_slow = J^T P_slow E0 / n, P_slow = modes with lam_k(0) T < 1
  per-layer H2 (direction restricted to layer l's parameters) for both directions
  the scalar second derivative ||f''[d,d]||/sqrt(n) for both directions
H2(d) = sigma_max( dJ/dtheta [d] ) / sqrt(n); sigma_max by power iteration with Mw = jvp(jvp(f, w), d), M^T u = grad(jvp(f, d) . u).
usage: python init_features.py OUT.csv glob1 glob2 ..."""
import numpy as np, glob, sys, os, csv, re, torch
torch.set_default_dtype(torch.float64)
import l63test_torch as LT
from l63test import make_data
from torch.func import jvp, grad
DEV = LT.DEV
def features(H, depth, seed, Delta, sc, T, K0, E0, n=200, ny=3, iters=25):
    LT.OUTSCALE = sc; net = LT.MLP(3, H, depth, 3, seed=seed); X = torch.tensor(make_data(n, Delta)[0], device=DEV)
    shapes = [W.shape for W in net.W] + [b.shape for b in net.b]; sizes = [int(np.prod(s)) for s in shapes]; L = len(net.W)
    th0 = net.pack().detach().clone()
    def unflat(th):
        out = []; i = 0
        for s, k in zip(shapes, sizes): out.append(th[i:i+k].reshape(s)); i += k
        return out[:L], out[L:]
    def raw(th):
        Ws, bs = unflat(th); a = X
        for l in range(L-1): a = torch.tanh(a @ Ws[l].T + bs[l])
        return (a @ Ws[-1].T + bs[-1]).reshape(-1)
    raw0 = raw(th0).detach()
    f = (lambda th: sc*(raw(th) - raw0)) if sc != 1.0 else raw
    E0 = torch.tensor(np.asarray(E0, dtype=np.float64), device=DEV); K0 = torch.tensor(np.asarray(K0, dtype=np.float64), device=DEV); lam, V = torch.linalg.eigh(K0)
    slow = lam*T < 1.0; Es = V[:, slow] @ (V[:, slow].T @ E0)
    layer_idx = []; i = 0
    for l in range(L): layer_idx.append((i, i+sizes[l])); i += sizes[l]
    bias_idx = []; 
    for l in range(L): bias_idx.append((i, i+sizes[L+l])); i += sizes[L+l]
    def vjp_dir(v): return grad(lambda th: f(th) @ v)(th0)/n
    def opnorm(d):
        Mw = lambda w: jvp(lambda th: jvp(f, (th,), (w,))[1], (th0,), (d,))[1]
        MTu = lambda u: grad(lambda th: jvp(f, (th,), (d,))[1] @ u)(th0)
        w = torch.randn(th0.numel(), device=DEV, generator=torch.Generator(device=DEV).manual_seed(0)); w /= w.norm(); s = 0.
        for _ in range(iters):
            u = Mw(w); s = float(u.norm()); w = MTu(u); w /= w.norm()
        return s/np.sqrt(n)
    def scalar2(d): return float(jvp(lambda th: jvp(f, (th,), (d,))[1], (th0,), (d,))[1].norm())/np.sqrt(n)
    out = dict(lam1=float(lam[-1]), n_slow=int(slow.sum()), Eslow_frac=float(Es.norm()/E0.norm()))
    for name, v in (('desc', E0), ('slow', Es)):
        g = vjp_dir(v); gn = float(g.norm()); d = g/gn; out[f'gnorm_{name}'] = gn
        out[f'H2_{name}'] = opnorm(d); out[f'S2_{name}'] = scalar2(d)
        for l in range(L):
            m = torch.zeros_like(d); a, b = layer_idx[l]; m[a:b] = 1.; a, b = bias_idx[l]; m[a:b] = 1.
            dl = d*m; nl = float(dl.norm()); out[f'dfrac_{name}_l{l}'] = nl
            out[f'H2_{name}_l{l}'] = opnorm(dl/nl) if nl > 1e-12 else 0.
    return out
rows = []
for pat in sys.argv[2:]:
    for fn in sorted(glob.glob(pat)):
        r = np.load(fn, allow_pickle=True).item()
        if r.get('opt', 'gd') != 'gd' or r.get('loss', 'sq') != 'sq' or r['n'] != 200 or r.get('data', 'l63') != 'l63': continue
        sc = float(r.get('outscale') or 1.0); T = float(np.asarray(r['t'])[-1])
        lam0 = np.asarray(r['lam0']); e2 = np.asarray(r['e0V2_0']); Elazy = float(np.sqrt((e2*np.exp(-2*lam0*T)).sum()))
        y2 = np.asarray(r['yV2_0'])[::-1]; k90 = int(np.searchsorted(np.cumsum(y2)/y2.sum(), 0.9))+1
        fe = features(int(r['H']), int(r['depth']), int(r['seed']), float(r['Delta']), sc, T, np.asarray(r['K0']), np.asarray(r['E0']))
        rec = dict(tag=r['tag'], H=int(r['H']), depth=int(r['depth']), seed=int(r['seed']), D=float(r['Delta']), sc=sc, T=T, ET=float(np.asarray(r['E'])[-1]), Ev=float(np.asarray(r['Ev'])[-1]),
                   H2_saved=float(r['H2_0']), Elazy=Elazy, k90=k90, E0=float(np.asarray(r['E'])[0])); rec.update(fe); rows.append(rec)
        print(f"{rec['tag']:28s} H2 saved {rec['H2_saved']:.3f} recomputed {rec['H2_desc']:.3f} | slow: n={rec['n_slow']} frac={rec['Eslow_frac']:.2f} H2_slow {rec['H2_slow']:.3f} | per-layer slow " + ' '.join(f"{rec[f'H2_slow_l{l}']:.2f}" for l in range(int(r['depth'])+1)), flush=True)
keys = sorted(set().union(*[set(x.keys()) for x in rows]), key=lambda k: (k not in ('tag',), k))
with open(sys.argv[1], 'w') as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow(x) for x in rows]
print('saved', sys.argv[1], len(rows))
