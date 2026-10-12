"""Second batch of init-only features: H2 per rank band (directions J^T P_band E0), third-order directional derivatives,
and spectrum-shape features from the saved init spectrum. Appends to the csv of init_features.py by tag."""
import numpy as np, glob, sys, csv, re, torch
torch.set_default_dtype(torch.float64)
import l63test_torch as LT
from l63test import make_data
from torch.func import jvp, grad
DEV = LT.DEV; BANDS = [(0, 10), (10, 50), (50, 200), (200, 600)]
def feats(H, depth, seed, Delta, sc, T, K0, E0, lam0, e2, n=200, iters=20):
    LT.OUTSCALE = sc; net = LT.MLP(3, H, depth, 3, seed=seed); X = torch.tensor(make_data(n, Delta)[0], device=DEV)
    shapes = [W.shape for W in net.W] + [b.shape for b in net.b]; sizes = [int(np.prod(s)) for s in shapes]; L = len(net.W); th0 = net.pack().detach().clone()
    def unflat(th):
        out = []; i = 0
        for s, k in zip(shapes, sizes): out.append(th[i:i+k].reshape(s)); i += k
        return out[:L], out[L:]
    def raw(th):
        Ws, bs = unflat(th); a = X
        for l in range(L-1): a = torch.tanh(a @ Ws[l].T + bs[l])
        return (a @ Ws[-1].T + bs[-1]).reshape(-1)
    raw0 = raw(th0).detach(); f = (lambda th: sc*(raw(th) - raw0)) if sc != 1.0 else raw
    E0 = torch.tensor(np.asarray(E0, np.float64), device=DEV); K0 = torch.tensor(np.asarray(K0, np.float64), device=DEV); lam, V = torch.linalg.eigh(K0); V = V.flip(1)
    def vjp_dir(v): return grad(lambda th: f(th) @ v)(th0)/n
    def opnorm(d):
        Mw = lambda w: jvp(lambda th: jvp(f, (th,), (w,))[1], (th0,), (d,))[1]; MTu = lambda u: grad(lambda th: jvp(f, (th,), (d,))[1] @ u)(th0)
        w = torch.randn(th0.numel(), device=DEV, generator=torch.Generator(device=DEV).manual_seed(0)); w /= w.norm(); s = 0.
        for _ in range(iters): u = Mw(w); s = float(u.norm()); w = MTu(u); w /= w.norm()
        return s/np.sqrt(n)
    def third(d):  # ||f'''[d,d,d]|| / sqrt(n)
        g2 = lambda th: jvp(lambda t2: jvp(f, (t2,), (d,))[1], (th,), (d,))[1]
        return float(jvp(g2, (th0,), (d,))[1].norm())/np.sqrt(n)
    out = {}
    for (a, b) in BANDS:
        Eb = V[:, a:b] @ (V[:, a:b].T @ E0); g = vjp_dir(Eb); d = g/g.norm(); out[f'H2_b{a}_{b}'] = opnorm(d); out[f'gn_b{a}_{b}'] = float(g.norm()); out[f'Efrac_b{a}_{b}'] = float(Eb.norm()/E0.norm())
    g = vjp_dir(E0); d = g/g.norm(); out['S3_desc'] = third(d); out['S2_desc'] = float(jvp(lambda th: jvp(f, (th,), (d,))[1], (th0,), (d,))[1].norm())/np.sqrt(n)
    # spectrum shape (saved, sorted descending) and lazy trajectory
    ls = np.sort(np.asarray(lam0))[::-1]; es = np.asarray(e2)[np.argsort(np.asarray(lam0))[::-1]]
    for k in (10, 50, 100, 200): out[f'lam{k}'] = float(ls[k-1])
    out['tail_a'] = float(-np.polyfit(np.log(np.arange(20, 301)), np.log(ls[19:300]), 1)[0])
    for frac, name in ((0.01, 'Elazy_T100'), (0.1, 'Elazy_T10')): out[name] = float(np.sqrt((es*np.exp(-2*ls*frac*T)).sum()))
    return out
prev = {x['tag']: x for x in csv.DictReader(open(sys.argv[1]))}; rows = []
for pat in sys.argv[2:]:
    for fn in sorted(glob.glob(pat)):
        r = np.load(fn, allow_pickle=True).item(); tag = r['tag']
        if tag not in prev or tag in [x['tag'] for x in rows]: continue
        sc = float(r.get('outscale') or 1.0); T = float(np.asarray(r['t'])[-1])
        fe = feats(int(r['H']), int(r['depth']), int(r['seed']), float(r['Delta']), sc, T, r['K0'], r['E0'], r['lam0'], r['e0V2_0'])
        x = dict(prev[tag]); x.update(fe); rows.append(x)
        print(f"{tag:28s} H2 bands " + ' '.join(f"{fe[f'H2_b{a}_{b}']:.2f}" for a, b in BANDS) + f"  S2 {fe['S2_desc']:.3f} S3 {fe['S3_desc']:.3f} tail_a {fe['tail_a']:.2f}", flush=True)
keys = list(rows[0].keys())
with open(sys.argv[1].replace('.csv', '2.csv'), 'w') as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow(x) for x in rows]
print('saved', len(rows))
