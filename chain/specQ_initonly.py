"""Fully init-only spectrum predictor. Inputs at theta_0: K0 eigenpairs, E0, per-layer q_lk, init variances, m_l.
Chain weights J_k and amplitudes A_l from the reduced model (DRIVE=0); mass trQ = KAPPA * sum_l A_l^2 m_l (KAPPA = 2.2, the
measured median bias of the init statistic, section (B)); per-variable Hermite spectra over the chain's 16 directions
per layer; product spectra; composition across layers; free convolution. Reports gains at ranks 50/100/200 vs measured
for a list of designs (references + held-out). usage: python specQ_initonly.py tag1 tag2 ...  (tags with saved ground truth)"""
import numpy as np, sys, os, torch, heapq, glob
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hermite_saturation import coeffs
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
import reduced_model2 as M
torch.set_default_dtype(torch.float64)
RUNS = L63_ROOT + '/runs'; MDEG = 16; KTOP = 600; KAPPA = float(os.environ.get('KAPPA', '2.2')); AMPFIX = float(os.environ.get('AMPFIX', '1.0'))
def espec(a, sig):
    c = coeffs(min(a, 20.0), sig, M=MDEG)**2; return c/c.sum()
def top_products(spectra, K):
    S = [np.sort(s)[::-1] for s in spectra]; idx0 = tuple(0 for _ in S); heap = [(-float(np.prod([s[0] for s in S])), idx0)]; seen = {idx0}; out = []
    while heap and len(out) < K:
        v, idx = heapq.heappop(heap); out.append(-v)
        for d in range(len(S)):
            if idx[d]+1 < len(S[d]):
                nidx = idx[:d] + (idx[d]+1,) + idx[d+1:]
                if nidx not in seen: seen.add(nidx); heapq.heappush(heap, (-(-v)*S[d][idx[d]+1]/S[d][idx[d]], nidx))
    return np.array(out), float(np.prod([s.sum() for s in S])) - sum(out)
def predict_spectrum(H, depth, seed, Delta, sc, T, rng):
    os.environ['L63_OUTSCALE'] = str(sc); n = 200; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV)
    net = MLP(3, H, depth, 3, seed=seed)
    if sc != 1.0: net.forward(X)
    out, Acts = net.forward(X); J0, _ = net.jacobian(X); K0 = kernel(J0, n); w0t, Vt = torch.linalg.eigh(K0); w0 = w0t.flip(0).cpu().numpy(); Vt = Vt.flip(1)
    E0 = out.reshape(-1) - Y.reshape(-1); c = ((Vt.T @ E0)**2/n).cpu().numpy(); lam0 = w0.copy()
    sizesW = [W.numel() for W in net.W]; offs = np.cumsum([0]+sizesW); L = len(net.W)
    q = np.array([torch.einsum('ij,ji->i', Vt.T @ (J0[:, offs[l]:offs[l+1]] @ J0[:, offs[l]:offs[l+1]].T / n), Vt).cpu().numpy() for l in range(L)])
    # reduced-model chain weights (v2 gain law only used to shape the chain; amplitudes scaled by AMPFIX)
    kk = np.arange(1, 601); fr = c[100:].sum()/c.sum(); B = M.B; Jk = np.zeros(600); ts = np.concatenate([[0.], np.geomspace(1e-3/lam0[0], T, 1500)]); Am = np.full(L, 1e-3); hid = list(range(1, L-1)) if L > 2 else [1]; cc = c.copy()
    for i in range(1, len(ts)):
        dt = ts[i]-ts[i-1]; P = max(np.prod(Am[hid]), 1e-6); lk = np.log(kk/100.)
        lift = 1 + np.exp(B[0]+B[1]*np.log(P)+B[2]*lk+B[3]*(depth-2)+B[4]*np.log(fr)+B[5]*lk*np.log(fr)+B[6]*lk*np.log(P)); lam = lam0*lift
        Jk += np.sqrt(np.maximum(cc, 0))*np.sqrt(lift)*dt; cc = cc*np.exp(-2*lam*dt); contrib = q*Jk[None, :]**2; Am = np.sqrt(np.sort(contrib, axis=1)[:, -16:].sum(1))+1e-6
    A = AMPFIX*Am; G = J0.T @ Vt/np.sqrt(n); Jt = torch.tensor(Jk, device=DEV)
    sig = [float((Acts[l] @ net.W[l].T + net.b[l]).std()) for l in range(depth)]
    mfile = f'{RUNS}/initm/m_H{H}_D{Delta}_depth{depth}_s{seed}' + (f'_os{sc:g}' if sc != 1 else '') + '.npz'
    m_l = np.load(mfile)['m']; trQ = KAPPA*(A**2*m_l).sum(); share = np.array([A[l]**2*m_l[l] for l in range(depth)]); share /= share.sum()
    layer_specs = []
    for l in range(depth):
        dWl = -(G[offs[l]:offs[l+1], :] @ Jt).reshape(net.W[l].shape); U, S, Vh = torch.linalg.svd(dWl, full_matrices=False); r16 = min(16, S.numel()); scale = A[l]/float(S[:r16].norm())
        a_j = [float(((Acts[l] @ Vh[j])*S[j]*scale).std())*float(U[:, j].abs().mean()) for j in range(r16)]
        layer_specs.append([espec(a, sig[l]) for a in a_j])
    eig_all = []
    for l in range(depth):
        specs = [s for ll in range(l, depth) for s in layer_specs[ll]]; vals, rest = top_products(specs, KTOP); vals = vals[1:]; vals = vals/(vals.sum()+rest)*trQ*share[l]; eig_all += list(vals)
    eig_all = np.sort(np.array(eig_all))[::-1][:600]; specQ = np.zeros(600); specQ[:len(eig_all)] = eig_all
    outl = []
    for _ in range(3):
        R, _ = np.linalg.qr(rng.normal(size=(600, 600))); outl.append(np.sort(np.linalg.eigvalsh(np.diag(w0) + R @ np.diag(specQ) @ R.T))[::-1])
    return w0, np.median(outl, 0), A, trQ
if __name__ == '__main__':
    rng = np.random.default_rng(0); rows = []
    for fn in sys.argv[1:]:
        r = np.load(fn, allow_pickle=True).item(); tag = r['tag']; sc = r.get('outscale', 1.0); seed = r['seed']
        w0, wp, A, trQ = predict_spectrum(r['H'], r['depth'], seed, r['Delta'], sc, r['t'][-1], rng)
        gm = (r['lam'][-1]/r['lam'][0])[::-1]; gp = wp/w0
        print(f"{tag:28s} A={np.round(A,1)} trQ={trQ:8.1f} | gain meas/pred k=50: {gm[49]:8.1f}/{gp[49]:8.1f}  k=100: {gm[99]:8.1f}/{gp[99]:8.1f}  k=200: {gm[199]:9.1f}/{gp[199]:9.1f}", flush=True)
        rows.append((gp[49]/gm[49], gp[99]/gm[99], gp[199]/gm[199]))
    R = np.array(rows); print(f"\n{len(R)} designs: pred/meas gain median k=50 {np.median(R[:,0]):.2f} [{np.percentile(R[:,0],10):.2f},{np.percentile(R[:,0],90):.2f}]  k=100 {np.median(R[:,1]):.2f} [{np.percentile(R[:,1],10):.2f},{np.percentile(R[:,1],90):.2f}]  k=200 {np.median(R[:,2]):.2f} [{np.percentile(R[:,2],10):.2f},{np.percentile(R[:,2],90):.2f}]")
