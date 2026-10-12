"""v6: fully init-only statistical model with error output.
Checkpoints t_i (geometric). At each: amplitudes A_l(t_i) from the chain (J_k integrated from the residual's K0 coefficients),
spec(Q(t_i)) from the layer statistics (specQ_initonly.predict pieces), K_i = K0 + R diag(specQ_i) R^T with ONE generic R per
design (persistent orientation), eigenpairs (lam_i, V_i); between checkpoints the residual vector decays exactly in that
basis. Output ||E(t)|| and the final spectrum. usage: python reduced_model6.py H DEPTH SEED DELTA OUTSCALE T NCK OUT.npz"""
import numpy as np, sys, os, torch, heapq
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hermite_saturation import coeffs
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
torch.set_default_dtype(torch.float64)
H, depth, seed, Delta, sc, T, NCK, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]), sys.argv[8]
os.environ['L63_OUTSCALE'] = str(sc); RUNS = L63_ROOT + '/runs'; MDEG = 16; KTOP = 600; KAPPA = float(os.environ.get('KAPPA', '2.2'))
def espec(a, sig):
    c = coeffs(min(a, 20.0), sig, M=MDEG)**2; return c/c.sum()
def top_products(spectra, K):
    S = [np.sort(s)[::-1] for s in spectra]; idx0 = tuple(0 for _ in S); heap = [(-float(np.prod([s[0] for s in S])), idx0)]; seen = {idx0}; outl = []
    while heap and len(outl) < K:
        v, idx = heapq.heappop(heap); outl.append(-v)
        for d in range(len(S)):
            if idx[d]+1 < len(S[d]):
                nidx = idx[:d] + (idx[d]+1,) + idx[d+1:]
                if nidx not in seen: seen.add(nidx); heapq.heappush(heap, (-(-v)*S[d][idx[d]+1]/S[d][idx[d]], nidx))
    return np.array(outl), float(np.prod([s.sum() for s in S])) - sum(outl)
n = 200; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV)
net = MLP(3, H, depth, 3, seed=seed)
if sc != 1.0: net.forward(X)
o, Acts = net.forward(X); J0, _ = net.jacobian(X); K0t = kernel(J0, n); w0t, V0t = torch.linalg.eigh(K0t); w0 = w0t.flip(0).cpu().numpy(); V0t = V0t.flip(1); V0 = V0t.cpu().numpy()
E = (o.reshape(-1) - Y.reshape(-1)).cpu().numpy(); K0 = K0t.cpu().numpy()
sizesW = [W.numel() for W in net.W]; offs = np.cumsum([0]+sizesW); L = len(net.W)
q = np.array([torch.einsum('ij,ji->i', V0t.T @ (J0[:, offs[l]:offs[l+1]] @ J0[:, offs[l]:offs[l+1]].T / n), V0t).cpu().numpy() for l in range(L)])
G = (J0.T @ V0t/np.sqrt(n)); sig = [float((Acts[l] @ net.W[l].T + net.b[l]).std()) for l in range(depth)]
m_l = np.load(f'{RUNS}/initm/m_H{H}_D{Delta}_depth{depth}_s{seed}' + (f'_os{sc:g}' if sc != 1 else '') + '.npz')['m']
rng = np.random.default_rng(seed); R, _ = np.linalg.qr(rng.normal(size=(600, 600)))
def spectrum_from_chain(Jk, A):
    trQ = KAPPA*(A**2*m_l).sum(); share = np.array([A[l]**2*m_l[l] for l in range(depth)]); share = share/max(share.sum(), 1e-300)
    Jt = torch.tensor(Jk, device=DEV); layer_specs = []
    for l in range(depth):
        dWl = -(G[offs[l]:offs[l+1], :] @ Jt).reshape(net.W[l].shape); U, S, Vh = torch.linalg.svd(dWl, full_matrices=False); r16 = min(16, S.numel()); scale = A[l]/max(float(S[:r16].norm()), 1e-30)
        a_j = [float(((Acts[l] @ Vh[j])*S[j]*scale).std())*float(U[:, j].abs().mean()) for j in range(r16)]
        layer_specs.append([espec(a, sig[l]) for a in a_j])
    eig_all = []
    for l in range(depth):
        specs = [s for ll in range(l, depth) for s in layer_specs[ll]]; vals, rest = top_products(specs, KTOP); vals = vals[1:]; vals = vals/(vals.sum()+rest)*trQ*share[l]; eig_all += list(vals)
    eig_all = np.sort(np.array(eig_all))[::-1][:600]; specQ = np.zeros(600); specQ[:len(eig_all)] = eig_all
    return specQ, A, trQ
import reduced_model2 as M
B = M.B; kk = np.arange(1, 601); c_init = (V0.T @ E)**2/n; fr = c_init[100:].sum()/c_init.sum(); hid = list(range(1, L-1)) if L > 2 else [1]
checks = np.geomspace(1e-2, T, NCK); ts = np.concatenate([[0.], np.geomspace(1e-3/w0[0], T, 2500)]); ic = 0
lam, V = w0.copy(), V0.copy(); Jk = np.zeros(600); cc = c_init.copy(); A = np.full(L, 1e-3); hist_t = []; hist_E = []; trQ = 0.
for i in range(1, len(ts)):
    dt = ts[i]-ts[i-1]
    # v2 chain bookkeeping (open loop): fitted mild lift shapes J_k and A_l only
    P = max(np.prod(A[hid]), 1e-6); lk = np.log(kk/100.)
    lift2 = 1 + np.exp(B[0]+B[1]*np.log(P)+B[2]*lk+B[3]*(depth-2)+B[4]*np.log(fr)+B[5]*lk*np.log(fr)+B[6]*lk*np.log(P))
    Jk += np.sqrt(np.maximum(cc, 0))*np.sqrt(lift2)*dt; cc = cc*np.exp(-2*w0*lift2*dt); contrib = q*Jk[None, :]**2; A = np.sqrt(np.sort(contrib, axis=1)[:, -16:].sum(1))+1e-6
    # residual decays in the statistical spectrum
    a = V.T @ E/np.sqrt(n); E = np.sqrt(n)*(V @ (a*np.exp(-lam*dt)))
    if ic < NCK and ts[i] >= checks[ic]:
        specQ, A_, trQ = spectrum_from_chain(Jk, A); K = K0 + R @ np.diag(specQ) @ R.T; w, Vn = np.linalg.eigh(K); lam, V = w[::-1], Vn[:, ::-1]; ic += 1
    hist_t.append(ts[i]); hist_E.append(np.sqrt(E@E/n))
np.savez(out, t=np.array(hist_t), E=np.array(hist_E), lamT=lam, lam0=w0, A=A, trQ=trQ, H=H, depth=depth, seed=seed, Delta=Delta, s=sc)
print(f"H{H} d{depth} s{seed} D{Delta} os{sc:g}: E(T)={hist_E[-1]:.4f} lift100={lam[99]/w0[99]:.1f} A={np.round(A,1)} trQ={trQ:.0f}", flush=True)
