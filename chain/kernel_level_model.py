"""Kernel-level statistical model. State: lam(rho,t) (spectrum by rank = roughness class), c(rho,t) (residual energy per
class), A_l(t) (amplitude per layer). Dynamics: dc/dt = -2 lam c ; dA_l/dt = sqrt(share_l * sum lam c) (descent identity,
share_l(t) = sum_k q_lk c_k / sum_k lam_k c_k from init blocks); supply: spec(Q) from layer statistics (per-variable Hermite
spectra at the per-unit shifts set by A_l, products within a layer, composition across layers, mass kappa*sum A_l^2 m_l);
lam(.,t) = spec(K0) (+) spec(Q(t)) by free convolution (generic orientation, one R). Inputs at theta_0 only; no eigenvectors
tracked; no network evaluation after initialization. usage: python kernel_level_model.py H DEPTH SEED DELTA OUTSCALE T NCK OUT.npz"""
import numpy as np, sys, os, torch, heapq
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hermite_saturation import coeffs
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
torch.set_default_dtype(torch.float64)
H, depth, seed, Delta, sc, T, NCK, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]), sys.argv[8]
os.environ['L63_OUTSCALE'] = str(sc); RUNS = L63_ROOT + '/runs'; MDEG = 16; KTOP = 600; KAPPA = float(os.environ.get('KAPPA', '2.2')); AMPSCALE = float(os.environ.get('AMPSCALE', '1.0'))
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
# ---- initialization inputs (one forward pass, one Jacobian, one SVD per layer)
n = 200; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV)
net = MLP(3, H, depth, 3, seed=seed)
if sc != 1.0: net.forward(X)
o, Acts = net.forward(X); J0, _ = net.jacobian(X); K0t = kernel(J0, n); w0t, V0t = torch.linalg.eigh(K0t); w0 = w0t.flip(0).cpu().numpy(); V0t = V0t.flip(1)
E0 = o.reshape(-1) - Y.reshape(-1); c = ((V0t.T @ E0)**2/n).cpu().numpy(); K0 = K0t.cpu().numpy()
sizesW = [W.numel() for W in net.W]; offs = np.cumsum([0]+sizesW); L = len(net.W)
q = np.array([torch.einsum('ij,ji->i', V0t.T @ (J0[:, offs[l]:offs[l+1]] @ J0[:, offs[l]:offs[l+1]].T / n), V0t).cpu().numpy() for l in range(L)])
sig = [float((Acts[l] @ net.W[l].T + net.b[l]).std()) for l in range(depth)]
m_l = np.load(f'{RUNS}/initm/m_H{H}_D{Delta}_depth{depth}_s{seed}' + (f'_os{sc:g}' if sc != 1 else '') + '.npz')['m']
# per-variable amplitude ratios per layer from the init gradient chain structure: gradient direction of the initial residual
# restricted to each layer, its SVD gives the 16 spike directions and their relative weights r_j and per-unit shift factors
gE = (J0.T @ E0/n); ratios = []
for l in range(depth):
    Gl = gE[offs[l]:offs[l+1]].reshape(net.W[l].shape); U, S, Vh = torch.linalg.svd(Gl, full_matrices=False); r16 = min(16, S.numel())
    f = [float(((Acts[l] @ Vh[j])).std())*float(U[:, j].abs().mean())*float(S[j]/S[:r16].norm()) for j in range(r16)]   # shift per unit amplitude, per direction
    ratios.append(np.array(f))
rng = np.random.default_rng(seed); R, _ = np.linalg.qr(rng.normal(size=(600, 600))); ORIENT_E = None; ORIENT_V = None; HIST = []; DRIVE_FRAC = float(os.environ.get('DRIVE_FRAC', '0.001'))
def spectrum(A):
    trQ = KAPPA*(A[:depth]**2*m_l[:depth]).sum(); share = A[:depth]**2*m_l[:depth]; share = share/max(share.sum(), 1e-300)
    layer_specs = [[espec(A[l]*f, sig[l]) for f in ratios[l]] for l in range(depth)]
    eig_all = []
    for l in range(depth):
        specs = [s for ll in range(l, depth) for s in layer_specs[ll]]; vals, rest = top_products(specs, KTOP); vals = vals[1:]; vals = vals/(vals.sum()+rest)*trQ*share[l]; eig_all += list(vals)
    eig_all = np.sort(np.array(eig_all))[::-1][:600]; specQ = np.zeros(600); specQ[:len(eig_all)] = eig_all
    global ORIENT_E, ORIENT_V, HIST
    if ORIENT_E is not None and len(HIST) > 0:
        # history-ordered orientation: Q's leading eigenvectors are the directions the residual has passed through
        # (recorded at earlier checkpoints, oldest = most fitted = largest), then the current residual direction gets
        # a component equal to the derived drive share, the rest generic
        lead = [h/np.linalg.norm(h) for h in HIST]
        Bm = np.column_stack(lead + [R[:, j] for j in range(600)]); Qo, _ = np.linalg.qr(Bm); Ro = Qo[:, :600]
        e = ORIENT_E/np.linalg.norm(ORIENT_E)
        Kq = Ro @ np.diag(specQ) @ Ro.T
        drive_q = DRIVE_FRAC*trQ                                  # Q energy along the current residual (statistical share)
        Kq = Kq + drive_q*np.outer(e, e)
    else: Ro = R; Kq = Ro @ np.diag(specQ) @ Ro.T
    w, Vn = np.linalg.eigh(K0 + Kq); return w[::-1], trQ, Vn[:, ::-1]
# ---- integration: residual carried through the generic rotation; amplitudes as sum of squares over classes
checks = np.geomspace(1e-2, T, NCK); ts = np.concatenate([[0.], np.geomspace(1e-3/w0[0], T, 2500)]); ic = 0
V0 = V0t.cpu().numpy(); E = E0.cpu().numpy(); lam = w0.copy(); V = V0.copy(); Pc = np.zeros((L, 600)); A = np.full(L, 1e-6); hist_t = []; hist_E = []; hist_A = []; trQ = 0.; lift_class = np.ones(600)
for i in range(1, len(ts)):
    dt = ts[i]-ts[i-1]
    a = V.T @ E/np.sqrt(n)                                     # residual in the current (rotated) basis
    cls = a*a                                                  # energy per class = per rank of the current spectrum
    Pc += np.sqrt(np.maximum(q*cls[None, :], 0))*np.sqrt(lift_class)[None, :]*dt      # per-layer, per-class path contribution (feature norm lifted by the class gain)
    A = np.sqrt(np.sort(Pc**2, axis=1)[:, -16:].sum(1)) + 1e-6
    E = np.sqrt(n)*(V @ (a*np.exp(-lam*dt)))
    if ic < NCK and ts[i] >= checks[ic]:
        ORIENT_E, ORIENT_V = E.copy(), V.copy(); lam_new, trQ, Vn = spectrum(AMPSCALE*A); HIST.append(E.copy()); lift_class = lam_new/w0; lam, V = lam_new, Vn; ic += 1
    hist_t.append(ts[i]); hist_E.append(np.sqrt(E@E/n)); hist_A.append(A.copy())
np.savez(out, t=np.array(hist_t), E=np.array(hist_E), A=np.array(hist_A), lamT=lam, lam0=w0, trQ=trQ, H=H, depth=depth, seed=seed, Delta=Delta, s=sc)
print(f"H{H} d{depth} s{seed} D{Delta} os{sc:g}: E(T)={hist_E[-1]:.4f} gains k=50,100,200: {lam[49]/w0[49]:.1f} {lam[99]/w0[99]:.1f} {lam[199]/w0[199]:.1f} A={np.round(A,1)} trQ={trQ:.0f}", flush=True)
