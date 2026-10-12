"""Exact degree->rank mapping. For each layer l: chain spike feature s_l(x) at theta_0 = top singular direction of the
chain's dW_l applied to the init activations a_{l-1}(x) (normalised to unit variance over the data); He_m(s_l) projected
onto the K0 eigenvectors (tiled over the 3 outputs) gives the rank distribution of degree m exactly. spec(Q) by rank =
trQ * sum_l share_l sum_m p_l(m) * dist_{l,m}(rank). Shape test on the two references with measured trQ."""
import numpy as np, sys, os, torch
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hermite_saturation import coeffs
from numpy.polynomial.hermite_e import hermeval
from scipy.special import factorial
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
import reduced_model2 as M
torch.set_default_dtype(torch.float64)
RUNS = L63_ROOT + '/runs'; MDEG = 30
def degree_dist(a, sig):
    c = coeffs(min(a, 20.0), sig, M=MDEG)**2; return c/c.sum()
def convolve(p, q):
    r = np.convolve(p, q); return r[:len(p)]/r[:len(p)].sum()
rng = np.random.default_rng(0)
for tag, Delta, depth in [('H96_D0.5_depth2_s0', 0.5, 2), ('H96_D1.0_depth3_s0', 1.0, 3)]:
    C = np.load(f'{RUNS}/sweepC/late_{tag}.npy', allow_pickle=True).item(); L2 = np.load(f'{RUNS}/layers2/lay_{tag}.npy', allow_pickle=True).item()
    K0 = C['snap_K'][0].astype(np.float64); KT = C['snap_K'][-1].astype(np.float64); Q = C['snap_Q'][-1].astype(np.float64)
    w0, V0 = np.linalg.eigh(K0); w0 = w0[::-1]; V0 = V0[:, ::-1]; wT = np.sort(np.linalg.eigvalsh(KT))[::-1]; wQ = np.sort(np.linalg.eigvalsh(Q))[::-1]; trQ = np.trace(Q)
    A = [np.linalg.norm(sv[:16]) for sv in L2['snap_svdW'][-1]]; H = 96; n = 200
    Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV); net = MLP(3, H, depth, 3, seed=0); out, Acts = net.forward(X)
    J0, _ = net.jacobian(X); E0 = out.reshape(-1) - Y.reshape(-1); w0t, Vt = torch.linalg.eigh(kernel(J0, n)); Vt = Vt.flip(1)
    sizesW = [W.numel() for W in net.W]; offs = np.cumsum([0]+sizesW)
    # chain weights (reduced model) and chain dW per layer at theta_0
    z = np.load(f'{RUNS}/initq/q_{tag}.npz'); lam0 = z['lam0']; c = z['c0'].copy(); q = z['q']; kk = np.arange(1, 601); fr = c[100:].sum()/c.sum(); B = M.B
    Jk = np.zeros(600); ts = np.concatenate([[0.], np.geomspace(1e-3/lam0[0], 3000, 1500)]); Am = np.full(depth+1, 1e-3); hid = list(range(1, depth)) if depth > 1 else [1]
    for i in range(1, len(ts)):
        dt = ts[i]-ts[i-1]; P = max(np.prod(Am[hid]), 1e-6); lk = np.log(kk/100.)
        lift = 1 + np.exp(B[0]+B[1]*np.log(P)+B[2]*lk+B[3]*(depth-2)+B[4]*np.log(fr)+B[5]*lk*np.log(fr)+B[6]*lk*np.log(P)); lam = lam0*lift
        Jk += np.sqrt(np.maximum(c, 0))*np.sqrt(lift)*dt; c = c*np.exp(-2*lam*dt); contrib = q*Jk[None, :]**2; Am = np.sqrt(np.sort(contrib, axis=1)[:, -16:].sum(1))+1e-6
    G = J0.T @ Vt/np.sqrt(n); Jt = torch.tensor(Jk, device=DEV)
    sig = [float((Acts[l] @ net.W[l].T + net.b[l]).std()) for l in range(depth)]
    m_l = np.load(f'{RUNS}/initm/m_{tag}.npz')['m']; share = np.array([A[l]**2*m_l[l] for l in range(depth)]); share /= share.sum()
    specQ = np.zeros(600); V0t = None
    dists = []; feats = []
    for l in range(depth):
        dWl = -(G[offs[l]:offs[l+1], :] @ Jt).reshape(net.W[l].shape); U, S, Vh = torch.linalg.svd(dWl, full_matrices=False)
        # per-unit pre-activation shift field from the rank-16 chain, scaled to the measured amplitude A_l
        shift = (Acts[l] @ (Vh[:16].T*S[:16]) @ U[:, :16].T) * (A[l]/float(S[:16].norm()))      # n x H_out
        a_eff = float(shift.std()); s_feat = (shift @ U[:, 0]); s_feat = (s_feat - s_feat.mean())/s_feat.std()   # dominant feature over x, unit variance
        dists.append(degree_dist(a_eff, sig[l])); feats.append(s_feat.cpu().numpy())
        print(f"   layer {l}: a_eff(per-unit shift std)={a_eff:.2f} sig0={sig[l]:.2f} share={share[l]:.2f}")
    eigs = []
    for l in range(depth):
        p = dists[l]
        for mm in range(l+1, depth): p = convolve(p, dists[mm])
        eigs += [trQ*share[l]*p[m] for m in range(1, MDEG)]        # degree 0 is a constant feature: no new direction
    eigs = np.array(eigs); eigs = eigs*trQ/eigs.sum()
    specQ = np.zeros(600); specQ[:len(eigs)] = np.sort(eigs)[::-1][:600]
    specQ_sorted = np.sort(specQ)[::-1]
    outl = []
    for _ in range(3):
        R, _ = np.linalg.qr(rng.normal(size=(600, 600))); outl.append(np.sort(np.linalg.eigvalsh(np.diag(w0) + R @ np.diag(specQ_sorted) @ R.T))[::-1])
    wp = np.median(outl, 0)
    print(f"{tag}: trQ={trQ:.1f}; gains at k=50,100,200,400:")
    print(f"   measured K(T)        : " + ' '.join(f'{wT[k]/w0[k]:10.1f}' for k in (49, 99, 199, 399)))
    print(f"   statistical spec(Q)  : " + ' '.join(f'{wp[k]/w0[k]:10.1f}' for k in (49, 99, 199, 399)))
    print(f"   spec(Q) pred/meas at ranks 10,50,100,200,400: " + ' '.join(f'{specQ_sorted[k]:.2e}/{wQ[k]:.2e}' for k in (9, 49, 99, 199, 399)))
