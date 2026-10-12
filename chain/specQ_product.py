"""spec(Q) as a product spectrum. Per layer l: the chain dW_l (model) has 16 singular directions with per-unit shift
amplitudes a_j (j=1..16); each variable j has a Hermite energy spectrum e_j(m) (incl. degree 0). Q's directions within
the layer are the multi-indices (m_1..m_16) != 0 with energy prod_j e_j(m_j); the backprop product over downstream layers
multiplies in their spectra. Top-K products by best-first search; remainder mass lumped into the K-th value. Mass per layer:
trQ * share_l. Free convolution with spec(K0). Shape test on the two references with measured trQ."""
import numpy as np, sys, os, torch, heapq
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hermite_saturation import coeffs
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
import reduced_model2 as M
torch.set_default_dtype(torch.float64)
RUNS = L63_ROOT + '/runs'; MDEG = 16; KTOP = 600
def espec(a, sig):
    c = coeffs(min(a, 20.0), sig, M=MDEG)**2; return c/c.sum()           # includes degree 0
def top_products(spectra, K):
    """spectra: list of 1-D arrays (each sorted desc with index map). Return top-K products over all index tuples and
    the total remaining mass. Best-first over the lattice with sorted factors."""
    S = [np.sort(s)[::-1] for s in spectra]; idx0 = tuple(0 for _ in S); val0 = float(np.prod([s[0] for s in S]))
    heap = [(-val0, idx0)]; seen = {idx0}; out = []
    while heap and len(out) < K:
        v, idx = heapq.heappop(heap); out.append(-v)
        for d in range(len(S)):
            if idx[d]+1 < len(S[d]):
                nidx = idx[:d] + (idx[d]+1,) + idx[d+1:]
                if nidx not in seen: seen.add(nidx); heapq.heappush(heap, (-(-v)*S[d][idx[d]+1]/S[d][idx[d]], nidx))
    total = float(np.prod([s.sum() for s in S])); return np.array(out), total - sum(out)
rng = np.random.default_rng(0)
for tag, Delta, depth in [('H96_D0.5_depth2_s0', 0.5, 2), ('H96_D1.0_depth3_s0', 1.0, 3)]:
    C = np.load(f'{RUNS}/sweepC/late_{tag}.npy', allow_pickle=True).item(); L2 = np.load(f'{RUNS}/layers2/lay_{tag}.npy', allow_pickle=True).item()
    K0 = C['snap_K'][0].astype(np.float64); KT = C['snap_K'][-1].astype(np.float64); Q = C['snap_Q'][-1].astype(np.float64)
    w0 = np.sort(np.linalg.eigvalsh(K0))[::-1]; wT = np.sort(np.linalg.eigvalsh(KT))[::-1]; wQ = np.sort(np.linalg.eigvalsh(Q))[::-1]; trQ = np.trace(Q)
    A = [np.linalg.norm(sv[:16]) for sv in L2['snap_svdW'][-1]]; H = 96; n = 200
    Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV); net = MLP(3, H, depth, 3, seed=0); out, Acts = net.forward(X)
    J0, _ = net.jacobian(X); w0t, Vt = torch.linalg.eigh(kernel(J0, n)); Vt = Vt.flip(1); sizesW = [W.numel() for W in net.W]; offs = np.cumsum([0]+sizesW)
    z = np.load(f'{RUNS}/initq/q_{tag}.npz'); lam0 = z['lam0']; c = z['c0'].copy(); q = z['q']; kk = np.arange(1, 601); fr = c[100:].sum()/c.sum(); B = M.B
    Jk = np.zeros(600); ts = np.concatenate([[0.], np.geomspace(1e-3/lam0[0], 3000, 1500)]); Am = np.full(depth+1, 1e-3); hid = list(range(1, depth)) if depth > 1 else [1]
    for i in range(1, len(ts)):
        dt = ts[i]-ts[i-1]; P = max(np.prod(Am[hid]), 1e-6); lk = np.log(kk/100.)
        lift = 1 + np.exp(B[0]+B[1]*np.log(P)+B[2]*lk+B[3]*(depth-2)+B[4]*np.log(fr)+B[5]*lk*np.log(fr)+B[6]*lk*np.log(P)); lam = lam0*lift
        Jk += np.sqrt(np.maximum(c, 0))*np.sqrt(lift)*dt; c = c*np.exp(-2*lam*dt); contrib = q*Jk[None, :]**2; Am = np.sqrt(np.sort(contrib, axis=1)[:, -16:].sum(1))+1e-6
    G = J0.T @ Vt/np.sqrt(n); Jt = torch.tensor(Jk, device=DEV)
    sig = [float((Acts[l] @ net.W[l].T + net.b[l]).std()) for l in range(depth)]
    m_l = np.load(f'{RUNS}/initm/m_{tag}.npz')['m']; share = np.array([A[l]**2*m_l[l] for l in range(depth)]); share /= share.sum()
    layer_specs = []
    for l in range(depth):
        dWl = -(G[offs[l]:offs[l+1], :] @ Jt).reshape(net.W[l].shape); U, S, Vh = torch.linalg.svd(dWl, full_matrices=False)
        r16 = min(16, S.numel()); scale = A[l]/float(S[:r16].norm())
        a_j = [float(((Acts[l] @ Vh[j])*S[j]*scale).std())*float(U[:, j].abs().mean()) for j in range(r16)]   # per-unit: times |U_ij| ~ 1/sqrt(H)          # per-variable shift amplitude over the data
        layer_specs.append([espec(a, sig[l]) for a in a_j])
        print(f"   layer {l}: a_j (top 5) = {np.round(a_j[:5], 2)}, sig0={sig[l]:.2f}, share={share[l]:.2f}")
    eig_all = []
    for l in range(depth):
        specs = [s for ll in range(l, depth) for s in layer_specs[ll]]          # own layer's tanh'' ~ same spectra, times downstream tanh' factors
        vals, rest = top_products(specs, KTOP)
        vals = vals[1:]                                                        # drop the all-degree-0 constant direction
        vals = vals/ (vals.sum()+rest) * trQ*share[l]; eig_all += list(vals)
    eig_all = np.sort(np.array(eig_all))[::-1][:600]; specQ = np.zeros(600); specQ[:len(eig_all)] = eig_all
    outl = []
    for _ in range(3):
        R, _ = np.linalg.qr(rng.normal(size=(600, 600))); outl.append(np.sort(np.linalg.eigvalsh(np.diag(w0) + R @ np.diag(specQ) @ R.T))[::-1])
    wp = np.median(outl, 0)
    print(f"{tag}: trQ={trQ:.1f}; gains at k=50,100,200,400:")
    print(f"   measured K(T)        : " + ' '.join(f'{wT[k]/w0[k]:10.1f}' for k in (49, 99, 199, 399)))
    print(f"   product-spectrum Q   : " + ' '.join(f'{wp[k]/w0[k]:10.1f}' for k in (49, 99, 199, 399)))
    print(f"   spec(Q) pred/meas at ranks 1,10,50,100,200,400: " + ' '.join(f'{specQ[k]:.2e}/{wQ[k]:.2e}' for k in (0, 9, 49, 99, 199, 399)))
