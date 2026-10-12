"""(C) Shape test: spec(Q) by rank = trQ * (composed degree distribution of the second-order features, mapped to rank
through the universal rank-roughness curve); spec(K) = spec(K0) (+) spec(Q) by random rotation. Compare gains with
measured on the two reference runs, using the MEASURED tr Q (sweep C at T) and measured amplitudes (layers2 at T)."""
import numpy as np, sys, os, torch
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hermite_saturation import coeffs
from l63test import make_data
from l63test_torch import MLP, DEV
torch.set_default_dtype(torch.float64)
RUNS = L63_ROOT + '/runs'
rho_of_rank = np.interp(np.arange(1, 601), [1, 5, 10, 20, 50, 100, 200, 300, 400, 600], [0.055, 0.06, 0.07, 0.15, 0.29, 0.56, 0.89, 1.12, 1.20, 1.20])  # measured universal curve (section 7)
def degree_dist(a, sig, M=40):
    c = coeffs(min(a, 20.0), sig, M=M)**2; return c/c.sum()
def convolve(p, q):
    r = np.convolve(p, q); return r[:len(p)]/r[:len(p)].sum()
rng = np.random.default_rng(0); SHIFT = os.environ.get('SHIFT', '0') == '1'
for tag, Delta, depth in [('H96_D0.5_depth2_s0', 0.5, 2), ('H96_D1.0_depth3_s0', 1.0, 3)]:
    C = np.load(f'{RUNS}/sweepC/late_{tag}.npy', allow_pickle=True).item(); L2 = np.load(f'{RUNS}/layers2/lay_{tag}.npy', allow_pickle=True).item()
    K0 = C['snap_K'][0].astype(np.float64); KT = C['snap_K'][-1].astype(np.float64); Q = C['snap_Q'][-1].astype(np.float64)
    w0 = np.sort(np.linalg.eigvalsh(K0))[::-1]; wT = np.sort(np.linalg.eigvalsh(KT))[::-1]; wQ = np.sort(np.linalg.eigvalsh(Q))[::-1]; trQ = np.trace(Q)
    A = [np.linalg.norm(sv[:16]) for sv in L2['snap_svdW'][-1]]; H = 96
    Xn, Yn = make_data(200, Delta); X = torch.tensor(Xn, device=DEV); net = MLP(3, H, depth, 3, seed=0); _, Acts = net.forward(X)
    sig = [float((Acts[l] @ net.W[l].T + net.b[l]).std()) for l in range(depth)]; rms_in = [float(Acts[l].norm()/np.sqrt(Acts[l].numel())) for l in range(depth)]
    a_eff = [A[l]*rms_in[l]*np.sqrt(Acts[l].shape[1])/np.sqrt(H) for l in range(depth)]
    dists = [degree_dist(a_eff[l], sig[l]) for l in range(depth)]
    # second-order features: weight-space mass per layer ~ A_l^2 m_l (use measured per-layer share of Q as proxy here: equal weights)
    # composed degree distribution of the backprop factor of layer l, then one more tanh'' factor ~ shift by +1 degree
    m_l = np.load(f'{RUNS}/initm/m_{tag}.npz')['m']; share = np.array([A[l]**2*m_l[l] for l in range(depth)]); share = share/share.sum()
    print(f"   per-layer mass share (A_l^2 m_l): {np.round(share, 3)}")
    p_tot = np.zeros(40)
    for l in range(depth):
        p = dists[l]
        for mm in range(l+1, depth): p = convolve(p, dists[mm])
        if SHIFT: p = np.roll(p, 1); p[0] = 0
        p_tot += share[l]*p/p.sum()
    p_tot /= p_tot.sum()
    # design-specific roughness per Hermite degree: the spike feature is the chain sum_k J_k G_k; its roughness is the
    # J_k^2 q_k-weighted roughness of the driving modes (init-only: chain weights from the reduced model, rank-roughness curve)
    import reduced_model2 as M
    z = np.load(f'{RUNS}/initq/q_{tag}.npz'); lam0 = z['lam0']; c = z['c0'].copy(); q = z['q']; kk = np.arange(1, 601); fr = c[100:].sum()/c.sum(); B = M.B
    Jn = np.zeros(600); Jk = np.zeros(600); ts = np.concatenate([[0.], np.geomspace(1e-3/lam0[0], 3000, 1500)]); Am = np.full(depth+1, 1e-3); hid = list(range(1, depth)) if depth > 1 else [1]
    for i in range(1, len(ts)):
        dt = ts[i]-ts[i-1]; P = max(np.prod(Am[hid]), 1e-6); lk = np.log(kk/100.)
        lift = 1 + np.exp(B[0]+B[1]*np.log(P)+B[2]*lk+B[3]*(depth-2)+B[4]*np.log(fr)+B[5]*lk*np.log(fr)+B[6]*lk*np.log(P)); lam = lam0*lift
        Jn += np.sqrt(np.maximum(c, 0))*dt; Jk += np.sqrt(np.maximum(c, 0))*np.sqrt(lift)*dt; c = c*np.exp(-2*lam*dt); contrib = q*Jk[None, :]**2; Am = np.sqrt(np.sort(contrib, axis=1)[:, -16:].sum(1))+1e-6
    wgt = (lam0*Jn**2); rho_f = float((wgt*rho_of_rank).sum()/wgt.sum())
    print(f"   chain roughness rho_f = {rho_f:.3f} (J_k^2-weighted roughness of the driving modes)")
    rho_deg = 0.025 + rho_f*np.arange(40)
    # distribute trQ over ranks: energy at degree m goes to ranks whose roughness is nearest rho_deg[m]
    specQ = np.zeros(600)
    for m in range(40):
        ks = np.where(np.abs(rho_of_rank - rho_deg[m]) <= 0.03)[0]
        if len(ks) == 0: ks = np.array([np.argmin(np.abs(rho_of_rank - rho_deg[m]))])
        specQ[ks] += trQ*p_tot[m]/len(ks)
    specQ = np.sort(specQ)[::-1]
    out = []
    for _ in range(3):
        R, _ = np.linalg.qr(rng.normal(size=(600, 600))); out.append(np.sort(np.linalg.eigvalsh(np.diag(w0) + R @ np.diag(specQ) @ R.T))[::-1])
    wp = np.median(out, 0)
    print(f"{tag}: a_eff={np.round(a_eff,2)} trQ={trQ:.1f}; gains at k=50,100,200,400:")
    print(f"   measured K(T)      : " + ' '.join(f'{wT[k]/w0[k]:10.1f}' for k in (49, 99, 199, 399)))
    print(f"   spec(K0)+measured Q: " + ' '.join(f'{np.median([np.sort(np.linalg.eigvalsh(np.diag(w0) + R @ np.diag(wQ) @ R.T))[::-1][k] for R in [np.linalg.qr(rng.normal(size=(600,600)))[0] for _ in range(3)]])/w0[k]:10.1f}' for k in (49, 99, 199, 399)))
    print(f"   statistical spec(Q): " + ' '.join(f'{wp[k]/w0[k]:10.1f}' for k in (49, 99, 199, 399)))
    print(f"   predicted spec(Q) vs measured spec(Q) at ranks 10,50,100,200,400: " + ' '.join(f'{specQ[k]:.2e}/{wQ[k]:.2e}' for k in (9, 49, 99, 199, 399)))
