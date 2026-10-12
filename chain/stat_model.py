"""Statistical model: iterated spiked-NTK (Gaussian over units) with coherent shifts from the integrated residual.
Tables per layer l (weights fan_in->fan_out), unit-SUMMED unless noted:
  A_l(x,x') = a_l(x).a_l(x') + 1 (bias feature)   [input activations of layer l]
  Zc_l(x,x') per-unit preactivation covariance (bulk) = (A_l - 1)/fan_in_l * sw2 + sb2, with sw2 = 1/fan_in (init), sb2 = 0
  Dsum_l(x,c;x',c') = delta_l(x,c).delta_l(x',c')   (backprop vectors, unit-summed)
Kernel: K[(x,c),(x',c')] = sum_l A_l(x,x') Dsum_l(x,c;x',c') / n.
Coherent shifts from u (integrated residual, n x ny):
  forward: s_l(x) = dW_l a_l(x) + db_l, dW_l = -sum_{i,c} delta_l(x_i,c) a_l(x_i)^T u_ic  ->  per-unit covariance
           S_l = A_l^T (Ucov_l) A_l / fan_out_l,  Ucov_l(i,i') = sum_cc' u_ic Dsum_l(i,c;i',c') u_i'c'
  backward: Dsum_l gets  Sdot_l * [ sum_ii' (A_{l+1}(i,i')-1)/1 * G_c(i,x) G_c'(i',x') ],  G_c(i,x) = sum_d Dsum_{l+1}(i,d;x,c) u_id
Gaussian expectations with covariance Zc_l + S_l give Sigma_{l+1} (per unit) and Sdot_l; measured init tables are the
base and the recursion supplies the change (table(t) = table_meas(0) + rec(t) - rec(0)).
Time loop: K -> E decays by kernel regression; u += int E dt / n.  usage: python stat_model.py H DEPTH SEED DELTA OUTSCALE T NSTEP OUT.npz"""
import numpy as np, sys, os, torch, time
os.environ['L63_OUTSCALE'] = sys.argv[5]
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
torch.set_default_dtype(torch.float64)
H, depth, seed, Delta, sc, T, NSTEP, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]), sys.argv[8]
n = 200; ny = 3; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV); Yv = Y.reshape(-1)
net = MLP(3, H, depth, 3, seed=seed)
if sc != 1.0: net.forward(X)
L = len(net.W); fan_in = [W.shape[1] for W in net.W]; fan_out = [W.shape[0] for W in net.W]
# ---- measured init tables
out0, Acts = net.forward(X); Zs = [Acts[l] @ net.W[l].T + net.b[l] for l in range(L)]
deltas = [torch.zeros(n, fan_out[l], ny, device=DEV) for l in range(L)]
for c in range(ny):
    dl = torch.zeros(n, ny, device=DEV); dl[:, c] = 1.; deltas[L-1][:, :, c] = dl; dl = dl @ net.W[-1]
    for l in range(L-2, -1, -1):
        dl = dl*(1-Acts[l+1]**2); deltas[l][:, :, c] = dl
        if l > 0: dl = dl @ net.W[l]
A0 = [Acts[l] @ Acts[l].T + 1. for l in range(L)]
D0 = [torch.einsum('ipc,jpd->icjd', deltas[l], deltas[l]) for l in range(L)]
Zc0 = [Zs[l] @ Zs[l].T / fan_out[l] for l in range(L)]
E = out0.reshape(-1) - Yv
def assemble(A, D):
    Kf = torch.zeros(n, ny, n, ny, device=DEV)
    for l in range(L): Kf += A[l][:, None, :, None]*D[l]
    return Kf.reshape(n*ny, n*ny)/n
K0 = assemble(A0, D0); Kj = kernel(net.jacobian(X)[0], n); print('init kernel from tables vs Jacobian: rel err', float((K0-Kj).norm()/Kj.norm()), flush=True)
# ---- Gaussian expectations
gx, gw = np.polynomial.hermite_e.hermegauss(20); gx = torch.tensor(gx, device=DEV); gw = torch.tensor(gw/np.sqrt(2*np.pi), device=DEV)
def gexp(C, f):
    d = torch.sqrt(torch.clamp(torch.diag(C), min=1e-14)); s1 = d[:, None]; s2 = d[None, :]; rho = torch.clamp(C/(s1*s2), -0.999999, 0.999999); sq = torch.sqrt(1-rho**2)
    acc = torch.zeros_like(C)
    for a_, wa in zip(gx, gw):
        f1 = f(s1*a_)
        for b_, wb in zip(gx, gw):
            acc += (wa*wb)*f1*f(s2*(rho*a_ + sq*b_))
    return acc
tanh = torch.tanh; dtanh = lambda z: 1-torch.tanh(z)**2
# recursion at zero shift (reference) : per-unit Sigma_{l+1}^rec(0), Sdot_l^rec(0)
def recursion(Zc_list, Sfwd, Tbwd):
    """Given per-unit preactivation covariances Zc_l(t) (bulk, already with current A_l) and shift covariances, return
    per-unit Sigma_{l+1} (activation covariance) and Sdot_l, then unit-summed A_l(t) and Dsum_l(t) via the chain."""
    Sig = [None]*(L+1); Sd = [None]*L
    for l in range(L):
        Ctot = Zc_list[l] + Sfwd[l]
        Sig[l+1] = gexp(Ctot, tanh) if l < L-1 else None; Sd[l] = gexp(Ctot, dtanh) if l < L-1 else None
    return Sig, Sd
# zero-shift reference tables for the additive-correction scheme
zero = [torch.zeros(n, n, device=DEV) for _ in range(L)]
Zc_ref = [(A0[l]-1.)/fan_in[l] for l in range(L)]
Sig_ref, Sd_ref = recursion(Zc_ref, zero, None)
def Dsum_chain(Sd, Tb, A_cur):
    """unit-summed backprop tables from the top: Dsum_{L-1} = I_c ; Dsum_l = Sd_l * (Dsum_{l+1} + Tb_l)"""
    D = [None]*L; Dtop = torch.zeros(n, ny, n, ny, device=DEV)
    for c in range(ny): Dtop[:, c, :, c] = 1.
    D[L-1] = Dtop
    for l in range(L-2, -1, -1):
        D[l] = Sd[l][:, None, :, None]*(D[l+1] + (Tb[l] if Tb is not None else 0.))
    return D
D_ref = Dsum_chain(Sd_ref, None, A0)
def tables(u, A_prev):
    """u: n x ny integrated residual. Returns A(t), Dsum(t) lists (unit-summed) with measured base + predicted change."""
    # forward pass of covariances: layer by layer, A_l(t) known -> Zc_l(t) -> shift S_l -> Sigma_{l+1}(t)
    A = [None]*L; A[0] = A0[0]; Zc = [None]*L; Sf = [None]*L; Sig = [None]*(L+1); Sd = [None]*L
    # backprop tables needed for the forward shift covariance: use previous-step Dsum (lagged by one step)
    for l in range(L):
        Zc[l] = (A[l]-1.)/fan_in[l]                                    # per-unit bulk preactivation covariance (sw2 = 1/fan_in)
        Ucov = torch.einsum('ic,icjd,jd->ij', u, D_prev[l], u)         # n x n
        Sf[l] = A[l].T @ Ucov @ A[l] / fan_out[l]
        Ctot = Zc[l] + Sf[l]
        if l < L-1:
            Sig_t = gexp(Ctot, tanh); Sd[l] = gexp(Ctot, dtanh)
            A[l+1] = A0[l+1] + fan_out[l]*(Sig_t - Sig_ref[l+1])     # measured base + predicted change (unit-summed)
    # backward coherent term T_l and Dsum chain (predicted) ; measured base + change
    Tb = [None]*L
    for l in range(L-1):
        G = torch.einsum('idxc,id->ixc', D_prev[l+1], u)                # n(i) x n(x) x ny(c)
        Tb[l] = torch.einsum('ij,ixc,jyd->xcyd', A[l+1]-1., G, G)/fan_out[l]
    D_rec = Dsum_chain(Sd, Tb, A)
    D = [D0[l] + (D_rec[l] - D_ref[l]) for l in range(L)]
    return A, D
# ---- optional diagnostic of the shift-covariance formula against the exact chain
if os.environ.get('DIAG_U'):
    rr = np.load(os.environ['DIAG_U'], allow_pickle=True).item(); tt = rr['t']; Ev = rr['Evec']; uu = np.zeros(n*ny)
    for i in range(1, len(tt)): uu += 0.5*(Ev[i]+Ev[i-1])*(tt[i]-tt[i-1])
    uu = torch.tensor(uu/n, device=DEV).reshape(n, ny)
    for l in range(L):
        dW = -torch.einsum('ipc,ic,iq->pq', deltas[l], uu, Acts[l]); db = -torch.einsum('ipc,ic->p', deltas[l], uu)
        sx = Acts[l] @ dW.T + db; S_exact = sx @ sx.T / fan_out[l]
        Ucov = torch.einsum('ic,icjd,jd->ij', uu, D0[l], uu); S_form = A0[l].T @ Ucov @ A0[l] / fan_out[l]
        print(f"   layer {l}: exact shift var (mean diag) {float(torch.diag(S_exact).mean()):.3e}  formula {float(torch.diag(S_form).mean()):.3e}  ratio {float(torch.diag(S_form).mean()/torch.diag(S_exact).mean()):.2f};  bulk Zc diag {float(torch.diag(Zc0[l]).mean()):.3e}", flush=True)
    # exact tables of the shifted network at this u (scale the shift by DIAG_SCALE to probe larger shifts)
    scl = float(os.environ.get('DIAG_SCALE', '1'))
    Wd = [net.W[l] - scl*torch.einsum('ipc,ic,iq->pq', deltas[l], uu, Acts[l]) for l in range(L)]; bd = [net.b[l] - scl*torch.einsum('ipc,ic->p', deltas[l], uu) for l in range(L)]
    a = X; Ae = [a]
    for l in range(L-1): a = torch.tanh(a @ Wd[l].T + bd[l]); Ae.append(a)
    de = [torch.zeros(n, fan_out[l], ny, device=DEV) for l in range(L)]
    for c in range(ny):
        dl = torch.zeros(n, ny, device=DEV); dl[:, c] = 1.; de[L-1][:, :, c] = dl; dl = dl @ Wd[-1]
        for l in range(L-2, -1, -1):
            dl = dl*(1-Ae[l+1]**2); de[l][:, :, c] = dl
            if l > 0: dl = dl @ Wd[l]
    A_ex = [Ae[l] @ Ae[l].T + 1. for l in range(L)]; D_ex = [torch.einsum('ipc,jpd->icjd', de[l], de[l]) for l in range(L)]
    D_prev = D0; A_pr, D_pr = tables(scl*uu, A0)
    K_ex = assemble(A_ex, D_ex); K_pr = assemble(A_pr, D_pr); w0d = torch.linalg.eigvalsh(K0).flip(0); wex = torch.linalg.eigvalsh(K_ex).flip(0); wpr = torch.linalg.eigvalsh(K_pr).flip(0)
    print(f"   shift scale {scl}: lift at k=50,100,200,400  exact network: " + ' '.join(f'{float(wex[k]/w0d[k]):8.1f}' for k in (49, 99, 199, 399)) + "   Gaussian tables: " + ' '.join(f'{float(wpr[k]/w0d[k]):8.1f}' for k in (49, 99, 199, 399)), flush=True)
    for l in range(1, L):
        ra = (A_pr[l]-A0[l]).norm()/(A_ex[l]-A0[l]).norm(); rd = (D_pr[l-1]-D0[l-1]).norm()/(D_ex[l-1]-D0[l-1]).norm()
        print(f"      layer {l}: |dA pred|/|dA exact| = {float(ra):.2f};  layer {l-1}: |dD pred|/|dD exact| = {float(rd):.2f}", flush=True)
    # (a) recursion-only assembly: K_rec(t)/K_rec(0) lift (no measured base, positive by construction)
    Sig_t = [None]*(L+1); Sd_t = [None]*L; A_t = [A0[0]]
    D_prev = D0; Acur = A0[0]
    for l in range(L):
        Zc_t = (A_t[l]-1.)/fan_in[l]; Ucov = torch.einsum('ic,icjd,jd->ij', scl*uu, D0[l], scl*uu); Sf = A_t[l].T @ Ucov @ A_t[l]/fan_out[l]
        if l < L-1: Sig_t = gexp(Zc_t+Sf, tanh); Sd_t[l] = gexp(Zc_t+Sf, dtanh); A_t.append(fan_out[l]*Sig_t + 1.)
    A_r0 = [A0[0]]
    for l in range(L-1): A_r0.append(fan_out[l]*Sig_ref[l+1] + 1.)
    Tb = [None]*L
    for l in range(L-1):
        G = torch.einsum('idxc,id->ixc', D0[l+1], scl*uu); Tb[l] = torch.einsum('ij,ixc,jyd->xcyd', A_t[l+1]-1., G, G)/fan_out[l]
    D_rt = Dsum_chain(Sd_t, Tb, A_t); D_r0 = Dsum_chain(Sd_ref, None, A_r0)
    K_rt = assemble(A_t, D_rt); K_r0 = assemble(A_r0, D_r0); wrt = torch.linalg.eigvalsh(K_rt).flip(0); wr0 = torch.linalg.eigvalsh(K_r0).flip(0)
    print(f"   (a) recursion-only lift K_rec(t)/K_rec(0): " + ' '.join(f'{float(wrt[k]/wr0[k]):8.1f}' for k in (49, 99, 199, 399)) + f"   [recursion init tail vs measured: lam100 {float(wr0[99]):.2e} vs {float(w0d[99]):.2e}, lam400 {float(wr0[399]):.2e} vs {float(w0d[399]):.2e}]", flush=True)
    # (b) backward chain with EXACT tanh' covariance of the shifted network (per unit), Gaussian elsewhere
    Sd_ex = [((1-Ae[l+1]**2) @ (1-Ae[l+1]**2).T)/fan_out[l] for l in range(L-1)]
    D_bx = Dsum_chain(Sd_ex, Tb, A_t); D_b0 = Dsum_chain([((1-Acts[l+1]**2) @ (1-Acts[l+1]**2).T)/fan_out[l] for l in range(L-1)], None, A0)
    for l in range(L-1):
        print(f"      layer {l}: |dD| ratio with exact tanh' cov: {float((D_bx[l]-D_b0[l]).norm()/(D_ex[l]-D0[l]).norm()):.2f}  (Gaussian tanh': {float((D_pr[l]-D0[l]).norm()/(D_ex[l]-D0[l]).norm()):.2f});  Sd change ratio Gaussian/exact: {float((Sd_t[l]-Sd_ref[l]).norm()/(Sd_ex[l]-((1-Acts[l+1]**2) @ (1-Acts[l+1]**2).T)/fan_out[l]).norm()):.2f}", flush=True)
    # exact decomposition of the backward change: (i) shifted tanh' with INIT weights in the backward pass, (ii) full
    def deltas_of(Wb, Ab):
        dd = [torch.zeros(n, fan_out[l], ny, device=DEV) for l in range(L)]
        for c in range(ny):
            dl = torch.zeros(n, ny, device=DEV); dl[:, c] = 1.; dd[L-1][:, :, c] = dl; dl = dl @ Wb[-1]
            for l in range(L-2, -1, -1):
                dl = dl*(1-Ab[l+1]**2); dd[l][:, :, c] = dl
                if l > 0: dl = dl @ Wb[l]
        return dd
    d_i = deltas_of([net.W[l] for l in range(L)], Ae)            # shifted activations, init weights backward
    D_i = [torch.einsum('ipc,jpd->icjd', d_i[l], d_i[l]) for l in range(L)]
    for l in range(L-1):
        tot = (D_ex[l]-D0[l]).norm(); part_i = (D_i[l]-D0[l]).norm(); part_w = (D_ex[l]-D_i[l]).norm()
        print(f"      exact backward change layer {l}: |total| {float(tot):.3e};  tanh'-only part {float(part_i/tot):.2f} of total;  dW-in-backward part {float(part_w/tot):.2f} of total;  model T_l norm / exact dW part: {float(Tb[l].norm()/part_w):.2f}", flush=True)
    # positive assembly: measured base scaled by recursion lift ratio per rank, applied in the measured eigenbasis up to rank 200
    w0m, V0m = torch.linalg.eigh(K0); w0m = w0m.flip(0); V0m = V0m.flip(1); ratio = torch.clamp(wrt/torch.clamp(wr0, min=1e-300), min=1.0); ratio[200:] = ratio[199]
    K_pos = (V0m*(w0m*ratio)) @ V0m.T; wpos = torch.linalg.eigvalsh(K_pos).flip(0)
    print(f"   positive assembly lift at k=50,100,200,400: " + ' '.join(f'{float(wpos[k]/w0m[k]):8.1f}' for k in (49, 99, 199, 399)) + "   (exact: " + ' '.join(f'{float(wex[k]/w0d[k]):.1f}' for k in (49, 99, 199, 399)) + ")", flush=True)
    sys.exit(0)
# ---- time loop
D_prev = D0; u = torch.zeros(n, ny, device=DEV); t = 0.; ts = np.concatenate([[0.], np.geomspace(1e-3/float(torch.linalg.eigvalsh(K0)[-1]), T, NSTEP)])
w0 = torch.linalg.eigvalsh(K0).flip(0); hist = []; t0 = time.time(); A_cur, D_cur = A0, D0; K = K0
for i in range(1, len(ts)):
    dt = ts[i]-ts[i-1]
    w, V = torch.linalg.eigh(K); w = torch.clamp(w, min=0); a = V.T @ E
    dec = torch.exp(-w*dt); intE = V @ (a*torch.where(w > 1e-300, (1-dec)/torch.clamp(w, min=1e-300), torch.full_like(w, dt)))
    E = V @ (a*dec); u += intE.reshape(n, ny)/n
    A_cur, D_cur = tables(u, A_cur); D_prev = D_cur; K = assemble(A_cur, D_cur)
    if i % max(1, NSTEP//12) == 0 or i == len(ts)-1:
        wk = torch.linalg.eigvalsh(K).flip(0); hist.append((ts[i], float(torch.sqrt(E@E/n)), float(wk[99]/w0[99]), float(wk[199]/w0[199])))
        print(f"   t={ts[i]:9.3f} ||E||={hist[-1][1]:.4f} lift100={hist[-1][2]:8.1f} lift200={hist[-1][3]:9.1f} |u|={float(u.norm()):.3f} ({time.time()-t0:.0f}s)", flush=True)
np.savez(out, hist=np.array(hist), lamT=torch.linalg.eigvalsh(K).flip(0).cpu().numpy(), lam0=w0.cpu().numpy())
