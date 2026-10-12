"""Checkpoint 1: virtual Gaussian unit ensemble with coherent maps, at a fixed shift, vs the exact shifted network.
Per layer l (output units of W_l): unit = (z_q(.), m_q(.)) over the n data points, independent Gaussian processes:
  z ~ GP(0, Zc_l) (per-unit preactivation covariance, measured at init),
  m ~ GP(0, M_l) with M_l(x,c;x',c') = D_{l+1}(x,c;x',c')/fan_out_l (incoming backward signal; last layer: m = e_c)
  delta_q = sigma'(z_q) * m_q ;  a_{l+1,q} = sigma(z_q).
Coherent maps with integrated residual u (n x ny), Grams G_l = A_l^T A_l (+1 for bias) and D_{l+1}:
  z_q(x) <- z_q(x) - sum_{i,c} u_ic delta_q^c(x_i) G_l(x_i, x)                      (forward shift, exact per unit)
  m_q^c(x) <- m_q^c(x) - sum_{i,d} sigma(z_q(x_i)) D_{l+1}(x_i,d; x,c) u_id              (backward shift, exact per unit)
Tables: A_{l+1} = fan_out_l * E_q[a a^T] + 1,  Dsum_l = fan_out_l * E_q[delta delta^T].
usage: python virtual_ensemble.py late_run.npy H DEPTH DELTA SCALE NVIRT"""
import numpy as np, sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
torch.set_default_dtype(torch.float64)
fn, H, depth, Delta, scl, NV = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
n = 200; ny = 3; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV)
net = MLP(3, H, depth, 3, seed=0); L = len(net.W); fan_out = [W.shape[0] for W in net.W]; fan_in = [W.shape[1] for W in net.W]
out0, Acts = net.forward(X); Zs = [Acts[l] @ net.W[l].T + net.b[l] for l in range(L)]
deltas = [torch.zeros(n, fan_out[l], ny, device=DEV) for l in range(L)]
for c in range(ny):
    dl = torch.zeros(n, ny, device=DEV); dl[:, c] = 1.; deltas[L-1][:, :, c] = dl; dl = dl @ net.W[-1]
    for l in range(L-2, -1, -1):
        dl = dl*(1-Acts[l+1]**2); deltas[l][:, :, c] = dl
        if l > 0: dl = dl @ net.W[l]
A0 = [Acts[l] @ Acts[l].T + 1. for l in range(L)]; D0 = [torch.einsum('ipc,jpd->icjd', deltas[l], deltas[l]) for l in range(L)]; Zc0 = [Zs[l] @ Zs[l].T/fan_out[l] for l in range(L)]
def assemble(A, D):
    Kf = torch.zeros(n, ny, n, ny, device=DEV)
    for l in range(L): Kf += A[l][:, None, :, None]*D[l]
    return Kf.reshape(n*ny, n*ny)/n
K0 = assemble(A0, D0); w0 = torch.linalg.eigvalsh(K0).flip(0)
# integrated residual from the measured run, scaled
r = np.load(fn, allow_pickle=True).item(); t = r['t']; Ev = r['Evec']; uu = np.zeros(n*ny)
for i in range(1, len(t)): uu += 0.5*(Ev[i]+Ev[i-1])*(t[i]-t[i-1])
u = scl*torch.tensor(uu/n, device=DEV).reshape(n, ny)
# ---- exact shifted network (chain from init clouds), as reference
Wd = [net.W[l] - torch.einsum('ipc,ic,iq->pq', deltas[l], u, Acts[l]) for l in range(L)]; bd = [net.b[l] - torch.einsum('ipc,ic->p', deltas[l], u) for l in range(L)]
a = X; Ae = [a]
for l in range(L-1): a = torch.tanh(a @ Wd[l].T + bd[l]); Ae.append(a)
de = [torch.zeros(n, fan_out[l], ny, device=DEV) for l in range(L)]
for c in range(ny):
    dl = torch.zeros(n, ny, device=DEV); dl[:, c] = 1.; de[L-1][:, :, c] = dl; dl = dl @ Wd[-1]
    for l in range(L-2, -1, -1):
        dl = dl*(1-Ae[l+1]**2); de[l][:, :, c] = dl
        if l > 0: dl = dl @ Wd[l]
A_ex = [Ae[l] @ Ae[l].T + 1. for l in range(L)]; D_ex = [torch.einsum('ipc,jpd->icjd', de[l], de[l]) for l in range(L)]
wex = torch.linalg.eigvalsh(assemble(A_ex, D_ex)).flip(0)
# ---- virtual ensemble
gen = torch.Generator(device=DEV).manual_seed(0)
def gp_sample(C, N):
    w, V = torch.linalg.eigh(C); w = torch.clamp(w, min=0); return (torch.randn(N, C.shape[0], generator=gen, device=DEV)*torch.sqrt(w)) @ V.T
A_v = [A0[0]]; D_v = [None]*L; Zc_cur = Zc0
for l in range(L):
    # sample units of layer l: z over data (N x n), m over data x outputs (N x n*ny)
    z = gp_sample(Zc0[l], NV)
    if l < L-1:
        M = (D0[l+1]/fan_out[l]).reshape(n*ny, n*ny); m = gp_sample(M, NV).reshape(NV, n, ny)
    else:
        m = torch.zeros(NV, n, ny, device=DEV); m[:, :, :] = torch.eye(ny, device=DEV)[torch.arange(NV, device=DEV) % ny][:, None, :]*np.sqrt(ny)  # last layer: delta = e_c, one unit per output
    sp = 1-torch.tanh(z)**2 if l < L-1 else torch.ones_like(z)
    delta = sp[:, :, None]*m                                   # N x n x ny  (old backprop signal of each unit)
    # forward shift (exact per unit): s(x) = - sum_{i,c} u_ic delta^c(x_i) G_l(x_i, x)
    G = A0[l]                                                  # init Gram of layer-l inputs (+1 bias)
    s = -torch.einsum('qic,ic,ix->qx', delta, u, G)
    z_new = z + s
    # backward shift (exact per unit): m^c(x) <- m^c(x) - sum_{i,d} sigma(z(x_i)) D_{l+1}(x_i,d;x,c) u_id
    if l < L-1:
        g = torch.einsum('idxc,id->ixc', D0[l+1], u)            # n(i) x n(x) x ny(c)
        m_new = m - torch.einsum('qi,ixc->qxc', torch.tanh(z), g)
        sp_new = 1-torch.tanh(z_new)**2
        delta_new = sp_new[:, :, None]*m_new
        a_new = torch.tanh(z_new)
        A_v.append(fan_out[l]*(a_new.T @ a_new)/NV + 1.)
    else:
        delta_new = m                                            # output layer delta unchanged (= e_c)
    D_v[l] = fan_out[l]*torch.einsum('qxc,qyd->xcyd', delta_new, delta_new)/NV
# NOTE: D_v for the last layer uses NV virtual "output units" -> normalise to the true output layer (ny units)
D_v[L-1] = D0[L-1]
wv = torch.linalg.eigvalsh(assemble(A_v, D_v)).flip(0)
print(f"shift scale {scl}, {NV} virtual units/layer")
print("  lift at k=50,100,200,400   exact network: " + ' '.join(f'{float(wex[k]/w0[k]):8.1f}' for k in (49, 99, 199, 399)) + "   virtual ensemble: " + ' '.join(f'{float(wv[k]/w0[k]):8.1f}' for k in (49, 99, 199, 399)))
for l in range(1, L):
    print(f"  layer {l}: |dA virt|/|dA exact| = {float((A_v[l]-A0[l]).norm()/(A_ex[l]-A0[l]).norm()):.2f};  layer {l-1}: |dD virt|/|dD exact| = {float((D_v[l-1]-D0[l-1]).norm()/(D_ex[l-1]-D0[l-1]).norm()):.2f}")
# also: how well does the virtual ensemble reproduce the INIT tables (sanity of the GP sampling)?
A_i = [A0[0]]; 
for l in range(L-1):
    z = gp_sample(Zc0[l], NV); A_i.append(fan_out[l]*(torch.tanh(z).T @ torch.tanh(z))/NV + 1.)
# tail handling (b): measured init spectrum times the ensemble's lift ratio by rank (ratio from a matched virtual init)
A_i2 = [A0[0]]; D_i2 = [None]*L; gen2 = torch.Generator(device=DEV).manual_seed(0)
for l in range(L):
    w_, V_ = torch.linalg.eigh(Zc0[l]); z = (torch.randn(NV, n, generator=gen2, device=DEV)*torch.sqrt(torch.clamp(w_, min=0))) @ V_.T
    if l < L-1:
        M = (D0[l+1]/fan_out[l]).reshape(n*ny, n*ny); wm, Vm = torch.linalg.eigh(M); m = ((torch.randn(NV, n*ny, generator=gen2, device=DEV)*torch.sqrt(torch.clamp(wm, min=0))) @ Vm.T).reshape(NV, n, ny)
        sp = 1-torch.tanh(z)**2; A_i2.append(fan_out[l]*(torch.tanh(z).T @ torch.tanh(z))/NV + 1.); D_i2[l] = fan_out[l]*torch.einsum('qxc,qyd->xcyd', sp[:, :, None]*m, sp[:, :, None]*m)/NV
D_i2[L-1] = D0[L-1]
wv0 = torch.linalg.eigvalsh(assemble(A_i2, D_i2)).flip(0); ratio = torch.clamp(wv/torch.clamp(wv0, min=1e-300), min=1.0)
print("  tail handling: virtual lift ratio K_v(shift)/K_v(0) by rank: " + ' '.join(f'{float(ratio[k]):8.1f}' for k in (49, 99, 199, 399)) + "  (exact: " + ' '.join(f'{float(wex[k]/w0[k]):.1f}' for k in (49, 99, 199, 399)) + ")")
print("  sanity: |A_virt(0)-A0|/|A0| per layer:", [round(float((A_i[l]-A0[l]).norm()/A0[l].norm()), 3) for l in range(1, L)])
