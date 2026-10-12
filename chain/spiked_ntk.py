"""Spiked NTK recursion (unit-averaged, Gaussian over units): kernel K(t) as a function of the integrated residual u.
Layers l=0..L-1 (weights W_l: fan_in -> fan_out), activations a_0 = x, a_{l+1} = tanh(z_l), z_l = W_l a_l + b_l.
Initialisation covariances on the data (n points), measured from the init network (finite width, exact):
  A_l(x,x') = a_l(x).a_l(x')/fan_in  (activation kernel feeding layer l),   D_l(x,x') = delta_l(x).delta_l(x') per output
  Zc_l(x,x') = z_l(x).z_l(x')/fan_out (preactivation covariance across units).
Shift field s_l(x) = dW_l a_l(x), dW_l = -sum_i delta_l(x_i) a_l(x_i)^T u_i  ->  across units Gaussian with covariance
  S_l(x,x') = sum_ii' A_l(x_i,x) A_l(x_i',x') Dsum_l(x_i,x_i') u_i u_i'  (Dsum over output components; scaled by fan_in).
Shifted activation covariance: E_j[tanh(z_j(x)+s_j(x)) tanh(z_j(x')+s_j(x'))] with (z,s) jointly Gaussian across units:
  z ~ N(0, Zc), s ~ N(0, S), independent (bulk weights independent of the spike field at leading order) -> total covariance Zc+S.
  => Sigma_{l+1} = E tanh tanh under covariance Zc_l + S_l ;  Sigmadot_l = E tanh' tanh' under the same.
Backprop kernel D_l(t) = D_{l+1}(t) (x) Sigmadot_l(t) via the chain; kernel K = sum_l A_l(t) * D_l(t) per output (3 outputs).
Gaussian expectations by Gauss-Hermite on the 2D Gaussian with covariance [[c11,c12],[c12,c22]] (vectorised).
This script: load the init network, build the measured init covariances, take u from a measured trajectory, compute K(T).
usage: python spiked_ntk.py late_run.npy H DEPTH DELTA"""
import numpy as np, sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
torch.set_default_dtype(torch.float64)
fn, H, depth, Delta = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
n = 200; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV); Yv = Y.reshape(-1); ny = 3
net = MLP(3, H, depth, 3, seed=0); L = len(net.W)
out, Acts = net.forward(X)                         # Acts[l] = input activations of layer l (n x fan_in)
Z = [Acts[l] @ net.W[l].T + net.b[l] for l in range(L)]   # preactivations z_l (n x fan_out)
# backprop vectors delta_l(x) per output c: n x fan_out x ny
deltas = [torch.zeros(n, net.W[l].shape[0], ny, device=DEV) for l in range(L)]
for c in range(ny):
    dl = torch.zeros(n, ny, device=DEV); dl[:, c] = 1.; deltas[L-1][:, :, c] = dl
    dl = dl @ net.W[-1]
    for l in range(L-2, -1, -1):
        dl = dl*(1-Acts[l+1]**2); deltas[l][:, :, c] = dl
        if l > 0: dl = dl @ net.W[l]
Akern = [Acts[l] @ Acts[l].T for l in range(L)]                             # n x n (unnormalised a.a')
Dkern = [torch.einsum('ipc,jpc->ijc', deltas[l], deltas[l]) for l in range(L)]   # n x n x ny
Zc = [Z[l] @ Z[l].T / Z[l].shape[1] for l in range(L)]                      # per-unit preactivation covariance
# check: init kernel from these = sum_l (A_l (x) D_l)/n  (per output block)
def assemble(Ak, Dk):
    Kb = torch.zeros(n*ny, n*ny, device=DEV)
    for c in range(ny):
        for c2 in range(ny):
            pass
    # outputs are decoupled at init only through the last layer; general: K[(i,c),(j,c')] = sum_l A_l(i,j) delta_l(i,c).delta_l(j,c')
    return Kb
# simpler exact assembly with full cross-output structure:
def K_from(Ak_list, delta_list):
    Kf = torch.zeros(n, ny, n, ny, device=DEV)
    for l in range(L):
        Kf += torch.einsum('ij,ipc,jpd->icjd', Ak_list[l], delta_list[l], delta_list[l])
    return Kf.reshape(n*ny, n*ny)/n
J, _ = net.jacobian(X); K0 = kernel(J, n); K0r = K_from(Akern, deltas)
print('init kernel reconstruction error (weights+biases vs activations/deltas, biases omitted):', float((K0-K0r).norm()/K0.norm()))
# ---- integrated residual u from the measured run
r = np.load(fn, allow_pickle=True).item(); t = r['t']; Ev = r['Evec']                     # Evec: logged E vectors (n*ny) at log times
u = np.zeros(n*ny)
for i in range(1, len(t)): u += 0.5*(Ev[i]+Ev[i-1])*(t[i]-t[i-1])
u = torch.tensor(u/n, device=DEV).reshape(n, ny)
# ---- coherent chain shift fields, exact per unit (finite width): s_l(x) = dW_l a_l(x), dW_l = -sum_i,c delta_l(x_i,c) a_l(x_i)^T u_ic
dWs = [-torch.einsum('ipc,ic,iq->pq', deltas[l], u, Acts[l]) for l in range(L)]
print('chain dW Frobenius norms:', [round(float(x.norm()), 2) for x in dWs])
# forward with the chain applied (this is the unit-level ground truth of the statistical recursion)
W1 = [net.W[l] + dWs[l] for l in range(L)]
a = X; Acts1 = [a]
for l in range(L-1): a = torch.tanh(a @ W1[l].T + net.b[l]); Acts1.append(a)
out1 = Acts1[-1] @ W1[-1].T + net.b[-1]
deltas1 = [torch.zeros_like(d) for d in deltas]
for c in range(ny):
    dl = torch.zeros(n, ny, device=DEV); dl[:, c] = 1.; deltas1[L-1][:, :, c] = dl; dl = dl @ W1[-1]
    for l in range(L-2, -1, -1):
        dl = dl*(1-Acts1[l+1]**2); deltas1[l][:, :, c] = dl
        if l > 0: dl = dl @ W1[l]
K1 = K_from([Acts1[l] @ Acts1[l].T for l in range(L)], deltas1)
w0 = torch.linalg.eigvalsh(K0).flip(0); w1 = torch.linalg.eigvalsh(K1).flip(0)
print('lift at k=50,100,200,400 from the exact chain at measured u(T):', [round(float(w1[k]/w0[k]), 1) for k in (49, 99, 199, 399)])
print('measured lift at T:', [round(float(r['lam'][-1][::-1][k]/r['lam'][0][::-1][k]), 1) for k in (49, 99, 199, 399)])
# ---- statistical (unit-averaged Gaussian) recursion for the same u
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(24); gx = torch.tensor(gh_x, device=DEV); gw = torch.tensor(gh_w/np.sqrt(2*np.pi), device=DEV)
def gauss_exp2(C11, C12, C22, f):
    """E[f(z1) f(z2)] for (z1,z2) ~ N(0, [[C11,C12],[C12,C22]]), elementwise over n x n matrices, Gauss-Hermite 24x24."""
    s1 = torch.sqrt(torch.clamp(C11, min=1e-12)); s2 = torch.sqrt(torch.clamp(C22, min=1e-12)); rho = torch.clamp(C12/(s1*s2), -0.999999, 0.999999)
    out = torch.zeros_like(C11)
    for a_, wa in zip(gx, gw):
        z1 = s1*a_
        for b_, wb in zip(gx, gw):
            z2 = s2*(rho*a_ + torch.sqrt(1-rho**2)*b_)
            out += wa*wb*f(z1)*f(z2)
    return out
fan_in = [3] + [H]*(L-1)
Sig = [None]*(L+1); Sig[0] = Akern[0]/fan_in[0]  # input layer activation covariance per unit... inputs are not Gaussian units; use measured A_0 directly
Sdot = [None]*L
Cov_z = []
for l in range(L):
    # shift-field covariance across units (per unit): S_l(x,x') = sum_ii' A_l(x_i,x)/fan_in A_l(x_i',x')/fan_in * Dsum(x_i,x_i') u u / fan_out... derive from dW directly:
    # s_j(x) = sum_q dW_jq a_q(x);  across units j: Cov(s(x),s(x')) = (1/fan_out) sum_j s_j(x) s_j(x')  (unit-averaged; exact second moment)
    s = Acts[l] @ dWs[l].T                                  # n x fan_out shift field (exact)
    S = s @ s.T / dWs[l].shape[0]
    Cz = Zc[l] + S                                          # total per-unit preactivation covariance (bulk + spike, assumed independent)
    Cov_z.append(Cz)
    if l < L-1:
        Sig[l+1] = gauss_exp2(torch.diag(Cz)[:, None].expand(n, n), Cz, torch.diag(Cz)[None, :].expand(n, n), torch.tanh)
        Sdot[l] = gauss_exp2(torch.diag(Cz)[:, None].expand(n, n), Cz, torch.diag(Cz)[None, :].expand(n, n), lambda z: 1-torch.tanh(z)**2)
# assemble statistical kernel: K^(l) = A_l * D_l with A_l = fan_in * Sig[l] (unit-summed), D_l = prod_{l'>l} Sdot_{l'} (times W variance factors)
# use the measured D at init rescaled by the ratio of shifted/unshifted Sdot products (keeps finite-width structure of the backprop kernel)
Sdot0 = [gauss_exp2(torch.diag(Zc[l])[:, None].expand(n, n), Zc[l], torch.diag(Zc[l])[None, :].expand(n, n), lambda z: 1-torch.tanh(z)**2) if l < L-1 else None for l in range(L)]
Sig0 = [None] + [gauss_exp2(torch.diag(Zc[l])[:, None].expand(n, n), Zc[l], torch.diag(Zc[l])[None, :].expand(n, n), torch.tanh) for l in range(L-1)]
Kst = torch.zeros(n, ny, n, ny, device=DEV)
for l in range(L):
    Aratio = torch.ones(n, n, device=DEV) if l == 0 else Sig[l]/torch.clamp(Sig0[l], min=1e-12)
    Dratio = torch.ones(n, n, device=DEV)
    for l2 in range(l, L-1): Dratio = Dratio*(Sdot[l2]/torch.clamp(Sdot0[l2], min=1e-12))
    Kst += torch.einsum('ij,ipc,jpd->icjd', Akern[l]*Aratio*Dratio, deltas[l], deltas[l])
Kst = Kst.reshape(n*ny, n*ny)/n; ws = torch.linalg.eigvalsh(Kst).flip(0)
print('lift at k=50,100,200,400 from the Gaussian (unit-averaged) recursion at the same u(T):', [round(float(ws[k]/w0[k]), 1) for k in (49, 99, 199, 399)])
