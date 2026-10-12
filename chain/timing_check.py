import sys, time, numpy as np
sys.path.insert(0, sys.argv[1])
import l63test as L
import scipy, matplotlib
print('numpy', np.__version__, 'scipy', scipy.__version__, 'mpl', matplotlib.__version__, flush=True)
X, Y = L.make_data(200, 0.5)
for depth, H in [(4, 96), (4, 256)]:
    net = L.MLP(3, H, depth, 3, seed=0); theta = net.pack(); n = 200
    t0 = time.time(); J, out = net.jacobian(X); tj = time.time()-t0
    t0 = time.time(); K = L.kernel(J, n); tk = time.time()-t0
    t0 = time.time(); lam1 = np.linalg.eigvalsh(K)[-1]; te = time.time()-t0
    E = out.reshape(-1) - Y.reshape(-1)
    gE = J.T@E/n
    t0 = time.time(); Pu = L.along_dir_second(net, X, theta, gE/np.linalg.norm(gE)); s = np.linalg.norm(Pu, 2); th = time.time()-t0
    # GD steps
    eta = 0.4/lam1; Yv = Y.reshape(-1)
    t0 = time.time()
    for _ in range(300):
        out, A, Z = net.forward(X); delta = (out - Y)/n
        gW = [None]*len(net.W); gb = [None]*len(net.b)
        gW[-1] = delta.T@A[-1]; gb[-1] = delta.sum(0); dl = delta@net.W[-1]
        for l in range(net.L-1, -1, -1):
            dl = dl*(1-A[l+1]**2); gW[l] = dl.T@A[l]; gb[l] = dl.sum(0)
            if l > 0: dl = dl@net.W[l]
        g = np.concatenate([q.ravel() for q in gW+gb]); theta -= eta*g; net.unpack(theta)
    ts = (time.time()-t0)/300
    print(f'depth={depth} H={H} d={theta.size} lam1_0={lam1:.2f} eta={eta:.2e}: jac {tj:.2f}s kernel {tk:.2f}s eig {te:.3f}s H2(2 jac+svd) {th:.2f}s step {ts*1e3:.2f}ms', flush=True)
