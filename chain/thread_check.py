import sys, time, numpy as np
sys.path.insert(0, sys.argv[1]); import l63test as L
X, Y = L.make_data(200, 0.5); n = 200
for depth, H in [(4, 96), (4, 256)]:
    net = L.MLP(3, H, depth, 3, seed=0); theta = net.pack(); eta = 1e-2
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
    t0 = time.time(); J, out = net.jacobian(X); K = L.kernel(J, n); tj = time.time()-t0
    print(f'  threads={sys.argv[2]} depth={depth} H={H}: step {ts*1e3:.2f}ms  jac+kernel {tj:.2f}s', flush=True)
