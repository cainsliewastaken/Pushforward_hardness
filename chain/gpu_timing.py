import sys, time, torch, numpy as np
sys.path.insert(0, sys.argv[1]); import l63test_torch as T
from l63test import make_data
Xn, Yn = make_data(200, 0.5); X = torch.tensor(Xn, device=T.DEV); Y = torch.tensor(Yn, device=T.DEV); n = 200
print('device', T.DEV, torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
for depth, H in [(4, 96), (4, 256)]:
    net = T.MLP(3, H, depth, 3, seed=0); theta = net.pack(); eta = 1e-3
    for rep in range(2):
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(1000):
            out, A = net.forward(X); g = net.grad(X, (out-Y)/n, A); theta -= eta*g; net.unpack(theta)
        torch.cuda.synchronize(); ts = (time.time()-t0)/1000
    torch.cuda.synchronize(); t0 = time.time(); J, out = net.jacobian(X); K = T.kernel(J, n); lam1 = float(torch.linalg.eigvalsh(K)[-1]); torch.cuda.synchronize(); tj = time.time()-t0
    E = out.reshape(-1) - Y.reshape(-1); gE = J.T@E/n
    torch.cuda.synchronize(); t0 = time.time(); Pu = T.along_dir_second(net, X, theta, gE/torch.linalg.norm(gE)); s = T.opnorm(Pu); torch.cuda.synchronize(); th = time.time()-t0
    print(f'depth={depth} H={H}: step {ts*1e3:.3f}ms  jac+kernel+eig {tj:.3f}s  H2 {th:.3f}s  mem {torch.cuda.max_memory_allocated()/1e9:.2f}GB', flush=True)
