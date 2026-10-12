"""Optimizer stepper for the ground-truth harness: gd (default), momentum (0.9), adam (b1 .9, b2 .999, eps 1e-8),
sgd (minibatch B, fixed random batches per step), weight decay WD; loss sq | l1 | huber (delta 0.1). Eager (no CUDA graph)
except plain gd+sq, which keeps the graph. Env: L63_OPT, L63_LOSS, L63_BATCH, L63_WD."""
import os, torch, numpy as np
from l63_late import GraphStepper
OPT = os.environ.get('L63_OPT', 'gd'); LOSS = os.environ.get('L63_LOSS', 'sq'); B = int(os.environ.get('L63_BATCH', '0')); WD = float(os.environ.get('L63_WD', '0'))
class OptStepper:
    def __init__(self, net, X, Y, n, eta0):
        self.net, self.X, self.Y, self.n = net, X, Y, n; self.theta = net.pack().clone(); self.eta = torch.full((), eta0, device=X.device)
        self.use_graph = False
        if OPT == 'gd' and LOSS == 'sq' and B == 0 and WD == 0:
            self.inner = GraphStepper(net, X, Y, n, eta0); self.theta = self.inner.theta; self.eta = self.inner.eta; self.use_graph = True; return
        d = self.theta.numel(); self.m = torch.zeros(d, device=X.device); self.v = torch.zeros(d, device=X.device); self.k = 0; self.gen = torch.Generator(device=X.device).manual_seed(0)
    def _grad(self):
        X, Y = self.X, self.Y
        if B > 0:
            idx = torch.randint(0, self.n, (B,), generator=self.gen, device=X.device); X = X[idx]; Y = Y[idx]
        out, A = self.net.forward(X); R = out - Y
        if LOSS == 'sq': dl = R
        elif LOSS == 'l1': dl = torch.sign(R)
        elif LOSS == 'huber': dl = torch.clamp(R, -0.1, 0.1)
        g = self.net.grad(X, dl/X.shape[0], A)
        if WD > 0: g = g + WD*self.theta
        return g
    def run(self, k):
        if self.use_graph: return self.inner.run(k)
        for _ in range(k):
            g = self._grad(); self.k += 1
            if OPT == 'momentum':
                self.m.mul_(0.9).add_(g); step = self.m
            elif OPT == 'adam':
                self.m.mul_(0.9).add_(0.1*g); self.v.mul_(0.999).add_(0.001*g*g)
                mh = self.m/(1-0.9**self.k); vh = self.v/(1-0.999**self.k); step = mh/(torch.sqrt(vh)+1e-8)
            else: step = g
            self.theta.sub_(self.eta*step); self.net.unpack(self.theta)
