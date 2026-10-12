"""Theorem 11 along the trajectory: at snapshot times, split the instantaneous (first-order) weight change of the last hidden layer
into core (per-unit mean of sigma') and bulk, compare the measured bulk/core energy with the prediction from the CURRENT
preactivation dispersion, and report the top-3 energy fraction of the cumulative change."""
import numpy as np, torch, os, sys
os.environ['L63_OUTSCALE']='1.0'; sys.argv=['x']; torch.set_default_dtype(torch.float64)
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
n=200; ny=3; H=96; depth=2
Xn,Yn=make_data(n,0.5); X=torch.tensor(Xn,device=DEV); Y=torch.tensor(Yn,device=DEV); Yv=Y.reshape(-1)
net=MLP(3,H,depth,3,seed=0); L=len(net.W); W1_0=net.W[1].clone()
J,o=net.jacobian(X); eta=0.4/float(torch.linalg.eigvalsh(kernel(J,n))[-1]); t=0.
print("t        sigma_z(last hidden)   pred bulk/core   measured bulk/core (instantaneous dW)   top-3 energy of cumulative dW   lam1*eta")
for target in (0.0,0.3,4.4,43.6,428.0):
    while t<target-1e-12:
        h=min(eta,target-t); J,o=net.jacobian(X); E=o.reshape(-1)-Yv; net.unpack(net.pack()-h*(J.T@E)/n); t+=h
        if t>1 and eta*float(torch.linalg.eigvalsh(kernel(J,n))[-1])>1.0: eta=0.5/float(torch.linalg.eigvalsh(kernel(J,n))[-1])
    o,A=net.forward(X); E=o-Y; z=A[L-2]@net.W[L-2].T+net.b[L-2]; S=1-torch.tanh(z)**2; M=E@net.W[L-1]; a_in=A[L-2]
    h0=S.mean(0,keepdim=True); C=S*M; Cc=h0*M; Cr=C-Cc; dWc=Cc.T@a_in/n; dWr=Cr.T@a_in/n
    pred=float(((S.var(0))*(M**2).mean(0)).sum()/((h0**2)*(M**2).mean(0)).sum()); meas=float((dWr**2).sum()/(dWc**2).sum())
    sv=torch.linalg.svdvals(net.W[1]-W1_0); top3=float((sv[:3]**2).sum()/(sv**2).sum()) if t>0 else float('nan')
    print(f"{t:7.1f}   {float(z.std()):6.2f}               {pred:6.3f}           {meas:6.3f}                               {top3:6.3f}                  {eta*float(torch.linalg.eigvalsh(kernel(net.jacobian(X)[0],n))[-1]):.2f}")
