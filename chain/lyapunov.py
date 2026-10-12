"""Is training chaotic? Perturb theta_0 by a relative 1e-7 in (a) a random direction and (b) the direction of the tail features
(J0^T v for v an init eigenvector at rank 300), train both copies by GD, track the separation in weights and in the residual
on a log grid. Exponential growth => positive finite-time Lyapunov exponent; algebraic => ill-conditioned but not chaotic."""
import numpy as np, torch, os, sys
os.environ['L63_OUTSCALE']='1.0'; sys.argv=['x']; torch.set_default_dtype(torch.float64)
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
n=200; H=96; depth=2
Xn,Yn=make_data(n,0.5); X=torch.tensor(Xn,device=DEV); Y=torch.tensor(Yn,device=DEV); Yv=Y.reshape(-1)
base=MLP(3,H,depth,3,seed=0); th0=base.pack().clone(); J0,_=base.jacobian(X); K0=kernel(J0,n); w,V=torch.linalg.eigh(K0); V=V.flip(1)
g=torch.Generator(device=DEV).manual_seed(7); rnd=torch.randn_like(th0,generator=g); rnd/=rnd.norm()
tail=J0.T@V[:,299]; tail/=tail.norm(); top=J0.T@V[:,0]; top/=top.norm()
rel=1e-7; pert={'random':rnd,'tail(rank300)':tail,'top(rank1)':top}
nets={'base':MLP(3,H,depth,3,seed=0)}
for k,d in pert.items(): m=MLP(3,H,depth,3,seed=0); m.unpack(th0+rel*th0.norm()*d); nets[k]=m
eta=0.4/float(w[-1]); t=0.; grid=np.concatenate([[0],np.geomspace(0.01,3000,31)]); gi=1
print(f"relative perturbation {rel:g} of ||theta_0||. columns: separation ratio ||dtheta(t)||/||dtheta(0)|| and ||dE(t)||/||E(t)|| for each perturbation direction")
print("   t        " + "   ".join(f"{k:>14s}:dth    dE/E " for k in pert))
def step(m,h):
    o,A=m.forward(X); E=o-Y; g=m.grad(X,E/n,A); m.unpack(m.pack()-h*g); return E.reshape(-1)
while t<3000-1e-9:
    target=grid[gi] if gi<len(grid) else 3000
    while t<target-1e-12:
        h=min(eta,target-t); Eb=None
        for k,m in nets.items():
            E=step(m,h); Eb=E if k=='base' else Eb
        t+=h
        if t>1:
            l1=float(torch.linalg.eigvalsh(kernel(nets['base'].jacobian(X)[0],n))[-1]) if abs(t-target)<1e-12 else None
            if l1 is not None and eta*l1>1.0: eta=0.5/l1
    Jb,ob=nets['base'].jacobian(X); Eb=ob.reshape(-1)-Yv; thb=nets['base'].pack(); row=[]
    for k in pert:
        m=nets[k]; _,o=m.jacobian(X); E=o.reshape(-1)-Yv; row.append(f"{float((m.pack()-thb).norm()/(rel*th0.norm())):10.3e} {float((E-Eb).norm()/Eb.norm()):9.2e}")
    print(f"{t:8.2f}  " + "   ".join(row) + f"   ||E||={float(torch.sqrt(Eb@Eb/n)):.4f}", flush=True); gi+=1
