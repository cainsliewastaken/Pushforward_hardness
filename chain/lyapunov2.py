"""Perturbation growth along training with the CUDA-graph stepper (fast). 1e-7 relative perturbations of theta_0 in a random
direction, the tail-feature direction (J0^T v_300) and the top-feature direction (J0^T v_1); separation in weights and residual on a log grid."""
import numpy as np, torch, os, sys
os.environ['L63_OUTSCALE']='1.0'; sys.argv=['x']; torch.set_default_dtype(torch.float64)
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
from l63_late import GraphStepper
n=200; H=96; depth=2
Xn,Yn=make_data(n,0.5); X=torch.tensor(Xn,device=DEV); Y=torch.tensor(Yn,device=DEV); Yv=Y.reshape(-1)
base=MLP(3,H,depth,3,seed=0); th0=base.pack().clone(); J0,_=base.jacobian(X); K0=kernel(J0,n); w,V=torch.linalg.eigh(K0); V=V.flip(1)
g=torch.Generator(device=DEV).manual_seed(7); rnd=torch.randn_like(th0,generator=g); rnd/=rnd.norm()
tail=J0.T@V[:,299]; tail/=tail.norm(); top=J0.T@V[:,0]; top/=top.norm(); rel=1e-7
dirs={'base':None,'random':rnd,'tail(rank300)':tail,'top(rank1)':top}
steppers={}
for k,d in dirs.items():
    m=MLP(3,H,depth,3,seed=0)
    if d is not None: m.unpack(th0+rel*th0.norm()*d)
    steppers[k]=GraphStepper(m,X,Y,n,0.4/float(w[-1]))
eta=0.4/float(w[-1]); t=0.; step=0; grid=np.geomspace(0.01,3000,40)
print(f"relative perturbation {rel:g}. columns per direction: ||dtheta(t)||/||dtheta(0)||, ||dE(t)||/||E(t)||")
print("      t   " + "   ".join(f"{k:>14s}: dth      dE/E" for k in dirs if k!='base'))
for target in grid:
    k_steps=max(1,int(np.ceil((target-t)/eta-1e-9)))
    for st in steppers.values(): st.run(k_steps)
    t+=k_steps*eta; step+=k_steps
    sb=steppers['base']; sb.net.unpack(sb.theta); ob,_=sb.net.forward(X); Eb=(ob-Y).reshape(-1)
    row=[]
    for k,st in steppers.items():
        if k=='base': continue
        st.net.unpack(st.theta); o,_=st.net.forward(X); E=(o-Y).reshape(-1)
        row.append(f"{float((st.theta-sb.theta).norm()/(rel*th0.norm())):10.3e} {float((E-Eb).norm()/Eb.norm()):9.2e}")
    # eta adaptation as in the harness (same eta for all copies)
    Jb,_=sb.net.jacobian(X); l1=float(torch.linalg.eigvalsh(kernel(Jb,n))[-1])
    if eta*l1>1.0:
        eta=0.5/l1
        for st in steppers.values(): st.eta.fill_(eta)
    print(f"{t:9.2f}   " + "   ".join(row) + f"   ||E||={float(torch.sqrt(Eb@Eb/n)):.4f}  eta*lam1={eta*l1:.2f}", flush=True)
print("done")
