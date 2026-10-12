"""Prop. 31 check: compare the measured propagator growth (lyapunov_full.txt) with the bound exp(int (1/n)||E|| H2 dt) and with
exp(int lambda_max(-H)_+ dt) estimated by power iteration on -H along the reference run (H96 d2 Delta 0.5)."""
import numpy as np, torch, os, sys
os.environ['L63_OUTSCALE']='1.0'; sys.argv=['x']; torch.set_default_dtype(torch.float64)
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
from l63_late import GraphStepper
from torch.func import grad, jvp
n=200; H=96; depth=2
Xn,Yn=make_data(n,0.5); X=torch.tensor(Xn,device=DEV); Y=torch.tensor(Yn,device=DEV); Yv=Y.reshape(-1)
net=MLP(3,H,depth,3,seed=0); L=len(net.W); shapes=[W.shape for W in net.W]+[b.shape for b in net.b]; sizes=[int(np.prod(s)) for s in shapes]
def f(th):
    out=[];i=0
    for s,k in zip(shapes,sizes): out.append(th[i:i+k].reshape(s)); i+=k
    Ws,bs=out[:L],out[L:]; a=X
    for l in range(L-1): a=torch.tanh(a@Ws[l].T+bs[l])
    return (a@Ws[-1].T+bs[-1]).reshape(-1)
loss=lambda th: 0.5*((f(th)-Yv)**2).sum()/n
def lam_max_negH(th,iters=40):   # power iteration on (c I - H) to get the most negative eigenvalue of H
    hvp=lambda v: jvp(grad(loss),(th,),(v,))[1]
    v=torch.randn_like(th); v/=v.norm(); c=None
    for _ in range(iters):   # first estimate lambda_max(H) for the shift
        w=hvp(v); c=float(v@w); v=w/w.norm()
    c=abs(c)*1.05; u=torch.randn_like(th); u/=u.norm(); mu=0.
    for _ in range(iters):
        w=c*u-hvp(u); mu=float(u@w); u=w/w.norm()
    return mu-c      # = lambda_max(-H) (positive if H has a negative eigenvalue)
def H2(th,E):        # operator norm of dJ/dtheta along the descent direction, /sqrt(n)
    g=grad(lambda t: f(t)@E)(th); d=g/g.norm()
    Mw=lambda w: jvp(lambda t: jvp(f,(t,),(w,))[1],(th,),(d,))[1]; MTu=lambda u: grad(lambda t: jvp(f,(t,),(d,))[1]@u)(th)
    w=torch.randn_like(th); w/=w.norm(); s=0.
    for _ in range(20): u=Mw(w); s=float(u.norm()); w=MTu(u); w/=w.norm()
    return s/np.sqrt(n)
J0,_=net.jacobian(X); eta=0.4/float(torch.linalg.eigvalsh(kernel(J0,n))[-1]); st=GraphStepper(net,X,Y,n,eta); t=0.
grid=np.concatenate([[0],np.geomspace(0.05,3000,22)]); rows=[]
print("   t      ||E||     H2      ||E||H2/n    lam_max(-H)_+    running int of bound A=int ||E||H2/n   B=int lam_max(-H)_+   exp(A)  exp(B)")
IA=IB=0.; tprev=0.; aprev=bprev=None
for target in grid:
    k=int(np.ceil((target-t)/eta-1e-9)) if target>t else 0
    if k>0: st.run(k); t+=k*eta
    st.net.unpack(st.theta); th=st.theta.clone(); E=f(th)-Yv; En=float(E.norm())
    h2=H2(th,E); lm=max(lam_max_negH(th),0.); a=En*h2/n; b=lm
    if aprev is not None: IA+=0.5*(a+aprev)*(t-tprev); IB+=0.5*(b+bprev)*(t-tprev)
    aprev,bprev,tprev=a,b,t
    print(f"{t:8.2f}  {En/np.sqrt(n):.4f}  {h2:7.3f}  {a:9.3e}    {lm:9.3e}        {IA:8.3f}            {IB:8.3f}      {np.exp(IA):7.2f}  {np.exp(IB):7.2f}", flush=True)
    l1=float(torch.linalg.eigvalsh(kernel(st.net.jacobian(X)[0],n))[-1])
    if eta*l1>1.0: eta=0.5/l1; st.eta.fill_(eta)
print("measured propagator over the run (lyapunov_full.txt): x6 random, x21 tail, x4.9 top")
