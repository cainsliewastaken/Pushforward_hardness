"""Theorem 25 to full second order in 1/s: adds (i) the second-order weight change from the Jacobian changing along the lazy path and
(ii) the residual feedback (lifted kernel accelerates the lazy residual) to a_2. Compares with measured fixed-direction lifts."""
import numpy as np, torch, os, sys
os.environ['L63_OUTSCALE']='1.0'; sys.argv=['x']; torch.set_default_dtype(torch.float64)
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
from torch.func import grad, jvp
n=200; H=96; depth=2
Xn,Yn=make_data(n,0.5); X=torch.tensor(Xn,device=DEV); Y=torch.tensor(Yn,device=DEV); Yv=Y.reshape(-1)
net=MLP(3,H,depth,3,seed=0); L=len(net.W); shapes=[W.shape for W in net.W]+[b.shape for b in net.b]; sizes=[int(np.prod(s)) for s in shapes]; th0=net.pack().clone()
def f(th):
    out=[];i=0
    for s,k in zip(shapes,sizes): out.append(th[i:i+k].reshape(s)); i+=k
    Ws,bs=out[:L],out[L:]; a=X
    for l in range(L-1): a=torch.tanh(a@Ws[l].T+bs[l])
    return (a@Ws[-1].T+bs[-1]).reshape(-1)
J0,_=net.jacobian(X); K0=kernel(J0,n); w,V=torch.linalg.eigh(K0); w=w.flip(0); V=V.flip(1); c=V.T@(-Yv)
E_lazy=lambda tau: V@(c*torch.exp(-w*tau)); U_lazy=lambda tau: V@(c*(1-torch.exp(-w*tau))/w)
HVP=lambda vec,dth: jvp(lambda th: grad(lambda t: f(t)@vec)(th),(th0,),(dth,))[1]          # (dJ/dtheta [dth])^T vec  =  Hess_{vec.f} dth
dJg=lambda g,dth: jvp(lambda th: jvp(f,(th,),(g,))[1],(th0,),(dth,))[1]                       # (dJ/dtheta [dth]) g
def second_order_terms(tau, nq=24):
    # quadrature nodes on [0,tau], log-spaced plus 0
    ts=np.concatenate([[0.0],np.geomspace(min(1e-3,tau/100),tau,nq)]); dth1=lambda t: -(J0.T@U_lazy(t))/n
    # (i) second-order weight change from the Jacobian changing: dtheta2a = -(1/n) int_0^tau (dJ[dth1(t')])^T E_lazy(t') dt'
    # (ii) residual feedback: dE1(t) = -int_0^t e^{-K0 (t-t')} dK1(t') E_lazy(t') dt',  dK1 E = [J0 (dJ[dth1]^T E) + dJ[dth1] (J0^T E)]/n ; dtheta2b = -(1/n) J0^T int_0^tau dE1
    g2a=torch.zeros_like(th0); src=[]; 
    for t in ts:
        d1=dth1(float(t)); E=E_lazy(float(t)); hE=HVP(E,d1); g2a_t=hE; dK1E=(J0@hE + dJg(J0.T@E,d1))/n
        src.append((float(t),g2a_t,dK1E))
    # trapezoid in t for dtheta2a
    for i in range(len(ts)-1):
        h=ts[i+1]-ts[i]; g2a+= -0.5*h*(src[i][1]+src[i+1][1])/n
    # dE1 at the quadrature nodes via variation of constants in the eigenbasis, then U2 = int dE1
    Vt=V.T; U2=torch.zeros(n*3,device=DEV); dE_prev=torch.zeros(n*3,device=DEV)
    for i in range(1,len(ts)):
        t=ts[i]; acc=torch.zeros(n*3,device=DEV)
        for j in range(i):
            h=ts[j+1]-ts[j]; 
            for (tt,_,s_) in (src[j],src[j+1]):
                acc+= -0.5*h*(V@(torch.exp(-w*(t-tt))*(Vt@s_)))
        dE=acc; U2+=0.5*(ts[i]-ts[i-1])*(dE+dE_prev); dE_prev=dE
    g2b=-(J0.T@U2)/n
    return g2a, g2b
files={2.0:'../runs/gt_lazy/des_H96_D0.5_depth2_s0_os2.npy',4.0:'../runs/gt_lazy/des_H96_D0.5_depth2_s0_os4.npy',8.0:'../runs/gt_lazy/des_H96_D0.5_depth2_s0_os8.npy',16.0:'../runs/gt_lazy/des_H96_D0.5_depth2_s0_os16.npy'}
meas={}
for s,fn in files.items():
    r=np.load(fn,allow_pickle=True).item(); K0r=np.asarray(r['K0'],np.float64); wr,Vr=np.linalg.eigh(K0r); meas[s]=(np.asarray(r['snap_t'])*s*s,[np.asarray(K,np.float64) for K in r['snap_K']],Vr[:,::-1],wr[::-1])
for k in (99,199):
    v=V[:,k]; lam=float(w[k]); Pv=J0.T@v
    print(f"\nrank {k+1}: two-term (a1 + Q + curvature) vs full second order (adds 2nd-order weight change + residual feedback)")
    print("     s     tau      a1/s     (q2+c2)/s^2   extra2/s^2    pred(2-term)   pred(full)    measured   ratio2  ratioFull")
    cache={}
    for s in (16.0,8.0,4.0,2.0):
        st,Ks,Vr,wr=meas[s]; vr=Vr[:,k]
        for tt,K in list(zip(st,Ks))[:7]:
            tau=float(tt); mex=float(vr@K@vr)/wr[k]-1
            if tau not in cache:
                d1=-(J0.T@U_lazy(tau))/n; dp1=HVP(v,d1); dp2=0.5*jvp(lambda th: jvp(lambda t: grad(lambda u: f(u)@v)(t),(th,),(d1,))[1],(th0,),(d1,))[1]
                g2a,g2b=second_order_terms(tau); extra=float(2*(Pv@HVP(v,g2a+g2b))/(n*lam))
                cache[tau]=(float(2*(Pv@dp1)/(n*lam)), float((dp1@dp1)/(n*lam)+2*(Pv@dp2)/(n*lam)), extra)
            a1,b2,x2=cache[tau]; p2=a1/s+b2/s**2; pf=p2+x2/s**2
            print(f"  {s:5g}  {tau:7.1f}  {a1/s:+9.2e}  {b2/s**2:+10.2e}  {x2/s**2:+10.2e}   {p2:+10.3e}  {pf:+10.3e}  {mex:+10.3e}   {p2/mex:5.2f}   {pf/mex:5.2f}")
