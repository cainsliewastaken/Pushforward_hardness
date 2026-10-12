"""Correlation-free rank bound along the trajectory: instantaneous dW_l = -(1/n)(S (.) M)^T A with rank M <= ny exactly, so rank(dW) <= ny * rank_delta(S).
Measure at snapshot times: spectrum of S = sigma'(Z) (data x units) for each hidden layer, the energy of the instantaneous dW in its top ny*r
singular directions, the cumulative dW top-3/top-9 energy, and the effective rank of Z (preactivation matrix)."""
import numpy as np, torch, os, sys
os.environ['L63_OUTSCALE']='1.0'; sys.argv=['x']; torch.set_default_dtype(torch.float64)
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
n=200; ny=3; H=96; depth=int(sys.argv[1]) if len(sys.argv)>1 else 2; D=float(sys.argv[2]) if len(sys.argv)>2 else 0.5
Xn,Yn=make_data(n,D); X=torch.tensor(Xn,device=DEV); Y=torch.tensor(Yn,device=DEV); Yv=Y.reshape(-1)
net=MLP(3,H,depth,3,seed=0); L=len(net.W); W0=[W.clone() for W in net.W]
J,o=net.jacobian(X); eta=0.4/float(torch.linalg.eigvalsh(kernel(J,n))[-1]); t=0.
def efrac(M,r): s=torch.linalg.svdvals(M); return float((s[:r]**2).sum()/(s**2).sum())
def erank(M): s=torch.linalg.svdvals(M); p=s**2/(s**2).sum(); p=p[p>0]; return float(torch.exp(-(p*torch.log(p)).sum()))
print(f"H{H} depth{depth} Delta{D}: per hidden layer l: [erank Z_l | S_l energy in top 1/3/10 sv | instantaneous dW_l energy in top 3/9/30 sv | cumulative dW_l top-3/top-9 | (S.M) measured energy outside rank 3*r_S(90%)]")
for target in (0.0,0.3,4.4,43.6,428.0,1500.0):
    while t<target-1e-12:
        h=min(eta,target-t); J,o=net.jacobian(X); E=o.reshape(-1)-Yv; net.unpack(net.pack()-h*(J.T@E)/n); t+=h
        if eta*float(torch.linalg.eigvalsh(kernel(J,n))[-1])>1.0: eta=0.5/float(torch.linalg.eigvalsh(kernel(J,n))[-1])
    o,A=net.forward(X); E=o-Y
    # backprop coefficients per layer: delta_l^c(x) = sigma'(z_l) * m_l^c ;  M_l[i,q] = sum_c m_{l,q}^c(x_i) E_c(x_i)  (rank <= ny only for the LAST hidden layer; deeper: through sigma' of layers above)
    row=[]
    m=E@net.W[L-1]                      # n x H: incoming backward signal to the last hidden layer, contracted with E (rank <= ny)
    for l in range(L-2,-1,-1):
        z=A[l]@net.W[l].T+net.b[l]; S=1-torch.tanh(z)**2; C=S*m; dW=C.T@A[l]/n
        sS=torch.linalg.svdvals(S); fS=(sS**2).cumsum(0)/(sS**2).sum(); rS=int((fS<0.9).sum())+1
        row.append(f"l{l}: erZ {erank(z):4.1f} | S {efrac(S,1):.2f}/{efrac(S,3):.2f}/{efrac(S,10):.2f} | dW {efrac(dW,3):.3f}/{efrac(dW,9):.3f}/{efrac(dW,30):.3f} | cum {efrac(net.W[l]-W0[l],3):.3f}/{efrac(net.W[l]-W0[l],9):.3f} | r_S(90%)={rS}, dW energy in top {3*rS}: {efrac(dW,3*rS):.3f}")
        m=(C)@net.W[l]                  # propagate the backward coefficient to the layer below (now contracted with E): n x H_{l-1}
    print(f"t={t:7.1f} ||E||={float(torch.sqrt((E**2).sum()/n)):.4f}  " + "  ||  ".join(row))
