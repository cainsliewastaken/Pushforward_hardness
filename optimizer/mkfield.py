"""Field-problem data, identical to muon_field.py (Section 44): GRF inputs on a 16x16 periodic grid with spectrum (1+k^2)^-2,
target tanh(2 N[(-Lap)^p u]); 2000 train / 500 test, torch seed 0. Saves npz with X, Y, Xv, Yv (flattened 256-vectors)."""
import sys, numpy as np, torch
p, out = int(sys.argv[1]), sys.argv[2]
ntr_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 2000; nte_arg = int(sys.argv[4]) if len(sys.argv) > 4 else 500
torch.set_default_dtype(torch.float64); torch.manual_seed(0)
G = 16; ntr, nte = ntr_arg, nte_arg
kx = torch.fft.fftfreq(G) * G; KX, KY = torch.meshgrid(kx, kx, indexing='ij'); K2 = KX ** 2 + KY ** 2
def grf(m):
    z = torch.randn(m, G, G, dtype=torch.complex128); amp = (1 + K2) ** -2.0; u = torch.fft.ifft2(z * amp).real
    return u / u.std()
def target(u, p):
    uh = torch.fft.fft2(u); v = torch.fft.ifft2(uh * (K2 ** p)).real if p > 0 else u.clone(); v = v / v.std()
    return torch.tanh(2 * v)
U = grf(ntr + nte); Yall = target(U, p)
X, Y = U[:ntr].reshape(ntr, -1).numpy(), Yall[:ntr].reshape(ntr, -1).numpy()
Xv, Yv = U[ntr:].reshape(nte, -1).numpy(), Yall[ntr:].reshape(nte, -1).numpy()
np.savez(out, X=X, Y=Y, Xv=Xv, Yv=Yv, p=p); print('saved', out, X.shape, Y.shape, Xv.shape)
