"""Unit-by-unit comparison of the own-unit model state with the real network at the same time (depth 2)."""
import torch, numpy as np, sys
m = torch.load(sys.argv[1]); r = torch.load(sys.argv[2])
Zm, Mm, Km, Em = m['Zu'][0].double(), m['Mu'][0].double(), m['K'].double(), m['E'].double()
Zr, Mr, Kr, Er = r['Z'][0].double(), r['M'][0].double(), r['K'].double(), r['E'].double()
n = Zm.shape[1]
print(f"t model {m['t']:.3f} real {r['t']:.3f}; ||E|| model {float(torch.sqrt(Em@Em/n)):.4f} real {float(torch.sqrt(Er@Er/n)):.4f}; cos(E) {float(Em@Er/Em.norm()/Er.norm()):.4f}")
wm = torch.linalg.eigvalsh(Km).flip(0); wr = torch.linalg.eigvalsh(Kr).flip(0)
print("sorted eigenvalues model/real at ranks 1,10,50,100,200:", [f"{float(wm[k]/wr[k]):.3f}" for k in (0, 9, 49, 99, 199)])
print(f"||K_m - K_r||_F / ||K_r - K_0||_F : need K0 -> skip; rel Frobenius diff {float((Km-Kr).norm()/Kr.norm()):.3e}")
dZ = Zm - Zr; dM = Mm - Mr
print(f"z: rms real {float(Zr.pow(2).mean().sqrt()):.4f}, rms(model-real) {float(dZ.pow(2).mean().sqrt()):.3e}; per-unit max rel diff {float((dZ.norm(dim=1)/Zr.norm(dim=1)).max()):.3e}")
print(f"m: rms real {float(Mr.pow(2).mean().sqrt()):.4f}, rms(model-real) {float(dM.pow(2).mean().sqrt()):.3e}")
# decompose z difference: mean over data (bias-like), input-linear part, remainder
X = None
try:
    import os; os.environ['L63_OUTSCALE'] = '1.0'; from l63test import make_data; Xn, _ = make_data(n, float(sys.argv[3])); X = torch.tensor(Xn).double()
    F = torch.cat([X, torch.ones(n, 1)], 1); P = F @ torch.linalg.solve(F.T @ F, F.T)
    inspan = dZ @ P; print(f"z diff: in input span {float(inspan.norm()/dZ.norm()):.3f} of norm, outside {float((dZ-inspan).norm()/dZ.norm()):.3f}")
    # how much did z move at all?
    print(f"(for scale) real z displacement from init is not stored here; |dZ|/|Zr| = {float(dZ.norm()/Zr.norm()):.3e}")
except Exception as e: print('span decomposition skipped:', e)
