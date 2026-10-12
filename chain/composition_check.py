"""(A) Composition of saturation statistics through the layers vs measured backprop-factor roughness.
For each layers2 run and snapshot: amplitudes A_l (rank-16 norm of dW_l), init pre-activation std sig_l per layer
(from a forward pass at theta_0), per-unit shift scale a_eff_l = A_l * rms(a_{l-1}) / sqrt(H_l) (spike spread over
units), Hermite degree distribution of tanh'(z0 + a_eff s) per layer (table), backprop factor of layer l = product of
tanh' factors of layers l..L-1 -> degree distributions convolve; roughness index = 0.06 * mean degree (+ 0.025 floor).
Compare with measured rough(delta_l) at each snapshot."""
import numpy as np, glob, os, sys, torch
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hermite_saturation import coeffs
from l63test import make_data
from l63test_torch import MLP, DEV
from scipy.stats import spearmanr
torch.set_default_dtype(torch.float64)
cache = {}
def degree_dist(a, sig, M=24):
    key = (round(a, 2), round(sig, 2))
    if key not in cache:
        c = coeffs(min(a, 12.0), sig, M=M)**2; cache[key] = c/c.sum()
    return cache[key]
def convolve(p, q):
    r = np.convolve(p, q); return r[:len(p)]/r[:len(p)].sum()
rows = []
for fn in sorted(glob.glob(L63_ROOT + '/runs/layers2/lay_*.npy')):
    r = np.load(fn, allow_pickle=True).item(); H = r['H']; depth = r['depth']; Delta = r['Delta']; sc = r['outscale']
    os.environ['L63_OUTSCALE'] = str(sc); Xn, Yn = make_data(200, Delta); X = torch.tensor(Xn, device=DEV)
    net = MLP(3, H, depth, 3, seed=0); _, Acts = net.forward(X)
    sig = [float((Acts[l] @ net.W[l].T + net.b[l]).std()) for l in range(depth)]          # init pre-activation std per hidden layer
    rms_in = [float(Acts[l].norm()/np.sqrt(Acts[l].numel())) for l in range(depth)]
    taus = r['snap_t']*r['lam1_0']
    for j in range(len(taus)):
        A = [np.linalg.norm(sv[:16]) for sv in r['snap_svdW'][j]]
        a_eff = [A[l]*rms_in[l]*np.sqrt(Acts[l].shape[1])/np.sqrt(H) for l in range(depth)]   # shift std per unit: |dW a| ~ A |a| / sqrt(H_out) * sqrt(fan_in)/... (first-order estimate)
        dists = [degree_dist(a_eff[l], sig[l]) for l in range(depth)]
        pred = []
        for l in range(depth):
            p = dists[l]
            for m in range(l+1, depth): p = convolve(p, dists[m])
            mean_deg = (np.arange(len(p))*p).sum(); pred.append(0.025 + 0.06*mean_deg)
        meas = r['snap_factor_rough'][j][:, 2]
        rows.append(dict(tag=r['tag'], tau=taus[j], depth=depth, a_eff=np.round(a_eff, 2), pred=np.round(pred, 3), meas=np.round(meas, 3)))
        print(f"{r['tag']:26s} tau={taus[j]:7.0f} a_eff={np.round(a_eff,2)} pred rough(delta_l)={np.round(pred,3)} measured={np.round(meas,3)}")
P = np.concatenate([q['pred'] for q in rows]); Mm = np.concatenate([q['meas'] for q in rows])
print(f"\nall layers/snapshots: Spearman(pred, meas) = {spearmanr(P, Mm)[0]:.2f}; median pred/meas (meas>0.05) = {np.median(P[Mm>0.05]/Mm[Mm>0.05]):.2f}")
