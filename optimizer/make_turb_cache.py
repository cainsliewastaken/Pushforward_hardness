"""One-off cache for the 2D turbulence one-step problem (scale_study.py --problem turb). Reads the vorticity snapshots of the
Flop_Constrained_Temporal_Scaling data (HDF5 'omega', 256 x 256 per global timestep; conventions from that repo's config:
spinup 20000, 400000 training frames, 50000 test frames, lead = time_step / reference_dt = 10), coarsens each frame to
res x res by block means, and stores (x_t, x_{t+lead}) pairs for a training slab and a test slab as float32 .npz on scratch.
Runs on a login node (I/O only, a few minutes); nothing is overwritten unless --force."""
import argparse, os, time, numpy as np, h5py
p = argparse.ArgumentParser()
p.add_argument('--h5', default='/global/homes/c/cainslie/Flop_Constrained_Temporal_Scaling/training_data/2d_turbulence/full_index.h5')
p.add_argument('--out', default=os.path.expandvars('$SCRATCH/l63_held_rough/data/turb64_lead10.npz'))
p.add_argument('--res', type=int, default=64); p.add_argument('--lead', type=int, default=10)
p.add_argument('--spinup', type=int, default=20000); p.add_argument('--ntrain', type=int, default=20000)
p.add_argument('--test_start', type=int, default=420000); p.add_argument('--nval', type=int, default=2048)
p.add_argument('--chunk', type=int, default=1000); p.add_argument('--force', type=int, default=0)
a = p.parse_args()
if os.path.exists(a.out) and not a.force: raise SystemExit(f'{a.out} exists; pass --force 1 to rebuild')
def coarsen(x):                       # (n, 256, 256) -> (n, res, res) block mean
    n, H, W = x.shape; f = H // a.res
    return x.reshape(n, a.res, f, a.res, f).mean(axis=(2, 4))
def slab(om, start, n):
    out = np.empty((n, a.res, a.res), np.float32)
    for i in range(0, n, a.chunk):
        j = min(n, i + a.chunk); out[i:j] = coarsen(np.asarray(om[start + i:start + j], dtype=np.float32))
    return out
t0 = time.time()
with h5py.File(a.h5, 'r') as h:
    om = h['omega']; print('omega', om.shape, flush=True)
    tr = slab(om, a.spinup, a.ntrain + a.lead); print(f'train slab read {time.time() - t0:.0f} s', flush=True)
    va = slab(om, a.test_start, a.nval + a.lead); print(f'test slab read {time.time() - t0:.0f} s', flush=True)
mean, std = float(tr.mean()), float(tr.std())
np.savez(a.out, xtr=tr[:-a.lead], ytr=tr[a.lead:], xva=va[:-a.lead], yva=va[a.lead:], mean=mean, std=std,
         res=a.res, lead=a.lead, spinup=a.spinup, test_start=a.test_start)
print(f'wrote {a.out}: train {a.ntrain} pairs, val {a.nval} pairs, res {a.res}, lead {a.lead}, mean {mean:.4g} std {std:.4g}, {time.time() - t0:.0f} s')
