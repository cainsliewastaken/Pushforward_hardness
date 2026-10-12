"""Full-batch Adam on an L63 design until it reaches the saved GD run's final training error; lr sweep; FLOP accounting.
usage: python adam_l63.py TAG MAXSTEPS lr1,lr2,..."""
import sys, os, re, time, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
torch.set_default_dtype(torch.float64); torch.set_num_threads(int(os.environ.get('SEGFLOW_THREADS', '8'))); torch.set_num_interop_threads(1)
from l63_design import make_data_stats
from l63test_torch import MLP
tag = sys.argv[1]; maxsteps = int(sys.argv[2]); lrs = [float(x) for x in sys.argv[3].split(',')]
mm = re.match(r'H(\d+)_D([\d.]+)_depth(\d+)_s(\d+)(_os([\d.]+))?', tag); H, Delta, depth, seed = int(mm[1]), float(mm[2]), int(mm[3]), int(mm[4]); osc = float(mm[6]) if mm[6] else 1.0
d = np.load(f'../runs/heldout/des_{tag}.npy', allow_pickle=True).item(); n = d['n']; E_target = float(d['E'][-1])
Xr, Yr = make_data_stats(n, Delta, 1); mu, sd = Xr.mean(0), Xr.std(0); X = torch.tensor((Xr-mu)/sd); Y = torch.tensor((Yr-mu)/sd).reshape(-1)
net = MLP(3, H, depth, 3, seed=seed)
res = {}
for lr in lrs:
    Ws = [torch.nn.Parameter(w.clone()) for w in net.W]; bs = [torch.nn.Parameter(b.clone()) for b in net.b]
    def fwd():
        a = X
        for l in range(depth): a = torch.tanh(a @ Ws[l].T + bs[l])
        return (a @ Ws[-1].T + bs[-1]).reshape(-1)
    with torch.no_grad(): out0 = fwd().clone()
    opt = torch.optim.Adam(Ws + bs, lr=lr); t0 = time.time(); best = 1e9; hit = None; E = None
    for step in range(1, maxsteps+1):
        opt.zero_grad(); o = fwd(); o = o if osc == 1.0 else osc*(o - out0); r = o - Y; loss = 0.5*torch.dot(r, r)/n; loss.backward(); opt.step()
        if step % 100 == 0 or step == 1:
            E = float(torch.sqrt(2*loss.detach())); best = min(best, E)
            if E <= E_target: hit = step; break
    res[lr] = (hit, best, time.time()-t0)
    print(f'{tag} Adam lr={lr:g}: ' + (f'reached {E_target:.4f} at step {hit}' if hit else f'NOT reached in {maxsteps} steps (best {best:.4f})') + f'  ({time.time()-t0:.0f}s)', flush=True)
good = [(h, lr) for lr, (h, b, w) in res.items() if h]
sf = np.load(f'../runs/segflow/{tag}_target_flow_m30_d0.25.npy', allow_pickle=True).item()['hist'][-1]
flop_sf = 2*sf['jvp'] + 3*sf['vjp'] + 1*sf['forward']
if good:
    h, lr = min(good); print(f'SUMMARY {tag}: Adam best lr {lr:g} steps {h} -> FLOP-eq {3*h} | SegFlow FLOP-eq {flop_sf} ({sf["jvp"]} Lanczos its) | GD steps {d["steps"]} -> FLOP-eq {3*d["steps"]} | ratio Adam/SegFlow {3*h/flop_sf:.1f}x', flush=True)
else:
    print(f'SUMMARY {tag}: Adam did not reach target in {maxsteps} steps (best {min(b for _, b, _ in res.values()):.4f}) -> FLOP-eq > {3*maxsteps} | SegFlow FLOP-eq {flop_sf} | ratio > {3*maxsteps/flop_sf:.1f}x', flush=True)
