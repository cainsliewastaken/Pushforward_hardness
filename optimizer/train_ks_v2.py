"""General MLP training driver: Adam / Muon-free baselines vs KronSign, full batch or minibatch, linear lr decay.
Data: npz with X, Y, Xv, Yv (raw; standardised here with training-set mean/std of X, applied to X and Y, as in kadam.py).
Output: one JSON line per (run, metric) to --out.
"""
import os, sys, json, math, time, argparse, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
from kronsign import KronSign

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--opt', required=True, choices=['adam', 'kronsign'])
p.add_argument('--lr', type=float, required=True)
p.add_argument('--widths', default='160,160', help='hidden widths, comma-separated')
p.add_argument('--act', default='tanh', choices=['tanh', 'gelu', 'relu'])
p.add_argument('--steps', type=int, default=3000)
p.add_argument('--batch', type=int, default=0, help='0 = full batch')
p.add_argument('--seed', type=int, default=1)
p.add_argument('--b1', type=float, default=0.9)
p.add_argument('--b2', type=float, default=0.999)
p.add_argument('--warmup', type=int, default=0)
p.add_argument('--rule', default='sign', choices=['sign', 'adam'])
p.add_argument('--S_mode', default='mc', choices=['mc', 'exact', 'emp'])
p.add_argument('--stats_every', type=int, default=10)
p.add_argument('--eig_every', type=int, default=10)
p.add_argument('--stats_n', type=int, default=2000, help='points per statistics update (subset)')
p.add_argument('--eig', default='qr', choices=['eigh', 'qr'])
p.add_argument('--init_from_kadam', type=int, default=0, help='1 = copy the tanh MLP init used by kadam.py (validation)')
p.add_argument('--float32', type=int, default=0)
p.add_argument('--log_every', type=int, default=100)
p.add_argument('--standardize', default='x', choices=['x', 'none'], help='x: standardise X and Y with the training X statistics (one-step maps); none: raw')
p.add_argument('--profile', type=int, default=0)
p.add_argument('--async_eig', type=int, default=0)
p.add_argument('--lowrank', type=int, default=1, help='kronsign: rotate the minibatch gradient on the batch side when cheaper')
p.add_argument('--stats_from_train', type=int, default=1, help='mc statistics from the training forward graph (no separate pass)')
p.add_argument('--out', required=True)
a = p.parse_args()

torch.set_default_dtype(torch.float32 if a.float32 else torch.float64)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(a.seed)
z = np.load(a.data)
Xr, Yr, Xvr, Yvr = z['X'], z['Y'], z['Xv'], z['Yv']
mu, sd = Xr.mean(0), Xr.std(0)
if a.standardize == 'none': mu, sd = 0.0, 1.0
def T(v): return torch.tensor(v, device=DEV, dtype=torch.get_default_dtype())
X, Y, Xv, Yv = T((Xr - mu) / sd), T((Yr - mu) / sd), T((Xvr - mu) / sd), T((Yvr - mu) / sd)
n, din = X.shape; dout = Y.shape[1]
widths = [int(w) for w in a.widths.split(',')]
Act = dict(tanh=torch.nn.Tanh, gelu=torch.nn.GELU, relu=torch.nn.ReLU)[a.act]
layers = []; d = din
for w in widths: layers += [torch.nn.Linear(d, w), Act()]; d = w
layers.append(torch.nn.Linear(d, dout))
model = torch.nn.Sequential(*layers).to(DEV)
if a.init_from_kadam:
    from l63test_torch import MLP
    ref = MLP(din, widths[0], len(widths), dout, seed=a.seed)
    lins = [m for m in model if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for m, W, b in zip(lins, ref.W, ref.b): m.weight.copy_(W.to(DEV)); m.bias.copy_(b.to(DEV))
P = sum(q.numel() for q in model.parameters())

gen = torch.Generator(device=DEV); gen.manual_seed(a.seed)
if a.opt == 'adam':
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(a.b1, a.b2))
else:
    opt = KronSign(model, a.lr, rule=a.rule, b1=a.b1, b2=a.b2, stats_every=a.stats_every, eig_every=a.eig_every,
                   S_mode=a.S_mode, eig=a.eig, seed=a.seed)

rows = []
def emit(metric, value, step=None):
    rows.append(dict(vars(a), n=n, P=P, step=step, metric=metric, value=value))

def rmse(Xe, Ye):
    with torch.no_grad():
        r = model(Xe) - Ye; return float(torch.sqrt((r * r).sum() / Xe.shape[0]))

prof = {}
_lins = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
WPS = sum(m.weight.numel() + (m.bias.numel() if m.bias is not None else 0) for m in _lins)   # weights per sample-forward
flops_total = 0.0
if a.opt == 'kronsign': opt.profile = bool(a.profile); opt.async_eig = bool(a.async_eig); opt.lowrank = bool(a.lowrank)
if DEV == 'cuda': torch.cuda.synchronize()
t0 = time.time(); diverged = False
for step in range(1, a.steps + 1):
    fac = (1 - (step - 1) / a.steps) * (min(1.0, step / a.warmup) if a.warmup > 0 else 1.0); lr = a.lr * fac
    if a.batch and a.batch < n:
        idx = torch.randint(0, n, (a.batch,), generator=gen, device=DEV); xb, yb = X[idx], Y[idx]
    else:
        xb, yb = X, Y
    if a.profile and DEV == 'cuda': torch.cuda.synchronize()
    tfb = time.perf_counter()
    model.zero_grad(set_to_none=True)
    emp = a.opt == 'kronsign' and a.S_mode == 'emp' and (step - 1) % a.stats_every == 0
    trs = a.opt == 'kronsign' and a.stats_from_train and a.S_mode == 'mc' and (step - 1) % a.stats_every == 0
    lrk = a.opt == 'kronsign' and a.lowrank
    if lrk: opt.begin_step(keep_out=trs)
    else:
        if emp: opt.capture_emp()
        if trs: opt.begin_train_stats()
    outp = model(xb)
    tf = opt.end_train_stats(outp) if trs else None
    r = outp - yb
    L = 0.5 * (r * r).sum() / xb.shape[0]
    L.backward()
    flops_total += 6.0 * xb.shape[0] * WPS + (4.0 * xb.shape[0] * WPS if trs else 0.0)
    if a.opt == 'kronsign': opt._last_B = xb.shape[0]
    if a.profile:
        if DEV == 'cuda': torch.cuda.synchronize()
        prof['fwd_bwd'] = prof.get('fwd_bwd', 0.0) + time.perf_counter() - tfb
    tst = time.perf_counter()
    if not math.isfinite(float(L.detach())) or float(L.detach()) > 1e6:
        diverged = True; emit('diverged', 1, step); break
    if a.opt == 'adam':
        for g_ in opt.param_groups: g_['lr'] = lr
        opt.step(); flops_total += 10.0 * P
    else:
        ef = (opt.emp_from_step(loss_scale=xb.shape[0]) if lrk else opt.finish_emp(loss_scale=xb.shape[0])) if emp else tf
        sx = None
        if a.S_mode != 'emp' and not a.stats_from_train and (step - 1) % a.stats_every == 0:
            sx = X[torch.randperm(n, generator=gen, device=DEV)[:a.stats_n]] if a.stats_n < n else X
        opt.step(stats_x=sx, emp_factors=ef, lr=lr)
    if a.profile:
        if DEV == 'cuda': torch.cuda.synchronize()
        prof['opt_step_total'] = prof.get('opt_step_total', 0.0) + time.perf_counter() - tst
    if step % a.log_every == 0 or step == a.steps:
        emit("E_train_batch", math.sqrt(2 * float(L.detach())), step)
if DEV == 'cuda': torch.cuda.synchronize()
wall = time.time() - t0
Et, Ev = rmse(X, Y), rmse(Xv, Yv)
flops_total += getattr(opt, 'flops', 0.0); emit('flops_total', flops_total); emit('n_lowrank_rotations', getattr(opt, 'n_lowrank', 0)); emit('E_train_final', Et, a.steps); emit('E_val_final', Ev, a.steps); emit('wall_s', wall)
if a.profile:
    allp = dict(prof); allp.update({'opt.' + k: v for k, v in getattr(opt, 'times', {}).items()})
    print('PROFILE ms/step: ' + '  '.join(f'{k}={1000*v/a.steps:.3f}' for k, v in allp.items()) + f'  total={1000*wall/a.steps:.3f}', flush=True)
    for k, v in allp.items(): emit('prof_ms_per_step_' + k, 1000 * v / a.steps)
with open(a.out, 'a') as fh:
    for rw in rows: fh.write(json.dumps(rw) + '\n')
print(f'{a.opt} rule={a.rule if a.opt=="kronsign" else "-"} lr={a.lr:g} widths={a.widths} batch={a.batch or "full"} n={n}: '
      f'E_train={Et:.4g} E_val={Ev:.4g} wall={wall:.1f}s flops={flops_total:.3e}', flush=True)
