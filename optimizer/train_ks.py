"""General training driver: Adam vs KronSign (sign / adam / kronv rules, GN or Shampoo statistics), MLP or periodic CNN,
full batch or minibatch, linear lr decay, optional DDP (torchrun).
Data: npz with X, Y, Xv, Yv (raw; --standardize x standardises X and Y with the training-set X statistics).
Output: one JSON line per (run, metric) to --out (rank 0 only under DDP).
"""
import os, sys, json, math, time, argparse, numpy as np, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chain')]  # chain/ holds l63_design, l63test_torch, l63_late
from kronsign import KronSign

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--opt', required=True, choices=['adam', 'kronsign', 'kronroot'])
p.add_argument('--root_every', type=int, default=10, help='kronroot: steps between Newton-Schulz root refreshes')
p.add_argument('--ns_iters', type=int, default=20, help='kronroot: Newton-Schulz iterations per square root')
p.add_argument('--damp', type=float, default=1e-3, help='kronroot: factor damping relative to mean eigenvalue')
p.add_argument('--lr', type=float, required=True)
p.add_argument('--arch', default='mlp', choices=['mlp', 'cnn'])
p.add_argument('--widths', default='160,160', help='mlp: hidden widths, comma-separated')
p.add_argument('--channels', default='32,32,32', help='cnn: hidden channels, comma-separated (periodic 3x3 convolutions on the 16x16 grid)')
p.add_argument('--ksize', type=int, default=3)
p.add_argument('--grid', type=int, default=16)
p.add_argument('--act', default='tanh', choices=['tanh', 'gelu', 'relu'])
p.add_argument('--steps', type=int, default=3000)
p.add_argument('--batch', type=int, default=0, help='0 = full batch (per rank under DDP)')
p.add_argument('--seed', type=int, default=1)
p.add_argument('--b1', type=float, default=0.9)
p.add_argument('--b2', type=float, default=0.999)
p.add_argument('--warmup', type=int, default=0)
p.add_argument('--rule', default='sign', choices=['sign', 'adam', 'kronv'])
p.add_argument('--stats', default='gn', choices=['gn', 'shampoo'], help='kronsign: factor statistics (shampoo + rule adam = SOAP)')
p.add_argument('--S_mode', default='mc', choices=['mc', 'exact', 'emp'])
p.add_argument('--last_S', default='stats', choices=['stats', 'identity'])
p.add_argument('--stats_every', type=int, default=10)
p.add_argument('--eig_every', type=int, default=10)
p.add_argument('--eig_batch', type=int, default=0)
p.add_argument('--eig_adapt', type=int, default=0)
p.add_argument('--eig_max', type=int, default=80)
p.add_argument('--beta_stats', type=float, default=0.95)
p.add_argument('--rot_dtype', default='none', choices=['none', 'float32', 'bfloat16'])
p.add_argument('--stats_n', type=int, default=2000, help='points per statistics update (subset) when not from the training pass')
p.add_argument('--eig', default='qr', choices=['eigh', 'qr'])
p.add_argument('--init_from_kadam', type=int, default=0)
p.add_argument('--float32', type=int, default=0)
p.add_argument('--tf32', type=int, default=0, help='allow TF32 matmuls (float32 runs)')
p.add_argument('--log_every', type=int, default=100)
p.add_argument('--standardize', default='x', choices=['x', 'none'])
p.add_argument('--profile', type=int, default=0)
p.add_argument('--async_eig', type=int, default=0)
p.add_argument('--lowrank', type=int, default=1, help='kronsign: rotate the minibatch gradient on the batch side when cheaper (off under DDP)')
p.add_argument('--stats_from_train', type=int, default=1, help='mc statistics from the training forward graph (no separate pass)')
p.add_argument('--reparam', type=int, default=0, help='kronsign: train Linear layers in the rotated basis (no per-step rotation)')
p.add_argument('--block', type=int, default=0, help='kronsign: block-diagonal factors with blocks of this size (0 = full)')
p.add_argument('--ddp', type=int, default=0, help='launch with torchrun; each rank draws its own minibatch')
p.add_argument('--out', required=True)
a = p.parse_args()

torch.set_default_dtype(torch.float32 if a.float32 else torch.float64)
if a.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
rank, world = 0, 1
if a.ddp:
    import torch.distributed as dist
    dist.init_process_group('nccl' if DEV == 'cuda' else 'gloo')
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get('LOCAL_RANK', rank))
    if DEV == 'cuda': torch.cuda.set_device(local); DEV = f'cuda:{local}'
    a.lowrank = 0
torch.manual_seed(a.seed)
z = np.load(a.data)
Xr, Yr, Xvr, Yvr = z['X'], z['Y'], z['Xv'], z['Yv']
mu, sd = Xr.mean(0), Xr.std(0)
if a.standardize == 'none': mu, sd = 0.0, 1.0
def T(v): return torch.tensor(v, device=DEV, dtype=torch.get_default_dtype())
X, Y, Xv, Yv = T((Xr - mu) / sd), T((Yr - mu) / sd), T((Xvr - mu) / sd), T((Yvr - mu) / sd)
n, din = X.shape; dout = Y.shape[1]
Act = dict(tanh=torch.nn.Tanh, gelu=torch.nn.GELU, relu=torch.nn.ReLU)[a.act]
if a.arch == 'mlp':
    widths = [int(w) for w in a.widths.split(',')]
    layers = []; d = din
    for w in widths: layers += [torch.nn.Linear(d, w), Act()]; d = w
    layers.append(torch.nn.Linear(d, dout))
else:
    g = a.grid; assert din == g * g and dout == g * g, 'cnn expects flattened grid fields'
    chs = [int(c) for c in a.channels.split(',')]
    layers = [torch.nn.Unflatten(1, (1, g, g))]; c = 1
    for w in chs: layers += [torch.nn.Conv2d(c, w, a.ksize, padding=a.ksize // 2, padding_mode='circular'), Act()]; c = w
    layers += [torch.nn.Conv2d(c, 1, a.ksize, padding=a.ksize // 2, padding_mode='circular'), torch.nn.Flatten(1)]
net = torch.nn.Sequential(*layers).to(DEV)
if a.init_from_kadam:
    from l63test_torch import MLP
    ref = MLP(din, widths[0], len(widths), dout, seed=a.seed)
    lins = [m for m in net if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for m, W, b in zip(lins, ref.W, ref.b): m.weight.copy_(W.to(DEV)); m.bias.copy_(b.to(DEV))
model = net
if a.ddp:
    model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local] if DEV.startswith('cuda') else None)
P = sum(q.numel() for q in net.parameters())

gen = torch.Generator(device=DEV); gen.manual_seed(a.seed + 7919 * rank)      # rank 0 keeps the single-process seed
if a.opt == 'adam':
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, betas=(a.b1, a.b2))
elif a.opt == 'kronroot':
    from kronroot import KronRoot
    opt = KronRoot(net, a.lr, b1=a.b1, b2=a.b2, beta_stats=a.beta_stats, stats_every=a.stats_every, root_every=a.root_every,
                   ns_iters=a.ns_iters, damp=a.damp, S_mode=a.S_mode, last_S=a.last_S, seed=a.seed)
else:
    rd = dict(none=None, float32=torch.float32, bfloat16=torch.bfloat16)[a.rot_dtype]
    opt = KronSign(net, a.lr, rule=a.rule, b1=a.b1, b2=a.b2, beta_stats=a.beta_stats, stats_every=a.stats_every, eig_every=a.eig_every,
                   S_mode=a.S_mode, eig=a.eig, seed=a.seed, block=a.block, last_S=a.last_S, stats=a.stats, eig_batch=bool(a.eig_batch),
                   eig_adapt=bool(a.eig_adapt), eig_max=a.eig_max, rot_dtype=rd, ddp=bool(a.ddp), reparam=bool(a.reparam))

rows = []
def emit(metric, value, step=None):
    rows.append(dict(vars(a), n=n, P=P, world=world, step=step, metric=metric, value=value))

def rmse(Xe, Ye):
    with torch.no_grad():
        r = net(Xe) - Ye; return float(torch.sqrt((r * r).sum() / Xe.shape[0]))

prof = {}
WPS = sum(q.numel() for q in net.parameters())            # weights per sample-forward (dense count; convolutions undercounted)
if a.arch == 'cnn':
    WPS = sum(m.weight.numel() * a.grid * a.grid + (m.bias.numel() if m.bias is not None else 0) for m in net if isinstance(m, torch.nn.Conv2d))
flops_total = 0.0
if a.opt in ('kronsign', 'kronroot'): opt.profile = bool(a.profile); opt.async_eig = bool(a.async_eig); opt.lowrank = bool(a.lowrank)
gn_stats = a.opt in ('kronsign', 'kronroot') and a.stats == 'gn'
if DEV.startswith('cuda'): torch.cuda.synchronize()
if a.ddp: dist.barrier()
t0 = time.time(); diverged = False; t_half = None
for step in range(1, a.steps + 1):
    if step == a.steps // 2 + 1:
        if DEV.startswith('cuda'): torch.cuda.synchronize()
        t_half = time.time()
    fac = (1 - (step - 1) / a.steps) * (min(1.0, step / a.warmup) if a.warmup > 0 else 1.0); lr = a.lr * fac
    if a.batch and a.batch < n:
        idx = torch.randint(0, n, (a.batch,), generator=gen, device=DEV); xb, yb = X[idx], Y[idx]
    else:
        xb, yb = X, Y
    if a.profile and DEV.startswith('cuda'): torch.cuda.synchronize()
    tfb = time.perf_counter()
    net.zero_grad(set_to_none=True)
    emp = gn_stats and a.S_mode == 'emp' and (step - 1) % a.stats_every == 0
    trs = gn_stats and a.stats_from_train and a.S_mode == 'mc' and (step - 1) % a.stats_every == 0
    lrk = a.opt in ('kronsign', 'kronroot') and a.lowrank
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
    if a.opt in ('kronsign', 'kronroot'): opt._last_B = xb.shape[0]
    if a.profile:
        if DEV.startswith('cuda'): torch.cuda.synchronize()
        prof['fwd_bwd'] = prof.get('fwd_bwd', 0.0) + time.perf_counter() - tfb
    tst = time.perf_counter()
    Lf = float(L.detach())
    if not math.isfinite(Lf) or Lf > 1e6:
        diverged = True; emit('diverged', 1, step); break
    if a.opt == 'adam':
        for g_ in opt.param_groups: g_['lr'] = lr
        opt.step(); flops_total += 10.0 * P
    else:
        ef = (opt.emp_from_step(loss_scale=xb.shape[0]) if lrk else opt.finish_emp(loss_scale=xb.shape[0])) if emp else tf
        sx = None
        if gn_stats and a.S_mode != 'emp' and not a.stats_from_train and (step - 1) % a.stats_every == 0:
            sx = X[torch.randperm(n, generator=gen, device=DEV)[:a.stats_n]] if a.stats_n < n else X
        opt.step(stats_x=sx, emp_factors=ef, lr=lr)
    if a.profile:
        if DEV.startswith('cuda'): torch.cuda.synchronize()
        prof['opt_step_total'] = prof.get('opt_step_total', 0.0) + time.perf_counter() - tst
    if step % a.log_every == 0 or step == a.steps:
        emit("E_train_batch", math.sqrt(2 * Lf), step)
if DEV.startswith('cuda'): torch.cuda.synchronize()
if a.ddp: dist.barrier()
t_end = time.time(); wall = t_end - t0
Et, Ev = rmse(X, Y), rmse(Xv, Yv)
flops_total = world * (flops_total + getattr(opt, 'flops', 0.0))
emit('flops_total', flops_total); emit('n_lowrank_rotations', getattr(opt, 'n_lowrank', 0)); emit('n_refresh', getattr(opt, 'n_refresh', 0))
emit('E_train_final', Et, a.steps); emit('E_val_final', Ev, a.steps); emit('wall_s', wall)
if t_half is not None and not diverged: emit('ms_per_step_steady', 1000 * (t_end - t_half) / (a.steps - a.steps // 2))
if a.opt in ('kronsign', 'kronroot') and a.eig_adapt: emit('eig_periods', [st['period'] for st in opt.st])
if a.profile:
    allp = dict(prof); allp.update({'opt.' + k: v for k, v in getattr(opt, 'times', {}).items()})
    if rank == 0: print('PROFILE ms/step: ' + '  '.join(f'{k}={1000*v/a.steps:.3f}' for k, v in allp.items()) + f'  total={1000*wall/a.steps:.3f}', flush=True)
    for k, v in allp.items(): emit('prof_ms_per_step_' + k, 1000 * v / a.steps)
if rank == 0:
    with open(a.out, 'a') as fh:
        for rw in rows: fh.write(json.dumps(rw) + '\n')
    tag = f'{a.opt}' + (f' rule={a.rule} stats={a.stats} block={a.block} last_S={a.last_S}' if a.opt in ('kronsign', 'kronroot') else '')
    print(f'{tag} lr={a.lr:g} arch={a.arch} batch={a.batch or "full"}x{world} n={n}: E_train={Et:.4g} E_val={Ev:.4g} wall={wall:.1f}s flops={flops_total:.3e}', flush=True)
if a.ddp: dist.destroy_process_group()
