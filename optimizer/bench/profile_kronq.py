"""Where does KronQ's step time go? (optimizer/bench/profile_kronq.py). CIFAR ResNet-18, batch 128, bf16 autocast, channels-last,
random inputs of the benchmark's shapes (timing does not depend on the values). One GPU, a few minutes.
  1. clean ms/step (no added syncs): SGD; KronQ as in the benchmark; KronQ stats_every=4; KronQ empirical-Fisher probe
  2. KronQ phase breakdown: each phase wrapped with cuda.synchronize (so phases add up; the total is a little above the clean time)
  3. torch.profiler over 10 KronQ steps: top ops by self CPU and by self CUDA time, kernel launches per step, GPU busy fraction
  4. TF32 statistics (stats_tf32, default on): the batch factors of one captured step against float64, with TF32 on and off
     (on: products exact for bf16 rows, so both errors should be fp32 accumulation error)
"""
import os, sys, time, collections, functools
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'llm'))
import kronq_lm
from kronq_lm import KronQLM, _Layer
from kronq_main import _Basis

dev = torch.device('cuda'); torch.backends.cudnn.benchmark = True
B, WARM, N = 128, 30, 60


def make_model():
    torch.manual_seed(0)
    m = torchvision.models.resnet18(num_classes=100); m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity()
    return m.to(dev).to(memory_format=torch.channels_last)


X = torch.randn(B, 3, 32, 32, device=dev).contiguous(memory_format=torch.channels_last); Y = torch.randint(0, 100, (B,), device=dev)


def fwd(model):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(X)
    return out, F.cross_entropy(out.float(), Y)


def clean_ms(kind, **kw):
    model = make_model()
    if kind == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
        def step():
            out, loss = fwd(model); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    else:
        opt = KronQLM(model, lr=1e-3, lr_adam=1e-3, loss='ce', exclude=(), weight_decay=5e-5, **kw)
        def step():
            out, loss = fwd(model); opt.backward(loss, out); opt.step(1e-3); opt.zero_grad()
    for _ in range(WARM): step()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(N): step()
    torch.cuda.synchronize(); return 1e3 * (time.time() - t) / N, (opt if kind != 'sgd' else None), step


print(f'GPU {torch.cuda.get_device_name(0)}, torch {torch.__version__}, batch {B}, {N} timed steps after {WARM} warm-up', flush=True)
res = {}
for name, kind, kw in [('SGD', 'sgd', {}), ('KronQ (benchmark settings)', 'kronq', {}), ('KronQ stats_every=4', 'kronq', dict(stats_every=4)),
                       ('KronQ empirical-Fisher probe', 'kronq', dict(probe='emp')), ('KronQ stats_rows=8192', 'kronq', dict(stats_rows=8192)), ('KronQ stats_rows=8192, stats_every=4', 'kronq', dict(stats_rows=8192, stats_every=4)),
                       ('KronQ fp32 statistics (stats_tf32=0)', 'kronq', dict(stats_tf32=False)), ('KronQ stats_rows=8192, stats_tf32=0', 'kronq', dict(stats_rows=8192, stats_tf32=False))]:
    ms, opt, _ = clean_ms(kind, **kw); res[name] = ms
    print(f'[1] {name:32s} {ms:7.1f} ms/step', flush=True)
nl = len(opt.layers)
print(f'    KronQ layers: {nl} ({sum(L.conv for L in opt.layers)} conv); block dims: '
      + ', '.join(sorted({f"{L.S.nf}x{L.S.b}+{L.S.rem} | {L.A.nf}x{L.A.b}+{L.A.rem}" for L in opt.layers})), flush=True)

# ---------------- 2. phase breakdown (synchronized wrappers)
T = collections.defaultdict(float); C = collections.Counter()


def timed(name, fn):
    @functools.wraps(fn)
    def w(*a, **k):
        torch.cuda.synchronize(); t = time.time(); r = fn(*a, **k); torch.cuda.synchronize()
        T[name] += time.time() - t; C[name] += 1; return r
    return w


model = make_model()
opt = KronQLM(model, lr=1e-3, lr_adam=1e-3, loss='ce', exclude=(), weight_decay=5e-5, stats_rows=8192)
patches = [(kronq_lm, '_batch_factor', 'stats: batch factor (hooks)'), (KronQLM, '_ema', 'stats: EMA + all-reduce'),
           (_Basis, 'refresh', 'refresh: subspace iteration + QR'), (KronQLM, '_sync', 'refresh: sync/overlap'),
           (_Basis, 'transport', 'refresh: state transport'), (KronQLM, '_push', 'refresh: first-order decrease'),
           (_Layer, 'to_basis', 'rotate in'), (_Layer, 'from_basis', 'rotate out'), (_Layer, 'grad', 'gather grad'),
           (_Layer, 'apply', 'apply update'), (kronq_lm, '_update_c', 'per-mode update (compiled)'), (_Layer, 'in_rows', 'stats: unfold / rows')]
orig = {}
for obj, attr, name in patches:
    orig[(obj, attr)] = getattr(obj, attr); setattr(obj, attr, timed(name, getattr(obj, attr)))
if opt.adam is not None: opt.adam.step = timed('AdamW (BatchNorm)', opt.adam.step)


def sync_t(name, fn):
    torch.cuda.synchronize(); t = time.time(); r = fn(); torch.cuda.synchronize(); T[name] += time.time() - t; C[name] += 1; return r


def step_phased():
    out, loss = sync_t('forward (incl. stats hooks)', lambda: fwd(model))
    sync_t('backward', lambda: loss.backward(retain_graph=True))
    sync_t('probe (sample labels + backward to layer outputs)', lambda: opt._probe(out))
    for L in opt.layers: L.z = None
    opt._anchors = []
    sync_t('step total', lambda: opt.step(1e-3)); opt.zero_grad()


for _ in range(WARM): step_phased()
T.clear(); C.clear(); torch.cuda.synchronize(); t0 = time.time()
for _ in range(N): step_phased()
torch.cuda.synchronize(); wall = 1e3 * (time.time() - t0) / N
print(f'[2] phase breakdown (synchronized, {wall:.1f} ms/step wall; nested items are inside "step total" or "forward"):', flush=True)
for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
    print(f'    {k:48s} {1e3 * v / N:7.2f} ms/step  ({C[k] / N:6.1f} calls/step)', flush=True)
for (obj, attr), fn in orig.items(): setattr(obj, attr, fn)

# ---------------- 3. torch.profiler
from torch.profiler import profile, ProfilerActivity
_, opt3, step3 = clean_ms('kronq', stats_rows=8192)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10): step3()
    torch.cuda.synchronize()
ka = prof.key_averages()
cuda_total = sum(e.self_device_time_total for e in ka) / 1e3 / 10       # ms per step of GPU kernel time
launches = sum(e.count for e in ka if e.key.startswith(('cudaLaunchKernel', 'cuLaunchKernel', 'cudaLaunchKernelExC'))) / 10
print(f'[3] profiler, 10 KronQ steps: GPU kernel time {cuda_total:.1f} ms/step vs {res["KronQ (benchmark settings)"]:.1f} ms/step wall '
      f'-> GPU busy ~{100 * cuda_total / res["KronQ (benchmark settings)"]:.0f}%; kernel launches ~{launches:.0f}/step', flush=True)
print('    top 20 by self CPU time:', flush=True)
print(ka.table(sort_by='self_cpu_time_total', row_limit=20, max_name_column_width=60), flush=True)
print('    top 15 by self CUDA time:', flush=True)
print(ka.table(sort_by='self_device_time_total', row_limit=15, max_name_column_width=60), flush=True)

# ---------------- 4. TF32 statistics: factors of one captured step against float64
bf = kronq_lm._batch_factor


def factor_errors(on):
    errs = []
    def spy(B, Z):
        F_ = bf(B, Z); R = bf(B, Z.double())
        errs.append(max(float((a.double() - r).abs().max() / r.abs().max()) for a, r in zip(F_, R) if r is not None)); return F_
    model = make_model(); opt = KronQLM(model, lr=1e-3, lr_adam=1e-3, loss='ce', exclude=(), weight_decay=5e-5, stats_tf32=on)
    kronq_lm._batch_factor = spy
    try:
        out, loss = fwd(model); opt.backward(loss, out); torch.cuda.synchronize()
    finally:
        kronq_lm._batch_factor = bf
    return errs


e_on, e_off = factor_errors(True), factor_errors(False)
print(f'[4] batch factors vs float64, {len(e_on)} factors of one step: TF32 statistics max rel err {max(e_on):.1e} (median '
      f'{sorted(e_on)[len(e_on) // 2]:.1e}); fp32 statistics {max(e_off):.1e} (median {sorted(e_off)[len(e_off) // 2]:.1e})', flush=True)
