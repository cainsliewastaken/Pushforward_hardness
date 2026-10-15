"""CIFAR-100 benchmark (optimizer/bench/cifar.py): CIFAR ResNet-18, the standard optimizer-paper recipe, one optimizer per run.

Recipe: ResNet-18 with the CIFAR stem (3x3 conv, stride 1, no max-pool), batch 128, 200 epochs, random crop (pad 4) + horizontal
flip, cosine learning-rate schedule with linear warmup, bf16 autocast, fp32 parameters. Test accuracy after every epoch.

Optimizers (--opt):
  sgd    SGD, Nesterov momentum 0.9, L2 weight decay --wd on all parameters (standard: lr 0.1, wd 5e-4)
  adamw  AdamW (betas 0.9 / 0.999), decoupled decay --wd on weights (not norms / biases)
  soap   SOAP (third_party/soap, unmodified): betas 0.95 / 0.95, preconditioning every 10 steps, conv dims merged
  muon   Muon (third_party/muon, SingleDeviceMuonWithAuxAdam, unmodified) on the 4-D conv kernels of layer1-4; AdamW (--lr_aux)
         on the stem conv, the classifier and the BatchNorm parameters
  kronq  KronQ (../llm/kronq_lm.py, the main optimizer's method; conv layers as KFC matrices) on every conv and the classifier;
         --beta_u 0 freezes the per-mode multipliers at 1 (ablation), --u_clamp none removes the clamp;
         AdamW (--lr_aux) on BatchNorm; decoupled decay --wd as the decay per step at peak lr (kronq_main convention)

Everything goes to --out (default $SCRATCH/kronq_bench/cifar100/<name>): config.json (arguments, git commit, uncommitted files,
torch / GPU), log.jsonl (one line per epoch), summary.json, and model.pt at the end. Nothing is written into the repository.
"""
import argparse, json, math, os, subprocess, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'llm'), os.path.join(HERE, 'third_party', 'soap'), os.path.join(HERE, 'third_party', 'muon')]

p = argparse.ArgumentParser()
p.add_argument('--opt', required=True, choices=['sgd', 'adamw', 'soap', 'muon', 'kronq', 'kronm'])
p.add_argument('--lr', type=float, required=True, help='peak learning rate of the main optimizer (Muon: of the Muon group)')
p.add_argument('--lr_aux', type=float, default=1e-3, help='AdamW learning rate for the parameters Muon / KronQ do not handle')
p.add_argument('--wd', type=float, default=None, help='weight decay (sgd: L2 5e-4; adamw/soap: decoupled 5e-2 / 1e-2; muon 0; '
               'kronq: decay per step at peak lr, 5e-5)')
p.add_argument('--epochs', type=int, default=200); p.add_argument('--batch', type=int, default=128)
p.add_argument('--warmup_epochs', type=float, default=1.0)
p.add_argument('--seed', type=int, default=1)
p.add_argument('--name', default=None); p.add_argument('--out', default=None)
p.add_argument('--data', default=os.path.expandvars('$SCRATCH/training_data/cifar100'))
p.add_argument('--device', default='cuda'); p.add_argument('--amp', type=int, default=1)
p.add_argument('--subset', type=int, default=0, help='use only this many training images (smoke tests)')
p.add_argument('--test_subset', type=int, default=0, help='evaluate on only this many test images (smoke tests)')
p.add_argument('--max_steps', type=int, default=0, help='stop after this many steps (smoke tests)')
# KronQ options (defaults = the main optimizer's)
p.add_argument('--block', type=int, default=256); p.add_argument('--stats_every', type=int, default=1)
p.add_argument('--stats_rows', type=int, default=0, help='rows (image x position pairs) per layer for the factor statistics; 0 = all'); p.add_argument('--conv_images', type=int, default=32); p.add_argument('--probe', default='fisher', choices=['fisher', 'emp'])
p.add_argument('--beta_u', type=float, default=0.01, help='KronQ: learning rate of the per-mode log-multipliers u (0 = multipliers frozen at 1)')
p.add_argument('--u_clamp', default='4.6', help="KronQ: bound on |u| (multiplier in [e^-c, e^c]); 'none' = no clamp")
p.add_argument('--km_rule', default='crit', choices=['crit', 'const'], help='kronm (TEST variant, ../llm/kronm_lm.py): per-mode momentum (crit) or one momentum 0.9 (const)')
p.add_argument('--km_beta_max', type=float, default=0.999)
p.add_argument('--u_sample_every', type=int, default=50, help='KronQ: steps between samples of the per-layer multiplier statistics')
a = p.parse_args()
# default weight decay and its source: sgd 5e-4 (standard CIFAR recipe, L2); kronq 1e-3 per step at peak lr (the method's documented
# finite-data setting, optimizer/README.md, OPTIMIZER_PLAN 2b); soap 1e-2 and muon 0 (their implementations' defaults); adamw 5e-2 (a
# common CIFAR choice, untuned)
if a.wd is None: a.wd = dict(sgd=5e-4, adamw=5e-2, soap=1e-2, muon=0.0, kronq=1e-3, kronm=1e-3)[a.opt]
U_CLAMP = float('inf') if str(a.u_clamp).lower() == 'none' else float(a.u_clamp)
a.name = a.name or f'{a.opt}_lr{a.lr:g}_wd{a.wd:g}_s{a.seed}'
a.out = a.out or os.path.expandvars(f'$SCRATCH/kronq_bench/cifar100/{a.name}')
dev = torch.device(a.device if torch.cuda.is_available() or a.device == 'cpu' else 'cpu')
torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.backends.cudnn.benchmark = True
os.makedirs(a.out, exist_ok=True)


def git(*cmd):
    try: return subprocess.run(['git', '-C', HERE, *cmd], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception: return ''


cfg = dict(vars(a), git_commit=git('rev-parse', 'HEAD'), git_uncommitted=git('status', '--short', '--', '.', '../llm', '../kronq_main.py').splitlines(),
           torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', host=os.uname().nodename,
           started=time.strftime('%Y-%m-%d %H:%M:%S'))
json.dump(cfg, open(os.path.join(a.out, 'config.json'), 'w'), indent=1)
print(json.dumps(cfg), flush=True)

# ---------------- data: whole set on the device, augmentation on the device
tr = torchvision.datasets.CIFAR100(a.data, train=True, download=True); te = torchvision.datasets.CIFAR100(a.data, train=False, download=True)
mean = torch.tensor([0.5071, 0.4865, 0.4409], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.2673, 0.2564, 0.2762], device=dev).view(1, 3, 1, 1)
Xtr = torch.tensor(tr.data, device=dev).permute(0, 3, 1, 2).contiguous(); Ytr = torch.tensor(tr.targets, device=dev)
Xte = torch.tensor(te.data, device=dev).permute(0, 3, 1, 2).contiguous(); Yte = torch.tensor(te.targets, device=dev)
if a.subset: Xtr, Ytr = Xtr[:a.subset], Ytr[:a.subset]
if a.test_subset: Xte, Yte = Xte[:a.test_subset], Yte[:a.test_subset]
norm = lambda x: (x.float() / 255 - mean) / std
gen = torch.Generator(device=dev); gen.manual_seed(a.seed)


def augment(x):
    """random crop 32 from the image zero-padded by 4, and a random horizontal flip (uint8 in, normalized float out)."""
    n = x.shape[0]; xp = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (n,), device=dev, generator=gen); j = torch.randint(0, 9, (n,), device=dev, generator=gen)
    ar = torch.arange(32, device=dev); rows = i[:, None] + ar; cols = j[:, None] + ar          # (n, 32) each
    x = xp[torch.arange(n, device=dev)[:, None, None, None], torch.arange(3, device=dev)[None, :, None, None],
           rows[:, None, :, None], cols[:, None, None, :]]                                       # (n, 3, 32, 32)
    flip = torch.rand(n, device=dev, generator=gen) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    return norm(x)


# ---------------- model: CIFAR ResNet-18
model = torchvision.models.resnet18(num_classes=100)
model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); model.maxpool = nn.Identity()
model = model.to(dev).to(memory_format=torch.channels_last)
nparam = sum(p.numel() for p in model.parameters())

# ---------------- optimizer
steps_per_epoch = Xtr.shape[0] // a.batch
total = a.epochs * steps_per_epoch if not a.max_steps else min(a.max_steps, a.epochs * steps_per_epoch)
warm = int(a.warmup_epochs * steps_per_epoch)
sched = lambda t: (t + 1) / max(1, warm) if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm)))
decay, no_decay = [p_ for p_ in model.parameters() if p_.ndim >= 2], [p_ for p_ in model.parameters() if p_.ndim < 2]
kq = None
if a.opt == 'sgd':
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, nesterov=True, weight_decay=a.wd)
elif a.opt == 'adamw':
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=a.wd), dict(params=no_decay, weight_decay=0.0)], lr=a.lr, betas=(0.9, 0.999))
elif a.opt == 'soap':
    from soap import SOAP
    opt = SOAP([dict(params=decay, weight_decay=a.wd), dict(params=no_decay, weight_decay=0.0)], lr=a.lr, betas=(0.95, 0.95),
               precondition_frequency=10, merge_dims=True)
elif a.opt == 'muon':
    from muon import SingleDeviceMuonWithAuxAdam
    hidden = [p_ for n_, p_ in model.named_parameters() if p_.ndim == 4 and n_.startswith('layer')]
    hid = {id(p_) for p_ in hidden}; aux = [p_ for p_ in model.parameters() if id(p_) not in hid]
    opt = SingleDeviceMuonWithAuxAdam([dict(params=hidden, lr=a.lr, momentum=0.95, weight_decay=a.wd, use_muon=True),
                                       dict(params=aux, lr=a.lr_aux, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0, use_muon=False)])
elif a.opt == 'kronm':
    from kronm_lm import KronMLM
    kq = KronMLM(model, lr=a.lr, lr_adam=a.lr_aux, loss='ce', probe=a.probe, exclude=(), block=a.block, stats_every=a.stats_every,
                 conv_images=a.conv_images, stats_rows=a.stats_rows, weight_decay=a.wd, seed=a.seed, rule=a.km_rule, beta_max=a.km_beta_max)
    a.beta_u = 0.0                                               # no multipliers to track
else:
    from kronq_lm import KronQLM
    kq = KronQLM(model, lr=a.lr, lr_adam=a.lr_aux, loss='ce', probe=a.probe, exclude=(), block=a.block, stats_every=a.stats_every,
                 conv_images=a.conv_images, stats_rows=a.stats_rows, weight_decay=a.wd, seed=a.seed, beta_u=a.beta_u, u_clamp=U_CLAMP)
    # per-layer multiplier statistics, sampled every u_sample_every steps (on the device; read once per epoch). "At the clamp" =
    # |u| >= 4.59 (the default bound 4.6 as stored in bf16); without a clamp the same level counts how often it WOULD have acted.
    U_REF = 4.59
    uacc = {L.name: torch.zeros(4, device=dev) for L in kq.layers}      # [at-level fraction, same at lr > 1e-8, samples, samples lr > 1e-8]
    umax = {L.name: torch.zeros((), device=dev) for L in kq.layers}
base_lrs = None if kq else [g['lr'] for g in opt.param_groups]
print(f'ResNet-18: {nparam / 1e6:.2f}M parameters; {steps_per_epoch} steps/epoch, {total} steps; '
      + (f'KronQ layers {len(kq.layers)}, AdamW tensors {sum(len(g["params"]) for g in kq.adam.param_groups)}' if kq else a.opt), flush=True)


def state_bytes():
    if kq: return kq.state_bytes()
    return sum(v.numel() * v.element_size() for st in opt.state.values() for v in st.values() if torch.is_tensor(v))


@torch.no_grad()
def evaluate():
    model.eval(); n = c = 0; ls = 0.0
    for i in range(0, Xte.shape[0], 1000):
        x, y = norm(Xte[i:i + 1000]).contiguous(memory_format=torch.channels_last), Yte[i:i + 1000]
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bool(a.amp)):
            out = model(x)
        ls += F.cross_entropy(out.float(), y, reduction='sum').item(); c += (out.argmax(1) == y).sum().item(); n += y.numel()
    model.train(); return ls / n, c / n


# ---------------- train
sync = (lambda: torch.cuda.synchronize()) if dev.type == 'cuda' else (lambda: None)
logf = open(os.path.join(a.out, 'log.jsonl'), 'a'); step = 0; t_train = 0.0; best = 0.0; diverged = False
for ep in range(a.epochs):
    perm = torch.randperm(Xtr.shape[0], device=dev, generator=gen); sync(); t0 = time.time()
    tl = torch.zeros((), device=dev); tc = torch.zeros((), device=dev); nb = 0
    for b in range(steps_per_epoch):
        if step >= total: break
        idx = perm[b * a.batch:(b + 1) * a.batch]; x = augment(Xtr[idx]).contiguous(memory_format=torch.channels_last); y = Ytr[idx]
        f = sched(step)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bool(a.amp)):
            out = model(x)
        loss = F.cross_entropy(out.float(), y)
        if kq:
            kq.backward(loss, out); kq.step(a.lr * f); kq.zero_grad()
            if a.beta_u > 0 and step % a.u_sample_every == 0:
                ok = float(a.lr * f > 1e-8)
                with torch.no_grad():
                    for L in kq.layers:
                        ua = L.u.float().abs(); fr = (ua >= U_REF).float().mean()
                        uacc[L.name] += torch.stack([fr, fr * ok, torch.ones_like(fr), torch.full_like(fr, ok)])
                        umax[L.name] = torch.maximum(umax[L.name], ua.max())
        else:
            for g, lr0 in zip(opt.param_groups, base_lrs): g['lr'] = lr0 * f
            loss.backward()
            if a.opt == 'muon':                                  # Muon's update uses .view(); channels-last conv grads are not contiguous
                for p_ in hidden: p_.grad = p_.grad.contiguous()
            opt.step(); opt.zero_grad(set_to_none=True)
        tl += loss.detach(); tc += (out.argmax(1) == y).sum(); nb += 1; step += 1
    sync(); t_train += time.time() - t0
    train_loss = (tl / max(nb, 1)).item()
    if not math.isfinite(train_loss): diverged = True
    test_loss, test_acc = evaluate(); best = max(best, test_acc)
    rec = dict(epoch=ep + 1, step=step, lr=a.lr * sched(max(step - 1, 0)), train_loss=train_loss, train_acc=(tc / max(nb * a.batch, 1)).item(),
               test_loss=test_loss, test_acc=test_acc, train_time_s=round(t_train, 1), ms_per_step=round(1e3 * t_train / max(step, 1), 2),
               peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev.type == 'cuda' else 0.0, opt_state_mb=round(state_bytes() / 2 ** 20, 1))
    if kq and a.opt == 'kronq':                                  # per-layer: clamp fraction (epoch mean of samples), max |u|, median |u|, |W|
        layers = {}; num = den_ = num_ok = 0.0
        with torch.no_grad():
            for L in kq.layers:
                acc = uacc[L.name].tolist(); n_ = L.u.numel()
                fr = acc[0] / acc[2] if acc[2] else 0.0; fr_ok = acc[1] / acc[3] if acc[3] else 0.0
                layers[L.name] = dict(clamp=round(fr, 6), clamp_lr_ok=round(fr_ok, 6), umax=round(float(umax[L.name]), 3),
                                      umed=round(float(L.u.float().abs().median()), 4), wnorm=round(float(L.mod.weight.norm()), 4))
                num += fr * n_; num_ok += fr_ok * n_; den_ += n_
                uacc[L.name].zero_(); umax[L.name].zero_()
        rec.update(clamp_frac=round(num / den_, 6), clamp_frac_lr_ok=round(num_ok / den_, 6),
                   clamp_frac_max_layer=max(v['clamp'] for v in layers.values()), umax_all=max(v['umax'] for v in layers.values()),
                   layers=layers)
    logf.write(json.dumps(rec) + '\n'); logf.flush()
    print(' '.join(f'{k} {v:.4g}' if isinstance(v, float) else f'{k} {v}' for k, v in rec.items() if k != 'layers'), flush=True)
    if diverged or step >= total: break

summ = dict(name=a.name, opt=a.opt, lr=a.lr, wd=a.wd, seed=a.seed, epochs_done=ep + 1, steps=step, final_test_acc=test_acc, best_test_acc=best,
            final_test_loss=test_loss, ms_per_step=rec['ms_per_step'], peak_mem_gb=rec['peak_mem_gb'], opt_state_mb=rec['opt_state_mb'],
            params_m=round(nparam / 1e6, 3), diverged=diverged, finished=time.strftime('%Y-%m-%d %H:%M:%S'))
if kq and a.opt == 'kronm': summ.update(km_rule=a.km_rule, km_beta_max=a.km_beta_max, stats_rows=a.stats_rows)
elif kq: summ.update(beta_u=a.beta_u, u_clamp=a.u_clamp, stats_rows=a.stats_rows)
json.dump(summ, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
torch.save(model.state_dict(), os.path.join(a.out, 'model.pt'))
print('SUMMARY ' + json.dumps(summ), flush=True)
