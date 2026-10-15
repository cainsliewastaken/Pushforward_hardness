"""CPU checks for KronQLM (seconds each; torch.compile disabled so they stay short):
  1. equivalence: on an MLP with squared loss KronQLM reproduces kronq_main.KronQ step by step (float32), with and without decay;
  2. a tiny GPT with cross entropy (Fisher probe, empirical-Fisher option, bf16 autocast) trains, against AdamW for reference;
  3. two-process DDP (gloo), two micro-batches per step with statistics from one: parameters, bases and refresh periods stay
     identical on both ranks;
  4. the streamed statistics equal the all-at-once ones, and micro-batch capture takes them from exactly the captured micro-batch;
  5. Conv2d: a 1x1-conv network takes the same KronQ steps as the equivalent Linear network; a small 3x3 CNN trains (CE probe).
Run: python test_kronq_lm.py [1|2|3|4|5 ...]  (all by default)."""
import os, sys
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import contextlib, copy, math, time, tempfile, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kronq_lm import KronQLM
from gpt import GPT, GPTConfig
from kronq_main import KronQ


def rademacher(seed):
    g = torch.Generator().manual_seed(seed)
    return lambda shape: torch.randint(0, 2, shape, generator=g).to(torch.get_default_dtype()) * 2 - 1


def test_equivalence(steps=60, exact=8, wd=0.0):
    """The two forwards differ by rounding only (nn.Linear computes a W^T + b, kronq_main [a, 1] [W | b]^T). Through the first
    `exact` steps (two basis refreshes) every quantity agrees to rounding and the refresh schedules are identical; later a
    near-zero momentum entry (stored in bf16) can take the other sign in one run and the two runs separate slowly, as two GPU
    runs of kronq_main itself do. So: strict agreement over the first steps, agreement of the loss to 3 % after `steps`."""
    torch.manual_seed(0)
    din, w, dout, B, lr = 16, 80, 8, 32, 3e-3                       # 81 = 2 blocks of 32 + a remainder of 17
    net = nn.Sequential(nn.Linear(din, w), nn.GELU(), nn.Linear(w, w), nn.GELU(), nn.Linear(w, dout))
    lins = [m for m in net if isinstance(m, nn.Linear)]
    X = torch.randn(512, din); Y = torch.sin(X @ torch.randn(din, dout))
    ref = KronQ(lins, F.gelu, dout, rademacher(1), lr, 'cpu', block=32, refresh=5, refresh_min=2, weight_decay=wd)
    new = KronQLM(net, lr=lr, loss='mse', draw_probe=rademacher(1), exclude=(), block=32, refresh=5, refresh_min=2,
                  weight_decay=wd, lam_scale=2.0 / dout)
    worst, wdiff, sched_ok = 0.0, 0.0, True
    for s in range(steps):
        i = (s * B) % 512; xb, yb = X[i:i + B], Y[i:i + B]
        for W in ref.Wt: W.grad = None
        o1 = ref.forward(xb, keep=True); l1 = F.mse_loss(o1, yb); l1.backward(retain_graph=True); ref.step(lr, o1)
        new.zero_grad(); o2 = net(xb); l2 = F.mse_loss(o2, yb); new.backward(l2, o2); new.step(lr)
        if s < exact:
            worst = max(worst, abs(l1.item() - l2.item()) / l1.item())
            wdiff = max([wdiff] + [float((W.data[:, :-1] - m.weight.data).abs().max()) for W, m in zip(ref.Wt, lins)])
            sched_ok &= ref.next_ref == [L.next_ref for L in new.layers]
    final = abs(l1.item() - l2.item()) / l1.item()
    ok = worst < 1e-6 and wdiff < 1e-5 and sched_ok and final < 0.03
    print(f'[1] equivalence wd={wd}: first {exact} steps: max rel loss diff {worst:.1e}, max weight diff {wdiff:.1e}, same refresh '
          f'schedule {sched_ok}; after {steps}: loss {l1.item():.5g} vs {l2.item():.5g} ({final:.1%}) -> {"PASS" if ok else "FAIL"}')
    return ok


def markov_batch(g, B, T, V, perm):
    x = torch.empty(B, T + 1, dtype=torch.long); x[:, 0] = torch.randint(0, V, (B,), generator=g)
    for t in range(T):
        rnd = torch.randint(0, V, (B,), generator=g); keep = torch.rand(B, generator=g) < 0.9
        x[:, t + 1] = torch.where(keep, perm[x[:, t]], rnd)
    return x[:, :-1], x[:, 1:]


def train_gpt(opt_name, steps=150, probe='fisher', amp=False, seed=0):
    torch.manual_seed(seed); V, T, B = 64, 32, 16
    model = GPT(GPTConfig(vocab_size=V, block_size=T, n_layer=2, n_head=4, n_embd=64))
    perm = torch.randperm(V, generator=torch.Generator().manual_seed(123)); g = torch.Generator().manual_seed(7)
    lr = 3e-3; sched = lambda s: lr * (1 - s / steps)
    if opt_name == 'kronq':
        opt = KronQLM(model, lr=lr, lr_adam=lr, loss='ce', probe=probe, block=64, refresh=10, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    t0 = time.time(); losses = []
    for s in range(steps):
        x, y = markov_batch(g, B, T, V, perm)
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
            logits = model(x); loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        if opt_name == 'kronq':
            opt.zero_grad(); opt.backward(loss, logits); opt.step(sched(s))
        else:
            for gr in opt.param_groups: gr['lr'] = sched(s)
            opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    tail = sum(losses[-10:]) / 10
    extra = ''
    if opt_name == 'kronq':
        us = opt.u_stats(); extra = (f', KronQ layers {len(opt.layers)}, AdamW tensors {sum(len(gr["params"]) for gr in opt.adam.param_groups)}, '
                                     f'max at-upper {max(v["at_upper"] for v in us.values()):.3f}, state {opt.state_bytes() / 1e6:.2f} MB')
    print(f'    {opt_name:5s} probe={probe:6s} amp={amp!s:5s}: loss {losses[0]:.3f} -> {tail:.3f} (floor ~0.75, uniform {math.log(V):.2f}), '
          f'{(time.time() - t0) / steps * 1e3:.0f} ms/step{extra}')
    return tail


def test_gpt():
    print('[2] tiny GPT, cross entropy, 150 steps (Markov chain data):')
    a = train_gpt('adamw'); k = train_gpt('kronq'); e = train_gpt('kronq', probe='emp'); b = train_gpt('kronq', amp=True, steps=40)
    ok = all(math.isfinite(v) for v in (k, e, b)) and k < 2.0 and e < 2.5
    print(f'    -> {"PASS" if ok else "FAIL"} (pass = finite and KronQ below 2.0; AdamW for reference {a:.3f})')
    return ok


def _ddp_worker(rank, world, initfile, steps, q):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    dist.init_process_group('gloo', init_method=f'file://{initfile}', rank=rank, world_size=world)
    torch.manual_seed(0); V, T, B = 64, 32, 8
    model = DDP(GPT(GPTConfig(vocab_size=V, block_size=T, n_layer=2, n_head=4, n_embd=64)))
    perm = torch.randperm(V, generator=torch.Generator().manual_seed(123)); g = torch.Generator().manual_seed(100 + rank)  # different data per rank
    opt = KronQLM(model, lr=3e-3, loss='ce', block=64, refresh=3, refresh_min=2, weight_decay=1e-4)
    for s in range(steps):                                      # two micro-batches per step, statistics from the first
        opt.zero_grad()
        for i in range(2):
            x, y = markov_batch(g, B, T, V, perm)
            ctx = model.no_sync() if i == 0 else contextlib.nullcontext()
            with ctx:
                with opt.capture(i == 0):
                    logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten()) / 2
                opt.backward(loss, logits)
        opt.step()
    flat = torch.cat([p.detach().flatten() for p in model.parameters()] + [L.S.U.flatten() for L in opt.layers] + [L.A.U.flatten() for L in opt.layers])
    per = torch.tensor([float(L.period) for L in opt.layers])
    fl = [torch.zeros_like(flat) for _ in range(world)]; pl = [torch.zeros_like(per) for _ in range(world)]
    dist.all_gather(fl, flat); dist.all_gather(pl, per)
    if rank == 0: q.put((float((fl[0] - fl[1]).abs().max()), float((pl[0] - pl[1]).abs().max()), loss.item(), per.tolist()))
    dist.destroy_process_group()


def _factors(opt):
    return [(L.nA[0], L.nS[0]) for L in opt.layers]


def test_streaming_and_microbatch():
    """(a) the statistics consumed by hooks as the probe's gradients pass (stream=True) equal those from all layer-output
    gradients at once (stream=False); (b) with 4 micro-batches and capture on the first, the statistics equal those of a run on
    that micro-batch alone, and training with micro-batches runs."""
    V, T, B = 64, 32, 8; cfg = GPTConfig(vocab_size=V, block_size=T, n_layer=2, n_head=4, n_embd=64)
    perm = torch.randperm(V, generator=torch.Generator().manual_seed(123))
    torch.manual_seed(0); m1 = GPT(cfg); m2 = copy.deepcopy(m1)
    o1 = KronQLM(m1, lr=3e-3, block=64, stream=True); o2 = KronQLM(m2, lr=3e-3, block=64, stream=False)
    x, y = markov_batch(torch.Generator().manual_seed(5), B, T, V, perm)
    for m, o in ((m1, o1), (m2, o2)):
        lg = m(x); o.backward(F.cross_entropy(lg.flatten(0, 1), y.flatten()), lg)
    da = max(float((a1 - a2).abs().max() / a2.abs().max()) for (a1, s1), (a2, s2) in zip(_factors(o1), _factors(o2)))
    ds = max(float((s1 - s2).abs().max() / s2.abs().max()) for (a1, s1), (a2, s2) in zip(_factors(o1), _factors(o2)))
    ok_a = da < 1e-6 and ds < 1e-6
    print(f'[4a] streamed vs all-at-once statistics: max rel diff A {da:.1e}, S {ds:.1e} -> {"PASS" if ok_a else "FAIL"}')

    torch.manual_seed(0); m3 = GPT(cfg); m4 = copy.deepcopy(m3)
    o3 = KronQLM(m3, lr=3e-3, block=64); o4 = KronQLM(m4, lr=3e-3, block=64)
    g = torch.Generator().manual_seed(9); mbs = [markov_batch(g, B, T, V, perm) for _ in range(4)]
    for i, (xi, yi) in enumerate(mbs):                          # o3: 4 micro-batches, statistics from the first
        with o3.capture(i == 0):
            lg = m3(xi)
        o3.backward(F.cross_entropy(lg.flatten(0, 1), yi.flatten()) / 4, lg)
    lg = m4(mbs[0][0]); o4.backward(F.cross_entropy(lg.flatten(0, 1), mbs[0][1].flatten()), lg)   # o4: that micro-batch alone
    dm = max(max(float((p - q).abs().max()) for p, q in zip(f3, f4)) for f3, f4 in zip(_factors(o3), _factors(o4)))
    o3.step(); losses = []
    for s in range(60):                                         # train on: 4 micro-batches per step
        o3.zero_grad(); tot = 0.0
        for i in range(4):
            xi, yi = markov_batch(g, B, T, V, perm)
            with o3.capture(i == 0):
                lg = m3(xi)
            loss = F.cross_entropy(lg.flatten(0, 1), yi.flatten()) / 4; o3.backward(loss, lg); tot += loss.item()
        o3.step(3e-3 * (1 - s / 60)); losses.append(tot)
    ok_b = dm == 0.0 and losses[-1] < 0.5 * losses[0]
    print(f'[4b] micro-batches: statistics equal to the captured micro-batch alone (max diff {dm:.1e}); 60 steps x 4 micro-batches: '
          f'loss {losses[0]:.3f} -> {sum(losses[-5:]) / 5:.3f} -> {"PASS" if ok_b else "FAIL"}')
    return ok_a and ok_b


def test_ddp(steps=12):
    import torch.multiprocessing as mp
    ctx = mp.get_context('spawn'); q = ctx.Queue(); d = tempfile.mkdtemp(); f = os.path.join(d, 'init')
    ps = [ctx.Process(target=_ddp_worker, args=(r, 2, f, steps, q)) for r in range(2)]
    for p in ps: p.start()
    for p in ps: p.join(timeout=300)
    pdiff, perdiff, loss, per = q.get(timeout=10)
    ok = pdiff == 0.0 and perdiff == 0.0 and all(p.exitcode == 0 for p in ps)
    print(f'[3] DDP 2 ranks, {steps} steps: max param/basis diff across ranks {pdiff:.1e}, period diff {perdiff}, periods {per}, '
          f'loss {loss:.3f} -> {"PASS" if ok else "FAIL"}')
    return ok


def test_conv(steps=12):
    """(a) a 1x1-conv network on (B, C, 1, 1) inputs is the same function as a Linear network, and KronQ must take the same
    steps on both (squared loss, same probe draws); (b) a small 3x3 CNN (zero and circular padding, strided conv, BatchNorm on
    AdamW) trains with the cross-entropy probe."""
    torch.manual_seed(0); din, w, dout, B, lr = 12, 40, 6, 16, 3e-3
    lin = nn.Sequential(nn.Linear(din, w), nn.GELU(), nn.Linear(w, w), nn.GELU(), nn.Linear(w, dout))
    conv = nn.Sequential(*[nn.Conv2d(m.in_features, m.out_features, 1) if isinstance(m, nn.Linear) else nn.GELU() for m in lin])
    with torch.no_grad():
        for a, b in zip(lin, conv):
            if isinstance(a, nn.Linear): b.weight.copy_(a.weight[:, :, None, None]); b.bias.copy_(a.bias)
    X = torch.randn(256, din); Y = torch.sin(X @ torch.randn(din, dout))
    o1 = KronQLM(lin, lr=lr, loss='mse', draw_probe=rademacher(3), exclude=(), block=16, refresh=4, refresh_min=2)
    o2 = KronQLM(conv, lr=lr, loss='mse', draw_probe=rademacher(3), exclude=(), block=16, refresh=4, refresh_min=2)
    # the two forwards round differently; through the first `exact` steps everything agrees to rounding (traced 2026-10-14:
    # factors ~1e-7, gradients ~1e-8, weights <= 1e-5 through step 9); later a near-zero bf16 momentum entry can take the other
    # sign (step 10 there: one entry, one sign step lr e^u ~ 2e-3), as in test 1. So: strict first steps, loss after.
    exact, worst, wdx = 8, 0.0, 0.0
    for s in range(steps):
        i = (s * B) % 256; xb, yb = X[i:i + B], Y[i:i + B]
        l1 = F.mse_loss(out1 := lin(xb), yb); o1.backward(l1, out1); o1.step(); o1.zero_grad()
        out2 = conv(xb[:, :, None, None]); l2 = F.mse_loss(out2[:, :, 0, 0], yb); o2.backward(l2, out2); o2.step(); o2.zero_grad()
        if s < exact:
            worst = max(worst, abs(l1.item() - l2.item()) / l1.item())
            with torch.no_grad():
                wdx = max([wdx] + [float((a.weight - b.weight[:, :, 0, 0]).abs().max()) for a, b in zip(lin, conv) if isinstance(a, nn.Linear)])
    final = abs(l1.item() - l2.item()) / l1.item()
    ok_a = wdx < 1e-4 and worst < 1e-5 and final < 0.01
    print(f'[5a] 1x1 conv vs Linear: first {exact} steps: max weight diff {wdx:.1e}, max rel loss diff {worst:.1e}; after {steps}: '
          f'loss diff {final:.1e} -> {"PASS" if ok_a else "FAIL"}')

    torch.manual_seed(1); nc = 5
    cnn = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
                        nn.Conv2d(16, 32, 3, stride=2, padding=1, padding_mode='circular'), nn.ReLU(),
                        nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, nc))
    g = torch.Generator().manual_seed(4); protos = torch.randn(nc, 3, 16, 16, generator=g)
    opt = KronQLM(cnn, lr=2e-3, lr_adam=2e-3, loss='ce', block=64, refresh=5, conv_images=8, weight_decay=1e-4)
    kinds = sorted({('conv' if L.conv else 'linear') for L in opt.layers})
    losses = []
    for s in range(60):
        yb = torch.randint(0, nc, (32,), generator=g); xb = protos[yb] + 0.8 * torch.randn(32, 3, 16, 16, generator=g)
        out = cnn(xb); loss = F.cross_entropy(out, yb); opt.backward(loss, out); opt.step(2e-3 * (1 - s / 60)); opt.zero_grad()
        losses.append(loss.item())
    ok_b = len(opt.layers) == 4 and sum(losses[-5:]) / 5 < 0.5 * losses[0]
    print(f'[5b] 3x3 CNN, cross entropy: KronQ layers {len(opt.layers)} ({kinds}), AdamW tensors '
          f'{sum(len(gr["params"]) for gr in opt.adam.param_groups)}; loss {losses[0]:.3f} -> {sum(losses[-5:]) / 5:.3f} -> {"PASS" if ok_b else "FAIL"}')
    return ok_a and ok_b


if __name__ == '__main__':
    which = sys.argv[1:] or ['1', '2', '3', '4', '5']
    res = []
    if '1' in which: res += [test_equivalence(wd=0.0), test_equivalence(wd=1e-3)]
    if '2' in which: res.append(test_gpt())
    if '3' in which: res.append(test_ddp())
    if '4' in which: res.append(test_streaming_and_microbatch())
    if '5' in which: res.append(test_conv())
    print('ALL PASS' if all(res) else 'SOME FAILED')
