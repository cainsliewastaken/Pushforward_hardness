"""Equal-budget tuning protocol on train_ks.py (any problem, full batch or minibatch).
Stage 1: lr grid (x1.5 steps) per method config, seed 1, selection by validation RMSE; extend past an edge up to two steps.
Stage 2: (Adam) warmup 100 at the best; kept if better.   Stage 3: seeds 2, 3 at the final best.
Usage: python tune2.py SUITE OUT
"""
import os, sys, json, math, subprocess
suite, OUT = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(os.environ['SCRATCH'], 'l63_held_rough', 'runs', 'kadam')
os.makedirs(os.path.join(OUT, 'runs'), exist_ok=True)
G_ADAM = [1.5e-3, 3e-3, 4.5e-3, 7e-3, 1e-2, 1.5e-2, 2e-2, 3e-2]
G_KS = [3e-4, 4.5e-4, 7e-4, 1e-3, 1.5e-3, 2e-3, 3e-3]
SUITES = {
  'minibatch_adam':      dict(problem=f'--data {D}/data_n20000_D1.0_ds1.npz --widths 160,160 --steps 10000 --batch 256',
                              configs={f'adam_b2{b2}': (f'--opt adam --b2 {b2}', G_ADAM) for b2 in (0.99, 0.999)}, warmup=True),
  'minibatch_ks_sign':   dict(problem=f'--data {D}/data_n20000_D1.0_ds1.npz --widths 160,160 --steps 10000 --batch 256',
                              configs={'ks_sign': ('--opt kronsign --rule sign', G_KS)}, warmup=False),
  'minibatch_ks_adam':   dict(problem=f'--data {D}/data_n20000_D1.0_ds1.npz --widths 160,160 --steps 10000 --batch 256',
                              configs={f'ks_adam_b2{b2}': (f'--opt kronsign --rule adam --b2 {b2}', G_ADAM) for b2 in (0.99, 0.999)}, warmup=False),
}
G_F = [1.5e-4, 2.2e-4, 3.3e-4, 5e-4, 7.5e-4, 1.1e-3, 1.7e-3, 2.5e-3, 3.8e-3, 5.7e-3, 8.5e-3]
G_FS = [1e-5, 1.5e-5, 2.2e-5, 3.3e-5, 5e-5, 7.5e-5, 1.1e-4, 1.7e-4, 2.5e-4, 3.8e-4, 5.7e-4, 8.5e-4]
G_FK = [1.1e-3, 1.7e-3, 2.5e-3, 3.8e-3, 5.7e-3, 8.5e-3, 1.3e-2, 1.9e-2, 2.9e-2]          # Kronecker-Adam on the field: best was 3.8e-3 / 1.9e-2
for _p in (0, 1, 2):
    _prob = f'--data {D}/field20k_p{_p}.npz --widths 512,512 --act gelu --steps 6000 --batch 100 --standardize none'
    SUITES[f'field_p{_p}_adam'] = dict(problem=_prob, configs={f'adam_b2{b2}': (f'--opt adam --b2 {b2}', G_F) for b2 in (0.99, 0.999)}, warmup=True)
    SUITES[f'field_p{_p}_ks_sign'] = dict(problem=_prob, configs={'ks_sign': ('--opt kronsign --rule sign', G_FS)}, warmup=False)
    SUITES[f'field_p{_p}_ks_adam'] = dict(problem=_prob, configs={f'ks_adam_b2{b2}': (f'--opt kronsign --rule adam --b2 {b2}', G_F) for b2 in (0.99, 0.999)}, warmup=False)
    # 2026-10-12 additions: SOAP baseline (Shampoo statistics every step, Adam in the eigenbasis), rank-1 second moment, blocked rank-1
    SUITES[f'field_p{_p}_soap'] = dict(problem=_prob, configs={f'soap_b2{b2}': (f'--opt kronsign --rule adam --stats shampoo --stats_every 1 --b2 {b2}', G_FK) for b2 in (0.99, 0.999)}, warmup=False)
    SUITES[f'field_p{_p}_kronv'] = dict(problem=_prob, configs={'kronv_b20.999': ('--opt kronsign --rule kronv --b2 0.999', G_FK)}, warmup=False)
    SUITES[f'field_p{_p}_kronv_b64'] = dict(problem=_prob, configs={'kronv_b64_b20.999': ('--opt kronsign --rule kronv --block 64 --b2 0.999', G_FK)}, warmup=False)
    # periodic CNN on the same fields (convolution support): 3 hidden layers of 32 channels, 3x3 circular
    _cnn = f'--data {D}/field20k_p{_p}.npz --arch cnn --channels 32,32,32 --act gelu --steps 6000 --batch 100 --standardize none'
    SUITES[f'cnn_p{_p}_adam'] = dict(problem=_cnn, configs={f'adam_b2{b2}': (f'--opt adam --b2 {b2}', G_F[2:9]) for b2 in (0.99, 0.999)}, warmup=True)
    SUITES[f'cnn_p{_p}_ks_adam'] = dict(problem=_cnn, configs={'ks_adam_b20.999': ('--opt kronsign --rule adam --b2 0.999', G_F[2:9])}, warmup=False)
    SUITES[f'cnn_p{_p}_kronv'] = dict(problem=_cnn, configs={'kronv_b20.999': ('--opt kronsign --rule kronv --b2 0.999', G_F[2:9])}, warmup=False)
if os.environ.get('TUNE_SMOKE'):   # tiny CPU smoke test of the protocol: TUNE_SMOKE=<small npz>
    import re
    def _sm(pr):
        pr = re.sub(r'--data \S+', f'--data {os.environ["TUNE_SMOKE"]}', pr)
        for a_, b_ in (('--steps 6000', '--steps 30'), ('--steps 10000', '--steps 30'), ('--widths 512,512', '--widths 16,16'),
                       ('--widths 160,160', '--widths 16,16'), ('--channels 32,32,32', '--channels 4,4'), ('--batch 256', '--batch 50')):
            pr = pr.replace(a_, b_)
        return pr
    SUITES = {k: dict(v, problem=_sm(v['problem'])) for k, v in SUITES.items()}
    G_F[:] = G_F[4:6]; G_FK[:] = G_FK[2:4]; G_ADAM[:] = G_ADAM[:2]; G_KS[:] = G_KS[:2]; G_FS[:] = G_FS[:2]
S = SUITES[suite]

def run(cfg_name, cfg, lr, warmup=0, seed=1):
    tag = f'{suite}_{cfg_name}_lr{lr:.4g}_wu{warmup}_s{seed}'
    path = os.path.join(OUT, 'runs', tag + '.jsonl')
    def read():
        if not os.path.exists(path): return None
        for l in open(path):
            d = json.loads(l)
            if d['metric'] == 'E_val_final': return d['value']
    if read() is None:
        if os.path.exists(path): os.remove(path)
        subprocess.run(f'python -u {HERE}/train_ks.py {cfg} {S["problem"]} --lr {lr:.6g} --warmup {warmup} --seed {seed} --out {path}',
                       shell=True, check=False)
    v = read()
    return v if (v is not None and math.isfinite(v)) else float('inf')

def log(rec):
    with open(os.path.join(OUT, f'{suite}_selection.jsonl'), 'a') as fh: fh.write(json.dumps(rec) + '\n')
    print('SELECT', json.dumps(rec), flush=True)

res = {}
for name, (cfg, grid) in S['configs'].items():
    lrs = list(grid)
    for lr in lrs: res[(name, lr)] = run(name, cfg, lr)
    for _ in range(2):
        best = min(lrs, key=lambda x: res[(name, x)])
        if best == lrs[0]: new = lrs[0] / 1.5; lrs.insert(0, new)
        elif best == lrs[-1]: new = lrs[-1] * 1.5; lrs.append(new)
        else: break
        res[(name, new)] = run(name, cfg, new)
(name, lr), v = min(res.items(), key=lambda kv: kv[1])
cfg = S['configs'][name][0]
lrs_n = sorted(k[1] for k in res if k[0] == name); edge = lr in (lrs_n[0], lrs_n[-1])
wu = 0
if S['warmup']:
    vw = run(name, cfg, lr, warmup=100)
    if vw < v: v, wu = vw, 100
log(dict(suite=suite, stage='grid', best_cfg=name, best_lr=lr, best_warmup=wu, val_seed1=v, lr_at_grid_edge=edge,
         grid={f'{k[0]}|{k[1]:.4g}': res[k] for k in sorted(res)}))
seeds = {1: v}
for s in (2, 3): seeds[s] = run(name, cfg, lr, warmup=wu, seed=s)
vals = list(seeds.values())
log(dict(suite=suite, stage='seeds', cfg=name, lr=lr, warmup=wu, vals=seeds, mean=sum(vals) / 3, min=min(vals), max=max(vals)))
