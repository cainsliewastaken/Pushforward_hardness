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
for _p in (0, 1, 2):
    _prob = f'--data {D}/field20k_p{_p}.npz --widths 512,512 --act gelu --steps 6000 --batch 100 --standardize none'
    SUITES[f'field_p{_p}_adam'] = dict(problem=_prob, configs={f'adam_b2{b2}': (f'--opt adam --b2 {b2}', G_F) for b2 in (0.99, 0.999)}, warmup=True)
    SUITES[f'field_p{_p}_ks_sign'] = dict(problem=_prob, configs={'ks_sign': ('--opt kronsign --rule sign', G_FS)}, warmup=False)
    SUITES[f'field_p{_p}_ks_adam'] = dict(problem=_prob, configs={f'ks_adam_b2{b2}': (f'--opt kronsign --rule adam --b2 {b2}', G_F) for b2 in (0.99, 0.999)}, warmup=False)
if os.environ.get('TUNE_SMOKE'):
    SUITES = {k: dict(v, problem=f'--data {os.environ["TUNE_SMOKE"]} --widths 16,16 --steps 30 --batch 64') for k, v in SUITES.items()}
S = SUITES[suite]

def run(cfg_name, cfg, lr, warmup=0, seed=1):
    tag = f'{suite}_{cfg_name}_lr{lr:.4g}_wu{warmup}_s{seed}'
    path = os.path.join(OUT, 'runs', tag + '.jsonl')
    def read():
        for l in open(path):
            d = json.loads(l)
            if d['metric'] == 'E_val_final': return d['value']
    if not (os.path.exists(path) and read() is not None):
        subprocess.run(f'python -u {HERE}/train_ks.py {cfg} {S["problem"]} --lr {lr:.6g} --warmup {warmup} --seed {seed} --out {path}',
                       shell=True, check=False)
    v = read() if os.path.exists(path) else None
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
