"""Equal-budget tuning protocol for one method on the three benchmark problems.

Stage 1: fine lr grid (x1.5 steps; Adam also crossed with beta2), seed 1, select by validation RMSE.
         If the best lr is at a grid edge, extend the grid by up to two x1.5 steps past that edge.
Stage 2: (Adam only) warmup 100 steps at the stage-1 best; kept if better.
Stage 3: seeds 2 and 3 at the final best.
Every run writes its own JSONL under OUT/runs/; selections are logged to OUT/<method>_selection.jsonl.
Usage: python tune.py METHOD OUT      (METHOD in adam, kron, muon)
"""
import os, sys, json, math, subprocess

method, OUT = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.environ['SCRATCH'], 'l63_held_rough', 'runs', 'kadam')
os.makedirs(os.path.join(OUT, 'runs'), exist_ok=True)

PROBLEMS = {
    'P1_smooth_n2000_3k': f'--Delta 0.5 --n 2000 --steps 3000 --data_cache {D}/data_n2000_D0.5_ds1.npz',
    'P2_rough_n20000_3k': f'--Delta 1.0 --n 20000 --steps 3000 --data_cache {D}/data_n20000_D1.0_ds1.npz',
    'P3_rough_n20000_10k': f'--Delta 1.0 --n 20000 --steps 10000 --data_cache {D}/data_n20000_D1.0_ds1.npz',
}
COMMON = '--depth 2 --nval 2000 --H 160'
if os.environ.get('TUNE_SMOKE'):
    PROBLEMS = {'P1_smoke': '--Delta 0.5 --n 100 --steps 20'}; COMMON = '--depth 2 --nval 50 --H 16'
LR_ADAM_MUON = [1e-2, 1.5e-2, 2e-2, 3e-2, 4.5e-2, 7e-2]
LR_KRON = [1e-3, 1.5e-3, 2e-3, 3e-3, 4.5e-3, 7e-3]

def method_args(prob):
    if method == 'adam': return '--opt adam'
    if method == 'muon': return '--opt muon'
    sub = 500 if prob.startswith('P1') else 2000
    return f'--opt kron --kron_stats_every 10 --kron_subsample {sub} --kron_S mc'

def run(prob, lr, b2=0.999, warmup=0, seed=1):
    tag = f'{method}_{prob}_lr{lr:g}_b2{b2:g}_wu{warmup}_s{seed}'
    path = os.path.join(OUT, 'runs', tag + '.jsonl')
    def read():
        for l in open(path):
            d = json.loads(l)
            if d['metric'] == 'E_val_final': return d['value'], d
        return None
    if os.path.exists(path):
        got = read()
        if got: return got[0]
    cmd = (f'python -u {HERE}/kadam.py {method_args(prob)} --lr {lr} --b2 {b2} --warmup {warmup} --seed {seed} '
           f'{PROBLEMS[prob]} {COMMON} --out {path}')
    subprocess.run(cmd, shell=True, check=False)
    got = read() if os.path.exists(path) else None
    v = got[0] if got else float('inf')
    return v if math.isfinite(v) else float('inf')

def log(rec):
    with open(os.path.join(OUT, f'{method}_selection.jsonl'), 'a') as fh: fh.write(json.dumps(rec) + '\n')
    print('SELECT', json.dumps(rec), flush=True)

for prob in PROBLEMS:
    grid = list(LR_KRON if method == 'kron' else LR_ADAM_MUON)
    b2s = [0.95, 0.99, 0.999] if method == 'adam' else [0.999]
    res = {}
    for b2 in b2s:
        lrs = list(grid)
        for lr in lrs: res[(lr, b2)] = run(prob, lr, b2)
        for _ in range(2):                                   # extend past an edge, at most two steps
            best_lr = min(lrs, key=lambda x: res[(x, b2)])
            if best_lr == lrs[0]: new = lrs[0] / 1.5; lrs.insert(0, new)
            elif best_lr == lrs[-1]: new = lrs[-1] * 1.5; lrs.append(new)
            else: break
            res[(new, b2)] = run(prob, new, b2)
    (lr, b2), v = min(res.items(), key=lambda kv: kv[1])
    lrs_b2 = sorted(k[0] for k in res if k[1] == b2); at_edge = lr in (lrs_b2[0], lrs_b2[-1])
    wu = 0
    if method == 'adam':
        vw = run(prob, lr, b2, warmup=100)
        if vw < v: v, wu = vw, 100
    log(dict(method=method, problem=prob, stage='grid', best_lr=lr, best_b2=b2, best_warmup=wu, val_seed1=v, lr_at_grid_edge=at_edge,
             grid={f'{k[0]:g}|{k[1]:g}': res[k] for k in sorted(res)}))
    seeds = {1: v}
    for s in (2, 3): seeds[s] = run(prob, lr, b2, warmup=wu, seed=s)
    vals = list(seeds.values())
    log(dict(method=method, problem=prob, stage='seeds', lr=lr, b2=b2, warmup=wu, vals=seeds,
             mean=sum(vals) / 3, min=min(vals), max=max(vals)))
