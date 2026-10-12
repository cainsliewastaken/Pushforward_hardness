"""KronRoot (Newton-Schulz Kronecker whitening + Adam in whitened coordinates) on the rough field problem, p = 1, seed 1.
Stage 1: profiles (1500 steps): Adam, KronSign reparam+identity (current best), KronRoot root_every 10 and 50.
Stage 2: KronRoot lr grid (6000 steps, root_every 50), edge extension; best lr also with root_every 10.
Stage 3: best KronRoot at Adam's FLOP budget and at Adam's wall-time budget (steady ms/step).
Resumable.  Usage: python kr_sweep.py OUT_DIR
"""
import os, sys, json, subprocess
OUT = sys.argv[1]; HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.environ['SCRATCH'], 'l63_held_rough', 'runs', 'kadam')
os.makedirs(os.path.join(OUT, 'runs'), exist_ok=True)
PROB = f'--data {D}/field20k_p1.npz --widths 512,512 --act gelu --batch 100 --standardize none --seed 1'
ADAM = '--opt adam --lr 3.8e-3 --b2 0.99'
KSRP = '--opt kronsign --rule adam --lr 1.9125e-2 --b2 0.999 --reparam 1 --last_S identity'
KR = '--opt kronroot --last_S identity --b2 0.999'

def run(tag, args):
    path = os.path.join(OUT, 'runs', tag + '.jsonl')
    def read():
        r = {}
        if os.path.exists(path):
            for l in open(path):
                d = json.loads(l); r[d['metric']] = d['value']
        return r
    if 'E_val_final' not in read():
        if os.path.exists(path): os.remove(path)
        subprocess.run(f'python -u {HERE}/train_ks.py {PROB} {args} --out {path}', shell=True, check=False)
    r = read()
    rec = dict(tag=tag, args=args, E_val=r.get('E_val_final', float('inf')), E_train=r.get('E_train_final', float('nan')),
               flops=r.get('flops_total', float('nan')), wall=r.get('wall_s', float('nan')), ms_step=r.get('ms_per_step_steady', float('nan')),
               prof={k[17:]: round(v, 3) for k, v in r.items() if k.startswith('prof_ms_per_step_')})
    with open(os.path.join(OUT, 'summary.jsonl'), 'a') as fh: fh.write(json.dumps(rec) + '\n')
    print('RESULT', json.dumps(rec), flush=True)
    return rec

# stage 1: profiles
for tag, args in (('prof_adam', ADAM), ('prof_ks_rp_id', KSRP), ('prof_kr_r10', f'{KR} --lr 3e-3 --root_every 10'),
                  ('prof_kr_r50', f'{KR} --lr 3e-3 --root_every 50')):
    run(tag, f'{args} --steps 1500 --profile 1')
# stage 2: lr grid
A = run('adam_s6000', f'{ADAM} --steps 6000')
grid = [1e-3, 2e-3, 3e-3, 5e-3, 8e-3]; res = {}
for lr in grid: res[lr] = run(f'kr_r50_lr{lr:g}_s6000', f'{KR} --lr {lr:g} --root_every 50 --steps 6000')['E_val']
for _ in range(2):
    best = min(grid, key=lambda x: res[x])
    if best == grid[0]: new = grid[0] / 1.6; grid.insert(0, new)
    elif best == grid[-1]: new = grid[-1] * 1.6; grid.append(new)
    else: break
    res[new] = run(f'kr_r50_lr{new:.4g}_s6000', f'{KR} --lr {new:.6g} --root_every 50 --steps 6000')['E_val']
lr = min(res, key=res.get)
B50 = run(f'kr_r50_lr{lr:.4g}_s6000', f'{KR} --lr {lr:.6g} --root_every 50 --steps 6000')
B10 = run(f'kr_r10_lr{lr:.4g}_s6000', f'{KR} --lr {lr:.6g} --root_every 10 --steps 6000')
print('BEST', json.dumps(dict(lr=lr, at_edge=lr in (grid[0], grid[-1]), r50=B50['E_val'], r10=B10['E_val'])), flush=True)
# stage 3: equal budgets for both refresh periods
for name, Bx, re in (('r50', B50, 50), ('r10', B10, 10)):
    n_flop = max(50, int(round(A['flops'] / (Bx['flops'] / 6000))))
    n_time = max(50, int(round(1000 * A['wall'] / Bx['ms_step'])))
    run(f'kr_{name}_eqflop_s{n_flop}', f'{KR} --lr {lr:.6g} --root_every {re} --steps {n_flop}')
    run(f'kr_{name}_eqtime_s{n_time}', f'{KR} --lr {lr:.6g} --root_every {re} --steps {n_time}')
print('KR SWEEP DONE', flush=True)
