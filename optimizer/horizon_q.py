"""Is the benefit of pushing deep modes one-step or multi-step? From the q x lr grid curves (E_train every 10 steps):
best q at horizons 300 / 1000 / 3000, at each lr; and the best (q, lr) pair over all lr at each horizon."""
import json, glob, collections, numpy as np
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
C = collections.defaultdict(dict)
def arg(a, k, d=None):
    t = a.split(); return t[t.index(k) + 1] if k in t else d
for f in glob.glob(L63_ROOT + '/runs/front/p6_*.jsonl'):
    for line in open(f):
        d = json.loads(line)
        if d['metric'] != 'E_train': continue
        a = d['args']
        if d['opt'] == 'kadam': key = ('exact', arg(a, '--Delta'), float(arg(a, '--qmode')), float(arg(a, '--lr')), '1')
        else:
            if arg(a, '--H', '96') != '96' or arg(a, '--depth', '2') != '2' or arg(a, '--lr_kappa', '0') != '0': continue
            key = ('kronq2', arg(a, '--Delta'), float(arg(a, '--q')), float(arg(a, '--lr')), arg(a, '--seed', '1'))
        C[key][d['step']] = d['value']
for meth in ('exact', 'kronq2'):
    for D in ('1.0', '0.5'):
        print(f'\n{meth}, Delta {D}: best q per lr at horizons 300 / 1000 / 3000 (seed-mean for KronQ2)')
        lrs = sorted({k[3] for k in C if k[0] == meth and k[1] == D}); qs = sorted({k[2] for k in C if k[0] == meth and k[1] == D})
        overall = {}
        for lr in lrs:
            row = []
            for hz in (300, 1000, 3000):
                vals = {}
                for q in qs:
                    e = [C[k].get(hz) for k in C if k[0] == meth and k[1] == D and k[2] == q and k[3] == lr and C[k].get(hz) is not None]
                    if e: vals[q] = float(np.mean(e))
                if vals:
                    qb = min(vals, key=vals.get); row.append(f'{qb:+g} ({vals[qb]:.4f})')
                    for q, v in vals.items():
                        if hz not in overall or v < overall[hz][2]: overall[hz] = (q, lr, v)
            print(f'  lr {lr:g}: ' + ' | '.join(row))
        print('  best (q, lr) overall: ' + ' | '.join(f'h{hz}: q {overall[hz][0]:+g}, lr {overall[hz][1]:g} ({overall[hz][2]:.4f})' for hz in (300, 1000, 3000) if hz in overall))
