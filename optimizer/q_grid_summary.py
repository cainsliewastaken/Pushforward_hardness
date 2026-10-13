"""Exhaustive q x lr grid (runs/front/p6_*.jsonl): exact-basis KAdam sign (+cap) and KronQ2 (2 seeds); best q per lr with
the measured Taylor remainder (is the optimal ridge a constant-remainder contour?); architecture transfer; lr controller."""
import json, glob, re, collections, numpy as np
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
R = collections.defaultdict(dict)
def arg(a, k, d=None):
    t = a.split(); return t[t.index(k) + 1] if k in t else d
for f in glob.glob(L63_ROOT + '/runs/front/p6_*.jsonl'):
    for line in open(f):
        d = json.loads(line); a = d['args']; opt = d['opt']
        if opt == 'kadam':
            key = ('exact', arg(a, '--Delta'), '96', '2', float(arg(a, '--qmode')), float(arg(a, '--lr')), '1', '0')
        else:
            key = ('kronq2', arg(a, '--Delta'), arg(a, '--H', '96'), arg(a, '--depth', '2'), float(arg(a, '--q')), float(arg(a, '--lr')), arg(a, '--seed', '1'), arg(a, '--lr_kappa', '0'))
        m = R[key]
        if d['metric'] in ('E_train_final', 'E_val_final', 'flop_eq_final'): m[d['metric']] = d['value']
        if d['metric'] in ('rf', 'rf_l0', 'rf_l1'): m.setdefault('rf_series', []).append((d['step'], d['metric'], d['value']))
def rf_mid(m):   # mean remainder of the hidden layers over steps 500-2000
    v = [x[2] for x in m.get('rf_series', []) if 500 <= x[0] <= 2000 and x[1] in ('rf', 'rf_l0', 'rf_l1')]
    return float(np.mean(v)) if v else float('nan')
qs = [0.25, 0.0, -0.125, -0.25, -0.375, -0.5]; lrs = [5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2]
for D in ('1.0', '0.5'):
    for meth in ('exact', 'kronq2'):
        print(f'\n### {meth}, Delta {D}: training error (rows q, columns lr){"; KronQ2 = mean of 2 seeds [spread]" if meth == "kronq2" else ""}')
        print('| q \\ lr | ' + ' | '.join(f'{lr:g}' for lr in lrs) + ' |'); print('|---' * (len(lrs) + 1) + '|')
        best = {}
        for q in qs:
            cells = []
            for lr in lrs:
                ms = [R[k] for k in R if k[0] == meth and k[1] == D and k[2] == '96' and k[3] == '2' and k[4] == q and k[5] == lr and k[7] == '0' and 'E_train_final' in R[k]]
                if not ms: cells.append('-'); continue
                e = [m['E_train_final'] for m in ms]; mu = float(np.mean(e))
                cells.append(f'{mu:.4f}' + (f' [{max(e) - min(e):.4f}]' if len(e) > 1 else ''))
                if lr not in best or mu < best[lr][1]: best[lr] = (q, mu, float(np.mean([rf_mid(m) for m in ms])))
            print(f'| {q:+g} | ' + ' | '.join(cells) + ' |')
        print('best q per lr (error, mid-run remainder at the optimum): ' + '; '.join(f'lr {lr:g}: q {best[lr][0]:+g} ({best[lr][1]:.4f}, rf {best[lr][2]:.3f})' for lr in lrs if lr in best))
        # remainder map for the ridge test
        print('remainder map (mid-run mean, hidden layers): ' + ' '.join(f'q{q:+g}:' + ','.join(f'{np.nanmean([rf_mid(R[k]) for k in R if k[0]==meth and k[1]==D and k[2]=="96" and k[3]=="2" and k[4]==q and k[5]==lr and k[7]=="0"] or [np.nan]):.2g}' for lr in lrs) for q in qs))
    print(f'\n### KronQ2 architecture transfer, Delta {D}: best (q, lr) per architecture and error at each q (best lr)')
    for H, dep in (('48', '2'), ('192', '2'), ('96', '3')):
        row = []
        for q in (0.0, -0.25, -0.5):
            ms = [(k[5], R[k]['E_train_final']) for k in R if k[0] == 'kronq2' and k[1] == D and k[2] == H and k[3] == dep and k[4] == q and k[7] == '0' and 'E_train_final' in R[k]]
            if ms: lr_b, e_b = min(ms, key=lambda x: x[1]); row.append(f'q {q:+g}: {e_b:.4f} (lr {lr_b:g})')
        print(f'  H{H} d{dep}: ' + '; '.join(row))
    print(f'\n### KronQ2 per-layer lr controller, Delta {D} (base lr 3e-3)')
    for k in sorted(k for k in R if k[0] == 'kronq2' and k[1] == D and k[7] != '0'):
        m = R[k]; print(f'  q {k[4]:+g} kappa {k[7]}: train {m.get("E_train_final", float("nan")):.4f}  val {m.get("E_val_final", float("nan")):.4f}')
