"""Summarise runs/front: per target and base method, each (rho, beta_mode) at its best lr (by train E), with val E,
block-1 spectrum at the last refresh (lift), residual energy below lam_front/10, and the front momentum."""
import json, glob, collections, sys, math
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
R = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(L63_ROOT + '/runs/front/p[12]_w*.jsonl'):
    for line in open(f):
        d = json.loads(line); a = d['args'].split()
        g = lambda k: a[a.index(k) + 1]
        key = (g('--Delta'), int(g('--signmode')), g('--rho'), g('--beta_mode'), float(g('--lr')))
        R[key][d['metric']].append((d['step'], d['value']))
def last(m, k): return m[k][-1][1] if m[k] else float('nan')
groups = collections.defaultdict(list)
for k, m in R.items(): groups[k[:4]].append((k[4], m))
for D in ('0.5', '1.0'):
    for sm in (1, 0):
        print(f'\n### Delta {D}, {"sign per mode" if sm else "RMS per mode"}  (best lr by train E; lrs tried in brackets)')
        print('| rho | beta | lr | E_train | E_val | E_train at other lrs | lam_r1 / r100 / r1000 (block 1, end) | resid. energy < lam_f/10 | beta at front (end) |')
        print('|---|---|---|---|---|---|---|---|---|')
        for rho, bm in (('inf','fixed'),('3','fixed'),('30','fixed'),('300','fixed'),('inf','front'),('30','front'),('inf','mode'),('30','mode')):
            runs = sorted(groups.get((D, sm, rho, bm), []), key=lambda x: x[0])
            if not runs: continue
            ok = [(lr, m) for lr, m in runs if m['E_train_final']]
            if not ok: print(f'| {rho} | {bm} | - | diverged |'); continue
            lr, m = min(ok, key=lambda x: last(x[1], 'E_train_final'))
            others = ' '.join(f'{l:g}:{last(mm,"E_train_final"):.3g}' for l, mm in runs if l != lr)
            print(f'| {rho} | {bm} | {lr:g} | {last(m,"E_train_final"):.4f} | {last(m,"E_val_final"):.4f} | {others} | '
                  f'{last(m,"lam_r1"):.3g} / {last(m,"lam_r100"):.3g} / {last(m,"lam_r1000"):.3g} | {last(m,"res_below_f10"):.2f} | {last(m,"beta_front"):.3f} |')
