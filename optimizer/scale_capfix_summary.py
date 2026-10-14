"""Summary of the divergence mechanism test (scale_turbfull res_j100+) and the capped-budget fix (scale_capfix): per run a label built
from the distinguishing flags (cap, rem_anchor, slow_adapt / sa_cap / kappa_max, wd, lr, target or field p), the final and best
validation, the validation at a few steps (to see degradation), and the wall."""
import json, glob, os, sys, collections
def arg(args, k): return args.split(f'--{k} ')[-1].split()[0] if f'--{k} ' in args else None
def label(a):
    parts = []
    if '--problem turb' in a: parts.append(f"turb {arg(a, 'turb_target')}")
    else: parts.append('field ' + ('rough' if arg(a, 'p') == '2' else 'smooth') + f" seed {arg(a, 'seed')}")
    if '--opt adamw' in a: parts.append(f"adamw lr {arg(a, 'lr')}"); return ' | '.join(parts)
    parts.append(f"lr {arg(a, 'lr')} budget {arg(a, 'rem')}")
    parts.append('cap on' if arg(a, 'cap') == '1' else 'cap off')
    if '--rem_anchor' in a: parts.append(f"anchor {arg(a, 'rem_anchor')}")
    if '--slow_adapt 1' in a: parts.append(('held-out CAP' if '--sa_cap 1' in a else 'held-out budget') + f" kappa {arg(a, 'sa_kappa_max') or '1.0'}")
    if '--wd' in a and arg(a, 'wd') not in (None, '0.0', '0'): parts.append(f"wd {arg(a, 'wd')}")
    return ' | '.join(parts)
def show(files, title):
    print(f'==== {title}')
    rows_all = []
    for fn in files:
        rows = [json.loads(l) for l in open(fn)]
        if not rows: continue
        a = rows[0]['args']; v = {r['step']: r['value'] for r in rows if r['metric'] == 'val'}
        vf = [r['value'] for r in rows if r['metric'] == 'val_final']; vb = [r['value'] for r in rows if r['metric'] == 'val_best']; w = [r['value'] for r in rows if r['metric'] == 'wall_total']
        steps = sorted(v); marks = [s for s in steps if s in (1000, 2000, 5000, 10000, 20000, 30000)] or steps[-3:]
        traj = ' '.join(f"{s}:{v[s]:.3g}" for s in marks)
        rows_all.append((label(a), vf[0] if vf else float('nan'), vb[0] if vb else float('nan'), w[0] if w else float('nan'), traj))
    for lab, f_, b_, w_, traj in sorted(rows_all):
        print(f"   {lab:70s} final {f_:9.4g}  best {b_:9.4g}  wall {w_:5.0f} s   [{traj}]")
S = os.path.expandvars('$SCRATCH/l63_held_rough/runs')
add = [f for f in glob.glob(f'{S}/scale_turbfull/res_j*.jsonl') if int(os.path.basename(f).split('_j')[1].split('.')[0]) >= 100]
if add: show(sorted(add), 'mechanism test (scale_turbfull, indices 100+)')
fix = glob.glob(f'{S}/scale_capfix/res_j*.jsonl')
if fix: show(sorted(fix), 'capped-budget fix (scale_capfix)')
