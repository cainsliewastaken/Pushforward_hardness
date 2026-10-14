"""Summary of the slow adaptation test (scale_slow.sh): final validation MSE and wall per (target, method), and the window
trajectory (c, r, kappa, budget, residual fast fraction, regime) of every slow-adapt run."""
import json, glob, os, sys, collections
F = sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars('$SCRATCH/l63_held_rough/runs/scale_slow')
def tag(args):
    t = []
    if '--problem coord' in args:
        t.append('coord' + (' ff8' if '--ff 8' in args else '') + (' lowpass8' if '--lowpass' in args else '') + ('' if ('--ff' in args or '--lowpass' in args) else ' raw'))
    else: t.append('field ' + ('rough' if '--p 2' in args else 'smooth'))
    if '--opt adamw' in args:
        m = 'adamw lr ' + args.split('--lr ')[1].split()[0] + (' 29000 steps' if '--steps 29000' in args else ' 5000 steps')
    else:
        m = 'kronqf fixed' if '--slow_adapt' not in args else ('kronqf slow_adapt' + (' decay none' if '--decay none' in args else ' linear decay'))
    return t[0], m
res = collections.OrderedDict(); traj = collections.OrderedDict()
for fn in sorted(glob.glob(f'{F}/res_j*.jsonl'), key=lambda s: int(s.split('_j')[1].split('.')[0])):
    rows = [json.loads(l) for l in open(fn)]
    if not rows: continue
    t, m = tag(rows[0]['args']); vals = {r['metric']: r['value'] for r in rows if r['step'] == rows[-1]['step'] or r['metric'] in ('val_final', 'wall_total')}
    vf = [r['value'] for r in rows if r['metric'] == 'val_final']; w = [r['value'] for r in rows if r['metric'] == 'wall_total']
    res[(t, m)] = (vf[0] if vf else float('nan'), w[0] if w else float('nan'), os.path.basename(fn))
    sa = [r for r in rows if r['metric'].startswith('sa_')]
    if sa:
        by = collections.OrderedDict()
        for r in sa: by.setdefault(r['step'], {})[r['metric']] = r['value']
        traj[(t, m)] = by
cur = None
for (t, m), (v, w, fn) in res.items():
    if t != cur: print(f'==== {t}'); cur = t
    print(f'   {m:40s} val {v:.4g}   wall {w:.0f} s   ({fn})')
for (t, m), by in traj.items():
    print(f'---- windows: {t} | {m}')
    print('   step    c      r     kappa  budget  resid_fastfrac  step_fastshare  smooth')
    for s, d in by.items():
        print(f"   {s:5d}  {d.get('sa_c', float('nan')):.3f}  {d.get('sa_r', float('nan')):+.3f}  {d.get('sa_kappa', float('nan')):.2f}   {d.get('sa_rem', float('nan')):.3f}   {d.get('sa_fastfrac', float('nan')):.4f}          {d.get('sa_step_fastshare', float('nan')):.3f}         {int(d.get('sa_smooth', 0))}")
