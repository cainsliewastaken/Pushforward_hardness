"""Summarise the equal-FLOP scaling study (scale_jobs_flops.py runs; res_s*.jsonl in $L63_ROOT/runs/scale_flops).
Every run had the same FLOP budget (AdamW x BUDGET steps at its width). Per width and target: the final validation MSE at
the best lr per method, with its step count, optimizer-state bytes per parameter, peak GPU memory and wall; grouped by
state-memory class so that comparisons are made at comparable memory. Also lists failed runs and the measured
q_true / q_Kronecker of the line-search runs."""
import os, re, glob, json, collections, csv, sys
import numpy as np
ROOT = os.environ.get('L63_ROOT', os.path.expandvars('$SCRATCH/l63_held_rough')); F = f'{ROOT}/runs/scale_flops'
recs = [json.loads(l) for f in sorted(glob.glob(f'{F}/res_s*.jsonl')) for l in open(f) if l.strip()]
runs = collections.defaultdict(dict)
for r in recs:
    d = runs[r['args']]; d.setdefault('curve', {}); d.setdefault('q', collections.defaultdict(list))
    if r['metric'] in ('val', 'wall'): d['curve'].setdefault(r['step'], {})[r['metric']] = r['value']
    elif r['metric'].startswith('gnls_qratio'): d['q'][r['metric']].append(r['value'])
    else: d[r['metric']] = r['value']
    d['opt'], d['width'], d['p'], d['git'] = r['opt'], r['width'], r['p'], r['git']
def variant(a):
    o = re.search(r'--opt (\w+)', a).group(1)
    if o == 'kronq':
        b = re.search(r'--block (\d+)', a); b = int(b.group(1)) if b else 0
        tag = 'kronq full' if b == 0 else f'kronq b{b}'
        if '--compress 1' in a: tag += '+c'
        if '--gnls' in a: tag += ' gnls'
        if '--gmask' in a: tag += '+gmask'
        if '--rem' in a: tag += ' rem'
        return tag
    if o == 'soap': return 'soap b256' if '--block' in a else 'soap full'
    return o
rows = []
for a, d in runs.items():
    if 'val_final' not in d: continue
    rows.append(dict(variant=variant(a), width=d['width'], p=d['p'], lr=float(re.search(r'--lr ([0-9.e-]+)', a).group(1)),
                     steps=int(re.search(r'--steps (\d+)', a).group(1)), val=d['val_final'], spp=d.get('state_per_param', np.nan),
                     peak_gb=d.get('peak_mem_bytes', np.nan) / 1e9, wall=d.get('wall_total', np.nan),
                     q1=np.median(d['q']['gnls_qratio_l1']) if d['q'].get('gnls_qratio_l1') else np.nan, git=d['git'], args=a))
print(f'{len(rows)} finished runs of {len(runs)} started')
if not rows: sys.exit(0)
os.makedirs(f'{F}/summary', exist_ok=True)
with open(f'{F}/summary/runs.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
best = {}
for r in rows:
    k = (r['variant'], r['width'], r['p'])
    if k not in best or r['val'] < best[k]['val']: best[k] = r
def mclass(spp): return '<= 2.0 (Adam)' if spp <= 2.05 else '2-3' if spp <= 3 else '3-6' if spp <= 6 else '> 6'
for p in sorted(set(r['p'] for r in rows)):
    for w in sorted(set(r['width'] for r in rows)):
        sel = sorted([r for k, r in best.items() if k[1] == w and k[2] == p], key=lambda r: (r['spp'], r['val']))
        if not sel: continue
        print(f'\n== p = {p:g} ({"smooth" if p == 0 else "rough"}), width {w}: equal FLOPs; best lr per method; sorted by state/param')
        print('   %-26s %8s %6s %7s %7s %8s %6s %s' % ('method', 'val', 'lr', 'steps', 'state/P', 'peak GB', 'wall s', 'memory class'))
        for r in sel:
            flag = '  [rem undo-to-0.2 code]' if ('rem' in r['variant'] and r['git'] in ('a36fd54', '6e29d3a')) else ''
            print('   %-26s %8.3e %6g %7d %7.2f %8.2f %6.0f %s%s%s' % (r['variant'], r['val'], r['lr'], r['steps'], r['spp'], r['peak_gb'], r['wall'], mclass(r['spp']),
                  '' if np.isnan(r['q1']) else f'   q_true/q_K(l1) med {r["q1"]:.0f}', flag))
        ref = best.get(('adamw', w, p))
        if ref:
            print('   ratio to AdamW (val_adamw / val):', '  '.join(f"{r['variant']}: {ref['val'] / r['val']:.2f}" for r in sel if r['variant'] != 'adamw'))
print('\nfailed runs (val > 0.5 on p = 2, > 1e-2 on p = 0):')
for r in sorted(rows, key=lambda r: (r['p'], r['width'], r['variant'], r['lr'])):
    if (r['p'] == 2.0 and r['val'] > 0.5) or (r['p'] == 0.0 and r['val'] > 1e-2): print('   p=%g w=%d %-24s lr=%g val=%.3g' % (r['p'], r['width'], r['variant'], r['lr'], r['val']))
print('\nwritten', f'{F}/summary/runs.csv')
