"""Summarise the optimizer scaling study (scale_study.py runs; res_s*.jsonl in $L63_ROOT/runs/scale).
Tables: (1) best final validation MSE per optimizer variant at each width and target, with lr, wall, memory;
(2) KronQ memory reductions relative to full KronQ; (3) equal wall-clock comparison (val at AdamW's total wall time);
(4) time split at the largest width. Single seed, 5000 steps, batch 256, no schedule / warmup / weight decay (see
THEORY.md 21(d) for what the study does and does not decide)."""
import os, re, glob, json, collections, csv, sys
import numpy as np
ROOT = os.environ.get('L63_ROOT', os.path.expandvars('$SCRATCH/l63_held_rough'))
F = f'{ROOT}/runs/scale'; OUT = f'{F}/summary'; os.makedirs(OUT, exist_ok=True)

recs = [json.loads(l) for f in sorted(glob.glob(f'{F}/res_s*.jsonl')) for l in open(f) if l.strip()]
runs = collections.defaultdict(dict)
for r in recs:
    d = runs[r['args']]
    d.setdefault('curve', {})
    if r['metric'] in ('val', 'wall'): d['curve'].setdefault(r['step'], {})[r['metric']] = r['value']
    else: d[r['metric']] = r['value']
    d['opt'], d['width'], d['p'], d['P'] = r['opt'], r['width'], r['p'], r['P']

def variant(a):
    o = re.search(r'--opt (\w+)', a).group(1)
    if o == 'kronq':
        if '--max_dim' in a: return 'kronq diag>1024'
        b = re.search(r'--block (\d+)', a)
        if b and '--compress 1' in a: return f'kronq b{b.group(1)}+compress'
        return f'kronq b{b.group(1)}' if b else 'kronq full'
    if o == 'soap': return 'soap b256' if '--block' in a else 'soap full'
    return o
ORDER = ['adamw', 'muon', 'soap full', 'soap b256', 'kronq full', 'kronq b1024', 'kronq b256', 'kronq b64', 'kronq b256+compress', 'kronq diag>1024']
rows = []
for a, d in runs.items():
    lr = float(re.search(r'--lr ([0-9.e-]+)', a).group(1))
    rows.append(dict(variant=variant(a), opt=d['opt'], width=d['width'], p=d['p'], lr=lr, P=d['P'], val_final=d.get('val_final', np.nan),
                     wall=d.get('wall_total', np.nan), spp=d.get('state_per_param', np.nan), peak_gb=d.get('peak_mem_bytes', np.nan) / 1e9,
                     t_fb=d.get('t_fb', np.nan), t_stats=d.get('t_stats', np.nan), t_eig=d.get('t_eig', np.nan), t_upd=d.get('t_upd', np.nan),
                     curve=d['curve'], args=a))
with open(f'{OUT}/runs.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k not in ('curve', 'args')] + ['args']); w.writeheader()
    for r in rows: w.writerow({k: r[k] for k in w.fieldnames})
widths = sorted(set(r['width'] for r in rows)); ps = sorted(set(r['p'] for r in rows))
best = {}   # (variant, width, p) -> row with the best val_final over lr
for r in rows:
    k = (r['variant'], r['width'], r['p'])
    if k not in best or r['val_final'] < best[k]['val_final']: best[k] = r
print('(1) best final validation MSE over the lr grid; per cell: val_final (lr) | wall s | state/param | peak GB')
for p in ps:
    print(f'\n== target p = {p:g} ({"smooth" if p == 0 else "rough"})')
    print('%-20s' % 'variant' + ''.join('%26s' % f'w={w}' for w in widths))
    for v in ORDER:
        line = '%-20s' % v
        for w in widths:
            r = best.get((v, w, p))
            line += '%26s' % (f"{r['val_final']:.3e} ({r['lr']:g}) {r['wall']:5.0f}s {r['spp']:4.1f} {r['peak_gb']:4.1f}" if r else '-')
        print(line)
print('\n(2) KronQ memory reductions: val_final relative to KronQ full at the same width (best lr each); state/param in brackets')
for p in ps:
    print(f'== p = {p:g}')
    for v in ORDER[5:]:
        line = '%-20s' % v
        for w in widths:
            r, f = best.get((v, w, p)), best.get(('kronq full', w, p))
            line += '%18s' % (f"{r['val_final'] / f['val_final']:6.2f}x [{r['spp']:.1f}]" if r and f else '-')
        print(line)
print('\n(3) equal wall-clock: val at the last eval with wall <= AdamW best run\'s wall_total at that width/target (best over lr); "-" = no eval within the budget')
for p in ps:
    print(f'== p = {p:g}')
    print('%-20s' % 'variant' + ''.join('%22s' % f'w={w}' for w in widths))
    for v in ORDER:
        line = '%-20s' % v
        for w in widths:
            ref = best.get(('adamw', w, p)); budget = ref['wall'] if ref else np.nan
            cands = []
            for r in rows:
                if (r['variant'], r['width'], r['p']) != (v, w, p): continue
                ok = [(s, c) for s, c in r['curve'].items() if 'wall' in c and 'val' in c and c['wall'] <= budget]
                if ok: s, c = max(ok); cands.append((c['val'], s, r['lr']))
            line += '%22s' % (f'{min(cands)[0]:.3e} @{min(cands)[1]:d}' if cands else '-')
        print(line)
print('\n(4) time split over the run at the largest width (best-lr runs), seconds: fwd/bwd | statistics | eigh | update | total wall; then ms per step.')
print('    NOTE: 3 runs shared each GPU (co-tenants differ by slot), so wall times are inflated and only roughly comparable.')
wmax = max(widths)
for p in ps:
    print(f'== p = {p:g}')
    for v in ORDER:
        r = best.get((v, wmax, p))
        if r: print('%-20s %7.0f %7.0f %7.0f %7.0f | %7.0f | %6.1f ms/step' % (v, r['t_fb'], r['t_stats'], r['t_eig'], r['t_upd'], r['wall'], 1e3 * r['wall'] / 5000))
print('\n(4b) failed runs (val_final above 0.5 on p = 2, i.e. the constant predictor, or above 1e-2 on p = 0):')
for r in sorted(rows, key=lambda r: (r['p'], r['width'], r['variant'], r['lr'])):
    if (r['p'] == 2.0 and r['val_final'] > 0.5) or (r['p'] == 0.0 and r['val_final'] > 1e-2):
        print('   p=%g w=%d %-20s lr=%g val_final=%.3g' % (r['p'], r['width'], r['variant'], r['lr'], r['val_final']))
print('\n(5) learning-rate sensitivity: val_final at each lr, p = 2, width', wmax)
for v in ORDER:
    rs = sorted([r for r in rows if (r['variant'], r['width'], r['p']) == (v, wmax, 2.0)], key=lambda r: r['lr'])
    print('%-20s' % v + '  '.join(f"{r['lr']:g}: {r['val_final']:.3e}" for r in rs))
print('\nwritten', OUT)
