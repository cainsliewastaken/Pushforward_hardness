"""Continuity check summary (scale_continuity.sh): (A) the other instance's script + data at its tuned settings against the
saved selections; (B) today's harness at width 512 / batch 100 / 6000 steps: best lr per method and ratio to AdamW."""
import os, re, glob, json, collections
F = os.path.expandvars('$SCRATCH/l63_held_rough/runs/scale_continuity')
SAVED = {('p0', 'adam'): 0.357, ('p0', 'sign'): 0.178, ('p0', 'ksadam'): 0.161, ('p2', 'adam'): 11.48, ('p2', 'sign'): 5.25, ('p2', 'ksadam'): 5.23}
print('(A) other instance\'s train_ks.py on field20k_p{0,2}.npz, width 512, batch 100, 6000 steps (seed 1 of the saved 3-seed means)')
for f in sorted(glob.glob(f'{F}/old_j*.jsonl'), key=lambda s: int(re.search(r'j(\d+)', s).group(1))):
    M = {}
    for l in open(f):
        r = json.loads(l); M[r['metric']] = r['value']
    args = M.get('args', '') if isinstance(M.get('args'), str) else ''
    j = int(re.search(r'old_j(\d+)', f).group(1)); p = 'p0' if j < 3 else 'p2'; meth = ['adam', 'sign', 'ksadam'][j % 3]
    v = M.get('E_val_final', float('nan')); print('   %s %-7s E_val_final=%.4g   saved mean %.4g   ratio %.3f' % (p, meth, v, SAVED[(p, meth)], v / SAVED[(p, meth)]))
print('\n(B) today\'s harness, online field problem, width 512, batch 100, 6000 steps; best lr per method')
rows = collections.defaultdict(list)
for f in glob.glob(f'{F}/new_j*.jsonl'):
    recs = [json.loads(l) for l in open(f) if l.strip()]
    if not recs: continue
    a = recs[0]['args']; p = float(re.search(r'--p (\d)', a).group(1)); lr = float(re.search(r'--lr ([0-9.e-]+)', a).group(1))
    opt = re.search(r'--opt (\w+)', a).group(1); v = opt + (' full + rem 0.2' if '--rem' in a else ' full' if opt in ('kronq', 'soap') else '')
    val = [r['value'] for r in recs if r['metric'] == 'val_final']
    if val: rows[(p, v)].append((val[0], lr))
for p in (0.0, 2.0):
    ref = min(rows.get((p, 'adamw'), [(float('nan'), 0)]))[0]
    print(f'  p = {p:g}:')
    for v in ('adamw', 'soap full', 'kronq full', 'kronq full + rem 0.2'):
        if (p, v) in rows:
            b = min(rows[(p, v)]); print('     %-22s %.4e (lr %g)   AdamW/this = %.2f   all lrs: %s' % (v, b[0], b[1], ref / b[0], ' '.join(f'{lr:g}:{val:.3g}' for val, lr in sorted(rows[(p, v)], key=lambda t: t[1]))))
