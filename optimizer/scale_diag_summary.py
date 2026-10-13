"""Summarise the width-4096 collapse diagnosis (scale_diag.sh; res_j*.jsonl in $L63_ROOT/runs/scale_diag).
Per job: the validation curve and, per layer and eval step, the push-weight quantiles, the clamp fraction, the fraction of
modes where the GN cap binds, the step energy on the lowest decile of modes by Kronecker eigenvalue, the hidden
preactivation rms and saturation fraction (|z| > 3), and the weight norms."""
import os, re, glob, json, collections, sys
ROOT = os.environ.get('L63_ROOT', os.path.expandvars('$SCRATCH/l63_held_rough')); F = f'{ROOT}/runs/scale_diag'
LABEL = {0: 'b256 hyper OFF', 1: 'b256 u_clamp 1', 2: 'b256 beta_u 1e-3', 3: 'b256 as run', 4: 'full as run', 5: 'diag>1024 hyper OFF'}
STEPS = [250, 500, 750, 1000, 1500, 2000, 3000, 5000]
for f in sorted(glob.glob(f'{F}/res_j*.jsonl')):
    j = int(re.search(r'res_j(\d+)', f).group(1)); recs = [json.loads(l) for l in open(f) if l.strip()]
    if not recs: print(f'job {j}: no records'); continue
    M = collections.defaultdict(dict)
    for r in recs: M[r['metric']][r['step']] = r['value']
    print(f"\n==== job {j}: {LABEL.get(j, '?')}   ({recs[0]['args'].split('--out')[0].strip()})")
    print('val:  ' + '  '.join(f"{s}:{M['val'].get(s, float('nan')):.4f}" for s in [0] + STEPS))
    L = sorted(set(int(m.split('_l')[-1]) for m in M if '_l' in m))
    def row(name, fmt='%.3g'):
        for l in L:
            key = f'{name}_l{l}'
            if key in M: print('  %-12s l%d ' % (name, l) + '  '.join(('%5d:' % s) + (fmt % M[key][s]) for s in STEPS if s in M[key]))
    for name in ('u_q10', 'u_q50', 'u_q90', 'u_clampfrac', 'wmean', 'wmax', 'capfrac', 'livefrac', 'e_lowdec', 'zrms', 'sat', 'wnorm'): row(name)
    fin = {k: M[k][5000] for k in ('val_final', 'wall_total', 'state_per_param') if k in M and 5000 in M[k]}
    print('  final:', ', '.join(f'{k}={v:.4g}' for k, v in fin.items()))
