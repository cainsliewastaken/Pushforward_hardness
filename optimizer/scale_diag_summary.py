"""Summarise the width-4096 collapse diagnosis (scale_diag.sh; res_j*.jsonl in $L63_ROOT/runs/scale_diag).
Per job: the validation curve and, per layer and eval step, the push-weight quantiles, the clamp fraction, the fraction of
modes where the GN cap binds, the step energy on the lowest decile of modes by Kronecker eigenvalue, the hidden
preactivation rms and saturation fraction (|z| > 3), and the weight norms."""
import os, re, glob, json, collections, sys
ROOT = os.environ.get('L63_ROOT', os.path.expandvars('$SCRATCH/l63_held_rough')); SUB = sys.argv[1] if len(sys.argv) > 1 else 'scale_diag'; F = f'{ROOT}/runs/{SUB}'
LABELS = {'scale_diag': {0: 'b256 hyper OFF', 1: 'b256 u_clamp 1', 2: 'b256 beta_u 1e-3', 3: 'b256 as run', 4: 'full as run', 5: 'diag>1024 hyper OFF'},
          'scale_fix': {0: 'b256 gmask 1e-3', 1: 'b256 gmask 1e-5', 2: 'diag>1024 gmask 1e-3', 3: 'full gmask 1e-3', 4: 'b256 batch-curvature cap', 5: 'b256 step budget rem 0.1'},
          'scale_alloc': {0: 'ROUGH hyper rem 0.1', 1: 'ROUGH hyper ctrl 0.1', 2: 'ROUGH hyper ctrl 0.03', 3: 'ROUGH alloc ctrl 0.1', 4: 'ROUGH alloc ctrl 0.03', 5: 'ROUGH alloc rem 0.1',
                          6: 'SMOOTH hyper rem 0.1', 7: 'SMOOTH hyper ctrl 0.1', 8: 'SMOOTH hyper ctrl 0.03', 9: 'SMOOTH alloc ctrl 0.1', 10: 'SMOOTH alloc ctrl 0.03', 11: 'SMOOTH alloc rem 0.1'},
          'scale_fix2': {0: 'ROUGH b256+compress rem 0.1', 1: 'ROUGH b256+compress gnls', 2: 'ROUGH b64 rem 0.1', 3: 'ROUGH diag rem 0.1', 4: 'SMOOTH b256+compress rem 0.1', 5: 'SMOOTH b256+compress gnls', 6: 'SMOOTH b64 rem 0.1', 7: 'SMOOTH diag rem 0.1'}}
LABEL = LABELS.get(SUB, {})
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
    for name in ('u_q50', 'wmean', 'wmax', 'capfrac', 'livefrac', 'glive', 'glive_and_live', 'e_lowdec', 'zrms', 'sat', 'wnorm'): row(name)
    for name in ('gnls_alpha', 'gnls_qratio', 'alloc_gamma_over_lmax', 'alloc_fitfrac', 'g_scale'): row(name)
    if 'g_scale' in M: print('  g_scale        ' + '  '.join(('%5d:' % st) + ('%.3g' % M['g_scale'][st]) for st in STEPS if st in M['g_scale']))
    # per-mode weight profile vs eigenvalue decile (bin 0 = slowest 10 % of modes, bin 9 = stiffest), at steps 1000 and 5000
    for st in (1000, 5000):
        for l in L:
            prof = [M.get(f'ld_bin{b}_l{l}', {}).get(st) for b in range(10)]
            if all(v is not None for v in prof):
                lam = [M.get(f'lam_bin{b}_l{l}', {}).get(st, float('nan')) for b in range(10)]
                print('  log|D|/rms by lambda decile, l%d @%d: ' % (l, st) + ' '.join('%+.2f' % v for v in prof) + '   (lambda/lam_max: ' + ' '.join('%.0e' % (x / max(lam)) for x in lam) + ')')
                uprof = [M.get(f'u_bin{b}_l{l}', {}).get(st) for b in range(10)]
                if all(v is not None for v in uprof): print('  u by lambda decile,        l%d @%d: ' % (l, st) + ' '.join('%+.2f' % v for v in uprof))
    if 'rem_rho' in M: print('  rem_rho        ' + '  '.join(('%5d:' % st) + ('%.3g' % M['rem_rho'][st]) for st in STEPS if st in M['rem_rho']))
    fin = {k: M[k][5000] for k in ('val_final', 'wall_total', 'state_per_param') if k in M and 5000 in M[k]}
    print('  final:', ', '.join(f'{k}={v:.4g}' for k, v in fin.items()))
