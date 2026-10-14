"""Summary of the 2D turbulence runs (scale_turb.sh): per target and method the mean final validation MSE over seeds (unit-variance
target, so 1.0 = predicting zero), seed values, and mean wall; the best KronQ against the best AdamW at equal wall."""
import json, glob, os, sys, collections
F = sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars('$SCRATCH/l63_held_rough/runs/scale_turb2')
def arg(args, k): return args.split(f'--{k} ')[-1].split()[0] if f'--{k} ' in args else None   # last occurrence (argparse semantics)
res = collections.defaultdict(list)
for fn in glob.glob(f'{F}/res_j*.jsonl'):
    rows = [json.loads(l) for l in open(fn)]
    if not rows: continue
    a = rows[0]['args']; vf = [r['value'] for r in rows if r['metric'] == 'val_final']; w = [r['value'] for r in rows if r['metric'] == 'wall_total']; vb = [r['value'] for r in rows if r['metric'] == 'val_best']
    if not vf: continue
    t = arg(a, 'turb_target') or 'delta'
    m = (f"adamw lr {arg(a, 'lr')} steps {arg(a, 'steps')}" if '--opt adamw' in a else f"kronq lr {arg(a, 'lr')} budget {arg(a, 'rem')}")
    res[(t, m)].append((vf[0], w[0] if w else float('nan'), vb[0] if vb else float('nan')))
for t in ('delta', 'direct'):
    print(f'==== turbulence target {t}')
    best = {}
    for k in sorted(res):
        if k[0] != t: continue
        v = [x[0] for x in res[k]]; w = [x[1] for x in res[k]]; b = [x[2] for x in res[k]]; mean = sum(v) / len(v)
        print(f"   {k[1]:34s} final {mean:.4g}  best-along-run {sum(b) / len(b):.4g}  seeds ({' '.join(f'{x:.3g}' for x in v)})  wall {sum(w) / len(w):.0f} s")
        fam = 'kronq' if k[1].startswith('kronq') else ('adamw long' if 'steps 25000' in k[1] else 'adamw 5000')
        if fam not in best or mean < best[fam][0]: best[fam] = (mean, k[1], sum(w) / len(w))
    if 'kronq' in best and 'adamw long' in best:
        print(f"   best kronq {best['kronq'][0]:.4g} ({best['kronq'][1]}, {best['kronq'][2]:.0f} s) vs best adamw equal-wall {best['adamw long'][0]:.4g} ({best['adamw long'][1]}, {best['adamw long'][2]:.0f} s): ratio adamw/kronq {best['adamw long'][0] / best['kronq'][0]:.2f}")
